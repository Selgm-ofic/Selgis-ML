"""Loss functions for memory-efficient training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ChunkedCrossEntropyLoss(nn.Module):
    """Memory-efficient Cross-Entropy with vocabulary chunking.

    Splits the VOCABULARY axis into chunks to reduce peak memory usage.
    For large vocabularies (50K+ tokens), chunked backward saves
    several GB of VRAM compared to computing softmax over the full
    [B, T, V] tensor at once.

    Algorithm:
        Pass 1 (no grad) — global max over the vocabulary, chunk by chunk.
        Pass 2 — per-chunk quantities (sum of exponentials, target logit,
        logit sum) computed inside ``torch.utils.checkpoint`` blocks.
        Nothing chunk-sized is stored for backward; each chunk is
        recomputed chunk-by-chunk during the backward pass, so peak
        memory stays proportional to ``chunk_size``.

    Label smoothing follows the PyTorch definition:
        loss = (1 - ε) * CE_hard + (ε / V) * Σ_i(-log p_i)
    The uniform term is ``log_z - mean(x)`` and carries gradient,
    matching ``F.cross_entropy(label_smoothing=ε)`` exactly.

    Args:
        chunk_size: Vocabulary chunk size. Default 1024.
            Smaller = less backward-pass memory, slightly slower.
        label_smoothing: Label smoothing factor in [0, 1). Default 0.0.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``. Default ``'mean'``.
        ignore_index: Label index to ignore (e.g. padding). Default -100.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.chunk_size = chunk_size
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss.

        Args:
            logits: ``[B, T, V]`` — batch × sequence × vocabulary.
                Also accepts ``[B, V]`` for non-sequence tasks.
            targets: ``[B, T]`` or ``[B]`` integer token IDs.

        Returns:
            Scalar loss (``reduction='mean'/'sum'``) or
            ``[B, T]`` tensor (``reduction='none'``).
        """
        # ── Normalise to 3-D ──────────────────────────────────────────────
        if logits.dim() == 2:
            # [B, V] → [B, 1, V]
            logits = logits.unsqueeze(1)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)

        if logits.dim() != 3:
            raise ValueError(f"logits must be 2-D or 3-D, got {logits.dim()}-D")

        batch_size, seq_len, vocab_size = logits.shape

        # targets shape validation + flatten to [B, T]
        if targets.dim() == 1:
            # [B] → [B, 1]
            targets = targets.unsqueeze(1)
        if targets.shape != (batch_size, seq_len):
            raise ValueError(
                f"targets shape {tuple(targets.shape)} is incompatible "
                f"with logits shape {tuple(logits.shape)}"
            )

        targets = targets.contiguous()

        if self.label_smoothing > 0.0:
            return self._forward_smoothed(logits, targets, vocab_size)
        return self._forward_chunked(logits, targets, vocab_size)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _chunk_bounds(self, vocab_size: int):
        for start in range(0, vocab_size, self.chunk_size):
            yield start, min(start + self.chunk_size, vocab_size)

    def _compute_global_max(
        self,
        logits: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Compute the global vocabulary max, chunk by chunk (no grad).

        Returns:
            ``[B, T]`` float32 tensor, detached.
        """
        global_max = torch.full(
            logits.shape[:2],
            float("-inf"),
            device=logits.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            for start, end in self._chunk_bounds(vocab_size):
                chunk_max = logits[:, :, start:end].max(dim=-1).values
                global_max = torch.maximum(global_max, chunk_max.float())
        return global_max.detach()

    def _sum_exp(
        self,
        logits: torch.Tensor,
        global_max: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Accumulate Σ exp(x - max) chunk by chunk (checkpointed).

        Returns:
            ``[B, T]`` float32 tensor with gradients through every chunk.
        """
        grad_enabled = torch.is_grad_enabled()

        def chunk_fn(chunk: torch.Tensor) -> torch.Tensor:
            return (chunk.float() - global_max.unsqueeze(-1)).exp().sum(dim=-1)

        total = None
        for start, end in self._chunk_bounds(vocab_size):
            chunk = logits[:, :, start:end]
            if grad_enabled:
                part = checkpoint(chunk_fn, chunk, use_reentrant=False)
            else:
                with torch.no_grad():
                    part = chunk_fn(chunk)
            total = part if total is None else total + part
        return total

    def _sum_logits(
        self,
        logits: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Accumulate Σ x over the vocabulary chunk by chunk (checkpointed).

        Used by label smoothing; ``Σ_i(-log p_i) = V·log_z - Σ x``.
        """
        grad_enabled = torch.is_grad_enabled()

        def chunk_fn(chunk: torch.Tensor) -> torch.Tensor:
            return chunk.float().sum(dim=-1)

        total = None
        for start, end in self._chunk_bounds(vocab_size):
            chunk = logits[:, :, start:end]
            if grad_enabled:
                part = checkpoint(chunk_fn, chunk, use_reentrant=False)
            else:
                with torch.no_grad():
                    part = chunk_fn(chunk)
            total = part if total is None else total + part
        return total

    def _gather_target_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Gather target logits chunk by chunk (checkpointed).

        Each chunk only gathers indices that fall inside it, so the
        gather backward never materialises a full ``[B, T, V]`` gradient.

        Returns:
            ``[B, T]`` tensor of target logits.
        """
        grad_enabled = torch.is_grad_enabled()
        result = torch.zeros(
            targets.shape,
            device=logits.device,
            dtype=logits.dtype,
        )

        for start, end in self._chunk_bounds(vocab_size):
            in_chunk = (targets >= start) & (targets < end)
            if not in_chunk.any():
                continue

            # NOTE: ``start`` is bound as a default argument so the closure
            # keeps its per-iteration value at checkpoint-recompute time.
            def chunk_fn(chunk: torch.Tensor, start: int = start) -> torch.Tensor:
                local_idx = (targets - start).clamp(0, chunk.shape[-1] - 1)
                return chunk.gather(dim=-1, index=local_idx.unsqueeze(-1)).squeeze(-1)

            chunk = logits[:, :, start:end]
            if grad_enabled:
                vals = checkpoint(chunk_fn, chunk, use_reentrant=False)
            else:
                with torch.no_grad():
                    vals = chunk_fn(chunk)
            result = result + vals * in_chunk.to(logits.dtype)
        return result

    def _forward_chunked(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Hard-target chunked CE (no label smoothing)."""
        valid_mask = targets != self.ignore_index  # [B, T]

        global_max = self._compute_global_max(logits, vocab_size)
        sum_exp = self._sum_exp(logits, global_max, vocab_size)
        log_z = global_max + sum_exp.log()  # [B, T]

        target_logits = self._gather_target_logits(
            logits, targets.clamp(min=0), vocab_size
        )

        # CE = -(target_logit - log_z)
        per_token_loss = (log_z - target_logits) * valid_mask.to(logits.dtype)

        return self._reduce(per_token_loss, valid_mask)

    def _forward_smoothed(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Label-smoothed chunked CE.

        Smoothing formula (identical to ``F.cross_entropy``):
            loss = (1 - ε) * CE_hard + (ε / V) * Σ_i(-log p_i)
        where Σ_i(-log p_i) = V·log_z - Σ x, so the uniform term
        carries gradient through every logit.
        """
        valid_mask = targets != self.ignore_index

        global_max = self._compute_global_max(logits, vocab_size)
        sum_exp = self._sum_exp(logits, global_max, vocab_size)
        log_z = global_max + sum_exp.log()  # [B, T]

        target_logits = self._gather_target_logits(
            logits, targets.clamp(min=0), vocab_size
        )
        sum_x = self._sum_logits(logits, vocab_size)  # [B, T]

        hard_ce = log_z - target_logits  # [B, T]
        uniform_ce = log_z - sum_x / vocab_size  # [B, T]

        per_token_loss = (
            (1.0 - self.label_smoothing) * hard_ce
            + self.label_smoothing * uniform_ce
        ) * valid_mask.to(logits.dtype)

        return self._reduce(per_token_loss, valid_mask)

    def _reduce(
        self,
        per_token_loss: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reduction to per-token losses."""
        if self.reduction == "mean":
            valid_count = valid_mask.sum().clamp(min=1)
            return per_token_loss.sum() / valid_count
        if self.reduction == "sum":
            return per_token_loss.sum()
        return per_token_loss  # 'none'

    def extra_repr(self) -> str:
        return (
            f"chunk_size={self.chunk_size}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}, "
            f"ignore_index={self.ignore_index}"
        )


class CrossEntropyLossV2(nn.Module):
    """Drop-in replacement for ``nn.CrossEntropyLoss``.

    Transparent wrapper that routes to ``ChunkedCrossEntropyLoss`` when
    ``chunk_size > 0``, and to the standard ``F.cross_entropy`` otherwise.
    Accepts both 2-D ``[B, V]`` and 3-D ``[B, T, V]`` logits.

    Args:
        chunk_size: Vocabulary chunk size. 0 = standard (no chunking).
        label_smoothing: Label smoothing factor. Default 0.0.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``. Default ``'mean'``.
        ignore_index: Label index to ignore. Default -100.
    """

    def __init__(
        self,
        chunk_size: int = 0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

        self._chunked: ChunkedCrossEntropyLoss | None = None
        if chunk_size > 0:
            self._chunked = ChunkedCrossEntropyLoss(
                chunk_size=chunk_size,
                label_smoothing=label_smoothing,
                reduction=reduction,
                ignore_index=ignore_index,
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self._chunked is not None:
            return self._chunked(logits, targets)

        # Standard path: flatten to [B*T, V] for F.cross_entropy
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

    def extra_repr(self) -> str:
        return (
            f"chunk_size={self.chunk_size}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}, "
            f"ignore_index={self.ignore_index}"
        )
