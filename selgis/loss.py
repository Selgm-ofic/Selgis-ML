"""Loss functions for memory-efficient training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedCrossEntropyLoss(nn.Module):
    """Memory-efficient Cross-Entropy with vocabulary chunking.

    Splits the VOCABULARY axis into chunks to reduce peak memory usage
    during the backward pass. For large vocabularies (50K+ tokens),
    chunked backward saves several GB of VRAM compared to computing
    softmax over the full [B, T, V] tensor at once.

    Algorithm (2-pass, numerically stable):
        Pass 1 — compute global logsumexp in chunks (no large intermediates).
        Pass 2 — gather target logits, compute CE = -(target - log_Z).

    Note: ``logits [B, T, V]`` must still fit in memory. The gain is
    during the backward pass where chunk-wise exp gradients are computed
    one chunk at a time instead of all at once.

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

    def _compute_log_z(
        self,
        logits: torch.Tensor,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute global max and logsumexp in one chunk-wise sweep.

        Returns:
            Tuple ``(global_max, log_z)`` both of shape ``[B, T]``.
        """
        # Pass 1 — global max (cheap; needed for numerical stability)
        global_max = torch.full(
            logits.shape[:2],
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        for start in range(0, vocab_size, self.chunk_size):
            end = min(start + self.chunk_size, vocab_size)
            chunk_max = logits[:, :, start:end].max(dim=-1).values
            global_max = torch.maximum(global_max, chunk_max)
        global_max = global_max.detach()  # don't diff through max

        # Pass 2 — accumulate exp(x - max) in chunks
        sum_exp = torch.zeros_like(global_max)
        for start in range(0, vocab_size, self.chunk_size):
            end = min(start + self.chunk_size, vocab_size)
            chunk = logits[:, :, start:end] - global_max.unsqueeze(-1)
            sum_exp = sum_exp + chunk.exp().sum(dim=-1)
            # chunk is freed here — only [B, T] sum_exp stays alive

        log_z = global_max + sum_exp.log()  # [B, T]
        return global_max, log_z

    def _forward_chunked(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Hard-target chunked CE (no label smoothing)."""
        valid_mask = targets != self.ignore_index  # [B, T]
        safe_targets = targets.clamp(min=0)  # avoid OOB gather on ignore_index

        _, log_z = self._compute_log_z(logits, vocab_size)

        # Gather target logit: logits[b, t, targets[b, t]]
        target_logits = logits.gather(
            dim=-1,
            index=safe_targets.unsqueeze(-1),
        ).squeeze(-1)  # [B, T]

        # CE = -(target_logit - log_z)
        per_token_loss = -(target_logits - log_z) * valid_mask.float()

        return self._reduce(per_token_loss, valid_mask)

    def _forward_smoothed(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """Label-smoothed chunked CE.

        Smoothing formula:
            loss = (1 - ε) * CE_hard + ε * CE_uniform
        where CE_uniform = log(vocab_size) (constant; uniform distribution).
        """
        valid_mask = targets != self.ignore_index
        safe_targets = targets.clamp(min=0)

        _, log_z = self._compute_log_z(logits, vocab_size)

        target_logits = logits.gather(
            dim=-1,
            index=safe_targets.unsqueeze(-1),
        ).squeeze(-1)

        hard_ce = -(target_logits - log_z)  # [B, T]

        # CE against uniform = -log(1/V) = log(V)
        uniform_ce = math.log(vocab_size)

        per_token_loss = (
            (1.0 - self.label_smoothing) * hard_ce
            + self.label_smoothing * uniform_ce
        ) * valid_mask.float()

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
