"""Learning rate finder (Leslie Smith style)."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from selgis.utils import is_dict_like, move_to_device, unpack_batch


class LRFinder:
    """Find optimal learning rate via exponential sweep.

    Works with any PyTorch or HuggingFace model.  Use
    ``trainable_only=True`` for large models (e.g. LLM with LoRA)
    to reduce memory during state save/restore.

    Args:
        model: Model to tune.
        optimizer: Optimizer whose LR will be swept.
        criterion: Loss function.  May be ``None`` if the model
            returns loss directly (e.g. HuggingFace models).
        device: Compute device.  Defaults to the device of the first
            model parameter.
        trainable_only: Save and restore only trainable parameters.
        amp_dtype: Optional dtype for ``torch.amp.autocast``
            (e.g. ``torch.float16``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module | Callable | None = None,
        device: torch.device | None = None,
        trainable_only: bool = False,
        amp_dtype: torch.dtype | None = None,
        save_optimizer_state: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainable_only = trainable_only
        self.amp_dtype = amp_dtype
        self.save_optimizer_state = save_optimizer_state

        self._has_device_map = bool(getattr(model, "hf_device_map", None))

        if device is not None:
            self.device = device
        elif not self._has_device_map:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cuda")

        self._trainable_names: set[str] | None = None
        if trainable_only:
            self._trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

        self._initial_state = self._clone_state()
        self._initial_optim_state = None
        self._initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
        if self.save_optimizer_state:
            self._initial_optim_state = self._clone_any(
                optimizer.state_dict(),
            )
        self._lrs: list[float] = []
        self._losses: list[float] = []

    def _clone_any(self, value: Any):
        """Deep-clone arbitrary optimizer state to CPU."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: self._clone_any(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone_any(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_any(v) for v in value)
        return value

    def _clone_state(self) -> dict[str, torch.Tensor]:
        """Clone model state to CPU.

        Returns:
            Dictionary of parameter names to CPU tensor copies.
        """
        state: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if self._trainable_names is not None and name not in self._trainable_names:
                continue
            if param.device.type == "cpu":
                state[name] = param.detach().clone()
            else:
                state[name] = param.detach().cpu()
        return state

    def _restore_state(self) -> None:
        """Restore initial model and optimizer state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._initial_state:
                    param.data.copy_(self._initial_state[name])

        if self._initial_optim_state is not None:
            self.optimizer.load_state_dict(self._initial_optim_state)
        else:
            for idx, pg in enumerate(self.optimizer.param_groups):
                if idx < len(self._initial_lrs):
                    pg["lr"] = self._initial_lrs[idx]

    def _get_amp_context(self) -> AbstractContextManager:
        """Return autocast context matching training configuration."""
        if self.amp_dtype is None:
            return nullcontext()
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        if self.device.type == "cpu":
            return torch.amp.autocast("cpu", dtype=self.amp_dtype)
        return nullcontext()

    def find(
        self,
        train_loader,
        forward_fn: (
            Callable[
                [nn.Module, Any],
                tuple[torch.Tensor, torch.Tensor],
            ]
            | None
        ) = None,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_steps: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 4.0,
    ) -> float:
        """Run LR sweep and return the suggested learning rate.

        Args:
            train_loader: Training data loader.
            forward_fn: Optional ``(model, batch) -> (loss, logits)``.
            start_lr: Starting learning rate.
            end_lr: Ending learning rate.
            num_steps: Number of sweep steps.
            smooth_f: Exponential smoothing factor for loss.
            diverge_th: Stop if smoothed loss exceeds
                ``diverge_th * best_loss``.

        Returns:
            Suggested optimal learning rate.
        """
        if num_steps <= 0:
            print("[WARN] num_steps must be positive, using default LR")
            return start_lr

        print("\n[INFO] Searching for optimal LR...")

        mult = (end_lr / start_lr) ** (1.0 / num_steps)
        lr = start_lr
        self._set_lr(lr)

        self.model.train()
        self._losses = []
        self._lrs = []
        smoothed_loss = 0.0
        best_loss = float("inf")
        skipped_batches = 0
        max_skipped = num_steps * 2  # Allow up to 2x skips

        data_iter = iter(train_loader)

        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = self._compute_loss(batch, forward_fn)

            # Skip batch if loss is None (e.g., wrong format)
            if loss is None:
                skipped_batches += 1
                if skipped_batches >= max_skipped:
                    print("[WARN] Too many invalid batches, stopping LR finder")
                    break
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Loss exploded at LR={lr:.2e}")
                break

            loss_val = loss.item()
            smoothed_loss = (
                loss_val if step == 0 else smooth_f * loss_val + (1 - smooth_f) * smoothed_loss
            )

            self._losses.append(smoothed_loss)
            self._lrs.append(lr)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            if smoothed_loss > diverge_th * best_loss:
                break

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            self.optimizer.step()

            lr *= mult
            self._set_lr(lr)

        self._restore_state()
        self._free_saved_state()

        optimal_lr = self._compute_optimal_lr(
            self._lrs,
            self._losses,
        )
        print(f"[INFO] Found optimal LR: {optimal_lr:.2e}")

        return optimal_lr

    def _free_saved_state(self) -> None:
        """Release saved state tensors to free memory."""
        self._initial_state = {}
        self._initial_optim_state = None
        self._lrs = []
        self._losses = []
        gc.collect()
        if self._trainable_names is not None:
            self._trainable_names = None

    def _compute_loss(
        self,
        batch: Any,
        forward_fn: Callable | None,
    ) -> torch.Tensor | None:
        """Compute loss for one batch.

        Args:
            batch: Raw batch from the data loader.
            forward_fn: Optional custom forward function.

        Returns:
            Loss tensor, or ``None`` on failure.
        """
        if not self._has_device_map:
            batch = move_to_device(batch, self.device)

        with self._get_amp_context():
            if forward_fn is not None:
                loss, _ = forward_fn(self.model, batch)
                return loss

            inputs, labels = unpack_batch(batch)

            if is_dict_like(inputs):
                inputs_dict = dict(inputs)
                # For causal LM, include labels if available
                if labels is not None:
                    inputs_dict["labels"] = labels
                # Ensure input_ids exists
                if "input_ids" not in inputs_dict:
                    if "text" in inputs_dict:
                        # Try tokenizing if tokenizer available
                        if self.criterion is not None and hasattr(self.criterion, "tokenizer"):
                            raise ValueError("Need tokenizer to process text input")
                    raise ValueError("batch must contain input_ids for LLM training")
                outputs = self.model(**inputs_dict)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    return outputs.loss
                if self.criterion is None:
                    raise ValueError("Model doesn't return loss, criterion required")
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                return self.criterion(logits, labels)

            outputs = self.model(inputs)
            if self.criterion is None:
                raise ValueError("Criterion required for tensor input")
            return self.criterion(outputs, labels)

    def _set_lr(self, lr: float) -> None:
        """Set learning rate for all parameter groups."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    @staticmethod
    def _compute_optimal_lr(
        lrs: list[float],
        losses: list[float],
    ) -> float:
        """Select LR at steepest descent of the loss curve.

        Args:
            lrs: Learning rates from the sweep.
            losses: Corresponding smoothed losses.

        Returns:
            Suggested learning rate.
        """
        if len(lrs) < 10:
            return lrs[len(lrs) // 2] if lrs else 1e-3

        min_grad = float("inf")
        min_idx = 0

        for i in range(1, len(losses)):
            lr_diff = math.log10(lrs[i]) - math.log10(lrs[i - 1])
            if abs(lr_diff) < 1e-12:
                continue
            grad = (losses[i] - losses[i - 1]) / lr_diff
            if grad < min_grad:
                min_grad = grad
                min_idx = i

        safe_idx = max(0, min_idx - 10)
        return lrs[safe_idx]

    @property
    def history(self) -> dict[str, list[float]]:
        """LR and loss history from the last sweep."""
        return {"lrs": self._lrs, "losses": self._losses}
