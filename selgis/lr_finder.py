"""Learning rate finder (Leslie Smith style): exponential LR sweep, optimal LR from loss curve."""

import math
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.optim as optim

from selgis.utils import move_to_device, unpack_batch, is_dict_like


class LRFinder:
    """
    Find learning rate by sweeping LR exponentially and tracking loss.
    Works with any PyTorch or HuggingFace model. Use trainable_only=True for
    large models (e.g. LLM with LoRA) to save memory.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module | Callable | None = None,
        device: torch.device | None = None,
        trainable_only: bool = False,
    ) -> None:
        """
        Args:
            model: Model to tune.
            optimizer: Optimizer (LR will be swept).
            criterion: Loss; can be None if model returns loss (e.g. HuggingFace).
            device: Device; defaults to model's device.
            trainable_only: If True, clone/restore only trainable parameters (saves memory for LoRA/LLM).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or next(model.parameters()).device
        self.trainable_only = trainable_only
        self._trainable_names = (
            {n for n, p in model.named_parameters() if p.requires_grad}
            if trainable_only
            else None
        )

        self._initial_state = self._clone_state()
        self._initial_optim_state = optimizer.state_dict()
        self._lrs: list[float] = []
        self._losses: list[float] = []

    def _clone_state(self) -> dict:
        """Clone model state to CPU (all params or trainable_only)."""
        state = self.model.state_dict()
        if self._trainable_names is not None:
            state = {k: v.cpu().clone() for k, v in state.items() if k in self._trainable_names}
        else:
            state = {k: v.cpu().clone() for k, v in state.items()}
        return state

    def _restore_state(self) -> None:
        """Restore initial model and optimizer state."""
        if self._trainable_names is not None:
            current = self.model.state_dict()
            for k, v in self._initial_state.items():
                if k in current:
                    current[k] = v.to(self.device)
            self.model.load_state_dict(current, strict=False)
        else:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self._initial_state.items()}
            )
        self.optimizer.load_state_dict(self._initial_optim_state)

    def find(
        self,
        train_loader,
        forward_fn: Callable[[nn.Module, Any], tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_steps: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 4.0,
    ) -> float:
        """
        Run LR sweep. forward_fn (model, batch) -> (loss, logits) optional.
        Returns suggested learning rate.
        """
        print("\n[INFO] Searching for optimal LR...")

        mult = (end_lr / start_lr) ** (1.0 / num_steps)
        lr = start_lr
        self._set_lr(lr)

        self.model.train()
        self._losses = []
        self._lrs = []
        smoothed_loss = 0.0
        best_loss = float("inf")

        data_iter = iter(train_loader)

        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = self._compute_loss(batch, forward_fn)

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Loss exploded at LR={lr:.2e}")
                break

            loss_val = loss.item()
            smoothed_loss = (
                loss_val if step == 0
                else smooth_f * loss_val + (1 - smooth_f) * smoothed_loss
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
        optimal_lr = self._compute_optimal_lr(self._lrs, self._losses)
        print(f"[INFO] Found optimal LR: {optimal_lr:.2e}")

        return optimal_lr

    def _compute_loss(
        self,
        batch: Any,
        forward_fn: Callable | None,
    ) -> torch.Tensor | None:
        """Compute loss for one batch (dict or tensor inputs)."""
        batch = move_to_device(batch, self.device)

        if forward_fn is not None:
            loss, _ = forward_fn(self.model, batch)
            return loss

        inputs, labels = unpack_batch(batch)

        if is_dict_like(inputs):
            inputs_dict = dict(inputs)
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
        """Set learning rate for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_optimal_lr(
        self,
        lrs: list[float],
        losses: list[float],
    ) -> float:
        """Pick LR at steepest descent (slightly before min)."""
        if len(lrs) < 10:
            return lrs[len(lrs) // 2] if lrs else 1e-3

        min_grad = float("inf")
        min_idx = 0

        for i in range(1, len(losses)):
            grad = (losses[i] - losses[i - 1])
            grad /= (math.log10(lrs[i]) - math.log10(lrs[i - 1]))
            if grad < min_grad:
                min_grad = grad
                min_idx = i

        safe_idx = max(0, min_idx - 10)
        return lrs[safe_idx]

    @property
    def history(self) -> dict[str, list]:
        """LR and loss history for plotting."""
        return {"lrs": self._lrs, "losses": self._losses}