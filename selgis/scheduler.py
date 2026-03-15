"""LR schedulers: warmup, cosine (with restarts), linear, polynomial."""

import math
import warnings
from typing import Optional

import torch.optim as optim

from selgis.config import SelgisConfig

DEFAULT_NUM_TRAINING_STEPS = 10_000


class SmartScheduler:
    """Epoch- or step-based LR scheduler.

    Supports warmup, cosine, cosine with warm restarts, linear,
    constant, and polynomial decay.  Provides ``reduce_lr`` and
    ``surge_lr`` for manual adjustment that persists across subsequent
    ``step`` / ``step_epoch`` calls.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        config: SelgisConfig,
        num_training_steps: Optional[int] = None,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self._base_lr = initial_lr
        self.config = config
        self.num_training_steps = num_training_steps

        self._step = 0
        self._epoch = 0
        self._lr_scale = 1.0

        if config.warmup_ratio > 0 and num_training_steps:
            self._warmup_steps = int(
                config.warmup_ratio * num_training_steps,
            )
        elif config.warmup_ratio > 0:
            warnings.warn(
                "warmup_ratio > 0 but num_training_steps not provided. "
                "Warmup disabled.",
                UserWarning,
                stacklevel=2,
            )
            self._warmup_steps = 0
        else:
            self._warmup_steps = 0

    @property
    def _effective_lr(self) -> float:
        """Base LR scaled by any manual adjustment."""
        return self._base_lr * self._lr_scale

    def step_epoch(self, epoch: int) -> float:
        """Update LR for epoch-based schedule.

        Args:
            epoch: Current epoch index (zero-based).

        Returns:
            New learning rate.
        """
        self._epoch = epoch
        warmup = self.config.warmup_epochs

        if epoch < warmup:
            lr = self._effective_lr * (epoch + 1) / warmup
        else:
            lr = self._compute_lr_after_warmup(epoch - warmup)

        self._set_lr(lr)
        return lr

    def step(self) -> float:
        """Update LR for step-based schedule.

        Returns:
            New learning rate.
        """
        self._step += 1

        if self._step <= self._warmup_steps:
            lr = self._effective_lr * self._step / self._warmup_steps
        else:
            adjusted = self._step - self._warmup_steps
            total = (
                self.num_training_steps or DEFAULT_NUM_TRAINING_STEPS
            ) - self._warmup_steps
            total = max(total, 1)
            lr = self._compute_lr_step_based(adjusted, total)

        self._set_lr(lr)
        return lr

    def _compute_lr_after_warmup(self, adjusted_epoch: int) -> float:
        """Compute LR for epoch-based schedules after warmup.

        Args:
            adjusted_epoch: Epochs elapsed since warmup ended.

        Returns:
            Computed learning rate (clamped to ``min_lr``).
        """
        stype = self.config.scheduler_type
        eff_lr = self._effective_lr
        min_lr = self.config.min_lr

        if stype == "cosine_restart":
            t_cur = self.config.t_0
            epoch = adjusted_epoch
            while epoch >= t_cur:
                epoch -= t_cur
                t_cur = int(t_cur * self.config.t_mult)
            lr = min_lr + (eff_lr - min_lr) * (
                1 + math.cos(math.pi * epoch / t_cur)
            ) / 2
            return max(lr, min_lr)

        if stype == "cosine":
            total = max(
                self.config.max_epochs - self.config.warmup_epochs, 1,
            )
            progress = min(adjusted_epoch / total, 1.0)
            lr = min_lr + (eff_lr - min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2
            return max(lr, min_lr)

        if stype == "linear":
            total = max(
                self.config.max_epochs - self.config.warmup_epochs, 1,
            )
            progress = min(adjusted_epoch / total, 1.0)
            lr = eff_lr * (1 - progress)
            return max(lr, min_lr)

        if stype == "polynomial":
            total = max(
                self.config.max_epochs - self.config.warmup_epochs, 1,
            )
            progress = min(adjusted_epoch / total, 1.0)
            lr = eff_lr * ((1 - progress) ** 2.0)
            return max(lr, min_lr)

        if stype == "constant":
            return eff_lr

        return eff_lr

    def _compute_lr_step_based(
        self, step: int, total: int,
    ) -> float:
        """Compute LR for step-based schedules.

        Args:
            step: Steps elapsed since warmup ended.
            total: Total post-warmup steps.

        Returns:
            Computed learning rate (clamped to ``min_lr``).
        """
        stype = self.config.scheduler_type
        eff_lr = self._effective_lr
        min_lr = self.config.min_lr
        progress = min(step / total, 1.0)

        if stype == "cosine_restart":
            t_cur = self.config.t_0
            s = step
            while s >= t_cur:
                s -= t_cur
                t_cur = int(t_cur * self.config.t_mult)
            lr = min_lr + (eff_lr - min_lr) * (
                1 + math.cos(math.pi * s / t_cur)
            ) / 2
            return max(lr, min_lr)

        if stype == "cosine":
            lr = min_lr + (eff_lr - min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2
            return max(lr, min_lr)

        if stype == "linear":
            lr = eff_lr * (1 - progress)
            return max(lr, min_lr)

        if stype == "polynomial":
            lr = eff_lr * ((1 - progress) ** 2.0)
            return max(lr, min_lr)

        if stype == "constant":
            return eff_lr

        return eff_lr

    def reduce_lr(self, factor: float = 0.5) -> float:
        """Reduce effective LR by *factor*.

        The reduction persists through subsequent ``step`` calls.

        Args:
            factor: Multiplicative factor (e.g. 0.5 halves LR).

        Returns:
            New learning rate.
        """
        self._lr_scale *= factor
        new_lr = max(self.get_lr() * factor, self.config.min_lr)
        self._set_lr(new_lr)
        return new_lr

    def surge_lr(self, factor: float = 3.0) -> float:
        """Increase effective LR by *factor* (capped at ``initial_lr``).

        Args:
            factor: Multiplicative factor.

        Returns:
            New learning rate.
        """
        self._lr_scale = min(
            self._lr_scale * factor,
            self.initial_lr / max(self._base_lr, 1e-12),
        )
        new_lr = min(self.get_lr() * factor, self.initial_lr)
        self._set_lr(new_lr)
        return new_lr

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _set_lr(self, lr: float) -> None:
        """Set LR for all optimizer parameter groups."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary with internal state.
        """
        return {
            "_step": self._step,
            "_epoch": self._epoch,
            "_lr_scale": self._lr_scale,
            "_base_lr": self._base_lr,
            "initial_lr": self.initial_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scheduler state from checkpoint.

        Args:
            state_dict: State dictionary from ``state_dict()``.
        """
        self._step = state_dict.get("_step", 0)
        self._epoch = state_dict.get("_epoch", 0)
        self._lr_scale = state_dict.get("_lr_scale", 1.0)
        self._base_lr = state_dict.get("_base_lr", self._base_lr)
        self.initial_lr = state_dict.get(
            "initial_lr", self.initial_lr,
        )


def get_transformer_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Build a HuggingFace-compatible scheduler.

    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Scheduler name (e.g. ``"linear"``, ``"cosine"``).
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.

    Returns:
        Scheduler instance from ``transformers.get_scheduler``.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """
    try:
        from transformers import get_scheduler

        return get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    except ImportError:
        raise ImportError(
            "Install transformers: pip install transformers"
        ) from None