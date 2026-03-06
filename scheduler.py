"""LR schedulers: warmup, cosine (with restarts), linear, polynomial, reduce/surge on plateau."""

import math
from typing import Literal

import torch.optim as optim

from selgis.config import SelgisConfig

# Fallback when step-based schedule is used but num_training_steps was not provided.
DEFAULT_NUM_TRAINING_STEPS = 10_000


class SmartScheduler:
    """
    Epoch- or step-based LR: warmup, cosine/cosine_restart, linear, constant, polynomial.
    reduce_lr / surge_lr for manual adjustment (e.g. after rollback or final surge).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        initial_lr: float,
        config: SelgisConfig,
        num_training_steps: int | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.config = config
        self.num_training_steps = num_training_steps

        self._step = 0
        self._epoch = 0

        if config.warmup_ratio > 0 and num_training_steps:
            self._warmup_steps = int(config.warmup_ratio * num_training_steps)
        else:
            self._warmup_steps = 0

    def step_epoch(self, epoch: int) -> float:
        """Update LR for epoch (epoch-based schedule). Returns new LR."""
        self._epoch = epoch
        warmup = self.config.warmup_epochs

        if epoch < warmup:
            lr = self.initial_lr * (epoch + 1) / warmup
        else:
            lr = self._compute_lr_after_warmup(epoch - warmup)

        self._set_lr(lr)
        return lr

    def step(self) -> float:
        """Update LR for step (step-based schedule). Returns new LR."""
        self._step += 1

        if self._step <= self._warmup_steps:
            lr = self.initial_lr * self._step / self._warmup_steps
        else:
            adjusted_step = self._step - self._warmup_steps
            total_steps = (self.num_training_steps or DEFAULT_NUM_TRAINING_STEPS) - self._warmup_steps
            lr = self._compute_lr_step_based(adjusted_step, total_steps)

        self._set_lr(lr)
        return lr

    def _compute_lr_after_warmup(self, adjusted_epoch: int) -> float:
        """Compute LR after warmup (epoch-based)."""
        stype = self.config.scheduler_type

        if stype == "cosine_restart":
            t_cur = self.config.t_0
            epoch = adjusted_epoch

            while epoch >= t_cur:
                epoch -= t_cur
                t_cur = int(t_cur * self.config.t_mult)

            return self.config.min_lr + (self.initial_lr - self.config.min_lr) * (
                1 + math.cos(math.pi * epoch / t_cur)
            ) / 2

        if stype == "cosine":
            total = self.config.max_epochs - self.config.warmup_epochs
            return self.config.min_lr + (self.initial_lr - self.config.min_lr) * (
                1 + math.cos(math.pi * adjusted_epoch / total)
            ) / 2

        if stype == "linear":
            total = self.config.max_epochs - self.config.warmup_epochs
            return self.initial_lr * (1 - adjusted_epoch / total)

        if stype == "constant":
            return self.initial_lr

        return self.initial_lr

    def _compute_lr_step_based(self, step: int, total: int) -> float:
        """Compute LR for step-based schedule."""
        stype = self.config.scheduler_type
        progress = min(step / total, 1.0)

        if stype in ("cosine", "cosine_restart"):
            return self.config.min_lr + (self.initial_lr - self.config.min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2

        if stype == "linear":
            return self.initial_lr * (1 - progress)

        if stype == "polynomial":
            power = 2.0
            return self.initial_lr * ((1 - progress) ** power)

        return self.initial_lr

    def reduce_lr(self, factor: float = 0.5) -> float:
        """Reduce LR by factor (e.g. after rollback). Returns new LR."""
        current = self.get_lr()
        new_lr = max(current * factor, self.config.min_lr)
        self._set_lr(new_lr)
        return new_lr

    def surge_lr(self, factor: float = 3.0) -> float:
        """Increase LR by factor (e.g. final surge). Returns new LR."""
        current = self.get_lr()
        new_lr = min(current * factor, self.initial_lr)
        self._set_lr(new_lr)
        return new_lr

    def get_lr(self) -> float:
        """Current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _set_lr(self, lr: float) -> None:
        """Set LR for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing and resume."""
        return {"_step": self._step, "_epoch": self._epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._step = state_dict.get("_step", 0)
        self._epoch = state_dict.get("_epoch", 0)


def get_transformer_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Build HuggingFace-compatible scheduler (requires transformers).

    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Scheduler name (e.g. linear, cosine).
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.

    Returns:
        Scheduler instance from transformers.get_scheduler.

    Raises:
        ImportError: If transformers is not installed.
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
        raise ImportError("Install transformers: pip install transformers")