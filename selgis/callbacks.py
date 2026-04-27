"""Callback system for training hooks."""

from __future__ import annotations

import json
import logging
import shutil
from abc import ABC
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from selgis.trainer import Trainer


class Callback(ABC):
    """Base callback with no-op defaults for all hooks."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called once at the start of training."""

    def on_train_end(self, trainer: Any) -> None:
        """Called once at the end of training."""

    def on_epoch_begin(
        self,
        trainer: Trainer,
        epoch: int,
    ) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called at the end of each epoch with computed metrics."""

    def on_step_begin(
        self,
        trainer: Trainer,
        step: int,
    ) -> None:
        """Called before each training step."""

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
    ) -> None:
        """Called after each training step."""

    def on_evaluate(
        self,
        trainer: Trainer,
        metrics: dict[str, float],
    ) -> None:
        """Called after evaluation."""


class EarlyStoppingCallback(Callback):
    """Early stopping based on a monitored metric.

    Args:
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        metric: Metric key to monitor.
        mode: ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        metric: str = "loss",
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.history = []
        self._best = float("inf") if mode == "min" else float("-inf")
        self._counter = 0
        self.should_stop = False

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Check metric improvement; set stop flag if exhausted."""
        self.history.append(
            {
                "epoch": epoch,
                "global_step": getattr(trainer, "_global_step", 0),
                "metrics": metrics,
            },
        )

        value = metrics.get(self.metric, 0.0)

        if self.mode == "min":
            improved = value < self._best - self.min_delta
        else:
            improved = value > self._best + self.min_delta

        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.should_stop = True
            logger.info(
                "Early stopping: no improvement for %d epochs",
                self.patience,
            )


BEST_MODEL_DIRNAME = "best_model"


class CheckpointCallback(Callback):
    """Save model checkpoints with optional rotation.

    Args:
        output_dir: Base directory for checkpoints.
        save_best_only: If True, only save on metric improvement.
        save_total_limit: Maximum number of regular checkpoints.
        metric: Metric key to determine best checkpoint.
        mode: ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        output_dir: str,
        save_best_only: bool = True,
        save_total_limit: int = 3,
        metric: str = "loss",
        mode: str = "min",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_best_only = save_best_only
        self.save_total_limit = save_total_limit
        self.metric = metric
        self.mode = mode
        self._best = float("inf") if mode == "min" else float("-inf")
        self._saved_checkpoints: list[Path] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save checkpoint (and best checkpoint) when appropriate."""
        value = metrics.get(self.metric, 0.0)

        if self.mode == "min":
            is_best = value < self._best
        else:
            is_best = value > self._best

        if is_best:
            self._best = value

        if self.save_best_only and not is_best:
            return

        ckpt_path = self.output_dir / f"checkpoint-epoch-{epoch}"
        self._save_checkpoint(trainer, ckpt_path, metrics, epoch)
        self._saved_checkpoints.append(ckpt_path)

        if is_best:
            best_path = self.output_dir / BEST_MODEL_DIRNAME
            self._save_checkpoint(trainer, best_path, metrics, epoch)

        self._cleanup()

    def _save_checkpoint(
        self,
        trainer: Trainer,
        path: Path,
        metrics: dict[str, float],
        epoch: int,
    ) -> None:
        """Save model weights, optimizer, scheduler, and metadata.

        Only trainable parameters are saved for the model to reduce
        checkpoint size when using PEFT/LoRA.
        """
        path.mkdir(parents=True, exist_ok=True)

        is_peft = getattr(getattr(trainer, "selgis", None), "_is_peft_model", False)
        if is_peft:
            state_dict = {}
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    if param.device.type == "cpu":
                        state_dict[name] = param.detach().clone()
                    else:
                        state_dict[name] = param.detach().cpu()
            state_format = "trainable_only"
        else:
            state_dict = {}
            for name, tensor in trainer.model.state_dict().items():
                if isinstance(tensor, torch.Tensor):
                    if tensor.device.type == "cpu":
                        state_dict[name] = tensor.detach().clone()
                    else:
                        state_dict[name] = tensor.detach().cpu()
                else:
                    state_dict[name] = tensor
            state_format = "full"
        torch.save(state_dict, path / "model.pt")

        torch.save(
            trainer.optimizer.state_dict(),
            path / "optimizer.pt",
        )

        if (
            hasattr(trainer, "scheduler")
            and trainer.scheduler is not None
            and hasattr(trainer.scheduler, "state_dict")
        ):
            torch.save(
                trainer.scheduler.state_dict(),
                path / "scheduler.pt",
            )

        global_step = getattr(trainer, "_global_step", 0)
        meta = {
            "epoch": epoch,
            "global_step": global_step,
            "state_format": state_format,
            **metrics,
        }
        with open(path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved checkpoint: %s", path)

    def _cleanup(self) -> None:
        """Remove oldest regular checkpoints beyond the limit.

        The ``best_model`` directory is never removed.
        """
        best_resolved = (self.output_dir / BEST_MODEL_DIRNAME).resolve()
        non_best = [p for p in self._saved_checkpoints if p.resolve() != best_resolved]

        while len(non_best) > self.save_total_limit:
            oldest = non_best.pop(0)
            self._saved_checkpoints.remove(oldest)
            if oldest.exists():
                shutil.rmtree(oldest)


class HistoryCallback(Callback):
    """Save training history to a JSON file.

    Args:
        output_dir: Directory where ``training_history.json`` is saved.
    """

    def __init__(self, output_dir: str = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.history: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Record epoch metrics."""
        self.history.append(
            {
                "epoch": epoch,
                "global_step": getattr(trainer, "_global_step", 0),
                "metrics": metrics,
            },
        )

    def on_train_end(self, trainer: Any) -> None:
        """Write full history to disk."""
        path = self.output_dir / "training_history.json"

        try:
            config_dict = asdict(trainer.config)
        except (TypeError, ValueError, AttributeError):
            config_dict = str(trainer.config)

        serializable_config: dict[str, Any] = {}
        if isinstance(config_dict, dict):
            for k, v in config_dict.items():
                try:
                    json.dumps(v)
                    serializable_config[k] = v
                except (TypeError, ValueError):
                    serializable_config[k] = str(v)
        else:
            serializable_config = {"raw": config_dict}

        info = {
            "config": serializable_config,
            "model_type": type(trainer.model).__name__,
            "total_epochs": len(self.history),
            "final_metrics": (self.history[-1]["metrics"] if self.history else {}),
            "history": self.history,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info("Training history saved: %s", path)


class LoggingCallback(Callback):
    """Console logging of step losses and epoch metrics.

    Args:
        log_every: Log every *n* steps.
    """

    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every

    def on_train_begin(self, trainer: Trainer) -> None:
        """Print training start banner."""
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer) -> None:
        """Print training end banner."""
        logger.info("Training complete")

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Print epoch summary metrics."""
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()
        )
        logger.info("Epoch %d | %s", epoch, metrics_str)

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
    ) -> None:
        """Print step loss and LR at configured intervals."""
        if step > 0 and step % self.log_every == 0:
            lr = trainer.optimizer.param_groups[0]["lr"]
            logger.debug("Step %6d | Loss: %.4f | LR: %.2e", step, loss, lr)


class WandBCallback(Callback):
    """Weights and Biases logging.

    Args:
        project: W&B project name.
        name: Optional run name.
        config: Optional config dict to log.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self._wandb = None

    def on_train_begin(self, trainer: Trainer) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
            )
        except ImportError:
            logger.warning("wandb not installed, skipping WandB logging")

    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        loss: float,
    ) -> None:
        """Log step loss."""
        if self._wandb is not None:
            self._wandb.log(
                {"train/loss": loss, "train/step": step},
            )

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Log epoch metrics."""
        if self._wandb is not None:
            self._wandb.log(
                {
                    "epoch": epoch,
                    **{f"eval/{k}": v for k, v in metrics.items()},
                },
            )

    def on_train_end(self, trainer: Trainer) -> None:
        """Finish W&B run."""
        if self._wandb is not None:
            self._wandb.finish()


class SparsityCallback(Callback):
    """Magnitude pruning for trainable parameters.

    Applies per-layer unstructured pruning to ``Linear`` and ``Conv``
    modules.  Suited for LoRA/PEFT where most weights are frozen.

    Args:
        target_sparsity: Fraction of weights to zero (0.0-1.0).
        start_epoch: First epoch to apply pruning.
        frequency: Apply every *n* epochs after ``start_epoch``.
        skip_lora: Skip LoRA adapter layers.
        min_params_to_prune: Minimum layer size to consider.
        log_details: Print per-layer sparsity.
    """

    _SKIP_SUFFIXES = ("bias", "norm", "layer_norm")
    _SKIP_PREFIXES = ("embed",)

    def __init__(
        self,
        target_sparsity: float = 0.5,
        start_epoch: int = 0,
        frequency: int = 1,
        skip_lora: bool = True,
        min_params_to_prune: int = 1000,
        log_details: bool = False,
    ) -> None:
        if not 0.0 <= target_sparsity <= 1.0:
            raise ValueError("target_sparsity must be in [0.0, 1.0]")

        self.target_sparsity = target_sparsity
        self.start_epoch = start_epoch
        self.frequency = max(frequency, 1)
        self.skip_lora = skip_lora
        self.min_params_to_prune = min_params_to_prune
        self.log_details = log_details
        self._applied_epochs: list[int] = []

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Apply magnitude pruning if the epoch matches the schedule."""
        if epoch < self.start_epoch:
            return
        if (epoch - self.start_epoch) % self.frequency != 0:
            return
        if self.target_sparsity <= 0:
            return

        pruned_layers = self._apply_local_pruning(trainer.model)

        if pruned_layers > 0:
            self._applied_epochs.append(epoch)
            total_sparsity = self._compute_model_sparsity(trainer.model)
            logger.info(
                "Sparsity applied: target=%.2f%%, actual=%.2f%%, pruned_layers=%d",
                self.target_sparsity * 100,
                total_sparsity * 100,
                pruned_layers,
            )

    def _should_skip(self, name: str) -> bool:
        """Return True if the named module should not be pruned."""
        lower = name.lower()
        if self.skip_lora and "lora" in lower:
            return True
        parts = lower.rsplit(".", 1)
        last_part = parts[-1] if parts else lower
        return last_part in self._SKIP_SUFFIXES or any(
            last_part.startswith(p) for p in self._SKIP_PREFIXES
        )

    def _apply_local_pruning(self, model: nn.Module) -> int:
        """Apply magnitude pruning to eligible layers."""
        pruned_count = 0
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            if not hasattr(module, "weight") or not module.weight.requires_grad:
                continue
            if self._should_skip(name):
                continue
            if module.weight.numel() < self.min_params_to_prune:
                continue
            self._prune_layer(module, name)
            pruned_count += 1
        return pruned_count

    def _prune_layer(self, module: nn.Module, name: str) -> None:
        """Zero the smallest weights in a single layer."""
        with torch.no_grad():
            weight = module.weight.data
            flat = weight.abs().view(-1)
            k = int(flat.numel() * self.target_sparsity)
            if k <= 0 or k >= flat.numel():
                return
            threshold = torch.kthvalue(flat, k).values.item()
            mask = weight.abs() > threshold
            weight.mul_(mask)
            if self.log_details:
                actual = (weight == 0).sum().item() / weight.numel()
                logger.debug("Layer %s: sparsity=%.2f%%", name, actual * 100)

    @staticmethod
    def _compute_model_sparsity(model: nn.Module) -> float:
        """Compute fraction of zero trainable parameters."""
        total_params = 0
        zero_params = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            total_params += param.numel()
            zero_params += (param.data == 0).sum().item()
        if total_params == 0:
            return 0.0
        return zero_params / total_params
