"""Callback system for training hooks (logging, checkpoints, early stopping, WandB, sparsity)."""
from __future__ import annotations
import json
import shutil
from abc import ABC
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

class Callback(ABC):
    """Base callback; override any of on_train_begin/end, on_epoch_begin/end, on_step_begin/end, on_evaluate."""
    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        pass

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        pass

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        pass

    def on_step_end(
        self,
        trainer: "Trainer",
        step: int,
        loss: float,
    ) -> None:
        pass

    def on_evaluate(
        self,
        trainer: "Trainer",
        metrics: dict[str, float],
    ) -> None:
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping based on a metric (min or max)."""
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

        self._best = float("inf") if mode == "min" else float("-inf")
        self._counter = 0
        self._should_stop = False

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
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
            self._should_stop = True
            print(f"\n[INFO] Early stopping: no improvement for {self.patience} epochs")

    @property
    def should_stop(self) -> bool:
        return self._should_stop


class HistoryCallback(Callback):
    """Save comprehensive training history to JSON."""
    def __init__(self, output_dir: str = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.history: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        # Collect all metrics and info
        record = {
            "epoch": epoch,
            "global_step": getattr(trainer, "_global_step", 0),
            "metrics": metrics,
        }
        self.history.append(record)

    def on_train_end(self, trainer: "Trainer") -> None:
        path = self.output_dir / "training_history.json"
        
        # Collect system/config info
        info = {
            "config": getattr(trainer.config, "__dict__", str(trainer.config)),
            "model_type": type(trainer.model).__name__,
            "total_epochs": len(self.history),
            "final_metrics": self.history[-1]["metrics"] if self.history else {},
            "history": self.history
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] Training history saved: {path}")


BEST_MODEL_DIRNAME = "best_model"


class CheckpointCallback(Callback):
    """Save checkpoints (best and/or periodic); optional total limit."""
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
        self._protected_paths: set[Path] = set()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        value = metrics.get(self.metric, 0.0)

        if self.mode == "min":
            is_best = value < self._best
        else:
            is_best = value > self._best

        if is_best:
            self._best = value

        if self.save_best_only and not is_best:
            return

        # Save checkpoint
        ckpt_path = self.output_dir / f"checkpoint-epoch-{epoch}"
        self._save_checkpoint(trainer, ckpt_path, metrics, epoch)

        if is_best:
            best_path = self.output_dir / BEST_MODEL_DIRNAME
            self._save_checkpoint(trainer, best_path, metrics, epoch)
            self._protected_paths.add(best_path.resolve())

        self._cleanup()

    def _save_checkpoint(
        self,
        trainer: "Trainer",
        path: Path,
        metrics: dict[str, float],
        epoch: int,
    ) -> None:
        """Save model (trainable), optimizer, scheduler, and training step for resume."""
        path.mkdir(parents=True, exist_ok=True)

        state_dict = {}
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.cpu()
        torch.save(state_dict, path / "model.pt")

        torch.save(trainer.optimizer.state_dict(), path / "optimizer.pt")

        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            if hasattr(trainer.scheduler, "state_dict"):
                torch.save(trainer.scheduler.state_dict(), path / "scheduler.pt")

        global_step = getattr(trainer, "_global_step", 0)
        with open(path / "metrics.json", "w") as f:
            json.dump({"epoch": epoch, "global_step": global_step, **metrics}, f, indent=2)

        self._saved_checkpoints.append(path)
        print(f"[SAVE] Saved: {path}")

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond save_total_limit; keep protected (e.g. best_model)."""
        protected_resolved = {p.resolve() for p in self._protected_paths}
        regular = [p for p in self._saved_checkpoints if p.resolve() not in protected_resolved]

        while len(regular) > self.save_total_limit:
            oldest = regular.pop(0)
            if oldest.exists():
                shutil.rmtree(oldest)
            self._saved_checkpoints.remove(oldest)


class LoggingCallback(Callback):
    """Console logging: step loss/LR and epoch metrics."""
    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every
        self._step_losses: list[float] = []

    def on_train_begin(self, trainer: "Trainer") -> None:
        print("-" * 40)
        print("[INFO] Training started")
        print("-" * 40)

    def on_train_end(self, trainer: "Trainer") -> None:
        print("-" * 40)
        print("[INFO] Training complete")
        print("-" * 40)

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"[INFO] Epoch {epoch} | {metrics_str}")

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        if step % self.log_every == 0:
            lr = trainer.optimizer.param_groups[0]["lr"]
            print(f"  Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e}")


class WandBCallback(Callback):
    """Weights & Biases logging (step loss, epoch metrics)."""
    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self._run = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
            )
        except ImportError:
            print("[WARN] wandb not installed, skipping WandB logging")

    def on_step_end(self, trainer: "Trainer", step: int, loss: float) -> None:
        if self._run:
            import wandb
            wandb.log({"train/loss": loss, "train/step": step})

    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        if self._run:
            import wandb
            wandb.log({"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}})

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._run:
            import wandb
            wandb.finish()


class SparsityCallback(Callback):
    """
    Memory-safe magnitude pruning: per-layer, trainable params only.
    Suited for LoRA/PEFT where most weights are frozen.
    """
    def __init__(
        self,
        target_sparsity: float = 0.5,
        start_epoch: int = 0,
        frequency: int = 1,
        skip_lora: bool = True,
        min_params_to_prune: int = 1000,
        log_details: bool = False,
    ) -> None:
        """
        target_sparsity: fraction of weights to zero (0–1). start_epoch/frequency: when to apply.
        skip_lora: skip LoRA adapter params. min_params_to_prune: min layer size. log_details: per-layer log.
        """
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
        trainer: "Trainer",
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Apply magnitude pruning after epoch (if frequency matches)."""
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
            
            print(
                f"[INFO] Sparsity applied: target={self.target_sparsity:.2%},  "
                f"actual={total_sparsity:.2%}, pruned_layers={pruned_layers}"
            )

    def _apply_local_pruning(self, model: nn.Module) -> int:
        """Apply magnitude pruning per layer (Linear/Conv). Returns number of pruned layers."""
        pruned_count = 0

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            if not hasattr(module, "weight") or not module.weight.requires_grad:
                continue
            if self.skip_lora and "lora" in name.lower():
                continue
            if module.weight.numel() < self.min_params_to_prune:
                continue
            # Fixed: precise layer name matching to avoid false positives
            if any(skip in name.lower() for skip in ["bias", ".norm", ".layer_norm", ".embed"]):
                continue
            self._prune_layer(module, name)
            pruned_count += 1

        return pruned_count

    def _prune_layer(self, module: nn.Module, name: str) -> None:
        """Prune one layer in-place by magnitude (zero smallest weights)."""
        with torch.no_grad():
            weight = module.weight.data
            flat = weight.abs().view(-1)
            k = int(flat.numel() * self.target_sparsity)

            if k > 0 and k < flat.numel():
                threshold = torch.kthvalue(flat, k).values.item()
                mask = weight.abs() >= threshold
                weight.mul_(mask)

                if self.log_details:
                    actual_sparsity = (weight == 0).sum().item() / weight.numel()
                    print(f"      Layer {name}: sparsity={actual_sparsity:.2%}")

    def _compute_model_sparsity(self, model: nn.Module) -> float:
        """Compute fraction of zero trainable params (0–1), memory-safe."""
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

    def get_layer_sparsity(self, model: nn.Module) -> dict[str, float]:
        """Return dict of layer_name -> sparsity (fraction of zeros) for trainable params."""
        result = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if param.numel() == 0:
                continue
                
            zeros = (param.data == 0).sum().item()
            sparsity = zeros / param.numel()
            result[name] = sparsity
            
        return result