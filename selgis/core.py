"""SELGIS core: training protection and optimization."""
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal
import torch
import torch.nn as nn
from selgis.config import SelgisConfig
from selgis.scheduler import SmartScheduler


class SelgisCore:
    """
    Training protection and optimization core.
    Features: NaN/Inf and loss spike protection, automatic rollback on anomalies,
    early stopping with final surge, gradient clipping, memory-efficient state
    management (trainable parameters only).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: SmartScheduler | Any,
        config: SelgisConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self._is_peft_model = self._check_is_peft_model()

        self._state_storage: str = getattr(config, "state_storage", "disk")
        if self._state_storage not in {"disk", "memory"}:
            raise ValueError("state_storage must be 'disk' or 'memory'")

        self._state_dir: Path | None = None
        if self._state_storage == "disk":
            base_dir = getattr(config, "state_dir", None) or config.output_dir
            self._state_dir = Path(base_dir) / "selgis_state"
            self._state_dir.mkdir(parents=True, exist_ok=True)

        self._loss_history: list[float] = []
        self._max_loss_history: int = max(self.config.min_history_len * 10, 1000)
        self._best_metric = float("-inf")
        self._best_loss = float("inf")
        self._best_state: dict | str | None = None
        self._no_improve = 0
        self._surge_done = False
        self._last_good_state: dict | str | None = None
        self._state_update_interval: int = getattr(
            config,
            "state_update_interval",
            100,
        )
        self._steps_since_last_state: int = 0

        self._trainable_param_names: set[str] = self._get_trainable_param_names()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Parameters: {trainable_params:,} trainable / {total_params:,} total  "
              f"({100 * trainable_params / total_params:.2f}%)")

        self._save_last_good_state()

        self._scaler = None
        if config.fp16 and device.type == "cuda":
            self._scaler = torch.amp.GradScaler("cuda")

        if config.fp16:
            self._amp_dtype = torch.float16
        elif config.bf16:
            self._amp_dtype = torch.bfloat16
        else:
            self._amp_dtype = None

    def _check_is_peft_model(self) -> bool:
        """Return True if the model is a PEFT (e.g. LoRA) model."""
        try:
            from peft import PeftModel
            return isinstance(self.model, PeftModel)
        except ImportError:
            return False

    def _get_trainable_param_names(self) -> set[str]:
        """Return names of all trainable parameters."""
        return {
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _clone_trainable_state(self) -> dict:
        """
        Clone only trainable parameters to CPU. Critical for LoRA/PEFT where
        most weights are frozen; saves ~0.1% of params instead of full model.
        """
        state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Используем .detach() вместо устаревшего .data
                state[name] = param.detach().cpu().clone()
        return state

    def _load_trainable_state(self, state: dict) -> None:
        """
        Load only trainable parameters from state dict with BITWISE PRECISION.
        """
        current_state = self.model.state_dict()
        trainable_names = self._trainable_param_names

        missing_in_model = [k for k in state if k not in current_state]
        missing_in_state = [k for k in trainable_names if k not in state]

        if missing_in_model:
            warnings.warn(
                f"SelgisCore: state has keys not in model (ignored): {missing_in_model[:5]}"
                + (f" ... and {len(missing_in_model) - 5} more" if len(missing_in_model) > 5 else ""),
                UserWarning,
                stacklevel=2,
            )
        if missing_in_state:
            warnings.warn(
                f"SelgisCore: model has trainable keys not in state (will keep current): {missing_in_state[:5]}"
                + (f" ... and {len(missing_in_state) - 5} more" if len(missing_in_state) > 5 else ""),
                UserWarning,
                stacklevel=2,
            )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state and param.requires_grad:
                    param.data.copy_(state[name].to(param.device, non_blocking=False))

    def _save_last_good_state(self) -> None:
        """Save last stable model state (trainable params only)."""
        if self._state_storage == "memory":
            self._last_good_state = self._clone_trainable_state()
            return

        if self._state_dir is None:
            return

        path = self._state_dir / "last_good_state.pt"
        trainable_state = self._clone_trainable_state()
        torch.save(trainable_state, path)
        self._last_good_state = str(path)

    def _load_last_good_state(self) -> None:
        """Load last stable model state."""
        if self._last_good_state is None:
            return

        if self._state_storage == "memory":
            self._load_trainable_state(self._last_good_state)  # type: ignore[arg-type]
            return

        state_path = Path(self._last_good_state)
        if state_path.is_file():
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            self._load_trainable_state(state_dict)

    def _save_best_state(self) -> None:
        """Save best model weights (trainable params only)."""
        if self._state_storage == "memory":
            self._best_state = self._clone_trainable_state()
            return

        if self._state_dir is None:
            return

        path = self._state_dir / "best_state.pt"
        trainable_state = self._clone_trainable_state()
        torch.save(trainable_state, path)
        self._best_state = str(path)

    def _load_best_state(self) -> bool:
        """Load best saved weights. Returns True if loaded, False if none saved."""
        if self._best_state is None:
            return False

        if self._state_storage == "memory":
            self._load_trainable_state(self._best_state)  # type: ignore[arg-type]
            return True

        state_path = Path(self._best_state)
        if not state_path.is_file():
            return False

        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        self._load_trainable_state(state_dict)
        return True

    def _rollback(self, reason: str) -> None:
        """Rollback to last stable state and reduce LR."""
        print(f"\n[WARN] Rollback triggered: {reason}")
        self._load_last_good_state()

        if hasattr(self.scheduler, "reduce_lr"):
            new_lr = self.scheduler.reduce_lr()
            print(f"   LR reduced to {new_lr:.2e}")

    def check_loss(self, loss: torch.Tensor) -> bool:
        """
        Check loss for anomalies (NaN/Inf/spike). Returns True if OK, False if rollback needed.
        """
        if not self.config.nan_recovery:
            return True

        if torch.isnan(loss) or torch.isinf(loss):
            self._rollback("NaN/Inf loss detected")
            return False

        loss_val = loss.item()
        min_len = self.config.min_history_len

        if len(self._loss_history) >= min_len:
            recent = self._loss_history[-min_len:]
            avg = sum(recent) / len(recent)
            threshold = self.config.spike_threshold * avg

            if loss_val > threshold:
                self._rollback(f"Spike: {loss_val:.3f} > {threshold:.3f}")
                return False

        self._loss_history.append(loss_val)

        if len(self._loss_history) > self._max_loss_history:
            overflow = len(self._loss_history) - self._max_loss_history
            del self._loss_history[:overflow]

        return True

    def backward_step(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> None:
        """
        Backward pass with optional mixed precision. Gradient clipping in optimizer_step.
        """
        if self._scaler is not None:
            self._scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _clip_grad_norm(self, max_norm: float) -> None:
        """Memory-efficient gradient clipping; computes L2 norm incrementally."""
        parameters = [
            p for p in self.model.parameters()
            if p.grad is not None and p.requires_grad
        ]
        if not parameters:
            return

        total_norm_sq = 0.0
        for param in parameters:
            grad = param.grad.detach()
            total_norm_sq += grad.data.float().norm(2).item() ** 2

        total_norm = total_norm_sq ** 0.5

        if total_norm == 0.0:
            return

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef >= 1.0:
            return

        for param in parameters:
            if param.grad is not None:
                param.grad.detach().mul_(clip_coef)

    def optimizer_step(self) -> None:
        """Optimizer step with gradient clipping and optional AMP."""
        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)

        if self.config.grad_clip_norm > 0:
            self._clip_grad_norm(max_norm=self.config.grad_clip_norm)

        if self.config.grad_clip_value is not None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_value_(trainable_params, clip_value=self.config.grad_clip_value)

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        self._steps_since_last_state += 1
        if self._steps_since_last_state >= max(self._state_update_interval, 1):
            self._save_last_good_state()
            self._steps_since_last_state = 0

    def eval_epoch(
        self,
        metrics: dict[str, float],
        epoch: int,
        primary_metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Literal["IMPROVED", "SURGE", "STOP", "CONTINUE"]:
        """
        Evaluate epoch result. Returns IMPROVED, SURGE, STOP, or CONTINUE.
        """
        # Only step by epoch when using epoch-based schedule (warmup_ratio == 0).
        if hasattr(self.scheduler, "step_epoch") and getattr(self.config, "warmup_ratio", 0) == 0:
            self.scheduler.step_epoch(epoch)

        metric_val = metrics.get(primary_metric, 0.0)
        val_loss = metrics.get("loss", float("inf"))

        if higher_is_better:
            improved = metric_val > self._best_metric + self.config.min_delta
        else:
            improved = metric_val < self._best_metric - self.config.min_delta

        loss_improved = val_loss < self._best_loss - self.config.min_delta

        if improved or loss_improved:
            if higher_is_better:
                self._best_metric = max(metric_val, self._best_metric)
            else:
                self._best_metric = min(metric_val, self._best_metric)
            self._best_loss = min(val_loss, self._best_loss)
            self._save_best_state()
            self._no_improve = 0
            return "IMPROVED"

        self._no_improve += 1

        if self._no_improve >= self.config.patience:
            if not self._surge_done and hasattr(self.scheduler, "surge_lr"):
                print("\n[INFO] Final surge triggered")
                self.scheduler.surge_lr(factor=5.0)
                self._surge_done = True
                self._no_improve = 0
                return "SURGE"
            return "STOP"

        return "CONTINUE"

    def load_best_weights(self) -> bool:
        """Load best saved weights. Returns True if loaded, False otherwise."""
        loaded = self._load_best_state()
        if loaded:
            print("[INFO] Best weights loaded")
        return loaded

    def get_amp_context(self):
        """Return autocast context for mixed precision, or nullcontext if disabled."""
        if self._amp_dtype is not None and self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self._amp_dtype)
        if self._amp_dtype is not None and self.device.type == "cpu":
            return torch.amp.autocast("cpu", dtype=self._amp_dtype)
        return nullcontext()

    @property
    def best_metric(self) -> float:
        """Best primary metric value seen so far."""
        return self._best_metric

    @property
    def best_loss(self) -> float:
        """Best validation loss seen so far."""
        return self._best_loss

    @property
    def trainable_params_count(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def total_params_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())