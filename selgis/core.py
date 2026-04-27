"""SELGIS core: training protection and optimization."""

from __future__ import annotations

import itertools
import logging
import warnings
from collections import deque
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from selgis.config import SelgisConfig
from selgis.scheduler import SmartScheduler

logger = logging.getLogger(__name__)


class SelgisCore:
    """Training protection and optimization core.

    Features: NaN/Inf and loss spike protection, automatic rollback
    on anomalies, early stopping with final surge, gradient clipping,
    memory-efficient state management (trainable parameters only),
    and optional CPU offload for optimizer states.
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
        self._cpu_offload: bool = getattr(config, "cpu_offload", False)

        self._state_storage: str = getattr(config, "state_storage", "disk")
        if self._state_storage not in {"disk", "memory"}:
            raise ValueError("state_storage must be 'disk' or 'memory'")

        self._state_dir: Path | None = None
        if self._state_storage == "disk":
            base_dir = getattr(config, "state_dir", None) or config.output_dir
            self._state_dir = Path(base_dir) / "selgis_state"
            self._state_dir.mkdir(parents=True, exist_ok=True)

        self._loss_history: deque[float] = deque(
            maxlen=max(self.config.min_history_len * 10, 1000),
        )
        self._higher_is_better: bool | None = None
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
        logger.info(
            "Parameters: %d trainable / %d total (%.2f%%)",
            trainable_params,
            total_params,
            100 * trainable_params / total_params if total_params > 0 else 0.0,
        )

        if self._cpu_offload:
            self._setup_cpu_offload()

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

        if self._cpu_offload:
            logger.info("CPU Offload enabled for optimizer states")

    def _check_is_peft_model(self) -> bool:
        """Return True if the model is a PEFT (e.g. LoRA) model."""
        try:
            from peft import PeftModel

            return isinstance(self.model, PeftModel)
        except ImportError:
            return False

    def _setup_cpu_offload(self) -> None:
        """Set up CPU offload for optimizer states.

        Optimizer state offload is handled lazily after each optimizer
        step, because optimizer states use lazy initialization and are
        empty until the first ``optimizer.step()`` call.

        Gradient offload is not used because PyTorch 2.x enforces
        device consistency between parameters and their gradients.
        For LoRA models, gradients are small enough to stay on GPU.
        """
        logger.debug("CPU offload initialized")

    def _onload_optimizer_state_to_device(self) -> None:
        """Move optimizer state tensors to compute device before step."""
        if not self._cpu_offload:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if not param.requires_grad:
                    continue
                if param not in self.optimizer.state:
                    continue
                state = self.optimizer.state[param]
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.device != param.device:
                        state[key] = val.to(param.device)

    def _offload_optimizer_state(self) -> None:
        """Move all optimizer state tensors to CPU after step."""
        if not self._cpu_offload:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if not param.requires_grad:
                    continue
                if param not in self.optimizer.state:
                    continue
                state = self.optimizer.state[param]
                for key, val in state.items():
                    if isinstance(val, torch.Tensor) and val.device.type != "cpu":
                        state[key] = val.to("cpu")

    def _get_trainable_param_names(self) -> set[str]:
        """Return names of all trainable parameters."""
        return {name for name, param in self.model.named_parameters() if param.requires_grad}

    def _clone_trainable_state(self) -> dict:
        """Clone only trainable parameters to CPU.

        For LoRA/PEFT models where most weights are frozen this saves
        significant memory by skipping frozen base model weights.
        When a parameter already resides on CPU ``clone`` is used;
        otherwise ``cpu`` already produces an independent copy.
        """
        state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.device.type == "cpu":
                    state[name] = param.detach().clone()
                else:
                    state[name] = param.detach().cpu()
        return state

    def _load_trainable_state(self, state: dict) -> None:
        """Load only trainable parameters from *state* with bitwise precision.

        Uses ``copy_`` which handles cross-device transfers natively
        without creating intermediate tensors.
        """
        current_state = self.model.state_dict()
        trainable_names = self._trainable_param_names

        missing_in_model = [k for k in state if k not in current_state]
        missing_in_state = [k for k in trainable_names if k not in state]

        if missing_in_model:
            warnings.warn(
                "SelgisCore: state has keys not in model (ignored): "
                f"{missing_in_model[:5]}"
                + (
                    f" ... and {len(missing_in_model) - 5} more"
                    if len(missing_in_model) > 5
                    else ""
                ),
                UserWarning,
                stacklevel=2,
            )
        if missing_in_state:
            warnings.warn(
                "SelgisCore: model has trainable keys not in state "
                f"(will keep current): {missing_in_state[:5]}"
                + (
                    f" ... and {len(missing_in_state) - 5} more"
                    if len(missing_in_state) > 5
                    else ""
                ),
                UserWarning,
                stacklevel=2,
            )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state and param.requires_grad:
                    param.data.copy_(state[name])

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
            self._load_trainable_state(self._last_good_state)
            return

        state_path = Path(self._last_good_state)
        if state_path.is_file():
            state_dict = torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
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
        """Load best saved weights.

        Returns:
            True if weights were loaded, False if no saved state exists.
        """
        if self._best_state is None:
            return False

        if self._state_storage == "memory":
            self._load_trainable_state(self._best_state)
            return True

        state_path = Path(self._best_state)
        if not state_path.is_file():
            return False

        state_dict = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
        self._load_trainable_state(state_dict)
        return True

    def _rollback(self, reason: str) -> None:
        """Rollback to last stable state, reset optimizer momentum, and reduce LR."""
        logger.warning("Rollback triggered: %s", reason)
        self._load_last_good_state()

        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param in self.optimizer.state:
                    del self.optimizer.state[param]

        if hasattr(self.scheduler, "reduce_lr"):
            new_lr = self.scheduler.reduce_lr()
            logger.info("LR reduced to %.2e", new_lr)

        self._steps_since_last_state = 0

    def check_loss(self, loss: torch.Tensor) -> bool:
        """Check loss for anomalies (NaN, Inf, or spike).

        Returns:
            True if the loss is normal, False if a rollback was triggered.
        """
        if not self.config.nan_recovery:
            return True

        if torch.isnan(loss) or torch.isinf(loss):
            self._rollback("NaN/Inf loss detected")
            return False

        loss_val = loss.item()
        min_len = self.config.min_history_len

        if len(self._loss_history) >= min_len:
            recent = list(itertools.islice(reversed(self._loss_history), min_len))
            avg = sum(recent) / min_len
            threshold = self.config.spike_threshold * avg

            if loss_val > threshold:
                self._rollback(f"Spike: {loss_val:.3f} > {threshold:.3f}")
                return False

        self._loss_history.append(loss_val)
        return True

    def backward_step(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
    ) -> None:
        """Backward pass with optional mixed-precision scaling."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _clip_grad_norm(self, max_norm: float) -> None:
        """Clip gradient L2 norm without allocating full fp32 gradient copies.

        Handles parameters distributed across multiple devices by
        accumulating per-device norms on CPU.
        """
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        if not parameters:
            return

        total_norm_sq = 0.0
        for param in parameters:
            param_norm = param.grad.detach().norm(2)
            total_norm_sq += (
                param_norm.to(
                    device="cpu",
                    dtype=torch.float32,
                ).item()
                ** 2
            )

        total_norm = total_norm_sq**0.5
        if total_norm == 0.0:
            return

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef >= 1.0:
            return

        for param in parameters:
            param.grad.detach().mul_(clip_coef)

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping, AMP, and CPU offload.

        The caller must invoke ``optimizer.zero_grad()`` before the
        corresponding ``backward_step`` call.
        """
        if self._cpu_offload:
            self._onload_optimizer_state_to_device()

        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)

        if self.config.grad_clip_norm > 0:
            self._clip_grad_norm(max_norm=self.config.grad_clip_norm)

        if self.config.grad_clip_value is not None:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_value_(
                trainable_params,
                clip_value=self.config.grad_clip_value,
            )

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        if self._cpu_offload:
            self._offload_optimizer_state()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._steps_since_last_state += 1
        if self._steps_since_last_state >= max(
            self._state_update_interval,
            1,
        ):
            self._save_last_good_state()
            self._steps_since_last_state = 0

    def eval_epoch(
        self,
        metrics: dict[str, float],
        epoch: int,
        primary_metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Literal["IMPROVED", "SURGE", "STOP", "CONTINUE"]:
        """Evaluate epoch result.

        Returns:
            One of ``'IMPROVED'``, ``'SURGE'``, ``'STOP'``,
            or ``'CONTINUE'``.
        """
        if hasattr(self.scheduler, "step_epoch"):
            self.scheduler._epoch = epoch

        metric_val = metrics.get(primary_metric, 0.0)
        val_loss = metrics.get("loss", float("inf"))

        if self._higher_is_better is None:
            self._higher_is_better = higher_is_better
            self._best_metric = float("-inf") if higher_is_better else float("inf")

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
            if (
                not self._surge_done
                and hasattr(self.scheduler, "surge_lr")
                and self.config.final_surge_factor > 0
            ):
                logger.info(
                    "Final surge triggered (factor=%.1f)",
                    self.config.final_surge_factor,
                )
                self.scheduler.surge_lr(
                    factor=self.config.final_surge_factor,
                )
                self._surge_done = True
                self._no_improve = 0
                return "SURGE"
            return "STOP"

        return "CONTINUE"

    def load_best_weights(self) -> bool:
        """Load best saved weights.

        Returns:
            True if weights were loaded, False otherwise.
        """
        loaded = self._load_best_state()
        if loaded:
            logger.info("Best weights loaded")
        return loaded

    def get_amp_context(self) -> AbstractContextManager:
        """Return autocast context for mixed precision, or ``nullcontext``."""
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

    def enable_flash_attention(self) -> bool:
        """Enable Flash Attention 2 (Flash SDP) for GPU.

        Uses PyTorch's scaled_dot_product_attention with Flash kernel.
        Requires:
        - CUDA device with compute capability >= 8.0 (Ampere or newer)
        - PyTorch >= 2.0 with CUDA

        Returns:
            True if Flash Attention was enabled, False if not available.
        """
        if self.device.type != "cuda":
            logger.debug("Flash Attention requires CUDA device")
            return False

        try:
            import torch.backends.cuda
        except ImportError:
            return False

        cuda_capable = torch.cuda.is_available()
        if not cuda_capable:
            logger.debug("CUDA not available")
            return False

        sm = torch.cuda.get_device_capability(self.device)
        if sm[0] < 8:
            logger.debug(f"Flash Attention requires compute capability >= 8.0, got {sm[0]}.{sm[1]}")
            self._enable_mem_efficient_attention()
            return False

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        logger.info(f"Flash Attention enabled (compute capability {sm[0]}.{sm[1]})")
        return True

    def _enable_mem_efficient_attention(self) -> bool:
        """Enable memory-efficient attention (SDPA fallback).

        Falls back to memory-efficient kernel when Flash is unavailable.
        Works on older GPUs but is slower.
        """
        if self.device.type != "cuda":
            return False

        try:
            import torch.backends.cuda

            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Memory-efficient attention enabled (fallback)")
            return True
        except ImportError:
            return False

    def is_flash_attention_enabled(self) -> bool:
        """Check if Flash Attention is currently enabled."""
        try:
            import torch.backends.cuda

            return torch.backends.cuda.flash_sdp_enabled()
        except (ImportError, AttributeError):
            return False

    def is_mem_efficient_attention_enabled(self) -> bool:
        """Check if memory-efficient attention is enabled."""
        try:
            import torch.backends.cuda

            return torch.backends.cuda.mem_efficient_sdp_enabled()
        except (ImportError, AttributeError):
            return False
