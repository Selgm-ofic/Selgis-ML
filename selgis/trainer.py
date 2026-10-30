"""Universal trainers for PyTorch and HuggingFace Transformers."""

from __future__ import annotations

import gc
import inspect
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from selgis.callbacks import (
    Callback,
    CheckpointCallback,
    HistoryCallback,
    LoggingCallback,
    SparsityCallback,
)
from selgis.checkpointing import GradientCheckpointingManager
from selgis.config import SelgisConfig, TransformerConfig
from selgis.core import SelgisCore
from selgis.loss import CrossEntropyLossV2
from selgis.lr_finder import LRFinder
from selgis.scheduler import SmartScheduler
from selgis.utils import (
    call_model_with_dict_inputs,
    get_device,
    get_optimizer_grouped_params,
    is_dict_like,
    move_to_device,
    seed_everything,
    unpack_batch,
)

logger = logging.getLogger(__name__)

# Metrics where a LOWER value is better; everything else is treated as
# "higher is better" when used as ``primary_metric``.
LOWER_IS_BETTER_METRICS = frozenset(
    {"loss", "perplexity", "ppl", "mse", "mae", "rmse", "mape", "smape", "wer", "cer"}
)


def _accepts_token_type_ids(model: nn.Module) -> bool:
    """Return True if *model*'s forward accepts ``token_type_ids``.

    Models like Llama/Mistral/Qwen have no ``token_type_ids`` input and
    raise ``TypeError`` if it is passed, so it must only be injected
    when the model actually supports it.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    if "token_type_ids" in params:
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        # **kwargs-style forward: use the config as a hint
        cfg = getattr(model, "config", None)
        return bool(getattr(cfg, "type_vocab_size", 0))
    return False


def _add_token_type_ids(inputs_dict: dict, model: nn.Module) -> None:
    """Add zero ``token_type_ids`` to *inputs_dict* if missing and supported."""
    if "token_type_ids" in inputs_dict:
        return
    input_ids = inputs_dict.get("input_ids")
    if input_ids is not None and _accepts_token_type_ids(model):
        inputs_dict["token_type_ids"] = torch.zeros_like(input_ids)




class Trainer:
    """Universal trainer for PyTorch models.

    Supports any architecture, custom forward functions, callbacks,
    mixed precision, LR finder, and gradient accumulation.

    Device management is handled internally by ``_forward``.
    The caller must not move batches to the device manually.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SelgisConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        criterion: nn.Module | None = None,
        optimizer: optim.Optimizer | None = None,
        callbacks: list[Callback] | None = None,
        forward_fn: Callable[[nn.Module, Any], tuple[torch.Tensor, torch.Tensor]] | None = None,
        compute_metrics: Callable[[torch.Tensor, torch.Tensor], dict[str, float]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.forward_fn = forward_fn
        self.compute_metrics = compute_metrics

        self.device = get_device(config.device)
        self._has_device_map = bool(getattr(self.model, "hf_device_map", None))

        if not self._has_device_map:
            self.model.to(self.device)

        seed_everything(config.seed)

        if optimizer is None:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer
        self._resume_meta: dict[str, Any] = {}
        self._start_epoch = 0
        amp_dtype: torch.dtype | None = None
        if config.fp16:
            amp_dtype = torch.float16
        elif config.bf16:
            amp_dtype = torch.bfloat16

        if config.lr_finder_enabled and not getattr(config, "resume_from_checkpoint", None):
            problem_type = getattr(config, "problem_type", "")

            if problem_type == "causal_lm":
                def causal_lm_forward(model, batch):
                    inputs, labels = unpack_batch(batch)
                    if is_dict_like(inputs):
                        inputs_dict = dict(inputs)
                        if "input_ids" not in inputs_dict and "text" in inputs_dict:
                            return None, None
                        input_ids = inputs_dict.get("input_ids")
                        if input_ids is None:
                            return None, None
                        if isinstance(input_ids, torch.Tensor) and input_ids.numel() == 0:
                            return None, None
                        _add_token_type_ids(inputs_dict, model)
                        if "attention_mask" not in inputs_dict:
                            inputs_dict["attention_mask"] = torch.ones_like(input_ids)
                        if labels is not None:
                            inputs_dict["labels"] = labels
                        inputs_dict = move_to_device(inputs_dict, self.device)
                        outputs = model(**inputs_dict)
                        return outputs.loss, outputs.logits if hasattr(outputs, "logits") else outputs.loss
                    return None, None

                finder_forward = causal_lm_forward
                finder_criterion = None
            else:
                finder_forward = forward_fn
                finder_criterion = criterion

            lr_finder = LRFinder(
                model,
                optimizer,
                criterion=finder_criterion,
                device=self.device,
                trainable_only=config.lr_finder_trainable_only,
                amp_dtype=amp_dtype,
                save_optimizer_state=config.lr_finder_save_optimizer_state,
            )
            self.initial_lr = lr_finder.find(
                train_dataloader,
                forward_fn=finder_forward,
                start_lr=config.lr_finder_start,
                end_lr=config.lr_finder_end,
                num_steps=config.lr_finder_steps,
                log_steps=config.lr_finder_log_steps,
            )
            self.initial_lr = max(self.initial_lr, config.min_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.initial_lr
            logger.info("LR finder found optimal LR: %.2e", self.initial_lr)
        else:
            self.initial_lr = self.optimizer.param_groups[0]["lr"]

        steps_per_epoch = max(
            (len(train_dataloader) + config.gradient_accumulation_steps - 1) // max(
                config.gradient_accumulation_steps, 1
            ),
            1,
        )
        num_training_steps = steps_per_epoch * config.max_epochs

        self.scheduler = SmartScheduler(
            self.optimizer,
            self.initial_lr,
            config,
            num_training_steps=num_training_steps,
        )

        self.selgis = SelgisCore(
            model,
            self.optimizer,
            self.scheduler,
            config,
            self.device,
        )

        self.callbacks = callbacks or [LoggingCallback()]

        if not any(isinstance(cb, HistoryCallback) for cb in self.callbacks):
            self.callbacks.append(
                HistoryCallback(output_dir=config.output_dir),
            )

        if not any(isinstance(cb, CheckpointCallback) for cb in self.callbacks):
            self.callbacks.append(
                CheckpointCallback(
                    output_dir=config.output_dir,
                    save_best_only=getattr(config, "save_best_only", True),
                    save_total_limit=getattr(config, "save_total_limit", 3),
                ),
            )

        if getattr(config, "sparsity_enabled", False) and not any(
            isinstance(cb, SparsityCallback) for cb in self.callbacks
        ):
            self.callbacks.append(
                SparsityCallback(
                    target_sparsity=getattr(config, "sparsity_target", 0.0),
                    start_epoch=getattr(config, "sparsity_start_epoch", 0),
                    frequency=getattr(config, "sparsity_frequency", 1),
                ),
            )

        self._global_step = 0
        self._current_epoch = 0
        self._empty_cache_steps = max(int(getattr(config, "empty_cache_steps", 0)), 0)
        self._gc_collect_steps = max(int(getattr(config, "gc_collect_steps", 0)), 0)
        self._resume_if_needed()

    # ── Checkpoint resume ─────────────────────────────────────────────────

    def _torch_load(
        self,
        path: Path,
        map_location: str | torch.device = "cpu",
        weights_only: bool = False,
    ):
        kwargs: dict[str, Any] = {"map_location": map_location}
        if weights_only:
            try:
                return torch.load(path, **kwargs, weights_only=True)
            except TypeError:
                return torch.load(path, **kwargs)
        return torch.load(path, **kwargs)

    def _resume_if_needed(self) -> None:
        ckpt_path_raw = getattr(self.config, "resume_from_checkpoint", None)
        if not ckpt_path_raw:
            return

        checkpoint_dir = Path(ckpt_path_raw)
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            raise ValueError(f"resume checkpoint directory not found: {checkpoint_dir}")

        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            state_format = "full"
            meta_path = checkpoint_dir / "metrics.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                state_format = meta.get("state_format", "full")

            strict = state_format == "full"
            self.load_model(str(model_path), weights_only=True, strict=strict)

        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            optim_state = self._torch_load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optim_state)

        scheduler_path = checkpoint_dir / "scheduler.pt"
        if scheduler_path.exists() and hasattr(self.scheduler, "load_state_dict"):
            scheduler_state = self._torch_load(scheduler_path, map_location="cpu")
            self.scheduler.load_state_dict(scheduler_state)

        metrics_path = checkpoint_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                meta = json.load(f)
            self._resume_meta = meta
            self._global_step = int(meta.get("global_step", 0))
            self._start_epoch = int(meta.get("epoch", -1)) + 1

        logger.info("Resumed from checkpoint: %s", checkpoint_dir)

    # ── Training loop ─────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        self._call_callbacks("on_train_begin")
        metrics: dict[str, Any] = {}
        primary_metric = getattr(self.config, "primary_metric", None)

        for epoch in range(self._start_epoch, self.config.max_epochs):
            self._current_epoch = epoch
            self._call_callbacks("on_epoch_begin", epoch=epoch)

            train_loss = self._train_epoch()
            metrics = {"train_loss": train_loss}

            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer and self.optimizer.param_groups else 0
            logger.info("Epoch %d  train_loss=%.4f  lr=%.2e", epoch, train_loss, lr)

            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)

            self._call_callbacks("on_epoch_end", epoch=epoch, metrics=metrics)

            if primary_metric and primary_metric in metrics:
                chosen_metric = primary_metric
                higher_is_better = primary_metric not in LOWER_IS_BETTER_METRICS
            elif "accuracy" in metrics:
                chosen_metric = "accuracy"
                higher_is_better = True
            else:
                chosen_metric = "loss"
                higher_is_better = False

            if getattr(self.config, "warmup_ratio", 0) > 0:
                # Step-based scheduling: scheduler.step() runs after every
                # optimizer step in _train_epoch. Do NOT call step_epoch()
                # here — its epoch-based formula would overwrite the LR
                # computed by the step schedule. Only track the epoch for
                # surge_lr().
                self.scheduler._epoch = epoch
            elif hasattr(self.scheduler, "step_epoch"):
                self.scheduler.step_epoch(epoch)

            if self.eval_dataloader is not None:
                status = self.selgis.eval_epoch(
                    metrics,
                    epoch,
                    primary_metric=chosen_metric,
                    higher_is_better=higher_is_better,
                )
            else:
                status = "CONTINUE"

            if status == "STOP":
                break

            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                break

        self.selgis.load_best_weights()
        self._call_callbacks("on_train_end")
        return metrics

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        accum_steps = max(self.config.gradient_accumulation_steps, 1)
        accum_count = 0
        log_steps = getattr(self.config, 'logging_steps', 10) or 10

        logger.info("Starting epoch %d, dataloader length: %d", self._current_epoch, len(self.train_dataloader) if self.train_dataloader else 0)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_dataloader):
            self._call_callbacks("on_step_begin", step=self._global_step)

            loss = self._training_step(batch)

            if batch_idx == 0:
                logger.debug("batch_idx=%d, loss returned=%s", batch_idx, loss)

            if loss is not None:
                loss_val = loss.item() if hasattr(loss, 'item') else loss
                total_loss += loss_val
                num_steps += 1
                accum_count += 1

                if self._global_step % log_steps == 0:
                    lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0
                    logger.info("[METRIC] step=%d epoch=%d loss=%.4f lr=%.2e", self._global_step, self._current_epoch, loss_val, lr)

                if accum_count >= accum_steps:
                    self.selgis.optimizer_step()
                    self._step_scheduler_if_needed()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    self._global_step += 1
            else:
                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            self._call_callbacks(
                "on_step_end",
                step=self._global_step,
                loss=loss if loss is not None else None,
            )
            self._maybe_release_memory()

        if accum_count > 0:
            self.selgis.optimizer_step()
            self._step_scheduler_if_needed()
            self.optimizer.zero_grad(set_to_none=True)
            self._global_step += 1

        return total_loss / max(num_steps, 1)

    def _maybe_release_memory(self) -> None:
        if self._gc_collect_steps > 0 and (
            self._global_step > 0 and self._global_step % self._gc_collect_steps == 0
        ):
            gc.collect()

        if (
            self._empty_cache_steps > 0
            and self.device.type == "cuda"
            and self._global_step > 0
            and self._global_step % self._empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

    def _step_scheduler_if_needed(self) -> None:
        """Step LR at each optimizer step when using warmup_ratio.

        Scheduling strategy:
        - ``warmup_ratio > 0`` : step-based scheduling via ``scheduler.step()``
          (called here after every optimizer step). Epoch-based stepping in
          ``train()`` only sets ``scheduler._epoch`` without updating LR.
        - ``warmup_ratio == 0`` (default) : epoch-based scheduling via
          ``scheduler.step_epoch()`` called once per epoch in ``train()``.
          Step-based updates are never invoked.
        """
        if getattr(self.config, "warmup_ratio", 0) <= 0:
            return
        if hasattr(self.scheduler, "step"):
            self.scheduler.step()

    def _training_step(self, batch: Any) -> float | None:
        with self.selgis.get_amp_context():
            loss, logits = self._forward(batch)

        if logits is not None:
            del logits

        if loss is None:
            logger.debug("_training_step: loss is None")
            return None

        try:
            loss_val = loss.item()
        except Exception as e:
            logger.debug("_training_step: loss.item() failed: %s", e)
            return None

        if not self.selgis.check_loss(loss):
            return None

        accum_steps = max(self.config.gradient_accumulation_steps, 1)
        scaled_loss = loss / accum_steps
        self.selgis.backward_step(scaled_loss)

        return loss_val

    def _forward(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.forward_fn is not None:
            if self._has_device_map:
                batch = move_to_device(batch, "cuda")
            else:
                batch = move_to_device(batch, self.device)
            return self.forward_fn(self.model, batch)

        inputs, labels = unpack_batch(batch)

        if is_dict_like(inputs):
            inputs_dict = dict(inputs)

            if "input_ids" not in inputs_dict and "text" in inputs_dict:
                raise ValueError("Dataset must return tokenized 'input_ids', not raw 'text'. Use data_type='text' with proper tokenizer.")

            _add_token_type_ids(inputs_dict, self.model)

            target_device = self.device
            if self._has_device_map:
                target_device = next(self.model.parameters()).device

            inputs_dict = move_to_device(inputs_dict, target_device)

            outputs = call_model_with_dict_inputs(self.model, inputs_dict)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                return outputs.loss, logits

            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            if labels is not None and self.criterion is not None:
                labels = move_to_device(labels, target_device)
                return self.criterion(logits, labels), logits

            raise ValueError("Model doesn't return loss and no criterion provided")

        target_device = self.device
        if self._has_device_map:
            target_device = next(self.model.parameters()).device

        inputs = move_to_device(inputs, target_device)

        outputs = self.model(inputs)

        if self.criterion is None:
            raise ValueError("Criterion required for non-dict input")
        if labels is None:
            raise ValueError("Labels required for training")

        labels = move_to_device(labels, target_device)

        return self.criterion(outputs, labels), outputs

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        correct = 0
        total = 0
        regression_se_sum = 0.0
        regression_ae_sum = 0.0
        regression_count = 0
        current_preds: torch.Tensor | None = None

        for batch in self.eval_dataloader:
            _, labels = unpack_batch(batch)

            with self.selgis.get_amp_context():
                loss, logits = self._forward(batch)

            total_loss += loss.item()
            num_batches += 1
            logits_shape = logits.shape

            labels_cpu = None
            if labels is not None:
                labels_cpu = (
                    labels.cpu()
                    if hasattr(labels, "device") and labels.device.type != "cpu"
                    else labels
                )

            if self.compute_metrics is not None or labels_cpu is not None:
                is_regression = labels_cpu is not None and (
                    torch.is_floating_point(labels_cpu)
                    and not (labels_cpu.dim() == 1 and logits_shape[-1] > 1)
                )

                if is_regression:
                    current_preds = logits.cpu()
                else:
                    current_preds = logits.argmax(dim=-1).cpu() if logits.dim() > 1 else logits.cpu()
            else:
                is_regression = False
                current_preds = None

            del logits

            if labels_cpu is not None:
                if self.compute_metrics is not None:
                    if current_preds is not None:
                        all_preds.append(current_preds)
                    all_labels.append(labels_cpu)
                elif is_regression and current_preds is not None:
                    if current_preds.shape != labels_cpu.shape:
                        current_preds = current_preds.view_as(labels_cpu)
                    diff = current_preds - labels_cpu
                    regression_se_sum += diff.pow(2).sum().item()
                    regression_ae_sum += diff.abs().sum().item()
                    regression_count += labels_cpu.numel()
                elif labels is not None and current_preds is not None:
                    if current_preds.shape != labels_cpu.shape:
                        current_preds = current_preds.view_as(labels_cpu)
                    correct += (current_preds == labels_cpu).sum().item()
                    total += labels_cpu.numel()

        metrics: dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }

        if self.compute_metrics is not None and all_preds and all_labels:
            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)
            metrics.update(self.compute_metrics(preds_cat, labels_cat))
        elif self.compute_metrics is None:
            if total > 0:
                # Fraction in [0, 1] — consistent with compute_metrics outputs
                metrics["accuracy"] = correct / total
            if regression_count > 0:
                metrics["mse"] = regression_se_sum / regression_count
                metrics["mae"] = regression_ae_sum / regression_count

        self._call_callbacks("on_evaluate", metrics=metrics)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return metrics

    # ── Utilities ─────────────────────────────────────────────────────────

    def _call_callbacks(self, method: str, **kwargs: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, method, None)
            if fn is not None:
                fn(self, **kwargs)

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, weights_only: bool = True, strict: bool = True) -> None:
        resolved_path = Path(path)
        if resolved_path.is_dir():
            model_file = resolved_path / "model.pt"
            if model_file.exists():
                resolved_path = model_file

        load_kwargs: dict[str, Any] = {"map_location": self.device}
        if weights_only:
            try:
                state = torch.load(resolved_path, **load_kwargs, weights_only=True)
            except TypeError:
                logger.warning(
                    "weights_only=True not supported (PyTorch < 2.0). "
                    "Loading with weights_only=False."
                )
                state = torch.load(resolved_path, **load_kwargs)
        else:
            state = torch.load(resolved_path, **load_kwargs)

        try:
            self.model.load_state_dict(state, strict=strict)
        except RuntimeError as exc:
            if strict:
                logger.warning("Strict load failed; retrying with strict=False (%s)", exc)
                self.model.load_state_dict(state, strict=False)
            else:
                raise


# ══════════════════════════════════════════════════════════════════════════════


class TransformerTrainer(Trainer):
    """Trainer for HuggingFace Transformers.

    Supports automatic model/tokenizer loading, PEFT/LoRA adapters,
    gradient checkpointing, quantization, device-map offloading,
    Flash Attention, Chunked Cross-Entropy, and memory-efficient
    evaluation for large-vocabulary generative models.
    """

    def __init__(
        self,
        model_or_path: nn.Module | str,
        config: TransformerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        self._transformer_config = config

        use_unsloth = getattr(config, "use_unsloth", False)

        # If using Unsloth, it will load the model itself - skip _load_model
        if use_unsloth and isinstance(model_or_path, str):
            model = None  # Will be loaded by _apply_unsloth
        elif isinstance(model_or_path, str):
            model = self._load_model(model_or_path, config)
        else:
            model = model_or_path

        if tokenizer is None and use_unsloth:
            tokenizer = None  # Unsloth returns its own tokenizer
        elif tokenizer is None and isinstance(model_or_path, str):
            tokenizer = self._try_load_tokenizer(model_or_path, config)

        if tokenizer is not None:
            self._sync_pad_token(model, tokenizer)
        else:
            self._ensure_pad_token_id(model)

        # ── Flash Attention ───────────────────────────────────────────────
        # BUG FIX: was never actually called in the original code.
        # Must happen BEFORE model is moved to device / wrapped.
        if getattr(config, "flash_attention", False):
            self._try_enable_flash_attention()

        # ── Unsloth ──────────────────────────────────────────────────
        use_unsloth = getattr(config, "use_unsloth", False)
        unsloth_applied_peft = False  # Track if Unsloth already applied PEFT

        if use_unsloth:
            model = self._apply_unsloth(model, config, model_or_path)
            # Check if Unsloth already applied PEFT (it does if peft_config provided)
            unsloth_applied_peft = (
                use_unsloth
                and getattr(config, "use_peft", False)
                and config.peft_config
            )

        # ── PEFT / Gradient Checkpointing ────────────────────────────────
        # Skip if Unsloth already applied PEFT
        if config.use_peft and not unsloth_applied_peft and (config.peft_config or config.adapter_name_or_path):
            model = self._apply_peft(
                model,
                config.peft_config,
                config.problem_type,
                gradient_checkpointing=config.gradient_checkpointing,
                adapter_name_or_path=getattr(config, "adapter_name_or_path", None),
            )
        elif config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )

        # ── Granular GC (every N layers) ─────────────────────────────────
        if config.gc_checkpoint_interval > 1:
            gc_manager = GradientCheckpointingManager(
                checkpoint_interval=config.gc_checkpoint_interval,
                use_reentrant=False,
            )
            model = gc_manager.apply_to_model(model)
            logger.info(
                "Applied layer-level gradient checkpointing (interval=%d)",
                config.gc_checkpoint_interval,
            )

        # ── Chunked CE setup ─────────────────────────────────────────────
        self._chunked_ce = config.chunked_ce
        self._ce_chunk_size = config.ce_chunk_size

        # Chunked CE is only meaningful for generative/masked tasks
        # (classification tasks have tiny vocab and don't OOM on logits)
        self._chunked_ce_active = self._chunked_ce and (
            config.problem_type in ("causal_lm", "seq2seq", "masked_lm")
        )

        if self._chunked_ce_active:
            logger.info(
                "Chunked CE active for %s (chunk_size=%d). Model's internal CE will be replaced.",
                config.problem_type,
                config.ce_chunk_size,
            )

        # ── Optimizer ────────────────────────────────────────────────────
        param_groups = get_optimizer_grouped_params(model, config.weight_decay)
        optimizer = self._create_optimizer(param_groups, config)

        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            **kwargs,
        )

        self.tokenizer = tokenizer if tokenizer is not None else getattr(self, "tokenizer", None)

        # ── Criterion override for chunked CE ────────────────────────────
        if self._chunked_ce_active and self.criterion is None:
            self.criterion = CrossEntropyLossV2(
                chunk_size=self._ce_chunk_size,
                label_smoothing=config.label_smoothing,
            )
            logger.info(
                "CrossEntropyLossV2 (chunked) set as criterion (chunk_size=%d)",
                self._ce_chunk_size,
            )

    # ── Flash Attention helper ────────────────────────────────────────────

    @staticmethod
    def _try_enable_flash_attention() -> None:
        """Enable Flash SDP (or mem-efficient fallback) via PyTorch SDPA.

        No-op if CUDA is unavailable.  Falls back to memory-efficient
        attention on GPUs older than Ampere (compute capability < 8.0).
        """
        if not torch.cuda.is_available():
            logger.debug("Flash Attention requires CUDA -- skipped on CPU/MPS")
            return

        try:
            sm_major, sm_minor = torch.cuda.get_device_capability()

            if sm_major >= 8:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info(
                    "Flash Attention 2 enabled (CUDA compute capability %d.%d)",
                    sm_major,
                    sm_minor,
                )
            else:
                # Turing / Volta / older - Flash Attention not supported
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_flash_sdp(False)
                logger.info(
                    "Memory-efficient attention enabled "
                    "(Flash Attention requires sm>=8.0; got sm%d.%d)",
                    sm_major,
                    sm_minor,
                )
        except Exception as exc:
            logger.warning(
                "Could not configure Flash/SDPA attention: %s. "
                "Training continues with default attention.",
                exc,
            )

    # ── _forward override (KEY FIX for chunked CE) ───────────────────────

    def _forward(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional chunked cross-entropy.

        When ``chunked_ce=True`` and the problem type is a generative
        task (causal_lm / seq2seq / masked_lm), this override:

        1. Calls the model WITHOUT the ``labels`` key, so HuggingFace
           does **not** compute its own CE over the full [B, T, V] logits.
        2. Gets ``outputs.logits`` from the model.
        3. Applies our ``CrossEntropyLossV2`` (chunked) to the logits,
           processing the vocabulary in chunks during the backward pass.

        This is the only way chunked CE actually saves VRAM — if we let
        the model compute its own loss it would materialise the full
        exp(logits) tensor regardless.

        Without chunked CE (or for classification tasks) falls back to
        the standard ``Trainer._forward``.
        """
        if not self._chunked_ce_active or self.criterion is None:
            return super()._forward(batch)

        if self.forward_fn is not None:
            if not self._has_device_map:
                batch = move_to_device(batch, self.device)
            return self.forward_fn(self.model, batch)

        inputs, labels = unpack_batch(batch)

        if not is_dict_like(inputs):
            # Non-dict input — can't reliably strip labels; fall back
            return super()._forward(batch)

        inputs_dict = dict(inputs)

        # Extract labels from dict if not returned separately
        if labels is None:
            labels = inputs_dict.get("labels")

        # Build model inputs WITHOUT labels so model doesn't compute CE
        model_inputs = {k: v for k, v in inputs_dict.items() if k != "labels"}

        # Add token_type_ids if the model supports them (e.g. Gemma3)
        _add_token_type_ids(model_inputs, self.model)

        if self._has_device_map:
            model_inputs = move_to_device(model_inputs, 'cuda')
        else:
            model_inputs = move_to_device(model_inputs, self.device)

        outputs = self.model(**model_inputs)

        if not hasattr(outputs, "logits") or outputs.logits is None:
            # Unusual: model doesn't expose logits → fall back gracefully
            logger.warning(
                "chunked_ce=True but model returned no logits attribute. "
                "Falling back to model's internal loss. "
                "Memory savings from Chunked CE will NOT apply."
            )
            # Re-run with labels so model computes loss
            full_inputs = (
                move_to_device(inputs_dict, self.device)
                if not self._has_device_map
                else inputs_dict
            )
            outputs2 = self.model(**full_inputs)
            logits2 = (
                outputs2.logits
                if hasattr(outputs2, "logits")
                else torch.zeros(1, device=self.device)
            )
            return outputs2.loss, logits2

        logits = outputs.logits  # [B, T, V] — still in memory but backward is chunked

        if labels is None:
            raise ValueError(
                "chunked_ce=True requires labels. "
                "Add 'labels' to the batch dict or provide them as the "
                "second element of the batch tuple."
            )

        # Labels must live on the same device as the logits (with
        # device_map the batch dict is not fully moved upfront).
        if labels.device != logits.device:
            labels = labels.to(logits.device)

        loss = self.criterion(logits, labels)

        # Return a detached dummy instead of logits to free memory faster.
        # Evaluation paths that need real logits use their own forward call.
        dummy = torch.zeros(1, device=self.device, requires_grad=False)
        del logits
        return loss, dummy

    # ── Memory-efficient evaluation ───────────────────────────────────────

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Memory-efficient evaluation for Transformer models.

        For causal/seq2seq/masked LM: computes loss + perplexity without
        storing full logits.  For classification/regression: falls back to
        base ``Trainer.evaluate`` which computes accuracy.
        """
        if self.eval_dataloader is None:
            return {}

        if self._transformer_config.problem_type not in (
            "causal_lm",
            "seq2seq",
            "masked_lm",
        ):
            return super().evaluate()

        self.model.eval()
        total_loss = 0.0
        total_weight = 0  # tokens (or elements) contributing to the loss
        total_tokens = 0
        num_batches = 0

        for batch in self.eval_dataloader:
            inputs, _ = unpack_batch(batch)

            if is_dict_like(inputs):
                inputs_dict = dict(inputs)
                if self._has_device_map:
                    inputs_dict = move_to_device(inputs_dict, 'cuda')
                else:
                    inputs_dict = move_to_device(inputs_dict, self.device)
            else:
                if self._has_device_map:
                    inputs = move_to_device(inputs, 'cuda')
                else:
                    inputs = move_to_device(inputs, self.device)
                inputs_dict = {"input_ids": inputs}

            with self.selgis.get_amp_context():
                outputs = self.model(**inputs_dict)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                labels = inputs_dict.get("labels")
                if labels is not None:
                    # Weight by the number of real (non-pad) tokens so that
                    # an undersized last batch doesn't bias the average.
                    weight = (labels != -100).sum().item()
                else:
                    input_ids = inputs_dict.get("input_ids")
                    weight = input_ids.numel() if input_ids is not None else 1
                weight = max(weight, 1)
                total_loss += outputs.loss.item() * weight
                total_weight += weight

                if labels is not None:
                    total_tokens += max((labels != -100).sum().item(), 0)
                else:
                    input_ids = inputs_dict.get("input_ids")
                    if input_ids is not None:
                        total_tokens += input_ids.numel()

            num_batches += 1
            del outputs

        if total_weight > 0:
            avg_loss = total_loss / total_weight
        else:
            avg_loss = 0.0

        try:
            perplexity = math.exp(min(avg_loss, 100))
        except OverflowError:
            perplexity = float("inf")

        logger.info("Eval_loss=%.4f perplexity=%.4f", avg_loss, perplexity)

        metrics: dict[str, float] = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        if total_tokens > 0:
            metrics["eval_tokens"] = float(total_tokens)

        self._call_callbacks("on_evaluate", metrics=metrics)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return metrics

    # ── Model / tokenizer loading ─────────────────────────────────────────

    @staticmethod
    def _try_load_tokenizer(model_path: str, config: TransformerConfig):
        try:
            from transformers import AutoTokenizer

            trust = getattr(config, "trust_remote_code", False)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust)
            logger.info("Tokenizer auto-loaded from %s", model_path)
            return tokenizer
        except Exception:
            logger.warning("Could not auto-load tokenizer. Pass tokenizer explicitly if needed.")
            return None

    @staticmethod
    def _sync_pad_token(model: nn.Module, tokenizer: Any) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("pad_token not set — using eos_token (%r)", tokenizer.eos_token)
        if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info("Model pad_token_id set to %d", tokenizer.pad_token_id)

    @staticmethod
    def _ensure_pad_token_id(model: nn.Module) -> None:
        if not hasattr(model, "config"):
            return
        if getattr(model.config, "pad_token_id", None) is not None:
            return
        eos_id = getattr(model.config, "eos_token_id", None)
        if eos_id is not None:
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            model.config.pad_token_id = eos_id
            logger.info("No tokenizer provided — pad_token_id set to eos_token_id (%s)", eos_id)

    def _load_model(self, path: str, config: TransformerConfig) -> nn.Module:
        # Clean memory before loading large model
        import torch
        torch.cuda.empty_cache()
        gc.collect()

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMaskedLM,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise ImportError("Install transformers: pip install transformers") from None

        bnb_config = self._build_bnb_config(config)
        device_map = getattr(config, "device_map", None)
        trust_remote = getattr(config, "trust_remote_code", False)

        if bnb_config is not None and device_map is None:
            device_map = "auto"
            logger.info("device_map='auto' set automatically (required for quantization)")

        load_kw: dict[str, Any] = {"trust_remote_code": trust_remote}
        if bnb_config is not None:
            load_kw["quantization_config"] = bnb_config
        if device_map is not None:
            load_kw["device_map"] = device_map

        model_loaders = {
            "causal_lm": lambda: AutoModelForCausalLM.from_pretrained(path, **load_kw),
            "seq2seq": lambda: AutoModelForSeq2SeqLM.from_pretrained(path, **load_kw),
            "masked_lm": lambda: AutoModelForMaskedLM.from_pretrained(path, **load_kw),
        }

        if config.problem_type in model_loaders:
            return model_loaders[config.problem_type]()

        if config.problem_type in (
            "single_label_classification",
            "multi_label_classification",
            "regression",
        ):
            return AutoModelForSequenceClassification.from_pretrained(
                path, num_labels=config.num_labels, **load_kw
            )

        supported = (
            "single_label_classification",
            "multi_label_classification",
            "regression",
            "seq2seq",
            "causal_lm",
            "masked_lm",
        )
        raise ValueError(
            f"Unsupported problem_type: {config.problem_type!r}. Supported: {', '.join(supported)}"
        )

    @staticmethod
    def _build_bnb_config(config: TransformerConfig):
        if config.quantization_type == "no":
            return None
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning("bitsandbytes/transformers not available. Proceeding without quantization.")
            return None

        try:
            if config.quantization_type == "8bit":
                logger.info("Quantization: 8-bit enabled")
                return BitsAndBytesConfig(load_in_8bit=True)
            if config.quantization_type == "4bit":
                compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
                logger.info("Quantization: 4-bit enabled (%s)", config.bnb_4bit_quant_type)
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                )
        except Exception as exc:
            logger.warning("Failed to configure quantization: %s. Proceeding without it.", exc)

        return None

    def _apply_peft(
        self,
        model: nn.Module,
        peft_config: dict,
        problem_type: str | None = None,
        gradient_checkpointing: bool = False,
        adapter_name_or_path: str | None = None,
    ) -> nn.Module:
        try:
            from peft import (
                LoraConfig,
                PeftModel,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError:
            raise ImportError(
                "PEFT is required when use_peft=True. Install with: pip install peft"
            ) from None

        is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(
            model, "is_loaded_in_4bit", False
        )

        if is_quantized:
            gc_kwarg: Any
            if gradient_checkpointing:
                gc_kwarg = {"use_reentrant": False}
            else:
                gc_kwarg = False

            try:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc_kwarg)
            except TypeError:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=bool(gradient_checkpointing)
                )
                if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
        elif gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        task_type_mapping = {
            "causal_lm": TaskType.CAUSAL_LM,
            "seq2seq": TaskType.SEQ_2_SEQ_LM,
            "token_classification": TaskType.TOKEN_CLS,
            "question_answering": TaskType.QUESTION_ANS,
            "feature_extraction": TaskType.FEATURE_EXTRACTION,
        }

        if "task_type" in peft_config:
            task_type_value = peft_config["task_type"]
            if isinstance(task_type_value, str):
                task_type = task_type_mapping.get(task_type_value.lower(), TaskType.SEQ_CLS)
            else:
                task_type = task_type_value
        elif problem_type:
            task_type = task_type_mapping.get(problem_type, TaskType.SEQ_CLS)
        else:
            task_type = TaskType.SEQ_CLS

        if adapter_name_or_path:
            peft_model = PeftModel.from_pretrained(model, adapter_name_or_path, is_trainable=True)
            logger.info("Loaded existing LoRA adapter from %s", adapter_name_or_path)
        else:
            filtered_config = {k: v for k, v in peft_config.items() if k != "task_type"}
            lora_config = LoraConfig(task_type=task_type, **filtered_config)
            peft_model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        logger.info("LoRA applied: %d / %d params trainable (%.2f%%)", trainable_params, total_params, 100 * trainable_params / total_params if total_params > 0 else 0.0)
        return peft_model

    def _apply_unsloth(
        self,
        model: nn.Module,
        config: TransformerConfig,
        model_or_path: str = "",
    ) -> nn.Module:
        """Apply Unsloth for faster training.

        Unsloth optimizes attention layers for ~2x speed and ~50% less VRAM.
        Works with Llama, Qwen, Mistral, Phi, Gemma, Gemma 4.

        Args:
            model: The model to optimize (may be unused if model_or_path provided).
            config: Configuration with Unsloth settings.
            model_or_path: Model path for Unsloth loading.

        Returns:
            Model wrapped with Unsloth if available.
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            logger.warning("Unsloth not installed. Install with: pip install unsloth")
            return model

        max_seq_length = getattr(config, "max_length", 512)
        dtype = torch.bfloat16 if config.bf16 else torch.float16
        load_in_4bit = config.quantization_type == "4bit"

        try:
            current_model_path = getattr(config, "model_name_or_path", "")
            if not current_model_path:
                logger.warning("Unsloth requires model_name_or_path. Skipping.")
                return model

            model_name_lower = current_model_path.lower()
            is_gemma_4 = "gemma-4" in model_name_lower or "gemma4" in model_name_lower
            if is_gemma_4:
                logger.info("Gemma 4 detected - using Unsloth with transformers>=5.5.0")

            logger.info("Applying Unsloth to %s...", current_model_path)

            unsloth_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=current_model_path,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )

            if is_gemma_4 and tokenizer:
                try:
                    tokenizer.chat_template = tokenizer.chat_template.replace(
                        "{% for message in messages %}",
                        "{% raw %}{% for message in messages %}{% endraw %}"
                    )
                except Exception:
                    pass

            if getattr(config, "use_peft", False) and config.peft_config:
                from peft import LoraConfig, get_peft_model

                filtered = {k: v for k, v in config.peft_config.items() if k != "task_type"}
                lora_config = LoraConfig(**filtered)
                unsloth_model = get_peft_model(unsloth_model, lora_config)

                trainable = sum(p.numel() for p in unsloth_model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in unsloth_model.parameters())
                logger.info("Unsloth + LoRA: %d / %d trainable (%.2f%%)", trainable, total, 100 * trainable / total if total > 0 else 0.0)

            self.tokenizer = tokenizer
            logger.info("Unsloth applied successfully")

            return unsloth_model

        except Exception as e:
            logger.warning("Failed to apply Unsloth: %s. Continuing without.", e)
            return model

    def _create_optimizer(
        self,
        param_groups: list[dict],
        config: TransformerConfig,
    ) -> optim.Optimizer:
        if config.optimizer_type == "adamw":
            return optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
        if config.optimizer_type == "adam":
            return optim.Adam(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
        if config.optimizer_type == "sgd":
            return optim.SGD(param_groups, lr=config.learning_rate, momentum=0.9)
        if config.optimizer_type == "adafactor":
            try:
                from transformers.optimization import Adafactor

                return Adafactor(
                    param_groups,
                    lr=config.learning_rate,
                    relative_step=False,
                    warmup_init=False,
                )
            except ImportError:
                logger.warning("Adafactor not available, falling back to AdamW")
        return optim.AdamW(param_groups, lr=config.learning_rate)

    # ── Save / push ───────────────────────────────────────────────────────

    def save_pretrained(self, path: str) -> None:
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
            logger.info("Model saved to %s", path)
        else:
            self.save_model(f"{path}/pytorch_model.pt")

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        if not hasattr(self.model, "push_to_hub"):
            raise ValueError("Model doesn't support push_to_hub. Use save_pretrained instead.")
        self.model.push_to_hub(repo_id, **kwargs)
        if self.tokenizer:
            self.tokenizer.push_to_hub(repo_id, **kwargs)
        logger.info("Model pushed to https://huggingface.co/%s", repo_id)
