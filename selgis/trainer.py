"""Universal trainers for PyTorch and HuggingFace Transformers."""
from typing import Any, Callable, Optional, Union

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
from selgis.config import SelgisConfig, TransformerConfig
from selgis.core import SelgisCore
from selgis.lr_finder import LRFinder
from selgis.scheduler import SmartScheduler
from selgis.utils import (
    get_device,
    get_optimizer_grouped_params,
    is_dict_like,
    move_to_device,
    seed_everything,
    unpack_batch,
)


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
        eval_dataloader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        callbacks: Optional[list[Callback]] = None,
        forward_fn: Optional[
            Callable[
                [nn.Module, Any], tuple[torch.Tensor, torch.Tensor]
            ]
        ] = None,
        compute_metrics: Optional[
            Callable[
                [torch.Tensor, torch.Tensor], dict[str, float]
            ]
        ] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            train_dataloader: Training data loader.
            eval_dataloader: Optional evaluation data loader.
            criterion: Loss function. Optional if model returns loss.
            optimizer: Optional optimizer; created automatically if absent.
            callbacks: List of callbacks. Defaults to ``[LoggingCallback()]``.
            forward_fn: Custom ``(model, batch) -> (loss, logits)`` callable.
            compute_metrics: ``(preds, labels) -> metrics_dict`` callable.
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.forward_fn = forward_fn
        self.compute_metrics = compute_metrics

        self.device = get_device(config.device)
        self._has_device_map = hasattr(self.model, "hf_device_map")

        if not self._has_device_map:
            self.model.to(self.device)

        seed_everything(config.seed)

        if optimizer is None:
            trainable_params = [
                p for p in model.parameters() if p.requires_grad
            ]
            optimizer = optim.AdamW(
                trainable_params,
                lr=1e-3,
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

        if config.lr_finder_enabled:
            lr_finder = LRFinder(
                model,
                self.optimizer,
                criterion,
                self.device,
                trainable_only=getattr(
                    config, "lr_finder_trainable_only", False,
                ),
            )
            self.initial_lr = lr_finder.find(
                train_dataloader,
                forward_fn=forward_fn,
                start_lr=config.lr_finder_start,
                end_lr=config.lr_finder_end,
                num_steps=config.lr_finder_steps,
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.initial_lr
        else:
            self.initial_lr = self.optimizer.param_groups[0]["lr"]

        steps_per_epoch = (
            len(train_dataloader)
            + config.gradient_accumulation_steps
            - 1
        ) // max(config.gradient_accumulation_steps, 1)
        num_training_steps = steps_per_epoch * config.max_epochs

        self.scheduler = SmartScheduler(
            self.optimizer,
            self.initial_lr,
            config,
            num_training_steps=num_training_steps,
        )

        self.selgis = SelgisCore(
            model, self.optimizer, self.scheduler, config, self.device,
        )

        self.callbacks = callbacks or [LoggingCallback()]

        if not any(
            isinstance(cb, HistoryCallback) for cb in self.callbacks
        ):
            self.callbacks.append(
                HistoryCallback(output_dir=config.output_dir),
            )

        if not any(
            isinstance(cb, CheckpointCallback) for cb in self.callbacks
        ):
            self.callbacks.append(
                CheckpointCallback(
                    output_dir=config.output_dir,
                    save_best_only=getattr(
                        config, "save_best_only", True,
                    ),
                    save_total_limit=getattr(
                        config, "save_total_limit", 3,
                    ),
                ),
            )

        if getattr(config, "sparsity_enabled", False) and not any(
            isinstance(cb, SparsityCallback) for cb in self.callbacks
        ):
            self.callbacks.append(
                SparsityCallback(
                    target_sparsity=getattr(
                        config, "sparsity_target", 0.0,
                    ),
                    start_epoch=getattr(
                        config, "sparsity_start_epoch", 0,
                    ),
                    frequency=getattr(config, "sparsity_frequency", 1),
                ),
            )

        self._global_step = 0
        self._current_epoch = 0

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Final metrics dictionary.
        """
        self._call_callbacks("on_train_begin")
        metrics: dict[str, Any] = {}
        primary_metric = getattr(
            self.config, "primary_metric", None,
        )

        for epoch in range(self.config.max_epochs):
            self._current_epoch = epoch
            self._call_callbacks("on_epoch_begin", epoch=epoch)

            train_loss = self._train_epoch()
            metrics = {"train_loss": train_loss}

            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)

            self._call_callbacks(
                "on_epoch_end", epoch=epoch, metrics=metrics,
            )

            if primary_metric and primary_metric in metrics:
                chosen_metric = primary_metric
                higher_is_better = primary_metric != "loss"
            elif "accuracy" in metrics:
                chosen_metric = "accuracy"
                higher_is_better = True
            else:
                chosen_metric = "loss"
                higher_is_better = False

            status = self.selgis.eval_epoch(
                metrics,
                epoch,
                primary_metric=chosen_metric,
                higher_is_better=higher_is_better,
            )

            if status == "STOP":
                break

            if any(
                getattr(cb, "should_stop", False)
                for cb in self.callbacks
            ):
                break

        self.selgis.load_best_weights()
        self._call_callbacks("on_train_end")
        return metrics

    def _train_epoch(self) -> float:
        """Run a single training epoch with gradient accumulation.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        accum_steps = max(self.config.gradient_accumulation_steps, 1)
        accum_count = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch in self.train_dataloader:
            self._call_callbacks("on_step_begin", step=self._global_step)

            loss = self._training_step(batch)

            if loss is not None:
                total_loss += loss
                num_steps += 1
                accum_count += 1

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
                loss=loss if loss is not None else 0.0,
            )

        if accum_count > 0:
            self.selgis.optimizer_step()
            self._step_scheduler_if_needed()
            self.optimizer.zero_grad(set_to_none=True)
            self._global_step += 1

        return total_loss / max(num_steps, 1)

    def _step_scheduler_if_needed(self) -> None:
        """Call scheduler step when using step-based schedule."""
        if getattr(self.config, "warmup_ratio", 0) <= 0:
            return
        if hasattr(self.scheduler, "step"):
            self.scheduler.step()

    def _training_step(self, batch: Any) -> Optional[float]:
        """Execute a single forward + backward pass.

        Device management is delegated to ``_forward``.
        No optimizer step is performed here (handled by ``_train_epoch``
        for gradient accumulation).

        Args:
            batch: A raw batch from the data loader.

        Returns:
            Loss value as float, or ``None`` if a rollback was triggered.
        """
        with self.selgis.get_amp_context():
            loss, logits = self._forward(batch)

        del logits

        if not self.selgis.check_loss(loss):
            return None

        accum_steps = max(self.config.gradient_accumulation_steps, 1)
        scaled_loss = loss / accum_steps
        self.selgis.backward_step(scaled_loss)

        return loss.item()

    def _forward(
        self, batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with automatic device management.

        Handles custom ``forward_fn``, dict-like and tuple-like batches,
        and models using ``hf_device_map``.

        Args:
            batch: Raw batch from the data loader (typically on CPU).

        Returns:
            Tuple of ``(loss, logits)``.

        Raises:
            ValueError: If loss cannot be computed.
        """
        if self.forward_fn is not None:
            if not self._has_device_map:
                batch = move_to_device(batch, self.device)
            return self.forward_fn(self.model, batch)

        inputs, labels = unpack_batch(batch)

        if is_dict_like(inputs):
            inputs_dict = dict(inputs)
            if not self._has_device_map:
                inputs_dict = move_to_device(inputs_dict, self.device)

            outputs = self.model(**inputs_dict)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                logits = (
                    outputs.logits
                    if hasattr(outputs, "logits")
                    else outputs
                )
                return outputs.loss, logits

            logits = (
                outputs.logits
                if hasattr(outputs, "logits")
                else outputs
            )

            if labels is not None and self.criterion is not None:
                if not self._has_device_map:
                    labels = move_to_device(labels, self.device)
                return self.criterion(logits, labels), logits

            raise ValueError(
                "Model doesn't return loss and no criterion provided"
            )

        if not self._has_device_map:
            inputs = move_to_device(inputs, self.device)

        outputs = self.model(inputs)

        if self.criterion is None:
            raise ValueError("Criterion required for non-dict input")
        if labels is None:
            raise ValueError("Labels required for training")

        if not self._has_device_map:
            labels = move_to_device(labels, self.device)

        return self.criterion(outputs, labels), outputs

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval data loader.

        Returns:
            Metrics dictionary (e.g. ``{'loss': 0.5, 'accuracy': 92.0}``).
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        correct = 0
        total = 0

        for batch in self.eval_dataloader:
            _, labels = unpack_batch(batch)

            with self.selgis.get_amp_context():
                loss, logits = self._forward(batch)

            total_loss += loss.item()
            num_batches += 1

            preds = (
                logits.argmax(dim=-1).cpu()
                if logits.dim() > 1
                else logits.cpu()
            )
            del logits

            if labels is not None:
                labels_cpu = (
                    labels.cpu()
                    if hasattr(labels, "device")
                    and labels.device.type != "cpu"
                    else labels
                )

                if self.compute_metrics is not None:
                    all_preds.append(preds)
                    all_labels.append(labels_cpu)
                else:
                    if preds.shape != labels_cpu.shape:
                        preds = preds.view_as(labels_cpu)
                    correct += (preds == labels_cpu).sum().item()
                    total += labels_cpu.numel()

        metrics: dict[str, float] = {
            "loss": total_loss / max(num_batches, 1),
        }

        if self.compute_metrics is not None and all_preds and all_labels:
            preds_cat = torch.cat(all_preds)
            labels_cat = torch.cat(all_labels)
            metrics.update(self.compute_metrics(preds_cat, labels_cat))
        elif self.compute_metrics is None and total > 0:
            metrics["accuracy"] = correct / total * 100

        self._call_callbacks("on_evaluate", metrics=metrics)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return metrics

    def _call_callbacks(self, method: str, **kwargs: Any) -> None:
        """Invoke a callback method on all registered callbacks."""
        for callback in self.callbacks:
            fn = getattr(callback, method, None)
            if fn is not None:
                fn(self, **kwargs)

    def save_model(self, path: str) -> None:
        """Save model ``state_dict`` to *path*.

        Args:
            path: Output file path.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, weights_only: bool = True) -> None:
        """Load model ``state_dict`` from *path*.

        Args:
            path: Path to checkpoint file.
            weights_only: If True (default), load only tensors.
                Requires PyTorch >= 2.0. Set False only for trusted
                checkpoints.
        """
        load_kwargs: dict[str, Any] = {"map_location": self.device}
        if weights_only:
            try:
                state = torch.load(
                    path, **load_kwargs, weights_only=True,
                )
            except TypeError:
                print(
                    "[WARN] weights_only=True not supported "
                    "(PyTorch < 2.0). Loading with weights_only=False "
                    "— only use with trusted checkpoints!"
                )
                state = torch.load(path, **load_kwargs)
        else:
            state = torch.load(path, **load_kwargs)
        self.model.load_state_dict(state, strict=True)


class TransformerTrainer(Trainer):
    """Trainer for HuggingFace Transformers.

    Supports automatic model/tokenizer loading, PEFT/LoRA adapters,
    gradient checkpointing, quantization, and device-map offloading.
    """

    def __init__(
        self,
        model_or_path: Union[nn.Module, str],
        config: TransformerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TransformerTrainer.

        Args:
            model_or_path: A model instance or a pretrained model path.
            config: Transformer training configuration.
            train_dataloader: Training data loader.
            eval_dataloader: Optional evaluation data loader.
            tokenizer: Optional HuggingFace tokenizer.
            **kwargs: Extra arguments forwarded to ``Trainer.__init__``.
        """
        self._transformer_config = config

        if isinstance(model_or_path, str):
            model = self._load_model(model_or_path, config)
        else:
            model = model_or_path

        if config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

        if config.use_peft and config.peft_config:
            model = self._apply_peft(
                model, config.peft_config, config.problem_type,
            )

        param_groups = get_optimizer_grouped_params(
            model, config.weight_decay,
        )
        optimizer = self._create_optimizer(param_groups, config)

        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            **kwargs,
        )

        self.tokenizer = tokenizer

    def _load_model(
        self, path: str, config: TransformerConfig,
    ) -> nn.Module:
        """Load a HuggingFace model by path and ``problem_type``.

        Args:
            path: Pretrained model name or path.
            config: Transformer configuration with ``problem_type``,
                quantization, and device-map settings.

        Returns:
            Loaded model instance.

        Raises:
            ImportError: If ``transformers`` is not installed.
            ValueError: If ``problem_type`` is not supported.
        """
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMaskedLM,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise ImportError(
                "Install transformers: pip install transformers"
            ) from None

        bnb_config = self._build_bnb_config(config)

        device_map = getattr(config, "device_map", None)
        trust_remote = getattr(config, "trust_remote_code", False)

        load_kw: dict[str, Any] = {
            "trust_remote_code": trust_remote,
        }
        if bnb_config is not None:
            load_kw["quantization_config"] = bnb_config
        if device_map is not None:
            load_kw["device_map"] = device_map

        model_loaders = {
            "causal_lm": lambda: AutoModelForCausalLM.from_pretrained(
                path, **load_kw,
            ),
            "seq2seq": lambda: AutoModelForSeq2SeqLM.from_pretrained(
                path, **load_kw,
            ),
            "masked_lm": lambda: AutoModelForMaskedLM.from_pretrained(
                path, **load_kw,
            ),
        }

        if config.problem_type in model_loaders:
            return model_loaders[config.problem_type]()

        if config.problem_type in (
            "single_label_classification",
            "multi_label_classification",
            "regression",
        ):
            return AutoModelForSequenceClassification.from_pretrained(
                path,
                num_labels=config.num_labels,
                **load_kw,
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
            f"Unsupported problem_type: {config.problem_type!r}. "
            f"Supported: {', '.join(supported)}"
        )

    @staticmethod
    def _build_bnb_config(config: TransformerConfig):
        """Build BitsAndBytes quantization config if requested.

        Args:
            config: Transformer configuration.

        Returns:
            A ``BitsAndBytesConfig`` instance, or ``None`` if
            quantization is disabled or unavailable.
        """
        if config.quantization_type == "no":
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            print(
                "[WARN] bitsandbytes/transformers not available. "
                "Proceeding without quantization."
            )
            return None

        try:
            if config.quantization_type == "8bit":
                print("[INFO] Quantization: 8-bit enabled")
                return BitsAndBytesConfig(load_in_8bit=True)

            if config.quantization_type == "4bit":
                compute_dtype = getattr(
                    torch, config.bnb_4bit_compute_dtype,
                )
                print(
                    f"[INFO] Quantization: 4-bit enabled "
                    f"({config.bnb_4bit_quant_type})"
                )
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=(
                        config.bnb_4bit_use_double_quant
                    ),
                )
        except Exception as exc:
            print(
                f"[WARN] Failed to configure quantization: {exc}. "
                f"Proceeding without it."
            )

        return None

    def _apply_peft(
        self,
        model: nn.Module,
        peft_config: dict,
        problem_type: Optional[str] = None,
    ) -> nn.Module:
        """Apply PEFT/LoRA adapters to the model.

        Args:
            model: Base model.
            peft_config: LoRA configuration dictionary.
            problem_type: Optional problem type for task-type inference.

        Returns:
            Model wrapped with PEFT adapters.

        Raises:
            ImportError: If ``peft`` is not installed.
        """
        try:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError:
            raise ImportError(
                "PEFT is required when use_peft=True. "
                "Install with: pip install peft"
            ) from None

        is_quantized = (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
        )
        if is_quantized:
            model = prepare_model_for_kbit_training(model)

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
                task_type = task_type_mapping.get(
                    task_type_value.lower(), TaskType.SEQ_CLS,
                )
            else:
                task_type = task_type_value
        elif problem_type:
            task_type = task_type_mapping.get(
                problem_type, TaskType.SEQ_CLS,
            )
        else:
            task_type = TaskType.SEQ_CLS

        filtered_config = {
            k: v for k, v in peft_config.items() if k != "task_type"
        }
        lora_config = LoraConfig(task_type=task_type, **filtered_config)
        peft_model = get_peft_model(model, lora_config)

        trainable_params = sum(
            p.numel()
            for p in peft_model.parameters()
            if p.requires_grad
        )
        total_params = sum(
            p.numel() for p in peft_model.parameters()
        )
        print(
            f"[INFO] LoRA applied: {trainable_params:,} / "
            f"{total_params:,} params trainable "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        return peft_model

    def _create_optimizer(
        self,
        param_groups: list[dict],
        config: TransformerConfig,
    ) -> optim.Optimizer:
        """Create optimizer from configuration.

        Supports ``adamw``, ``adam``, ``sgd``, and ``adafactor``.

        Args:
            param_groups: Parameter groups with weight decay settings.
            config: Transformer configuration.

        Returns:
            Configured optimizer instance.
        """
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
            return optim.SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=0.9,
            )

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
                print(
                    "[WARN] Adafactor not available, "
                    "falling back to AdamW"
                )

        return optim.AdamW(param_groups, lr=config.learning_rate)

    def save_pretrained(self, path: str) -> None:
        """Save model in HuggingFace format (adapters only for PEFT).

        Args:
            path: Output directory path.
        """
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
            print(f"[SAVE] Model saved to {path}")
        else:
            self.save_model(f"{path}/pytorch_model.pt")

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        """Push model and tokenizer to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub.
            **kwargs: Extra arguments for ``push_to_hub``.

        Raises:
            ValueError: If the model doesn't support ``push_to_hub``.
        """
        if not hasattr(self.model, "push_to_hub"):
            raise ValueError(
                "Model doesn't support push_to_hub. "
                "Use save_pretrained instead."
            )
        self.model.push_to_hub(repo_id, **kwargs)
        if self.tokenizer:
            self.tokenizer.push_to_hub(repo_id, **kwargs)
        print(
            f"[INFO] Model pushed to "
            f"https://huggingface.co/{repo_id}"
        )