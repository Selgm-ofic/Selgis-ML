"""Universal trainers for PyTorch and HuggingFace Transformers."""
from typing import Any, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from selgis.callbacks import Callback, LoggingCallback, SparsityCallback, HistoryCallback
from selgis.config import SelgisConfig, TransformerConfig
from selgis.core import SelgisCore
from selgis.lr_finder import LRFinder
from selgis.scheduler import SmartScheduler
from selgis.utils import (
    get_device,
    move_to_device,
    unpack_batch,
    seed_everything,
    is_dict_like,
)


class Trainer:
    """
    Universal trainer for PyTorch models: any architecture, custom forward,
    callbacks, mixed precision, LR finder.
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
        """
        model: Model to train. config: SelgisConfig. train/eval_dataloader: DataLoaders.
        criterion: Loss (optional if model returns loss). optimizer: optional, else created.
        forward_fn: (model, batch) -> (loss, logits). compute_metrics: (preds, labels) -> dict.
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.forward_fn = forward_fn
        self.compute_metrics = compute_metrics

        self.device = get_device(config.device)
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)

        seed_everything(config.seed)

        if optimizer is None:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_params,
                lr=1e-3,  # Will be updated by LR Finder
                weight_decay=config.weight_decay,
            )
        self.optimizer = optimizer

        # LR Finder
        if config.lr_finder_enabled:
            lr_finder = LRFinder(
                model,
                self.optimizer,
                criterion,
                self.device,
                trainable_only=getattr(config, "lr_finder_trainable_only", False),
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

        # Scheduler: total optimizer steps (accounting for gradient accumulation)
        steps_per_epoch = (len(train_dataloader) + config.gradient_accumulation_steps - 1) // max(
            config.gradient_accumulation_steps, 1
        )
        num_training_steps = steps_per_epoch * config.max_epochs
        self.scheduler = SmartScheduler(
            self.optimizer,
            self.initial_lr,
            config,
            num_training_steps=num_training_steps,
        )

        # SelgisCore
        self.selgis = SelgisCore(
            model, self.optimizer, self.scheduler, config, self.device
        )

        self.callbacks = callbacks or [LoggingCallback()]
        
        # Ensure HistoryCallback is present
        if not any(isinstance(cb, HistoryCallback) for cb in self.callbacks):
            self.callbacks.append(HistoryCallback(output_dir=config.output_dir))

        if (
            getattr(config, "sparsity_enabled", False)
            and not any(isinstance(cb, SparsityCallback) for cb in self.callbacks)
        ):
            sparsity_cb = SparsityCallback(
                target_sparsity=getattr(config, "sparsity_target", 0.0),
                start_epoch=getattr(config, "sparsity_start_epoch", 0),
                frequency=getattr(config, "sparsity_frequency", 1),
            )
            self.callbacks.append(sparsity_cb)

        # State
        self._global_step = 0
        self._current_epoch = 0

    def train(self) -> dict[str, Any]:
        """Run training loop. Returns final metrics dict."""
        self._call_callbacks("on_train_begin")

        metrics = {}

        for epoch in range(self.config.max_epochs):
            self._current_epoch = epoch
            self._call_callbacks("on_epoch_begin", epoch=epoch)

            # Training
            train_loss = self._train_epoch()

            # Evaluation
            metrics = {"train_loss": train_loss}
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                metrics.update(eval_metrics)

            self._call_callbacks("on_epoch_end", epoch=epoch, metrics=metrics)

            # Check stopping
            status = self.selgis.eval_epoch(
                metrics,
                epoch,
                primary_metric="accuracy" if "accuracy" in metrics else "loss",
                higher_is_better="accuracy" in metrics,
            )

            if status == "STOP":
                break

            # Check callback stopping
            should_stop = False
            for cb in self.callbacks:
                if hasattr(cb, "should_stop") and cb.should_stop:
                    should_stop = True
                    break

            if should_stop:
                break

        self.selgis.load_best_weights()
        self._call_callbacks("on_train_end")

        return metrics

    def _train_epoch(self) -> float:
        """Single training epoch with gradient accumulation; returns average loss."""
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
                # Rollback: reset accumulation
                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            self._call_callbacks(
                "on_step_end",
                step=self._global_step,
                loss=loss or 0.0,
            )

        # Last partial accumulation in epoch
        if accum_count > 0:
            self.selgis.optimizer_step()
            self._step_scheduler_if_needed()
            self._global_step += 1

        return total_loss / max(num_steps, 1)

    def _step_scheduler_if_needed(self) -> None:
        """Call scheduler.step() when using step-based schedule (warmup_ratio > 0)."""
        if getattr(self.config, "warmup_ratio", 0) <= 0:
            return
        if hasattr(self.scheduler, "step"):
            self.scheduler.step()

    def _training_step(self, batch: Any) -> float | None:
        """
        Single training step: forward, loss check, backward. No optimizer step here
        (handled in _train_epoch for gradient accumulation). Returns loss or None if rollback.
        """
        batch = move_to_device(batch, self.device)

        with self.selgis.get_amp_context():
            loss, _ = self._forward(batch)

        if not self.selgis.check_loss(loss):
            return None

        # Scale loss for accumulation (so mean over accum_steps)
        accum_steps = max(self.config.gradient_accumulation_steps, 1)
        scaled_loss = loss / accum_steps
        self.selgis.backward_step(scaled_loss)

        return loss.item()

    def _forward(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: custom forward_fn or model(**inputs) / model(inputs) + criterion."""
        if self.forward_fn is not None:
            return self.forward_fn(self.model, batch)

        inputs, labels = unpack_batch(batch)

        if is_dict_like(inputs):
            inputs_dict = dict(inputs)
            outputs = self.model(**inputs_dict)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                return outputs.loss, logits

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if labels is not None and self.criterion is not None:
                loss = self.criterion(logits, labels)
                return loss, logits

            raise ValueError("Model doesn't return loss and no criterion provided")

        outputs = self.model(inputs)

        if self.criterion is None:
            raise ValueError("Criterion required for non-dict input")

        if labels is None:
            raise ValueError("Labels required for training")

        loss = self.criterion(outputs, labels)
        return loss, outputs

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation; returns metrics dict (e.g. loss, accuracy)."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        correct = 0
        total = 0

        for batch in self.eval_dataloader:
            batch = move_to_device(batch, self.device)

            with self.selgis.get_amp_context():
                loss, logits = self._forward(batch)

            total_loss += loss.item()

            if logits.dim() > 1:
                preds = logits.argmax(dim=-1)
            else:
                preds = logits

            _, labels = unpack_batch(batch)

            if labels is not None:
                if self.compute_metrics is not None:
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
                else:
                    if preds.shape != labels.shape:
                        preds = preds.view_as(labels)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

        metrics = {"loss": total_loss / len(self.eval_dataloader)}

        if self.compute_metrics is not None and all_preds and all_labels:
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            metrics.update(self.compute_metrics(preds, labels))
        elif self.compute_metrics is None and total > 0:
            metrics["accuracy"] = correct / total * 100

        self._call_callbacks("on_evaluate", metrics=metrics)
        return metrics

    def _call_callbacks(self, method: str, **kwargs) -> None:
        """Invoke callback method on all callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, method):
                getattr(callback, method)(self, **kwargs)

    def save_model(self, path: str) -> None:
        """Save model state_dict to path."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, weights_only: bool = True) -> None:
        """
        Load model state_dict from path.

        Args:
            path: Path to checkpoint file.
            weights_only: If True (default), load only tensors (safe for untrusted files).
                Requires PyTorch >= 2.0. Set False only for trusted checkpoints.
        """
        load_kwargs: dict = {"map_location": self.device}
        if weights_only:
            try:
                state = torch.load(path, **load_kwargs, weights_only=True)
            except TypeError:
                # Fallback with explicit warning for security
                print("[WARN] weights_only=True not supported (PyTorch < 2.0). Loading with weights_only=False — only use with trusted checkpoints!")
                state = torch.load(path, **load_kwargs)
        else:
            state = torch.load(path, **load_kwargs)
        self.model.load_state_dict(state, strict=True)


class TransformerTrainer(Trainer):
    """
    Trainer for HuggingFace Transformers: auto load model/tokenizer,
    PEFT/LoRA, gradient checkpointing.
    """
    def __init__(
        self,
        model_or_path: nn.Module | str,
        config: TransformerConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        tokenizer: Any = None,
        **kwargs,
    ) -> None:
        """model_or_path: model or pretrained path. config: TransformerConfig. tokenizer: optional."""
        self._transformer_config = config

        # Load model if path provided
        if isinstance(model_or_path, str):
            model = self._load_model(model_or_path, config)
        else:
            model = model_or_path

        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

        # PEFT/LoRA
        if config.use_peft and config.peft_config:
            model = self._apply_peft(model, config.peft_config, config.problem_type)

        # Create optimizer with proper param groups
        from selgis.utils import get_optimizer_grouped_params

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

        self.tokenizer = tokenizer

    def _load_model(self, path: str, config: TransformerConfig) -> nn.Module:
        """
        Load HuggingFace model by path and problem_type.

        Raises:
            ImportError: If transformers is not installed.
            ValueError: If problem_type is not supported.
        """
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForMaskedLM,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise ImportError("Install transformers: pip install transformers") from None

        # BitsAndBytes Config
        bnb_config = None
        if config.quantization_type != "no":
            try:
                from transformers import BitsAndBytesConfig
                import torch
                
                if config.quantization_type == "8bit":
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    print("[INFO] Quantization: 8-bit enabled")
                elif config.quantization_type == "4bit":
                    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
                    )
                    print(f"[INFO] Quantization: 4-bit enabled ({config.bnb_4bit_quant_type})")
            except ImportError:
                print("[WARN] bitsandbytes or transformers not installed/compatible. Proceeding without quantization.")
            except Exception as e:
                print(f"[WARN] Failed to configure quantization: {e}. Proceeding without it.")

        num_labels = config.num_labels
        trust = True
        load_kw = {
            "trust_remote_code": trust,
            "quantization_config": bnb_config
        }
        # Remove quantization_config if None (to avoid errors if not supported by all loaders)
        if bnb_config is None:
            load_kw.pop("quantization_config")

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
                path,
                num_labels=num_labels,
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
            f"Unsupported problem_type: {config.problem_type!r}.  "
            f"Supported: {', '.join(supported)}"
        )

    def _apply_peft(
        self,
        model: nn.Module,
        peft_config: dict,
        problem_type: str | None = None,
    ) -> nn.Module:
        """Apply PEFT/LoRA to model. Returns model with adapters."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
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
                    task_type = task_type_mapping.get(task_type_value.lower(), TaskType.SEQ_CLS)
                else:
                    task_type = task_type_value
            elif problem_type:
                task_type = task_type_mapping.get(problem_type, TaskType.SEQ_CLS)
            else:
                task_type = TaskType.SEQ_CLS

            filtered_config = {k: v for k, v in peft_config.items() if k != "task_type"}

            lora_config = LoraConfig(
                task_type=task_type,
                **filtered_config,
            )

            peft_model = get_peft_model(model, lora_config)

            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in peft_model.parameters())
            print(f"[INFO] LoRA applied: {trainable_params:,} / {total_params:,} params trainable  "
                  f"({100 * trainable_params / total_params:.2f}%)")

            return peft_model

        except ImportError:
            raise ImportError(
                "PEFT is required when use_peft=True. Install with: pip install peft"
            ) from None

    def _create_optimizer(
        self,
        param_groups: list[dict],
        config: TransformerConfig,
    ) -> optim.Optimizer:
        """Create optimizer from config (adamw, adam, sgd, adafactor)."""
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
                print("[WARN] Adafactor not available, falling back to AdamW")

        return optim.AdamW(param_groups, lr=config.learning_rate)

    def save_pretrained(self, path: str) -> None:
        """Save model in HuggingFace format (adapters only for PEFT)."""
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
            print(f"[SAVE] Model saved to {path}")
        else:
            self.save_model(f"{path}/pytorch_model.pt")

    def push_to_hub(self, repo_id: str, **kwargs) -> None:
        """Push model (and tokenizer) to HuggingFace Hub."""
        if hasattr(self.model, "push_to_hub"):
            self.model.push_to_hub(repo_id, **kwargs)
            if self.tokenizer:
                self.tokenizer.push_to_hub(repo_id, **kwargs)
            print(f"[INFO] Model pushed to https://huggingface.co/{repo_id}")
        else:
            raise ValueError("Model doesn't support push_to_hub. Use save_pretrained instead.")