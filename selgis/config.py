"""Configuration dataclasses for generic and Transformer training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SelgisConfig:
    """Base training configuration.

    Controls training loop, early stopping, gradient clipping, anomaly
    detection, learning rate scheduling, regularization, mixed precision,
    logging, checkpointing, device placement, and reproducibility.
    """

    # === Training ===
    batch_size: int = 32
    eval_batch_size: int = 64
    max_epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3

    # === Early Stopping ===
    patience: int = 5
    min_delta: float = 1e-4
    final_surge_factor: float = 5.0
    primary_metric: str | None = None

    # === Gradient ===
    grad_clip_norm: float = 1.0
    grad_clip_value: float | None = None

    # === Anomaly Detection ===
    spike_threshold: float = 3.0
    min_history_len: int = 10
    nan_recovery: bool = True

    # === LR Finder ===
    lr_finder_enabled: bool = False
    lr_finder_trainable_only: bool = False
    lr_finder_save_optimizer_state: bool = False
    lr_finder_start: float = 1e-7
    lr_finder_end: float = 1.0
    lr_finder_steps: int = 100

    # === Scheduler ===
    warmup_epochs: int = 0
    warmup_ratio: float = 0.0
    min_lr: float = 1e-7
    scheduler_type: Literal[
        "cosine",
        "cosine_restart",
        "linear",
        "constant",
        "polynomial",
    ] = "cosine_restart"
    t_0: int = 10
    t_mult: int = 2

    # === Regularization ===
    label_smoothing: float = 0.1
    weight_decay: float = 0.01
    sparsity_enabled: bool = False
    sparsity_target: float = 0.0
    sparsity_start_epoch: int = 0
    sparsity_frequency: int = 1

    # === Mixed Precision ===
    fp16: bool = False
    bf16: bool = False

    # === Logging ===
    logging_steps: int = 10
    eval_steps: int | None = None
    save_steps: int | None = None

    # === Checkpointing ===
    output_dir: str = "./output"
    save_total_limit: int = 3
    save_best_only: bool = True
    state_storage: Literal["disk", "memory"] = "disk"
    state_dir: str | None = None
    state_update_interval: int = 100
    resume_from_checkpoint: str | None = None

    # === Device ===
    device: str = "auto"
    cpu_offload: bool = False
    empty_cache_steps: int = 0
    gc_collect_steps: int = 0

    # === Reproducibility ===
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration constraints.

        Raises:
            ValueError: If ``fp16`` and ``bf16`` are both enabled,
                both warmup modes are set, or
                ``gradient_accumulation_steps`` is not positive.
        """
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        if self.warmup_epochs > 0 and self.warmup_ratio > 0:
            raise ValueError("Use either warmup_epochs or warmup_ratio, not both")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be positive, got {self.eval_batch_size}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.save_total_limit is not None and self.save_total_limit < 0:
            raise ValueError(f"save_total_limit must be non-negative, got {self.save_total_limit}")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}")
        if self.spike_threshold <= 0:
            raise ValueError(f"spike_threshold must be positive, got {self.spike_threshold}")
        if self.min_history_len <= 0:
            raise ValueError(f"min_history_len must be positive, got {self.min_history_len}")
        if self.state_update_interval <= 0:
            raise ValueError(
                f"state_update_interval must be positive, got {self.state_update_interval}"
            )


@dataclass
class TransformerConfig(SelgisConfig):
    """Configuration for Transformer (HuggingFace) models.

    Extends ``SelgisConfig`` with model loading, tokenizer, optimizer,
    PEFT/LoRA, gradient checkpointing, quantization, and device-map
    settings.
    """

    # === Model ===
    model_name_or_path: str = ""
    num_labels: int = 2
    problem_type: Literal[
        "single_label_classification",
        "multi_label_classification",
        "regression",
        "seq2seq",
        "causal_lm",
        "masked_lm",
    ] = "single_label_classification"
    trust_remote_code: bool = False
    device_map: str | None = None

    # === Tokenizer ===
    max_length: int = 512
    padding: Literal[
        "max_length",
        "longest",
        "do_not_pad",
    ] = "max_length"
    truncation: bool = True

    # === Optimizer ===
    optimizer_type: Literal[
        "adamw",
        "adam",
        "sgd",
        "adafactor",
    ] = "adamw"
    learning_rate: float = 2e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # === LoRA / PEFT ===
    use_peft: bool = False
    peft_config: dict[str, Any] = field(default_factory=dict)
    adapter_name_or_path: str | None = None

    # === Gradient Checkpointing ===
    gradient_checkpointing: bool = False
    gc_checkpoint_interval: int = 1

    # === Chunked Cross-Entropy ===
    chunked_ce: bool = False
    ce_chunk_size: int = 1024

    # === Flash Attention ===
    flash_attention: bool = False

    # === Unsloth ===
    use_unsloth: bool = False

    # === DeepSpeed ===
    deepspeed_config: str | None = None

    # === Quantization (BitsAndBytes) ===
    quantization_type: Literal["no", "8bit", "4bit"] = "no"
    bnb_4bit_compute_dtype: Literal[
        "float16",
        "bfloat16",
        "float32",
    ] = "float16"
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    bnb_4bit_use_double_quant: bool = False

    def __post_init__(self) -> None:
        """Validate Transformer-specific configuration constraints.

        Raises:
            ValueError: If ``use_peft`` is enabled with an empty
                ``peft_config``, or quantization requires GPU but
                ``device`` is set to CPU.
        """
        super().__post_init__()

        if self.use_peft and not self.peft_config and not self.adapter_name_or_path:
            raise ValueError(
                "peft_config or adapter_name_or_path must be provided "
                "when use_peft=True. Example: "
                "peft_config={'r': 16, 'lora_alpha': 32, "
                "'lora_dropout': 0.05}"
            )

        if self.quantization_type != "no" and self.device == "cpu":
            raise ValueError(
                f"quantization_type={self.quantization_type!r} requires GPU, but device='cpu'"
            )

        if self.chunked_ce and self.ce_chunk_size <= 0:
            raise ValueError(f"ce_chunk_size must be positive, got {self.ce_chunk_size}")

        if self.gc_checkpoint_interval < 1:
            raise ValueError(
                f"gc_checkpoint_interval must be >= 1, got {self.gc_checkpoint_interval}"
            )

        if self.use_unsloth and not self.use_peft:
            print("[WARN] use_unsloth=True without use_peft=True. Applying Unsloth only.")
