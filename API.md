

## Обновлённая документация API


# Selgis ML Library - Complete API Documentation

**Version:** 0.2.3
**Description:** Universal Training Framework for PyTorch and HuggingFace Transformers

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Datasets](#datasets)
5. [Training](#training)
6. [Callbacks](#callbacks)
7. [Utilities](#utilities)
8. [Advanced Features](#advanced-features)
9. [Complete Examples](#complete-examples)
10. [CLI](#cli-command-line-interface)
11. [Error Handling & Troubleshooting](#error-handling--troubleshooting)
12. [Security & Breaking Changes](#security--breaking-changes)
13. [License & Support](#license--support)

---

## Installation

```bash
# Basic installation
pip install selgis

# With all dependencies
pip install selgis[all]

# Specific components
pip install selgis[transformers]  # HuggingFace support
pip install selgis[peft]          # LoRA/PEFT support
pip install selgis[llm]           # LLM training (4-bit/8-bit)
pip install selgis[tracking]      # Weights & Biases
```

---

## Quick Start

### Basic Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from selgis import Trainer, SelgisConfig

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

# Create dataset
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Configure training
config = SelgisConfig(
    max_epochs=10,
    batch_size=32,
    lr_finder_enabled=True,
    fp16=False,
)

# Create trainer and train
trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    eval_dataloader=loader,
    criterion=nn.CrossEntropyLoss(),
)

trainer.train()
```

### Transformer Fine-tuning Example

```python
from transformers import AutoTokenizer
from selgis import TransformerTrainer, TransformerConfig, create_dataloaders, DatasetConfig

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Configure dataset
dataset_config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",
    tokenizer=tokenizer,
    max_length=512,
    batch_size=32,
)

train_loader, eval_loader = create_dataloaders(dataset_config)

# Configure Transformer training
config = TransformerConfig(
    model_name_or_path="bert-base-uncased",
    use_peft=True,
    peft_config={
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["query", "value"],
    },
    learning_rate=2e-5,
    max_epochs=3,
)

# Train
trainer = TransformerTrainer(
    model_or_path="bert-base-uncased",
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

trainer.train()
```

---

## Configuration

### SelgisConfig

Base training configuration dataclass.

```python
from selgis import SelgisConfig

config = SelgisConfig(
    # Training
    batch_size=32,
    eval_batch_size=64,
    max_epochs=100,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,

    # Early Stopping
    patience=5,
    min_delta=1e-4,
    final_surge_factor=5.0,
    primary_metric=None,

    # Gradient
    grad_clip_norm=1.0,
    grad_clip_value=None,

    # Anomaly Detection
    spike_threshold=3.0,
    min_history_len=10,
    nan_recovery=True,

    # LR Finder
    lr_finder_enabled=False,
    lr_finder_trainable_only=False,
    lr_finder_start=1e-7,
    lr_finder_end=1.0,
    lr_finder_steps=100,

    # Scheduler
    warmup_epochs=0,
    warmup_ratio=0.0,
    min_lr=1e-7,
    scheduler_type="cosine_restart",
    t_0=10,
    t_mult=2,

    # Regularization
    label_smoothing=0.1,
    weight_decay=0.01,
    sparsity_enabled=False,
    sparsity_target=0.0,
    sparsity_start_epoch=0,
    sparsity_frequency=1,

    # Mixed Precision
    fp16=False,
    bf16=False,

    # Logging
    logging_steps=10,
    eval_steps=None,
    save_steps=None,

    # Checkpointing
    output_dir="./output",
    save_total_limit=3,
    save_best_only=True,
    state_storage="disk",
    state_dir=None,
    state_update_interval=100,

    # Device
    device="auto",
    cpu_offload=False,

    # Reproducibility
    seed=42,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Training batch size |
| `eval_batch_size` | int | 64 | Evaluation batch size |
| `max_epochs` | int | 100 | Maximum number of training epochs |
| `gradient_accumulation_steps` | int | 1 | Steps for gradient accumulation. Must be >= 1. |
| `learning_rate` | float | 1e-3 | Initial learning rate for auto-created optimizer. Overridden by LR Finder when enabled. |
| `patience` | int | 5 | Epochs to wait before early stopping triggers |
| `min_delta` | float | 1e-4 | Minimum change in metric to qualify as improvement |
| `final_surge_factor` | float | 5.0 | LR multiplier for final surge before early stopping (set to 0 to disable). When training stalls, this gives the model one last chance to improve by temporarily increasing the learning rate. |
| `primary_metric` | str or None | None | Primary metric for early stopping (e.g. `"accuracy"`, `"f1"`, `"loss"`). When None, auto-selects `"accuracy"` if present, otherwise `"loss"`. |
| `grad_clip_norm` | float | 1.0 | Maximum gradient norm for clipping |
| `grad_clip_value` | float or None | None | Maximum gradient value for clipping |
| `spike_threshold` | float | 3.0 | Loss spike detection threshold (multiplier). If current loss exceeds this factor times the running average, it's flagged as a spike. |
| `min_history_len` | int | 10 | Minimum history length for spike detection. Ensures enough data points before anomaly detection activates. |
| `nan_recovery` | bool | True | Enable automatic recovery from NaN/Inf loss. When enabled, the trainer will attempt to restore previous good state and reduce LR. |
| `lr_finder_enabled` | bool | False | Enable learning rate finder at training start. Automatically searches for optimal initial LR. Set to True when you want auto-tuned LR instead of using the configured `learning_rate`. |
| `lr_finder_trainable_only` | bool | False | Only tune trainable parameters during LR search. Saves memory for large/LoRA models. |
| `lr_finder_start` | float | 1e-7 | Starting LR for finder search. |
| `lr_finder_end` | float | 1.0 | Ending LR for finder search. |
| `lr_finder_steps` | int | 100 | Number of steps for LR search. More steps = finer granularity. |
| `warmup_epochs` | int | 0 | Number of warmup epochs. LR linearly increases from 0 to initial_lr during warmup. |
| `warmup_ratio` | float | 0.0 | Warmup ratio of total training steps. Alternative to warmup_epochs (use one or the other). |
| `min_lr` | float | 1e-7 | Minimum learning rate. LR will never go below this value during scheduling. |
| `scheduler_type` | str | "cosine_restart" | LR scheduler type: `cosine` (smooth decay), `cosine_restart` (with restarts), `linear` (linear decay), `constant` (no decay), `polynomial` (power-law decay). |
| `t_0` | int | 10 | Initial period for cosine restart scheduler (in epochs). After t_0 epochs, the cycle restarts. |
| `t_mult` | int | 2 | Period multiplier for cosine restart. After each restart, the period is multiplied by t_mult. |
| `label_smoothing` | float | 0.1 | Label smoothing factor. Reduces overconfidence by smoothing target labels (0 = no smoothing, 1 = uniform). |
| `weight_decay` | float | 0.01 | Weight decay (L2 regularization) for optimizer. Helps prevent overfitting. |
| `fp16` | bool | False | Enable FP16 mixed precision. Reduces memory usage and can speed up training on NVIDIA GPUs. |
| `bf16` | bool | False | Enable BF16 mixed precision. Better for training stability on Ampere+ GPUs (A100, RTX 30xx+). |
| `logging_steps` | int | 10 | Log training metrics every N steps. Lower values give finer-grained logs but increase output. |
| `output_dir` | str | "./output" | Output directory for checkpoints and training state. |
| `save_total_limit` | int | 3 | Maximum number of checkpoints to keep. Older checkpoints are automatically deleted. |
| `save_best_only` | bool | True | Save only the best checkpoint (based on eval metric). Set False to save all epochs. |
| `state_storage` | str | "disk" | State storage mode: `disk` (saves RAM, slower) or `memory` (faster, uses more RAM). Used for rollback on NaN/Inf. |
| `device` | str | "auto" | Device selection: `auto` (cuda > mps > cpu), `cuda`, `cpu`, `mps`. |
| `cpu_offload` | bool | False | Offload optimizer states and gradients to CPU. Reduces GPU VRAM usage at slight speed cost. Independent from `device_map`. |
| `seed` | int | 42 | Random seed for reproducibility. Sets seeds for random, numpy, torch, and CUDA. Also sets `cudnn.deterministic=True` and `cudnn.benchmark=False` for full reproducibility. |

#### Validation Rules

The following constraints are enforced in `__post_init__`:

| Constraint | Exception |
|------------|-----------|
| `fp16` and `bf16` cannot both be True | `ValueError` |
| `warmup_epochs` and `warmup_ratio` cannot both be > 0 | `ValueError` |
| `gradient_accumulation_steps` must be >= 1 | `ValueError` |

### TransformerConfig

Extended configuration for HuggingFace Transformers.

```python
from selgis import TransformerConfig

config = TransformerConfig(
    # Inherited from SelgisConfig (all parameters above)

    # Model
    model_name_or_path="bert-base-uncased",
    num_labels=2,
    problem_type="single_label_classification",
    trust_remote_code=False,
    device_map=None,

    # Tokenizer
    max_length=512,
    padding="max_length",
    truncation=True,

    # Optimizer
    optimizer_type="adamw",
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,

    # LoRA / PEFT
    use_peft=False,
    peft_config={
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["query", "value"],
        "bias": "none",
        "task_type": "SEQ_CLS",
    },

    # Gradient Checkpointing
    gradient_checkpointing=False,

    # Quantization
    quantization_type="no",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)
```

#### TransformerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | "" | Model name or path to load from HuggingFace Hub or local directory. |
| `num_labels` | int | 2 | Number of labels for classification tasks. Ignored for causal_lm. |
| `problem_type` | str | "single_label_classification" | Task type: `single_label_classification`, `multi_label_classification`, `regression`, `seq2seq`, `causal_lm`, `masked_lm`. |
| `trust_remote_code` | bool | False | Allow execution of custom code from HuggingFace Hub repositories. **Security warning:** only enable for trusted models. |
| `device_map` | str or None | None | Device placement strategy for model layers. Set to `"auto"` for automatic distribution across available devices (GPU + CPU). Independent from `cpu_offload`. Common values: `None` (single device), `"auto"`, `"balanced"`, `"sequential"`. |
| `max_length` | int | 512 | Maximum sequence length. Longer sequences are truncated. |
| `padding` | str | "max_length" | Padding strategy: `max_length` (pad to max_length), `longest` (pad to longest in batch), `do_not_pad`. |
| `truncation` | bool | True | Truncate sequences longer than max_length. |
| `optimizer_type` | str | "adamw" | Optimizer type: `adamw` (recommended), `adam`, `sgd`, `adafactor`. |
| `learning_rate` | float | 2e-5 | Initial learning rate. Overrides `SelgisConfig.learning_rate`. Use LR Finder for optimal value. |
| `adam_beta1` | float | 0.9 | AdamW beta1 parameter (exponential decay rate for first moment). |
| `adam_beta2` | float | 0.999 | AdamW beta2 parameter (exponential decay rate for second moment). |
| `adam_epsilon` | float | 1e-8 | AdamW epsilon parameter (numerical stability). |
| `use_peft` | bool | False | Enable PEFT/LoRA for parameter-efficient fine-tuning. Requires `peft_config` to be non-empty. |
| `peft_config` | dict | {} | LoRA configuration: `r` (rank), `lora_alpha` (scaling), `lora_dropout`, `target_modules`, `bias`, `task_type`. Must be provided when `use_peft=True`. |
| `gradient_checkpointing` | bool | False | Enable gradient checkpointing. Reduces memory by ~40% at slight speed cost. |
| `quantization_type` | str | "no" | Quantization type: `no`, `8bit`, `4bit`. Requires GPU (cannot be used with `device="cpu"`). |
| `bnb_4bit_compute_dtype` | str | "float16" | Compute dtype for 4-bit quantization: `float16`, `bfloat16` (recommended for Ampere+). |
| `bnb_4bit_quant_type` | str | "nf4" | Quantization type: `nf4` (Normal Float 4, better for weights) or `fp4`. |
| `bnb_4bit_use_double_quant` | bool | False | Enable double quantization. Saves ~0.4 bits/parameter. |

#### TransformerConfig Validation Rules

| Constraint | Exception |
|------------|-----------|
| All `SelgisConfig` validations | (inherited) |
| `use_peft=True` requires non-empty `peft_config` | `ValueError` |
| `quantization_type` other than `"no"` requires `device != "cpu"` | `ValueError` |

#### `cpu_offload` vs `device_map`

These are **independent** features that can be used separately or together:

| Feature | Purpose | Effect |
|---------|---------|--------|
| `cpu_offload=True` | Offload optimizer states and gradients to CPU | Saves VRAM (~30-40%), model stays on GPU |
| `device_map="auto"` | Distribute model layers across devices | Splits large models across GPU + CPU |

```python
# Optimizer offload only (model on single GPU)
config = TransformerConfig(cpu_offload=True)

# Model layer distribution only
config = TransformerConfig(device_map="auto")

# Both (maximum memory savings)
config = TransformerConfig(cpu_offload=True, device_map="auto")
```

#### LoRA PEFT Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | LoRA rank. Higher = more parameters, better quality. Common: 8, 16, 32. |
| `lora_alpha` | int | 16 | LoRA alpha scaling. Typically 2 * r. |
| `lora_dropout` | float | 0.05 | Dropout for LoRA layers. |
| `target_modules` | list | [] | Modules to apply LoRA to. For LLM: `["q_proj", "v_proj"]` or `["query", "value"]`. |
| `bias` | str | "none" | Which biases to train: `none`, `all`, `lora_only`. |
| `task_type` | str | None | Task type for PEFT: `SEQ_CLS` (classification), `CAUSAL_LM`, `SEQ_2_SEQ_LM`, etc. |

#### Problem Types

| Problem Type | Description |
|--------------|-------------|
| `single_label_classification` | Single label classification |
| `multi_label_classification` | Multi-label classification |
| `regression` | Regression task |
| `seq2seq` | Sequence-to-sequence (T5, BART) |
| `causal_lm` | Causal language modeling (GPT) |
| `masked_lm` | Masked language modeling (BERT) |

---

## Datasets

### DatasetConfig

Configuration for creating datasets.

```python
from selgis import DatasetConfig

config = DatasetConfig(
    # Data Type
    data_type="text",  # text, image, multimodal, custom, streaming, tabular

    # Paths
    data_path="./data.jsonl",
    train_path="./train.jsonl",
    eval_path="./eval.jsonl",

    # For Multimodal/Tabular
    image_path="./images",
    image_column="image",
    text_column="text",
    label_column="label",

    # Loading Parameters
    batch_size=32,
    eval_batch_size=64,
    num_workers=0,
    prefetch_factor=None,
    pin_memory=True,
    persistent_workers=False,

    # Tokenization / Transforms
    tokenizer=None,
    image_processor=None,
    transform=None,
    format_fn=None,

    # Caching
    cache_dir="./cache",
    use_cache=True,
    pre_tokenize=False,
    pre_compute_features=False,

    # Streaming
    streaming=False,
    buffer_size=1000,

    # Data Split
    train_split=0.9,
    seed=42,

    # Distributed Training
    world_size=1,
    rank=0,

    # Additional
    max_length=512,
    custom_kwargs={},
)
```

### create_dataset

Create a dataset from configuration.

```python
from selgis import create_dataset, DatasetConfig

config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",
    tokenizer=tokenizer,
    max_length=512,
)

dataset = create_dataset(config)
```

### create_dataloaders

Create train and eval DataLoaders.

```python
from selgis import create_dataloaders, DatasetConfig

config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",
    tokenizer=tokenizer,
    batch_size=32,
    num_workers=4,
)

train_loader, eval_loader = create_dataloaders(config)

print(f"Train batches: {len(train_loader)}")
print(f"Eval batches: {len(eval_loader)}")
```

### TextDataset

Dataset for text data with optimized loading.

```python
from selgis import TextDataset

dataset = TextDataset(
    data_path="./data/dialogues.jsonl",
    tokenizer=tokenizer,
    max_length=512,
    cache_dir="./cache",
    pre_tokenize=True,
    use_mmap=True,  # Memory-mapped file for fast access
)

print(f"Dataset size: {len(dataset)}")
print(f"Stats: {dataset.get_stats()}")
```

#### Features

- **Memory-mapped files** for fast lazy loading
- **Pre-tokenization** with disk caching
- **Dialogue formatting** for LLM fine-tuning
- **Performance metrics** tracking

### HFTextDataset

Dataset from HuggingFace datasets with auto-caching.

```python
from selgis import HFTextDataset

dataset = HFTextDataset(
    dataset_name="tatsu-lab/alpaca",
    tokenizer=tokenizer,
    max_length=512,
    cache_dir="./cache",
    streaming=False,
    text_column="instruction",
)
```

#### Features

- **Automatic caching** of tokenized data
- **Batched tokenization** (10x faster)
- **Streaming support** for large datasets
- **HuggingFace Hub integration**

### ImageDataset

Dataset for image classification.

```python
from torchvision import transforms
from selgis import ImageDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
])

dataset = ImageDataset(
    data_path="./data/imagenet",
    transform=transform,
    file_format="folder",  # folder, csv, json
)

print(f"Classes: {dataset._class_names}")
print(f"Samples: {len(dataset)}")
```

#### Folder Structure

```
data/imagenet/
├── class_0/
│   ├── img1.jpg
│   └── img2.jpg
├── class_1/
│   ├── img3.jpg
│   └── img4.jpg
└── ...
```

### MultimodalDataset

Dataset for text + image data (LLaVA, BLIP style).

```python
from transformers import AutoTokenizer, AutoImageProcessor
from selgis import MultimodalDataset

tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
image_processor = AutoImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

dataset = MultimodalDataset(
    data_path="./data/llava_dataset.jsonl",
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_length=512,
    image_root="./data/images",
)

# Sample format in JSONL:
# {"image": "img1.jpg", "question": "What is this?", "answer": "A cat"}
```

### StreamingTextDataset

Streaming dataset for large files (>100GB).

```python
from selgis import StreamingTextDataset

dataset = StreamingTextDataset(
    data_path="./data/huge_dataset.jsonl",
    tokenizer=tokenizer,
    max_length=512,
    buffer_size=1000,
)

# Works with DataLoader and num_workers
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

#### Features

- **No RAM loading** — reads line by line
- **Multi-worker support** with data partitioning
- **Compressed file support** (.gz, .zip)
- **Automatic file handle management**

### CustomDataset

Wrapper for custom PyTorch datasets.

```python
from torch.utils.data import Dataset
from selgis import CustomDataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = [...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.data[idx]),
            "labels": torch.tensor(self.target[idx]),
        }

# Wrap for Selgis
custom_dataset = CustomDataset(
    dataset=MyDataset(),
    wrap_key="inputs",
    label_key="labels",
)
```

---

## Training

### Trainer

Universal trainer for PyTorch models.

Device management is handled internally. The caller must not move batches to the device manually. The caller is responsible for calling `optimizer.zero_grad()` only when using `SelgisCore` directly; `Trainer` handles this automatically.

```python
from selgis import Trainer, SelgisConfig

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=None,  # Auto-created if None
    callbacks=None,  # Auto-created: Logging, History, Checkpoint
    forward_fn=None,  # Custom forward function
    compute_metrics=None,  # Custom metrics function
)

# Train
metrics = trainer.train()

# Save model
trainer.save_model("./model.pt")

# Load model
trainer.load_model("./model.pt")
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | Model to train |
| `config` | SelgisConfig | Training configuration |
| `train_dataloader` | DataLoader | Training data loader |
| `eval_dataloader` | DataLoader | Evaluation data loader (optional) |
| `criterion` | nn.Module | Loss function (optional if model returns loss) |
| `optimizer` | Optimizer | Optimizer (auto-created with `config.learning_rate` if None) |
| `callbacks` | List[Callback] | List of callbacks |
| `forward_fn` | Callable | Custom forward: `(model, batch) -> (loss, logits)` |
| `compute_metrics` | Callable | Custom metrics: `(preds, labels) -> dict` |

#### Custom Forward Function

```python
def forward_fn(model, batch):
    inputs = batch["input_ids"]
    labels = batch["labels"]

    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    return loss, outputs

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    forward_fn=forward_fn,
)
```

#### Custom Metrics

```python
def compute_metrics(preds, labels):
    preds = preds.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    compute_metrics=compute_metrics,
)
```

### TransformerTrainer

Trainer for HuggingFace Transformers with LoRA/PEFT support.

```python
from selgis import TransformerTrainer, TransformerConfig

config = TransformerConfig(
    model_name_or_path="bert-base-uncased",
    use_peft=True,
    peft_config={
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["query", "value"],
    },
    quantization_type="4bit",
    learning_rate=2e-5,
)

trainer = TransformerTrainer(
    model_or_path="bert-base-uncased",
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
)

trainer.train()

# Save model
trainer.save_pretrained("./saved_model")

# Push to HuggingFace Hub
trainer.push_to_hub("username/my-model")
```

#### Features

- **Automatic model loading** from path or HuggingFace Hub
- **PEFT/LoRA** for parameter-efficient fine-tuning
- **4-bit/8-bit quantization** via BitsAndBytes
- **Gradient checkpointing** for memory efficiency
- **Push to Hub** for easy sharing
- **`trust_remote_code`** control for secure model loading
- **`device_map`** for multi-device model distribution

---

## Callbacks

### Callback (Base Class)

Base class for all callbacks.

```python
from selgis import Callback

class MyCallback(Callback):
    def on_train_begin(self, trainer):
        print("Training started!")

    def on_epoch_begin(self, trainer, epoch):
        print(f"Epoch {epoch} starting...")

    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch} completed: {metrics}")

    def on_step_begin(self, trainer, step):
        pass

    def on_step_end(self, trainer, step, loss):
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}")

    def on_evaluate(self, trainer, metrics):
        print(f"Evaluation: {metrics}")

    def on_train_end(self, trainer):
        print("Training completed!")
```

### LoggingCallback

Console logging for training progress.

```python
from selgis import LoggingCallback

callback = LoggingCallback(log_every=10)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    callbacks=[callback],
)
```

#### Output Example

```
----------------------------------------
[INFO] Training started
----------------------------------------
  Step     10 | Loss: 0.6893 | LR: 1.00e-03
  Step     20 | Loss: 0.6471 | LR: 1.00e-03
[INFO] Epoch 0 | train_loss: 0.6949 | loss: 0.6818 | accuracy: 50.5000
----------------------------------------
[INFO] Training complete
----------------------------------------
```

### EarlyStoppingCallback

Early stopping based on a metric.

```python
from selgis import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=5,
    min_delta=1e-4,
    metric="loss",
    mode="min",  # min for loss, max for accuracy
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    callbacks=[callback],
)
```

### CheckpointCallback

Save model checkpoints during training.

```python
from selgis import CheckpointCallback

callback = CheckpointCallback(
    output_dir="./checkpoints",
    save_best_only=True,
    save_total_limit=3,
    metric="loss",
    mode="min",
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    callbacks=[callback],
)
```

#### Checkpoint Structure

```
checkpoints/
├── best_model/
│   ├── model.pt
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── metrics.json
├── checkpoint-epoch-0/
│   ├── model.pt
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── metrics.json
└── checkpoint-epoch-1/
    └── ...
```

### HistoryCallback

Save training history to JSON.

```python
from selgis import HistoryCallback

callback = HistoryCallback(output_dir="./output")

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    callbacks=[callback],
)

trainer.train()

# Access history
history = callback.history
for record in history:
    print(f"Epoch {record['epoch']}: loss={record['metrics']['loss']:.4f}")
```

#### Output JSON Structure

```json
{
  "config": {...},
  "model_type": "Sequential",
  "total_epochs": 10,
  "final_metrics": {"loss": 0.2873, "accuracy": 91.0},
  "history": [
    {"epoch": 0, "global_step": 10, "metrics": {"loss": 0.6818}},
    {"epoch": 1, "global_step": 20, "metrics": {"loss": 0.6628}}
  ]
}
```

### WandBCallback

Weights & Biases integration.

```python
from selgis import WandBCallback

callback = WandBCallback(
    project="my-project",
    name="experiment-1",
    config={"lr": 1e-3, "batch_size": 32},
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    callbacks=[callback],
)
```

### SparsityCallback

Magnitude pruning for model sparsity.

```python
from selgis import SparsityCallback

callback = SparsityCallback(
    target_sparsity=0.5,
    start_epoch=5,
    frequency=1,
    skip_lora=True,
    min_params_to_prune=1000,
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    callbacks=[callback],
)
```

---

## Utilities

### Device Management

```python
from selgis import get_device

# Auto-select best device
device = get_device("auto")  # cuda > mps > cpu

# Force specific device
device = get_device("cuda")
device = get_device("cuda:1")  # Specific GPU
device = get_device("cpu")
device = get_device("mps")
```

#### Output

```
[INFO] Device: cuda
   GPU: NVIDIA GeForce GTX 1660 Ti
   Memory: 6.00 GB
```

### Seeding

```python
from selgis import seed_everything

# Full reproducibility (default)
seed_everything(42)

# Reproducibility with cudnn.benchmark for speed (non-deterministic)
seed_everything(42, deterministic=False)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | — | Random seed value |
| `deterministic` | bool | True | When True, sets `cudnn.deterministic=True` and `cudnn.benchmark=False` for full reproducibility. When False, enables `cudnn.benchmark=True` for potential speed gains at the cost of non-determinism. |

### Parameter Counting

```python
from selgis import count_parameters, format_params

model = MyModel()

# Count parameters
trainable = count_parameters(model, trainable_only=True)
total = count_parameters(model, trainable_only=False)

# Format for display
print(f"Trainable: {format_params(trainable)}")  # e.g., "1.2M"
print(f"Total: {format_params(total)}")  # e.g., "3.4B"
```

### Batch Utilities

```python
from selgis import move_to_device, unpack_batch, is_dict_like, to_dict

# Move batch to device (non_blocking=False by default for safety)
batch = move_to_device(batch, device)

# With pinned memory (DataLoader(pin_memory=True))
batch = move_to_device(batch, device, non_blocking=True)

# Unpack batch
inputs, labels = unpack_batch(batch)

# Check if dict-like
if is_dict_like(inputs):
    inputs_dict = to_dict(inputs)
```

### Optimizer Parameter Groups

```python
from selgis import get_optimizer_grouped_params

# Create parameter groups with weight decay exclusion
# Empty groups are automatically omitted
param_groups = get_optimizer_grouped_params(
    model,
    weight_decay=0.01,
    no_decay_keywords=("bias", "LayerNorm", "layer_norm"),
)

optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
```

---

## Advanced Features

### SelgisCore

Low-level training protection and optimization core.

The caller is responsible for invoking `optimizer.zero_grad()` before each `backward_step` call.

```python
from selgis import SelgisCore, SelgisConfig, SmartScheduler
import torch.optim as optim

# Create components
model = MyModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = SmartScheduler(optimizer, initial_lr=1e-3, config=config)
device = torch.device("cuda")

# Create core
core = SelgisCore(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    device=device,
)

# Training loop
for epoch in range(config.max_epochs):
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        loss = compute_loss(model, batch)

        # Check for anomalies
        if not core.check_loss(loss):
            continue  # Rollback triggered, optimizer state cleared

        # Backward pass
        core.backward_step(loss)

        # Optimizer step (includes gradient clipping, AMP, offload)
        core.optimizer_step()
```

#### Features

- **NaN/Inf detection** with automatic rollback and optimizer state reset
- **Loss spike detection** based on running history
- **Gradient clipping** (norm and value) without full fp32 copies
- **Mixed precision** support (FP16/BF16)
- **CPU offload** with full onload/offload cycle for optimizer states
- **Memory-efficient** state management (trainable parameters only)

### LRFinder

Learning rate finder using Leslie Smith's method.

```python
from selgis import LRFinder
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

lr_finder = LRFinder(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    device=torch.device("cuda"),
    trainable_only=False,  # Save memory for LoRA models
    amp_dtype=torch.float16,  # Match training precision
)

optimal_lr = lr_finder.find(
    train_loader=train_loader,
    start_lr=1e-7,
    end_lr=1.0,
    num_steps=100,
)

print(f"Optimal LR: {optimal_lr:.2e}")

# Plot results (using matplotlib)
import matplotlib.pyplot as plt
history = lr_finder.history
plt.plot(history["lrs"], history["losses"])
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.show()
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | — | Model to tune |
| `optimizer` | Optimizer | — | Optimizer (LR will be swept) |
| `criterion` | nn.Module or None | None | Loss function. None if model returns loss. |
| `device` | torch.device or None | None | Device. Defaults to model's device. |
| `trainable_only` | bool | False | Clone/restore only trainable parameters. |
| `amp_dtype` | torch.dtype or None | None | Autocast dtype (e.g. `torch.float16`) to match training precision. |

### SmartScheduler

Learning rate scheduler with warmup, restarts, and persistent manual adjustments.

```python
from selgis import SmartScheduler, SelgisConfig
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

config = SelgisConfig(
    scheduler_type="cosine_restart",
    warmup_ratio=0.1,
    min_lr=1e-7,
    t_0=10,
    t_mult=2,
)

scheduler = SmartScheduler(
    optimizer=optimizer,
    initial_lr=1e-3,
    config=config,
    num_training_steps=10000,
)

# Step-based (for warmup_ratio > 0)
for step in range(10000):
    train_step()
    scheduler.step()

# Or epoch-based (for warmup_epochs > 0)
for epoch in range(max_epochs):
    train_epoch()
    scheduler.step_epoch(epoch)

# Manual adjustment — persists across subsequent step() calls
scheduler.reduce_lr(factor=0.5)  # Reduce by 50%
scheduler.surge_lr(factor=3.0)   # Increase by 3x (capped at initial_lr)
```

#### Persistent LR Adjustments

Unlike previous versions, `reduce_lr` and `surge_lr` now **persist** across subsequent `step()` / `step_epoch()` calls. The scheduler maintains an internal `_lr_scale` factor that modifies the base LR used by all schedule computations.

```python
# Before reduce_lr: cosine computes LR from initial_lr=1e-3
scheduler.reduce_lr(factor=0.5)
# After reduce_lr: cosine computes LR from effective_lr=5e-4
# This persists through all future step() calls
```

#### State Dict

```python
# Save scheduler state
state = scheduler.state_dict()
# Returns: {"_step": 100, "_epoch": 5, "_lr_scale": 1.0, "_base_lr": 1e-3, "initial_lr": 1e-3}

# Restore scheduler state
scheduler.load_state_dict(state)
```

#### Scheduler Types

| Type | Description |
|------|-------------|
| `cosine` | Cosine annealing (clamped to `min_lr`) |
| `cosine_restart` | Cosine with warm restarts / SGDR (works in both epoch and step modes) |
| `linear` | Linear decay (clamped to `min_lr`) |
| `constant` | Constant LR |
| `polynomial` | Polynomial decay with power 2.0 (clamped to `min_lr`) |

### CPU Offload

Offload optimizer states and gradients to CPU.

```python
from selgis import SelgisConfig

config = SelgisConfig(
    cpu_offload=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
)
```

#### How It Works

1. **After backward**: Gradients are moved to CPU via hooks
2. **Before optimizer step**: Gradients and optimizer states are moved back to GPU
3. **After optimizer step**: Optimizer states are moved back to CPU, GPU cache cleared

This full onload/offload cycle ensures correct optimizer behavior while minimizing GPU memory usage.

#### Benefits

- **Reduced GPU memory** usage (up to 40%)
- **Larger batch sizes** possible
- **Trade-off:** Slightly slower training due to CPU↔GPU transfers

### Mixed Precision Training

```python
from selgis import SelgisConfig

# FP16 (faster on NVIDIA GPUs)
config = SelgisConfig(fp16=True, bf16=False)

# BF16 (better for stability, requires Ampere+)
config = SelgisConfig(fp16=False, bf16=True)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
)
```

### Gradient Accumulation

```python
from selgis import SelgisConfig

# Effective batch size = 32 * 4 = 128
config = SelgisConfig(
    batch_size=32,
    gradient_accumulation_steps=4,
)
```

---

## Complete Examples

### Example 1: Text Classification with BERT

```python
from transformers import AutoTokenizer
from selgis import (
    TransformerTrainer,
    TransformerConfig,
    create_dataloaders,
    DatasetConfig,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Configure dataset
dataset_config = DatasetConfig(
    data_type="text",
    data_path="./data/imdb.jsonl",
    tokenizer=tokenizer,
    max_length=256,
    batch_size=32,
    num_workers=4,
)

train_loader, eval_loader = create_dataloaders(dataset_config)

# Configure training
config = TransformerConfig(
    model_name_or_path="bert-base-uncased",
    num_labels=2,
    problem_type="single_label_classification",
    use_peft=True,
    peft_config={
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["query", "value"],
    },
    learning_rate=2e-5,
    max_epochs=3,
    fp16=True,
)

# Train
trainer = TransformerTrainer(
    model_or_path="bert-base-uncased",
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_pretrained("./bert-imdb")
```

### Example 2: Custom Architecture Training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from selgis import Trainer, SelgisConfig

# Define model
class ResNetLike(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=10):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, **kwargs):
        x = torch.relu(self.stem(inputs))
        x = self.block1(x) + x  # Residual connection
        return self.head(x)

# Create data
X = torch.randn(1000, 20)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Configure
config = SelgisConfig(
    max_epochs=10,
    batch_size=32,
    lr_finder_enabled=True,
    fp16=False,
)

# Train
model = ResNetLike(input_dim=20, num_classes=10)
trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    eval_dataloader=loader,
    criterion=nn.CrossEntropyLoss(),
)

metrics = trainer.train()
print(f"Final metrics: {metrics}")
```

### Example 3: LLM Fine-tuning with QLoRA

```python
from transformers import AutoTokenizer
from selgis import TransformerTrainer, TransformerConfig, create_dataloaders, DatasetConfig

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Configure for 4-bit QLoRA
config = TransformerConfig(
    model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_peft=True,
    peft_config={
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    },
    quantization_type="4bit",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    learning_rate=2e-4,
    max_epochs=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
)

# Create dataloaders
dataset_config = DatasetConfig(
    data_type="text",
    data_path="./data/alpaca.jsonl",
    tokenizer=tokenizer,
    max_length=512,
    batch_size=2,
)

train_loader, eval_loader = create_dataloaders(dataset_config)

# Define forward for causal LM
def forward_fn(model, batch):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss, outputs.logits

# Train
trainer = TransformerTrainer(
    model_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
    forward_fn=forward_fn,
)

trainer.train()
trainer.save_pretrained("./llama-finetuned")
```

---

## CLI (Command Line Interface)

Selgis includes a CLI for diagnostics and quick execution.

```bash
# Check device availability
selgis device

# Run demo training
selgis train

# Run training with config file
selgis train --config ./my_config.yaml

# Check version
selgis version
```

### CLI Output Example

```bash
$ selgis device
Device: cuda
   GPU: NVIDIA GeForce GTX 1660 Ti
   Memory: 6.00 GB
```

---

## Error Handling & Troubleshooting

### Automatic Rollback (Self-Healing)

When `nan_recovery=True` (default), Selgis monitors training for anomalies:

1. **Detection**: `[WARN] Rollback triggered: NaN/Inf loss detected`
2. **Action**: Weights restored from last stable state, optimizer momentum reset
3. **Recovery**: Learning rate permanently reduced by 50% (persists through scheduler)
4. **Resumption**: Training continues from safe point automatically

```python
from selgis import SelgisConfig

config = SelgisConfig(
    nan_recovery=True,           # Auto-rollback on NaN/Spike
    spike_threshold=3.0,         # Rollback if loss jumps 3x
    min_history_len=10,          # History length for spike detection
    final_surge_factor=5.0,      # LR boost when stuck (0 to disable)
    patience=5,                  # Epochs before early stopping
)
```

---

### Common Exceptions

| Exception | Trigger | Solution |
|-----------|---------|----------|
| `ImportError` | Missing `transformers`, `peft`, or `bitsandbytes` | `pip install selgis[all]` |
| `ValueError` | Conflicting config (e.g., `fp16=True` & `bf16=True`) | Fix configuration |
| `ValueError` | `use_peft=True` without `peft_config` | Provide `peft_config={...}` |
| `ValueError` | `quantization_type="4bit"` with `device="cpu"` | Use a GPU device |
| `ValueError` | `gradient_accumulation_steps=0` or negative | Set to >= 1 |
| `ValueError` | Both `warmup_epochs` and `warmup_ratio` > 0 | Use one or the other |
| `ZeroDivisionError` | Model has no trainable parameters | Unfreeze layers or fix LoRA |
| `FileNotFoundError` | Data path doesn't exist | Check file paths |
| `RuntimeError` | CUDA out of memory | Reduce batch size or enable `cpu_offload` |

---

### Troubleshooting Guide

#### Out of Memory (OOM)

| Solution | Code | Memory Savings |
|----------|------|----------------|
| Reduce batch size | `config = SelgisConfig(batch_size=8)` | ~20-50% |
| Gradient accumulation | `gradient_accumulation_steps=4` | ~50% |
| CPU offload | `cpu_offload=True` | ~40% |
| Device map | `device_map="auto"` | Distributes model |
| 4-bit quantization | `quantization_type="4bit"` | ~60-70% |
| Mixed precision | `fp16=True` | ~30-50% |

**Recommended combination for LLMs:**
```python
config = TransformerConfig(
    batch_size=2,
    gradient_accumulation_steps=8,  # Effective: 16
    cpu_offload=True,
    device_map="auto",
    quantization_type="4bit",
    fp16=True,
)
```

#### Slow Training

| Problem | Solution | Expected Speedup |
|---------|----------|------------------|
| Slow data loading | `num_workers=4` | 2-3x |
| Tokenization bottleneck | `pre_tokenize=True` | 5-10x |
| Large file I/O | `use_mmap=True` | 2-5x |
| Memory pressure | `state_storage="memory"` | 10-20% |
| Gradient checkpointing | Enable only if OOM | Trade-off |

```python
# Optimized for speed
config = DatasetConfig(
    num_workers=4,
    prefetch_factor=2,
    pre_tokenize=True,
    cache_dir="./cache",
)

dataset = TextDataset(data_path="./data.jsonl", use_mmap=True)
```

#### Non-converging Model

| Symptom | Solution |
|---------|----------|
| Loss oscillates | Reduce LR, increase warmup |
| Loss stuck | Use LR finder, increase LR |
| Accuracy plateaus | Final surge, check data quality |
| Spikes frequent | Increase `spike_threshold`, enable clipping |

```python
config = SelgisConfig(
    lr_finder_enabled=True,      # Find optimal LR
    warmup_ratio=0.1,            # Gradual warmup
    grad_clip_norm=1.0,          # Prevent explosions
    scheduler_type="cosine",     # Smooth decay
    nan_recovery=True,           # Auto-recovery
)
```

#### Loss Spikes

| Severity | Solution |
|----------|----------|
| Micro-spikes (1.5-2x) | Increase `min_history_len` |
| Medium spikes (2-5x) | Increase `spike_threshold` |
| Large spikes (5x+) | Reduce LR, check data quality |

```python
config = SelgisConfig(
    spike_threshold=5.0,       # Default: 3.0
    min_history_len=20,        # Default: 10
    grad_clip_norm=0.5,        # Stricter clipping
    scheduler_type="cosine",   # Smoother than cosine_restart
)
```

---

## Security & Breaking Changes

### Security Considerations

#### Checkpoint Loading

```python
# Safe loading (default) - PyTorch >= 2.0
trainer.load_model("./checkpoint.pt", weights_only=True)

# For PyTorch < 2.0 (warning issued)
trainer.load_model("./checkpoint.pt", weights_only=False)  # Only trusted files!
```

#### Remote Code Execution

```python
# Default: remote code is blocked
config = TransformerConfig(trust_remote_code=False)

# Enable only for trusted models
config = TransformerConfig(trust_remote_code=True)
```

#### HuggingFace Hub Authentication

```bash
# Login required for push_to_hub
huggingface-cli login

# Or set environment variable
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

---

### Breaking Changes

#### v0.2.3

| Change | Description | Migration |
|--------|-------------|-----------|
| **`lr_finder_enabled` default** | Changed from `True` to `False` | Add `lr_finder_enabled=True` explicitly if you relied on auto-LR |
| **`learning_rate` in SelgisConfig** | New field (default `1e-3`) | No action needed; auto-created optimizer now uses this value |
| **`primary_metric` in SelgisConfig** | New field (default `None`) | No action needed; behavior unchanged when None |
| **`trust_remote_code` in TransformerConfig** | New field (default `False`) | Add `trust_remote_code=True` if your model requires custom code |
| **`device_map` in TransformerConfig** | New field, separated from `cpu_offload` | Use `device_map="auto"` instead of relying on `cpu_offload` for model distribution |
| **`use_peft=True` validation** | Now requires non-empty `peft_config` | Always provide `peft_config={...}` when `use_peft=True` |
| **Quantization + CPU validation** | `quantization_type != "no"` with `device="cpu"` now raises error | Use a GPU device for quantization |
| **`reduce_lr` / `surge_lr` persistence** | LR adjustments now persist across scheduler `step()` calls | No action needed; this fixes the previous bug where adjustments were overwritten |
| **Optimizer state on rollback** | Optimizer state is now cleared on rollback | No action needed; improves recovery behavior |
| **`seed_everything` deterministic** | New `deterministic` parameter (default `True`); `cudnn.benchmark` no longer set in `get_device` | If you need `cudnn.benchmark=True`, use `seed_everything(42, deterministic=False)` |
| **Checkpoint format** | `scheduler.pt` now included in checkpoints | No action needed |

## License

Apache 2.0 License — Free for commercial and research use.

## Support

- **Documentation:** [API.md](API.md)
- **GitHub Issues:** For bugs and feature requests ( its empty:( )
