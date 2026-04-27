# Selgis ML API Documentation

**Version:** 0.2.5
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
11. [Error Handling](#error-handling)
12. [License](#license)

---

## Installation

```bash
# Base (PyTorch only)
pip install selgis

# Full (Transformers, LoRA, quantization)
pip install "selgis[all]"
```

---

## Quick Start

### Any PyTorch Model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from selgis import Trainer, SelgisConfig

# Create model
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

# Configure
config = SelgisConfig(
    max_epochs=10,
    learning_rate=1e-3,
    nan_recovery=True,
)

# Train
trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    criterion=nn.CrossEntropyLoss(),
)

trainer.train()
```

### LLM with LoRA

```python
from selgis import TransformerTrainer, TransformerConfig

config = TransformerConfig(
    model_name_or_path="Qwen/Qwen2-0.5B",
    use_peft=True,
    peft_config={"r": 16, "target_modules": ["q_proj", "v_proj"]},
    quantization_type="4bit",
)

trainer = TransformerTrainer("Qwen/Qwen2-0.5B", config=config)
trainer.train()
```

---

## Configuration

### SelgisConfig

Base training configuration.

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
    lr_finder_save_optimizer_state=False,
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

    # Mixed Precision
    fp16=False,
    bf16=False,

    # Logging
    logging_steps=10,

    # Checkpointing
    output_dir="./output",
    save_total_limit=3,
    save_best_only=True,
    state_storage="disk",
    state_update_interval=100,
    resume_from_checkpoint=None,

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
| `max_epochs` | int | 100 | Maximum epochs |
| `learning_rate` | float | 1e-3 | Initial learning rate |
| `nan_recovery` | bool | True | Auto-rollback on NaN/Inf |
| `spike_threshold` | float | 3.0 | Spike detection multiplier |
| `grad_clip_norm` | float | 1.0 | Gradient norm clip |
| `cpu_offload` | bool | False | Offload optimizer to CPU |
| `fp16` | bool | False | FP16 mixed precision |
| `bf16` | bool | False | BF16 mixed precision |
| `save_best_only` | bool | True | Save only best checkpoint |
| `final_surge_factor` | float | 5.0 | LR boost on plateau (0 to disable) |
| `scheduler_type` | str | "cosine_restart" | cosine, cosine_restart, linear, polynomial, constant |

### TransformerConfig

Extended configuration for HuggingFace Transformers.

```python
from selgis import TransformerConfig

config = TransformerConfig(
    # Model
    model_name_or_path="",
    num_labels=2,
    problem_type="single_label_classification",
    trust_remote_code=False,
    device_map=None,

    # Tokenizer
    max_length=512,
    padding="max_length",

    # Optimizer
    optimizer_type="adamw",
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,

    # LoRA / PEFT
    use_peft=False,
    peft_config={},
    adapter_name_or_path=None,

    # Gradient Checkpointing
    gradient_checkpointing=False,

    # Chunked Cross-Entropy
    chunked_ce=False,
    ce_chunk_size=1024,

    # Flash Attention
    flash_attention=False,

    # Quantization
    quantization_type="no",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | "" | Model path or HF hub name |
| `use_peft` | bool | False | Enable LoRA/PEFT |
| `peft_config` | dict | {} | LoRA config: r, lora_alpha, target_modules |
| `quantization_type` | str | "no" | no, 8bit, 4bit |
| `gradient_checkpointing` | bool | False | Enable gradient checkpointing |
| `device_map` | str | None | auto, balanced, sequential |

---

## Datasets

### DatasetConfig

```python
from selgis import DatasetConfig, create_dataloaders

config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",
    train_path=None,
    eval_path=None,
    batch_size=32,
    num_workers=0,
    max_length=512,
    tokenizer=None,
    image_processor=None,
    transform=None,
    cache_dir=None,
    pre_tokenize=False,
    streaming=False,
    buffer_size=1000,
    train_split=0.9,
    seed=42,
)

train_loader, eval_loader = create_dataloaders(config)
```

### Available Dataset Types

- `text` - Text data (JSONL, TXT, CSV)
- `image` - Image classification
- `multimodal` - Text + image
- `streaming` - Large file streaming
- `custom` - Wrap any PyTorch Dataset
- `tabular` - CSV/JSON tabular

### create_dataset

```python
from selgis import create_dataset, DatasetConfig

config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",
)

dataset = create_dataset(config)
```

---

## Training

### Trainer

Universal trainer for PyTorch models.

```python
from selgis import Trainer, SelgisConfig

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=None,
    callbacks=None,
    forward_fn=None,
    compute_metrics=None,
)

metrics = trainer.train()
trainer.save_model("./model.pt")
trainer.load_model("./model.pt")
```

### TransformerTrainer

Trainer for HuggingFace Transformers.

```python
from selgis import TransformerTrainer, TransformerConfig

trainer = TransformerTrainer(
    model_or_path="Qwen/Qwen2-0.5B",
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
    forward_fn=None,
)

metrics = trainer.train()
trainer.save_pretrained("./output")
trainer.push_to_hub("username/model")
```

### Custom Forward Function

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
    forward_fn=forward_fn,
)
```

### Custom Metrics

```python
def compute_metrics(preds, labels):
    preds = preds.argmax(dim=-1)
    accuracy = (preds == labels).float().mean()
    return {"accuracy": accuracy * 100}

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)
```

---

## Callbacks

### Callback (Base Class)

```python
from selgis import Callback

class MyCallback(Callback):
    def on_train_begin(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, metrics): pass
    def on_step_begin(self, trainer, step): pass
    def on_step_end(self, trainer, step, loss): pass
    def on_evaluate(self, trainer, metrics): pass
    def on_train_end(self, trainer): pass
```

### LoggingCallback

```python
from selgis import LoggingCallback

callback = LoggingCallback(log_every=10)
```

### CheckpointCallback

```python
from selgis import CheckpointCallback

callback = CheckpointCallback(
    output_dir="./checkpoints",
    save_best_only=True,
    save_total_limit=3,
)
```

### EarlyStoppingCallback

```python
from selgis import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    patience=5,
    metric="accuracy",
    mode="max",
)
```

### HistoryCallback

```python
from selgis import HistoryCallback

callback = HistoryCallback(output_dir="./output")
history = callback.history
```

### WandBCallback

```python
from selgis import WandBCallback

callback = WandBCallback(
    project="my-project",
    name="experiment",
)
```

### SparsityCallback

```python
from selgis import SparsityCallback

callback = SparsityCallback(
    target_sparsity=0.5,
    start_epoch=5,
    frequency=1,
)
```

---

## Utilities

### get_device

```python
from selgis import get_device

device = get_device("auto")  # cuda > mps > cpu
```

### seed_everything

```python
from selgis import seed_everything

seed_everything(42)                    # Full reproducibility
seed_everything(42, deterministic=False)  # Faster but non-deterministic
```

### count_parameters / format_params

```python
from selgis import count_parameters, format_params

trainable = count_parameters(model, trainable_only=True)
print(format_params(trainable))  # "1.20M"
```

### move_to_device / unpack_batch

```python
from selgis import move_to_device, unpack_batch

batch = move_to_device(batch, device)
inputs, labels = unpack_batch(batch)
```

### get_optimizer_grouped_params

```python
from selgis import get_optimizer_grouped_params

param_groups = get_optimizer_grouped_params(model, weight_decay=0.01)
optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
```

---

## Advanced Features

### SelgisCore

Low-level training protection.

```python
from selgis import SelgisCore, SelgisConfig, SmartScheduler

core = SelgisCore(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    device=device,
)

# Training loop
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch)

    if not core.check_loss(loss):  # Returns False if rollback triggered
        continue

    core.backward_step(loss)
    core.optimizer_step()
```

### LRFinder

```python
from selgis import LRFinder

lr_finder = LRFinder(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    device=device,
)

optimal_lr = lr_finder.find(
    train_loader,
    start_lr=1e-7,
    end_lr=1.0,
    num_steps=100,
)
```

### SmartScheduler

```python
from selgis import SmartScheduler

scheduler = SmartScheduler(
    optimizer=optimizer,
    initial_lr=1e-3,
    config=config,
)

# Manual adjustments
scheduler.reduce_lr(factor=0.5)  # Reduce by 50%
scheduler.surge_lr(factor=3.0)    # Increase by 3x
```

### ChunkedCrossEntropyLoss

Memory-efficient cross-entropy for large vocabularies.

```python
from selgis import ChunkedCrossEntropyLoss

loss_fn = ChunkedCrossEntropyLoss(
    chunk_size=1024,
    label_smoothing=0.1,
)
```

### CrossEntropyLossV2

Drop-in replacement for nn.CrossEntropyLoss with optional chunking.

```python
from selgis import CrossEntropyLossV2

loss_fn = CrossEntropyLossV2(
    chunk_size=1024,
    label_smoothing=0.1,
)
```

### GradientCheckpointingManager

Granular gradient checkpointing for transformer layers.

```python
from selgis.checkpointing import GradientCheckpointingManager

manager = GradientCheckpointingManager(
    checkpoint_interval=1,  # 1 = every layer, 2 = every other
    use_reentrant=False,
)
model = manager.apply_to_model(model)
```

### get_transformer_scheduler

HuggingFace-compatible scheduler.

```python
from selgis import get_transformer_scheduler

scheduler = get_transformer_scheduler(
    optimizer=optimizer,
    scheduler_type="cosine",
    num_warmup_steps=100,
    num_training_steps=10000,
)

### GradientCheckpointingManager

```python
from selgis.checkpointing import GradientCheckpointingManager

manager = GradientCheckpointingManager(
    checkpoint_interval=1,  # 1 = every layer, 2 = every 2nd layer
    use_reentrant=False,
)
model = manager.apply_to_model(model)
```

---

## Complete Examples

### Example 1: Basic Classification

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from selgis import Trainer, SelgisConfig

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

config = SelgisConfig(
    max_epochs=10,
    learning_rate=1e-3,
    nan_recovery=True,
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=loader,
    criterion=nn.CrossEntropyLoss(),
)

trainer.train()
```

### Example 2: LLM with LoRA

```python
from selgis import TransformerTrainer, TransformerConfig

config = TransformerConfig(
    model_name_or_path="Qwen/Qwen2-0.5B",
    use_peft=True,
    peft_config={
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    },
    quantization_type="4bit",
    gradient_checkpointing=True,
    max_epochs=3,
)

trainer = TransformerTrainer("Qwen/Qwen2-0.5B", config=config)
trainer.train()
trainer.save_pretrained("./model")
```

### Example 3: Custom Metrics

```python
from selgis import Trainer, SelgisConfig

def compute_metrics(preds, labels):
    preds = preds.argmax(dim=-1)
    return {
        "accuracy": (preds == labels).float().mean().item() * 100,
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)
```

---

## CLI

```bash
# Demo training
selgis train

# From config
selgis train --config config.yaml

# Check device
selgis device

# Run tests
selgis test
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ImportError` | Missing deps | `pip install selgis[all]` |
| `ValueError` | Config conflict | Check mutually exclusive options |
| `ValueError` | use_peft without config | Provide peft_config |
| `ValueError` | Quant + CPU | Use GPU for quantization |
| `RuntimeError` | OOM | Reduce batch_size, enable cpu_offload |

---

## License

Apache 2.0 License