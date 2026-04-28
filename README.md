# Selgis ML

> Universal Training Framework for PyTorch and HuggingFace Transformers.

**Selgis** (Self-Guided Intelligent Stability) is a training framework with automatic failure protection.

[![PyPI version](https://img.shields.io/pypi/v/selgis)](https://pypi.org/project/selgis/)
[![Python versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/selgis/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## What is Selgis

```
03:47 — Training started.
07:00 — Loss: NaN. Training crashed.
07:01 — You realize: 8 hours of work are gone.
```

Neural network training is fragile. Loss spikes, NaN, OOM and plateaus can destroy hours of computation. Standard trainers log the error and stop — you debug and restart manually.

Selgis automatically:
- Detects anomalies (NaN, spikes)
- Rolls back to stable state
- Lowers learning rate
- Continues training without your intervention

---

## Installation

```bash
# Base (PyTorch only)
pip install selgis

# Full (Transformers, LoRA, quantization)
pip install "selgis[all]"

# Unsloth support (recommended for LLM training)
pip install unsloth
```

---

## Quick Start

### Any PyTorch model

```python
from selgis import Trainer, SelgisConfig
from torch.utils.data import DataLoader

config = SelgisConfig(max_epochs=10)
trainer = Trainer(model, config, train_dataloader)
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

## Features

### 1. Self-Healing

Automatic recovery from anomalies:

```python
config = SelgisConfig(
    nan_recovery=True,        # Auto-rollback on NaN/Inf
    spike_threshold=3.0,     # Rollback when loss 3x spike
    min_history_len=10,      # Detection window
)
```

**What happens:**
1. Loss becomes NaN — loads last stable state
2. Loss spikes sharply — rollback + LR reduced 50%
3. Optimizer momentum cleared
4. Training continues

### 2. Memory Optimization

Techniques for large models on small GPUs:

| Technique | Savings |
|-----------|---------|
| 4-bit quantization | 75% |
| CPU offload | 40% |
| Gradient checkpointing | 40% |
| LoRA (trainable only) | 99.9% |
| **Unsloth** | **50% less VRAM, 2x faster** |

```python
config = TransformerConfig(
    quantization_type="4bit",
    cpu_offload=True,
    gradient_checkpointing=True,
    use_peft=True,
    peft_config={"r": 16},
)
```

### 2.1 Unsloth (NEW)

~2x faster training with ~50% less VRAM:

```python
config = TransformerConfig(
    model_name_or_path="Qwen/Qwen2-0.5B",
    use_unsloth=True,
    use_peft=True,
    peft_config={"r": 16},
)
```

Works with: Llama, Qwen, Mistral, Phi, Gemma, **Gemma 4**.

### 3. Final Surge

Automatic plateau escape:

```python
config = SelgisConfig(
    patience=5,               # epochs without improvement
    final_surge_factor=5.0,   # LR boost multiplier
)
```

If 5 epochs no improvement — LR multiplies to escape local minima.

### 4. LR Finder

Automatic learning rate search:

```python
config = SelgisConfig(
    lr_finder_enabled=True,
    lr_finder_steps=100,
    lr_finder_start=1e-7,
    lr_finder_end=1.0,
)
```

Leslie Smith style — finds optimal LR in 100 steps.

### 5. Schedulers

Built-in schedulers:

```python
config = SelgisConfig(
    scheduler_type="cosine_restart",  # cosine, linear, polynomial, constant
    warmup_ratio=0.1,
    min_lr=1e-7,
    t_0=10,
    t_mult=2,
)
```

### 6. Mixed Precision

```python
config = SelgisConfig(
    fp16=True,   # FP16 mixed precision
    # bf16=True, # or BF16 for Ampere+
)
```

### 7. Gradient Management

```python
config = SelgisConfig(
    grad_clip_norm=1.0,
    # grad_clip_value=0.5,
    gradient_accumulation_steps=4,
)
```

### 8. Checkpointing

```python
config = SelgisConfig(
    output_dir="./output",
    save_best_only=True,
    save_total_limit=3,
    state_storage="disk",     # or "memory"
)
```

### 9. Callbacks

Extend functionality:

```python
from selgis import (
    LoggingCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    HistoryCallback,
    WandBCallback,
    SparsityCallback,
)

callbacks = [
    LoggingCallback(log_every=10),
    CheckpointCallback(output_dir="./checkpoints"),
    EarlyStoppingCallback(patience=5, metric="accuracy", mode="max"),
    WandBCallback(project="my-project"),
]
```

### 10. Datasets

Unified data API:

```python
from selgis import create_dataloaders, DatasetConfig

# Text (JSONL) - auto-detects format by extension
config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",  # .jsonl, .json, .csv, .txt
    max_length=512,
)

# Chat datasets - auto-detects alpaca/sharegpt/messages
config = DatasetConfig(
    data_type="text",
    data_path="./alpaca_data.jsonl",  # auto-detects: alpaca, sharegpt, messages
)
# or manually:
config = DatasetConfig(
    data_type="text",
    data_path="./chat.jsonl",
    chat_format="messages",
    user_role="user",      # custom role (default)
    assistant_role="assistant",
)

# HuggingFace datasets
config = DatasetConfig(
    data_type="text",
    data_path="tatsu-lab/alpaca",  # auto-downloads from HF
)

# Image
config = DatasetConfig(
    data_type="image",
    data_path="./images",
)

# Streaming (large files)
config = DatasetConfig(
    data_type="streaming",
    data_path="./large.jsonl",
    buffer_size=1000,
)

train_loader, eval_loader = create_dataloaders(config)
```

---

## CLI

```bash
# Demo mode
selgis train

# From config
selgis train --config config.yaml

# Check device
selgis device

# Run tests
selgis test
```

---

## Configuration

| Parameter | Default | Description |
|------------|---------|-------------|
| `max_epochs` | 100 | Max epochs |
| `learning_rate` | 1e-3 | Base LR |
| `batch_size` | 32 | Batch size |
| `nan_recovery` | True | Auto-rollback |
| `spike_threshold` | 3.0 | Spike detection |
| `grad_clip_norm` | 1.0 | Gradient clip |
| `save_best_only` | True | Save best only |
| `cpu_offload` | False | CPU optimizer |
| `final_surge_factor` | 5.0 | LR boost on plateau |

---

## Examples

Full examples: [example_selgis.py](https://github.com/Selgm-ofic/Selgis-ML/blob/main/example_selgis.py)

```python
# Basic
from selgis import Trainer, SelgisConfig
config = SelgisConfig(max_epochs=10)
trainer = Trainer(model, config, loader)
trainer.train()

# LoRA
from selgis import TransformerTrainer, TransformerConfig
config = TransformerConfig(model_name_or_path="Qwen/Qwen2-0.5B", use_peft=True)
trainer = TransformerTrainer("Qwen/Qwen2-0.5B", config)
trainer.train()

# Callbacks
from selgis import LoggingCallback, CheckpointCallback
trainer = Trainer(model, config, loader, callbacks=[
    LoggingCallback(log_every=10),
    CheckpointCallback(output_dir="./ckpt"),
])
```

---

## Dependencies

```toml
# Base
torch>=2.0, numpy>=1.20, tqdm

# Optional
transformers>=4.30, datasets, accelerate>=0.21.0
peft>=0.5.0
bitsandbytes>=0.41.0
wandb
pytest
```

---

## Limitations

- **DeepSpeed** — partial support (v0.3.0)
- **FSDP** — in development

---

## Future Plans

- [x] **Unsloth integration** — DONE (v0.2.6)
  - 2x faster training, 50% less VRAM
  - Llama, Qwen, Mistral, Phi, Gemma, Gemma 4 support
  - Run locally or from HuggingFace

- [ ] **DeepSpeed full** — complete ZeRO, pipeline

- [ ] **FSDP** — Fully Sharded Data Parallel

- [ ] **Distributed Training** — DDP, multi-GPU

- [ ] **More schedulers** — OneCycle, ReduceLROnPlateau

- [ ] **MLflow integration** — W&B alternative

---

## Links

- [PyPI](https://pypi.org/project/selgis/)
- [GitHub](https://github.com/Selgm-ofic/Selgis-ML)
- [API Docs](https://github.com/Selgm-ofic/Selgis-ML/blob/main/API.md)
- [Tests](https://github.com/Selgm-ofic/Selgis-ML/blob/main/test_selgis.py)
- [Examples](https://github.com/Selgm-ofic/Selgis-ML/blob/main/example_selgis.py)

---

License: Apache 2.0