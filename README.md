# Selgis ML

> Universal Training Framework for PyTorch and HuggingFace Transformers.

**Selgis** (Self-Guided Intelligent Stability) is a training framework with automatic failure protection.

[![PyPI version](https://img.shields.io/pypi/v/selgis)](https://pypi.org/project/selgis/)
[![Python versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/selgis/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## What is Selgis

```
23:47 — Training started.
06:12 — Loss: NaN. Training crashed.
06:13 — You realize: 6 hours of work are gone.
```

Neural network training is fragile. Loss spikes, NaN and plateaus can destroy hours of computation. Standard trainers log the error and stop — you debug and restart manually. (OOM is a different beast: it is a capacity problem, attacked by the memory techniques below — not by automatic recovery.)

Selgis automatically:
- Detects anomalies (NaN, loss spikes)
- Rolls back to a stable state
- Lowers the learning rate
- Continues training without your intervention

---

## Installation

```bash
# Base (PyTorch only)
pip install selgis

# Full (Transformers, LoRA, quantization)
pip install "selgis[all]"

# Unsloth support (optional; vendor-claimed ~2x speed, not benchmarked here)
pip install "selgis[unsloth]"
```

---

## Quick Start

### Any PyTorch model

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from selgis import Trainer, SelgisConfig

config = SelgisConfig(max_epochs=10)

# criterion is required for models that don't compute loss internally
trainer = Trainer(model, config, train_dataloader, criterion=nn.CrossEntropyLoss())
trainer.train()
```

### LLM with LoRA

```python
from transformers import AutoTokenizer
from selgis import TransformerTrainer, TransformerConfig, DatasetConfig, create_dataloaders

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

config = TransformerConfig(
    model_name_or_path="Qwen/Qwen2-0.5B",
    use_peft=True,
    peft_config={"r": 16, "target_modules": ["q_proj", "v_proj"]},
    quantization_type="4bit",
)

data_config = DatasetConfig(
    data_type="text",
    data_path="./data.jsonl",   # jsonl with {"text": ...} or chat format
    tokenizer=tokenizer,        # without it the dataset can't produce input_ids
    batch_size=4,
    max_length=512,
)
train_loader, eval_loader = create_dataloaders(data_config)

trainer = TransformerTrainer(
    "Qwen/Qwen2-0.5B",
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    tokenizer=tokenizer,
)
trainer.train()
```

---

## Features

### 1. Self-Healing

Automatic recovery from anomalies (exact mechanics in [How It Works](#how-it-works)):

```python
config = SelgisConfig(
    nan_recovery=True,        # Auto-rollback on NaN/Inf
    spike_threshold=3.0,      # Rollback when loss exceeds 3x the mean of the last 10 losses
    min_history_len=10,       # Detection window
)
```

**What happens:**
1. Loss becomes NaN — the last stable state is restored (trainable params only)
2. Loss spikes — rollback, LR halved
3. Optimizer momentum is zeroed (not restored)
4. The offending batch is skipped, training continues

Note: state snapshots are taken every `state_update_interval=100` optimizer
steps, so a rollback can lose up to 100 steps of progress.

### 2. Memory Optimization

Techniques for large models on small GPUs (all numbers are approximate and workload-dependent):

| Technique | Typical savings | Notes |
|-----------|-----------------|-------|
| 4-bit quantization | up to ~75% | vs fp16 weights |
| CPU offload | varies | moves optimizer state to RAM |
| Gradient checkpointing | ~30–40% VRAM | ~20–30% slower |
| LoRA (trainable only) | 99%+ trainable-fraction reduction | base weights stay frozen |
| `chunked_ce=True` | forward CE peak ∝ `ce_chunk_size` | verified vs `F.cross_entropy` (see notebook below) |
| Unsloth | vendor claim: ~2x faster, ~50% less VRAM | not independently benchmarked here |

```python
config = TransformerConfig(
    quantization_type="4bit",
    cpu_offload=True,
    gradient_checkpointing=True,
    use_peft=True,
    peft_config={"r": 16},
)
```

### 2.1 Unsloth

~2x faster training with ~50% less VRAM (vendor-claimed; not benchmarked in this repo):

```python
config = TransformerConfig(
    model_name_or_path="Qwen/Qwen2-0.5B",
    use_unsloth=True,
    use_peft=True,
    peft_config={"r": 16},
)
```

Works with: Llama, Qwen, Mistral, Phi, Gemma, **Gemma 4**.

### 2.2 Chunked Cross-Entropy (`chunked_ce`)

**What it saves and what it doesn't:** the model's `lm_head` still produces
the full `[batch, seq, vocab]` logits — that cost is unavoidable without
model-level changes. What `chunked_ce` removes is everything the loss
*adds on top*: fp32 copies of logits and softmax intermediates that a
standard CE materializes for its backward pass. The chunked CE computes
the loss inside `torch.utils.checkpoint` blocks over vocabulary chunks, so
those intermediates peak at `ce_chunk_size` instead of the full tensor.

Measured on T4, GPT-2 + LoRA (seq=512): the fixed chunked CE is at parity
with (or slightly below) `F.cross_entropy` peak VRAM, while the pre-fix
implementation added **+13–16%** at batch sizes 4–16 (verified
independently on GTX 1660 Ti and Colab T4 — see CHANGELOG).

```python
config = TransformerConfig(
    problem_type="causal_lm",
    chunked_ce=True,      # replace HF's internal CE with chunked CE
    ce_chunk_size=1024,   # vocab chunk; smaller = less VRAM, slightly slower
)
```

Values and gradients match `F.cross_entropy` (label smoothing included)
to ~1e-6 / 1e-5.

### 3. Final Surge

Automatic plateau escape:

```python
config = SelgisConfig(
    patience=5,               # epochs without improvement
    final_surge_factor=5.0,   # LR boost multiplier
)
```

If the metric hasn't improved for 5 epochs, the LR is multiplied once
(never exceeding the initial LR, one surge per training run) to escape the
plateau. If it doesn't help, training stops early as usual.

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

Implements Leslie Smith's LR range test (2015) — sweeps the LR
exponentially and *suggests* a LR at the steepest loss descent
(see [How It Works](#how-it-works)).

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

## How It Works

For those who want to know exactly what happens under the hood.

### Anomaly detection & rollback

After every forward pass `SelgisCore.check_loss()` inspects the training loss:

1. **NaN / Inf** → immediate rollback.
2. **Spike** — once at least `min_history_len` (default 10) losses are recorded,
   a spike is `loss > spike_threshold × mean(last 10 losses)`.
   Default `spike_threshold=3.0`.

A rollback:

1. Restores the **last stable state** — a snapshot of *trainable parameters only*
   (taken at training start and every `state_update_interval=100` optimizer steps;
   for LoRA this is megabytes, not gigabytes).
2. **Zeroes** the optimizer's Adam moments (`exp_avg`, `exp_avg_sq`) — they are
   not restored, they are reset, so stale momentum can't re-trigger the divergence.
3. Halves the learning rate (`reduce_lr`, floored at `min_lr`).
4. Skips the offending batch entirely (no backward pass) and continues.

Honest limitations: RNG state and data-loader position are not rolled back;
the loss history window is a fixed-size moving average — a threshold based on
the last 10 losses can trigger false positives if the loss is dropping very fast.
There is currently **no limit on consecutive rollbacks**: if every batch
keeps failing (e.g., corrupted data), Selgis keeps rolling back and halving
the LR (floored at `min_lr`) instead of stopping — watch your logs.
OOM is not auto-recovered; use the memory techniques above.

### LR Finder — Smith's LR Range Test

`LRFinder` implements the LR range test from Leslie Smith,
*"Cyclical Learning Rates for Training Neural Networks"* (2015):

1. LR sweeps exponentially from `lr_finder_start` to `lr_finder_end`
   over `lr_finder_steps` batches.
2. Loss is exponentially smoothed (α = 0.05); the sweep stops early when the
   smoothed loss exceeds 4× its best value (divergence threshold).
3. The suggested LR is the **steepest descent** point of the smoothed loss
   over log₁₀(LR), stepped 10 sweep positions back as a safety margin
   (the fastai-style selection heuristic).

The sweep trains on real batches, so the model is fully restored afterwards:
parameters **and buffers** (e.g. BatchNorm running stats) are saved before
and restored after; optimizer state is restored too if
`lr_finder_save_optimizer_state=True`.

### Final surge

When the monitored metric hasn't improved by `min_delta` for `patience`
epochs, before stopping Selgis once multiplies the LR by
`final_surge_factor` (capped at the initial LR) and resets the patience
counter. If the plateau persists afterwards, training stops — the surge
is a one-shot escape attempt, never repeated.

### Schedulers

`SmartScheduler` supports cosine, cosine with warm restarts (SGDR-style
`t_0`/`t_mult`), linear, polynomial and constant decay, all with a `min_lr`
floor. With `warmup_epochs` the schedule is epoch-based; with `warmup_ratio`
it switches to per-optimizer-step scheduling (warmup ramp + decay per step).

### Chunked cross-entropy

With `chunked_ce=True`, the CE over the logits is computed as a two-pass
log-sum-exp over vocabulary chunks (`ce_chunk_size`, default 1024). The
model's `lm_head` still materializes the full `[batch, seq, vocab]` logits —
what chunking removes is the loss's own overhead (fp32 copies and softmax
intermediates that a standard CE keeps for backward). Each chunk runs
inside a `torch.utils.checkpoint` block: nothing chunk-sized is kept for
backward — it is recomputed chunk by chunk during the backward pass, so
the loss's peak memory scales with the chunk, not with the vocabulary.

### Checkpointing

`CheckpointCallback` saves `model.pt` (trainable parameters only when PEFT
is active), `optimizer.pt`, `scheduler.pt` and `metrics.json` per checkpoint,
with rotation down to `save_total_limit` regular checkpoints. The
`best_model/` directory is exempt from rotation.

---

## CLI

```bash
# Demo mode
selgis train

# From config
selgis train --config config.yaml

# Override config fields from the command line
selgis train --config config.yaml --set max_epochs=10 batch_size=64

# Check device
selgis device

# Validate a config file
selgis config validate --config config.yaml

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

# LoRA (dataloaders required — see Quick Start above)
from selgis import TransformerTrainer, TransformerConfig
config = TransformerConfig(model_name_or_path="Qwen/Qwen2-0.5B", use_peft=True)
trainer = TransformerTrainer(
    "Qwen/Qwen2-0.5B", config, train_dataloader=train_loader, eval_dataloader=eval_loader
)
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
# Base (installed automatically)
torch>=2.0, numpy>=1.20, pyyaml>=6.0

# Optional extras
selgis[transformers]  # transformers>=4.30, datasets, accelerate
selgis[peft]          # peft>=0.5.0 — LoRA
selgis[llm]           # bitsandbytes, accelerate, scipy — quantization
selgis[tracking]      # wandb
selgis[all]           # everything above
```

---

## Limitations

- **Single-GPU only** — DDP / multi-GPU / FSDP are not supported yet (see plans below)
- **DeepSpeed** — partial support today; full ZeRO/pipeline integration is planned for v0.3.0
- **OOM** is not auto-recovered — see the memory techniques above
- Repeated rollbacks are not capped (see [How It Works](#how-it-works))

## When NOT to use Selgis

| You need | Better fit |
|----------|------------|
| Multi-GPU / distributed training | HF `accelerate`, PyTorch DDP/FSDP, DeepSpeed |
| Battle-tested LLM trainer with the widest ecosystem | HF `Trainer`, Lightning |
| Full control over every step of the loop | a plain PyTorch training loop |

Selgis adds value on **single-GPU training that must survive unattended**:
long fine-tunes and experiments where a 3 AM NaN shouldn't cost you a night.
The self-healing path is ~20 lines of extra logic per step (loss check +
snapshot bookkeeping); if your runs are short and supervised, a plain
loop or HF Trainer with `gradient_checkpointing` + manual checkpoints is
a perfectly good choice.

---

## Future Plans

- [x] **Unsloth integration** — DONE (v0.2.6)
  - 2x faster training, 50% less VRAM (vendor-claimed)
  - Llama, Qwen, Mistral, Phi, Gemma, Gemma 4 support
  - Model loading from a local path or the HuggingFace Hub

- [ ] **Rollback loop guard** — cap consecutive rollbacks (`max_consecutive_rollbacks`) and stop with a clear error instead of halving LR forever

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
- [Tests](https://github.com/Selgm-ofic/Selgis-ML/tree/main/tests) — including regression tests for every bug fixed in 0.2.7.2 (green CI badge = all passing)
- [Examples](https://github.com/Selgm-ofic/Selgis-ML/blob/main/example_selgis.py)

---

License: Apache 2.0