
# SELGIS API Reference

**Version:** 0.2.1 (Stable)
**Python:** 3.10+

SELGIS is a universal training framework for PyTorch and HuggingFace Transformers. It provides training protection (NaN/loss spike recovery, rollback), learning-rate scheduling, LR finder, callbacks, and specialized support for quantization (4-bit/8-bit) and PEFT/LoRA.

---

## OpenAPI 3.0 Specification (CLI)

SELGIS exposes its main functionality via a CLI that follows a structured schema. Below is a conceptual OpenAPI representation of the training endpoint.

```yaml
openapi: 3.0.0
info:
  title: SELGIS CLI Training API
  version: 0.2.1
  description: API for managing ML training runs via CLI.
paths:
  /train:
    post:
      summary: Start a training run
      description: Initiates a training process with a specified configuration.
      parameters:
        - name: config
          in: query
          required: false
          schema:
            type: string
          description: Path to a YAML or JSON configuration file.
      responses:
        '200':
          description: Training completed successfully.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TrainingMetrics'
        '400':
          description: Invalid configuration parameters.
        '500':
          description: Training failed due to hardware or logic error.

components:
  schemas:
    TrainingMetrics:
      type: object
      properties:
        train_loss:
          type: number
        eval_loss:
          type: number
        accuracy:
          type: number
        epoch:
          type: integer
```

---

## Authentication & Security

### HuggingFace Hub
When using `TransformerTrainer.push_to_hub()`, SELGIS uses the standard HuggingFace authentication token.

**Setup:**
```bash
huggingface-cli login
```
Alternatively, set the `HUGGING_FACE_HUB_TOKEN` environment variable.

### Weight Security
SELGIS prioritizes security when loading external checkpoints.
- **`weights_only=True`**: (Default in `load_model`) Prevents arbitrary code execution by loading only tensors.
- **Breaking Change (v0.2.0)**: In PyTorch < 2.0, where `weights_only` is not supported, a `[WARN]` is issued, and a standard load is performed. Only use trusted checkpoints in legacy environments.

---

## Multi-Language Examples

### Python (Standard API)
```python
from selgis import TransformerConfig, TransformerTrainer

config = TransformerConfig(
    model_name_or_path="facebook/opt-125m",
    quantization_type="4bit",
    use_peft=True
)
trainer = TransformerTrainer(model_or_path=config.model_name_or_path, config=config, ...)
trainer.train()
```

### Bash (CLI)
```bash
# Basic training
selgis train

# Training with custom config
selgis train --config ./my_config.yaml

# Check device availability
selgis device
```

### Node.js (via Child Process)
```javascript
const { exec } = require('child_process');

exec('selgis train --config custom.json', (error, stdout, stderr) => {
  if (error) {
    console.error(`Error: ${error.message}`);
    return;
  }
  console.log(`Output: ${stdout}`);
});
```

---

## Table of Contents

1. [Installation & CLI](#installation--cli)
2. [Package overview](#package-overview)
3. [Config](#config)
4. [Core](#core)
5. [Trainers](#trainers)
6. [Callbacks](#callbacks)
7. [Scheduler](#scheduler)
8. [LR Finder](#lr-finder)
9. [Utils](#utils)

---

## Installation & CLI

Install in development mode:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[transformers,peft]"
```

**CLI** (entry point `selgis`):

| Command | Description |
|--------|-------------|
| `selgis --version` | Print package version |
| `selgis version` | Same as `--version` |
| `selgis device` | Print compute device (CUDA/MPS/CPU) and GPU info |
| `selgis train` | Run minimal synthetic demo training |
| `selgis train --config PATH` | Run training with configuration from YAML/JSON file |

Example:

```bash
selgis device
selgis train
```

---

## Package overview

**Public API** (import from `selgis`):

```python
from selgis import (
    __version__,
    SelgisConfig,
    TransformerConfig,
    SelgisCore,
    LRFinder,
    SmartScheduler,
    get_transformer_scheduler,
    Trainer,
    TransformerTrainer,
    Callback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback,
    WandBCallback,
    SparsityCallback,
    get_device,
    seed_everything,
    count_parameters,
    format_params,
    move_to_device,
    unpack_batch,
    get_optimizer_grouped_params,
    is_dict_like,
    to_dict,
)
```

**Note:** When using models loaded with `device_map="auto"` (e.g. large LLMs), the trainer does not move the model to a single device; each part stays on its mapped device. If no `eval_dataloader` is provided, `evaluate()` returns an empty dict and primary metric for early stopping defaults to `"loss"` with `higher_is_better=False`. Early stopping: SelgisCore uses `config.patience` and primary metric; if you add `EarlyStoppingCallback`, it runs in addition (callback's `should_stop` is checked after SelgisCore).

**Scheduler and warmup:** Use either `warmup_epochs` (epoch-based) or `warmup_ratio` (step-based), not both. When `warmup_ratio` > 0, the trainer calls `scheduler.step()` on every optimizer step; when `warmup_ratio` is 0, it calls `scheduler.step_epoch(epoch)` at the end of each epoch.

**Safe checkpoint loading:** For untrusted checkpoint files, use `trainer.load_model(path, weights_only=True)` (default). This loads only tensors and avoids arbitrary code execution. PyTorch 2.0+ supports `weights_only=True`; older versions fall back to full load.

---

## Config

### `SelgisConfig`

Base training configuration (dataclass).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | 32 | Training batch size |
| `eval_batch_size` | int | 64 | Evaluation batch size |
| `max_epochs` | int | 100 | Maximum epochs |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `patience` | int | 5 | Early stopping patience (epochs without improvement) |
| `min_delta` | float | 1e-4 | Minimum improvement to count as better |
| `grad_clip_norm` | float | 1.0 | Max gradient L2 norm (0 = no clip) |
| `grad_clip_value` | float \| None | None | Optional per-parameter clip value |
| `spike_threshold` | float | 3.0 | Loss spike = value > threshold × recent average |
| `min_history_len` | int | 10 | Min loss history length before spike check |
| `nan_recovery` | bool | True | Enable NaN/Inf and spike rollback |
| `lr_finder_enabled` | bool | True | Run LR finder before training |
| `lr_finder_trainable_only` | bool | False | LR finder: clone/restore only trainable params (saves memory for LoRA/LLM) |
| `lr_finder_start` | float | 1e-7 | LR finder start LR |
| `lr_finder_end` | float | 1.0 | LR finder end LR |
| `lr_finder_steps` | int | 100 | LR finder steps |
| `warmup_epochs` | int | 0 | Warmup epochs (use with step_epoch scheduler) |
| `warmup_ratio` | float | 0.0 | Warmup as fraction of total steps |
| `min_lr` | float | 1e-7 | Minimum LR for schedulers |
| `scheduler_type` | str | `"cosine_restart"` | One of: cosine, cosine_restart, linear, constant, polynomial |
| `t_0` | int | 10 | First restart period (cosine_restart) |
| `t_mult` | int | 2 | Period multiplier after each restart |
| `label_smoothing` | float | 0.1 | Label smoothing (if used in criterion) |
| `weight_decay` | float | 0.01 | Weight decay for optimizer |
| `sparsity_enabled` | bool | False | Enable SparsityCallback from config |
| `sparsity_target` | float | 0.0 | Target sparsity (0–1) |
| `sparsity_start_epoch` | int | 0 | Epoch to start pruning |
| `sparsity_frequency` | int | 1 | Prune every N epochs |
| `fp16` | bool | False | Use FP16 mixed precision (CUDA) |
| `bf16` | bool | False | Use BF16 mixed precision |
| `logging_steps` | int | 10 | Log every N steps |
| `eval_steps` | int \| None | None | Eval every N steps (if used) |
| `save_steps` | int \| None | None | Save every N steps (if used) |
| `output_dir` | str | `"./output"` | Default output directory |
| `save_total_limit` | int | 3 | Max checkpoints to keep |
| `save_best_only` | bool | True | Save only best checkpoint |
| `state_storage` | `"disk" \| "memory"` | `"disk"` | Where to store rollback/best state |
| `state_dir` | str \| None | None | Override directory for state (default: output_dir/selgis_state) |
| `state_update_interval` | int | 100 | Steps between saving "last good" state |
| `device` | str | `"auto"` | Device: auto, cuda, cpu, mps |
| `cpu_offload` | bool | `False` | Offload optimizer states and gradients to CPU (saves VRAM) |
| `seed` | int | 42 | Random seed |

**Validation:** `fp16` and `bf16` cannot both be True. Either `warmup_epochs` or `warmup_ratio` can be non-zero, not both.

---

### `TransformerConfig`

Extends `SelgisConfig` for HuggingFace models.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name_or_path` | str | `""` | Pretrained model name or path |
| `num_labels` | int | 2 | Number of labels (classification) |
| `problem_type` | str | `"single_label_classification"` | single_label_classification, multi_label_classification, regression, seq2seq, causal_lm, masked_lm |
| `max_length` | int | 512 | Max sequence length |
| `padding` | str | `"max_length"` | max_length, longest, do_not_pad |
| `truncation` | bool | True | Truncate to max_length |
| `optimizer_type` | str | `"adamw"` | adamw, adam, sgd, adafactor |
| `learning_rate` | float | 2e-5 | Learning rate |
| `adam_beta1` | float | 0.9 | Adam/AdamW beta1 |
| `adam_beta2` | float | 0.999 | Adam/AdamW beta2 |
| `adam_epsilon` | float | 1e-8 | Adam/AdamW epsilon |
| `use_peft` | bool | False | Apply PEFT (e.g. LoRA) |
| `peft_config` | dict | {} | PEFT config (e.g. LoraConfig kwargs) |
| `gradient_checkpointing` | bool | False | Enable gradient checkpointing |
| `deepspeed_config` | str \| None | None | Path to DeepSpeed config |
| `quantization_type` | `"no" \| "8bit" \| "4bit"` | `"no"` | BitsAndBytes quantization strategy |
| `bnb_4bit_compute_dtype` | `"float16" \| "bfloat16" \| "float32"` | `"float16"` | Compute dtype for 4-bit |
| `bnb_4bit_quant_type` | `"fp4" \| "nf4"` | `"nf4"` | 4-bit quantization type |
| `bnb_4bit_use_double_quant` | bool | False | Use nested quantization for 4-bit |

---

## Core

### `SelgisCore(model, optimizer, scheduler, config, device)`

Training protection and optimization: NaN/Inf and loss-spike detection, rollback to last good state, early stopping with optional "final surge", gradient clipping, mixed precision, memory-efficient state handling (trainable parameters only; important for LoRA/PEFT), and **CPU Offload** for optimizer states and gradients.

**CPU Offload:** When `config.cpu_offload=True`, the optimizer states (e.g., Adam's exp_avg, exp_avg_sq) and gradients are offloaded to CPU after each step, reducing GPU VRAM usage at the cost of slight speed overhead. This is especially useful for large models on consumer GPUs.

**Methods:**

- **`check_loss(loss: Tensor) -> bool`**  
  Check loss for NaN/Inf or spike. Returns `True` if OK, `False` if rollback was performed.

- **`backward_step(loss, retain_graph=False)`**  
  Backward pass (with optional AMP). Clipping is done in `optimizer_step`.

- **`optimizer_step()`**  
  Gradient clipping (norm and optionally value), then optimizer step; periodically saves “last good” state.

- **`eval_epoch(metrics, epoch, primary_metric="accuracy", higher_is_better=True) -> Literal["IMPROVED","SURGE","STOP","CONTINUE"]`**  
  Update scheduler, compare primary metric and loss to best; save best state if improved. Returns `IMPROVED`, `SURGE` (one final LR surge), `STOP`, or `CONTINUE`.

- **`load_best_weights() -> bool`**  
  Load best saved weights. Returns `True` if loaded.

- **`get_amp_context()`**  
  Returns `torch.amp.autocast(...)` context if fp16/bf16, else `nullcontext()`.

**Properties:** `best_metric`, `best_loss`, `trainable_params_count`, `total_params_count`.

---

## Trainers

### `Trainer(model, config, train_dataloader, eval_dataloader=None, criterion=None, optimizer=None, callbacks=None, forward_fn=None, compute_metrics=None)`

Generic trainer for any PyTorch model.

- **model:** `nn.Module` to train.
- **config:** `SelgisConfig`.
- **train_dataloader / eval_dataloader:** `DataLoader` instances.
- **criterion:** Loss module (e.g. `CrossEntropyLoss`). Optional if model returns `loss` (e.g. HuggingFace).
- **optimizer:** Optional; created by default (AdamW, lr from LR finder or 1e-3).
- **callbacks:** List of `Callback`; default `[LoggingCallback()]`. `SparsityCallback` is added automatically if `config.sparsity_enabled` is True.
- **forward_fn:** Optional `(model, batch) -> (loss, logits)`. If None, forward is inferred from batch (dict vs tensor) and criterion.
- **compute_metrics:** Optional `(preds, labels) -> dict[str, float]`. If None, accuracy is computed when possible.

**Methods:**

- **`train() -> dict[str, Any]`**  
  Run full training loop; returns final metrics.

- **`evaluate() -> dict[str, float]`**  
  Run evaluation; returns metrics (e.g. loss, accuracy).

- **`save_model(path)`**  
  Save `model.state_dict()` to path.

- **`load_model(path)`**  
  Load `state_dict` from path.

**Attributes:** `model`, `config`, `device`, `optimizer`, `selgis` (SelgisCore), `callbacks`, `train_dataloader`, `eval_dataloader`.

---

### `TransformerTrainer(model_or_path, config, train_dataloader, eval_dataloader=None, tokenizer=None, **kwargs)`

Trainer for HuggingFace Transformers: loads model from path if string, applies PEFT from config, gradient checkpointing, and grouped params for weight decay.

- **model_or_path:** Model instance or pretrained model name/path.
- **config:** `TransformerConfig`.
- **tokenizer:** Optional; used for `save_pretrained` / `push_to_hub`.

**Methods (in addition to Trainer):**

- **`save_pretrained(path)`**  
  Save model (and tokenizer) in HuggingFace format; for PEFT, saves adapters only.

- **`push_to_hub(repo_id, **kwargs)`**  
  Push model and tokenizer to HuggingFace Hub.

---

## Callbacks

Base: **`Callback`** — abstract base with hooks: `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_step_begin`, `on_step_end`, `on_evaluate`. Override as needed.

### `EarlyStoppingCallback(patience=5, min_delta=1e-4, metric="loss", mode="min")`

Stops training when `metric` does not improve for `patience` epochs. `mode`: `"min"` or `"max"`. Expose `should_stop` for trainer to break.

### `CheckpointCallback(output_dir, save_best_only=True, save_total_limit=3, metric="loss", mode="min")`

Saves checkpoints (trainable state_dict, optimizer, metrics) per epoch; keeps best and up to `save_total_limit` others.

### `HistoryCallback(output_dir="./output")`

**New in v0.2.0.** Automatically added by `Trainer`. Saves `training_history.json` on training end.

**JSON Structure:**
```json
{
  "config": { ... },
  "model_type": "...",
  "history": [
    { "epoch": 0, "metrics": { "loss": 0.5, "accuracy": 85.0 } },
    ...
  ],
  "final_metrics": { ... }
}
```

### `LoggingCallback(log_every=10)`

**Redesigned in v0.2.0.** Minimalist style.
- `[INFO]`: General progress.
- `[WARN]`: Non-critical issues (e.g. missing bitsandbytes).
- `[SAVE]`: File saving events.

Logs step loss/LR every `log_every` steps and epoch metrics at end of epoch.

### `WandBCallback(project, name=None, config=None)`

Logs to Weights & Biases (step loss, epoch metrics). Requires `wandb`.

### `SparsityCallback(target_sparsity=0.5, start_epoch=0, frequency=1, skip_lora=True, min_params_to_prune=1000, log_details=False)`

Magnitude pruning per layer (trainable only; optionally skip LoRA). Applies every `frequency` epochs starting at `start_epoch`.  
**Methods:** `get_layer_sparsity(model) -> dict[str, float]` — per-layer sparsity.

---

## Scheduler

### `SmartScheduler(optimizer, initial_lr, config, num_training_steps=None)`

Epoch- or step-based LR schedule: warmup (ratio or epochs), then cosine/cosine_restart/linear/constant/polynomial.

**Methods:**

- **`step_epoch(epoch) -> float`**  
  Update LR by epoch; returns new LR.

- **`step() -> float`**  
  Update LR by step (when `num_training_steps` and warmup_ratio used); returns new LR.

- **`reduce_lr(factor=0.5) -> float`**  
  Reduce current LR by factor (e.g. after rollback).

- **`surge_lr(factor=3.0) -> float`**  
  Increase LR (e.g. final surge).

- **`get_lr() -> float`**  
  Current learning rate.

### `get_transformer_scheduler(optimizer, scheduler_type, num_warmup_steps, num_training_steps)`

Returns a HuggingFace-style scheduler (requires `transformers`).

---

## LR Finder

### `LRFinder(model, optimizer, criterion=None, device=None)`

Finds a good learning rate by exponentially sweeping LR and choosing a point near steepest descent. Works with any model (including HuggingFace); `criterion` can be None if model returns `loss`.

**Method:**

- **`find(train_loader, forward_fn=None, start_lr=1e-7, end_lr=1.0, num_steps=100, smooth_f=0.05, diverge_th=4.0) -> float`**  
  Runs sweep, restores initial state, returns suggested LR.  
  **forward_fn:** Optional `(model, batch) -> (loss, logits)`.

**Property:** `history -> dict` with `"lrs"` and `"losses"` for plotting.

---

## Utils

- **`__version__`**: Current library version.
- **`get_device(preference="auto") -> torch.device`**  
  Resolves device: auto (cuda/mps/cpu), cuda, cpu, mps. Prints device and GPU info.

- **`seed_everything(seed: int)`**  
  Sets random seed for Python, NumPy, and PyTorch (and CUDA if available).

- **`count_parameters(model, trainable_only=True) -> int`**  
  Counts parameters (optionally trainable only).

- **`format_params(num: int) -> str`**  
  Formats count (e.g. `"1.2M"`, `"3.4B"`).

- **`is_dict_like(obj) -> bool`**  
  True if obj is dict-like (e.g. BatchEncoding).

- **`to_dict(obj) -> dict`**  
  Converts dict-like to plain dict.

- **`move_to_device(batch, device, non_blocking=True)`**  
  Recursively moves batch (dict, tuple, list, tensor) to device.

- **`unpack_batch(batch) -> (inputs, labels)`**  
  Unpacks batch into inputs and labels (supports dict with `labels`/`label`, (inputs, labels) tuple, or single tensor).

- **`get_optimizer_grouped_params(model, weight_decay, no_decay_keywords=("bias", "LayerNorm", "layer_norm")) -> list[dict]`**  
  Returns param groups for optimizer (decay vs no-decay).

---

## Error Handling & Exceptions

SELGIS uses structured exceptions and logging to handle common ML issues.

### Common Exceptions
| Exception | Trigger | Recommended Action |
|-----------|---------|-------------------|
| `ImportError` | Missing `transformers`, `peft`, or `bitsandbytes`. | Install via `pip install selgis[all]`. |
| `ValueError` | Conflicting config (e.g. `fp16=True` & `bf16=True`). | Correct the configuration file/object. |
| `ZeroDivisionError` | Model has no trainable parameters. | Ensure LoRA is correctly applied or layers are unfrozen. |

### Training Protection (Rollback)
If `nan_recovery=True` (default), SELGIS monitors loss for anomalies.
1. **Detection**: `[WARN] Anomaly detected: loss=NaN`
2. **Action**: Weights are restored from `last_good_state.pt`.
3. **Recovery**: LR is reduced by 50% automatically.
4. **Resumption**: Training continues from the last stable point.

---

## Breaking Changes

### v0.2.0
- **Log Format**: All emojis removed (sticker-style). Automated log parsers relying on emojis will need updating.
- **Weights Loading**: `load_model` now defaults to `weights_only=True` for security.
- **Default Callbacks**: `HistoryCallback` is now automatically added to all trainers.
