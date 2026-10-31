---
name: selgis
description: >
  Verified API cheat sheet and pitfalls for the selgis training framework
  (self-healing PyTorch/HF training: NaN/loss-spike detection, automatic
  rollback). Use when writing or debugging selgis training scripts —
  config fields, Trainer vs TransformerTrainer, dataset wiring, rollback
  behavior, and common error messages with fixes.
---

# Selgis quick reference

Single-GPU training framework. `Trainer` wraps any PyTorch model;
`TransformerTrainer` adds HuggingFace model/tokenizer loading, LoRA/PEFT,
quantization and chunked cross-entropy. Both protect training automatically:
on NaN/Inf loss or a spike, the last stable state is restored, optimizer
moments are zeroed, the LR is halved, and the offending batch is skipped.

## Imports

```python
from selgis import Trainer, TransformerTrainer, SelgisConfig, TransformerConfig
from selgis import DatasetConfig, create_dataloaders
from selgis import (LoggingCallback, EarlyStoppingCallback, CheckpointCallback,
                    HistoryCallback, WandBCallback, SparsityCallback)
```

## SelgisConfig — key fields (defaults)

`batch_size=32` · `max_epochs=100` · `learning_rate=1e-3` · `patience=5` ·
`nan_recovery=True` · `spike_threshold=3.0` (rollback when loss > 3× mean of
last `min_history_len=10` losses) · `max_consecutive_rollbacks=10` (stop with
`RuntimeError` after N rollbacks in a row with no healthy batch between; `0`
disables) · `grad_clip_norm=1.0` · `state_update_interval=100` (snapshot every
N optimizer steps; a rollback loses up to that many steps) ·
`state_storage="disk"|"memory"` · `output_dir="./output"` · `seed=42` ·
`warmup_epochs=0` / `warmup_ratio=0.0` (mutually exclusive) ·
`scheduler_type="cosine_restart"` · `fp16`/`bf16` · `lr_finder_enabled=False`.

Constructor: `Trainer(model, config, train_dataloader, eval_dataloader=None,
criterion=None, optimizer=None, callbacks=None, forward_fn=None,
compute_metrics=None)` → `.train()` returns the final metrics dict.

## LLM path (TransformerTrainer + LoRA)

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
config = TransformerConfig(
    model_name_or_path="Qwen/Qwen2-0.5B",
    use_peft=True, peft_config={"r": 16},
    quantization_type="4bit",        # 4bit/8bit need a CUDA GPU
    problem_type="causal_lm",
    chunked_ce=True,                 # memory-efficient CE for generative tasks
)
data_config = DatasetConfig(
    data_type="text", data_path="data.jsonl",
    tokenizer=tokenizer,             # REQUIRED: produces input_ids
    batch_size=4, max_length=512,
)
train_loader, eval_loader = create_dataloaders(data_config)
trainer = TransformerTrainer("Qwen/Qwen2-0.5B", config=config,
                             train_dataloader=train_loader,
                             eval_dataloader=eval_loader, tokenizer=tokenizer)
trainer.train()
```

`batch_size` belongs in `DatasetConfig`, not `TransformerConfig`
(selgis/config.py:151). `TransformerConfig` overrides `learning_rate=2e-5`.

## Five traps (verified against source)

1. **`criterion=` is required for non-dict batches** — a plain
   `(inputs, labels)` batch with no criterion raises
   `ValueError: Criterion required for non-dict input`
   (selgis/trainer.py:527). Dict batches `{"inputs": ..., "labels": ...}` work
   without a criterion only if the model computes its own loss.
2. **`tokenizer=` in `DatasetConfig` is required for text** — without it the
   dataset yields raw `{"text": ...}` records and training fails with
   "Dataset must return tokenized 'input_ids', not raw 'text'"
   (selgis/trainer.py:494; tokenizer field: selgis/datasets/config.py:197).
3. **Built-in accuracy is a fraction in [0, 1]**, not a percentage
   (selgis/trainer.py:617). `EarlyStoppingCallback(mode="max")` thresholds
   must be written accordingly (e.g. `min_delta=0.01`, not `1`).
4. **`warmup_ratio > 0` switches the scheduler to per-optimizer-step mode**
   (selgis/trainer.py:337, 435). With `warmup_epochs` the schedule is
   epoch-based. Setting both raises `ValueError`.
5. **`chunked_ce=True` only applies to `problem_type` in
   {`causal_lm`, `seq2seq`, `masked_lm`}** (selgis/trainer.py:766); for
   classification it is silently ignored (tiny vocab, no benefit).

## Pattern 1 — basic training

```python
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from selgis import Trainer, SelgisConfig

config = SelgisConfig(max_epochs=10, batch_size=32)
trainer = Trainer(model, config, train_dataloader=loader,
                  eval_dataloader=eval_loader, criterion=nn.CrossEntropyLoss())
metrics = trainer.train()   # {"train_loss": ..., "loss": ..., "accuracy": ...}
```

## Pattern 2 — self-healing A/B (NaN sabotage)

Poison the **real** loss of one batch (a bare NaN leaf never touches the
graph and poisons nothing). Full runnable version:
`demo/selfhealing_demo.ipynb`.

```python
class SabotageLoss(nn.Module):
    def __init__(self, ce, poison_call):
        super().__init__(); self.ce, self.poison_call, self.calls = ce, poison_call, 0
    def forward(self, logits, labels):
        loss = self.ce(logits, labels); self.calls += 1
        return loss * float("nan") if self.calls == self.poison_call else loss

config = SelgisConfig(max_epochs=5, batch_size=16, nan_recovery=True,
                      state_update_interval=5, state_storage="memory")
trainer = Trainer(model, config, train_dataloader=loader, criterion=SabotageLoss(nn.CrossEntropyLoss(), 21))
metrics = trainer.train()   # NaN detected BEFORE backward: rollback, skip, finish finite
```

Vanilla loop on the same task dies: NaN gradients → NaN weights →
`train_loss=nan` forever. Selgis logs "Rollback triggered: NaN/Inf loss
detected" and continues. Capture it: `logging.getLogger("selgis")` +
a logging handler, then assert `"Rollback triggered"` appears.

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `Criterion required for non-dict input` | tensor batch without a criterion | pass `criterion=nn.CrossEntropyLoss()` (or return loss from the model) |
| `Dataset must return tokenized 'input_ids', not raw 'text'` | `DatasetConfig` without `tokenizer=` | pass the tokenizer to `DatasetConfig` |
| `peft_config or adapter_name_or_path must be provided when use_peft=True` | LoRA enabled without adapter config | add `peft_config={"r": 16, "lora_alpha": 32}` (selgis/config.py:234) |
| `quantization_type='4bit' requires GPU, but device='cpu'` | quantization is CUDA-only | run on GPU (`device="auto"`), drop `quantization_type` on CPU |
| `Use either warmup_epochs or warmup_ratio, not both` | two warmup modes set | keep exactly one |
| Loss NaN and training stopped with `RuntimeError ... data is likely corrupted` | every batch keeps failing (corrupted data/labels) | inspect the dataset; raise `max_consecutive_rollbacks` or set `0` to restore the old unlimited-rollback behavior |
| `forward() got an unexpected keyword argument 'inputs'` | dict batch against a plain module | fixed in ≥0.2.7.2 (positional fallback); update selgis |
