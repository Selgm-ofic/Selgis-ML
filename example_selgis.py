"""
Selgis ML - Working Examples
======================

This file contains practical templates for training with Selgis.
Copy and adapt for your use case.

Quick Reference:
    - Basic PyTorch: Example 1
    - LLMs with LoRA: Example 2
    - Custom Dataset: Example 3
    - Advanced Features: Example 4
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# EXAMPLE 1: Basic PyTorch Model Training
# =============================================================================
"""
Minimal example for training any PyTorch model.
Perfect for: CNN, RNN, custom architectures.
"""

def example_basic_training():
    from selgis import Trainer, SelgisConfig

    # 1. Define your model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # 2. Create dataset (replace with your data)
    class MyDataset(Dataset):
        def __init__(self, size=1000):
            self.X = torch.randn(size, 784)
            self.y = torch.randint(0, 10, (size,))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return {"inputs": self.X[idx], "labels": self.y[idx]}

    train_dataset = MyDataset(1000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(train_dataset, batch_size=32)

    # 3. Configure training
    config = SelgisConfig(
        max_epochs=10,
        learning_rate=1e-3,
        batch_size=32,
        nan_recovery=True,          # Auto-rollback on NaN
        grad_clip_norm=1.0,   # Prevent exploding gradients
        save_best_only=True,    # Save memory
    )

    # 4. Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        criterion=nn.CrossEntropyLoss(),
    )

    metrics = trainer.train()
    print(f"Final metrics: {metrics}")

    # 5. Save model
    trainer.save_model("model.pt")


# =============================================================================
# EXAMPLE 2: LLM Fine-tuning with LoRA
# =============================================================================
"""
Fine-tune large language models with 4-bit quantization and LoRA.
Perfect for: Qwen, Llama, Mistral, other transformers.
"""

def example_llm_training():
    from selgis import TransformerTrainer, TransformerConfig
    from selgis.datasets import DatasetConfig, create_dataloaders

    # 1. Configure for your LLM
    config = TransformerConfig(
        model_name_or_path="Qwen/Qwen2-0.5B",  # Or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        use_peft=True,

        # LoRA config (train only 0.1% of parameters)
        peft_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "task_type": "CAUSAL_LM",
        },

        # Quantization (75% memory savings)
        quantization_type="4bit",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,

        # Memory optimization
        gradient_checkpointing=True,
        cpu_offload=True,
        device_map="auto",

        # Training params
        max_epochs=3,
        learning_rate=2e-5,
        warmup_ratio=0.1,

        # Self-healing
        nan_recovery=True,
        final_surge_factor=5.0,
    )

    # 2. Create data loaders (JSONL format: {"text": "..."})
    ds_config = DatasetConfig(
        data_type="text",
        data_path="./data.jsonl",
        batch_size=4,
        max_length=512,
    )
    train_loader, eval_loader = create_dataloaders(ds_config)

    # 3. Create trainer
    trainer = TransformerTrainer(
        model_or_path=config.model_name_or_path,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    # 4. Train
    metrics = trainer.train()

    # 5. Save
    trainer.save_pretrained("./output/model")


# =============================================================================
# EXAMPLE 3: Custom Dataset
# =============================================================================
"""
Working with your own data format.
"""

def example_custom_dataset():
    from selgis import Trainer, SelgisConfig
    from selgis.datasets import CustomDataset, DatasetConfig, create_dataloaders
    import torch

    # Your custom PyTorch dataset
    class MyCustomDataset(Dataset):
        def __init__(self, path):
            # Load your data here
            self.data = torch.load(path)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                "inputs": item["features"],
                "labels": item["label"],
            }

    # Method 1: Direct
    dataset = MyCustomDataset("./data.pt")
    wrapped = CustomDataset(dataset)

    config = DatasetConfig(
        data_type="custom",
        custom_kwargs={"dataset": wrapped},
        batch_size=32,
    )
    train_loader, eval_loader = create_dataloaders(config)

    # Method 2: Via factory
    # Just pass your dataset in custom_kwargs


# =============================================================================
# EXAMPLE 4: Advanced Features
# =============================================================================
"""
Using callbacks, LR finder, and custom metrics.
"""

def example_advanced():
    from selgis import (
        Trainer,
        SelgisConfig,
        LoggingCallback,
        CheckpointCallback,
        EarlyStoppingCallback,
        HistoryCallback,
    )
    from selgis.callbacks import WandBCallback, SparsityCallback
    from torch.utils.data import Dataset, DataLoader
    import json

    # Custom metrics for your task
    def compute_metrics(preds, labels):
        """Compute task-specific metrics."""
        preds = preds.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()

        # Add more metrics: F1, precision, recall, etc.
        return {
            "accuracy": accuracy.item() * 100,
        }

    # 1. Advanced config
    config = SelgisConfig(
        max_epochs=100,
        learning_rate=1e-3,

        # LR Finder (auto-tune learning rate)
        lr_finder_enabled=True,
        lr_finder_steps=100,
        lr_finder_start=1e-6,
        lr_finder_end=1.0,

        # Early stopping
        patience=10,
        min_delta=0.01,
        primary_metric="accuracy",

        # Checkpointing
        output_dir="./checkpoints",
        save_best_only=True,
        save_total_limit=3,

        # Self-healing
        nan_recovery=True,
        spike_threshold=3.0,
        final_surge_factor=5.0,

        # Gradient
        grad_clip_norm=1.0,
        gradient_accumulation_steps=4,

        # Mixed precision
        fp16=True,

        # Regularization
        weight_decay=0.01,
        label_smoothing=0.1,

        # Memory
        cpu_offload=True,
        empty_cache_steps=100,
    )

    # 2. Custom callbacks
    callbacks = [
        LoggingCallback(log_every=10),
        EarlyStoppingCallback(
            patience=10,
            metric="accuracy",
            mode="max",
        ),
        CheckpointCallback(
            output_dir="./checkpoints",
            save_best_only=True,
            save_total_limit=3,
        ),
        HistoryCallback(output_dir="./checkpoints"),
        # WandBCallback(project="my-project"),
        # SparsityCallback(target_sparsity=0.5, start_epoch=10),
    ]

    model = nn.Linear(10, 2)
    dataset = MySimpleDataset(1000)
    loader = DataLoader(dataset, batch_size=32)

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        eval_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))


class MySimpleDataset(Dataset):
    def __init__(self, size):
        self.X = torch.randn(size, 10)
        self.y = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"inputs": self.X[idx], "labels": self.y[idx]}


# =============================================================================
# EXAMPLE 5: CLI Usage
# =============================================================================
"""
Train from command line using YAML config:

```bash
# Create config.yaml
cat > config.yaml << 'EOF'
model_name_or_path: "Qwen/Qwen2-0.5B"
use_peft: true
peft_config:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
quantization_type: "4bit"
max_epochs: 3
learning_rate: 2e-5
nan_recovery: true
EOF

# Train
selgis train --config config.yaml

# Or demo mode (no config needed)
selgis train
```
"""


# =============================================================================
# EXAMPLE 6: Resume from Checkpoint
# =============================================================================
"""
Continue training from a saved checkpoint.
"""

def example_resume():
    from selgis import Trainer, SelgisConfig

    config = SelgisConfig(
        max_epochs=20,
        resume_from_checkpoint="./checkpoints/checkpoint-epoch-4",
    )

    # Model and data must match checkpoint
    model = nn.Linear(10, 2)
    loader = DataLoader(MySimpleDataset(1000), batch_size=32)

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )

    metrics = trainer.train()


# =============================================================================
# Run Examples
# =============================================================================
if __name__ == "__main__":
    # Uncomment to run:
    # example_basic_training()
    # example_llm_training()
    # example_custom_dataset()
    example_advanced()

    print("Selgis examples loaded. Uncomment a function to run.")