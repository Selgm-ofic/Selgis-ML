# 🛡️ Selgis AI

**Autonomous Self-Healing Training Framework for PyTorch & Transformers.**

[![PyPI](https://img.shields.io/pypi/v/selgis?color=blue)](https://pypi.org/project/selgis/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/selgis/)

**Selgis** (Self-Guided Intelligent Stability) is a library that turns unstable neural network training into a reliable, predictable process. It automatically detects **Loss Spikes**, **NaN/Inf values**, and **plateaus**, applying dynamic weight **Rollback** mechanisms and Learning Rate **Surges** to recover the run.

Especially effective for **LoRA/QLoRA finetuning of LLMs** (Llama, Qwen, Mistral) on consumer hardware, where standard trainers often crash with `OutOfMemory` errors or degrade due to fp16 instability.

---

## 🔥 Why Selgis?

Have you ever woken up in the morning to find your overnight run crashed with `Loss: NaN` at 80%? Or that the model "forgot" everything it learned due to a bad batch? Selgis solves this.

*   **🛡️ Self-Healing Loop:** Automatic rollback to the last stable state upon detecting anomalies (loss spikes / NaN).
*   **🧠 Memory-Safe Architecture:** State preservation logic tracks *only* trainable parameters (`trainable-only`). This allows training **Qwen-4B / Llama-7B** on cards with **8-12 GB VRAM** without OOM during checkpoints.
*   **⚡ Final Surge:** If the model gets stuck on a plateau, Selgis can automatically boost the LR by 5-10x to break through local minima ("defibrillator effect").
*   **📉 Smart Defaults:** Built-in LR Finder and adaptive scheduler presets.

---

## 📊 Benchmarks

We tested Selgis under extreme conditions on real hardware (Tesla T4 16GB). Here are the results:

| Task | Model | Problem | Selgis Solution | Result |
| :--- | :--- | :--- | :--- | :--- |
| **LLM Finetuning** | **Qwen-2.5-4B** (QLoRA) | OOM on 12GB cards + Loss Spike | Trainable-only state + Rollback | **Memory: 8.2 GB**, Loss < 0.001 |
| **Seq2Seq** | LSTM (1.4M) | Catastrophic Spike (Acc 52% → 44%) | Rollback + Surge | **+7% Accuracy** (Recovered to 59.04%) |
| **NLP** | BERT-base | Instability on small batch (16) | Stable LR Finder | **100.0% Accuracy** (in 3 epochs) |
| **CV** | CNN (MNIST) | Overfitting & micro-spikes | Micro-rollbacks | **99.09%** (Held at generalization peak) |

> *"Selgis doesn't just prevent explosions. It returns training to a productive track."*

---

## 🚀 Installation

```bash
# Base version (PyTorch only)
pip install selgis

# Full version (with Transformers, LoRA, quantization, and WandB support)
pip install "selgis[all]"
```

---

## 🛠️ Quick Start

### 1. Robust LLM Training (Llama / Qwen)

Selgis handles protection while you use the familiar Transformers API.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from selgis import TransformerTrainer, TransformerConfig

# Configuration with protection enabled
config = TransformerConfig(
    model_name_or_path="Qwen/Qwen-2.5-3B",
    use_peft=True,
    peft_config={
        "r": 8, 
        "target_modules": ["q_proj", "v_proj"]
    },
    
    # Enable Selgis protection
    nan_recovery=True,      # Auto-rollback on NaN/Spike
    state_storage="disk",   # Save RAM (store state on disk)
    patience=3              # Wait 3 epochs of stagnation before intervention
)

# Load model (4-bit for memory efficiency)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path, 
    load_in_4bit=True, 
    device_map="auto"
)

# Start training
trainer = TransformerTrainer(model, config, train_loader)
trainer.train() 
# You can go to sleep now. If the loss spikes, Selgis fixes it.
```

### 2. Standard PyTorch (Any Model)

```python
from selgis import Trainer, SelgisConfig
import torch

# Your model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
)

# Config
config = SelgisConfig(
    max_epochs=10,
    lr_finder_enabled=True,  # Auto-find optimal LR before start
    spike_threshold=3.0      # Rollback if loss jumps 3x
)

trainer = Trainer(
    model=model, 
    config=config, 
    train_dataloader=loader, 
    criterion=torch.nn.CrossEntropyLoss()
)
trainer.train()
```

---

## 💻 CLI (Command Line Interface)

Selgis ships with a handy CLI for diagnostics and quick execution.

| Command | Description |
| :--- | :--- |
| `selgis device` | Check GPU/CUDA/MPS availability and print device info. |
| `selgis train` | Run a minimal demo training on synthetic data (Smoke Test). |
| `selgis train --config <path>` | Run training using a config file (JSON supported, YAML coming soon). |
| `selgis version` | Print the current library version. |

Example environment check:
```bash
$ selgis device
🚀 Device: cuda
   GPU: NVIDIA Tesla T4
   Memory: 14.75 GB
```

---

## 📚 API Reference

Full technical documentation for `SelgisCore`, `Trainer`, `Callbacks`, and configuration classes is available in [API.md](API.md).

Key components:
*   **SelgisCore**: The brain of the system (protection, rollback, state management).
*   **TransformerTrainer**: Wrapper for the HuggingFace ecosystem.
*   **LRFinder**: Tool for finding the optimal learning rate.

---

## 📄 License

Apache 2.0 License. Free for commercial and research use.

**Selgis AI** — Make training boring (in a good way).
