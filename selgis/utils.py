"""Training utilities: device, seeding, batch handling, param groups."""

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def get_device(preference: str = "auto") -> torch.device:
    """Resolve compute device. Options: auto, cuda, cpu, mps.

    Args:
        preference: Device string; "auto" selects cuda/mps/cpu.

    Returns:
        The selected torch device. Prints device and GPU info to stdout.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    print(f"[INFO] Device: {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True

    return device


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters (optionally trainable only)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num: int) -> str:
    """Format parameter count (e.g. 1.2M, 3.4B)."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def is_dict_like(obj: Any) -> bool:
    """Return True if obj is dict-like (e.g. BatchEncoding, dict)."""
    return isinstance(obj, Mapping)


def to_dict(obj: Any) -> dict:
    """Convert dict-like object to plain dict."""
    if isinstance(obj, dict):
        return obj
    if is_dict_like(obj):
        return dict(obj)
    return obj


def move_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    """Move batch (tensors, dicts, tuples, lists) to device recursively."""
    if is_dict_like(batch):
        return {
            k: move_to_device(v, device, non_blocking)
            for k, v in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(move_to_device(x, device, non_blocking) for x in batch)
    if isinstance(batch, list):
        return [move_to_device(x, device, non_blocking) for x in batch]
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    return batch


def unpack_batch(batch: Any) -> tuple[Any, torch.Tensor | None]:
    """Unpack batch into (inputs, labels).

    Supports dict/Mapping (keys labels or label), (inputs, labels) tuple/list,
    or single tensor (labels=None).

    Returns:
        Tuple (inputs, labels); labels may be None.
    """
    if is_dict_like(batch):
        labels = None
        if "labels" in batch:
            labels = batch["labels"]
        elif "label" in batch:
            labels = batch["label"]
        return batch, labels
    
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    
    if isinstance(batch, (tuple, list)) and len(batch) == 1:
        return batch[0], None
    
    if isinstance(batch, torch.Tensor):
        return batch, None
    
    return batch, None


def get_optimizer_grouped_params(
    model: nn.Module,
    weight_decay: float,
    no_decay_keywords: tuple[str, ...] = ("bias", "LayerNorm", "layer_norm"),
) -> list[dict]:
    """Group parameters for optimizer: decay and no_decay (bias, LayerNorm excluded).

    Returns:
        List of param group dicts with "params" and "weight_decay".
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]