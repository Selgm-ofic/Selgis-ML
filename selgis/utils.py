"""Training utilities: device, seeding, batch handling, param groups."""

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def get_device(preference: str = "auto") -> torch.device:
    """Resolve compute device.

    Args:
        preference: Device string such as ``"auto"``, ``"cuda"``,
            ``"cuda:1"``, ``"cpu"``, or ``"mps"``.  ``"auto"`` selects
            the best available device.

    Returns:
        The selected ``torch.device``.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    print(f"[INFO] Device: {device}")

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        print(f"   GPU: {torch.cuda.get_device_name(idx)}")
        print(f"   Memory: {props.total_memory / 1024 ** 3:.2f} GB")

    return device


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for full reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic CUDA algorithms
            at the cost of potential performance reduction.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def count_parameters(
    model: nn.Module, trainable_only: bool = True,
) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only parameters with
            ``requires_grad=True``.

    Returns:
        Total number of (trainable) parameters.
    """
    if trainable_only:
        return sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    return sum(p.numel() for p in model.parameters())


def format_params(num: int) -> str:
    """Format parameter count for display.

    Args:
        num: Number of parameters.

    Returns:
        Human-readable string (e.g. ``"1.20B"``, ``"3.40M"``).
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def is_dict_like(obj: Any) -> bool:
    """Return True if *obj* is dict-like (dict, BatchEncoding, etc.)."""
    return isinstance(obj, Mapping)


def to_dict(obj: Any) -> dict:
    """Convert a dict-like object to a plain ``dict``.

    Args:
        obj: Object to convert.

    Returns:
        A plain ``dict``, or *obj* unchanged if not dict-like.
    """
    if isinstance(obj, dict):
        return obj
    if is_dict_like(obj):
        return dict(obj)
    return obj


def move_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = False,
) -> Any:
    """Recursively move tensors in *batch* to *device*.

    Handles dicts, tuples, lists, and bare tensors.  Non-tensor
    values are returned unchanged.

    Args:
        batch: Input data structure containing tensors.
        device: Target device.
        non_blocking: Use non-blocking transfers.  Only safe when
            source tensors reside in pinned memory
            (``DataLoader(pin_memory=True)``).

    Returns:
        A structure of the same shape with tensors on *device*.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if is_dict_like(batch):
        return {
            k: move_to_device(v, device, non_blocking)
            for k, v in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(
            move_to_device(x, device, non_blocking) for x in batch
        )
    if isinstance(batch, list):
        return [
            move_to_device(x, device, non_blocking) for x in batch
        ]
    return batch


def unpack_batch(
    batch: Any,
) -> tuple[Any, torch.Tensor | None]:
    """Unpack a batch into ``(inputs, labels)``.

    Supports several formats:

    - **dict/Mapping**: labels extracted from key ``"labels"`` or
      ``"label"``; the full dict is returned as inputs.
    - **tuple/list of length >= 2**: ``(inputs, labels, ...)``.
    - **tuple/list of length 1** or **single tensor**: labels is
      ``None``.

    Args:
        batch: Raw batch from a ``DataLoader``.

    Returns:
        Tuple ``(inputs, labels)`` where *labels* may be ``None``.
    """
    if is_dict_like(batch):
        labels = batch.get("labels", batch.get("label"))
        return batch, labels

    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None

    if isinstance(batch, torch.Tensor):
        return batch, None

    return batch, None


def get_optimizer_grouped_params(
    model: nn.Module,
    weight_decay: float,
    no_decay_keywords: tuple[str, ...] = (
        "bias",
        "LayerNorm",
        "layer_norm",
    ),
) -> list[dict]:
    """Group parameters for optimizer with selective weight decay.

    Bias and normalization parameters are placed in a group with zero
    weight decay.  Empty groups are omitted.

    Args:
        model: PyTorch model.
        weight_decay: Weight decay for the main parameter group.
        no_decay_keywords: Parameter name substrings that should
            receive zero weight decay.

    Returns:
        List of parameter group dicts with ``"params"`` and
        ``"weight_decay"`` keys.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict] = []
    if decay_params:
        groups.append(
            {"params": decay_params, "weight_decay": weight_decay},
        )
    if no_decay_params:
        groups.append(
            {"params": no_decay_params, "weight_decay": 0.0},
        )

    return groups