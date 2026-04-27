"""Dataset classes for backwards compatibility.

Note: This module provides placeholder classes. The full dataset
implementation was removed since it depended on external packages
that are not core dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Configuration for data loading.

    Args:
        data_type: Type of data (text, image, custom).
        data_path: Path to data file or directory.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        train_split: Fraction for training split.
        num_workers: Number of data loading workers.
    """

    data_type: str = "text"
    data_path: str = ""
    batch_size: int = 32
    max_length: int = 512
    train_split: float = 0.8
    num_workers: int = 0
    tokenizer: Any = None
    transform: Any = None
    buffer_size: int = 1000
    custom_kwargs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "data_type": self.data_type,
            "data_path": self.data_path,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "train_split": self.train_split,
            "num_workers": self.num_workers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DatasetConfig:
        """Create config from dictionary."""
        return cls(**data)


class BaseDataset:
    """Base dataset class."""

    def __init__(self, data=None):
        self.data = data

    def __len__(self) -> int:
        return len(self.data) if self.data else 0

    def __getitem__(self, idx):
        raise NotImplementedError

    def validate(self) -> bool:
        """Validate dataset."""
        return True

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {"size": len(self)}


class CustomDataset(BaseDataset):
    """Wrapper for custom PyTorch Dataset."""

    def __init__(self, dataset: Any):
        super().__init__(dataset)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TextDataset(BaseDataset):
    """Text dataset (placeholder)."""

    def __init__(self, data_path: str = "", tokenizer: Any = None, max_length: int = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length


class HFTextDataset(BaseDataset):
    """HuggingFace text dataset (placeholder)."""

    def __init__(self, data_path: str = "", tokenizer: Any = None, max_length: int = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length


class ImageDataset(BaseDataset):
    """Image dataset (placeholder)."""

    def __init__(self, data_path: str = "", transform: Any = None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform


class MultimodalDataset(BaseDataset):
    """Multimodal dataset (placeholder)."""

    def __init__(self, data_path: str = ""):
        super().__init__()
        self.data_path = data_path


class StreamingTextDataset(BaseDataset):
    """Streaming text dataset (placeholder)."""

    def __init__(self, data_path: str = "", buffer_size: int = 1000):
        super().__init__()
        self.data_path = data_path
        self.buffer_size = buffer_size


class StreamingDataset(BaseDataset):
    """Streaming dataset (placeholder)."""

    def __init__(self, data_path: str = "", buffer_size: int = 1000):
        super().__init__()
        self.data_path = data_path
        self.buffer_size = buffer_size


def create_dataset(config: DatasetConfig) -> BaseDataset:
    """Create dataset from config."""
    if config.data_type == "custom" and config.custom_kwargs:
        dataset_class = config.custom_kwargs.get("dataset")
        if dataset_class:
            return CustomDataset(dataset_class())
    return BaseDataset()


def create_dataloaders(config: DatasetConfig):
    """Create train/eval DataLoaders from config.

    Returns:
        Tuple of (train_loader, eval_loader).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if config.data_type == "custom" and config.custom_kwargs:
        dataset = config.custom_kwargs.get("dataset")
        if dataset:
            ds = CustomDataset(dataset)
            train_size = int(len(ds) * config.train_split)
            eval_size = len(ds) - train_size
            train_ds, eval_ds = torch.utils.data.random_split(ds, [train_size, eval_size])
            return (
                DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
                DataLoader(eval_ds, batch_size=config.batch_size),
            )

    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    train_size = int(len(dataset) * config.train_split)
    eval_size = len(dataset) - train_size
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_size, eval_size])
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        DataLoader(eval_ds, batch_size=config.batch_size),
    )


def prepare_data_for_trainer(data: Any, tokenizer: Any = None) -> Any:
    """Prepare data for trainer."""
    return data
