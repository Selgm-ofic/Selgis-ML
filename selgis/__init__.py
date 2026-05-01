"""
SELGIS Library - Universal Training Framework for PyTorch and HuggingFace Transformers.

Supports: PyTorch models, HuggingFace Transformers, custom architectures,
training protection (NaN/spike rollback), LR finder, callbacks, PEFT/LoRA,
and universal data loading (text, image, multimodal, streaming).
"""

from selgis.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    SparsityCallback,
    WandBCallback,
)
from selgis.checkpointing import GradientCheckpointingManager
from selgis.config import SelgisConfig, TransformerConfig
from selgis.core import SelgisCore
from selgis.datasets import (
    BaseDataset,
    CustomDataset,
    DatasetConfig,
    HFTextDataset,
    ImageDataset,
    MultimodalDataset,
    StreamingDataset,
    StreamingTextDataset,
    TextDataset,
    create_dataloaders,
    create_dataset,
    prepare_data_for_trainer,
)
from selgis.loss import ChunkedCrossEntropyLoss, CrossEntropyLossV2
from selgis.lr_finder import LRFinder
from selgis.scheduler import SmartScheduler, get_transformer_scheduler
from selgis.trainer import Trainer, TransformerTrainer
from selgis.utils import (
    count_parameters,
    format_params,
    get_device,
    get_optimizer_grouped_params,
    is_dict_like,
    move_to_device,
    seed_everything,
    to_dict,
    unpack_batch,
)


def __get_version() -> str:
    """Return package version from pyproject.toml (lazy loading)."""
    try:
        from importlib.metadata import version as _v
        return _v("selgis")
    except Exception:
        return "0.2.7.1"


def __getattr__(name: str):
    if name == "__version__":
        return __get_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "SelgisConfig",
    "TransformerConfig",
    # Core
    "SelgisCore",
    # Datasets
    "BaseDataset",
    "StreamingDataset",
    "DatasetConfig",
    "TextDataset",
    "HFTextDataset",
    "ImageDataset",
    "MultimodalDataset",
    "StreamingTextDataset",
    "CustomDataset",
    "create_dataset",
    "create_dataloaders",
    "prepare_data_for_trainer",
    # Training
    "LRFinder",
    "SmartScheduler",
    "get_transformer_scheduler",
    "Trainer",
    "TransformerTrainer",
    # Loss
    "ChunkedCrossEntropyLoss",
    "CrossEntropyLossV2",
    # Checkpointing
    "GradientCheckpointingManager",
    # Callbacks
    "Callback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "WandBCallback",
    "SparsityCallback",
    "HistoryCallback",
    # Utils
    "get_device",
    "seed_everything",
    "count_parameters",
    "format_params",
    "move_to_device",
    "unpack_batch",
    "get_optimizer_grouped_params",
    "is_dict_like",
    "to_dict",
]
