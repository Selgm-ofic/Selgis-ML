"""
SELGIS Library - Universal Training Framework for PyTorch and HuggingFace Transformers.

Supports: PyTorch models, HuggingFace Transformers, custom architectures,
training protection (NaN/spike rollback), LR finder, callbacks, PEFT/LoRA,
and universal data loading (text, image, multimodal, streaming).
"""


def _get_version() -> str:
    """Return package version from metadata (single source: pyproject.toml)."""
    try:
        from importlib.metadata import version
        return version("selgis")
    except Exception:
        return "0.2.40"


__version__ = _get_version()

from selgis.callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    SparsityCallback,
    WandBCallback,
)
from selgis.config import SelgisConfig, TransformerConfig
from selgis.core import SelgisCore
from selgis.datasets import (
    BaseDataset,
    StreamingDataset,
    DatasetConfig,
    TextDataset,
    HFTextDataset,
    ImageDataset,
    MultimodalDataset,
    StreamingTextDataset,
    CustomDataset,
    create_dataset,
    create_dataloaders,
    prepare_data_for_trainer,
)
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

__all__ = [
    # Config
    "SelgisConfig",
    "TransformerConfig",
    
    # Core
    "SelgisCore",
    
    # Datasets (NEW)
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
