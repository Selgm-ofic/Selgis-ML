"""
Configuration and data schemas for Selgis datasets.

TypedDict for output data validation and DatasetConfig for factory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

# =============================================================================
# Data Schemas (TypedDict for validation)
# =============================================================================


class TextSample(TypedDict, total=False):
    """
    Schema for text data (LLM, NLP).

    Example:
        {
            "input_ids": list[int],        # (seq_len,)
            "attention_mask": list[int],   # (seq_len,)
            "labels": list[int],           # (seq_len,)
        }
    """

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    text: str


class ImageSample(TypedDict, total=False):
    """
    Schema for images.

    Example:
        {
            "inputs": list[float],          # Image
            "labels": int,                 # Class
            "image_path": str,              # File path
        }
    """

    inputs: list[float]
    labels: int
    image_path: str
    pixel_values: list[float]


class MultimodalSample(TypedDict, total=False):
    """
    Schema for multimodal data (text + images).

    Example:
        {
            "inputs": {
                "pixel_values": list[float],
                "input_ids": list[int],
                "attention_mask": list[int],
            },
            "labels": str,
            "metadata": dict,
        }
    """

    inputs: dict
    labels: str | int
    metadata: dict


class TabularSample(TypedDict, total=False):
    """
    Schema for tabular data.

    Example:
        {
            "inputs": dict[str, list[float]],
            "labels": list[float],
            "row_id": int,
        }
    """

    inputs: dict[str, list[float]]
    labels: list[float]
    row_id: int


# Schema registry by data types
SCHEMA_REGISTRY: dict[str, type] = {
    "text": TextSample,
    "image": ImageSample,
    "multimodal": MultimodalSample,
    "tabular": TabularSample,
}


# =============================================================================
# Dataset Configuration
# =============================================================================


@dataclass
class DatasetConfig:
    """
    Configuration for creating dataset.

    Serializable to YAML/JSON for convenience.

    Example usage:
        config = DatasetConfig(
            data_type="text",
            data_path="./data.jsonl",
            batch_size=32,
            tokenizer=tokenizer,
            max_length=512,
        )
        dataset = create_dataset(config)

    Attributes:
        data_type: Data type ("text", "image", "multimodal", "custom", "streaming", "tabular")
        data_path: Path to main data file/folder
        train_path: Path to train dataset (if separate from eval)
        eval_path: Path to eval dataset (if separate from train)

        # For multimodal/tabular data
        image_path: Path to images folder
        image_column: Name of image column
        text_column: Name of text column
        label_column: Name of label column

        # Loading parameters
        batch_size: Batch size for train
        eval_batch_size: Batch size for eval
        num_workers: Number of workers for DataLoader
        prefetch_factor: Number of batches to prefetch
        pin_memory: Use pinned memory
        persistent_workers: Do not restart workers between epochs

        # Tokenization / transforms
        tokenizer: HuggingFace tokenizer (for text)
        image_processor: HuggingFace image processor (for images)
        transform: Torchvision transforms (for images)
        format_fn: Custom data formatting function

        # Caching and preprocessing
        cache_dir: Directory for cache (tokens, images)
        use_cache: Whether to use caching
        pre_tokenize: Pre-tokenize text on first run
        pre_compute_features: Pre-compute features for images

        # Streaming for large data
        streaming: Use streaming mode
        buffer_size: Buffer size for streaming

        # Data splitting
        train_split: Fraction of train data (if no eval_path)
        seed: Random seed for reproducibility

        # Distributed training
        world_size: Number of GPUs (for DDP)
        rank: Rank of current process (for DDP)

        # Custom parameters
        custom_kwargs: Additional parameters for specific datasets
    """

    # Data type
    data_type: Literal["text", "image", "multimodal", "custom", "streaming", "tabular"] = "text"

    # Data paths
    data_path: str | Path | None = None
    train_path: str | Path | None = None
    eval_path: str | Path | None = None

    # For multimodal/tabular data
    image_path: str | Path | None = None
    image_column: str | None = None
    text_column: str | None = None
    label_column: str | None = None

    # Loading parameters
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 0
    prefetch_factor: int | None = None
    pin_memory: bool = True
    persistent_workers: bool = False

    # Tokenization / transforms
    tokenizer: Any = None
    image_processor: Any = None
    transform: Any = None
    format_fn: Callable | None = None

    # Caching and preprocessing
    cache_dir: str | Path | None = None
    use_cache: bool = True
    pre_tokenize: bool = False
    pre_compute_features: bool = False

    # Streaming for large data
    streaming: bool = False
    buffer_size: int = 1000

    # Data splitting
    train_split: float = 0.9
    seed: int = 42

    # Distributed training
    world_size: int = 1
    rank: int = 0

    # Chat format (auto-detect if None)
    chat_format: str | None = None

    # Custom chat roles (for messages format)
    user_role: str = "user"
    assistant_role: str = "assistant"

    # Additional parameters
    max_length: int = 512
    custom_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert paths
        if self.data_path and isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if self.train_path and isinstance(self.train_path, str):
            self.train_path = Path(self.train_path)
        if self.eval_path and isinstance(self.eval_path, str):
            self.eval_path = Path(self.eval_path)
        if self.image_path and isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        if self.cache_dir and isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        # Validate train_split
        if not 0.0 < self.train_split <= 1.0:
            raise ValueError(f"train_split must be in (0, 1], got {self.train_split}")

        # Validate streaming
        if self.streaming and self.data_type != "streaming":
            # Automatically switch to streaming for large files
            if self.data_path and isinstance(self.data_path, Path):
                if self.data_path.exists() and self.data_path.stat().st_size > 10 * 1024**3:  # 10GB
                    print("[WARN] File > 10GB, streaming=True recommended")

        # Check conflicting parameters
        if self.pre_tokenize and self.streaming:
            print("[WARN] pre_tokenize ignored in streaming mode")
            self.pre_tokenize = False

        # Set default prefetch_factor
        if self.prefetch_factor is None and self.num_workers > 0:
            self.prefetch_factor = 2

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for JSON/YAML)."""
        from dataclasses import asdict

        # Exclude non-serializable fields
        exclude = {"tokenizer", "image_processor", "transform", "format_fn"}

        result = {}
        for key, value in asdict(self).items():
            if key in exclude:
                continue
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        """Deserialize from dict."""
        # Convert paths back to Path
        for key in ["data_path", "train_path", "eval_path", "image_path", "cache_dir"]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])

        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[SAVE] Configuration saved: {path}")

    @classmethod
    def load(cls, path: str | Path) -> DatasetConfig:
        """Load configuration from JSON file."""
        import json

        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
