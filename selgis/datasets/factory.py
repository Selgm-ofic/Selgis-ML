"""
Factory for creating datasets and DataLoader.

Single entry point for creating datasets of all types.
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    DistributedSampler,
    IterableDataset,
    Subset,
)

from selgis.datasets.config import DatasetConfig
from selgis.datasets.base import BaseDataset, StreamingDataset
from selgis.utils import seed_everything

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Factory
# =============================================================================


def create_dataset(config: DatasetConfig) -> BaseDataset:
    """
    Create a dataset based on configuration.

    Args:
        config: Dataset configuration

    Returns:
        Dataset object

    Raises:
        ValueError: If data type is not supported or configuration is invalid
        ImportError: If additional dependencies are required

    Example usage:
        config = DatasetConfig(
            data_type="text",
            data_path="./data.jsonl",
            tokenizer=tokenizer,
            max_length=512,
        )
        dataset = create_dataset(config)
    """
    data_type = config.data_type

    if data_type == "text":
        return _create_text_dataset(config)

    elif data_type == "image":
        return _create_image_dataset(config)

    elif data_type == "multimodal":
        return _create_multimodal_dataset(config)

    elif data_type == "streaming":
        return _create_streaming_dataset(config)

    elif data_type == "custom":
        return _create_custom_dataset(config)

    else:
        raise ValueError(
            f"Unsupported data type: {data_type}. "
            f"Available: text, image, multimodal, streaming, custom"
        )


def _create_text_dataset(config: DatasetConfig):
    """Create a text dataset."""
    from selgis.datasets.text import TextDataset, HFTextDataset

    data_path = config.data_path or config.train_path

    if data_path is None:
        raise ValueError("Text dataset requires data_path or train_path")

    # Check if this is HF dataset or local file
    data_path_str = str(data_path)
    path_obj = Path(data_path_str)
    if "/" in data_path_str and not path_obj.exists() and path_obj.suffix == "":
        # This is HF dataset name (e.g., "tatsu-lab/alpaca")
        return HFTextDataset(
            dataset_name=str(data_path),
            tokenizer=config.tokenizer,
            max_length=config.max_length,
            cache_dir=config.cache_dir if config.use_cache else None,
            streaming=config.streaming,
            text_column=config.text_column or "text",
            format_fn=config.format_fn,
        )

    # Local file
    return TextDataset(
        data_path=data_path,
        tokenizer=config.tokenizer,
        max_length=config.max_length,
        format_fn=config.format_fn,
        cache_dir=config.cache_dir if config.use_cache else None,
        file_format=config.custom_kwargs.get("file_format", "jsonl"),
        text_column=config.text_column,
        pre_tokenize=config.pre_tokenize,
        use_mmap=config.custom_kwargs.get("use_mmap", True),
    )


def _create_image_dataset(config: DatasetConfig):
    """Create an image dataset."""
    from selgis.datasets.image import ImageDataset

    data_path = config.data_path or config.train_path

    if data_path is None:
        raise ValueError("Image dataset requires data_path or train_path")

    return ImageDataset(
        data_path=data_path,
        labels_path=config.custom_kwargs.get("labels_path"),
        transform=config.transform,
        image_processor=config.image_processor,
        cache_dir=config.cache_dir if config.use_cache else None,
        file_format=config.custom_kwargs.get("file_format", "folder"),
        image_column=config.image_column,
        label_column=config.label_column,
    )


def _create_multimodal_dataset(config: DatasetConfig):
    """Create a multimodal dataset."""
    from selgis.datasets.multimodal import MultimodalDataset

    data_path = config.data_path or config.train_path

    if data_path is None:
        raise ValueError("Multimodal dataset requires data_path or train_path")

    return MultimodalDataset(
        data_path=data_path,
        tokenizer=config.tokenizer,
        image_processor=config.image_processor,
        max_length=config.max_length,
        cache_dir=config.cache_dir if config.use_cache else None,
        format_fn=config.format_fn,
        image_root=config.image_path,
    )


def _create_streaming_dataset(config: DatasetConfig):
    """Create a streaming dataset."""
    from selgis.datasets.streaming import StreamingTextDataset, StreamingCSVDataset

    data_path = config.data_path or config.train_path

    if data_path is None:
        raise ValueError("Streaming dataset requires data_path or train_path")

    # Determine format by extension
    suffix = Path(data_path).suffix.lower()

    if suffix == ".csv":
        return StreamingCSVDataset(
            data_path=data_path,
            text_column=config.text_column or "text",
            label_column=config.label_column,
            buffer_size=config.buffer_size,
            format_fn=config.format_fn,
            total_lines=config.custom_kwargs.get("total_lines"),
        )
    else:
        # Default is JSONL
        return StreamingTextDataset(
            data_path=data_path,
            tokenizer=config.tokenizer,
            max_length=config.max_length,
            buffer_size=config.buffer_size,
            format_fn=config.format_fn,
            total_lines=config.custom_kwargs.get("total_lines"),
        )


def _create_custom_dataset(config: DatasetConfig):
    """Create a wrapper for custom dataset."""
    from selgis.datasets.custom import CustomDataset

    custom_dataset = config.custom_kwargs.get("dataset")

    if custom_dataset is None:
        raise ValueError("For custom type, you need to pass dataset in custom_kwargs['dataset']")

    return CustomDataset(
        dataset=custom_dataset,
        wrap_key=config.custom_kwargs.get("wrap_key", "inputs"),
        label_key=config.custom_kwargs.get("label_key", "labels"),
        collate_fn=config.custom_kwargs.get("collate_fn"),
    )


# =============================================================================
# DataLoader Factory
# =============================================================================


def create_dataloaders(
    config: DatasetConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and eval DataLoader.

    Args:
        config: Dataset configuration

    Returns:
        (train_loader, eval_loader)

    Example usage:
        config = DatasetConfig(
            data_type="text",
            data_path="./data.jsonl",
            batch_size=32,
            num_workers=4,
        )
        train_loader, eval_loader = create_dataloaders(config)
    """
    # Create train dataset
    train_dataset = create_dataset(config)

    # Validation
    logger.info("Validating train dataset...")
    try:
        train_dataset.validate()
    except Exception as e:
        logger.warning(f"Dataset validation failed: {e}")

    # Split into train/eval
    if config.eval_path:
        # Separate eval dataset
        eval_config = DatasetConfig(
            data_type=config.data_type,
            data_path=config.eval_path,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            tokenizer=config.tokenizer,
            image_processor=config.image_processor,
            transform=config.transform,
            cache_dir=config.cache_dir if config.use_cache else None,
            streaming=config.streaming,
            seed=config.seed,
            custom_kwargs=config.custom_kwargs,
            max_length=config.max_length,
        )
        eval_dataset = create_dataset(eval_config)
        try:
            eval_dataset.validate()
        except Exception as e:
            logger.warning(f"Eval dataset validation failed: {e}")
        logger.info("Validating eval dataset...")
    else:
        # Split single dataset
        if isinstance(train_dataset, IterableDataset):
            # Streaming datasets do not support random_split
            logger.warning("Streaming datasets do not support eval_split. Eval loader not created.")
            eval_dataset = None
        else:
            train_size = int(config.train_split * len(train_dataset))
            eval_size = len(train_dataset) - train_size

            if eval_size > 0:
                train_dataset, eval_dataset = random_split(
                    train_dataset,
                    [train_size, eval_size],
                    generator=torch.Generator().manual_seed(config.seed),
                )
                logger.info(
                    "Dataset split: train=%d, eval=%d",
                    train_size,
                    eval_size,
                )
            else:
                eval_dataset = None

    # Create DataLoader
    train_loader = _create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        shuffle=not config.streaming and not isinstance(train_dataset, IterableDataset),
        seed=config.seed,
        world_size=config.world_size,
        rank=config.rank,
        persistent_workers=config.persistent_workers,
    )

    if eval_dataset is not None:
        eval_loader = _create_dataloader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            shuffle=False,
            seed=config.seed,
            world_size=config.world_size,
            rank=config.rank,
            persistent_workers=config.persistent_workers,
        )
    else:
        eval_loader = None

    return train_loader, eval_loader


def _create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    world_size: int = 1,
    rank: int = 0,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create DataLoader with proper settings for reproducibility and DDP.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of workers
        prefetch_factor: Number of batches to prefetch
        pin_memory: Use pinned memory
        shuffle: Whether to shuffle data
        seed: Random seed for reproducibility
        world_size: Number of GPUs (for DDP)
        rank: Rank of current process (for DDP)
        persistent_workers: Do not restart workers between epochs

    Returns:
        DataLoader
    """
    from torch.utils.data import IterableDataset

    # Check for streaming dataset
    is_streaming = isinstance(dataset, IterableDataset)

    if is_streaming and shuffle:
        logger.warning("IterableDataset does not support shuffle. Disabling.")
        shuffle = False

    def _resolve_collate_fn(ds: Dataset):
        if isinstance(ds, Subset):
            return _resolve_collate_fn(ds.dataset)
        fn = getattr(ds, "collate_fn", None)
        return fn if callable(fn) else None

    # Collate function
    collate_fn = _resolve_collate_fn(dataset)

    # Sampler for DDP
    sampler = None
    if world_size > 1 and not is_streaming:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        shuffle = False  # Sampler already shuffles

    # Worker init function for seed (created only if needed)
    worker_init_fn = None
    if num_workers > 0:
        def worker_init_fn(worker_id: int) -> None:
            # Seed for each worker
            worker_seed = seed + worker_id
            seed_everything(worker_seed)

    # Configure prefetch_factor
    if num_workers == 0:
        prefetch_factor = None
    effective_pin_memory = pin_memory and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=effective_pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed) if not is_streaming else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )


# =============================================================================
# Trainer Integration
# =============================================================================


def prepare_data_for_trainer(
    trainer: Any,
    config: DatasetConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare data for Trainer.

    Args:
        trainer: Trainer or TransformerTrainer
        config: Dataset configuration

    Returns:
        (train_loader, eval_loader)

    Example usage:
        config = DatasetConfig(
            data_type="text",
            data_path="./data.jsonl",
            tokenizer=tokenizer,
            batch_size=32,
        )
        train_loader, eval_loader = prepare_data_for_trainer(trainer, config)

        trainer.train_dataloader = train_loader
        trainer.eval_dataloader = eval_loader
    """
    train_loader, eval_loader = create_dataloaders(config)

    # Check compatibility with trainer
    if hasattr(trainer, "eval_dataloader") and eval_loader is None:
        logger.warning("Eval DataLoader not created. Training without validation.")

    # Output statistics
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, "get_stats"):
        stats = train_dataset.get_stats()
        logger.info("Dataset statistics: %s", stats)

    return train_loader, eval_loader
