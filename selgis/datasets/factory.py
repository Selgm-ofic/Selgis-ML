"""
Фабрика для создания датасетов и DataLoader.

Единая точка входа для создания датасетов всех типов.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, IterableDataset

from selgis.datasets.config import DatasetConfig
from selgis.datasets.base import BaseDataset, StreamingDataset
from selgis.utils import seed_everything


# =============================================================================
# Фабрика датасетов
# =============================================================================

def create_dataset(config: DatasetConfig) -> BaseDataset:
    """
    Создать датасет на основе конфигурации.
    
    Args:
        config: Конфигурация датасета
    
    Returns:
        Dataset объект
    
    Raises:
        ValueError: Если тип данных не поддерживается или конфигурация некорректна
        ImportError: Если требуются дополнительные зависимости
    
    Пример использования:
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
            f"Неподдерживаемый тип данных: {data_type}. "
            f"Доступные: text, image, multimodal, streaming, custom"
        )


def _create_text_dataset(config: DatasetConfig):
    """Создать текстовый датасет."""
    from selgis.datasets.text import TextDataset, HFTextDataset
    
    data_path = config.data_path or config.train_path
    
    if data_path is None:
        raise ValueError("Для text датасета требуется data_path или train_path")
    
    # Проверка — это HF dataset или локальный файл
    if isinstance(data_path, str) and "/" in data_path and not Path(data_path).exists():
        # Это HF dataset name (например, "tatsu-lab/alpaca")
        return HFTextDataset(
            dataset_name=str(data_path),
            tokenizer=config.tokenizer,
            max_length=config.max_length,
            cache_dir=config.cache_dir if config.use_cache else None,
            streaming=config.streaming,
            text_column=config.text_column or "text",
            format_fn=config.format_fn,
        )
    
    # Локальный файл
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
    """Создать датасет изображений."""
    from selgis.datasets.image import ImageDataset
    
    data_path = config.data_path or config.train_path
    
    if data_path is None:
        raise ValueError("Для image датасета требуется data_path или train_path")
    
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
    """Создать мультимодальный датасет."""
    from selgis.datasets.multimodal import MultimodalDataset
    
    data_path = config.data_path or config.train_path
    
    if data_path is None:
        raise ValueError("Для multimodal датасета требуется data_path или train_path")
    
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
    """Создать streaming датасет."""
    from selgis.datasets.streaming import StreamingTextDataset, StreamingCSVDataset
    
    data_path = config.data_path or config.train_path
    
    if data_path is None:
        raise ValueError("Для streaming датасета требуется data_path или train_path")
    
    # Определение формата по расширению
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
        # По умолчанию — JSONL
        return StreamingTextDataset(
            data_path=data_path,
            tokenizer=config.tokenizer,
            max_length=config.max_length,
            buffer_size=config.buffer_size,
            format_fn=config.format_fn,
            total_lines=config.custom_kwargs.get("total_lines"),
        )


def _create_custom_dataset(config: DatasetConfig):
    """Создать обёртку для кастомного датасета."""
    from selgis.datasets.custom import CustomDataset
    
    custom_dataset = config.custom_kwargs.get("dataset")
    
    if custom_dataset is None:
        raise ValueError("Для custom типа нужно передать dataset в custom_kwargs['dataset']")
    
    return CustomDataset(
        dataset=custom_dataset,
        wrap_key=config.custom_kwargs.get("wrap_key", "inputs"),
        label_key=config.custom_kwargs.get("label_key", "labels"),
        collate_fn=config.custom_kwargs.get("collate_fn"),
    )


# =============================================================================
# Фабрика DataLoader
# =============================================================================

def create_dataloaders(
    config: DatasetConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Создать train и eval DataLoader.
    
    Args:
        config: Конфигурация датасета
    
    Returns:
        (train_loader, eval_loader)
    
    Пример использования:
        config = DatasetConfig(
            data_type="text",
            data_path="./data.jsonl",
            batch_size=32,
            num_workers=4,
        )
        train_loader, eval_loader = create_dataloaders(config)
    """
    # Создание train датасета
    train_dataset = create_dataset(config)
    
    # Валидация
    print(f"[INFO] Валидация train датасета...")
    train_dataset.validate()
    
    # Разделение на train/eval
    if config.eval_path:
        # Отдельный eval датасет
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
        eval_dataset.validate()
        print(f"[INFO] Валидация eval датасета...")
    else:
        # Разделение одного датасета
        if isinstance(train_dataset, IterableDataset):
            # Streaming датасеты не поддерживают random_split
            print("[WARN] Streaming датасеты не поддерживают eval_split. Eval loader не создан.")
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
                print(f"[INFO] Разделение: train={train_size}, eval={eval_size}")
            else:
                eval_dataset = None
    
    # Создание DataLoader
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
    Создать DataLoader с правильными настройками для воспроизводимости и DDP.
    
    Args:
        dataset: Dataset для загрузки
        batch_size: Размер батча
        num_workers: Количество worker'ов
        prefetch_factor: Количество батчей для prefetch
        pin_memory: Использовать pinned memory
        shuffle: Перемешивать ли данные
        seed: Random seed для воспроизводимости
        world_size: Количество GPU (для DDP)
        rank: Rank текущего процесса (для DDP)
        persistent_workers: Не перезапускать workers между эпохами
    
    Returns:
        DataLoader
    """
    from torch.utils.data import IterableDataset
    
    # Проверка на streaming датасет
    is_streaming = isinstance(dataset, IterableDataset)
    
    if is_streaming and shuffle:
        print("[WARN] IterableDataset не поддерживает shuffle. Отключаю.")
        shuffle = False
    
    # Collate функция
    collate_fn = getattr(dataset, 'collate_fn', None)
    
    # Sampler для DDP
    sampler = None
    if world_size > 1 and not is_streaming:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        shuffle = False  # Sampler уже перемешивает
    
    # Worker init function для seed
    def worker_init_fn(worker_id: int) -> None:
        # Seed для каждого воркера
        worker_seed = seed + worker_id
        seed_everything(worker_seed)
    
    # Настройка prefetch_factor
    if num_workers == 0:
        prefetch_factor = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )


# =============================================================================
# Интеграция с Trainer
# =============================================================================

def prepare_data_for_trainer(
    trainer: Any,
    config: DatasetConfig,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Подготовить данные для Trainer.
    
    Args:
        trainer: Trainer или TransformerTrainer
        config: Конфигурация датасета
    
    Returns:
        (train_loader, eval_loader)
    
    Пример использования:
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
    
    # Проверка совместимости с тренером
    if hasattr(trainer, 'eval_dataloader') and eval_loader is None:
        print("[WARN] Eval DataLoader не создан. Обучение без валидации.")
    
    # Вывод статистики
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'get_stats'):
        stats = train_dataset.get_stats()
        print(f"[INFO] Статистика датасета: {stats}")
    
    return train_loader, eval_loader
