"""
selgis/datasets — Универсальная фабрика загрузчиков данных.

Поддерживает:
- Текст (JSONL, TXT, CSV, HuggingFace datasets)
- Изображения (папки, CSV, WebDataset)
- Мультимодальные данные (текст + изображения)
- Кастомные датасеты пользователя
- Streaming для больших данных (>100GB)

Архитектура:
- BaseDataset — единый интерфейс для всех датасетов
- Конкретные реализации под каждый тип данных
- Фабрика для создания через конфигурацию
"""

from selgis.datasets.base import BaseDataset, StreamingDataset
from selgis.datasets.config import DatasetConfig
from selgis.datasets.text import TextDataset, HFTextDataset
from selgis.datasets.image import ImageDataset
from selgis.datasets.multimodal import MultimodalDataset
from selgis.datasets.streaming import StreamingTextDataset
from selgis.datasets.custom import CustomDataset
from selgis.datasets.factory import create_dataset, create_dataloaders, prepare_data_for_trainer

__all__ = [
    # Базовые классы
    "BaseDataset",
    "StreamingDataset",
    
    # Конфигурация
    "DatasetConfig",
    
    # Текстовые датасеты
    "TextDataset",
    "HFTextDataset",
    
    # Датасеты изображений
    "ImageDataset",
    
    # Мультимодальные датасеты
    "MultimodalDataset",
    
    # Streaming датасеты
    "StreamingTextDataset",
    
    # Кастомные датасеты
    "CustomDataset",
    
    # Фабрика
    "create_dataset",
    "create_dataloaders",
    "prepare_data_for_trainer",
]
