"""
Конфигурация и схемы данных для датасетов Selgis.

TypedDict для валидации выходных данных и DatasetConfig для фабрики.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal, TypedDict
from pathlib import Path
import torch


# =============================================================================
# Схемы данных (TypedDict для валидации)
# =============================================================================

class TextSample(TypedDict, total=False):
    """
    Схема для текстовых данных (LLM, NLP).
    
    Пример:
        {
            "input_ids": tensor([1, 2, 3, ...]),        # (seq_len,)
            "attention_mask": tensor([1, 1, 1, ...]),   # (seq_len,)
            "labels": tensor([1, 2, 3, ...]),           # (seq_len,)
        }
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    text: str


class ImageSample(TypedDict, total=False):
    """
    Схема для изображений.
    
    Пример:
        {
            "inputs": tensor(C, H, W),          # Изображение
            "labels": tensor(class_id),         # Класс
            "image_path": str,                  # Путь к файлу
        }
    """
    inputs: torch.Tensor
    labels: torch.Tensor
    image_path: str
    pixel_values: torch.Tensor


class MultimodalSample(TypedDict, total=False):
    """
    Схема для мультимодальных данных (текст + изображения).
    
    Пример:
        {
            "inputs": {
                "pixel_values": tensor(C, H, W),
                "input_ids": tensor(seq_len,),
                "attention_mask": tensor(seq_len,),
            },
            "labels": str,
            "metadata": {...},
        }
    """
    inputs: Dict[str, Any]
    labels: Union[torch.Tensor, str]
    metadata: Dict[str, Any]


class TabularSample(TypedDict, total=False):
    """
    Схема для табличных данных.
    
    Пример:
        {
            "inputs": {
                "feature1": tensor(...),
                "feature2": tensor(...),
            },
            "labels": tensor(target),
        }
    """
    inputs: Dict[str, torch.Tensor]
    labels: torch.Tensor
    row_id: int


# Реестр схем по типам данных
SCHEMA_REGISTRY: Dict[str, type[TypedDict]] = {
    "text": TextSample,
    "image": ImageSample,
    "multimodal": MultimodalSample,
    "tabular": TabularSample,
}


# =============================================================================
# Конфигурация датасета
# =============================================================================

@dataclass
class DatasetConfig:
    """
    Конфигурация для создания датасета.
    
    Сериализуется в YAML/JSON для удобства.
    
    Пример использования:
        config = DatasetConfig(
            data_type="text",
            data_path="./data.jsonl",
            batch_size=32,
            tokenizer=tokenizer,
            max_length=512,
        )
        dataset = create_dataset(config)
    
    Attributes:
        data_type: Тип данных ("text", "image", "multimodal", "custom", "streaming", "tabular")
        data_path: Путь к основному файлу/папке данных
        train_path: Путь к train датасету (если отдельно от eval)
        eval_path: Путь к eval датасету (если отдельно от train)
        
        # Для мультимодальных/табличных данных
        image_path: Путь к папке с изображениями
        image_column: Название колонки с изображениями
        text_column: Название колонки с текстом
        label_column: Название колонки с метками
        
        # Параметры загрузки
        batch_size: Размер батча для train
        eval_batch_size: Размер батча для eval
        num_workers: Количество worker'ов для DataLoader
        prefetch_factor: Количество батчей для prefetch
        pin_memory: Использовать pinned memory
        persistent_workers: Не перезапускать workers между эпохами
        
        # Токенизация / трансформы
        tokenizer: HuggingFace tokenizer (для текста)
        image_processor: HuggingFace image processor (для изображений)
        transform: Torchvision трансформы (для изображений)
        format_fn: Кастомная функция форматирования данных
        
        # Кэширование и препроцессинг
        cache_dir: Директория для кэша (токены, изображения)
        use_cache: Использовать ли кэширование
        pre_tokenize: Pre-токенизировать текст при первом запуске
        pre_compute_features: Pre-вычислить фичи для изображений
        
        # Streaming для больших данных
        streaming: Использовать streaming режим
        buffer_size: Размер буфера для streaming
        
        # Разделение данных
        train_split: Доля train данных (если нет eval_path)
        seed: Random seed для воспроизводимости
        
        # Distributed training
        world_size: Количество GPU (для DDP)
        rank: Rank текущего процесса (для DDP)
        
        # Кастомные параметры
        custom_kwargs: Дополнительные параметры для конкретных датасетов
    """
    
    # Тип данных
    data_type: Literal["text", "image", "multimodal", "custom", "streaming", "tabular"] = "text"
    
    # Пути к данным
    data_path: Optional[Union[str, Path]] = None
    train_path: Optional[Union[str, Path]] = None
    eval_path: Optional[Union[str, Path]] = None
    
    # Для мультимодальных/табличных данных
    image_path: Optional[Union[str, Path]] = None
    image_column: Optional[str] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    
    # Параметры загрузки
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    pin_memory: bool = True
    persistent_workers: bool = False
    
    # Токенизация / трансформы
    tokenizer: Any = None
    image_processor: Any = None
    transform: Any = None
    format_fn: Optional[Callable] = None
    
    # Кэширование и препроцессинг
    cache_dir: Optional[Union[str, Path]] = None
    use_cache: bool = True
    pre_tokenize: bool = False
    pre_compute_features: bool = False
    
    # Streaming для больших данных
    streaming: bool = False
    buffer_size: int = 1000
    
    # Разделение данных
    train_split: float = 0.9
    seed: int = 42
    
    # Distributed training
    world_size: int = 1
    rank: int = 0
    
    # Дополнительные параметры
    max_length: int = 512
    custom_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Валидация конфигурации после инициализации."""
        # Преобразование путей
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
        
        # Валидация train_split
        if not 0.0 < self.train_split <= 1.0:
            raise ValueError(f"train_split должен быть в (0, 1], получил {self.train_split}")
        
        # Валидация streaming
        if self.streaming and self.data_type != "streaming":
            # Автоматически переключаем на streaming для больших файлов
            if self.data_path and isinstance(self.data_path, Path):
                if self.data_path.exists() and self.data_path.stat().st_size > 10 * 1024**3:  # 10GB
                    print(f"[WARN] Файл > 10GB, рекомендуется streaming=True")
        
        # Проверка conflicting параметров
        if self.pre_tokenize and self.streaming:
            print("[WARN] pre_tokenize игнорируется в streaming режиме")
            self.pre_tokenize = False
        
        # Установка prefetch_factor по умолчанию
        if self.prefetch_factor is None and self.num_workers > 0:
            self.prefetch_factor = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в dict (для JSON/YAML)."""
        import json
        from dataclasses import asdict
        
        # Исключаем не-сериализуемые поля
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
    def from_dict(cls, data: Dict[str, Any]) -> DatasetConfig:
        """Десериализация из dict."""
        # Конвертируем пути обратно в Path
        for key in ["data_path", "train_path", "eval_path", "image_path", "cache_dir"]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        
        return cls(**data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Сохранить конфигурацию в JSON файл."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] Конфигурация сохранена: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> DatasetConfig:
        """Загрузить конфигурацию из JSON файла."""
        import json
        
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
