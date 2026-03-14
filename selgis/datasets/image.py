"""
Датасеты для изображений.

Поддерживает:
- Папки с изображениями (по классам)
- CSV/JSON с путями к изображениям
- Кастомные трансформы (torchvision)
- HuggingFace image processors

Оптимизации:
- Prefetching изображений
- Кэширование обработанных изображений
- Поддержка WebDataset для больших архивов
"""
from __future__ import annotations
import json
import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from selgis.datasets.base import BaseDataset
from selgis.datasets.config import ImageSample


class ImageDataset(BaseDataset):
    """
    Датасет для изображений с оптимизированной загрузкой.
    
    Особенности:
    - Поддержка разных форматов (folder, csv, json)
    - Кастомные трансформы (torchvision)
    - HuggingFace image processors
    - Метрики производительности
    
    Пример использования:
        from torchvision import transforms
        
        dataset = ImageDataset(
            data_path="./data/imagenet",
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]),
        )
    """
    
    output_schema = ImageSample
    
    def __init__(
        self,
        data_path: Union[str, Path],
        labels_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        image_processor: Optional[Any] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        file_format: Literal["folder", "csv", "json"] = "folder",
        image_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.transform = transform
        self.image_processor = image_processor
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.file_format = file_format
        self.image_column = image_column or "image_path"
        self.label_column = label_column or "label"
        
        # Индекс файлов
        self._samples: List[Tuple[Path, Optional[int]]] = []
        self._class_names: List[str] = []
        
        # Метрики
        self._metrics = {
            "total_samples": 0,
            "load_time_ms": [],
        }
        
        # Валидация пути
        if not self.data_path.exists():
            raise FileNotFoundError(f"Путь не найден: {self.data_path}")
        
        self._build_index()
    
    def _build_index(self) -> None:
        """Построить индекс изображений."""
        if self.file_format == "folder":
            self._build_index_folder()
        elif self.file_format == "csv":
            self._build_index_csv()
        elif self.file_format == "json":
            self._build_index_json()
    
    def _build_index_folder(self) -> None:
        """Индекс для папки с классами."""
        class_to_idx = {}
        
        for class_dir in sorted(self.data_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_to_idx[class_name] = len(class_to_idx)
            self._class_names.append(class_name)
            
            # Поиск изображений
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
                for img_path in class_dir.glob(ext):
                    self._samples.append((img_path, class_to_idx[class_name]))
        
        print(f"[INFO] Найдено {len(self._samples)} изображений в {len(class_to_idx)} классах")
    
    def _build_index_csv(self) -> None:
        """Индекс для CSV файла."""
        data_file = self.labels_path or self.data_path
        
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for record in reader:
                img_path = record.get(self.image_column)
                label = record.get(self.label_column)
                
                if img_path:
                    if label is not None:
                        label = int(label)
                    self._samples.append((Path(img_path), label))
    
    def _build_index_json(self) -> None:
        """Индекс для JSON файла."""
        data_file = self.labels_path or self.data_path
        
        with open(data_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
            
            for record in records:
                img_path = record.get(self.image_column)
                label = record.get(self.label_column)
                
                if img_path:
                    if label is not None:
                        label = int(label)
                    self._samples.append((Path(img_path), label))
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Получить изображение с метриками производительности."""
        start_time = time.perf_counter()
        
        img_path, label = self._samples[idx]
        
        # Загрузка изображения
        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки изображения {img_path}: {e}")
        
        # Трансформы
        if self.image_processor:
            # HuggingFace processor
            image_input = self.image_processor(image, return_tensors="pt")
            if isinstance(image_input, dict):
                image_input = image_input.get("pixel_values", image_input.get("input_ids"))
            if isinstance(image_input, torch.Tensor):
                image_input = image_input.squeeze(0)
        elif self.transform:
            # Torchvision transforms
            image_input = self.transform(image)
        else:
            # По умолчанию — конвертация в tensor
            import numpy as np
            image_input = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Метрики
        elapsed = (time.perf_counter() - start_time) * 1000
        self._metrics["load_time_ms"].append(elapsed)
        self._metrics["total_samples"] += 1
        
        # Формирование результата
        result = {
            "inputs": image_input,
            "image_path": str(img_path),
        }
        
        if label is not None:
            result["labels"] = torch.tensor(label, dtype=torch.long)
        
        return result
    
    @property
    def collate_fn(self) -> Callable | None:
        """Custom collate для изображений."""
        return image_collate_fn
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика работы датасета."""
        stats = super().get_stats()
        
        load_times = self._metrics["load_time_ms"]
        stats.update({
            "avg_load_time_ms": sum(load_times) / len(load_times) if load_times else 0,
            "p95_load_time_ms": sorted(load_times)[int(len(load_times) * 0.95)] if len(load_times) > 20 else 0,
            "num_classes": len(self._class_names),
            "class_names": self._class_names[:10],  # Первые 10 для отладки
        })
        
        return stats


def image_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate функция для изображений.
    
    Stack'ит изображения в батч и собирает метки.
    """
    if not batch:
        return {}
    
    # Stack изображений
    images = torch.stack([item["inputs"] for item in batch])
    
    result = {"inputs": images}
    
    # Сбор меток
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
        
        if labels[0].dim() == 0:
            # Скалярные метки (классы)
            result["labels"] = torch.stack(labels)
        else:
            # Векторные метки (multi-label)
            result["labels"] = torch.stack(labels)
    
    return result
