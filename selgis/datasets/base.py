"""
Базовые классы для всех датасетов Selgis.

Единый интерфейс гарантирует совместимость с Trainer и DataLoader.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, IterableDataset


class BaseDataset(Dataset, ABC):
    """
    Базовый класс для всех датасетов Selgis.
    
    Гарантирует единый интерфейс для любого типа данных:
    - Текст, изображения, аудио, табличные данные
    - Мультимодальные данные
    
    Пример наследования:
        class MyDataset(BaseDataset):
            def __init__(self, data_path):
                super().__init__()
                self.data_path = Path(data_path)
                self._build_index()
            
            def __len__(self):
                return len(self._index)
            
            def __getitem__(self, idx):
                item = self._load_item(idx)
                return {
                    "inputs": item["feature"],
                    "labels": item["target"],
                }
    """
    
    def __init__(self) -> None:
        self._validated = False
    
    @abstractmethod
    def __len__(self) -> int:
        """Вернуть размер датасета."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Вернуть один элемент данных.
        
        Returns:
            dict с ключами, которые ожидает модель:
            - "inputs": входные данные (тензоры, dict, tuple)
            - "labels": таргеты (опционально)
            - Дополнительные ключи (напр. "pixel_values", "metadata")
        """
        pass
    
    def validate(self) -> bool:
        """
        Валидировать датасет перед использованием.
        
        Проверяет:
        - Датасет не пустой
        - __getitem__ возвращает dict
        - Есть ключ "inputs" или "input_ids"
        
        Returns:
            True если валидация прошла успешно
            
        Raises:
            ValueError: Если датасет некорректен
        """
        if len(self) == 0:
            raise ValueError("Dataset is empty!")
        
        # Проверка первого элемента
        try:
            sample = self[0]
        except Exception as e:
            raise ValueError(f"Error loading first sample: {e}")
        
        if not isinstance(sample, dict):
            raise ValueError(
                f"__getitem__ должен возвращать dict, получил {type(sample)}"
            )
        
        # Проверка обязательных ключей
        required_keys = {"inputs", "input_ids"}
        actual_keys = set(sample.keys())
        
        if not required_keys.intersection(actual_keys):
            raise ValueError(
                f"Элемент должен содержать один из ключей: {required_keys}. "
                f"Получены ключи: {list(actual_keys)}"
            )
        
        self._validated = True
        return True
    
    @property
    def collate_fn(self) -> Callable | None:
        """
        Вернуть функцию для collate батчей.
        
        Переопределите в подклассах для кастомной логики.
        
        Returns:
            Функция collate_fn для DataLoader, или None для стандартной
        """
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Вернуть статистику датасета.
        
        Returns:
            dict со статистикой (размер, среднее время загрузки, и т.д.)
        """
        return {
            "length": len(self),
            "validated": self._validated,
            "type": self.__class__.__name__,
        }


class StreamingDataset(IterableDataset, ABC):
    """
    Базовый класс для streaming датасетов.
    
    Наследуется от IterableDataset, что корректно работает с DataLoader:
    - Поддерживает num_workers > 0
    - Не требует индексации
    - Данные загружаются потоком по одному элементу
    
    Пример использования:
        class MyStreamingDataset(StreamingDataset):
            def __init__(self, data_path):
                super().__init__()
                self.data_path = Path(data_path)
            
            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                
                # Разделение данных между воркерами
                if worker_info is None:
                    start, end = 0, float('inf')
                else:
                    per_worker = self.total_lines // worker_info.num_workers
                    start = worker_info.id * per_worker
                    end = start + per_worker
                
                with open(self.data_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i < start:
                            continue
                        if i >= end:
                            break
                        
                        record = json.loads(line)
                        yield self._process(record)
    """
    
    def __init__(self) -> None:
        self._total_length: Optional[int] = None
    
    def __len__(self) -> int:
        """
        Вернуть размер датасета (опционально для streaming).
        
        Returns:
            Размер если известен, иначе 0
        """
        return self._total_length or 0
    
    @abstractmethod
    def __iter__(self):
        """
        Итератор для streaming режима.
        
        Должен быть реализован в подклассах.
        
        Yields:
            Элементы данных (dict)
        """
        pass
    
    def set_total_length(self, length: int) -> None:
        """
        Установить известную длину датасета.
        
        Args:
            length: Количество элементов
        """
        self._total_length = length
    
    def validate(self) -> bool:
        """Валидация streaming датасета."""
        # Проверка первого элемента через итерацию
        try:
            iterator = iter(self)
            sample = next(iterator)
            
            if not isinstance(sample, dict):
                raise ValueError(
                    f"__iter__ должен возвращать dict, получил {type(sample)}"
                )
            
            required_keys = {"inputs", "input_ids"}
            actual_keys = set(sample.keys())
            
            if not required_keys.intersection(actual_keys):
                raise ValueError(
                    f"Элемент должен содержать один из ключей: {required_keys}. "
                    f"Получены ключи: {list(actual_keys)}"
                )
            
            self._validated = True
            return True
            
        except StopIteration:
            raise ValueError("StreamingDataset пустой!")
        except Exception as e:
            raise ValueError(f"Error validating streaming dataset: {e}")
    
    @property
    def collate_fn(self) -> Callable | None:
        """Collate функция для streaming датасетов."""
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика streaming датасета."""
        return {
            "length": self._total_length if self._total_length else "unknown",
            "type": self.__class__.__name__,
            "streaming": True,
        }
