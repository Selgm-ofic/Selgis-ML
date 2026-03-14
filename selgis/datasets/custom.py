"""
Обёртка для кастомных датасетов пользователя.

Позволяет использовать любые пользовательские Dataset с Selgis,
автоматически приводя их к единому интерфейсу.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from selgis.datasets.base import BaseDataset


class CustomDataset(BaseDataset):
    """
    Обёртка для кастомных датасетов пользователя.
    
    Автоматически оборачивает любой PyTorch Dataset в интерфейс Selgis:
    - __getitem__ должен возвращать dict или будет обёрнут
    - Поддерживает кастомные collate функции
    
    Пример использования:
        # Пользовательский датасет
        class MyDataset(Dataset):
            def __init__(self):
                self.data = [...]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {
                    "inputs": torch.tensor(...),
                    "labels": torch.tensor(...),
                }
        
        # Обёртка для Selgis
        custom_dataset = CustomDataset(
            dataset=MyDataset(),
        )
        
        # Или с кастомными ключами
        custom_dataset = CustomDataset(
            dataset=MyDataset(),
            wrap_key="features",  # Если __getitem__ возвращает не dict
            label_key="target",
        )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        wrap_key: str = "inputs",
        label_key: str = "labels",
        collate_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        self.dataset = dataset
        self.wrap_key = wrap_key
        self.label_key = label_key
        self._collate_fn = collate_fn
        
        # Проверка длины
        try:
            length = len(dataset)
            if length == 0:
                print("[WARN] CustomDataset пустой!")
        except (TypeError, AttributeError):
            print("[WARN] CustomDataset не поддерживает len()")
    
    def __len__(self) -> int:
        try:
            return len(self.dataset)
        except (TypeError, AttributeError):
            # Для IterableDataset
            return 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Получить элемент и привести к интерфейсу Selgis."""
        item = self.dataset[idx]
        
        # Если элемент уже dict — возвращаем как есть
        if isinstance(item, dict):
            return item
        
        # Обёртка в dict
        return {
            self.wrap_key: item,
        }
    
    def __iter__(self):
        """Итератор для IterableDataset."""
        if hasattr(self.dataset, '__iter__'):
            for item in self.dataset:
                if isinstance(item, dict):
                    yield item
                else:
                    yield {self.wrap_key: item}
        else:
            # Fallback для обычных Dataset
            for i in range(len(self)):
                yield self[i]
    
    @property
    def collate_fn(self) -> Callable | None:
        """Вернуть collate функцию датасета."""
        return self._collate_fn or getattr(self.dataset, 'collate_fn', None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика датасета."""
        stats = super().get_stats()
        stats.update({
            "wrapped_dataset_type": self.dataset.__class__.__name__,
            "wrap_key": self.wrap_key,
            "label_key": self.label_key,
        })
        return stats
