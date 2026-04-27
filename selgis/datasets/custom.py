"""
Wrapper for custom user datasets.

Allows using any custom Dataset with Selgis,
automatically adapting them to a unified interface.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from selgis.datasets.base import BaseDataset


class CustomDataset(BaseDataset):
    """
    Wrapper for custom user datasets.
    
    Automatically wraps any PyTorch Dataset in Selgis interface:
    - __getitem__ should return dict or will be wrapped
    - Supports custom collate functions
    
    Example usage:
        # Custom dataset
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
        
        # Wrapper for Selgis
        custom_dataset = CustomDataset(
            dataset=MyDataset(),
        )
        
        # Or with custom keys
        custom_dataset = CustomDataset(
            dataset=MyDataset(),
            wrap_key="features",  # If __getitem__ returns non-dict
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
        
        # Check length
        try:
            length = len(dataset)
            if length == 0:
                print("[WARN] CustomDataset is empty!")
        except (TypeError, AttributeError):
            print("[WARN] CustomDataset does not support len()")
    
    def __len__(self) -> int:
        try:
            return len(self.dataset)
        except (TypeError, AttributeError):
            # For IterableDataset
            return 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get element and adapt to Selgis interface."""
        item = self.dataset[idx]
        
        # If element is already dict - return as is
        if isinstance(item, dict):
            return item
        
        # Wrap in dict
        return {
            self.wrap_key: item,
        }
    
    def __iter__(self):
        """Iterator for IterableDataset."""
        if hasattr(self.dataset, '__iter__'):
            for item in self.dataset:
                if isinstance(item, dict):
                    yield item
                else:
                    yield {self.wrap_key: item}
        else:
            # Fallback for regular Dataset
            for i in range(len(self)):
                yield self[i]
    
    @property
    def collate_fn(self) -> Callable | None:
        """Return dataset collate function."""
        return self._collate_fn or getattr(self.dataset, 'collate_fn', None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Dataset statistics."""
        stats = super().get_stats()
        stats.update({
            "wrapped_dataset_type": self.dataset.__class__.__name__,
            "wrap_key": self.wrap_key,
            "label_key": self.label_key,
        })
        return stats
