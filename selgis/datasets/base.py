"""
Base classes for all Selgis datasets.

Unified interface ensures compatibility with Trainer and DataLoader.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, IterableDataset


class BaseDataset(Dataset, ABC):
    """
    Base class for all Selgis datasets.
    
    Ensures unified interface for any data type:
    - Text, images, audio, tabular data
    - Multimodal data
    
    Example inheritance:
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
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one data element.
        
        Returns:
            dict with keys expected by model:
            - "inputs": input data (tensors, dict, tuple)
            - "labels": targets (optional)
            - Additional keys (e.g. "pixel_values", "metadata")
        """
        pass
    
    def validate(self) -> bool:
        """
        Validate dataset before use.
        
        Checks:
        - Dataset is not empty
        - __getitem__ returns dict
        - Has "inputs" or "input_ids" key
        
        Returns:
            True if validation passed
            
        Raises:
            ValueError: If dataset is invalid
        """
        if len(self) == 0:
            raise ValueError("Dataset is empty!")
        
        # Check first element
        try:
            sample = self[0]
        except Exception as e:
            raise ValueError(f"Error loading first sample: {e}")
        
        if not isinstance(sample, dict):
            raise ValueError(
                f"__getitem__ must return dict, got {type(sample)}"
            )
        
        # Check required keys
        required_keys = {"inputs", "input_ids"}
        actual_keys = set(sample.keys())
        
        if not required_keys.intersection(actual_keys):
            raise ValueError(
                f"Element must contain one of keys: {required_keys}. "
                f"Got keys: {list(actual_keys)}"
            )
        
        self._validated = True
        return True
    
    @property
    def collate_fn(self) -> Callable | None:
        """
        Return function for collating batches.
        
        Override in subclasses for custom logic.
        
        Returns:
            Collate_fn function for DataLoader, or None for default
        """
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return dataset statistics.
        
        Returns:
            dict with statistics (size, average load time, etc.)
        """
        return {
            "length": len(self),
            "validated": self._validated,
            "type": self.__class__.__name__,
        }


class StreamingDataset(IterableDataset, ABC):
    """
    Base class for streaming datasets.
    
    Inherits from IterableDataset, which works correctly with DataLoader:
    - Supports num_workers > 0
    - No indexing required
    - Data loaded as stream one element at a time
    
    Example usage:
        class MyStreamingDataset(StreamingDataset):
            def __init__(self, data_path):
                super().__init__()
                self.data_path = Path(data_path)
            
            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                
                # Split data between workers
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
        Return dataset size (optional for streaming).
        
        Returns:
            Size if known, otherwise 0
        """
        return self._total_length or 0
    
    @abstractmethod
    def __iter__(self):
        """
        Iterator for streaming mode.
        
        Must be implemented in subclasses.
        
        Yields:
            Data elements (dict)
        """
        pass
    
    def set_total_length(self, length: int) -> None:
        """
        Set known dataset length.
        
        Args:
            length: Number of elements
        """
        self._total_length = length
    
    def validate(self) -> bool:
        """Validate streaming dataset."""
        # Check first element via iteration
        try:
            iterator = iter(self)
            sample = next(iterator)
            
            if not isinstance(sample, dict):
                raise ValueError(
                    f"__iter__ must return dict, got {type(sample)}"
                )
            
            required_keys = {"inputs", "input_ids"}
            actual_keys = set(sample.keys())
            
            if not required_keys.intersection(actual_keys):
                raise ValueError(
                    f"Element must contain one of keys: {required_keys}. "
                    f"Got keys: {list(actual_keys)}"
                )
            
            self._validated = True
            return True
            
        except StopIteration:
            raise ValueError("StreamingDataset is empty!")
        except Exception as e:
            raise ValueError(f"Error validating streaming dataset: {e}")
    
    @property
    def collate_fn(self) -> Callable | None:
        """Collate function for streaming datasets."""
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Streaming dataset statistics."""
        return {
            "length": self._total_length if self._total_length else "unknown",
            "type": self.__class__.__name__,
            "streaming": True,
        }
