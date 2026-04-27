"""
Multimodal datasets (text + images).

Supports:
- LLaVA-style datasets (question-answer about image)
- BLIP-style datasets (image captions)
- Custom formats via format_fn

Example JSONL structure:
    {
        "image": "path/to/image.jpg",
        "text": "Image description",
        "question": "What is in the image?",
        "answer": "In the image..."
    }
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from selgis.datasets.base import BaseDataset
from selgis.datasets.config import MultimodalSample


class MultimodalDataset(BaseDataset):
    """
    Dataset for multimodal data (text + images).
    
    Features:
    - Support for LLaVA, BLIP, InstructBLIP formats
    - Custom formatting via format_fn
    - Performance metrics
    
    Example usage:
        from transformers import AutoTokenizer, AutoImageProcessor
        
        tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        image_processor = AutoImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        
        dataset = MultimodalDataset(
            data_path="./data/llava_dataset.jsonl",
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=512,
        )
    """
    
    output_schema = MultimodalSample
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        max_length: int = 512,
        cache_dir: Optional[Union[str, Path]] = None,
        format_fn: Optional[Callable] = None,
        image_root: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.format_fn = format_fn or self._default_format
        self.image_root = Path(image_root) if image_root else None
        
        # Record index
        self._records: List[Dict] = []
        
        # Metrics
        self._metrics = {
            "total_samples": 0,
            "load_time_ms": [],
            "image_load_time_ms": [],
            "text_load_time_ms": [],
        }
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        self._load_records()
    
    def _load_records(self) -> None:
        """Load record index."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line)
                    self._records.append(record)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping line {line_num}: {e}")
        
        print(f"[INFO] Loaded {len(self._records)} records")
    
    def __len__(self) -> int:
        return len(self._records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get multimodal element with metrics."""
        start_time = time.perf_counter()
        
        record = self._records[idx]
        
        # Text formatting
        text_start = time.perf_counter()
        text_input, text_target = self.format_fn(record)
        text_time = (time.perf_counter() - text_start) * 1000
        
        # Image loading
        image_start = time.perf_counter()
        image_input = self._load_image(record)
        image_time = (time.perf_counter() - image_start) * 1000
        
        # Text tokenization
        if self.tokenizer:
            text_encoded = self.tokenizer(
                text_input,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            text_input_ids = text_encoded["input_ids"].squeeze(0)
            attention_mask = text_encoded["attention_mask"].squeeze(0)
        else:
            text_input_ids = torch.tensor([])
            attention_mask = torch.tensor([])
        
        # Metrics
        elapsed = (time.perf_counter() - start_time) * 1000
        self._metrics["load_time_ms"].append(elapsed)
        self._metrics["image_load_time_ms"].append(image_time)
        self._metrics["text_load_time_ms"].append(text_time)
        self._metrics["total_samples"] += 1
        
        # Format result
        return {
            "inputs": {
                "pixel_values": image_input,
                "input_ids": text_input_ids,
                "attention_mask": attention_mask,
            },
            "labels": text_target,
            "metadata": {
                "text_input": text_input,
                "text_target": text_target,
            },
        }
    
    def _load_image(self, record: Dict) -> torch.Tensor:
        """Load and process image."""
        from PIL import Image
        
        image_path = record.get("image", "")
        
        # Add image_root if specified
        if self.image_root and not Path(image_path).is_absolute():
            image_path = self.image_root / image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Image processing
        if self.image_processor:
            image_input = self.image_processor(image, return_tensors="pt")
            if isinstance(image_input, dict):
                image_input = image_input.get("pixel_values", image_input.get("input_ids"))
            if isinstance(image_input, torch.Tensor):
                image_input = image_input.squeeze(0)
        else:
            # Default - convert to tensor
            import numpy as np
            image_input = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        return image_input
    
    def _default_format(self, record: Dict) -> Tuple[str, str]:
        """Default formatting (LLaVA-style)."""
        # LLaVA format: question + answer
        question = record.get("question", "")
        answer = record.get("answer", "")
        
        if question and answer:
            return question, answer
        
        # BLIP format: image captioning
        text = record.get("text", "")
        if text:
            return f"Describe the image:", text
        
        # Generic format
        text = record.get("text", "")
        return text, text
    
    @property
    def collate_fn(self) -> Callable | None:
        """Custom collate for multimodal data."""
        return multimodal_collate_fn
    
    def get_stats(self) -> Dict[str, Any]:
        """Dataset performance statistics."""
        stats = super().get_stats()
        
        load_times = self._metrics["load_time_ms"]
        image_times = self._metrics["image_load_time_ms"]
        text_times = self._metrics["text_load_time_ms"]
        
        stats.update({
            "avg_load_time_ms": sum(load_times) / len(load_times) if load_times else 0,
            "avg_image_load_time_ms": sum(image_times) / len(image_times) if image_times else 0,
            "avg_text_load_time_ms": sum(text_times) / len(text_times) if text_times else 0,
            "total_samples": self._metrics["total_samples"],
        })
        
        return stats


def multimodal_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for multimodal batches.
    
    Stacks images and tokens, handles variable-length sequences.
    """
    if not batch:
        return {}
    
    # Collect images
    pixel_values = [item["inputs"]["pixel_values"] for item in batch]
    
    # Stack images (if same size)
    try:
        pixel_values_batched = torch.stack(pixel_values)
    except RuntimeError:
        # Different sizes - return list
        pixel_values_batched = pixel_values
    
    # Tokenized text - padding via rnn
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["inputs"]["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["inputs"]["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    
    # Labels
    labels = [item["labels"] for item in batch]
    
    return {
        "inputs": {
            "pixel_values": pixel_values_batched,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        "labels": labels,
    }
