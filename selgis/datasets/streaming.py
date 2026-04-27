"""
Streaming datasets for large files (>100GB).

Inherit from IterableDataset for proper DataLoader integration:
- Support num_workers > 0
- No indexing required
- Data loaded as stream

Optimizations:
- Data splitting between workers
- Buffering to reduce I/O
- Support for compressed files (.gz, .zip)
"""

from __future__ import annotations
import csv
import gzip
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import torch
from torch.utils.data import IterableDataset

from selgis.datasets.base import StreamingDataset


class StreamingTextDataset(StreamingDataset):
    """
    Streaming dataset for large text files.

    Features:
    - Does not load entire file into RAM
    - Reads one line at a time
    - Supports splitting between workers
    - Automatic file closing

    Important:
        __iter__ returns self, which means:
        - NOT thread-safe: do not use one object from multiple threads
        - DO NOT create multiple iterators from one object in one process
        - Repeated calls to iter(dataset) will reset the first iterator's state
        - For parallel processing, use num_workers in DataLoader

    Example usage:
        dataset = StreamingTextDataset(
            data_path="./data/huge_dataset.jsonl",
            tokenizer=tokenizer,
            max_length=512,
            buffer_size=1000,
        )

        loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        buffer_size: int = 1000,
        format_fn: Optional[Callable] = None,
        total_lines: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.format_fn = format_fn
        self._total_lines = total_lines

        # Iteration state
        self._file = None
        self._buffer: Deque[str] = deque()
        self._line_count = 0
        self._global_line_idx = 0

        # For splitting between workers
        self._worker_id = 0
        self._num_workers = 1

        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

    def __iter__(self):
        """
        Iterator with multi-processing support.

        Each worker gets its own portion of data.
        """
        # Get worker information
        worker_info = torch.utils.data.get_worker_info()

        # Reset state
        self._buffer = deque()
        self._line_count = 0

        # Open new file for each worker - close old one if exists
        self._close_file()

        # Worker information
        if worker_info is None:
            self._worker_id = 0
            self._num_workers = 1
        else:
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers

        # Open file (compression support)
        if self.data_path.suffix == ".gz":
            try:
                self._file = gzip.open(self.data_path, "rt", encoding="utf-8")
            except Exception as e:
                raise RuntimeError(f"Failed to open file {self.data_path}: {e}") from None
        else:
            try:
                self._file = open(self.data_path, "r", encoding="utf-8")
            except Exception as e:
                raise RuntimeError(f"Failed to open file {self.data_path}: {e}") from None

        self._global_line_idx = 0

        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next element from stream."""
        if self._file is None:
            raise StopIteration

        # Fill buffer if empty
        if not self._buffer:
            self._fill_buffer()

        # Buffer empty and file read to end
        if not self._buffer:
            self._close_file()
            raise StopIteration

        # Get element from buffer
        text = self._buffer.popleft()
        self._line_count += 1

        # Tokenization
        if self.tokenizer:
            return self._tokenize(text)

        return {"inputs": text, "text": text}

    def _fill_buffer(self) -> None:
        """Fill buffer with next lines."""
        if self._file is None:
            return

        self._buffer = deque()

        for _ in range(self.buffer_size):
            line = self._file.readline()
            if not line:
                break
            line_idx = self._global_line_idx
            self._global_line_idx += 1
            if self._num_workers > 1 and (line_idx % self._num_workers != self._worker_id):
                continue

            try:
                # Parse JSONL
                record = json.loads(line)

                # Formatting
                if self.format_fn:
                    text = self.format_fn(record)
                else:
                    text = record.get("text", record.get("content", ""))

                if text:
                    self._buffer.append(text)

            except json.JSONDecodeError:
                # Skip invalid lines
                continue

    def _tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required")

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone(),
        }

    def _close_file(self) -> None:
        """Close file."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        """Guaranteed file closing."""
        self._close_file()

    def set_total_length(self, length: int) -> None:
        """Set known dataset length."""
        self._total_lines = length
        super().set_total_length(length)

    def get_stats(self) -> Dict[str, Any]:
        """Streaming dataset statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "data_path": str(self.data_path),
                "buffer_size": self.buffer_size,
                "lines_read": self._line_count,
            }
        )
        return stats


class StreamingCSVDataset(StreamingDataset):
    """
    Streaming dataset for large CSV files.

    Important:
        __iter__ returns self, which means:
        - NOT thread-safe: do not use one object from multiple threads
        - DO NOT create multiple iterators from one object in one process
        - Repeated calls to iter(dataset) will reset the first iterator's state
        - For parallel processing, use num_workers in DataLoader

    Example usage:
        dataset = StreamingCSVDataset(
            data_path="./data/huge_dataset.csv",
            text_column="review",
            label_column="sentiment",
        )
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        text_column: str = "text",
        label_column: Optional[str] = None,
        buffer_size: int = 1000,
        format_fn: Optional[Callable] = None,
        total_lines: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.text_column = text_column
        self.label_column = label_column
        self.buffer_size = buffer_size
        self.format_fn = format_fn
        self._total_lines = total_lines

        self._file = None
        self._buffer: Deque[Dict[str, Any]] = deque()
        self._line_count = 0
        self._headers: Optional[List[str]] = None
        self._reader: Optional[csv.DictReader] = None
        self._global_line_idx = 0
        self._worker_id = 0
        self._num_workers = 1

        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

    def __iter__(self):
        """Iterator with multi-processing support."""
        worker_info = torch.utils.data.get_worker_info()

        # Reset state
        self._buffer = deque()
        self._line_count = 0

        # Close old file before opening new one
        self._close_file()
        self._reader = None

        if worker_info is None:
            self._worker_id = 0
            self._num_workers = 1
        else:
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers

        # Open CSV file with error handling
        try:
            self._file = open(self.data_path, "r", encoding="utf-8", newline="")
            self._reader = csv.DictReader(self._file)
            self._headers = self._reader.fieldnames
        except Exception as e:
            raise RuntimeError(f"Failed to open CSV file {self.data_path}: {e}") from None

        self._global_line_idx = 0

        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next element."""
        if self._file is None:
            raise StopIteration

        if not self._buffer:
            self._fill_buffer()

        if not self._buffer:
            self._close_file()
            raise StopIteration

        record = self._buffer.popleft()
        self._line_count += 1

        # Formatting
        if self.format_fn:
            return self.format_fn(record)

        # Extract text and label
        result = {"inputs": record.get(self.text_column, "")}

        if self.label_column and self.label_column in record:
            result["labels"] = record[self.label_column]

        return result

    def _fill_buffer(self) -> None:
        """Fill buffer with next lines."""
        if self._file is None:
            return

        self._buffer = deque()
        if self._reader is None:
            return

        for _ in range(self.buffer_size):
            try:
                record = next(self._reader)
                line_idx = self._global_line_idx
                self._global_line_idx += 1
                if self._num_workers > 1 and (line_idx % self._num_workers != self._worker_id):
                    continue
                self._buffer.append(record)
            except StopIteration:
                break

    def _close_file(self) -> None:
        """Close file."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self._close_file()

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "data_path": str(self.data_path),
                "columns": self._headers,
                "text_column": self.text_column,
                "label_column": self.label_column,
            }
        )
        return stats
