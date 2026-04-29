"""Text dataset implementation."""

from __future__ import annotations

import json
import mmap
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Text dataset for language model training.

    Supports JSONL, TXT, and CSV formats with optional tokenization,
    caching, and memory-mapped indexing for large files.

    Args:
        data_path: Path to the data file or directory.
        tokenizer: Optional tokenizer for on-the-fly tokenization.
        max_length: Maximum sequence length (after tokenization).
        format_fn: Optional function to transform raw text.
        cache_dir: Optional directory for tokenized cache.
        file_format: Format hint ("jsonl", "txt", "csv").
        text_column: Column name for text (CSV/JSON).
        pre_tokenize: Whether to tokenize upfront.
        use_mmap: Use memory mapping for large files.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any = None,
        max_length: int = 512,
        format_fn: Any = None,
        cache_dir: str | None = None,
        file_format: str = "jsonl",
        text_column: str = "text",
        pre_tokenize: bool = False,
        use_mmap: bool = True,
        chat_format: str | None = None,
        user_role: str = "user",
        assistant_role: str = "assistant",
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_fn = format_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.file_format = file_format
        self.text_column = text_column
        self.use_mmap = use_mmap and file_format == "jsonl"
        self.chat_format = chat_format
        self.user_role = user_role
        self.assistant_role = assistant_role

        self._file = None  # type: ignore[assignment]
        self._mmap = None  # type: ignore[assignment]
        self._index: list = []
        self._records: list = []  # type: ignore[assignment]
        self._tokenized_cache = None

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._metrics = {
            "total_samples": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "load_time_ms": deque(maxlen=1000),
        }

        if file_format == "jsonl":
            self._build_index()
        else:
            self._load_all()

        if pre_tokenize and tokenizer:
            self._preprocess()

    def _build_index(self) -> None:
        """Build line offset index for fast random access."""
        if self.use_mmap:
            self._file = open(self.data_path)
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

            pos = 0
            line_num = 0
            while True:
                line = self._mmap.readline()
                if not line:
                    break
                self._index.append((pos, len(line)))
                pos = self._mmap.tell()
                line_num += 1
        else:
            with open(self.data_path, encoding="utf-8") as f:
                pos = 0
                for line in f:
                    line_len = len(line.encode("utf-8"))
                    self._index.append((pos, line_len))
                    f.seek(pos + line_len)

    def _load_record(self, idx: int) -> dict[str, Any]:
        """Load a single record by index."""
        start, length = self._index[idx]

        if self._mmap:
            self._mmap.seek(start)
            line = self._mmap.readline().decode("utf-8").rstrip("\n")
        else:
            with open(self.data_path, encoding="utf-8") as f:
                f.seek(start)
                line = f.read(length).rstrip("\n")

        return self._parse_line(line)

    def _load_record_seek(self, idx: int) -> dict[str, Any]:
        """Load record using file seek (alternative method)."""
        start, length = self._index[idx]

        if self._file:
            self._file.seek(start)
            line = self._file.read(length).rstrip("\n")
        else:
            with open(self.data_path, encoding="utf-8") as f:
                f.seek(start)
                line = f.read(length).rstrip("\n")

        return self._parse_line(line)

    def _parse_line(self, line: str) -> dict[str, Any]:
        """Parse a line into a record dict."""
        if self.file_format == "jsonl":
            return json.loads(line)
        if self.file_format == "csv":
            return {self.text_column: line}
        return {"text": line}

    def _load_all(self) -> None:
        """Load all records into memory (for small files)."""
        self._records: list[dict[str, Any]] = []

        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                record = self._parse_line(line.rstrip("\n"))
                self._records.append(record)

        self._metrics["total_samples"] = len(self._records)

    def _preprocess(self) -> None:
        """Pre-tokenize and cache all data."""
        if not self.tokenizer or self._tokenized_cache:
            return

        cache_path = self.cache_dir / f"{self.data_path.stem}_tokenized.pt"

        if cache_path.exists():
            self._tokenized_cache = torch.load(cache_path)
            return

        if not self._records:
            raise ValueError("Cannot pre-tokenize without loading records first")

        tokenized = self.tokenizer(
            [r.get(self.text_column, r.get("text", "")) for r in self._records],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        if self.cache_dir:
            torch.save(tokenized, cache_path)

        self._tokenized_cache = tokenized

    def __len__(self) -> int:
        if self._tokenized_cache:
            return self._tokenized_cache["input_ids"].shape[0]
        if self._records:
            return len(self._records)
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        start_time = time.perf_counter()

        if self._tokenized_cache:
            return {
                "input_ids": self._tokenized_cache["input_ids"][idx],
                "attention_mask": self._tokenized_cache["attention_mask"][idx],
                "labels": self._tokenized_cache["input_ids"][idx],
            }

        record = self._load_record(idx)

        # Auto-detect chat format if not specified
        chat_format = self.chat_format
        if chat_format is None:
            chat_format = self._detect_chat_format(record)

        # Apply chat format conversion
        if chat_format and self.tokenizer:
            text = self._format_chat(record, chat_format)
        else:
            text = record.get(self.text_column, record.get("text", ""))

        if self.format_fn:
            text = self.format_fn(text)

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
            }

        elapsed = (time.perf_counter() - start_time) * 1000
        self._metrics["load_time_ms"].append(elapsed)

        return {"text": text, "labels": text}

    def validate(self) -> None:
        """Validate the dataset."""
        if len(self) == 0:
            raise ValueError("Dataset is empty")

        sample = self[0]
        if not sample:
            raise ValueError("Sample is empty")

    def get_stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        load_times = self._metrics["load_time_ms"]  # type: ignore[attr-defined]

        if load_times:
            sorted_times = sorted(load_times)  # type: ignore[arg-type]
            p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 0 else 0  # type: ignore[arg-type]
        else:
            p95 = 0

        return {
            "total_samples": len(self),
            "cache_hits": self._metrics["cache_hits"],
            "cache_misses": self._metrics["cache_misses"],
            "avg_load_time_ms": sum(load_times) / len(load_times) if load_times else 0,
            "p95_load_time_ms": p95,
        }

    def _detect_chat_format(self, record: dict) -> str | None:
        """Auto-detect chat dataset format."""
        keys = set(record.keys())

        # Alpaca: instruction + input + output
        if "instruction" in keys:
            return "alpaca"

        # ShareGPT: conversations array
        if "conversations" in keys:
            return "sharegpt"

        # Messages format: messages array
        if "messages" in keys:
            return "messages"

        return None

    def _format_chat(self, record: dict, chat_format: str) -> str:
        """Convert chat format to text for training."""
        if chat_format == "alpaca":
            instruction = record.get("instruction", "")
            input_ = record.get("input", "")
            output = record.get("output", "")

            if input_:
                text = f"Instruction: {instruction}\nInput: {input_}\nOutput: {output}"
            else:
                text = f"Instruction: {instruction}\nOutput: {output}"
            return text

        if chat_format == "sharegpt":
            conversations = record.get("conversations", [])
            parts = []
            for msg in conversations:
                role = msg.get("from", msg.get("role", ""))
                value = msg.get("value", msg.get("content", ""))
                if role == "human":
                    parts.append(f"User: {value}")
                else:
                    parts.append(f"Assistant: {value}")
            return "\n".join(parts)

        if chat_format == "messages":
            messages = record.get("messages", [])
            parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == self.user_role:
                    parts.append(f"User: {content}")
                elif role == self.assistant_role:
                    parts.append(f"Assistant: {content}")
                elif role == "system":
                    parts.append(f"System: {content}")
            return "\n".join(parts)

        # Default: return as text
        return record.get(self.text_column, record.get("text", ""))

    def __del__(self) -> None:
        """Clean up resources."""
        if self._mmap:
            try:
                self._mmap.close()
            except Exception:
                pass
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass


class HFTextDataset(Dataset):
    """HuggingFace dataset wrapper.

    Provides a unified interface for HuggingFace datasets with
    automatic tokenization and caching.

    Args:
        dataset_name: HF dataset name or path.
        tokenizer: Tokenizer for processing.
        max_length: Maximum sequence length.
        cache_dir: Cache directory.
        streaming: Use streaming mode.
        text_column: Text column name.
        format_fn: Optional formatting function.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any = None,
        max_length: int = 512,
        cache_dir: str | None = None,
        streaming: bool = False,
        text_column: str = "text",
        format_fn: Any = None,
    ) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.format_fn = format_fn

        try:
            from datasets import load_dataset

            self._dataset = load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                streaming=streaming,
            )
            if hasattr(self._dataset, "split"):
                self._dataset = self._dataset["train"]

        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

    def __len__(self) -> int:
        if hasattr(self._dataset, "__len__"):
            return len(self._dataset)
        return 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._dataset[idx]

        text = record.get(self.text_column, "")
        if self.format_fn:
            text = self.format_fn(text)

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }

        return {"text": text}
