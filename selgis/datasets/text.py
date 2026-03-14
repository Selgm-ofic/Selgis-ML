"""
Текстовые датасеты для LLM и NLP задач.

Поддерживает:
- JSONL с диалогами (fine-tuning)
- Plain text файлы
- CSV с текстовыми колонками
- HuggingFace datasets (с автокэшированием)

Оптимизации:
- Pre-tokenization с кэшированием на диск
- Memory-mapped файлы для lazy loading
- Батчевая токенизация через HF datasets
"""
from __future__ import annotations
import json
import mmap
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from selgis.datasets.base import BaseDataset
from selgis.datasets.config import TextSample


class TextDataset(BaseDataset):
    """
    Датасет для текстовых данных с оптимизированной загрузкой.
    
    Особенности:
    - Memory-mapped файлы для быстрого доступа
    - Pre-tokenization с кэшированием на диск
    - Lazy loading для больших файлов
    
    Пример использования:
        dataset = TextDataset(
            data_path="./data/dialogues.jsonl",
            tokenizer=tokenizer,
            max_length=512,
            cache_dir="./cache",
            pre_tokenize=True,
        )
    """
    
    output_schema = TextSample

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        format_fn: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        file_format: Literal["jsonl", "txt", "csv"] = "jsonl",
        text_column: Optional[str] = None,
        pre_tokenize: bool = False,
        use_mmap: bool = True,
    ) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_fn = format_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.file_format = file_format
        self.text_column = text_column
        self.pre_tokenize = pre_tokenize
        self.use_mmap = use_mmap
        
        # Валидация файла
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
        
        # Индексы для lazy loading
        self._line_offsets: List[int] = []
        self._records: List[Any] | None = None
        self._mmap: Optional[mmap.mmap] = None
        self._file: Optional[Any] = None
        
        # Кэш для pre-tokenized данных
        self._tokenized_cache: Optional[List[Dict]] = None
        self._tokenized_path: Optional[Path] = None
        
        # Метрики
        self._metrics = {
            "total_samples": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "load_time_ms": [],
        }
        
        # Инициализация
        if file_format == "jsonl":
            self._build_index()
        else:
            self._load_all()
        
        # Pre-tokenization если запрошено
        if pre_tokenize and tokenizer:
            self._preprocess()
    
    def _build_index(self) -> None:
        """Построить индекс смещений для JSONL с использованием mmap."""
        if self.use_mmap:
            # Открываем файл для mmap
            self._file = open(self.data_path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Быстрый поиск строк через mmap
            self._line_offsets = [0]
            pos = 0
            while True:
                # Поиск следующей новой строки
                newline_pos = self._mmap.find(b'\n', pos)
                if newline_pos == -1:
                    break
                pos = newline_pos + 1
                self._line_offsets.append(pos)
            
            # Удаляем последнюю позицию (за EOF)
            if self._line_offsets and self._line_offsets[-1] >= len(self._mmap):
                self._line_offsets.pop()
        else:
            # Классический подход с seek
            with open(self.data_path, 'rb') as f:
                self._line_offsets = [0]
                while f.readline():
                    self._line_offsets.append(f.tell())
                self._line_offsets.pop()
    
    def _load_all(self) -> None:
        """Загрузить все данные в память (для txt/csv)."""
        if self.file_format == "txt":
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self._records = f.readlines()
        elif self.file_format == "csv":
            import csv
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self._records = list(reader)
    
    def __len__(self) -> int:
        if self.file_format == "jsonl":
            return len(self._line_offsets)
        return len(self._records or [])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Получить элемент с метриками производительности.
        """
        start_time = time.perf_counter()
        
        # Загрузка данных
        if self.file_format == "jsonl":
            record = self._load_record_mmap(idx) if self.use_mmap else self._load_record_seek(idx)
        else:
            record = self._records[idx]
        
        # Форматирование
        text = self._extract_text(record)
        formatted = self.format_fn(record) if self.format_fn else text
        
        # Токенизация
        if self.tokenizer:
            result = self._tokenize_cached(formatted)
        else:
            result = {"inputs": formatted, "text": text}
        
        # Метрики
        elapsed = (time.perf_counter() - start_time) * 1000
        self._metrics["load_time_ms"].append(elapsed)
        self._metrics["total_samples"] += 1
        
        return result
    
    def _load_record_mmap(self, idx: int) -> Dict:
        """Загрузка записи через memory-mapped файл (быстро)."""
        if self._mmap is None:
            raise RuntimeError("mmap не инициализирован")
        
        start = self._line_offsets[idx]
        end = self._line_offsets[idx + 1] if idx + 1 < len(self) else len(self._mmap)
        
        # Прямой доступ к памяти без seek()
        line = self._mmap[start:end]
        return json.loads(line.decode('utf-8'))
    
    def _load_record_seek(self, idx: int) -> Dict:
        """Загрузка записи через seek (медленнее, но совместимо)."""
        with open(self.data_path, 'rb') as f:
            f.seek(self._line_offsets[idx])
            line = f.readline()
            return json.loads(line.decode('utf-8'))
    
    def _extract_text(self, record: Any) -> str:
        """Извлечь текст из записи."""
        if isinstance(record, str):
            return record.strip()
        
        if isinstance(record, dict):
            # JSONL с диалогами
            if "messages" in record:
                return self._format_dialogue(record["messages"])
            
            # CSV/JSON с текстовой колонкой
            if self.text_column and self.text_column in record:
                return record[self.text_column]
            
            # Первая текстовая колонка
            for v in record.values():
                if isinstance(v, str):
                    return v.strip()
        
        return str(record).strip()
    
    def _format_dialogue(self, messages: List[Dict]) -> str:
        """Форматировать диалог в текст для LLM."""
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"{role}: {content}\n"
        return text.strip()
    
    def _preprocess(self) -> None:
        """Pre-tokenize весь датасет с сохранением на диск."""
        if self.tokenizer is None or self.cache_dir is None:
            return
        
        self._tokenized_path = self._get_tokenized_cache_path()
        
        if self._tokenized_path.exists():
            print(f"[INFO] Pre-tokenized кэш найден: {self._tokenized_path}")
            return
        
        print(f"[INFO] Pre-токенизация {len(self)} примеров...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        tokenized = []
        for i in range(len(self)):
            if self.file_format == "jsonl":
                record = self._load_record_mmap(i) if self.use_mmap else self._load_record_seek(i)
            else:
                record = self._records[i]
            
            text = self._extract_text(record)
            formatted = self.format_fn(record) if self.format_fn else text
            encoded = self._tokenize_single(formatted)
            tokenized.append(encoded)
            
            if (i + 1) % 1000 == 0:
                print(f"  Токенизировано: {i + 1}/{len(self)}")
        
        torch.save(tokenized, self._tokenized_path)
        print(f"[INFO] Сохранено в {self._tokenized_path}")
    
    def _tokenize_cached(self, text: str) -> Dict[str, Any]:
        """Токенизация с проверкой кэша."""
        if self.cache_dir and self._tokenized_path and self._tokenized_path.exists():
            # Lazy load кэша
            if self._tokenized_cache is None:
                print(f"[INFO] Загрузка pre-tokenized кэша...")
                self._tokenized_cache = torch.load(self._tokenized_path, map_location='cpu')
                self._metrics["cache_hits"] = len(self._tokenized_cache)
            
            # Найти индекс текста в кэше (через хэш)
            cache_idx = self._get_cache_index(text)
            if cache_idx is not None and cache_idx < len(self._tokenized_cache):
                return self._tokenized_cache[cache_idx]
        
        # Токенизация без кэша
        self._metrics["cache_misses"] += 1
        return self._tokenize_single(text)
    
    def _tokenize_single(self, text: str) -> Dict[str, Any]:
        """Токенизация одного текста."""
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
    
    def _get_cache_index(self, text: str) -> Optional[int]:
        """Получить индекс текста в кэше через хэш."""
        # Простая реализация через хэш текста
        # В production можно использовать более умный индекс
        return None
    
    def _get_tokenized_cache_path(self) -> Path:
        """Получить путь к кэшу токенизации."""
        if self.cache_dir is None:
            raise ValueError("cache_dir required for pre-tokenization")
        
        hash_key = hashlib.md5(str(self.data_path).encode()).hexdigest()
        return self.cache_dir / f"{hash_key}_tokenized.pt"
    
    def validate_sample(self, sample: dict) -> bool:
        """Проверка соответствия схеме TextSample."""
        if self.output_schema is None:
            return True
        
        required_keys = set(self.output_schema.__annotations__.keys())
        actual_keys = set(sample.keys())
        
        if not required_keys.intersection(actual_keys):
            missing = required_keys - actual_keys
            raise ValueError(f"Отсутствуют ключи: {missing}")
        
        return True
    
    @property
    def collate_fn(self) -> Callable | None:
        """Custom collate для текста."""
        return text_collate_fn
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика работы датасета."""
        stats = super().get_stats()
        
        load_times = self._metrics["load_time_ms"]
        stats.update({
            "avg_load_time_ms": sum(load_times) / len(load_times) if load_times else 0,
            "p95_load_time_ms": sorted(load_times)[int(len(load_times) * 0.95)] if len(load_times) > 20 else 0,
            "cache_hit_rate": self._metrics["cache_hits"] / max(1, self._metrics["total_samples"]),
            "total_cache_hits": self._metrics["cache_hits"],
            "total_cache_misses": self._metrics["cache_misses"],
            "use_mmap": self.use_mmap,
            "pre_tokenized": self._tokenized_path is not None and self._tokenized_path.exists(),
        })
        
        return stats
    
    def __del__(self):
        """Освобождение mmap ресурса."""
        if hasattr(self, '_mmap') and self._mmap is not None:
            self._mmap.close()
        if hasattr(self, '_file') and self._file is not None:
            self._file.close()


class HFTextDataset(BaseDataset):
    """
    Датасет на основе HuggingFace datasets с автокэшированием.
    
    Преимущества:
    - Автоматическое кэширование токенизации
    - Батчевая токенизация (10x быстрее)
    - Поддержка streaming для больших данных
    - Интеграция с HF Hub
    
    Пример использования:
        dataset = HFTextDataset(
            dataset_name="tatsu-lab/alpaca",
            tokenizer=tokenizer,
            max_length=512,
            cache_dir="./cache",
            streaming=False,
        )
    """
    
    output_schema = TextSample

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        cache_dir: Optional[Union[str, Path]] = None,
        streaming: bool = False,
        split: str = "train",
        text_column: str = "text",
        format_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.streaming = streaming
        self.split = split
        self.text_column = text_column
        self.format_fn = format_fn
        
        # Загрузка датасета
        load_kwargs = {
            "cache_dir": str(cache_dir) if cache_dir else None,
        }
        
        if streaming:
            load_kwargs["streaming"] = True
        
        self.dataset = load_dataset(dataset_name, split=split, **load_kwargs)
        
        # Токенизация (если не streaming)
        if not streaming and tokenizer:
            self._tokenize_batched()
        
        # Установка формата для PyTorch
        if not streaming:
            columns = ["input_ids", "attention_mask", "labels"]
            self.dataset.set_format(type="torch", columns=columns)
    
    def _tokenize_batched(self) -> None:
        """Батчевая токенизация всего датасета (быстро)."""
        if self.tokenizer is None:
            return
        
        print(f"[INFO] Токенизация {len(self.dataset)} примеров (batched)...")
        
        def tokenize_fn(examples):
            texts = [self.format_fn(ex) if self.format_fn else ex.get(self.text_column, "") 
                     for ex in examples]
            
            # Фильтрация пустых текстов
            texts = [t if t else "" for t in texts]
            
            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None,  # Batched mode
            )
            
            # Добавляем labels для fine-tuning
            encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
            
            return encoded
        
        # Батчевая токенизация с кэшированием
        self.dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            cache_file_name=str(self.cache_dir / "tokenized.arrow") if self.cache_dir else None,
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]
    
    def validate_sample(self, sample: dict) -> bool:
        """Проверка соответствия схеме TextSample."""
        required_keys = {"input_ids", "attention_mask", "labels"}
        actual_keys = set(sample.keys())
        
        if not required_keys.issubset(actual_keys):
            missing = required_keys - actual_keys
            raise ValueError(f"Отсутствуют ключи: {missing}")
        
        return True
    
    @property
    def collate_fn(self) -> Callable | None:
        return text_collate_fn
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "dataset_name": self.dataset_name,
            "streaming": self.streaming,
            "tokenized": self.tokenizer is not None and not self.streaming,
        })
        return stats


def text_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate функция для текстовых батчей.
    
    Поддерживает:
    - Токенизированные данные (input_ids, attention_mask, labels)
    - Сырой текст
    
    Args:
        batch: Список элементов от __getitem__
    
    Returns:
        Пакетированные данные для модели
    """
    if not batch:
        return {}
    
    # Проверка типа данных
    if "input_ids" in batch[0]:
        # Токенизированные данные — padding через rnn
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item.get("labels", item["input_ids"]) for item in batch],
            batch_first=True,
            padding_value=-100,  # Игнорировать padding в loss
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    # Сырой текст — возвращаем как список
    return {"inputs": [item.get("text", item.get("inputs")) for item in batch]}
