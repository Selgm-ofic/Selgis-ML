"""
Streaming датасеты для больших файлов (>100GB).

Наследуются от IterableDataset для корректной работы с DataLoader:
- Поддерживают num_workers > 0
- Не требуют индексации
- Данные загружаются потоком

Оптимизации:
- Разделение данных между воркерами
- Буферизация для уменьшения I/O
- Поддержка сжатых файлов (.gz, .zip)
"""
from __future__ import annotations
import json
import gzip
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from selgis.datasets.base import StreamingDataset


class StreamingTextDataset(StreamingDataset):
    """
    Streaming датасет для больших текстовых файлов.
    
    Особенности:
    - Не загружает весь файл в RAM
    - Читает по одной строке за раз
    - Поддерживает разделение между воркерами
    - Автоматическое закрытие файлов
    
    Пример использования:
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
        
        # Состояние для итерации
        self._file = None
        self._buffer: List[str] = []
        self._line_count = 0
        
        # Для разделения между воркерами
        self._worker_start = 0
        self._worker_end = float('inf')
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
    
    def __iter__(self):
        """
        Итератор с поддержкой multi-processing.
        
        Каждый воркер получает свою часть данных.
        """
        # Получаем информацию о воркере
        worker_info = torch.utils.data.get_worker_info()
        
        # Сброс состояния
        self._buffer = []
        self._line_count = 0
        
        # Открываем новый файл для каждого воркера
        if self._file is not None:
            try:
                self._file.close()
            except:
                pass
        
        # Определение диапазона для этого воркера
        if worker_info is None:
            # Один процесс
            self._worker_start = 0
            self._worker_end = float('inf')
        else:
            # Разделение между воркерами
            if self._total_lines:
                per_worker = self._total_lines // worker_info.num_workers
                self._worker_start = worker_info.id * per_worker
                self._worker_end = self._worker_start + per_worker
            else:
                # Если длина неизвестна, каждый воркер читает свои строки
                self._worker_start = 0
                self._worker_end = float('inf')
        
        # Открытие файла (поддержка сжатия)
        if self.data_path.suffix == '.gz':
            self._file = gzip.open(self.data_path, 'rt', encoding='utf-8')
        else:
            self._file = open(self.data_path, 'r', encoding='utf-8')
        
        # Пропуск строк до начала диапазона воркера
        if self._worker_start > 0:
            for _ in range(self._worker_start):
                line = self._file.readline()
                if not line:
                    break
        
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """Получить следующий элемент из потока."""
        if self._file is None:
            raise StopIteration
        
        # Заполнение буфера если пуст
        if not self._buffer:
            self._fill_buffer()
        
        # Буфер пуст и файл прочитан до конца
        if not self._buffer:
            self._close_file()
            raise StopIteration
        
        # Получение элемента из буфера
        text = self._buffer.pop(0)
        self._line_count += 1
        
        # Токенизация
        if self.tokenizer:
            return self._tokenize(text)
        
        return {"inputs": text, "text": text}
    
    def _fill_buffer(self) -> None:
        """Заполнить буфер следующими строками."""
        if self._file is None:
            return
        
        self._buffer = []
        
        for _ in range(self.buffer_size):
            # Проверка достижения конца диапазона воркера
            if self._line_count >= self._worker_end:
                break
            
            line = self._file.readline()
            if not line:
                break
            
            try:
                # Парсинг JSONL
                record = json.loads(line)
                
                # Форматирование
                if self.format_fn:
                    text = self.format_fn(record)
                else:
                    text = record.get("text", record.get("content", ""))
                
                if text:
                    self._buffer.append(text)
                    
            except json.JSONDecodeError:
                # Пропуск некорректных строк
                continue
    
    def _tokenize(self, text: str) -> Dict[str, Any]:
        """Токенизация текста."""
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
        """Закрыть файл."""
        if self._file is not None:
            try:
                self._file.close()
            except:
                pass
            self._file = None
    
    def __del__(self):
        """Гарантированное закрытие файла."""
        self._close_file()
    
    def set_total_length(self, length: int) -> None:
        """Установить известную длину датасета."""
        self._total_lines = length
        super().set_total_length(length)
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика streaming датасета."""
        stats = super().get_stats()
        stats.update({
            "data_path": str(self.data_path),
            "buffer_size": self.buffer_size,
            "lines_read": self._line_count,
        })
        return stats


class StreamingCSVDataset(StreamingDataset):
    """
    Streaming датасет для CSV файлов.
    
    Пример использования:
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
        self._buffer: List[Dict] = []
        self._line_count = 0
        self._headers: Optional[List[str]] = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.data_path}")
    
    def __iter__(self):
        """Итератор с поддержкой multi-processing."""
        import csv
        
        worker_info = torch.utils.data.get_worker_info()
        
        # Сброс состояния
        self._buffer = []
        self._line_count = 0
        
        if self._file is not None:
            try:
                self._file.close()
            except:
                pass
        
        # Определение диапазона для воркера
        if worker_info is None:
            self._worker_start = 0
            self._worker_end = float('inf')
        else:
            if self._total_lines:
                per_worker = self._total_lines // worker_info.num_workers
                self._worker_start = worker_info.id * per_worker
                self._worker_end = self._worker_start + per_worker
            else:
                self._worker_start = 0
                self._worker_end = float('inf')
        
        # Открытие CSV файла
        self._file = open(self.data_path, 'r', encoding='utf-8', newline='')
        reader = csv.DictReader(self._file)
        self._headers = reader.fieldnames
        
        # Пропуск строк до начала диапазона
        if self._worker_start > 0:
            for _ in range(self._worker_start):
                try:
                    next(reader)
                except StopIteration:
                    break
        
        return self
    
    def __next__(self) -> Dict[str, Any]:
        """Получить следующий элемент."""
        if self._file is None:
            raise StopIteration
        
        if not self._buffer:
            self._fill_buffer()
        
        if not self._buffer:
            self._close_file()
            raise StopIteration
        
        record = self._buffer.pop(0)
        self._line_count += 1
        
        # Форматирование
        if self.format_fn:
            return self.format_fn(record)
        
        # Извлечение текста и метки
        result = {"inputs": record.get(self.text_column, "")}
        
        if self.label_column and self.label_column in record:
            result["labels"] = record[self.label_column]
        
        return result
    
    def _fill_buffer(self) -> None:
        """Заполнить буфер следующими строками."""
        if self._file is None:
            return
        
        import csv
        
        self._buffer = []
        reader = csv.DictReader(self._file, fieldnames=self._headers)
        
        for _ in range(self.buffer_size):
            if self._line_count >= self._worker_end:
                break
            
            try:
                record = next(reader)
                self._buffer.append(record)
            except StopIteration:
                break
    
    def _close_file(self) -> None:
        """Закрыть файл."""
        if self._file is not None:
            try:
                self._file.close()
            except:
                pass
            self._file = None
    
    def __del__(self):
        self._close_file()
    
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "data_path": str(self.data_path),
            "columns": self._headers,
            "text_column": self.text_column,
            "label_column": self.label_column,
        })
        return stats
