"""Regression tests for bugs found in the 0.2.7.2 code audit.

Each test is pinned to a specific bug fixed in the 0.2.7.2 release;
if any of these bugs is reintroduced, the corresponding test must fail.

All tests are CPU-only and fast (no CUDA, no heavy training).
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from selgis.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)
from selgis.cli import _build_config
from selgis.config import SelgisConfig
from selgis.core import SelgisCore
from selgis.datasets.text import HFTextDataset, TextDataset
from selgis.loss import ChunkedCrossEntropyLoss
from selgis.lr_finder import LRFinder
from selgis.scheduler import SmartScheduler
from selgis.trainer import Trainer
from selgis.utils import seed_everything

V_SMALL = 100


# ══════════════════════════════════════════════════════════════════════════════
# TextDataset fixes
# ══════════════════════════════════════════════════════════════════════════════


class TestTextDatasetIndexing:
    """Bug: non-mmap _build_index drifted byte offsets (CRLF/UTF-8) and
    looped forever, growing memory unboundedly — even on a 3-line file."""

    def _write_jsonl(self, path: Path, rows: list[str]) -> None:
        import json

        path.write_text(
            "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in rows) + "\n",
            encoding="utf-8",
        )

    def test_non_mmap_index_terminates_on_utf8(self, tmp_path: Path):
        """The pre-fix code hung forever here (byte-offset drift). Run in a
        subprocess with a hard timeout so a regression fails instead of
        hanging the whole suite."""
        data = tmp_path / "data.jsonl"
        rows = ["привет мир", "вторая строка", "третья строка"]
        self._write_jsonl(data, rows)

        script = (
            "import json, sys\n"
            "sys.path.insert(0, '.')\n"
            "from selgis.datasets.text import TextDataset\n"
            "ds = TextDataset(data_path=sys.argv[1], file_format='jsonl', use_mmap=False)\n"
            "texts = [json.loads(json.dumps(ds[i]))['text'] for i in range(len(ds))]\n"
            "expected = ['привет мир', 'вторая строка', 'третья строка']\n"
            "assert texts == expected, texts\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(data)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=".",
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_non_mmap_records_match_mmap_records(self, tmp_path: Path):
        data = tmp_path / "data.jsonl"
        rows = [f"строка {i} текст" for i in range(50)]
        self._write_jsonl(data, rows)

        ds_mmap = TextDataset(data_path=data, file_format="jsonl", use_mmap=True)
        ds_plain = TextDataset(data_path=data, file_format="jsonl", use_mmap=False)
        assert len(ds_mmap) == len(ds_plain) == 50
        for i in (0, 1, 25, 49):
            assert ds_plain[i] == ds_mmap[i]

    def test_crlf_file_has_no_carriage_returns(self, tmp_path: Path):
        """Pre-fix: trailing '\\r' leaked into text from CRLF files."""
        data = tmp_path / "crlf.txt"
        data.write_bytes("строка один\r\nстрока два\r\n".encode("utf-8"))
        ds = TextDataset(data_path=data, file_format="txt")
        assert ds[0]["text"] == "строка один"
        assert ds[1]["text"] == "строка два"
        for i in range(len(ds)):
            assert "\r" not in ds[i]["text"]

    def test_txt_and_csv_getitem_do_not_crash(self, tmp_path: Path):
        """Pre-fix: __getitem__ read from the empty byte index for .txt/.csv
        and raised IndexError on any non-jsonl dataset."""
        txt = tmp_path / "doc.txt"
        txt.write_text("alpha\nbeta\n", encoding="utf-8")
        ds_txt = TextDataset(data_path=txt, file_format="txt")
        assert len(ds_txt) == 2
        assert ds_txt[1]["text"] == "beta"

        csv_file = tmp_path / "tab.csv"
        csv_file.write_text("text,score\nпривет,1\nмир,2\n", encoding="utf-8")
        ds_csv = TextDataset(data_path=csv_file, file_format="csv", text_column="text")
        assert len(ds_csv) == 2  # header row must not be a data record
        assert ds_csv[0]["text"] == "привет"
        assert ds_csv[1]["text"] == "мир"

    def test_pre_tokenize_without_cache_dir(self, tmp_path: Path):
        """Pre-fix: `None / str` TypeError when pre_tokenize=True, cache_dir=None."""

        class StubTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    "input_ids": torch.ones(len(texts), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
                }

        data = tmp_path / "data.jsonl"
        self._write_jsonl(data, ["a", "b", "c"])
        ds = TextDataset(
            data_path=data,
            file_format="jsonl",
            use_mmap=False,
            pre_tokenize=True,
            cache_dir=None,
            tokenizer=StubTokenizer(),
        )
        assert len(ds) == 3
        assert ds[0]["input_ids"].shape == (4,)


class TestHFTextDataset:
    """Bug: load_dataset("name") returns a DatasetDict; the
    hasattr(..., "split") check never unwrapped it, so len() returned the
    number of splits instead of the number of rows."""

    def test_datasetdict_is_unwrapped(self, tmp_path: Path):
        datasets = pytest.importorskip("datasets")

        fake = datasets.DatasetDict(
            {"train": datasets.Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})}
        )
        with mock.patch("datasets.load_dataset", return_value=fake):
            ds = HFTextDataset(dataset_name="dummy/name")
        assert len(ds) == 5  # pre-fix: 1 (number of splits)

        with mock.patch("datasets.load_dataset", return_value=fake):
            ds_nt = HFTextDataset(dataset_name="dummy/name")
        record = ds_nt[0]
        assert record["text"] == "a"


# ══════════════════════════════════════════════════════════════════════════════
# ChunkedCrossEntropyLoss fixes
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkedCECorrectness:
    """Bug 1: label smoothing was eps*log(V) — a constant with zero gradient,
    values diverged from F.cross_entropy.
    Bug 2: all fp32 chunk intermediates were retained for backward."""

    @pytest.mark.parametrize("eps", [0.0, 0.1])
    def test_values_match_reference(self, eps):
        torch.manual_seed(0)
        logits = torch.randn(4, 8, V_SMALL)
        targets = torch.randint(0, V_SMALL, (4, 8))
        ours = ChunkedCrossEntropyLoss(chunk_size=32, label_smoothing=eps)(logits, targets)
        ref = F.cross_entropy(
            logits.view(-1, V_SMALL), targets.view(-1), label_smoothing=eps
        )
        assert torch.allclose(ours, ref, atol=1e-5)

    @pytest.mark.parametrize("eps", [0.0, 0.1])
    def test_gradients_match_reference(self, eps):
        torch.manual_seed(0)
        logits = torch.randn(4, 8, V_SMALL)
        targets = torch.randint(0, V_SMALL, (4, 8))

        lg = logits.clone().requires_grad_(True)
        ChunkedCrossEntropyLoss(chunk_size=32, label_smoothing=eps)(lg, targets).backward()

        lg_ref = logits.clone().requires_grad_(True)
        F.cross_entropy(
            lg_ref.view(-1, V_SMALL), targets.view(-1), label_smoothing=eps
        ).backward()

        assert (lg.grad - lg_ref.grad).abs().max().item() < 1e-5

    def test_ignore_index_matches_reference(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 8, V_SMALL)
        targets = torch.randint(0, V_SMALL, (4, 8))
        targets[:, 0] = -100
        ours = ChunkedCrossEntropyLoss(chunk_size=32, ignore_index=-100)(logits, targets)
        ref = F.cross_entropy(
            logits.view(-1, V_SMALL), targets.view(-1), ignore_index=-100
        )
        assert torch.allclose(ours, ref, atol=1e-5)

    def test_reduction_none_matches_reference(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 8, V_SMALL)
        targets = torch.randint(0, V_SMALL, (4, 8))
        ours = ChunkedCrossEntropyLoss(chunk_size=32, reduction="none")(logits, targets)
        ref = F.cross_entropy(
            logits.view(-1, V_SMALL), targets.view(-1), reduction="none"
        ).view(4, 8)
        assert ours.shape == (4, 8)
        assert torch.allclose(ours, ref, atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer fixes
# ══════════════════════════════════════════════════════════════════════════════


class TestTrainerFixes:
    def _tiny_loader(self, n=32, dim=4):
        torch.manual_seed(0)
        X = torch.randn(n, dim)
        y = torch.randint(0, 2, (n,))
        return DataLoader(TensorDataset(X, y), batch_size=8)

    def test_lower_is_better_metric_direction(self):
        """Bug: perplexity/mse/mae as primary_metric were treated as
        'higher is better' — a worse perplexity was marked IMPROVED."""
        model = nn.Linear(4, 2)
        cfg = SelgisConfig(patience=10, nan_recovery=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        core = SelgisCore(
            model, opt, SmartScheduler(opt, 1e-3, cfg, 100), cfg, torch.device("cpu")
        )
        s1 = core.eval_epoch(
            {"perplexity": 100.0}, 0, primary_metric="perplexity", higher_is_better=False
        )
        s2 = core.eval_epoch(
            {"perplexity": 200.0}, 1, primary_metric="perplexity", higher_is_better=False
        )
        assert s1 == "IMPROVED"
        assert s2 == "CONTINUE"  # pre-fix: IMPROVED (worse metric accepted)

    def test_step_based_lr_not_overwritten_at_epoch_boundary(self, tmp_path: Path):
        """Bug: with warmup_ratio > 0 the epoch-end step_epoch() reset the LR
        schedule — LR jumped back to peak at every epoch boundary."""
        seed_everything(0)
        model = nn.Linear(4, 2)
        cfg = SelgisConfig(
            max_epochs=3,
            batch_size=8,
            warmup_ratio=0.5,
            learning_rate=1e-2,
            nan_recovery=False,
            grad_clip_norm=0.0,
            output_dir=str(tmp_path),
        )
        trainer = Trainer(
            model=model,
            config=cfg,
            train_dataloader=self._tiny_loader(),
            eval_dataloader=None,
            criterion=nn.CrossEntropyLoss(),
        )
        lrs: list[float] = []
        original_step = trainer.scheduler.step

        def spy_step():
            lrs.append(original_step())
            return lrs[-1]

        trainer.scheduler.step = spy_step
        trainer.train()

        warmup = trainer.scheduler._warmup_steps
        assert warmup == 6
        assert len(lrs) == 12  # 4 steps/epoch x 3 epochs
        # warmup ramps up
        assert all(a < b for a, b in zip(lrs[:warmup - 1], lrs[1:warmup]))
        # post-warmup: monotone decay, no jumps to peak at epoch boundaries
        post = lrs[warmup:]
        assert all(b <= a + 1e-12 for a, b in zip(post, post[1:])), lrs

    def test_token_type_ids_not_injected_when_unsupported(self):
        """Bug: token_type_ids was injected into every dict batch — models
        without the parameter (Llama-style) crashed with TypeError."""

        class PickyOutput:
            def __init__(self, loss):
                self.loss = loss

        class PickyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, labels=None):
                return PickyOutput((input_ids.float().mean() * self.dummy).sum())

        cfg = SelgisConfig(
            max_epochs=1, batch_size=2, nan_recovery=False, grad_clip_norm=0.0
        )
        trainer = Trainer(
            model=PickyModel(),
            config=cfg,
            train_dataloader=self._tiny_loader(),
            eval_dataloader=None,
            criterion=nn.MSELoss(),
        )
        batch = {
            "input_ids": torch.randint(0, 10, (2, 4)),
            "labels": torch.tensor([1.0, 0.0]),
        }
        loss, _ = trainer._forward(batch)  # pre-fix: TypeError
        assert isinstance(loss, torch.Tensor)

    def test_token_type_ids_passed_when_supported(self):
        class TTOutput:
            def __init__(self, loss):
                self.loss = loss

        class TTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
                self.seen = None

            def forward(self, input_ids, labels=None, token_type_ids=None):
                self.seen = token_type_ids
                return TTOutput(torch.zeros(()))

        cfg = SelgisConfig(
            max_epochs=1, batch_size=2, nan_recovery=False, grad_clip_norm=0.0
        )
        model = TTModel()
        trainer = Trainer(
            model=model,
            config=cfg,
            train_dataloader=self._tiny_loader(),
            eval_dataloader=None,
            criterion=nn.MSELoss(),
        )
        batch = {
            "input_ids": torch.randint(0, 10, (2, 4)),
            "labels": torch.tensor([1.0, 0.0]),
        }
        trainer._forward(batch)
        assert model.seen is not None and model.seen.shape == (2, 4)

    def test_dict_batch_positional_fallback(self):
        """Bug: dict batches ({"inputs": ...}) were passed as **kwargs —
        plain nn.Module inputs crashed with TypeError."""

        class PlainOutput:
            def __init__(self, value):
                self.value = value

        cfg = SelgisConfig(
            max_epochs=1, batch_size=2, nan_recovery=False, grad_clip_norm=0.0
        )
        model = nn.Linear(4, 2)
        trainer = Trainer(
            model=model,
            config=cfg,
            train_dataloader=self._tiny_loader(),
            eval_dataloader=None,
            criterion=nn.MSELoss(),
        )
        batch = {"inputs": torch.randn(2, 4), "labels": torch.randn(2, 2)}
        loss, _ = trainer._forward(batch)  # pre-fix: TypeError('inputs')
        assert isinstance(loss, torch.Tensor)


# ══════════════════════════════════════════════════════════════════════════════
# CLI, callbacks, LRFinder fixes
# ══════════════════════════════════════════════════════════════════════════════


class TestCliOverrides:
    """Bug: --set KEY=VALUE overrides were parsed but never applied."""

    def test_set_overrides_applied(self):
        raw = {"batch_size": 16, "max_epochs": 7, "learning_rate": 0.001}
        config, _ = _build_config(raw, ["max_epochs=2", "batch_size=8"])
        assert config.max_epochs == 2
        assert config.batch_size == 8
        assert config.learning_rate == 0.001  # untouched keys stay


class TestCallbackFixes:
    """Bug: metrics.get(metric, 0.0) counted a missing metric as a fake
    improvement (or fake regression), corrupting early stopping."""

    def test_early_stopping_missing_metric_is_ignored(self):
        """Bug: metrics.get(metric, 0.0) fabricated a value for a missing
        metric. Now a missing metric is skipped entirely: it must not count
        as improvement (pre-fix, mode='max' saw a fake 0.0 > -inf) and must
        not advance the patience counter."""
        cb = EarlyStoppingCallback(patience=2, metric="f1", mode="max", min_delta=0.0)
        cb.on_epoch_end(trainer=None, epoch=0, metrics={"f1": 0.5})
        cb.on_epoch_end(trainer=None, epoch=1, metrics={"loss": 0.5})  # no "f1"
        assert cb._counter == 0
        assert cb.should_stop is False

        # real non-improving epochs still exhaust patience
        cb.on_epoch_end(trainer=None, epoch=2, metrics={"f1": 0.5})
        cb.on_epoch_end(trainer=None, epoch=3, metrics={"f1": 0.5})
        assert cb.should_stop is True

    def test_early_stopping_real_metric_still_works(self):
        cb = EarlyStoppingCallback(patience=2, metric="f1", mode="max", min_delta=0.0)
        cb.on_epoch_end(trainer=None, epoch=0, metrics={"f1": 0.5})
        cb.on_epoch_end(trainer=None, epoch=1, metrics={"f1": 0.6})
        assert cb.should_stop is False
        assert cb._best == 0.6

    def test_checkpoint_missing_metric_creates_no_checkpoint(self, tmp_path: Path):
        cb = CheckpointCallback(
            output_dir=str(tmp_path), save_best_only=True, metric="accuracy", mode="max"
        )
        class FakeCfg:
            output_dir = str(tmp_path)

        class FakeTrainer:
            config = FakeCfg()
            model = nn.Linear(2, 2)

        cb.on_epoch_end(trainer=FakeTrainer(), epoch=0, metrics={"loss": 0.5})
        assert not (tmp_path / "best_model").exists()  # pre-fix: saved with fake 0.0


class TestLoggingCallbackNoneLoss:
    def test_none_loss_is_skipped(self):
        """Skipped batches pass loss=None; callback must not format None."""
        model = nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        class FakeTrainer:
            optimizer = opt

        cb = LoggingCallback(log_every=1)
        cb.on_step_end(FakeTrainer(), step=1, loss=None)  # must not raise
        cb.on_step_end(FakeTrainer(), step=1, loss=0.5)  # must not raise


class TestLRFinderBuffers:
    """Bug: LRFinder saved/restored only parameters; BatchNorm running
    stats stayed mutated after the sweep."""

    def test_buffers_restored_after_sweep(self):
        seed_everything(0)
        model = nn.Sequential(
            nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2)
        )
        model.train()
        before = {n: b.clone() for n, b in model.named_buffers()}

        X = torch.randn(64, 4)
        y = torch.randint(0, 2, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)

        finder = LRFinder(
            model,
            torch.optim.SGD(model.parameters(), lr=1e-5),
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
        finder.find(loader, start_lr=1e-6, end_lr=1e-2, num_steps=4, log_steps=100)

        for name, buf in model.named_buffers():
            assert torch.equal(before[name], buf.cpu()), f"buffer {name} not restored"
