"""Tests for the max_consecutive_rollbacks guard.

Feature: if every batch keeps failing (NaN/Inf or spikes) with no healthy
batch in between, training must stop with a clear RuntimeError ("data is
likely corrupted") instead of rolling back and halving the LR forever,
and the best saved weights must be restored before raising.

All tests are CPU-only and fast (no CUDA, no heavy training).
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from selgis.config import SelgisConfig
from selgis.core import SelgisCore
from selgis.scheduler import SmartScheduler
from selgis.trainer import Trainer


def _make_core(config: SelgisConfig, model: nn.Module | None = None) -> tuple[SelgisCore, nn.Module]:
    model = model if model is not None else nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = SmartScheduler(optimizer, 0.1, config)
    core = SelgisCore(model, optimizer, scheduler, config, torch.device("cpu"))
    return core, model


class TestConsecutiveRollbackGuard:
    def test_consecutive_nan_batches_exhaust_limit(self):
        """A run of NaN batches with no healthy batch in between must stop
        training with RuntimeError after max_consecutive_rollbacks, and the
        best saved weights (not the last-good snapshot) must be restored."""
        cfg = SelgisConfig(
            max_epochs=5,
            nan_recovery=True,
            max_consecutive_rollbacks=3,
            state_update_interval=1,
            state_storage="memory",
        )
        core, model = _make_core(cfg)

        # Register best weights (= initial weights).
        status = core.eval_epoch(
            {"loss": 1.0}, 0, primary_metric="loss", higher_is_better=False
        )
        assert status == "IMPROVED"
        best_weights = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }

        # Simulate real training progress: one optimizer step moves the
        # weights away from best; state_update_interval=1 snapshots them
        # as the new last-good state.
        x = torch.randn(8, 4)
        core.backward_step(model(x).sum())
        core.optimizer_step()
        assert not torch.equal(model.weight, best_weights["weight"])

        # Every batch fails: the limit must be hit exactly on the 3rd NaN.
        with pytest.raises(RuntimeError, match="data is likely corrupted"):
            for _ in range(3):
                core.check_loss(torch.tensor(float("nan")))

        # Weights restored to best saved state, not to the poisoned
        # last-good snapshot taken after the optimizer step.
        for name, param in model.named_parameters():
            assert torch.equal(param, best_weights[name]), (
                f"{name} was not restored to best weights"
            )

    def test_interleaved_healthy_batches_never_trigger(self):
        """NaN batches interleaved with healthy ones reset the counter —
        the guard must never fire."""
        cfg = SelgisConfig(
            max_epochs=5,
            nan_recovery=True,
            max_consecutive_rollbacks=2,
            min_history_len=3,
            state_storage="memory",
        )
        core, _model = _make_core(cfg)

        sequence = [float("nan"), 0.5, float("nan"), 0.4,
                    float("nan"), 0.3, float("nan"), 0.2]
        for value in sequence:
            loss = torch.tensor(value)
            ok = core.check_loss(loss)
            assert ok is not math.isnan(value)

        # A healthy batch always resets the counter.
        assert core._consecutive_rollbacks == 0

    def test_zero_disables_guard(self):
        """max_consecutive_rollbacks=0 keeps the old behavior: unlimited
        rollbacks, training never stops on its own."""
        cfg = SelgisConfig(
            max_epochs=5,
            nan_recovery=True,
            max_consecutive_rollbacks=0,
            state_storage="memory",
        )
        core, model = _make_core(cfg)

        for _ in range(30):
            assert core.check_loss(torch.tensor(float("nan"))) is False

        # Rollbacks still restore the last-good state: no NaN in weights.
        for param in model.parameters():
            assert not torch.isnan(param.data).any()

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError, match="max_consecutive_rollbacks"):
            SelgisConfig(max_consecutive_rollbacks=-1)

    def test_trainer_stops_with_runtime_error(self, tmp_path):
        """End to end: a Trainer whose every batch produces NaN must stop
        training by raising RuntimeError from train()."""

        class AlwaysNaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(4, 2)

            def forward(self, inputs, **kwargs):
                # NaN flows through the graph into the loss (a bare NaN leaf
                # would not poison anything — see demo notebook).
                return self.net(inputs) * float("nan")

        torch.manual_seed(0)
        X = torch.randn(32, 4)
        y = torch.randint(0, 2, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=4)  # 8 batches/epoch

        cfg = SelgisConfig(
            max_epochs=2,
            batch_size=4,
            nan_recovery=True,
            max_consecutive_rollbacks=3,
            logging_steps=1000,
            state_storage="memory",
            output_dir=str(tmp_path),
        )
        trainer = Trainer(
            model=AlwaysNaNModel(),
            config=cfg,
            train_dataloader=loader,
            eval_dataloader=None,
            criterion=nn.CrossEntropyLoss(),
        )
        with pytest.raises(RuntimeError, match="data is likely corrupted"):
            trainer.train()
