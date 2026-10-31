"""Regression test for a bug found during real-hardware validation (W4).

Bug: TransformerTrainer("model-name", config=TransformerConfig(use_unsloth=True))
defers model loading to _apply_unsloth (model=None), but when unsloth is not
installed, _apply_unsloth returned None and the trainer crashed later with an
obscure `AttributeError: 'NoneType' object has no attribute 'to'` instead of
falling back to the standard loading path.
"""

import sys
from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from selgis.config import TransformerConfig
from selgis.trainer import TransformerTrainer


class TestUnslothFallback:
    def _tiny_loader(self, n=8, dim=4):
        X = nn.init.normal_(torch.empty(n, dim))
        y = torch.randint(0, 2, (n,))
        return DataLoader(TensorDataset(X, y), batch_size=4)

    def test_use_unsloth_without_unsloth_falls_back_to_standard_loading(self):
        """use_unsloth=True + unsloth missing (ImportError) must load the
        model via the standard path instead of leaving model=None."""
        pytest.importorskip("transformers")

        cfg = TransformerConfig(
            model_name_or_path="dummy/model",
            use_unsloth=True,  # without use_peft: peft stays with the standard path
            state_storage="memory",
        )

        with mock.patch.dict(sys.modules, {"unsloth": None}), mock.patch.object(
            TransformerTrainer,
            "_load_model",
            return_value=nn.Linear(4, 2),
        ) as load_model:
            trainer = TransformerTrainer(
                "dummy/model",
                config=cfg,
                train_dataloader=self._tiny_loader(),
            )

        load_model.assert_called_once()  # pre-fix: never called, model stayed None
        assert isinstance(trainer.model, nn.Linear)
