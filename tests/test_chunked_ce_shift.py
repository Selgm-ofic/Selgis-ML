"""Regression tests for a bug found during real-hardware validation (W4).

Bug: with chunked_ce=True the TransformerTrainer strips ``labels`` from the
model call (so HF does not compute its own CE) but then fed the raw labels
to the chunked criterion WITHOUT the causal shift. For causal_lm/seq2seq the
model was therefore trained to copy the CURRENT token (logits[t] -> labels[t])
instead of predicting the NEXT one (logits[t] -> labels[t+1], the HF
semantics). Symptom on a real run: train_loss fell to 1.58 while eval loss
(HF-computed, shifted) stayed at 9.87 — the model learned nothing useful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from selgis.config import TransformerConfig
from selgis.trainer import TransformerTrainer

V = 32


class TinyCausalLM(nn.Module):
    """Minimal stand-in exposing HF-style ``logits`` (no internal loss)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 16)
        self.head = nn.Linear(16, V)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return type(
            "Output", (), {"logits": self.head(self.emb(input_ids)), "loss": None}
        )()


class TestChunkedCEShift:
    def _trainer(self, problem_type: str) -> tuple[TransformerTrainer, TinyCausalLM]:
        torch.manual_seed(0)
        model = TinyCausalLM()
        config = TransformerConfig(
            model_name_or_path="dummy/tiny",
            problem_type=problem_type,
            chunked_ce=True,
            ce_chunk_size=8,
            label_smoothing=0.0,  # reference CE below is unsmoothed
            max_epochs=1,
            device="cpu",
            state_storage="memory",
        )
        loader = DataLoader(TensorDataset(torch.zeros(4, 6, dtype=torch.long)), batch_size=2)
        trainer = TransformerTrainer(model, config=config, train_dataloader=loader)
        return trainer, model

    def test_causal_lm_chunked_loss_uses_hf_shift_semantics(self):
        """chunked train loss must equal the SHIFTED CE (logits[:, :-1] vs
        labels[:, 1:]), matching what HF computes for eval."""
        trainer, model = self._trainer("causal_lm")

        torch.manual_seed(1)
        ids = torch.randint(0, V, (2, 6))
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }

        loss, _ = trainer._forward(batch)

        with torch.no_grad():
            logits = model(input_ids=ids).logits
            ref = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, V), ids[:, 1:].reshape(-1)
            )

        assert torch.allclose(loss, ref, atol=1e-5), (
            f"chunked loss {loss.item():.4f} != shifted reference {ref.item():.4f} "
            "(pre-fix: labels were not shifted, model learned to copy)"
        )

    def test_masked_lm_chunked_loss_is_position_aligned(self):
        """masked_lm must NOT be shifted: logits[t] predicts labels[t]."""
        trainer, model = self._trainer("masked_lm")

        torch.manual_seed(1)
        ids = torch.randint(0, V, (2, 6))
        batch = {"input_ids": ids, "labels": ids.clone()}

        loss, _ = trainer._forward(batch)

        with torch.no_grad():
            logits = model(input_ids=ids).logits
            ref = F.cross_entropy(
                logits.reshape(-1, V), ids.reshape(-1)
            )

        assert torch.allclose(loss, ref, atol=1e-5), (
            f"masked_lm loss {loss.item():.4f} != position-aligned reference "
            f"{ref.item():.4f}"
        )
