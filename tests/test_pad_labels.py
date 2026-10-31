"""Regression test for a bug found during real-hardware validation (W4).

Bug: TextDataset / HFTextDataset / StreamingTextDataset returned
``labels = input_ids`` with pad-token ids left in place, so the LM loss
was computed over padding too (diluting it, and training the model on
pad positions). Padding positions must carry ``ignore_index=-100``.
"""

import torch
from torch.utils.data import DataLoader

from selgis.datasets.streaming import StreamingTextDataset
from selgis.datasets.text import HFTextDataset, TextDataset


class PadTokenizer:
    """Minimal tokenizer: token = position, pads to max_length with id 0."""

    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, texts, max_length=8, truncation=True, padding="max_length",
                 return_tensors=None, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            n = min(max_length, max(1, len(t.split())))
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _write_jsonl(path, rows):
    import json

    path.write_text(
        "\n".join(json.dumps({"text": t}) for t in rows) + "\n",
        encoding="utf-8",
    )


class TestPadLabelsMasked:
    def _expect_masked(self, sample, ids):
        labels = sample["labels"]
        mask = sample["attention_mask"]
        # real tokens keep their ids
        assert torch.equal(labels[mask == 1], ids[mask == 1])
        # padding positions are ignored
        assert (labels[mask == 0] == -100).all()

    def test_textdataset_live_tokenize_masks_padding(self, tmp_path):
        data = tmp_path / "data.jsonl"
        _write_jsonl(data, ["one two", "three"])
        ds = TextDataset(
            data_path=data, file_format="jsonl", tokenizer=PadTokenizer(), max_length=8
        )
        ids = ds[1]["input_ids"]
        self._expect_masked(ds[1], ids)

    def test_textdataset_pre_tokenized_masks_padding(self, tmp_path):
        data = tmp_path / "data.jsonl"
        _write_jsonl(data, ["one two", "three", "four five six"])
        ds = TextDataset(
            data_path=data,
            file_format="jsonl",
            tokenizer=PadTokenizer(),
            max_length=8,
            pre_tokenize=True,
            use_mmap=False,
        )
        ids = ds[0]["input_ids"]
        self._expect_masked(ds[0], ids)

    def test_hf_text_dataset_masks_padding(self):
        import datasets as hf_datasets
        from unittest import mock

        fake = hf_datasets.DatasetDict(
            {"train": hf_datasets.Dataset.from_dict({"text": ["one two", "three"]})}
        )
        with mock.patch("datasets.load_dataset", return_value=fake):
            ds = HFTextDataset(
                dataset_name="dummy/name", tokenizer=PadTokenizer(), max_length=8
            )
        ids = ds[1]["input_ids"]
        self._expect_masked(ds[1], ids)

    def test_streaming_text_dataset_masks_padding(self, tmp_path):
        data = tmp_path / "data.jsonl"
        _write_jsonl(data, ["one two", "three"])
        ds = StreamingTextDataset(
            data_path=data, tokenizer=PadTokenizer(), max_length=8, buffer_size=2
        )
        sample = next(iter(ds))
        self._expect_masked(sample, sample["input_ids"])

    def test_batched_loss_ignores_padding(self):
        """End to end: collated batch loss is computed on real tokens only —
        the chunked CE with ignore_index=-100 must agree with a manual
        computation over non-pad positions."""
        import torch.nn as nn
        import torch.nn.functional as F

        from selgis.loss import ChunkedCrossEntropyLoss

        class PadDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor([1, 2, 3, 0, 0]),
                    "attention_mask": torch.tensor([1, 1, 1, 0, 0]),
                    "labels": torch.tensor([1, 2, 3, -100, -100]),
                }

        loader = DataLoader(PadDataset(), batch_size=4)
        batch = next(iter(loader))
        logits = torch.randn(4, 5, 16, requires_grad=True)
        loss = ChunkedCrossEntropyLoss(chunk_size=4, ignore_index=-100)(
            logits, batch["labels"]
        )
        ref = F.cross_entropy(
            logits.reshape(-1, 16), batch["labels"].reshape(-1), ignore_index=-100
        )
        assert torch.allclose(loss, ref)
        ref_real = F.cross_entropy(
            logits[:, :3, :].reshape(-1, 16), batch["labels"][:, :3].reshape(-1)
        )
        assert torch.allclose(loss, ref_real, atol=1e-5)
