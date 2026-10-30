import gc
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataclasses import asdict

from selgis import (
    Callback,
    CheckpointCallback,
    CustomDataset,
    DatasetConfig,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    SelgisConfig,
    SmartScheduler,
    Trainer,
    TransformerConfig,
    create_dataloaders,
    create_dataset,
    get_device,
    seed_everything,
    count_parameters,
    format_params,
    move_to_device,
    unpack_batch,
    get_optimizer_grouped_params,
    is_dict_like,
)
from selgis.checkpointing import GradientCheckpointingManager
from selgis.loss import ChunkedCrossEntropyLoss, CrossEntropyLossV2


# =============================================================================
# Test 1: Imports
# =============================================================================

def test_imports():
    from selgis import (
        SelgisConfig, TransformerConfig, Trainer, TransformerTrainer,
        SelgisCore, LRFinder, SmartScheduler, DatasetConfig,
        BaseDataset, TextDataset, ImageDataset, MultimodalDataset,
        StreamingTextDataset, CustomDataset, create_dataset,
        create_dataloaders, Callback, LoggingCallback,
        EarlyStoppingCallback, CheckpointCallback, HistoryCallback,
    )
    assert SelgisConfig is not None


# =============================================================================
# Test 2: Configuration
# =============================================================================

class TestConfig:
    def test_selgis_config(self):
        config = SelgisConfig(max_epochs=10, batch_size=32, lr_finder_enabled=False)
        assert config.max_epochs == 10
        assert config.batch_size == 32

    def test_transformer_config(self):
        tf_config = TransformerConfig(
            model_name_or_path="test-model",
            use_peft=True,
            peft_config={"r": 16, "lora_alpha": 32},
            quantization_type="no",
        )
        assert tf_config.model_name_or_path == "test-model"
        assert tf_config.use_peft

    def test_dataset_config_serialization(self):
        ds_config = DatasetConfig(
            data_type="text", data_path="./data.jsonl", batch_size=32, max_length=512
        )
        ds_dict = ds_config.to_dict()
        ds_config2 = DatasetConfig.from_dict(ds_dict)
        assert ds_config2.data_type == ds_config.data_type
        assert ds_config2.data_path == ds_config.data_path

    def test_config_fp16_bf16_conflict(self):
        with pytest.raises(ValueError):
            SelgisConfig(fp16=True, bf16=True)

    def test_config_dual_warmup_conflict(self):
        with pytest.raises(ValueError):
            SelgisConfig(warmup_epochs=5, warmup_ratio=0.1)

    def test_transformer_config_empty_peft(self):
        with pytest.raises(ValueError):
            TransformerConfig(use_peft=True, peft_config={}, quantization_type="no")

    def test_transformer_config_no_model_name(self):
        config = TransformerConfig(use_peft=False)
        assert config.model_name_or_path == ""


# =============================================================================
# Test 3: Datasets
# =============================================================================

class TestDatasets:
    def test_custom_dataset(self):
        class SimpleDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {"inputs": torch.randn(10), "labels": torch.randint(0, 2, (1,)).squeeze()}

        custom_ds = CustomDataset(SimpleDataset(100))
        assert len(custom_ds) == 100
        sample = custom_ds[0]
        assert "inputs" in sample
        custom_ds.validate()
        stats = custom_ds.get_stats()
        assert isinstance(stats, dict)

    def test_dataset_config_creations(self):
        for dtype in ("text", "image", "streaming"):
            config = DatasetConfig(data_type=dtype, data_path=f"./{dtype}")
            assert config.data_type == dtype


# =============================================================================
# Test 4: DataLoader
# =============================================================================

class TestDataLoader:
    def test_create_dataloaders(self, seed):
        class SimpleDatasetGlobal(Dataset):
            def __init__(self, size=100):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {"inputs": torch.randn(10), "labels": torch.randint(0, 2, (1,)).squeeze()}

        config = DatasetConfig(
            data_type="custom",
            custom_kwargs={"dataset": SimpleDatasetGlobal(100)},
            batch_size=10,
            num_workers=0,
            train_split=0.8,
        )
        train_loader, eval_loader = create_dataloaders(config)
        assert len(train_loader) > 0
        assert len(eval_loader) > 0
        batch = next(iter(train_loader))
        assert batch["inputs"].shape[0] <= 10
        assert batch["labels"].shape[0] <= 10


# =============================================================================
# Test 5: Trainer basic
# =============================================================================

class TestTrainer:
    def test_trainer_creation(self, simple_model, simple_loader):
        config = SelgisConfig(max_epochs=2, batch_size=16, lr_finder_enabled=False)
        trainer = Trainer(
            model=simple_model,
            config=config,
            train_dataloader=simple_loader,
            eval_dataloader=simple_loader,
            criterion=nn.CrossEntropyLoss(),
        )
        assert trainer.device is not None
        assert trainer.selgis is not None

    def test_trainer_training_step(self, simple_model, simple_loader):
        config = SelgisConfig(max_epochs=1, batch_size=16, lr_finder_enabled=False, logging_steps=5)
        trainer = Trainer(
            model=simple_model,
            config=config,
            train_dataloader=simple_loader,
            eval_dataloader=simple_loader,
            criterion=nn.CrossEntropyLoss(),
        )
        trainer.selgis._save_last_good_state()

        train_loss = 0.0
        num_batches = 0
        model = simple_model
        model.train()
        for batch in simple_loader:
            inputs = batch["inputs"].to(trainer.device)
            labels = batch["labels"].to(trainer.device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            train_loss += loss.item()
            num_batches += 1
            if num_batches >= 2:
                break
        train_loss /= num_batches
        assert 0 <= train_loss < 10, f"Loss {train_loss} out of expected range"


# =============================================================================
# Test 6: Callbacks
# =============================================================================

class TestCallbacks:
    def test_callback_creation(self):
        class TestCallback(Callback):
            def on_train_begin(self, trainer): pass
            def on_epoch_end(self, trainer, epoch, metrics): pass
        cb = TestCallback()
        assert isinstance(cb, Callback)

    def test_logging_callback(self):
        cb = LoggingCallback(log_every=10)
        assert cb.log_every == 10

    def test_early_stopping_callback(self):
        cb = EarlyStoppingCallback(patience=5, metric="loss", mode="min")
        assert cb.patience == 5

    def test_checkpoint_callback(self, tmp_output_dir):
        cb = CheckpointCallback(output_dir=str(tmp_output_dir))
        assert str(cb.output_dir) == str(tmp_output_dir)

    def test_history_callback(self, tmp_output_dir):
        cb = HistoryCallback(output_dir=str(tmp_output_dir))
        assert str(cb.output_dir) == str(tmp_output_dir)


# =============================================================================
# Test 7: E2E Loss Decrease
# =============================================================================

class TestE2ELossDecrease:
    def test_loss_decreases(self, device):
        class SimpleClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2),
                )
            def forward(self, inputs, **kwargs):
                return self.net(inputs)

        model = SimpleClassifier().to(device)

        class SyntheticDataset(Dataset):
            def __init__(self, size=200):
                self.size = size
                torch.manual_seed(42)
                self.X = torch.randn(size, 10)
                self.y = (self.X[:, :5].sum(dim=1) > 0).long()
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {"inputs": self.X[idx], "labels": self.y[idx]}

        dataset = SyntheticDataset(200)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        eval_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        config = SelgisConfig(
            max_epochs=15, batch_size=16, lr_finder_enabled=False,
            nan_recovery=False, logging_steps=1,
        )
        trainer = Trainer(
            model=model, config=config,
            train_dataloader=train_loader, eval_dataloader=eval_loader,
            criterion=nn.CrossEntropyLoss(),
        )
        trainer.train()

        history_cb = next(cb for cb in trainer.callbacks if isinstance(cb, HistoryCallback))
        history = history_cb.history
        assert len(history) > 0, "History is empty"

        first_loss = history[0]["metrics"]["loss"]
        last_loss = history[-1]["metrics"]["loss"]

        assert last_loss < first_loss * 0.8, (
            f"Loss didn't decrease enough: {first_loss:.4f} -> {last_loss:.4f}"
        )


# =============================================================================
# Test 8: Utils
# =============================================================================

class TestUtils:
    def test_get_device(self):
        device = get_device("auto")
        assert device.type in ("cpu", "cuda")

    def test_seed_everything(self):
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_count_parameters(self):
        model = nn.Linear(10, 10)
        params = count_parameters(model)
        assert params == 110

    def test_format_params(self):
        assert "1.23M" in format_params(1234567)

    def test_move_to_device(self, device):
        tensor = torch.randn(3, 3)
        moved = move_to_device(tensor, device)
        assert moved.device.type == device.type

    def test_unpack_batch_dict(self):
        batch = {"input_ids": torch.randn(2, 3), "labels": torch.randint(0, 2, (2,))}
        inputs, labels = unpack_batch(batch)
        assert isinstance(inputs, dict)
        assert labels.shape == (2,)

    def test_unpack_batch_tuple(self):
        batch = (torch.randn(2, 3), torch.randint(0, 2, (2,)))
        inputs, labels = unpack_batch(batch)
        assert inputs.shape == (2, 3)

    def test_is_dict_like(self):
        assert is_dict_like({})
        assert not is_dict_like([])

    def test_get_optimizer_grouped_params(self):
        model = nn.Linear(10, 10)
        params = get_optimizer_grouped_params(model, weight_decay=0.01)
        assert len(params) == 2


# =============================================================================
# Test 9: Custom Architectures
# =============================================================================

class TestCustomArchitectures:
    def _make_trainer(self, model, dataset, config=None):
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        if config is None:
            config = SelgisConfig(max_epochs=3, batch_size=16, lr_finder_enabled=False, logging_steps=10)
        trainer = Trainer(
            model=model, config=config,
            train_dataloader=loader, eval_dataloader=loader,
            criterion=nn.CrossEntropyLoss(),
        )
        trainer.train()
        history_cb = next((cb for cb in trainer.callbacks if isinstance(cb, HistoryCallback)), None)
        return trainer, history_cb

    def _assert_training_ok(self, history_cb, name):
        assert history_cb is not None, f"{name}: HistoryCallback not found"
        assert len(history_cb.history) >= 1, f"{name}: history empty"

    def test_resnet_like(self):
        class ResidualBlock(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.fc1 = nn.Linear(in_f, out_f)
                self.fc2 = nn.Linear(out_f, out_f)
                self.residual = nn.Linear(in_f, out_f) if in_f != out_f else nn.Identity()
                self.norm = nn.LayerNorm(out_f)
            def forward(self, x, **kwargs):
                return F.relu(self.norm(self.fc2(F.relu(self.fc1(x))) + self.residual(x)))

        class ResNetLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Linear(20, 64)
                self.block1 = ResidualBlock(64, 64)
                self.block2 = ResidualBlock(64, 64)
                self.head = nn.Linear(64, 2)
            def forward(self, inputs, **kwargs):
                x = self.block2(self.block1(F.relu(self.stem(inputs))))
                return self.head(x)

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randn(100, 20)
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i].float(), "labels": self.y[i]}

        _, hist = self._make_trainer(ResNetLike(), DS())
        self._assert_training_ok(hist, "ResNet")
        first = hist.history[0]["metrics"]["loss"]
        last = hist.history[-1]["metrics"]["loss"]
        assert last < first, f"ResNet: loss {first:.4f} -> {last:.4f}"

    def test_transformer_like(self):
        class TransformerLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
                self.norm = nn.LayerNorm(64)
                self.fc = nn.Linear(64, 2)
            def forward(self, inputs, **kwargs):
                x = self.embed(inputs)
                attn_out, _ = self.attn(x, x, x)
                x = self.norm(x + attn_out)
                return self.fc(x.mean(dim=1))

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randint(0, 100, (100, 8))
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

        _, hist = self._make_trainer(TransformerLike(), DS())
        self._assert_training_ok(hist, "Transformer")

    def test_cnn_like(self):
        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm1d(64)
                self.fc = nn.Linear(64, 2)
            def forward(self, inputs, **kwargs):
                x = F.relu(self.bn1(self.conv1(inputs.unsqueeze(1))))
                x = F.relu(self.bn2(self.conv2(x)))
                return self.fc(x.mean(dim=-1))

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randn(100, 10)
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i].float(), "labels": self.y[i]}

        _, hist = self._make_trainer(CNN1D(), DS())
        self._assert_training_ok(hist, "CNN")

    def test_lstm_like(self):
        class LSTMLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 64, 2, batch_first=True, dropout=0.1)
                self.fc = nn.Linear(64, 2)
            def forward(self, inputs, **kwargs):
                _, (h_n, _) = self.lstm(inputs)
                return self.fc(h_n[-1])

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randn(100, 8, 10)
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i].float(), "labels": self.y[i]}

        _, hist = self._make_trainer(LSTMLike(), DS())
        self._assert_training_ok(hist, "LSTM")


# =============================================================================
# Test 10: CUDA / Mixed Precision
# =============================================================================

class TestCUDA:
    def test_cuda_info(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA not available")

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= 1
        device = get_device("auto")
        assert device.type == "cuda"

    def test_mixed_precision_fp16(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
            def forward(self, inputs, **kwargs):
                return self.net(inputs)

        model = MLP().to(device)

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randn(100, 10)
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

        loader = DataLoader(DS(), batch_size=16)
        config = SelgisConfig(max_epochs=2, batch_size=16, lr_finder_enabled=False, logging_steps=10, fp16=True)
        trainer = Trainer(model=model, config=config, train_dataloader=loader, eval_dataloader=loader, criterion=nn.CrossEntropyLoss())
        trainer.train()
        assert trainer.selgis._amp_dtype == torch.float16

    def test_gradient_accumulation(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
            def forward(self, inputs, **kwargs):
                return self.net(inputs)

        model = MLP().to(device)

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randn(100, 10)
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

        loader = DataLoader(DS(), batch_size=16)
        config = SelgisConfig(max_epochs=2, batch_size=16, gradient_accumulation_steps=4, lr_finder_enabled=False, logging_steps=10)
        trainer = Trainer(model=model, config=config, train_dataloader=loader, eval_dataloader=loader, criterion=nn.CrossEntropyLoss())
        trainer.train()
        assert trainer.config.gradient_accumulation_steps == 4


# =============================================================================
# Test 11: LLM Fine-tuning (config only — skipped if no GPU)
# =============================================================================

def test_llm_finetune_config():
    config = TransformerConfig(
        model_name_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_peft=True,
        peft_config={"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"], "task_type": "CAUSAL_LM"},
        quantization_type="4bit",
        max_epochs=1,
    )
    assert config.use_peft
    assert config.peft_config["r"] == 8
    assert "q_proj" in config.peft_config["target_modules"]


# =============================================================================
# Test 12: Pretrain Minimal
# =============================================================================

class _MinimalGPT(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        block = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(block, num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, inputs, **kwargs):
        batch, seq_len = inputs.shape
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(inputs) + self.position_embedding(positions)
        x = self.transformer(x)
        return self.head(x)


class _PretrainDataset(Dataset):
    def __init__(self, vocab_size=100, seq_len=16, size=200):
        self.size = size
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (size, seq_len + 1))
        self.vocab_size = vocab_size

    def __len__(self): return self.size
    def __getitem__(self, idx):
        return {"inputs": self.data[idx, :-1], "labels": self.data[idx, 1:]}


class TestPretrain:
    @pytest.fixture
    def pretrain_model(self):
        return _MinimalGPT(vocab_size=100, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=16)

    @pytest.fixture
    def pretrain_loader(self):
        dataset = _PretrainDataset(vocab_size=100, seq_len=16, size=200)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def test_pretrain_3_epochs(self, pretrain_model, pretrain_loader):
        def lm_criterion(logits, labels):
            B, T, V = logits.shape
            return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

        def forward_fn(model, batch):
            logits = model(batch["inputs"])
            loss = lm_criterion(logits, batch["labels"])
            return loss, logits

        config = SelgisConfig(max_epochs=3, batch_size=32, lr_finder_enabled=False, logging_steps=10, nan_recovery=True)
        trainer = Trainer(
            model=pretrain_model, config=config,
            train_dataloader=pretrain_loader, eval_dataloader=pretrain_loader,
            forward_fn=forward_fn,
        )
        trainer.train()

        history_cb = next(cb for cb in trainer.callbacks if isinstance(cb, HistoryCallback))
        assert len(history_cb.history) >= 2
        first_loss = history_cb.history[0]["metrics"]["loss"]
        last_loss = history_cb.history[-1]["metrics"]["loss"]
        assert last_loss < first_loss, f"Loss {first_loss:.4f} -> {last_loss:.4f}"

    def test_pretrain_15_epochs(self, pretrain_model):
        dataset = _PretrainDataset(vocab_size=100, seq_len=16, size=500)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        def lm_criterion(logits, labels):
            B, T, V = logits.shape
            return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

        def forward_fn(model, batch):
            logits = model(batch["inputs"])
            loss = lm_criterion(logits, batch["labels"])
            return loss, logits

        config = SelgisConfig(max_epochs=15, batch_size=32, lr_finder_enabled=False, logging_steps=10, nan_recovery=True)
        trainer = Trainer(
            model=pretrain_model, config=config,
            train_dataloader=loader, eval_dataloader=loader,
            forward_fn=forward_fn,
        )
        trainer.train()

        history_cb = next(cb for cb in trainer.callbacks if isinstance(cb, HistoryCallback))
        first_loss = history_cb.history[0]["metrics"]["loss"]
        last_loss = history_cb.history[-1]["metrics"]["loss"]
        assert last_loss < first_loss * 0.95, f"Loss {first_loss:.4f} -> {last_loss:.4f} (insufficient decrease)"


# =============================================================================
# Test 13: Rollback Procedure
# =============================================================================

class TestRollback:
    def test_rollback_on_nan(self):
        import torch.optim as optim
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        config = SelgisConfig(
            max_epochs=5, batch_size=16, nan_recovery=True,
            min_history_len=5, spike_threshold=2.0, state_storage="memory",
        )
        device = torch.device("cpu")
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)

        from selgis import SelgisCore
        core = SelgisCore(model, optimizer, scheduler, config, device)

        initial_state = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}

        model.train()
        for i in range(3):
            x = torch.randn(16, 10)
            y = torch.randint(0, 2, (16,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            assert core.check_loss(loss), "Normal loss should pass"

        nan_loss = torch.tensor(float("nan"))
        assert not core.check_loss(nan_loss), "NaN should trigger rollback"

        for name, param in model.named_parameters():
            if name in initial_state:
                assert torch.allclose(param, initial_state[name].to(device), atol=1e-6), f"{name} not rolled back"

    def test_spike_detection(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        config = SelgisConfig(max_epochs=5, batch_size=16, nan_recovery=True, min_history_len=5, spike_threshold=2.0)
        device = torch.device("cpu")
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)

        from selgis import SelgisCore
        core = SelgisCore(model, optimizer, scheduler, config, device)

        for i in range(10):
            core.check_loss(torch.tensor(1.0))

        spike = torch.tensor(5.0)
        assert not core.check_loss(spike), "Spike should trigger rollback"

    def test_nan_recovery_disabled(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        config = SelgisConfig(max_epochs=5, batch_size=16, nan_recovery=False)
        device = torch.device("cpu")
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)

        from selgis import SelgisCore
        core = SelgisCore(model, optimizer, scheduler, config, device)
        assert core.check_loss(torch.tensor(float("nan"))), "nan_recovery=False should skip check"


# =============================================================================
# Test 14: Self-healing
# =============================================================================

class TestSelfHealing:
    def test_self_healing_with_nan(self):
        class BreakableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(10, 2)
                self.break_mode = False
            def forward(self, inputs, **kwargs):
                if self.break_mode:
                    out = self.net(inputs)
                    return out * float("nan")
                return self.net(inputs)

        model = BreakableModel()

        class DS(Dataset):
            def __init__(self):
                self.X = torch.randn(100, 10)
                self.y = torch.randint(0, 2, (100,))
            def __len__(self): return 100
            def __getitem__(self, i): return {"inputs": self.X[i], "labels": self.y[i]}

        loader = DataLoader(DS(), batch_size=16, shuffle=True)
        config = SelgisConfig(max_epochs=3, batch_size=16, nan_recovery=True, min_history_len=3, spike_threshold=2.0, logging_steps=10)
        trainer = Trainer(model=model, config=config, train_dataloader=loader, eval_dataloader=loader, criterion=nn.CrossEntropyLoss())

        class NaNInjector(Callback):
            def __init__(self):
                self.injected = False
            def on_step_begin(self, trainer, step):
                if step == 3 and not self.injected:
                    trainer.model.break_mode = True
                    self.injected = True
            def on_step_end(self, trainer, step, loss):
                if self.injected and step == 3:
                    trainer.model.break_mode = False

        trainer.callbacks.insert(0, NaNInjector())
        trainer.train()

        for param in model.parameters():
            assert not torch.isnan(param.data).any(), "Model has NaN after self-healing"


# =============================================================================
# Test 15: Gradient Checkpointing
# =============================================================================

def test_gradient_checkpointing():
    class DummyBlock(nn.Module):
        def forward(self, x, **kwargs): return x

    model = nn.Sequential(DummyBlock(), DummyBlock(), DummyBlock())
    manager = GradientCheckpointingManager(checkpoint_interval=2, use_reentrant=False)
    model = manager.apply_to_model(model)
    stats = manager.get_stats(model)
    assert isinstance(stats, dict)
    manager.remove_from_model(model)


# =============================================================================
# Test 16: Chunked Cross-Entropy
# =============================================================================

class TestChunkedCrossEntropy:
    def test_chunked_matches_standard(self):
        B, T, V = 4, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))

        chunked = ChunkedCrossEntropyLoss(chunk_size=32)
        loss_chunked = chunked(logits, targets)
        loss_standard = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        assert torch.allclose(loss_chunked, loss_standard, atol=1e-5)

    def test_2d_input(self):
        B, V = 4, 100
        logits = torch.randn(B, V)
        targets = torch.randint(0, V, (B,))
        chunked = ChunkedCrossEntropyLoss(chunk_size=32)
        loss = chunked(logits, targets)
        assert loss is not None

    def test_label_smoothing(self):
        B, T, V = 4, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        smoothed = ChunkedCrossEntropyLoss(chunk_size=32, label_smoothing=0.1)
        loss = smoothed(logits, targets)
        assert loss is not None

    def test_ignore_index(self):
        B, T, V = 4, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        targets[:, 0] = -100
        ce = ChunkedCrossEntropyLoss(chunk_size=32, ignore_index=-100)
        loss = ce(logits, targets)
        assert loss is not None

    def test_reduction_sum(self):
        B, T, V = 4, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        ce = ChunkedCrossEntropyLoss(chunk_size=32, reduction="sum")
        loss = ce(logits, targets)
        assert loss is not None

    def test_reduction_none(self):
        B, T, V = 4, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        ce = ChunkedCrossEntropyLoss(chunk_size=32, reduction="none")
        loss = ce(logits, targets)
        assert loss.shape == (B, T)

    def test_v2_wrapper(self):
        B, T, V = 4, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        v2 = CrossEntropyLossV2(chunk_size=0)
        loss = v2(logits, targets)
        assert loss is not None


# =============================================================================
# Test 17: TransformerTrainer Config
# =============================================================================

def test_transformer_trainer_config():
    config = TransformerConfig(
        model_name_or_path="test-model",
        use_peft=True,
        peft_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
        quantization_type="no",
        max_epochs=2,
        chunked_ce=True,
        ce_chunk_size=1024,
        flash_attention=False,
        gradient_checkpointing=True,
    )
    assert config.use_peft
    assert config.chunked_ce


# =============================================================================
# Test 18: LRFinder
# =============================================================================

class TestLRFinder:
    def test_lr_finder_integration(self, simple_model, simple_loader):
        config = SelgisConfig(max_epochs=1, batch_size=16, lr_finder_enabled=True)
        def forward_fn(model, batch):
            logits = model(batch["inputs"])
            loss = nn.CrossEntropyLoss()(logits, batch["labels"])
            return loss, logits
        trainer = Trainer(
            model=simple_model, config=config,
            train_dataloader=simple_loader, eval_dataloader=simple_loader,
            forward_fn=forward_fn,
        )
        assert trainer.initial_lr > 0


# =============================================================================
# Test 19: SmartScheduler
# =============================================================================

class TestSmartScheduler:
    def test_scheduler_creation(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        config = SelgisConfig(max_epochs=10, batch_size=16)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)
        assert scheduler is not None

    def test_scheduler_warmup(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        config = SelgisConfig(max_epochs=10, batch_size=16, warmup_epochs=3)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)
        initial_lr = optimizer.param_groups[0]["lr"]
        lr = scheduler.step_epoch(epoch=1)
        assert scheduler._lr_scale == 1.0
        assert lr > 0

    def test_scheduler_warmup_completion(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        config = SelgisConfig(max_epochs=10, batch_size=16, warmup_epochs=3)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)
        lr = scheduler.step_epoch(epoch=3)
        assert abs(lr - 0.01) < 1e-6, f"Expected ~0.01 after warmup, got {lr}"

    def test_scheduler_cosine_decay(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        config = SelgisConfig(max_epochs=10, batch_size=16, warmup_epochs=0)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)
        lr = scheduler.step_epoch(epoch=5)
        assert lr < 0.01, f"Expected LR < 0.01 after cosine decay, got {lr}"

    def test_scheduler_reduce_lr(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        config = SelgisConfig(max_epochs=10, batch_size=16, warmup_epochs=0)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)
        scheduler.reduce_lr(factor=0.5)
        assert scheduler._lr_scale == 0.5

    def test_scheduler_surge_lr(self):
        import torch.optim as optim
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        config = SelgisConfig(max_epochs=10, batch_size=16, warmup_epochs=0)
        scheduler = SmartScheduler(optimizer, initial_lr=0.01, config=config)
        scheduler.reduce_lr(factor=0.5)
        assert scheduler._lr_scale == 0.5
        new_lr = scheduler.surge_lr(factor=2.0)
        assert new_lr > 0


# =============================================================================
# Test 20: Checkpoint round-trip
# =============================================================================

class TestCheckpointRoundTrip:
    def test_checkpoint_save_load(self, tmp_output_dir):
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        model_state = model.state_dict()
        ckpt_path = tmp_output_dir / "test_checkpoint.pt"
        torch.save({"model_state_dict": model_state}, ckpt_path)
        assert ckpt_path.exists()

        loaded = torch.load(ckpt_path, weights_only=True)
        restored_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        restored_model.load_state_dict(loaded["model_state_dict"])

        x = torch.randn(4, 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = restored_model(x)
        assert torch.allclose(out1, out2), "Checkpoint round-trip: model outputs differ"


# =============================================================================
# Test 21: CLI integration
# =============================================================================

class TestCLI:
    def test_cli_version(self):
        result = subprocess.run([sys.executable, "-m", "selgis.cli", "--version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_cli_help(self):
        result = subprocess.run([sys.executable, "-m", "selgis.cli", "--help"], capture_output=True, text=True)
        assert result.returncode == 0

    def test_cli_device(self):
        result = subprocess.run([sys.executable, "-m", "selgis.cli", "device"], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_cli_train_no_config(self):
        result = subprocess.run(
            [sys.executable, "-m", "selgis.cli", "train"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0


# =============================================================================
# Test 22: Edge cases
# =============================================================================

class TestEdgeCases:
    def test_empty_dataloader(self):
        class EmptyDataset(Dataset):
            def __len__(self): return 0
            def __getitem__(self, idx): raise IndexError

        config = DatasetConfig(data_type="custom", custom_kwargs={"dataset": EmptyDataset()}, batch_size=16, num_workers=0, train_split=0.5)
        with pytest.raises((ValueError, RuntimeError)):
            create_dataloaders(config)

    def test_single_batch_dataset(self):
        class SingleBatchDataset(Dataset):
            def __init__(self):
                self.X = torch.randn(5, 10)
                self.y = torch.randint(0, 2, (5,))
            def __len__(self): return 5
            def __getitem__(self, idx): return {"inputs": self.X[idx], "labels": self.y[idx]}

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
            def forward(self, inputs, **kwargs):
                return self.net(inputs)

        model = MLP()
        loader = DataLoader(SingleBatchDataset(), batch_size=16)
        config = SelgisConfig(max_epochs=1, batch_size=16, lr_finder_enabled=False)
        trainer = Trainer(model=model, config=config, train_dataloader=loader, eval_dataloader=loader, criterion=nn.CrossEntropyLoss())
        trainer.train()
        assert True

    def test_seed_reproducibility(self):
        seed_everything(42)
        x1 = torch.randn(5, 5)
        seed_everything(42)
        x2 = torch.randn(5, 5)
        assert torch.allclose(x1, x2)

    def test_forward_fn_custom(self, simple_model):
        class DS(Dataset):
            def __len__(self): return 20
            def __getitem__(self, i): return {"inputs": torch.randn(10), "labels": torch.randint(0, 2, (1,)).squeeze()}

        loader = DataLoader(DS(), batch_size=10)
        config = SelgisConfig(max_epochs=1, batch_size=10, lr_finder_enabled=False)

        def forward_fn(model, batch):
            logits = model(batch["inputs"])
            loss = F.cross_entropy(logits, batch["labels"])
            return loss, logits

        trainer = Trainer(model=simple_model, config=config, train_dataloader=loader, eval_dataloader=loader, forward_fn=forward_fn)
        trainer.train()
        assert True
