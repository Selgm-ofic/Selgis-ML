import gc
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cuda_available():
    return torch.cuda.is_available()


@pytest.fixture(autouse=True)
def cleanup():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


@pytest.fixture
def simple_model():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


@pytest.fixture
def simple_dataset():
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            self.X = torch.randn(size, 10)
            self.y = torch.randint(0, 2, (size,))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"inputs": self.X[idx], "labels": self.y[idx]}

    return SimpleDataset(100)


@pytest.fixture
def simple_loader(simple_dataset):
    return torch.utils.data.DataLoader(simple_dataset, batch_size=16, shuffle=True)
