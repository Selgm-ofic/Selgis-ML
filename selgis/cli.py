"""
Command-line interface for the SELGIS training framework.
Usage:
selgis --version
selgis device
selgis train --config path/to/config.yaml
"""
import argparse
import sys
from pathlib import Path


def _cmd_version() -> int:
    """Print package version and exit."""
    from selgis import __version__
    print(__version__)
    return 0


def _cmd_device() -> int:
    """Print compute device info (CUDA/MPS/CPU)."""
    from selgis.utils import get_device
    # Here we explicitly want to print info
    get_device("auto")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Run training from config file or default demo."""
    if getattr(args, "config", None):
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            return 1
        print(f"Config-based training: {config_path}")
        print("Tip: implement config loading in your project (YAML/JSON).")
        return 0

    from selgis import Trainer, SelgisConfig
    from selgis.utils import seed_everything
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("Running minimal demo (synthetic data)...")
    
    # Removed get_device("auto") here to avoid double printing.
    # Trainer will call it internally and print the info once.
    
    seed_everything(42)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    X = torch.randn(200, 10)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    config = SelgisConfig(
        max_epochs=2,
        lr_finder_enabled=False,
        nan_recovery=True,
    )
    
    # Trainer initializes device and prints info automatically
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        eval_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )
    trainer.train()
    print("Demo finished.")
    return 0


def main() -> int:
    """Entry point for the selgis command."""
    parser = argparse.ArgumentParser(
        prog="selgis",
        description="SELGIS - Universal Training Framework for PyTorch and Transformers",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("version", help="Show version")
    subparsers.add_parser("device", help="Show compute device (CUDA/MPS/CPU)")

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to training config file (YAML/JSON)",
    )

    args = parser.parse_args()

    if args.version or args.command == "version":
        return _cmd_version()
    if args.command == "device":
        return _cmd_device()
    if args.command == "train":
        return _cmd_train(args)
    if args.command is None:
        parser.print_help()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())