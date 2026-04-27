"""Command-line interface for the SELGIS training framework.

Usage::

    selgis --version
    selgis device
    selgis train [--config path/to/config.yaml]
    selgis test
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path


def _cmd_version() -> int:
    """Print package version and exit."""
    from selgis import __version__

    print(__version__)
    return 0


def _cmd_device() -> int:
    """Print compute device info (CUDA/MPS/CPU)."""
    from selgis.utils import get_device

    get_device("auto")
    return 0


def _cmd_test() -> int:
    """Run the complete test suite."""
    import subprocess

    result = subprocess.run([sys.executable, "-m", "test_selgis"])
    return result.returncode


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return a plain dict.

    Falls back to JSON if PyYAML is not installed.

    Args:
        path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        SystemExit: If the file cannot be parsed.
    """
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            print(
                "Error: PyYAML is required for .yaml configs. Install with: pip install pyyaml",
                file=sys.stderr,
            )
            sys.exit(1)
    elif path.suffix == ".json":
        import json

        data = json.loads(text)
    else:
        try:
            import yaml

            data = yaml.safe_load(text)
        except ImportError:
            import json

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                print(
                    f"Error: cannot parse {path}. Install PyYAML or use .json format.",
                    file=sys.stderr,
                )
                sys.exit(1)

    if not isinstance(data, dict):
        print(
            f"Error: config must be a YAML/JSON mapping, got {type(data).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def _build_config(raw: dict):
    """Build a SelgisConfig or TransformerConfig from a raw dict.

    If the dict contains Transformer-specific keys (``model_name_or_path``,
    ``use_peft``, ``problem_type``, etc.) a ``TransformerConfig`` is
    returned; otherwise a ``SelgisConfig``.

    Args:
        raw: Flat configuration dictionary.

    Returns:
        A ``SelgisConfig`` or ``TransformerConfig`` instance.
    """
    from selgis import SelgisConfig, TransformerConfig
    from selgis.datasets import DatasetConfig

    transformer_keys = {
        "model_name_or_path",
        "use_peft",
        "peft_config",
        "problem_type",
        "optimizer_type",
        "quantization_type",
        "gradient_checkpointing",
        "gc_checkpoint_interval",
        "chunked_ce",
        "ce_chunk_size",
        "flash_attention",
        "num_labels",
        "trust_remote_code",
        "device_map",
    }

    trainer_cls = TransformerConfig if transformer_keys.intersection(raw.keys()) else SelgisConfig
    trainer_field_names = {f.name for f in fields(trainer_cls)}
    dataset_field_names = {f.name for f in fields(DatasetConfig)}

    trainer_raw = {k: v for k, v in raw.items() if k in trainer_field_names}
    dataset_raw = {k: v for k, v in raw.items() if k in dataset_field_names}

    unknown_keys = sorted(
        k for k in raw.keys() if k not in trainer_field_names and k not in dataset_field_names
    )
    if unknown_keys:
        print("[WARN] Ignoring unknown config keys: " + ", ".join(unknown_keys))

    return trainer_cls(**trainer_raw), dataset_raw


def _build_dataloaders(config, dataset_raw: dict | None = None):
    """Create train/eval DataLoaders from config fields or defaults.

    Looks for ``data_path`` / ``train_path`` in *config*. When found,
    builds real data loaders via ``create_dataloaders``.  Otherwise
    returns a synthetic demo loader pair.

    Args:
        config: A ``SelgisConfig`` or ``TransformerConfig``.

    Returns:
        Tuple ``(train_loader, eval_loader)``.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset_raw = dataset_raw or {}
    data_path = dataset_raw.get("data_path")
    train_path = dataset_raw.get("train_path")

    if data_path or train_path:
        from selgis import DatasetConfig, create_dataloaders

        ds_kwargs = dict(dataset_raw)
        ds_kwargs.setdefault("batch_size", config.batch_size)
        ds_kwargs.setdefault("num_workers", 0)
        if hasattr(config, "max_length"):
            ds_kwargs.setdefault("max_length", config.max_length)
        ds_kwargs.setdefault("data_type", "text")

        tokenizer = _try_load_tokenizer(config)
        if tokenizer is not None:
            ds_kwargs.setdefault("tokenizer", tokenizer)

        ds_config = DatasetConfig(**ds_kwargs)
        return create_dataloaders(ds_config)

    print("[INFO] No data_path in config — using synthetic demo data")
    from selgis.utils import seed_everything

    seed_everything(getattr(config, "seed", 42))

    X = torch.randn(200, 10)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    return loader, loader


def _try_load_tokenizer(config):
    """Attempt to load a HuggingFace tokenizer from config.

    Args:
        config: Training configuration with optional
            ``model_name_or_path`` field.

    Returns:
        A tokenizer instance, or ``None`` if unavailable.
    """
    model_name = getattr(config, "model_name_or_path", None)
    if not model_name:
        return None

    try:
        from transformers import AutoTokenizer

        trust = getattr(config, "trust_remote_code", False)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except (ImportError, OSError) as e:
        import logging

        logging.getLogger(__name__).debug(f"Could not load tokenizer: {e}")
        return None
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Error loading tokenizer: {e}")
        return None


def _cmd_train(args: argparse.Namespace) -> int:
    """Run training from a config file or the built-in demo.

    When ``--config`` is provided the file is parsed, an appropriate
    config object is built, data loaders are created (from paths in
    the config or synthetic data), and training runs via ``Trainer``
    or ``TransformerTrainer``.

    Without ``--config`` a minimal demo on synthetic data is executed.
    """
    if getattr(args, "config", None):
        return _train_from_config(Path(args.config))

    return _train_demo()


def _train_from_config(config_path: Path) -> int:
    """Parse a YAML/JSON config and run full training.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    if not config_path.exists():
        print(
            f"Error: config file not found: {config_path}",
            file=sys.stderr,
        )
        return 1


    from selgis import TransformerConfig

    raw = _load_yaml(config_path)
    print(f"[INFO] Loaded config from {config_path}")

    try:
        config, dataset_raw = _build_config(raw)
    except (TypeError, ValueError) as exc:
        print(f"Error building config: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Config type: {type(config).__name__}")

    train_loader, eval_loader = _build_dataloaders(config, dataset_raw=dataset_raw)

    if isinstance(config, TransformerConfig) and config.model_name_or_path:
        return _train_transformer(config, train_loader, eval_loader)

    return _train_pytorch(config, train_loader, eval_loader)


def _train_transformer(config, train_loader, eval_loader) -> int:
    """Run TransformerTrainer with the given config and loaders.

    Args:
        config: A ``TransformerConfig`` instance.
        train_loader: Training data loader.
        eval_loader: Evaluation data loader.

    Returns:
        Exit code.
    """
    from selgis import TransformerTrainer

    tokenizer = _try_load_tokenizer(config)

    forward_fn = None
    if config.problem_type == "causal_lm":
        forward_fn = _causal_lm_forward

    try:
        trainer = TransformerTrainer(
            model_or_path=config.model_name_or_path,
            config=config,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            tokenizer=tokenizer,
            forward_fn=forward_fn,
        )
    except Exception as exc:
        print(f"Error creating TransformerTrainer: {exc}", file=sys.stderr)
        return 1

    metrics = trainer.train()
    _print_final_metrics(metrics)

    output_dir = getattr(config, "output_dir", "./output")
    trainer.save_pretrained(f"{output_dir}/final_model")

    return 0


def _train_pytorch(config, train_loader, eval_loader) -> int:
    """Run Trainer on a simple model with the given config and loaders.

    When no ``model_name_or_path`` is set, a small MLP is used as a
    placeholder so that the training pipeline can be validated.

    Args:
        config: A ``SelgisConfig`` instance.
        train_loader: Training data loader.
        eval_loader: Evaluation data loader.

    Returns:
        Exit code.
    """
    import torch.nn as nn

    from selgis import Trainer

    try:
        sample = next(iter(train_loader))
    except StopIteration:
        print("Error: train_loader is empty", file=sys.stderr)
        return 1

    if isinstance(sample, (tuple, list)):
        if len(sample) == 0:
            print("Error: train_loader returned empty batch", file=sys.stderr)
            return 1
        input_dim = sample[0].shape[-1]
    elif hasattr(sample, "keys"):
        first_key = next(iter(sample.keys()))
        input_dim = sample[first_key].shape[-1]
    else:
        input_dim = sample.shape[-1]

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        criterion=nn.CrossEntropyLoss(),
    )

    metrics = trainer.train()
    _print_final_metrics(metrics)

    output_dir = getattr(config, "output_dir", "./output")
    trainer.save_model(f"{output_dir}/model.pt")

    return 0


def _train_demo() -> int:
    """Run a minimal demo training on synthetic data.

    Returns:
        Exit code.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from selgis import SelgisConfig, Trainer
    from selgis.utils import seed_everything

    print("[INFO] Running demo training (synthetic data)...")
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
        max_epochs=3,
        lr_finder_enabled=False,
        nan_recovery=True,
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=loader,
        eval_dataloader=loader,
        criterion=nn.CrossEntropyLoss(),
    )

    metrics = trainer.train()
    _print_final_metrics(metrics)
    print("[INFO] Demo finished.")
    return 0


def _causal_lm_forward(model, batch):
    """Default forward function for causal language models.

    Expects batch to contain ``input_ids`` and ``labels``.

    Args:
        model: A HuggingFace causal LM.
        batch: Dict with ``input_ids`` and ``labels``.

    Returns:
        Tuple ``(loss, logits)``.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
    )
    return outputs.loss, outputs.logits


def _print_final_metrics(metrics: dict) -> None:
    """Print final training metrics summary.

    Args:
        metrics: Dictionary of metric names to values.
    """
    if not metrics:
        return
    print("\n" + "=" * 50)
    print("Final Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 50)


def main() -> int:
    """Entry point for the ``selgis`` command."""
    parser = argparse.ArgumentParser(
        prog="selgis",
        description=("SELGIS — Universal Training Framework for PyTorch and Transformers"),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    subparsers.add_parser("version", help="Show version")
    subparsers.add_parser(
        "device",
        help="Show compute device (CUDA/MPS/CPU)",
    )
    subparsers.add_parser(
        "test",
        help="Run the complete test suite (16 tests)",
    )

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
    if args.command == "test":
        return _cmd_test()
    if args.command == "train":
        return _cmd_train(args)
    if args.command is None:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
