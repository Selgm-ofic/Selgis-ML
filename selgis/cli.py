"""Command-line interface for the SELGIS training framework.

Usage::

    selgis --version
    selgis device
    selgis train [--config path/to/config.yaml] [--set key=value ...]
    selgis config validate [--config path/to/config.yaml]
    selgis test
"""

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _cmd_version() -> int:
    from selgis import __version__
    print(__version__)
    return 0


def _cmd_device() -> int:
    from selgis.utils import get_device
    device = get_device("auto")
    print(device)
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    import pytest
    args_list = ["-x", "-v", "tests/"] if args.verbose else ["-x", "tests/"]
    return pytest.main(args_list)


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            print("Error: PyYAML is required for .yaml configs. Install with: pip install pyyaml", file=sys.stderr)
            return {}
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
                print(f"Error: cannot parse {path}. Install PyYAML or use .json format.", file=sys.stderr)
                return {}

    if not isinstance(data, dict):
        print(f"Error: config must be a YAML/JSON mapping, got {type(data).__name__}", file=sys.stderr)
        return {}

    return data


def _get_transformer_keys() -> set:
    from selgis import TransformerConfig
    return {f.name for f in fields(TransformerConfig)}


def _get_selgis_keys() -> set:
    from selgis import SelgisConfig
    return {f.name for f in fields(SelgisConfig)}


def _get_dataset_keys() -> set:
    from selgis.datasets import DatasetConfig
    return {f.name for f in fields(DatasetConfig)}


def _build_config(raw: dict, overrides: list[str] | None = None):
    from selgis import SelgisConfig, TransformerConfig

    if overrides:
        for override in overrides:
            if "=" not in override:
                print(f"Error: --set must be in key=value format, got '{override}'", file=sys.stderr)
                return None, {}
            key, value = override.split("=", 1)
            raw[key] = _parse_value(value)

    transformer_keys = _get_transformer_keys()
    trainer_cls = TransformerConfig if transformer_keys.intersection(raw.keys()) else SelgisConfig
    trainer_field_names = _get_selgis_keys() | _get_transformer_keys()
    dataset_field_names = _get_dataset_keys()

    trainer_raw = {k: v for k, v in raw.items() if k in trainer_field_names}
    dataset_raw = {k: v for k, v in raw.items() if k in dataset_field_names}

    unknown_keys = sorted(k for k in raw.keys() if k not in trainer_field_names and k not in dataset_field_names)
    if unknown_keys:
        print("[WARN] Ignoring unknown config keys: " + ", ".join(unknown_keys))

    try:
        return trainer_cls(**trainer_raw), dataset_raw
    except (TypeError, ValueError) as exc:
        print(f"Error building config: {exc}", file=sys.stderr)
        return None, {}


def _parse_value(value: str) -> Any:
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _build_dataloaders(config, dataset_raw: dict | None = None):
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
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return loader, loader


def _try_load_tokenizer(config):
    model_name = getattr(config, "model_name_or_path", None)
    if not model_name:
        return None

    try:
        from transformers import AutoTokenizer
        trust = getattr(config, "trust_remote_code", False)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except (ImportError, OSError) as e:
        logger.debug(f"Could not load tokenizer: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading tokenizer: {e}")
        return None


def _cmd_config(args: argparse.Namespace) -> int:

    if args.config_action == "validate":
        if not getattr(args, "config", None):
            print("Error: --config is required for 'config validate'", file=sys.stderr)
            return 1

        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            return 1

        raw = _load_yaml(config_path)
        if not raw:
            return 1

        config, dataset_raw = _build_config(raw)
        if config is None:
            return 1

        print(f"[OK] Config is valid: {type(config).__name__}")
        print(f"[OK] Dataset config keys: {list(dataset_raw.keys())}")

        unknown = sorted(k for k in raw.keys()
                         if k not in _get_selgis_keys() | _get_transformer_keys() | _get_dataset_keys())
        if unknown:
            print(f"[WARN] Unknown keys: {unknown}")

        return 0

    print(f"Unknown config action: {args.config_action}", file=sys.stderr)
    return 1


def _cmd_train(args: argparse.Namespace) -> int:
    if getattr(args, "config", None):
        return _train_from_config(Path(args.config), overrides=args.set)

    return _train_demo()


def _train_from_config(config_path: Path, overrides: list[str] | None = None) -> int:
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    from selgis import TransformerConfig

    raw = _load_yaml(config_path)
    if not raw:
        return 1
    print(f"[INFO] Loaded config from {config_path}")

    config, dataset_raw = _build_config(raw, overrides)
    if config is None:
        return 1

    print(f"[INFO] Config type: {type(config).__name__}")

    train_loader, eval_loader = _build_dataloaders(config, dataset_raw=dataset_raw)

    if isinstance(config, TransformerConfig) and config.model_name_or_path:
        return _train_transformer(config, train_loader, eval_loader)

    return _train_pytorch(config, train_loader, eval_loader)


def _train_transformer(config, train_loader, eval_loader) -> int:
    from selgis import TransformerTrainer

    tokenizer = _try_load_tokenizer(config)
    forward_fn = _causal_lm_forward if config.problem_type == "causal_lm" else None

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

    try:
        metrics = trainer.train()
    except Exception as exc:
        print(f"Error during Transformer training: {exc}", file=sys.stderr)
        return 1

    _print_final_metrics(metrics)
    output_dir = getattr(config, "output_dir", "./output")
    trainer.save_pretrained(f"{output_dir}/final_model")
    return 0


def _train_pytorch(config, train_loader, eval_loader) -> int:
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
        nn.Linear(input_dim, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )

    trainer = Trainer(
        model=model, config=config,
        train_dataloader=train_loader, eval_dataloader=eval_loader,
        criterion=nn.CrossEntropyLoss(),
    )

    try:
        metrics = trainer.train()
    except Exception as exc:
        print(f"Error during PyTorch training: {exc}", file=sys.stderr)
        return 1

    _print_final_metrics(metrics)
    output_dir = getattr(config, "output_dir", "./output")
    trainer.save_model(f"{output_dir}/model.pt")
    return 0


def _train_demo() -> int:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from selgis import SelgisConfig, Trainer
    from selgis.utils import seed_everything

    print("[INFO] Running demo training (synthetic data)...")
    seed_everything(42)

    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    X = torch.randn(200, 10)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    config = SelgisConfig(max_epochs=3, lr_finder_enabled=False, nan_recovery=True)
    trainer = Trainer(model=model, config=config, train_dataloader=loader, eval_dataloader=loader, criterion=nn.CrossEntropyLoss())

    try:
        metrics = trainer.train()
    except Exception as exc:
        print(f"Error during demo training: {exc}", file=sys.stderr)
        return 1

    _print_final_metrics(metrics)
    print("[INFO] Demo finished.")
    return 0


def _causal_lm_forward(model, batch):
    has_device_map = bool(getattr(model, "hf_device_map", None))
    if has_device_map:
        batch = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items()}
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
    )
    return outputs.loss, outputs.logits


def _print_final_metrics(metrics: dict) -> None:
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
    parser = argparse.ArgumentParser(
        prog="selgis",
        description="SELGIS — Universal Training Framework for PyTorch and Transformers",
        epilog="Examples:\n"
               "  selgis --version\n"
               "  selgis device\n"
               "  selgis train\n"
               "  selgis train --config config.yaml\n"
               "  selgis train --config config.yaml --set max_epochs=10 batch_size=64\n"
               "  selgis config validate --config config.yaml\n"
               "  selgis test\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("device", help="Show compute device (CUDA/MPS/CPU)")
    subparsers.add_parser("test", help="Run the complete test suite with pytest")

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, default=None, metavar="PATH", help="Path to training config file (YAML/JSON)")
    train_parser.add_argument("--set", type=str, default=None, nargs="*", metavar="KEY=VALUE", help="Override config fields from CLI")

    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_sub = config_parser.add_subparsers(dest="config_action", help="Config action")
    validate_parser = config_sub.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("--config", type=str, required=True, metavar="PATH", help="Path to config file to validate")

    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.ERROR, stream=sys.stdout, format="%(message)s", force=True)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(message)s", force=True)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s", force=True)

    if args.version:
        return _cmd_version()

    if args.command == "device":
        return _cmd_device()
    if args.command == "test":
        return _cmd_test(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "config":
        return _cmd_config(args)
    if args.command is None:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
