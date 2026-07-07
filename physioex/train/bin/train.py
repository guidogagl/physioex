"""CLI entry point for training a model on PhysioEx datasets.

Supports BOTH:
  - Legacy preprocessed datasets (``--dataset_type legacy``)
  - New raw-EDF lazy-loading datasets (``--dataset_type raw``, default)

Examples::

    # TinySleepNet on HMC with time-domain preprocessing (raw waveforms)
    python -m physioex.train.bin.train \\
        --model physioex.models.tinysleepnet:TinySleepNet \\
        --dataset hmc \\
        --pipelines time_domain \\
        --channels EEG EOG EMG \\
        --model_kwargs '{"n_classes": 5, "in_chan": 3}' \\
        --max_epochs 20 --train_batch_size 64 --gpu_id 0

    # SeqSleepNet on HMC with time-frequency preprocessing (spectrograms)
    python -m physioex.train.bin.train \\
        --model physioex.models.seqsleepnet:SeqSleepNet \\
        --dataset hmc \\
        --pipelines time_frequency \\
        --channels EEG EOG EMG \\
        --model_kwargs '{"n_classes": 5, "in_chan": 3}' \\
        --max_epochs 20 --train_batch_size 64 --lr 1e-4 --gpu_id 0
"""
import argparse
import importlib
import json
import os

import yaml

from physioex.train.trainer import Trainer
from physioex.train.logger import add_logger_cli_args, logger_train_kwargs


def _import_class(spec: str):
    """Import a class from 'module.path:ClassName' spec."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _parse_model_kwargs(raw: str) -> dict:
    """Parse JSON or YAML string into a dict."""
    try:
        return json.loads(raw)
    except Exception:
        return yaml.safe_load(raw) or {}


def train_script():
    parser = argparse.ArgumentParser(
        description="Train a PhysioEx model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model", type=str, required=True, help="Model class: 'module.path:ClassName'"
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help="JSON/YAML model constructor kwargs",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g. hmc, sleepedf, dcsm, mesa)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Override default data root directory",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["EEG", "EOG", "EMG"],
        help="Channels to load (modality names or physical names)",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="time_domain",
        help="Preset pipeline name: raw, time_domain, time_frequency, "
        "seqsleepnet, eeg, emg, etc.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=21,
        help="Epoch sequence length (L). -1 for full recordings.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Override default cache directory"
    )

    # Training
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=None)

    # Logging / experiment tracking
    add_logger_cli_args(parser)

    # Optional config overlay
    parser.add_argument(
        "--config", type=str, default=None, help="YAML config file (overrides CLI args)"
    )

    args = parser.parse_args()

    # YAML config overlay
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        for k, v in config.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)

    # Dataset (new raw-EDF system) — build BEFORE model so we can infer in_chan
    from physioex.data.datasets import get_dataset
    from physioex.data.presets import get_preset

    dataset_class = get_dataset(args.dataset)
    ds_kwargs = dict(
        channels=args.channels,
        pipelines=args.pipelines,
        sequence_length=args.seqlen,
        cache_dir=args.cache_dir,
    )
    if args.dataset_root is not None:
        ds_kwargs["root"] = args.dataset_root
    dataset = dataset_class(**ds_kwargs)

    print(
        f"[Info] Dataset: {args.dataset} | Subjects: {dataset.get_n_subjects()} | "
        f"Epochs indexed: {len(dataset)} | Channels: {args.channels} | "
        f"Pipeline: {args.pipelines} | SeqLen: {args.seqlen}"
    )

    # Model — auto-inject in_chan from channel count if not explicitly set
    model_kwargs = _parse_model_kwargs(args.model_kwargs)
    n_channels = len(args.channels)
    if "in_chan" not in model_kwargs and "in_channels" not in model_kwargs:
        model_kwargs["in_chan"] = n_channels
        print(f"[Info] Auto-set in_chan={n_channels} from --channels {args.channels}")
    model_class = _import_class(args.model)
    model = model_class(**model_kwargs)

    # Train
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        fold=args.fold,
        checkpoint_path=args.checkpoint_path,
        gpu_id=args.gpu_id,
        seed=args.seed,
        accumulate_grad_batches=args.accumulate_grad_batches,
        early_stopping_patience=args.early_stopping_patience,
        **logger_train_kwargs(args),
    )

    print(f"[Info] Training complete.")
    return model


if __name__ == "__main__":
    train_script()
