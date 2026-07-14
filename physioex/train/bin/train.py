"""CLI entry point for training a model on PhysioEx datasets.

Uses the raw-EDF lazy-loading data layer (``physioex.data.datasets``), shared
with ``finetune`` and ``test_model`` via ``physioex.train.bin._common``.

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

from physioex.train.bin._common import (
    add_dataset_cli_args,
    apply_config_overlay,
    build_dataset_from_args,
    import_class,
    inject_in_chan,
    parse_kwargs,
)
from physioex.train.trainer import Trainer
from physioex.train.logger import add_logger_cli_args, logger_train_kwargs


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

    # Dataset (shared flags)
    add_dataset_cli_args(parser)

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
    apply_config_overlay(args)

    # Dataset (new raw-EDF layer) — build BEFORE model so we can infer in_chan.
    dataset, n_channels = build_dataset_from_args(args)
    print(
        f"[Info] Dataset: {args.dataset} | Subjects: {dataset.get_n_subjects()} | "
        f"Epochs indexed: {len(dataset)} | Channels: {args.channels} | "
        f"Pipeline: {args.pipelines} | SeqLen: {args.sequence_length}"
    )

    # Model — auto-inject in_chan from channel count if not explicitly set.
    model_kwargs = inject_in_chan(parse_kwargs(args.model_kwargs), n_channels)
    if "in_chan" in model_kwargs:
        print(f"[Info] in_chan={model_kwargs['in_chan']} (from --channels {args.channels})")
    model_class = import_class(args.model)
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

    print("[Info] Training complete.")
    return model


if __name__ == "__main__":
    train_script()
