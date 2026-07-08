"""CLI entry point for fine-tuning a pretrained PhysioEx model."""
import argparse
import importlib
import os
import yaml

from physioex.data.dataset import PhysioExDataset
from physioex.train.trainer import Trainer
from physioex.train.logger import add_logger_cli_args, logger_train_kwargs


def _import_class(spec):
    """Import a class from 'module.path:ClassName' spec."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def finetune_script():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained PhysioEx model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model class spec: 'module.path:ClassName'",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to pretrained checkpoint to finetune from",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True, help="One or more dataset names"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (merged over defaults)",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help="JSON/YAML string with model constructor kwargs",
    )
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="Eval batch size"
    )
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Directory to save finetuned checkpoints",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--selected_channels", nargs="+", default=["EEG"])
    parser.add_argument("--seqlen", type=int, default=21)
    parser.add_argument("--preprocessing", type=str, default="raw")
    add_logger_cli_args(parser)
    args = parser.parse_args()

    # Merge YAML config if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        for k, v in config.items():
            if hasattr(args, k) and v is not None:
                setattr(args, k, v)

    # Parse model_kwargs string (supports JSON and YAML)
    try:
        import json

        model_kwargs = json.loads(args.model_kwargs)
    except Exception:
        model_kwargs = yaml.safe_load(args.model_kwargs) or {}

    # Build dataset
    dataset = PhysioExDataset(
        datasets=args.datasets,
        selected_channels=args.selected_channels,
        seqlen=args.seqlen,
        preprocessing=args.preprocessing,
    )

    # Build model
    model_class = _import_class(args.model)
    model = model_class(**model_kwargs)

    # Load pretrained weights
    model, _, _ = Trainer.load_checkpoint(model, args.ckpt_path)

    # Finetune (lower LR)
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
        **logger_train_kwargs(args),
    )
    print("[Info] Finetune complete.")
    return model


if __name__ == "__main__":
    finetune_script()
