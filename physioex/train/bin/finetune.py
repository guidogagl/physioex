"""CLI entry point for fine-tuning a pretrained PhysioEx model.

Uses the same raw-EDF data layer as ``train`` and ``test_model`` (see
``physioex.train.bin._common``), so the ``--dataset/--channels/--pipelines``
spec matches across all three commands.
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
        "--model_kwargs",
        type=str,
        default="{}",
        help="JSON/YAML string with model constructor kwargs",
    )
    add_dataset_cli_args(parser)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file (merged over defaults)",
    )
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Directory to save finetuned checkpoints",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader workers (0 = main process)"
    )
    add_logger_cli_args(parser)
    args = parser.parse_args()

    apply_config_overlay(args)

    # Dataset (new raw-EDF layer) — build BEFORE model so we can infer in_chan.
    dataset, n_channels = build_dataset_from_args(args)

    # Model
    model_kwargs = inject_in_chan(parse_kwargs(args.model_kwargs), n_channels)
    model_class = import_class(args.model)
    model = model_class(**model_kwargs)

    # Load pretrained weights, then finetune (lower LR).
    model, _, _ = Trainer.load_checkpoint(model, args.ckpt_path)
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
