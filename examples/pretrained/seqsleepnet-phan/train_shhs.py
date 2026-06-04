"""Train SeqSleepNet on SHHS visit 1 with 3 channels (EEG, EOG, EMG).

Single-epoch test run to verify the pipeline works end-to-end on Sofia.

Usage:
    python examples/pretrained/seqsleepnet-phan/train_shhs.py --gpu_id 0 --num_workers 20
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.train.trainer import Trainer

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 3,
    "F": 129,
    "D": 32,
    "nfft": 256,
    "lowfreq": 0,
    "highfreq": 50,
    "fs": 100,
    "seqnhidden1": 64,
    "seqnlayer1": 4,
    "attentionsize": 32,
    "seqnhidden2": 64,
    "seqnlayer2": 4,
}

TRAIN_CONFIG = {
    "dataset": "shhs",
    "visit": 1,
    "channels": ["EEG", "EOG", "EMG"],
    "pipeline_preset": "seqsleepnet",
    "sequence_length": 20,
    "max_epochs": 1,
    "lr": 1e-4,
    "weight_decay": 0,
    "batch_size": 32,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": None,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train SeqSleepNet on SHHS (3-channel)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device id (None for CPU)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrained_output/seqsleepnet-shhs-3ch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset: SHHS visit 1 ──────────────────────────────────────────
    SHHS = get_dataset("shhs")
    dataset = SHHS(
        visit=TRAIN_CONFIG["visit"],
        channels=TRAIN_CONFIG["channels"],
        pipelines=TRAIN_CONFIG["pipeline_preset"],
        sequence_length=TRAIN_CONFIG["sequence_length"],
    )
    print(f"Dataset: {dataset}")

    # ── Model ───────────────────────────────────────────────────────────
    model = SeqSleepNet(**MODEL_KWARGS)

    # ── Train ───────────────────────────────────────────────────────────
    nw = args.num_workers
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=TRAIN_CONFIG["max_epochs"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        train_batch_size=TRAIN_CONFIG["batch_size"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"],
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    results = Trainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=TRAIN_CONFIG["sequence_length"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts ──────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    config = {
        "model_class": "physioex.models.seqsleepnet:SeqSleepNet",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print(
        f"Results: accuracy={results['accuracy']:.4f}, "
        f"f1={results['f1_score']:.4f}, kappa={results['cohen_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
