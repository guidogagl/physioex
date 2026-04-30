"""Train Tsinalis CNN on Sleep-EDF following Tsinalis et al. (2016).

Replicates the experiment from:
    Tsinalis et al., "Automatic Sleep Stage Scoring with Single-Channel
    EEG Using Convolutional Neural Networks", arXiv:1610.01683.

Configuration (from the paper):
    - Dataset: Sleep-EDF Expanded (paper used 20-subject version)
    - Input: 5 concatenated 30s epochs (15000 samples at 100Hz), Fpz-Cz
    - Preprocessing: none (paper uses raw signal without preprocessing)
    - Optimizer: SGD with L2 regularization
    - Loss: softmax with L2 regularization (= CrossEntropyLoss + weight_decay)
    - Evaluation: 20-fold leave-one-out in paper; here single fold

Differences from the paper:
    - Sleep-EDF Expanded (78 subjects) instead of original 20 subjects
    - Pipeline "identity" (no filtering) instead of per-paper raw signal
      (Sleep-EDF is already at 100Hz, no resampling needed)
    - Single fold split instead of 20-fold leave-one-subject-out
    - No class-balanced sampling (paper uses balanced batches per SGD epoch)
    - lr, batch_size, epochs not reported in paper; we use reasonable defaults

Usage:
    python examples/pretrained/tsinalis-2016/train.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.tsinalis import TsinalisCNN
from physioex.train.trainer import Trainer

MODEL_NAME = "tsinalis-2016"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "sfreq": 100,
    "n_filters_c1": 20,
    "n_filters_c2": 400,
    "fc_size": 500,
    "dropout": 0.5,
}

TRAIN_CONFIG = {
    "dataset": "sleepedf",
    "channels": ["EEG"],
    "pipeline_preset": "identity",
    "sequence_length": 5,
    "max_epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 20,
    "optimizer": "SGD",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train Tsinalis CNN (Tsinalis et al. 2016)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device id (None for CPU)"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrained_output/tsinalis-2016",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of Sleep-EDF data",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    # sequence_length=5: provides 5 consecutive epochs per sample
    # label_transform: keep only the central (3rd) epoch label
    SleepEDF = get_dataset("sleepedf")
    ds_kwargs = dict(
        channels=TRAIN_CONFIG["channels"],
        pipelines=TRAIN_CONFIG["pipeline_preset"],
        sequence_length=TRAIN_CONFIG["sequence_length"],
        label_transform=lambda labels: labels[2:3],
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = SleepEDF(**ds_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    model = TsinalisCNN(**MODEL_KWARGS)

    # ── Optimizer (SGD as in the paper) ──────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        momentum=0.9,
    )

    # ── Train ────────────────────────────────────────────────────────────
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=TRAIN_CONFIG["max_epochs"],
        optimizer=optimizer,
        train_batch_size=TRAIN_CONFIG["batch_size"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"],
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts ───────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    config = {
        "model_class": "physioex.models.tsinalis:TsinalisCNN",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Tsinalis et al. 2016 - Automatic Sleep Stage Scoring with Single-Channel EEG Using Convolutional Neural Networks (arXiv:1610.01683)",
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

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        for fname in ["model.pt", "config.json", "metrics.json"]:
            local = os.path.join(args.output_dir, fname)
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=f"{MODEL_NAME}/{fname}",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded {fname} to {HF_REPO_ID}/{MODEL_NAME}/")


if __name__ == "__main__":
    main()
