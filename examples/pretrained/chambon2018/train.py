"""Train Chambon2018 on MASS-SS3 following Chambon et al. (2018).

Replicates the experiment from:
    Chambon et al., "A deep learning architecture for temporal sleep stage
    classification using multivariate and multimodal time series",
    IEEE TNSRE 2018 (arXiv:1707.03321).

Configuration (from the paper):
    - Dataset: MASS Session 3 (61 subjects), single EEG channel
    - Temporal context: L epochs (paper explores k=0 to 5, optimal k=1 -> L=3)
    - Preprocessing: bandpass 0.3-40 Hz, resample 100 Hz (raw waveforms)
    - Optimizer: Adam, lr=1e-3
    - Batch size: 128
    - Loss: CrossEntropyLoss (balanced sampling)
    - Early stopping: patience 5
    - Dropout: 0.25
    - 5-fold cross-validation in paper; here single fold

Note: the paper uses MASS-SS3 as primary dataset. If MASS data is not
available, falls back to Sleep-EDF.

IMPORTANT: This model outputs (B, 1, n_classes) — it predicts ONLY the
central epoch of the sequence. The Trainer must handle this shape.

Usage:
    python examples/pretrained/chambon2018/train.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.chambon2018 import Chambon2018Net
from physioex.train.trainer import Trainer

MODEL_NAME = "chambon2018"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "in_channels": 1,
    "sf": 100,
    "n_times": 3000,
    "dropout": 0.25,
}

TRAIN_CONFIG = {
    "dataset": "sleepedf",
    "channels": ["EEG"],
    "pipeline_preset": "raw",
    "sequence_length": 3,
    "max_epochs": 100,
    "lr": 1e-3,
    "weight_decay": 0,
    "batch_size": 128,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 5,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train Chambon2018Net (Chambon et al. 2018)"
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
        default="pretrained_output/chambon2018",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of dataset",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    # label_transform: keep only the central epoch label (index 1 of L=3)
    DatasetClass = get_dataset(TRAIN_CONFIG["dataset"])
    ds_kwargs = dict(
        channels=TRAIN_CONFIG["channels"],
        pipelines=TRAIN_CONFIG["pipeline_preset"],
        sequence_length=TRAIN_CONFIG["sequence_length"],
        label_transform=lambda labels: labels[1:2],
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = DatasetClass(**ds_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    model = Chambon2018Net(**MODEL_KWARGS)

    # ── Train ────────────────────────────────────────────────────────────
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
        "model_class": "physioex.models.chambon2018:Chambon2018Net",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Chambon et al. 2018 - A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series (arXiv:1707.03321)",
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
