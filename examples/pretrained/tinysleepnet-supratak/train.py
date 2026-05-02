"""Train TinySleepNet on Sleep-EDF following Supratak & Guo (EMBC 2020).

Replicates the experiment from:
    Supratak & Guo, "TinySleepNet: An Efficient Deep Learning Model for
    Sleep Stage Scoring based on Raw Single-Channel EEG", IEEE EMBC 2020.

Configuration (from the official repo akaraspt/tinysleepnet):
    - Dataset: Sleep-EDF (Fpz-Cz channel only)
    - Sequence length: L = 20 epochs
    - Preprocessing: bandpass 0.3-40 Hz, resample 100 Hz (raw waveforms)
    - Optimizer: Adam, lr = 1e-4
    - Weight decay: 1e-3
    - Loss: CrossEntropyLoss (ignore_index=-1 for padded epochs)
    - Epochs: 200 (early stopping patience 50)
    - Batch size: 15
    - Fold: 0 (PhysioEx single-fold split)

Usage:
    python examples/pretrained/tinysleepnet-supratak/train.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.tinysleepnet import TinySleepNet
from physioex.train.trainer import Trainer

MODEL_NAME = "tinysleepnet-supratak"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 1,
    "sf": 100,
    "n_rnn_units": 128,
    "n_rnn_layers": 1,
}

TRAIN_CONFIG = {
    "dataset": "sleepedf",
    "channels": ["EEG"],
    "pipeline_preset": "raw",
    "sequence_length": 20,
    "max_epochs": 200,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "batch_size": 15,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 50,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train TinySleepNet (Supratak & Guo, EMBC 2020)"
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
        default="pretrained_output/tinysleepnet-supratak",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of Sleep-EDF data (overrides dataset default)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name (for smoke tests on different data)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max training epochs (for smoke tests)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Override early stopping patience",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── CLI overrides ────────────────────────────────────────────
    if args.dataset is not None:
        TRAIN_CONFIG["dataset"] = args.dataset
        # Clear dataset_kwargs when overriding dataset
        TRAIN_CONFIG.pop("dataset_kwargs", None)
    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience

    # ── Dataset ──────────────────────────────────────────────────────────
    SleepEDF = get_dataset("sleepedf")
    ds_kwargs = dict(
        channels=TRAIN_CONFIG["channels"],
        pipelines=TRAIN_CONFIG["pipeline_preset"],
        sequence_length=TRAIN_CONFIG["sequence_length"],
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = SleepEDF(**ds_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    model = TinySleepNet(**MODEL_KWARGS)

    # ── Train ────────────────────────────────────────────────────────────
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

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = Trainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=TRAIN_CONFIG["sequence_length"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts ───────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    config = {
        "model_class": "physioex.models.tinysleepnet:TinySleepNet",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Supratak & Guo 2020 - TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG",
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

    # ── Upload to HuggingFace ────────────────────────────────────────────
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
        print("Upload complete.")


if __name__ == "__main__":
    main()
