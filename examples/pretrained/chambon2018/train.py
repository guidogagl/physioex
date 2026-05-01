"""Train Chambon2018 on MASS-SS3 following Chambon et al. (2018).

Replicates the experiment from:
    Chambon et al., "A deep learning architecture for temporal sleep stage
    classification using multivariate and multimodal time series",
    IEEE TNSRE 2018 (arXiv:1707.03321).

Configuration (from the paper):
    - Dataset: MASS Session 3 (62 subjects, 256Hz native, downsampled to 128Hz)
    - Temporal context: k=1 (optimal) -> L=3 epochs (current +/- 1)
    - Preprocessing: bandpass 0.3-40 Hz, resample to 128 Hz
    - Optimizer: Adam, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
    - Batch size: 128
    - Loss: CrossEntropyLoss (paper uses balanced sampling ~20% per stage)
    - Early stopping: patience 5 on validation loss
    - Dropout: 0.25
    - Weight init: Normal(0, 0.1) in paper; PyTorch default here
    - 5-fold cross-validation in paper; here single fold

Differences from the paper:
    - Single fold split instead of 5-fold by subject
    - No class-balanced sampling (paper uses ~20% per stage per batch)
    - Default PyTorch weight init instead of Normal(0, 0.1)

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
    "sf": 128,
    "n_times": 3840,  # 30s * 128Hz
    "dropout": 0.25,
}

TRAIN_CONFIG = {
    "dataset": "mass",
    "dataset_kwargs": {"cohort": 3},
    "channels": ["EEG"],
    "pipeline_preset": "raw",
    "pipeline_kwargs": {"target_fs": 128.0},
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
        help="Root directory of MASS data",
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
    # MASS Session 3, single EEG, resample to 128Hz (paper spec)
    # label_transform: keep only the central epoch label (index 1 of L=3)
    from physioex.data.presets import get_preset

    pipeline = get_preset(
        TRAIN_CONFIG["pipeline_preset"], **TRAIN_CONFIG.get("pipeline_kwargs", {})
    )

    DatasetClass = get_dataset(TRAIN_CONFIG["dataset"])
    ds_kwargs = dict(
        channels=TRAIN_CONFIG["channels"],
        pipelines=pipeline,
        sequence_length=TRAIN_CONFIG["sequence_length"],
        label_transform=lambda labels: labels[1:2],
        **TRAIN_CONFIG.get("dataset_kwargs", {}),
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = DatasetClass(**ds_kwargs)

    # ── Central-epoch model: disable voting during training validation ──
    # Chambon2018Net outputs (B, 1, n_classes), not (B, L, n_classes),
    # so the voting eval step is replaced with the standard eval step.
    Trainer._voting_eval_step = Trainer._eval_step

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
