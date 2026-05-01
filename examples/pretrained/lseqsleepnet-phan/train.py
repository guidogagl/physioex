"""Train L-SeqSleepNet on SHHS following Phan et al. (2023).

Replicates the experiment from:
    Phan et al., "L-SeqSleepNet: Whole-cycle Long Sequence Modelling
    for Automatic Sleep Staging", IEEE TNSRE 2023 (arXiv:2301.03441).

Configuration (from the paper):
    - Dataset: SHHS visit 1, 5463 subjects, C4-A1 EEG channel
    - Split: 70% train / 30% test in paper; PhysioEx single fold here
    - Input: STFT spectrograms (T=29, F=129), seqsleepnet pipeline
    - Sequence length: L = 200 epochs (~100 min, one sleep cycle)
    - Fold-process-unfold: B=10 subsequences of K=20
    - Optimizer: Adam, lr=1e-4, eps=1e-7
    - L2 regularization: lambda=1e-4 (weight_decay)
    - Batch size: 8
    - Dropout: 0.1
    - Loss: CrossEntropyLoss
    - Validation: every 100 steps, early stopping after 5000 steps (SHHS)

Differences from the paper:
    - PhysioEx single-fold split instead of paper's 70/30 split
    - Standard PyTorch BiLSTM (no recurrent batch normalization)
    - Early stopping by epochs, not by validation steps

Usage:
    python examples/pretrained/lseqsleepnet-phan/train.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.lseqsleepnet import LSeqSleepNet
from physioex.train.trainer import Trainer

MODEL_NAME = "lseqsleepnet-phan"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 1,
    "F": 129,
    "D": 32,
    "nfft": 256,
    "sf": 100,
    "lowfreq": 0,
    "highfreq": 50,
    "epoch_hidden": 64,
    "epoch_attention": 64,
    "B": 10,
    "K": 20,
    "seq_hidden_ss": 64,
    "seq_hidden_ms": 64,
    "d_clf": 512,
    "dropout": 0.1,
}

TRAIN_CONFIG = {
    "dataset": "shhs",
    "dataset_kwargs": {"visit": 1},
    "channels": ["EEG"],
    "pipeline_preset": "seqsleepnet",
    "sequence_length": 200,
    "max_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 8,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 10,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train L-SeqSleepNet (Phan et al. 2023)"
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
        default="pretrained_output/lseqsleepnet-phan",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of SHHS data",
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
    DatasetClass = get_dataset(TRAIN_CONFIG["dataset"])
    ds_kwargs = dict(
        channels=TRAIN_CONFIG["channels"],
        pipelines=TRAIN_CONFIG["pipeline_preset"],
        sequence_length=TRAIN_CONFIG["sequence_length"],
        **TRAIN_CONFIG.get("dataset_kwargs", {}),
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = DatasetClass(**ds_kwargs)

    # ── Model ────────────────────────────────────────────────────────────
    model = LSeqSleepNet(**MODEL_KWARGS)

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
        "model_class": "physioex.models.lseqsleepnet:LSeqSleepNet",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Phan et al. 2023 - L-SeqSleepNet: Whole-cycle Long Sequence Modelling for Automatic Sleep Staging (arXiv:2301.03441)",
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
