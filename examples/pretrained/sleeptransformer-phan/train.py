"""Train SleepTransformer on SHHS following Phan et al. (2022).

Replicates the experiment from:
    Phan et al., "SleepTransformer: Automatic Sleep Staging with
    Interpretability and Uncertainty Quantification",
    IEEE TBIOM 2022 (arXiv:2105.11043).

Configuration (from the paper):
    - Dataset: SHHS visit 1, 5791 subjects, C4-A1 EEG channel
    - Split: 70% train / 30% test, 100 subjects for validation
    - Input: STFT spectrograms (T=29, F=129), seqsleepnet pipeline
    - Sequence length: L = 21 epochs
    - Optimizer: Adam, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-7
    - Batch size: 32
    - Loss: CrossEntropyLoss
    - Early stopping: 200 validation steps without improvement
    - Dropout: 0.1

Differences from the paper:
    - PhysioEx single-fold split instead of paper's 70/30 split
    - Standard early_stopping_patience (epochs) vs paper's step-based

Usage:
    python examples/pretrained/sleeptransformer-phan/train.py --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.sleeptransformer import SleepTransformer
from physioex.train.trainer import Trainer

MODEL_NAME = "sleeptransformer-phan"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 1,
    "d_model": 128,
    "n_heads": 8,
    "n_epoch_layers": 4,
    "n_seq_layers": 4,
    "d_ff": 1024,
    "d_clf": 1024,
    "dropout": 0.1,
    "attention_size": 128,
}

TRAIN_CONFIG = {
    "dataset": "shhs",
    "dataset_kwargs": {"visit": 1},
    "channels": ["EEG"],
    "pipeline_preset": "seqsleepnet",
    "sequence_length": 21,
    "max_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 0,
    "batch_size": 32,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 10,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train SleepTransformer (Phan et al. 2022)"
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
        default="pretrained_output/sleeptransformer-phan",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of SHHS data",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    model = SleepTransformer(**MODEL_KWARGS)

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
        "model_class": "physioex.models.sleeptransformer:SleepTransformer",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Phan et al. 2022 - SleepTransformer: Automatic Sleep Staging with Interpretability and Uncertainty Quantification (arXiv:2105.11043)",
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
