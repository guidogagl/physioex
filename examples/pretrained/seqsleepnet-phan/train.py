"""Train SeqSleepNet on Sleep-EDF following Phan et al. 2019.

Replicates the experiment from:
    Phan et al., "SeqSleepNet: End-to-End Hierarchical Recurrent Neural
    Network for Sequence-to-Sequence Automatic Sleep Staging", IEEE TNSRE 2019.

Configuration:
    - Dataset: Sleep-EDF (Fpz-Cz channel only)
    - Sequence length: L = 20 epochs
    - Preprocessing: bandpass 0.3-40 Hz, resample 100 Hz, STFT spectrogram
    - Optimizer: Adam, lr = 1e-4
    - Loss: CrossEntropyLoss (ignore_index=-1 for padded epochs)
    - Epochs: 10
    - Batch size: 32
    - Fold: 0 (PhysioEx single-fold split)

Note: the original paper uses per-fold z-score normalization on the
training set.  PhysioEx skips this step intentionally.

After training, the script:
    1. Evaluates on the test set
    2. Saves model.pt (state_dict), config.json, and metrics.json locally
    3. Uploads to HuggingFace Hub (4rooms/physioex/seqsleepnet-phan/)

Usage:
    python examples/pretrained/seqsleepnet/train.py [--gpu_id 0]
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.train.trainer import Trainer

# ── Paper configuration ──────────────────────────────────────────────────────

MODEL_NAME = "seqsleepnet-phan"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 1,
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
    "dataset": "sleepedf",
    "channels": ["EEG"],
    "pipeline_preset": "seqsleepnet",
    "sequence_length": 20,
    "max_epochs": 10,
    "lr": 1e-4,
    "weight_decay": 0,
    "batch_size": 32,
    "loss": "CrossEntropyLoss",
    "fold": 0,
}


def main():
    parser = argparse.ArgumentParser(description="Train SeqSleepNet (Phan et al. 2019)")
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device id (None for CPU)"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir", type=str, default="pretrained_output/seqsleepnet-phan"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of Sleep-EDF data (overrides dataset default)",
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

    # ── CLI overrides ────────────────────────────────────────────
    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience

    # ── Model ────────────────────────────────────────────────────────────
    model = SeqSleepNet(**MODEL_KWARGS)

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
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts ───────────────────────────────────────────────────
    # model.pt — pure state_dict
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    # config.json
    config = {
        "model_class": "physioex.models.seqsleepnet:SeqSleepNet",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Phan et al. 2019 - SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging",
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # metrics.json
    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print(
        f"Results: accuracy={results['accuracy']:.4f}, f1={results['f1_score']:.4f}, kappa={results['cohen_kappa']:.4f}"
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
