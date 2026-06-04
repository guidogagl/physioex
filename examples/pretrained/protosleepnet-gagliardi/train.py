"""Train ProtoSleepTransformer on multiple datasets.

ProtoSleepTransformer is a quantized and robust SleepTransformer that uses:
  - Per-channel EpochTransformer encoding
  - Accuracy-weighted ChannelsDropout for robust multi-channel training
  - SimVQ vector quantization with prototype-based representations
  - SequenceTransformer for temporal context

Training configuration:
    - Datasets: SHHS, SleepEDF, HMC (3 channels: EEG, EOG, EMG)
    - Input: STFT spectrograms (T=29, F=129), seqsleepnet pipeline
    - Sequence length: L = 21 epochs
    - Optimizer: Adam, lr=1e-3
    - Batch size: 32
    - Loss: CrossEntropyLoss (multi-loss: main + proto + commit + per-channel)
    - Early stopping: 10 epochs without improvement

Usage:
    python examples/pretrained/protosleepnet-gagliardi/train.py --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/train.py --gpu_id 0 --datasets shhs sleepedf hmc
    python examples/pretrained/protosleepnet-gagliardi/train.py --gpu_id 0 --datasets sleepedf --max_epochs 5
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import available_datasets, get_dataset
from physioex.models.prosleepnet import ProtoSleepTransformer, ProtoSleepTransformerTrainer

MODEL_NAME = "protosleepnet-gagliardi"
HF_REPO_ID = "4rooms/physioex"

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 3,
    "d_model": 128,
    "n_heads": 8,
    "n_epoch_layers": 4,
    "n_seq_layers": 4,
    "d_ff": 1024,
    "d_clf": 1024,
    "dropout": 0.1,
    "attention_size": 128,
    "cdropout": 0.5,
    "cm_n_heads": 4,
    "cm_d_ff": 256,
    "cm_n_layers": 1,
    "n_prototypes": 15,
    "T": 29,
    "F": 129,
}

DEFAULT_DATASETS = ["shhs", "sleepedf", "hmc"]

# Per-dataset kwargs (e.g. SHHS needs visit=1)
DATASET_KWARGS = {
    "shhs": {"visit": 1},
}

TRAIN_CONFIG = {
    "channels": CHANNELS,
    "pipeline_preset": PIPELINE,
    "sequence_length": SEQ_LEN,
    "max_epochs": 200,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 20,
    "valid_interval_ratio": 0.008,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train ProtoSleepTransformer (quantized+robust SleepTransformer)"
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
        default="pretrained_output/protosleepnet-gagliardi",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to train on (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory for dataset data",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override max training epochs",
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
    dataset_names = args.datasets if args.datasets else DEFAULT_DATASETS
    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience

    # ── Dataset ──────────────────────────────────────────────────
    # Train on the first dataset; for multi-dataset, iterate and combine
    # For single-dataset training (most common), use it directly
    ds_name = dataset_names[0]
    DatasetClass = get_dataset(ds_name)
    ds_kwargs = dict(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=SEQ_LEN,
        **DATASET_KWARGS.get(ds_name, {}),
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root
    dataset = DatasetClass(**ds_kwargs)

    n_subjects = dataset.get_n_subjects()
    print(f"Dataset: {ds_name}, {n_subjects} subjects, channels={CHANNELS}")

    # ── Model ────────────────────────────────────────────────────
    model = ProtoSleepTransformer(**MODEL_KWARGS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ProtoSleepTransformer: {n_params:,} parameters")

    # ── Train ────────────────────────────────────────────────────
    nw = args.num_workers
    model = ProtoSleepTransformerTrainer.train(
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
        valid_interval_ratio=TRAIN_CONFIG["valid_interval_ratio"],
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
    )

    # ── Evaluate ─────────────────────────────────────────────────
    results = ProtoSleepTransformerTrainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=SEQ_LEN,
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts ───────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    config = {
        "model_class": "physioex.models.prosleepnet:ProtoSleepTransformer",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "datasets": dataset_names,
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
