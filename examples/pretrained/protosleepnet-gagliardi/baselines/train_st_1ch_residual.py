"""Train SleepTransformer (Phan) 1ch with residual sequence encoder on SHHS.

Same architecture as sleeptransformer-phan but with:
  - Residual skip around SequenceTransformer: z = h + seq(h)
  - Deep supervision: CE loss at both epoch and sequence levels
  - Sequence encoder zero-initialized (starts as identity)

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/train_st_1ch_residual.py --gpu_id 0
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from residual_model import ResidualSequenceWrapper, ResidualTrainer
from physioex.data.datasets import get_dataset
from physioex.models.sleeptransformer import EpochTransformer, SequenceTransformer

MODEL_NAME = "sleeptransformer-phan-1ch-residual"

D_MODEL = 128
N_CLASSES = 5
D_CLF = 1024
DROPOUT = 0.1

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


def build_model():
    epoch_encoder = EpochTransformer(
        d_model=D_MODEL, in_chan=1, n_heads=8, n_layers=4,
        d_ff=1024, dropout=DROPOUT, attention_size=128,
    )
    sequence_encoder = SequenceTransformer(
        d_model=D_MODEL, n_heads=8, n_layers=4,
        d_ff=1024, dropout=DROPOUT,
    )
    classifier = nn.Sequential(
        nn.Linear(D_MODEL, D_CLF),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(D_CLF, D_CLF),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(D_CLF, N_CLASSES),
    )
    epoch_classifier = nn.Sequential(
        nn.Linear(D_MODEL, D_CLF),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(D_CLF, D_CLF),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(D_CLF, N_CLASSES),
    )
    return ResidualSequenceWrapper(epoch_encoder, sequence_encoder, classifier, epoch_classifier)


def main():
    parser = argparse.ArgumentParser(
        description="Train SleepTransformer 1ch + residual sequence encoder"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_output/sleeptransformer-phan-1ch-residual")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_every", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience

    # ── Dataset ──────────────────────────────────────────────────────
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

    # ── Model ────────────────────────────────────────────────────────
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_NAME}, params: {n_params:,}")

    # ── Valid interval ───────────────────────────────────────────────
    n_train = len(dataset.split(fold=TRAIN_CONFIG["fold"])[0])
    steps_per_epoch = max(1, n_train // TRAIN_CONFIG["batch_size"])
    valid_interval_ratio = args.valid_every / steps_per_epoch
    print(f"Validation every {args.valid_every} steps (ratio={valid_interval_ratio:.4f}, ~{steps_per_epoch} steps/epoch)")

    # ── Train ────────────────────────────────────────────────────────
    nw = args.num_workers
    model = ResidualTrainer.train(
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
        valid_interval_ratio=valid_interval_ratio,
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    results = ResidualTrainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=TRAIN_CONFIG["sequence_length"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save ─────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    config = {
        "model_class": "residual_model:ResidualSequenceWrapper",
        "build_fn": "train_st_1ch_residual:build_model",
        "training": TRAIN_CONFIG,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Results: accuracy={results['accuracy']:.4f}, "
        f"f1={results['f1_score']:.4f}, kappa={results['cohen_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
