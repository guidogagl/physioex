"""Train SleepTransformer (Phan) 3ch with channel mixer on SHHS.

Per-channel EpochTransformer(in_chan=1) + modality embeddings +
accuracy-weighted ZeroEmbeddingDropout + TransformerEncoder mixer (no residual)
+ attention pooling + SequenceTransformer + classifier.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/train_st_3ch_mixer.py --gpu_id 0
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from channel_mixer import ChannelMixerWrapper, MixerTrainer
from physioex.data.datasets import get_dataset
from physioex.models.sleeptransformer import EpochTransformer, SequenceTransformer

MODEL_NAME = "sleeptransformer-phan-3ch-mixer"
HF_REPO_ID = "4rooms/physioex"

EPOCH_KWARGS = {
    "d_model": 128,
    "in_chan": 1,
    "n_heads": 8,
    "n_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "attention_size": 128,
}

SEQ_KWARGS = {
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
}

D_MODEL = 128
N_CLASSES = 5
D_CLF = 1024
DROPOUT = 0.1

TRAIN_CONFIG = {
    "dataset": "shhs",
    "dataset_kwargs": {"visit": 1},
    "channels": ["EEG", "EOG", "EMG"],
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


def build_model(cdropout=0.5):
    epoch_encoder = EpochTransformer(**EPOCH_KWARGS)
    sequence_encoder = SequenceTransformer(**SEQ_KWARGS)
    classifier = nn.Sequential(
        nn.Linear(D_MODEL, D_CLF),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(D_CLF, D_CLF),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(D_CLF, N_CLASSES),
    )
    return ChannelMixerWrapper(
        epoch_encoder=epoch_encoder,
        sequence_encoder=sequence_encoder,
        classifier=classifier,
        n_channels=3,
        n_classes=N_CLASSES,
        d_model=D_MODEL,
        cdropout=cdropout,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SleepTransformer 3ch + mixer"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_output/sleeptransformer-phan-3ch-mixer")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_every", type=int, default=1000)
    parser.add_argument("--cdropout", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience
    TRAIN_CONFIG["cdropout"] = args.cdropout

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
    model = build_model(cdropout=args.cdropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_NAME}, params: {n_params:,}, cdropout={args.cdropout}")

    # ── Valid interval ───────────────────────────────────────────────
    n_train = len(dataset.split(fold=TRAIN_CONFIG["fold"])[0])
    steps_per_epoch = max(1, n_train // TRAIN_CONFIG["batch_size"])
    valid_interval_ratio = args.valid_every / steps_per_epoch
    print(f"Validation every {args.valid_every} steps (ratio={valid_interval_ratio:.4f}, ~{steps_per_epoch} steps/epoch)")

    # ── Train ────────────────────────────────────────────────────────
    nw = args.num_workers
    model = MixerTrainer.train(
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
    results = MixerTrainer.voting_evaluate(
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
        "model_class": "channel_mixer:ChannelMixerWrapper",
        "build_fn": "train_st_3ch_mixer:build_model",
        "model_kwargs": {
            "epoch_kwargs": EPOCH_KWARGS,
            "seq_kwargs": SEQ_KWARGS,
            "d_model": D_MODEL,
            "n_classes": N_CLASSES,
            "d_clf": D_CLF,
            "cdropout": args.cdropout,
        },
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
