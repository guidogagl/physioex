"""Ablation study for ProtoSeqSleepNet components (SeqSleepNet backbone).

Trains 4 variants to measure the contribution of each component:

  baseline      — Per-channel SeqSleepNet (no dropout, no mixer)
  dropout       — + random input-level channel dropout
  mixer         — + accuracy-weighted channel dropout + ChannelMixer
  protosleepnet — + SimVQ 100 prototypes

All variants use per-channel encoding (filterbank + BiLSTM + attention)
with 3 channels (EEG, EOG, EMG), trained on SHHS by default.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/train_ablation_seqsleepnet.py --variant baseline --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/train_ablation_seqsleepnet.py --variant dropout  --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/train_ablation_seqsleepnet.py --variant mixer    --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/train_ablation_seqsleepnet.py --variant protosleepnet --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.proseqsleepnet import ProtoSeqSleepNet, ProtoSeqSleepNetTrainer
from physioex.models.prosleepnet import AblationTrainer
from physioex.train.trainer import Trainer

HF_REPO_ID = "4rooms/physioex"

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21

_COMMON_MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 3,
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

VARIANTS = {
    "baseline": {
        "hf_name": "seqsleepnet-gagliardi-3ch",
        "model_kwargs": {
            **_COMMON_MODEL_KWARGS,
            "cdropout": 0.0,
            "use_channel_mixer": False,
            "use_prototypes": False,
        },
        "trainer_class": Trainer,
        "description": "Per-channel SeqSleepNet 3ch baseline (no dropout, no mixer)",
    },
    "dropout": {
        "hf_name": "seqsleepnet-gagliardi-3ch-dropout",
        "model_kwargs": {
            **_COMMON_MODEL_KWARGS,
            "random_input_dropout": 0.5,
            "cdropout": 0.0,
            "use_channel_mixer": False,
            "use_prototypes": False,
        },
        "trainer_class": Trainer,
        "description": "Per-channel SeqSleepNet 3ch + random channel dropout",
    },
    "mixer": {
        "hf_name": "seqsleepnet-gagliardi-3ch-dropout-mixer",
        "model_kwargs": {
            **_COMMON_MODEL_KWARGS,
            "cdropout": 0.5,
            "cm_n_heads": 4,
            "cm_d_ff": 256,
            "cm_n_layers": 1,
            "use_channel_mixer": True,
            "use_prototypes": False,
        },
        "trainer_class": AblationTrainer,
        "description": "Per-channel SeqSleepNet 3ch + accuracy-weighted dropout + ChannelMixer",
    },
    "protosleepnet": {
        "hf_name": "protoseqsleepnet-gagliardi",
        "model_kwargs": {
            **_COMMON_MODEL_KWARGS,
            "cdropout": 0.5,
            "cm_n_heads": 4,
            "cm_d_ff": 256,
            "cm_n_layers": 1,
            "n_prototypes": 48,
            "use_channel_mixer": True,
            "use_prototypes": True,
        },
        "trainer_class": ProtoSeqSleepNetTrainer,
        "description": "Per-channel SeqSleepNet 3ch + dropout + mixer + SimVQ 48 prototypes",
    },
}

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
    "valid_interval_ratio": 0.008,  # ~1000 steps on SHHS (paper: every 100 steps)
}


def main():
    parser = argparse.ArgumentParser(
        description="Train ProtoSeqSleepNet ablation variants"
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(VARIANTS.keys()),
        help="Ablation variant to train",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=["shhs"])
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--n_prototypes", type=int, default=None,
        help="Override number of prototypes (only for protosleepnet variant)",
    )
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    hf_name = variant["hf_name"]

    # Override n_prototypes for protosleepnet variant
    if args.n_prototypes is not None and args.variant == "protosleepnet":
        variant["model_kwargs"]["n_prototypes"] = args.n_prototypes
        hf_name = f"protoseqsleepnet-gagliardi-m{args.n_prototypes}"

    if args.output_dir is None:
        args.output_dir = f"pretrained_output/{hf_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience

    # ── Dataset ──────────────────────────────────────────────────
    ds_name = args.datasets[0]
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

    print(f"Variant: {args.variant} ({variant['description']})")
    print(f"HF name: {hf_name}")
    print(f"Dataset: {ds_name}, {dataset.get_n_subjects()} subjects")

    # ── Model ────────────────────────────────────────────────────
    model_kwargs = variant["model_kwargs"]
    model = ProtoSeqSleepNet(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ProtoSeqSleepNet: {n_params:,} parameters")

    # ── Train ────────────────────────────────────────────────────
    trainer_class = variant["trainer_class"]
    nw = args.num_workers
    model = trainer_class.train(
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
    results = trainer_class.voting_evaluate(
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
        "model_class": "physioex.models.proseqsleepnet:ProtoSeqSleepNet",
        "model_kwargs": model_kwargs,
        "training": {**TRAIN_CONFIG, "datasets": args.datasets},
        "variant": args.variant,
        "description": variant["description"],
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
                path_in_repo=f"{hf_name}/{fname}",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded {fname} to {HF_REPO_ID}/{hf_name}/")


if __name__ == "__main__":
    main()
