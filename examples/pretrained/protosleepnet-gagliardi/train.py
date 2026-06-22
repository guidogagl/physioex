"""Train ProtoSleepNet on sleep staging datasets.

ProtoSleepNet wraps a backbone (SleepTransformer or SeqSleepNet) with:
  - Dual residual connections (channel mixer + sequence encoder)
  - Optional channel mixer for multi-channel robustness
  - VQ-compatible epoch embeddings via deep supervision

Usage:
    # SeqSleepNet backbone, 3ch + mixer, on MASS
    python examples/pretrained/protosleepnet-gagliardi/train.py \
        --backbone seq --dataset mass --n_channels 3 --use_mixer --gpu_id 0

    # SleepTransformer backbone, 3ch + mixer, on SHHS
    python examples/pretrained/protosleepnet-gagliardi/train.py \
        --backbone st --dataset shhs --n_channels 3 --use_mixer --gpu_id 0

    # SeqSleepNet backbone, 1ch (no mixer), on MASS
    python examples/pretrained/protosleepnet-gagliardi/train.py \
        --backbone seq --dataset mass --n_channels 1 --gpu_id 0
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.protosleepnet import ProtoSleepNet, ProtoSleepNetTrainer

MODEL_NAME = "protosleepnet-gagliardi"
HF_REPO_ID = "4rooms/physioex"

# ── Backbone defaults ────────────────────────────────────────────────

BACKBONE_CONFIGS = {
    "st": {
        "factory": "from_sleep_transformer",
        "default_dataset": "shhs",
        "seq_len": 21,
        "max_epochs": 50,
        "channels_1ch": ["EEG"],
        "channels_3ch": ["EEG", "EOG", "EMG"],
        "dataset_kwargs": {"shhs": {"visit": 1}},
    },
    "seq": {
        "factory": "from_seq_sleep_net",
        "default_dataset": "mass",
        "seq_len": 20,
        "max_epochs": 10,
        "channels_1ch": ["EEG"],
        "channels_3ch": ["EEG", "EOG", "EMG"],
        "dataset_kwargs": {},
    },
}

# ── Mixer defaults ───────────────────────────────────────────────────

MIXER_KWARGS = {
    "use_channel_mixer": True,
    "cdropout": 0.5,
    "cm_n_heads": 4,
    "cm_d_ff": 256,
    "cm_n_layers": 1,
}


def build_dataset(dataset_name, channels, pipeline, seq_len, dataset_kwargs=None):
    """Build dataset, handling MASS multi-cohort case."""
    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(cohort=c, channels=channels, pipelines=pipeline,
                      sequence_length=seq_len)
            if ds.get_n_subjects() > 0:
                cohorts.append(ds)
                print(f"  MASS SS{c:02d}: {ds.get_n_subjects()} subjects")
        return MultiDataset(cohorts)
    else:
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = dict(
            channels=channels, pipelines=pipeline,
            sequence_length=seq_len,
            **(dataset_kwargs or {}),
        )
        dataset = DatasetClass(**ds_kwargs)
        print(f"  {dataset_name}: {dataset.get_n_subjects()} subjects")
        return dataset


def main():
    parser = argparse.ArgumentParser(description="Train ProtoSleepNet")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"],
                        help="Backbone: seq (SeqSleepNet) or st (SleepTransformer)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset (default: mass for seq, shhs for st)")
    parser.add_argument("--n_channels", type=int, default=3, choices=[1, 3])
    parser.add_argument("--use_mixer", action="store_true",
                        help="Enable channel mixer (requires n_channels > 1)")
    parser.add_argument("--cdropout", type=float, default=0.5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--valid_every", type=int, default=None,
                        help="Validate every N steps (default: 100 for seq, 1000 for st)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    # ── Resolve config ───────────────────────────────────────────
    cfg = BACKBONE_CONFIGS[args.backbone]
    dataset_name = args.dataset or cfg["default_dataset"]
    seq_len = cfg["seq_len"]
    max_epochs = args.max_epochs or cfg["max_epochs"]
    valid_every = args.valid_every or (100 if args.backbone == "seq" else 1000)
    channels = cfg["channels_3ch"] if args.n_channels == 3 else cfg["channels_1ch"]

    # Output dir
    suffix = f"{args.backbone}-{args.n_channels}ch"
    if args.use_mixer:
        suffix += "-mixer"
    output_dir = args.output_dir or f"pretrained_output/protosleepnet-{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"ProtoSleepNet: backbone={args.backbone}, channels={channels}, "
          f"mixer={args.use_mixer}, dataset={dataset_name}")

    # ── Dataset ──────────────────────────────────────────────────
    ds_extra = cfg["dataset_kwargs"].get(dataset_name, {})
    dataset = build_dataset(dataset_name, channels, "seqsleepnet", seq_len, ds_extra)

    # ── Model ────────────────────────────────────────────────────
    factory = getattr(ProtoSleepNet, cfg["factory"])
    mixer_kwargs = {}
    if args.use_mixer and args.n_channels > 1:
        mixer_kwargs = {**MIXER_KWARGS, "cdropout": args.cdropout}
    model = factory(n_channels=args.n_channels, **mixer_kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ── Valid interval ───────────────────────────────────────────
    n_train = len(dataset.split(fold=0)[0])
    steps_per_epoch = max(1, n_train // args.batch_size)
    valid_interval_ratio = valid_every / steps_per_epoch
    print(f"Validation every {valid_every} steps (ratio={valid_interval_ratio:.4f}, "
          f"~{steps_per_epoch} steps/epoch)")

    # ── Train ────────────────────────────────────────────────────
    nw = args.num_workers
    model = ProtoSleepNetTrainer.train(
        model=model,
        dataset=dataset,
        max_epochs=max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_batch_size=args.batch_size,
        fold=0,
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(output_dir, "checkpoints"),
        early_stopping_patience=args.early_stopping_patience,
        valid_interval_ratio=valid_interval_ratio,
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
        seed=args.seed,
    )

    # ── Evaluate ─────────────────────────────────────────────────
    results = ProtoSleepNetTrainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=seq_len,
        fold=0,
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts ───────────────────────────────────────────
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    train_config = {
        "backbone": args.backbone,
        "dataset": dataset_name,
        "channels": channels,
        "sequence_length": seq_len,
        "n_channels": args.n_channels,
        "use_mixer": args.use_mixer,
        "cdropout": args.cdropout,
        "max_epochs": max_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    config = {
        "model_class": "physioex.models.protosleepnet:ProtoSleepNet",
        "factory": cfg["factory"],
        "factory_kwargs": {"n_channels": args.n_channels, **mixer_kwargs},
        "training": train_config,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    metrics_path = os.path.join(output_dir, "metrics.json")
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
        hf_name = f"protosleepnet-{suffix}"
        for fname in ["model.pt", "config.json", "metrics.json"]:
            local = os.path.join(output_dir, fname)
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=f"{hf_name}/{fname}",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded {fname} to {HF_REPO_ID}/{hf_name}/")


if __name__ == "__main__":
    main()
