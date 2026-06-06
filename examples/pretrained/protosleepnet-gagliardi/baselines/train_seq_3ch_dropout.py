"""Train SeqSleepNet (Phan) 3ch with channel dropout on MASS.

Same as train_seq_3ch.py but wraps the model in ChannelDropoutWrapper to
randomly zero input channels during training, improving robustness to
missing channels at inference time.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/train_seq_3ch_dropout.py --gpu_id 0
    python examples/pretrained/protosleepnet-gagliardi/baselines/train_seq_3ch_dropout.py --gpu_id 0 --cdropout 0.25
"""
import argparse
import json
import os
import sys

import torch

# Allow relative import of channel_dropout from same directory
sys.path.insert(0, os.path.dirname(__file__))

from channel_dropout import ChannelDropoutWrapper
from physioex.data.datasets import get_dataset
from physioex.data.multi import MultiDataset
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.train.trainer import Trainer

MODEL_NAME = "seqsleepnet-phan-3ch-dropout"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
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

TRAIN_CONFIG = {
    "dataset": "mass",
    "dataset_cohorts": [1, 2, 3, 4, 5],
    "channels": ["EEG", "EOG", "EMG"],
    "pipeline_preset": "seqsleepnet",
    "sequence_length": 20,
    "max_epochs": 10,
    "lr": 1e-4,
    "weight_decay": 0,
    "batch_size": 32,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    "early_stopping_patience": 10,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train SeqSleepNet 3ch + channel dropout (Phan et al. 2019)"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_output/seqsleepnet-phan-3ch-dropout")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_every", type=int, default=100)
    parser.add_argument("--cdropout", type=float, default=0.5,
                        help="Per-channel dropout probability (default 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience
    TRAIN_CONFIG["cdropout"] = args.cdropout

    # ── Dataset: MASS all cohorts combined ───────────────────────────────
    MASS = get_dataset("mass")
    ds_kwargs = dict(
        channels=TRAIN_CONFIG["channels"],
        pipelines=TRAIN_CONFIG["pipeline_preset"],
        sequence_length=TRAIN_CONFIG["sequence_length"],
    )
    if args.dataset_root:
        ds_kwargs["root"] = args.dataset_root

    cohort_datasets = []
    for cohort in TRAIN_CONFIG["dataset_cohorts"]:
        ds = MASS(cohort=cohort, **ds_kwargs)
        n = ds.get_n_subjects()
        print(f"  MASS SS{cohort:02d}: {n} subjects")
        if n > 0:
            cohort_datasets.append(ds)

    dataset = MultiDataset(cohort_datasets)
    print(f"Combined: {dataset}")

    # ── Model + channel dropout wrapper ──────────────────────────────────
    inner_model = SeqSleepNet(**MODEL_KWARGS)
    model = ChannelDropoutWrapper(inner_model, p=args.cdropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {MODEL_NAME}, params: {n_params:,}, cdropout={args.cdropout}")

    # ── Compute valid_interval_ratio from --valid_every ──────────────────
    n_train = len(dataset.split(fold=TRAIN_CONFIG["fold"])[0])
    steps_per_epoch = max(1, n_train // TRAIN_CONFIG["batch_size"])
    valid_interval_ratio = args.valid_every / steps_per_epoch
    print(f"Validation every {args.valid_every} steps (ratio={valid_interval_ratio:.4f}, ~{steps_per_epoch} steps/epoch)")

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
        valid_interval_ratio=valid_interval_ratio,
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
    )

    # ── Evaluate (wrapper in eval mode = no dropout) ─────────────────────
    results = Trainer.voting_evaluate(
        model=model,
        dataset=dataset,
        L=TRAIN_CONFIG["sequence_length"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
    )

    # ── Save artifacts (inner model only, without wrapper prefix) ────────
    model_cpu = model.cpu()
    inner = model_cpu.model if isinstance(model_cpu, ChannelDropoutWrapper) else model_cpu
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(inner.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    config = {
        "model_class": "physioex.models.seqsleepnet:SeqSleepNet",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Phan et al. 2019 - SeqSleepNet (IEEE TNSRE), 3ch + channel dropout",
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
