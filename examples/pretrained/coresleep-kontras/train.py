"""Train CoRe-Sleep on SHHS following Kontras et al. (2024).

Replicates the experiment from:
    Kontras et al., "CoRe-Sleep: A Multimodal Fusion Framework for Time
    Series Robust to Imperfect Modalities", IEEE TNSRE 2024.

Configuration (from the paper):
    - Dataset: SHHS visit 1, C4-A1 EEG + EOG channels
    - Input: STFT spectrograms (T=29, F=129), per-modality pipelines
    - EEG: BP 0.3-40 Hz, resample 100 Hz, STFT 2s/1s
    - EOG: BP 0.3-23 Hz, resample 100 Hz, STFT 2s/1s
    - Sequence length: L = 21 epochs
    - Optimizer: Adam, weight_decay=1e-4
    - Scheduler: linear warmup (20K steps) to max_lr=0.03, cosine decay
    - Batch size: 16
    - Loss: CrossEntropyLoss
    - Early stopping: patience 9 (~100K steps)

Usage:
    python examples/pretrained/coresleep-kontras/train.py --gpu_id 0
    python examples/pretrained/coresleep-kontras/train.py --gpu_id 0 --channels EEG  # unimodal
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.coresleep import CoReSleep
from physioex.train.trainer import Trainer

MODEL_NAME = "coresleep-kontras"
HF_REPO_ID = "4rooms/physioex"

MODEL_KWARGS = {
    "n_classes": 5,
    "in_chan": 2,
    "F": 129,
    "d_model": 128,
    "n_heads": 8,
    "n_inner_layers": 4,
    "n_outer_layers": 4,
    "d_ff": 1024,
    "dropout": 0.3,
}

TRAIN_CONFIG = {
    "dataset": "shhs",
    "dataset_kwargs": {"visit": 1},
    "channels": ["EEG", "EOG"],
    "pipeline_preset": "coresleep",
    "sequence_length": 21,
    "max_epochs": 50,
    "lr": 0.03,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "loss": "CrossEntropyLoss",
    "fold": 0,
    # Paper: "converged when not improved in last 100k steps (~9 epochs)"
    "early_stopping_patience": 9,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train CoRe-Sleep (Kontras et al. 2024)"
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
        default="pretrained_output/coresleep-kontras",
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
        help="Override dataset name",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Override channels (e.g. --channels EEG for unimodal)",
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
    if args.dataset is not None:
        TRAIN_CONFIG["dataset"] = args.dataset
        TRAIN_CONFIG.pop("dataset_kwargs", None)
    if args.channels is not None:
        TRAIN_CONFIG["channels"] = args.channels
    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.early_stopping_patience is not None:
        TRAIN_CONFIG["early_stopping_patience"] = args.early_stopping_patience

    # Adjust model in_chan based on channels
    n_chan = len(TRAIN_CONFIG["channels"])
    MODEL_KWARGS["in_chan"] = n_chan

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
    model = CoReSleep(**MODEL_KWARGS)

    # ── Optimizer & Scheduler (paper spec) ──────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # Paper: cosine annealing with max_lr=0.03, 20k warmup steps.
    # Trainer calls scheduler.step() every ~25K steps (10 intervals/epoch).
    # 20K warmup ≈ 1 scheduler step.
    warmup_steps = 1
    total_sched_steps = TRAIN_CONFIG["max_epochs"] * 10  # 10 intervals/epoch

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4 / 0.03,  # start at ~1e-4, ramp to 0.03
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_sched_steps - warmup_steps,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    # ── Train ────────────────────────────────────────────────────────────
    nw = args.num_workers
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=TRAIN_CONFIG["max_epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
        train_batch_size=TRAIN_CONFIG["batch_size"],
        fold=TRAIN_CONFIG["fold"],
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"],
        num_workers=nw,
        pin_memory=nw > 0,
        persistent_workers=nw > 0,
        prefetch_factor=2,
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
        "model_class": "physioex.models.coresleep:CoReSleep",
        "model_kwargs": MODEL_KWARGS,
        "training": TRAIN_CONFIG,
        "reference": "Kontras et al. 2024 - CoRe-Sleep: A Multimodal Fusion Framework for Time Series Robust to Imperfect Modalities (IEEE TNSRE)",
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
