"""Train CoRe-Sleep on SHHS following Kontras et al. (2024).

Replicates the experiment from:
    Kontras et al., "CoRe-Sleep: A Multimodal Fusion Framework for Time
    Series Robust to Imperfect Modalities", IEEE TNSRE 2024.

Original repo: https://github.com/kkontras/CoRe-Sleep

Training specifics matching the original implementation:
    - Dataset: SHHS visit 1, C4-A1 EEG + EOG channels
    - Input: STFT spectrograms (T=29, F=129), per-modality pipelines
    - EEG: BP 0.3-40 Hz, resample 100 Hz, STFT 2s/1s
    - EOG: BP 0.3-23 Hz, resample 100 Hz, STFT 2s/1s
    - Sequence length: L = 21 epochs
    - Optimizer: Adam, lr=1e-4, weight_decay=1e-4
    - Scheduler: CosineAnnealingWarmRestarts(T_0=4, T_mult=2)
    - Batch size: 16
    - Multi-task loss: CE_combined + CE_eeg + CE_eog + 0.1 * CLIP_alignment
    - Early stopping: patience 9 (~100K steps)

Usage:
    python examples/pretrained/coresleep-kontras/train.py --gpu_id 0
    python examples/pretrained/coresleep-kontras/train.py --gpu_id 0 --channels EEG  # unimodal
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.data.datasets import get_dataset
from physioex.models.coresleep import CoReSleep
from physioex.train.trainer import Trainer
from physioex.train.metrics import accuracy_score

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
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "loss": "CoReSleepLoss",
    "fold": 0,
    # Paper: "converged when not improved in last 100k steps (~9 epochs)"
    "early_stopping_patience": 9,
}


# ── Custom multi-task loss ──────────────────────────────────────────────


class CoReSleepLoss(nn.Module):
    """Multi-task loss for CoRe-Sleep training.

    Combines:
        - CrossEntropyLoss on combined (fused) prediction
        - CrossEntropyLoss on EEG-only prediction (bimodal)
        - CrossEntropyLoss on EOG-only prediction (bimodal)
        - CLIP-style symmetric contrastive alignment loss (bimodal)

    Total = CE_combined + CE_eeg + CE_eog + align_weight * CLIP_align
    """

    def __init__(self, n_classes: int = 5, align_weight: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.align_weight = align_weight
        self.n_classes = n_classes

    def forward(self, outputs: dict, targets: torch.Tensor) -> torch.Tensor:
        targets_flat = targets.reshape(-1)

        # Combined CE (always present)
        combined = outputs["combined"].reshape(-1, self.n_classes)
        loss = self.ce(combined, targets_flat)

        # Per-modality CE (bimodal only)
        if "eeg" in outputs:
            eeg = outputs["eeg"].reshape(-1, self.n_classes)
            loss = loss + self.ce(eeg, targets_flat)
        if "eog" in outputs:
            eog = outputs["eog"].reshape(-1, self.n_classes)
            loss = loss + self.ce(eog, targets_flat)

        # CLIP alignment loss (bimodal only)
        if "align_eeg" in outputs and "align_eog" in outputs:
            eeg_norm = outputs["align_eeg"]  # (B, L, D), already L2-normalized
            eog_norm = outputs["align_eog"]  # (B, L, D)
            B, L, D = eeg_norm.shape
            # Cosine similarity matrix per batch: (B, L, L)
            sim = torch.einsum("bld,bmd->blm", eeg_norm, eog_norm)
            # Target: diagonal (epoch i of EEG matches epoch i of EOG)
            target = (
                torch.arange(L, device=sim.device).unsqueeze(0).expand(B, -1)
            )
            align_loss = (
                F.cross_entropy(sim, target)
                + F.cross_entropy(sim.transpose(1, 2), target)
            ) / 2
            loss = loss + self.align_weight * align_loss

        return loss


# ── Monkey-patch Trainer._step for dict-output models ───────────────────


_original_step = Trainer._step.__func__ if hasattr(Trainer._step, '__func__') else Trainer._step


@staticmethod
def _coresleep_step(model, batch, loss_fn, device):
    """Custom _step that handles CoReSleep's dict output."""
    if isinstance(batch, dict) and "signals" in batch:
        from physioex.data.collate import stack_channels

        inputs = stack_channels(batch).to(device)
        targets = batch["labels"].to(device)
    else:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

    with torch.autocast(device.type if "cuda" in device.type else "cpu"):
        outputs = model(inputs)

    # If model returns a dict (CoReSleep forward), use custom loss
    if isinstance(outputs, dict):
        loss = loss_fn(outputs, targets)
        # Accuracy on combined prediction
        combined = outputs["combined"].reshape(-1, outputs["combined"].shape[-1])
        targets_flat = targets.reshape(-1)
        acc = accuracy_score(
            combined, targets_flat, ignore_index=-1
        )
    else:
        # Fallback: standard tensor output
        outputs_flat = outputs.reshape(-1, outputs.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = loss_fn(outputs_flat, targets_flat)
        acc = accuracy_score(outputs_flat, targets_flat, ignore_index=-1)

    return loss, acc


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

    # ── Optimizer & Scheduler (matching original repo) ──────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # Original repo: CosineAnnealingWarmRestarts(T_0=4, T_mult=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=4,
        T_mult=2,
        eta_min=1e-6,
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_fn = CoReSleepLoss(n_classes=5, align_weight=0.1)

    # ── Monkey-patch Trainer._step for dict output ──────────────────────
    Trainer._step = _coresleep_step

    # ── Train ────────────────────────────────────────────────────────────
    nw = args.num_workers
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=TRAIN_CONFIG["max_epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss_fn,
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

    # ── Restore original _step for voting evaluation ─────────────────────
    Trainer._step = _original_step

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
