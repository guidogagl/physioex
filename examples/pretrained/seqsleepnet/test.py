"""Evaluate a pretrained SeqSleepNet (Phan et al. 2019) from HuggingFace.

Downloads the model via ``load_from_pretrained("seqsleepnet-huy")``,
evaluates on Sleep-EDF test set, and produces:
    - Console table with per-class and overall metrics
    - Confusion matrix plot (saved as PNG)
    - Results CSV

Usage:
    python examples/pretrained/seqsleepnet/test.py [--gpu_id 0] [--output_dir results/]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train.trainer import Trainer


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save a normalized confusion matrix."""
    cm_norm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
    cm_np = cm_norm.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_np, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SeqSleepNet (Phan et al. 2019) — Sleep-EDF")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_np[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_np[i, j]:.2f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained SeqSleepNet")
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="GPU device id (None for CPU)"
    )
    parser.add_argument("--output_dir", type=str, default="results/seqsleepnet-huy")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load pretrained model ────────────────────────────────────────────
    print("Loading pretrained SeqSleepNet from HuggingFace...")
    model = load_from_pretrained("seqsleepnet-huy")
    print(f"Model loaded: {type(model).__name__}")

    # ── Dataset (same config as training) ────────────────────────────────
    SleepEDF = get_dataset("sleepedf")
    dataset = SleepEDF(
        channels=["EEG"],
        pipelines="seqsleepnet",
        sequence_length=20,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("Evaluating on Sleep-EDF test set (fold 0)...")
    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
        fold=0,
        gpu_id=args.gpu_id,
    )

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SeqSleepNet (Phan et al. 2019) — Sleep-EDF Results")
    print("=" * 50)
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  F1 Score:     {results['f1_score']:.4f}")
    print(f"  Cohen Kappa:  {results['cohen_kappa']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")

    if "support" in results:
        print(f"\n  Per-class support: {results['support']}")
    print("=" * 50)

    # ── Confusion matrix ─────────────────────────────────────────────────
    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        if isinstance(cm, torch.Tensor):
            plot_confusion_matrix(
                cm, STAGE_NAMES, os.path.join(args.output_dir, "confusion_matrix.png")
            )

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
