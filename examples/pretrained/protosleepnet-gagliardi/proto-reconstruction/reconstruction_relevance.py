"""Reconstruction relevance via contrastive Integrated Gradients.

For each prototype, computes which input spectrogram components (channel,
time, frequency) are most responsible for making epoch_encode(x) close to
the prototype, using contrastive one-vs-rest class baselines.

Usage:
    python reconstruction_relevance.py \
        --backbone seq \
        --method data_driven \
        --reconstruction_dir /path/to/data_driven \
        --data_driven_dir /path/to/data_driven  \
        --output_dir /path/to/output
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physioex.explain.posthoc.gradients import IntegratedGradients

from utils import (
    CONFIGS, add_common_args, get_device,
    load_frozen_model, load_codebook,
)

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Disable cuDNN — LSTM backward requires train mode with cuDNN
torch.backends.cudnn.enabled = False


# ── Objective function ───────────────────────────────────────────────

def make_prototype_distance_fn(model, prototype_vec, device):
    """Returns f(x_batch) -> (B,) negative squared L2 distance."""
    p_k = torch.from_numpy(prototype_vec).float().to(device)

    def f(x_batch):
        # x_batch: (B, C, T, F)
        h = model.epoch_encode(x_batch.unsqueeze(1), quantize=False)  # (B, 1, d)
        h = h.squeeze(1)  # (B, d)
        return -((h - p_k.unsqueeze(0)) ** 2).sum(dim=1)  # (B,)

    return f


# ── Class means from data_driven ─────────────────────────────────────

def compute_class_means(data_driven_dir):
    """Compute per-class mean spectrograms from data_driven (GT labels).

    Returns:
        class_means: (5, C, T, F) numpy array
    """
    data_dir = Path(data_driven_dir)
    per_class = {s: [] for s in range(5)}

    for pd in sorted(data_dir.glob("proto_*")):
        labels_path = pd / "labels.npy"
        epochs_path = pd / "epochs.npy"
        if not labels_path.exists() or not epochs_path.exists():
            continue
        labels = np.load(labels_path)
        epochs = np.load(epochs_path)
        for stage in range(5):
            mask = labels == stage
            if mask.any():
                per_class[stage].append(epochs[mask])

    class_means = np.zeros((5,) + epochs.shape[1:], dtype=np.float32)
    for stage in range(5):
        if per_class[stage]:
            all_epochs = np.concatenate(per_class[stage], axis=0)
            class_means[stage] = all_epochs.mean(axis=0)
            print(f"  Class mean {STAGE_NAMES[stage]}: {len(all_epochs)} epochs")
        else:
            print(f"  Class mean {STAGE_NAMES[stage]}: no epochs (zeros)")

    return class_means


# ── Predict classes ──────────────────────────────────────────────────

@torch.no_grad()
def predict_classes(model, epochs_tensor, device, batch_size=256):
    """Predict sleep stage for each epoch. Returns (N,) int array."""
    N = epochs_tensor.shape[0]
    preds = []
    for i in range(0, N, batch_size):
        batch = epochs_tensor[i:i + batch_size].unsqueeze(1).to(device)
        model.epoch_encode(batch, quantize=False)
        logits = model.get_metrics()["epoch_logits"].squeeze(1)  # (B, 5)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ── Convergence sweep ────────────────────────────────────────────────

def convergence_sweep(f, sample_epochs, baseline, device, n_samples=16):
    """Test IG convergence across step counts. Returns optimal steps."""
    steps_to_test = [32, 64, 128, 256, 512]
    idx = np.random.choice(len(sample_epochs), size=min(n_samples, len(sample_epochs)), replace=False)
    x = torch.from_numpy(sample_epochs[idx]).float().to(device)
    bl = torch.from_numpy(baseline).float().to(device)
    bl = bl.unsqueeze(0).expand_as(x)

    # Compute f(x) and f(baseline) for completeness check
    with torch.no_grad():
        fx = f(x)  # (n_samples,)
        fb = f(bl)  # (n_samples,)
    diff = (fx - fb).cpu().numpy()  # expected sum of attributions

    print("  Convergence sweep:")
    for steps in steps_to_test:
        ig = IntegratedGradients(f, steps=steps, expects_batch=True)
        attr = ig(x, baseline=bl)  # (n_samples, C, T, F)
        attr_sum = attr.detach().sum(dim=(1, 2, 3)).cpu().numpy()  # (n_samples,)
        gaps = np.abs(attr_sum - diff) / (np.abs(diff) + 1e-8)
        mean_gap = gaps.mean()
        max_gap = gaps.max()
        print(f"    steps={steps:>3}: mean_gap={mean_gap:.4f}, max_gap={max_gap:.4f}")
        if mean_gap < 0.05:
            print(f"  -> Selected steps={steps} (mean_gap < 5%)")
            return steps

    print(f"  -> Using max steps={steps_to_test[-1]} (gap didn't converge)")
    return steps_to_test[-1]


# ── Main analysis ────────────────────────────────────────────────────

def analyze_prototype(
    model, codebook, k, epochs_np, pred_labels, class_means,
    device, steps, output_dir, batch_size=64,
):
    """Compute contrastive IG for one prototype."""
    proto_dir = Path(output_dir) / f"proto_{k:03d}"
    proto_dir.mkdir(parents=True, exist_ok=True)

    N, C, T, F = epochs_np.shape
    dominant_class = int(np.bincount(pred_labels, minlength=5).argmax())

    f = make_prototype_distance_fn(model, codebook[k], device)
    ig = IntegratedGradients(f, steps=steps, expects_batch=True)

    # Contrastive baselines: one per other class
    contrast_stages = [s for s in range(5) if s != dominant_class]
    attr_per_contrast = {}

    for other_stage in contrast_stages:
        bl = torch.from_numpy(class_means[other_stage]).float().to(device)
        # bl: (C, T, F) — broadcast to (B, C, T, F)

        # Process in batches to manage GPU memory
        attr_accum = np.zeros((C, T, F), dtype=np.float64)
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            x_batch = torch.from_numpy(epochs_np[i:j]).float().to(device)
            bl_batch = bl.unsqueeze(0).expand_as(x_batch)
            attr_batch = ig(x_batch, baseline=bl_batch)  # (B, C, T, F)
            attr_accum += attr_batch.detach().cpu().numpy().astype(np.float64).sum(axis=0)

        attr_mean = (attr_accum / N).astype(np.float32)
        attr_per_contrast[other_stage] = attr_mean
        np.save(proto_dir / f"attr_vs_{STAGE_NAMES[other_stage]}.npy", attr_mean)

    # Overall mean (across contrasts)
    overall = np.mean(list(attr_per_contrast.values()), axis=0)
    np.save(proto_dir / f"attr_mean.npy", overall)

    # Band relevance (EEG channel, per contrast)
    from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges
    bin_ranges = bands_to_bin_ranges(SLEEP_BANDS, fs=100.0, signal_length=256)
    band_names = [b[0] for b in bin_ranges]
    n_bands = len(bin_ranges)

    band_rel = np.zeros((len(contrast_stages), n_bands), dtype=np.float32)
    for ci, other_stage in enumerate(contrast_stages):
        attr_eeg = np.abs(attr_per_contrast[other_stage][0])  # (T, F)
        for bi, (_, bstart, bend) in enumerate(bin_ranges):
            band_rel[ci, bi] = attr_eeg[:, bstart:bend].sum()
    np.save(proto_dir / "band_relevance.npy", band_rel)

    # Temporal relevance (per contrast)
    temp_rel = np.zeros((len(contrast_stages), T), dtype=np.float32)
    for ci, other_stage in enumerate(contrast_stages):
        temp_rel[ci] = np.abs(attr_per_contrast[other_stage]).sum(axis=(0, 2))  # sum over C, F
    np.save(proto_dir / "temporal_relevance.npy", temp_rel)

    # Channel relevance (per contrast)
    chan_rel = np.zeros((len(contrast_stages), C), dtype=np.float32)
    for ci, other_stage in enumerate(contrast_stages):
        chan_rel[ci] = np.abs(attr_per_contrast[other_stage]).sum(axis=(1, 2))
    np.save(proto_dir / "channel_relevance.npy", chan_rel)

    # Metadata
    meta = {
        "prototype_idx": k,
        "dominant_class": STAGE_NAMES[dominant_class],
        "n_samples": N,
        "steps": steps,
        "contrast_stages": [STAGE_NAMES[s] for s in contrast_stages],
        "band_names": band_names,
    }
    with open(proto_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return dominant_class


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction relevance via contrastive Integrated Gradients"
    )
    add_common_args(parser)
    parser.add_argument("--method", type=str, required=True,
                        choices=["data_driven", "model_driven", "hybrid"])
    parser.add_argument("--reconstruction_dir", type=str, required=True)
    parser.add_argument("--data_driven_dir", type=str, required=True,
                        help="Path to data_driven dir (for class means)")
    parser.add_argument("--steps", type=int, default=0,
                        help="IG steps (0=auto sweep)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for IG computation")
    parser.add_argument("--proto_idx", type=int, default=None,
                        help="Run only this prototype (for testing)")
    args = parser.parse_args()

    device = get_device(args)
    print(f"Device: {device}")

    # Load model and codebook
    model = load_frozen_model(args.backbone, device,
                              checkpoint_path=args.checkpoint_path)
    codebook = load_codebook(args.backbone, m=args.m,
                             codebook_path=args.codebook_path)
    M = codebook.shape[0]
    print(f"Model loaded, codebook M={M}")

    # Compute class means from data_driven
    print("Computing class means from data_driven...")
    class_means = compute_class_means(args.data_driven_dir)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.reconstruction_dir) / "relevance"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "class_means.npy", class_means)

    # Determine which prototypes to process
    recon_dir = Path(args.reconstruction_dir)
    if args.proto_idx is not None:
        proto_indices = [args.proto_idx]
    else:
        proto_indices = sorted(
            int(p.name.split("_")[1])
            for p in recon_dir.glob("proto_*")
            if (p / "epochs.npy").exists()
        )
    print(f"Processing {len(proto_indices)} prototypes")

    # Convergence sweep (on first prototype)
    if args.steps <= 0:
        print("\nRunning convergence sweep...")
        first_k = proto_indices[0]
        sample_epochs = np.load(recon_dir / f"proto_{first_k:03d}" / "epochs.npy")
        f = make_prototype_distance_fn(model, codebook[first_k], device)
        # Use dominant class baseline for sweep
        sample_preds = predict_classes(
            model, torch.from_numpy(sample_epochs).float(), device
        )
        dom = int(np.bincount(sample_preds, minlength=5).argmax())
        other = [s for s in range(5) if s != dom][0]
        steps = convergence_sweep(f, sample_epochs, class_means[other], device)
    else:
        steps = args.steps
    print(f"Using steps={steps}")

    # Process each prototype
    for k in proto_indices:
        proto_path = recon_dir / f"proto_{k:03d}"
        if not (proto_path / "epochs.npy").exists():
            continue

        print(f"\nPrototype {k}:")
        epochs_np = np.load(proto_path / "epochs.npy")
        epochs_t = torch.from_numpy(epochs_np).float()

        # Predict classes
        pred_labels = predict_classes(model, epochs_t, device)
        counts = np.bincount(pred_labels, minlength=5)
        dom = STAGE_NAMES[counts.argmax()]
        print(f"  Predicted: {dom} ({counts.max()}/{len(pred_labels)})")

        analyze_prototype(
            model=model,
            codebook=codebook,
            k=k,
            epochs_np=epochs_np,
            pred_labels=pred_labels,
            class_means=class_means,
            device=device,
            steps=steps,
            output_dir=out_dir,
            batch_size=args.batch_size,
        )
        print(f"  Saved to {out_dir}/proto_{k:03d}/")

    print(f"\nDone. Output at {out_dir}/")


if __name__ == "__main__":
    main()
