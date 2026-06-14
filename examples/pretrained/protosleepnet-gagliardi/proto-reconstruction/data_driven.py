"""Data-driven prototype reconstruction.

For each prototype in the codebook, finds the training epochs whose
epoch embeddings are closest (L2) and that are actually assigned to
that prototype. Extracts epoch embeddings on-the-fly from the dataset
(no pre-extracted files needed). Saves raw spectrogram epochs for
visualization and as initialization for hybrid reconstruction.

Usage:
    python data_driven.py --backbone seq --top_k 256
    python data_driven.py --backbone st  --top_k 256 \
        --codebook_path /path/to/codebook_vq_kmeans_m24.npy \
        --checkpoint_path /path/to/model.pt
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physioex.data.collate import stack_channels

from utils import (
    CONFIGS, STAGE_NAMES,
    add_common_args, resolve_output_dir, get_device,
    load_frozen_model, load_codebook,
    build_prototype_index, build_train_loader,
    save_prototype_results, save_summary,
)


@torch.no_grad()
def extract_epoch_embeddings(model, train_loader, device, batch_size=256):
    """Extract epoch embeddings for all training subjects on-the-fly.

    Returns:
        subjects: list of (subject_id, Z, Y) tuples
            Z: (n_epochs, d_model) float32 numpy
            Y: (n_epochs,) int64 numpy
    """
    subjects = []
    for batch in tqdm(train_loader, desc="Extracting epoch embeddings"):
        subject_id = batch["subject"][0]["id"]
        x = stack_channels(batch).squeeze(0)  # (n_epochs, C, T, F)
        y = batch["labels"].squeeze(0).numpy()
        N = x.shape[0]

        embs = []
        for i in range(0, N, batch_size):
            chunk = x[i:i + batch_size].unsqueeze(0).to(device)  # (1, chunk, C, T, F)
            h = model.epoch_encode(chunk, quantize=False)  # (1, chunk, d_model)
            embs.append(h.squeeze(0).cpu().numpy())

        Z = np.concatenate(embs, axis=0).astype(np.float32)
        Y = y.astype(np.int64)
        subjects.append((subject_id, Z, Y))

    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="Data-driven prototype reconstruction"
    )
    add_common_args(parser)
    parser.add_argument(
        "--embed_batch_size", type=int, default=256,
        help="Batch size for epoch_encode() during extraction",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, "data_driven")
    if output_dir.exists() and not args.force:
        print(f"Output exists: {output_dir}  (use --force to overwrite)")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args)
    print(f"Device: {device}")

    # ── Phase 1: Extract epoch embeddings on-the-fly ─────────────────
    print(f"Phase 1: Loading model and dataset for {args.backbone}...")
    model = load_frozen_model(
        args.backbone, device, checkpoint_path=args.checkpoint_path
    )
    print(f"  Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    _, train_loader = build_train_loader(args.backbone)
    print(f"  Dataset ready, {len(train_loader)} training subjects")

    subjects = extract_epoch_embeddings(
        model, train_loader, device, batch_size=args.embed_batch_size
    )
    print(f"  Extracted embeddings for {len(subjects)} subjects")

    # ── Phase 2: Build prototype index ───────────────────────────────
    print("Phase 2: Computing prototype assignments...")
    codebook = load_codebook(
        args.backbone, m=args.m, codebook_path=args.codebook_path
    )

    proto_index, cluster_sizes = build_prototype_index(
        subjects, codebook, top_k=args.top_k
    )
    M = codebook.shape[0]
    print(f"  Cluster sizes: min={cluster_sizes.min()}, "
          f"max={cluster_sizes.max()}, mean={cluster_sizes.mean():.0f}")

    # Determine which subjects and epoch indices we need
    needed = defaultdict(set)
    for k in range(M):
        for subject_id, epoch_idx, dist, label in proto_index[k]:
            needed[subject_id].add(epoch_idx)

    n_subjects_needed = len(needed)
    n_epochs_needed = sum(len(v) for v in needed.values())
    print(f"  Need {n_epochs_needed} epochs from {n_subjects_needed} subjects")

    # ── Phase 3: Load raw epochs for selected subjects ───────────────
    print("Phase 3: Loading raw spectrogram epochs...")
    epoch_store = {}  # (subject_id, epoch_idx) -> (C, T, F) numpy
    for batch in tqdm(train_loader, desc="Loading raw epochs"):
        subject_id = batch["subject"][0]["id"]
        if subject_id not in needed:
            continue

        x = stack_channels(batch).squeeze(0)  # (n_epochs, C, T, F)
        for epoch_idx in needed[subject_id]:
            if epoch_idx < x.shape[0]:
                epoch_store[(subject_id, epoch_idx)] = (
                    x[epoch_idx].cpu().numpy().astype(np.float32)
                )

    print(f"  Retrieved {len(epoch_store)} / {n_epochs_needed} epochs")

    # ── Phase 4: Save per-prototype results ──────────────────────────
    print("Phase 4: Saving results...")
    summary_per_proto = []

    for k in range(M):
        entries = proto_index[k]
        if not entries:
            print(f"  proto_{k:03d}: empty cluster, skipping")
            summary_per_proto.append({
                "prototype": k, "n_samples": 0,
                "cluster_size": int(cluster_sizes[k]),
            })
            continue

        epochs_list, labels_list, distances_list = [], [], []
        subject_ids, epoch_indices = [], []

        for subject_id, epoch_idx, dist, label in entries:
            key = (subject_id, epoch_idx)
            if key not in epoch_store:
                continue
            epochs_list.append(epoch_store[key])
            labels_list.append(label)
            distances_list.append(dist)
            subject_ids.append(subject_id)
            epoch_indices.append(epoch_idx)

        if not epochs_list:
            print(f"  proto_{k:03d}: no epochs retrieved, skipping")
            summary_per_proto.append({
                "prototype": k, "n_samples": 0,
                "cluster_size": int(cluster_sizes[k]),
            })
            continue

        epochs_arr = np.stack(epochs_list)       # (N, C, T, F)
        labels_arr = np.array(labels_list, dtype=np.int64)
        distances_arr = np.array(distances_list, dtype=np.float32)

        # Label distribution
        label_dist = {}
        for c, name in enumerate(STAGE_NAMES):
            label_dist[name] = int((labels_arr == c).sum())

        metadata = {
            "prototype_idx": k,
            "n_samples": len(epochs_list),
            "cluster_size": int(cluster_sizes[k]),
            "method": "data_driven",
            "backbone": args.backbone,
            "model_name": CONFIGS[args.backbone]["model_name"],
            "codebook_size": int(M),
            "mean_distance": float(distances_arr.mean()),
            "std_distance": float(distances_arr.std()),
            "label_distribution": label_dist,
            "source_subjects": subject_ids,
            "epoch_indices": epoch_indices,
        }

        save_prototype_results(
            output_dir, k,
            epochs=epochs_arr,
            labels=labels_arr,
            distances=distances_arr,
            metadata=metadata,
        )

        valid_labels = labels_arr[labels_arr >= 0]
        if len(valid_labels) > 0:
            dominant = STAGE_NAMES[np.bincount(valid_labels, minlength=5).argmax()]
        else:
            dominant = "N/A"
        summary_per_proto.append({
            "prototype": k,
            "n_samples": len(epochs_list),
            "cluster_size": int(cluster_sizes[k]),
            "mean_distance": float(distances_arr.mean()),
            "dominant_stage": dominant,
        })
        print(f"  proto_{k:03d}: {len(epochs_list)} epochs, "
              f"dist={distances_arr.mean():.3f}±{distances_arr.std():.3f}, "
              f"cluster={cluster_sizes[k]}")

    # Global summary
    total_epochs = sum(len(Z) for _, Z, _ in subjects)
    save_summary(output_dir, {
        "method": "data_driven",
        "backbone": args.backbone,
        "model_name": CONFIGS[args.backbone]["model_name"],
        "n_prototypes": int(M),
        "top_k": args.top_k,
        "total_train_epochs": total_epochs,
        "cluster_sizes": cluster_sizes.tolist(),
        "per_prototype": summary_per_proto,
    })

    print(f"\nDone. Output at {output_dir}/")


if __name__ == "__main__":
    main()
