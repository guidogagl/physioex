"""Learn post-hoc codebook via K-Means only (no supervised refinement).

Fits a discrete codebook on pre-extracted epoch embeddings using
MiniBatchKMeans. No model needed — purely unsupervised on embeddings.

This is the non-optimized baseline for comparison with the supervised
VQ training in ``learn_prototypes_vq.py``.

Usage:
    python learn_prototypes_vq_kmeans.py \
        --emb_dir /path/to/embeddings/st-baseline \
        --n_prototypes 48 \
        --output_dir /path/to/save
"""
import argparse
import json
import os

import numpy as np

from physioex.explain.prototypes.posthoc import (
    learn_codebook_kmeans,
    quantize_embeddings,
    load_epoch_embeddings,
)

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser(
        description="Learn VQ codebook via K-Means (no supervised refinement)"
    )
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory with train/ embeddings")
    parser.add_argument("--n_prototypes", type=int, default=48)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: emb_dir)")
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if codebook exists")
    args = parser.parse_args()

    output_dir = args.output_dir or args.emb_dir
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"vq_kmeans_m{args.n_prototypes}"
    codebook_path = os.path.join(output_dir, f"codebook_{suffix}.npy")
    meta_path = os.path.join(output_dir, f"codebook_{suffix}_meta.json")

    # Skip if already done
    if not args.force and os.path.exists(codebook_path) and os.path.exists(meta_path):
        print(f"SKIP: {codebook_path} already exists (use --force to rerun)")
        return

    # Load training embeddings
    print(f"Loading training embeddings from {args.emb_dir}")
    Z_train, Y_train = load_epoch_embeddings(args.emb_dir, split="train")
    valid = Y_train >= 0
    Z_train = Z_train[valid]
    Y_train = Y_train[valid]
    print(f"  {Z_train.shape[0]} valid epochs, d_model={Z_train.shape[1]}")

    # Class distribution
    for c, name in enumerate(CLASS_NAMES):
        n = (Y_train == c).sum()
        print(f"  {name}: {n} epochs ({100*n/len(Y_train):.1f}%)")

    # K-Means
    print(f"\nRunning K-Means with M={args.n_prototypes} clusters...")
    codebook = learn_codebook_kmeans(
        Z_train, n_prototypes=args.n_prototypes, max_iter=args.max_iter,
    )
    print(f"  Codebook: {codebook.shape}")

    # Stats
    Z_q, assignments = quantize_embeddings(Z_train, codebook)
    recon_error = np.sqrt(((Z_train - Z_q) ** 2).sum(axis=1).mean())
    unique, counts = np.unique(assignments, return_counts=True)
    print(f"  Reconstruction error (L2): {recon_error:.4f}")
    print(f"  Active clusters: {len(unique)} / {args.n_prototypes}")
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.0f}, median={np.median(counts):.0f}")

    # Save
    np.save(codebook_path, codebook)

    meta = {
        "method": "vq_kmeans",
        "n_prototypes": args.n_prototypes,
        "d_model": int(codebook.shape[1]),
        "n_train_epochs": int(len(Z_train)),
        "active_clusters": int(len(unique)),
        "mean_recon_error_l2": float(recon_error),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {codebook_path}")


if __name__ == "__main__":
    main()
