"""Learn post-hoc codebook via K-Means on pre-extracted epoch embeddings.

Fits a discrete codebook (K-Means centroids) on the training embeddings.
At test time, epoch embeddings are quantized to nearest codebook entry
before being passed to the frozen sequence encoder + classifier.

Reference: Rymarczyk et al., "ProtoQuant", 2025 (arXiv:2602.06592)
           Ge et al., "Vector Quantized Latent Concepts", 2025 (arXiv:2602.02726)

Usage:
    python examples/pretrained/protosleepnet-gagliardi/learn_prototypes_vq.py \
        --emb_dir /path/to/embeddings/sleeptransformer-gagliardi \
        --n_prototypes 50 \
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
    evaluate_metrics,
)

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser(
        description="Learn VQ codebook from epoch embeddings"
    )
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory with train_embeddings.npy + train_labels.npy")
    parser.add_argument("--n_prototypes", type=int, default=50,
                        help="Number of codebook entries (K-Means clusters)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: emb_dir)")
    parser.add_argument("--max_iter", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir or args.emb_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load training embeddings
    print(f"Loading training embeddings from {args.emb_dir}")
    Z_train, Y_train = load_epoch_embeddings(args.emb_dir, split="train")

    # Filter unscored
    valid = Y_train >= 0
    Z_train = Z_train[valid]
    Y_train = Y_train[valid]
    print(f"  {Z_train.shape[0]} valid epochs, d_model={Z_train.shape[1]}")

    # Learn codebook
    print(f"\nRunning K-Means with M={args.n_prototypes} clusters...")
    codebook = learn_codebook_kmeans(
        Z_train,
        n_prototypes=args.n_prototypes,
        max_iter=args.max_iter,
    )
    print(f"  Codebook: {codebook.shape}")

    # Sanity check: quantize training data and show cluster stats
    print("\nSanity check: quantization on training set...")
    Z_q, assignments = quantize_embeddings(Z_train, codebook)

    # Reconstruction error
    recon_error = np.sqrt(((Z_train - Z_q) ** 2).sum(axis=1).mean())
    print(f"  Mean reconstruction error (L2): {recon_error:.4f}")

    # Cluster usage
    unique, counts = np.unique(assignments, return_counts=True)
    print(f"  Active clusters: {len(unique)} / {args.n_prototypes}")
    print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.0f}, median={np.median(counts):.0f}")

    # Class distribution per cluster (for interpretability)
    print("\nCluster-class mapping (top 3 clusters per class):")
    for c, name in enumerate(CLASS_NAMES):
        mask = Y_train == c
        class_assignments = assignments[mask]
        cluster_counts = np.bincount(class_assignments, minlength=args.n_prototypes)
        top3 = np.argsort(cluster_counts)[::-1][:3]
        top3_str = ", ".join(
            f"c{idx}({cluster_counts[idx]})" for idx in top3
        )
        print(f"  {name}: {top3_str}")

    # Save
    suffix = f"vq_m{args.n_prototypes}"
    np.save(os.path.join(output_dir, f"codebook_{suffix}.npy"), codebook)

    # Save metadata
    meta = {
        "method": "vq_kmeans",
        "n_prototypes": args.n_prototypes,
        "d_model": int(codebook.shape[1]),
        "n_train_epochs": int(len(Z_train)),
        "active_clusters": int(len(unique)),
        "mean_recon_error_l2": float(recon_error),
    }
    with open(os.path.join(output_dir, f"codebook_{suffix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  codebook_{suffix}.npy: {codebook.shape}")


if __name__ == "__main__":
    main()
