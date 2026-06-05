"""Learn post-hoc prototypes via NMF on pre-extracted epoch embeddings.

Applies per-class Non-negative Matrix Factorization on the training
embeddings to discover k prototypes per sleep stage.

Reference: Tan, Zhou & Chen, "Post-hoc Part-Prototype Networks", ICML 2024

Usage:
    python examples/pretrained/protosleepnet-gagliardi/learn_prototypes_nmf.py \
        --emb_dir /path/to/embeddings/sleeptransformer-gagliardi \
        --k_per_class 10 \
        --output_dir /path/to/save
"""
import argparse
import json
import os

import numpy as np

from physioex.explain.prototypes.posthoc import (
    discover_prototypes_nmf,
    load_epoch_embeddings,
    nearest_prototype_classify,
    evaluate_metrics,
)

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser(
        description="Learn NMF prototypes from epoch embeddings"
    )
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory with train_embeddings.npy + train_labels.npy")
    parser.add_argument("--k_per_class", type=int, default=10,
                        help="Number of prototypes per class")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: emb_dir)")
    parser.add_argument("--max_iter", type=int, default=500)
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

    # Class distribution
    for c, name in enumerate(CLASS_NAMES):
        n = (Y_train == c).sum()
        print(f"  {name}: {n} epochs ({100*n/len(Y_train):.1f}%)")

    # Discover prototypes
    print(f"\nRunning NMF with k={args.k_per_class} per class...")
    prototypes, proto_labels = discover_prototypes_nmf(
        Z_train, Y_train,
        k_per_class=args.k_per_class,
        n_classes=5,
        max_iter=args.max_iter,
    )
    print(f"  Discovered {len(prototypes)} prototypes "
          f"({args.k_per_class} x {5} classes)")

    # Quick sanity check: classify training data with prototypes
    print("\nSanity check: NMF prototype classification on training set...")
    Y_pred = nearest_prototype_classify(Z_train, prototypes, proto_labels, metric="cosine")
    train_metrics = evaluate_metrics(Y_train, Y_pred, CLASS_NAMES)
    print(f"  Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Train F1-macro: {train_metrics['f1_macro']:.4f}")
    print(f"  Train kappa:    {train_metrics['kappa']:.4f}")
    for name, f1 in train_metrics["per_class_f1"].items():
        print(f"    {name}: F1={f1:.4f}")

    # Save
    suffix = f"nmf_k{args.k_per_class}"
    np.save(os.path.join(output_dir, f"prototypes_{suffix}.npy"), prototypes)
    np.save(os.path.join(output_dir, f"proto_labels_{suffix}.npy"), proto_labels)

    # Save metadata
    meta = {
        "method": "nmf",
        "k_per_class": args.k_per_class,
        "n_prototypes": len(prototypes),
        "n_classes": 5,
        "d_model": int(prototypes.shape[1]),
        "n_train_epochs": int(len(Z_train)),
        "train_metrics": train_metrics,
    }
    with open(os.path.join(output_dir, f"prototypes_{suffix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  prototypes_{suffix}.npy: {prototypes.shape}")
    print(f"  proto_labels_{suffix}.npy: {proto_labels.shape}")


if __name__ == "__main__":
    main()
