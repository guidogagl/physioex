"""Train a linear probe on MASS SS03 for CSD explanation.

This script:
1. Checks that CBRAMod embeddings are cached for MASS SS03
2. Loads subject IDs from the embedding cache
3. Creates a random train/valid split of subjects
4. Loads cached embeddings for train/valid subjects
5. Trains a LinearProbeWithLN (LayerNorm + Linear)
6. Saves probe weights for use with CSD

Prerequisites:
    Run this first to extract embeddings (if not already done):
    python examples/foundation/cbramod/extract_embeddings.py --gpu_id 0 --datasets mass_ss03

Usage:
    python examples/explain/conceptualspectraldecomposition/train_probe.py --gpu_id 0
    python examples/explain/conceptualspectraldecomposition/train_probe.py --gpu_id 0 --train_ratio 0.8 --max_epochs 100
"""
import argparse
import os

from physioex.data.datasets import MASSDataset
from physioex.models.embed import load_embeddings, _cache_root
from examples.explain.conceptualspectraldecomposition.utils import (
    split_subjects,
    train_and_save_probe,
)

MODEL_NAME = "cbramod"
DATASET_NAME = "mass_ss03"
COHORT = 3  # SS03
EMBEDDING_DIM = 200  # CBRAMod embedding dimension
N_CLASSES = 5  # Sleep stages: W, N1, N2, N3, REM


def main():
    parser = argparse.ArgumentParser(
        description=f"Train linear probe on {DATASET_NAME} for CSD"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./csd_checkpoints")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"

    print("=" * 60)
    print(f"CSD Linear Probe Training: {MODEL_NAME} on {DATASET_NAME}")
    print("=" * 60)

    # 1. Check embeddings are cached
    print(f"\n[1] Checking cached embeddings...")
    emb_dir = _cache_root() / MODEL_NAME / DATASET_NAME

    if not emb_dir.exists():
        print(f"  ERROR: Embeddings not found at {emb_dir}")
        print(f"\n  Please extract embeddings first:")
        print(f"    python examples/foundation/cbramod/extract_embeddings.py --gpu_id {args.gpu_id} --datasets mass_ss03")
        return 1

    # Get subject IDs from cache directory
    subject_ids = sorted([d.name for d in emb_dir.iterdir() if d.is_dir()])

    if not subject_ids:
        print(f"  ERROR: No subject embeddings found in {emb_dir}")
        return 1

    print(f"  Found {len(subject_ids)} subjects with cached embeddings")
    print(f"  Cache dir: {emb_dir}")

    # Optional: verify dataset is accessible
    try:
        dataset = MASSDataset(
            cohort=COHORT,
            channels=["EEG", "EOG", "EMG"],
            pipelines="cbramod",
            sequence_length=1,
        )
        print(f"  Dataset channels: {len(dataset.channels)} available")
    except Exception as e:
        print(f"  WARNING: Could not load dataset: {e}")
        print(f"  Continuing with cached embeddings only...")

    # 2. Split subjects
    print(f"\n[2] Splitting subjects (train_ratio={args.train_ratio})...")
    train_subjects, valid_subjects = split_subjects(
        subject_ids, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"  Train: {len(train_subjects)} subjects")
    print(f"  Valid: {len(valid_subjects)} subjects")

    # 3. Train probe (using cached embeddings)
    print(f"\n[3] Training linear probe...")
    print(f"  Output: {args.output_dir}")
    print(f"  Config: max_epochs={args.max_epochs}, lr={args.lr}, wd={args.weight_decay}")

    results = train_and_save_probe(
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        train_subjects=train_subjects,
        valid_subjects=valid_subjects,
        output_path=args.output_dir,
        embedding_dim=EMBEDDING_DIM,
        device=device,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_classes=N_CLASSES,
    )

    # 4. Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nProbe saved to: {args.output_dir}/probe.pt")
    print(f"Metrics saved to: {args.output_dir}/metrics.json")
    print(f"\nValidation metrics:")
    print(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
    print(f"  Macro F1:  {results['metrics']['macro_f1']:.4f}")
    print(f"  Kappa:     {results['metrics']['kappa']:.4f}")

    print(f"\nTo use with CSD:")
    print(f"  python explain_csd.py --gpu_id {args.gpu_id} --probe_path {args.output_dir}/probe.pt --subject_id {valid_subjects[0]} --target_class 3")

    return 0


if __name__ == "__main__":
    exit(main())
