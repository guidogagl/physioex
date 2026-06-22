"""Codebook randomization test: K-Means vs random exemplar codebooks.

For a given M, evaluates the real K-Means codebook and K random codebooks
(M random training embeddings as centroids) on in-domain staging. Reports
kappa for each, plus a permutation p-value.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/test_prototype_randomization.py \
        --backbone seq --checkpoint /path/to/model.pt \
        --codebook_path /path/to/vq_kmeans/{M}/codebook.npy \
        --emb_dir /path/to/epoch-embeddings/protosleepnet-seq-3ch-mixer \
        --dataset mass --seq_len 20 --fold 0 --gpu_id 0 \
        --n_random 50 --seed 42 --output_dir /results
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

# Reuse functions from test_prototypes.py (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from test_prototypes import (
    load_model,
    evaluate_subject,
    compute_metrics,
    build_dataset,
    CLASS_NAMES,
)
from physioex.data.collate import stack_channels
from physioex.train.trainer import Trainer


def load_training_embeddings(emb_dir):
    """Load training-split epoch embeddings for random codebook sampling.

    Tries two directory layouts:
      1. {emb_dir}/train/*_embeddings.npy          (Leonardo layout)
      2. {emb_dir}/mass_cohort{1..5}/all/*_embeddings.npy  (local layout)

    Returns:
        Z_train: (N, d_model) float32 numpy array.
    """
    # Layout 1: train/valid/test split
    train_pattern = os.path.join(emb_dir, "train", "*_embeddings.npy")
    files = sorted(glob.glob(train_pattern))

    # Layout 2: per-cohort all/
    if not files:
        for c in range(1, 6):
            pat = os.path.join(emb_dir, f"mass_cohort{c}", "all", "*_embeddings.npy")
            files.extend(sorted(glob.glob(pat)))

    if not files:
        raise FileNotFoundError(
            f"No embedding files found in {emb_dir}/train/ or "
            f"{emb_dir}/mass_cohort*/all/"
        )

    arrays = [np.load(f) for f in files]
    Z = np.concatenate(arrays, axis=0).astype(np.float32)
    print(f"Loaded {len(files)} embedding files → Z_train: {Z.shape} ({Z.nbytes / 1e6:.1f} MB)")
    return Z


def evaluate_all_subjects(model, test_loader, seq_len, device):
    """Run sliding-window evaluation on all test subjects.

    Returns:
        metrics: dict with accuracy, f1_macro, cohen_kappa, f1_per_class.
        predictions: list of per-subject dicts (proba, labels).
    """
    all_proba = []
    all_targets = []
    predictions = []

    for subj_idx, batch in enumerate(test_loader):
        if isinstance(batch, dict) and "signals" in batch:
            inputs = stack_channels(batch)
            targets = batch["labels"]
        else:
            inputs, targets = batch

        proba = evaluate_subject(model, inputs, seq_len, device)
        targets_flat = targets.reshape(-1)

        predictions.append({
            "subject_idx": subj_idx,
            "proba": proba.tolist(),
            "labels": targets_flat.tolist(),
        })
        all_proba.append(proba)
        all_targets.append(targets_flat)

    metrics = compute_metrics(all_proba, all_targets)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Codebook randomization test")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--codebook_path", type=str, required=True,
                        help="Path to real K-Means codebook .npy")
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Root dir of epoch-embeddings (contains mass_cohort*/all/)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--channels", nargs="+", default=["EEG", "EOG", "EMG"])
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_random", type=int, default=50,
                        help="Number of random codebook trials")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load real codebook to get M
    codebook_real = np.load(args.codebook_path).astype(np.float32)
    M = codebook_real.shape[0]
    prefix = f"M{M}"
    print(f"=== Randomization test: M={M}, n_random={args.n_random}, seed={args.seed} ===")

    # Load model
    model = load_model(args.backbone, args.checkpoint, device)
    print(f"Model loaded on {device}")

    # Load training embeddings for random sampling
    Z_train = load_training_embeddings(args.emb_dir, args.dataset)

    # Build test dataloader
    dataset = build_dataset(args.dataset, args.channels, "seqsleepnet", args.seq_len)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=args.fold,
    )
    print(f"Test subjects: {len(test_loader)}")

    # --- Real codebook evaluation (shared, skip if exists) ---
    real_metrics_path = os.path.join(args.output_dir, f"{prefix}_real_metrics.json")
    real_preds_path = os.path.join(args.output_dir, f"{prefix}_real_predictions.json")

    if os.path.exists(real_metrics_path):
        print(f"SKIP real: {real_metrics_path} exists")
        with open(real_metrics_path) as f:
            real_metrics = json.load(f)
    else:
        print(f"Evaluating real K-Means codebook (M={M})...")
        model.set_codebook(codebook_real)
        t0 = time.time()
        real_metrics, real_predictions = evaluate_all_subjects(
            model, test_loader, args.seq_len, device
        )
        elapsed = time.time() - t0
        print(f"  Real: kappa={real_metrics['cohen_kappa']:.4f}  "
              f"acc={real_metrics['accuracy']:.4f}  f1={real_metrics['f1_macro']:.4f}  "
              f"({elapsed:.1f}s)")

        with open(real_metrics_path, "w") as f:
            json.dump(real_metrics, f, indent=2)
        with open(real_preds_path, "w") as f:
            json.dump(real_predictions, f)

    real_kappa = real_metrics["cohen_kappa"]

    # --- Random codebook evaluations ---
    rng = np.random.RandomState(args.seed)
    random_kappas = []
    n_skipped = 0

    for k in range(args.n_random):
        metrics_path = os.path.join(args.output_dir, f"{prefix}_random_{k:03d}_metrics.json")
        codebook_out_path = os.path.join(args.output_dir, f"{prefix}_random_{k:03d}_codebook.npy")

        # Always draw from RNG to maintain determinism across restarts
        idx = rng.choice(len(Z_train), size=M, replace=False)

        if os.path.exists(metrics_path):
            n_skipped += 1
            with open(metrics_path) as f:
                rand_metrics = json.load(f)
            random_kappas.append(rand_metrics["cohen_kappa"])
            continue

        codebook_rand = Z_train[idx].astype(np.float32)
        np.save(codebook_out_path, codebook_rand)

        model.set_codebook(codebook_rand)
        t0 = time.time()
        rand_metrics, _ = evaluate_all_subjects(
            model, test_loader, args.seq_len, device
        )
        elapsed = time.time() - t0

        with open(metrics_path, "w") as f:
            json.dump(rand_metrics, f, indent=2)

        random_kappas.append(rand_metrics["cohen_kappa"])
        print(f"  Random {k:03d}: kappa={rand_metrics['cohen_kappa']:.4f}  "
              f"acc={rand_metrics['accuracy']:.4f}  ({elapsed:.1f}s)")

    if n_skipped > 0:
        print(f"Skipped {n_skipped}/{args.n_random} random trials (already computed)")

    # --- Summary ---
    if len(random_kappas) == args.n_random:
        p_value = float(np.mean([kr >= real_kappa for kr in random_kappas]))
        summary = {
            "M": M,
            "n_random": args.n_random,
            "seed": args.seed,
            "real_kappa": real_kappa,
            "random_kappas": random_kappas,
            "random_mean": float(np.mean(random_kappas)),
            "random_std": float(np.std(random_kappas)),
            "p_value": p_value,
            "delta": real_kappa - float(np.mean(random_kappas)),
        }
        summary_path = os.path.join(args.output_dir, f"{prefix}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"M={M}: real_kappa={real_kappa:.4f}  "
              f"random={summary['random_mean']:.4f}±{summary['random_std']:.4f}  "
              f"delta={summary['delta']:+.4f}  p={p_value:.4f}")
        print(f"{'='*60}")
    else:
        print(f"WARNING: only {len(random_kappas)}/{args.n_random} random trials complete, "
              f"summary not written")


if __name__ == "__main__":
    main()
