"""Codebook randomization test: K-Means vs random exemplar codebooks.

For a given M, evaluates the real K-Means codebook and K random codebooks
(M random training embeddings as centroids) on in-domain staging. Reports
kappa for each, plus a permutation p-value.

Supports multi-worker parallelism via --worker_id / --n_workers: each worker
evaluates a disjoint subset of random codebooks (k % n_workers == worker_id).
RNG is advanced for every k to maintain determinism regardless of partitioning.

Usage:
    # Single process (original behavior):
    python test_prototype_randomization.py \
        --backbone seq --checkpoint MODEL.pt \
        --codebook_path codebook.npy --emb_dir EMBDIR \
        --dataset mass --seq_len 20 --fold 0 --gpu_id 0 \
        --n_random 50 --seed 42 --output_dir OUTDIR

    # Parallel (8 workers, launched from bash):
    for w in 0 1 2 3 4 5 6 7; do
        python test_prototype_randomization.py ... \
            --worker_id $w --n_workers 8 &
    done; wait
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


def _atomic_json_write(path, data, indent=None):
    """Write JSON atomically via tmp + os.replace."""
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent)
    os.replace(tmp, path)


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
    parser.add_argument("--worker_id", type=int, default=0,
                        help="Worker ID for parallel partitioning (0-based)")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Total number of parallel workers")
    args = parser.parse_args()

    assert 0 <= args.worker_id < args.n_workers, \
        f"worker_id={args.worker_id} must be in [0, n_workers={args.n_workers})"

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
    print(f"=== Randomization test: M={M}, n_random={args.n_random}, seed={args.seed}, "
          f"worker={args.worker_id}/{args.n_workers} ===")

    # Load model
    model = load_model(args.backbone, args.checkpoint, device)
    print(f"Model loaded on {device}")

    # Load training embeddings for random sampling
    Z_train = load_training_embeddings(args.emb_dir)

    # Build test dataloader
    dataset = build_dataset(args.dataset, args.channels, "seqsleepnet", args.seq_len)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=args.fold,
    )
    print(f"Test subjects: {len(test_loader)}")

    # --- Real codebook evaluation (worker 0 only, others skip) ---
    real_metrics_path = os.path.join(args.output_dir, f"{prefix}_real_metrics.json")
    real_preds_path = os.path.join(args.output_dir, f"{prefix}_real_predictions.json")

    if args.worker_id == 0:
        if os.path.exists(real_metrics_path):
            print(f"SKIP real: {real_metrics_path} exists")
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
            _atomic_json_write(real_metrics_path, real_metrics, indent=2)
            _atomic_json_write(real_preds_path, real_predictions)
    else:
        print(f"Worker {args.worker_id}: skipping real eval (worker 0's job)")

    # --- Random codebook evaluations (partitioned across workers) ---
    rng = np.random.RandomState(args.seed)
    n_skipped = 0
    n_evaluated = 0

    for k in range(args.n_random):
        # Always advance RNG to maintain determinism regardless of partitioning
        idx = rng.choice(len(Z_train), size=M, replace=False)

        # Only evaluate codebooks assigned to this worker
        if k % args.n_workers != args.worker_id:
            continue

        metrics_path = os.path.join(args.output_dir, f"{prefix}_random_{k:03d}_metrics.json")
        codebook_out_path = os.path.join(args.output_dir, f"{prefix}_random_{k:03d}_codebook.npy")

        if os.path.exists(metrics_path):
            n_skipped += 1
            continue

        codebook_rand = Z_train[idx].astype(np.float32)
        np.save(codebook_out_path, codebook_rand)

        model.set_codebook(codebook_rand)
        t0 = time.time()
        rand_metrics, _ = evaluate_all_subjects(
            model, test_loader, args.seq_len, device
        )
        elapsed = time.time() - t0

        _atomic_json_write(metrics_path, rand_metrics, indent=2)

        n_evaluated += 1
        print(f"  Random {k:03d}: kappa={rand_metrics['cohen_kappa']:.4f}  "
              f"acc={rand_metrics['accuracy']:.4f}  ({elapsed:.1f}s)")

    print(f"Worker {args.worker_id}: evaluated {n_evaluated}, skipped {n_skipped}")

    # --- Summary: scan all metrics files from disk ---
    all_kappas = {}
    for k in range(args.n_random):
        mp = os.path.join(args.output_dir, f"{prefix}_random_{k:03d}_metrics.json")
        if os.path.exists(mp):
            with open(mp) as f:
                all_kappas[k] = json.load(f)["cohen_kappa"]

    n_complete = len(all_kappas)
    print(f"Total completed: {n_complete}/{args.n_random} random trials")

    if n_complete == args.n_random:
        # Read real kappa from disk (may have been computed by worker 0)
        if os.path.exists(real_metrics_path):
            with open(real_metrics_path) as f:
                real_kappa = json.load(f)["cohen_kappa"]
        else:
            print("WARNING: real metrics not found, summary deferred")
            return

        random_kappas = [all_kappas[k] for k in range(args.n_random)]
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
        _atomic_json_write(summary_path, summary, indent=2)

        print(f"\n{'='*60}")
        print(f"M={M}: real_kappa={real_kappa:.4f}  "
              f"random={summary['random_mean']:.4f}±{summary['random_std']:.4f}  "
              f"delta={summary['delta']:+.4f}  p={p_value:.4f}")
        print(f"{'='*60}")
    else:
        missing = [k for k in range(args.n_random) if k not in all_kappas]
        print(f"Summary deferred: {args.n_random - n_complete} trials missing "
              f"(first missing: k={missing[0]})")


if __name__ == "__main__":
    main()
