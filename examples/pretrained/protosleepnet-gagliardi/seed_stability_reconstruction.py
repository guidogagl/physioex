"""Seed stability analysis via data-driven reconstruction MMD.

Compares prototype reconstructions across seeds in input space (spectrograms).
Since all reconstructions are real PSG epochs, they live in the same space
regardless of the seed's embedding geometry.

Pipeline:
1. Load data-driven reconstructions for all seeds (proto_k/epochs.npy)
2. Hungarian matching on mean spectrograms (input-space L2)
3. Pairwise MMD between matched prototype distributions
4. Report stability metrics

Usage:
    python seed_stability_reconstruction.py \
        --results_dir /results/seed_stability/seq \
        --seeds 42 123 456 789 1024 2048 3141
"""
import argparse
import json
import os
import glob

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def gaussian_mmd(X, Y, sigma=None):
    """MMD with Gaussian RBF kernel (median heuristic for bandwidth)."""
    X_flat = X.reshape(X.shape[0], -1).astype(np.float64)
    Y_flat = Y.reshape(Y.shape[0], -1).astype(np.float64)

    XX = cdist(X_flat, X_flat, metric="sqeuclidean")
    YY = cdist(Y_flat, Y_flat, metric="sqeuclidean")
    XY = cdist(X_flat, Y_flat, metric="sqeuclidean")

    if sigma is None:
        all_dists = np.concatenate([XX.ravel(), YY.ravel(), XY.ravel()])
        sigma = np.sqrt(np.median(all_dists[all_dists > 0]) / 2.0)
        if sigma == 0:
            sigma = 1.0

    gamma = 1.0 / (2.0 * sigma ** 2)
    K_XX = np.exp(-gamma * XX)
    K_YY = np.exp(-gamma * YY)
    K_XY = np.exp(-gamma * XY)

    mmd_sq = K_XX.mean() - 2.0 * K_XY.mean() + K_YY.mean()
    return float(np.sqrt(max(0.0, mmd_sq)))


def load_seed_reconstructions(seed_dir, M=12):
    """Load data-driven reconstructions for one seed."""
    recon_dir = os.path.join(seed_dir, "data_driven")
    prototypes = {}
    for k in range(M):
        # Try both naming conventions
        for pattern in [f"proto_{k:03d}", f"proto_{k}"]:
            epochs_path = os.path.join(recon_dir, pattern, "epochs.npy")
            if os.path.exists(epochs_path):
                prototypes[k] = np.load(epochs_path)  # (N, C, T, F)
                break
        if k not in prototypes:
            print(f"  WARNING: seed_dir={seed_dir}, proto {k} not found")
    return prototypes


def dominant_stage(recon_dir, k):
    """Get dominant stage label for prototype k."""
    for pattern in [f"proto_{k:03d}", f"proto_{k}"]:
        labels_path = os.path.join(recon_dir, pattern, "labels.npy")
        if os.path.exists(labels_path):
            labels = np.load(labels_path)
            valid = labels[labels >= 0]
            if len(valid) > 0:
                return STAGE_NAMES[np.bincount(valid, minlength=5).argmax()]
    return "?"


def main():
    parser = argparse.ArgumentParser(description="Seed stability via reconstruction MMD")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024, 2048, 3141])
    parser.add_argument("--M", type=int, default=12)
    args = parser.parse_args()

    M = args.M
    seeds = args.seeds

    # ── Load all reconstructions ────────────────────────────────
    print(f"Loading reconstructions for {len(seeds)} seeds, M={M}")
    all_recon = {}
    for s in seeds:
        seed_dir = os.path.join(args.results_dir, f"seed_{s}")
        protos = load_seed_reconstructions(seed_dir, M)
        if len(protos) == M:
            all_recon[s] = protos
            print(f"  seed {s}: {M} prototypes loaded, "
                  f"shapes: {protos[0].shape}")
        else:
            print(f"  seed {s}: INCOMPLETE ({len(protos)}/{M}), skipping")

    if len(all_recon) < 2:
        print("ERROR: need at least 2 seeds with complete reconstructions")
        return

    valid_seeds = sorted(all_recon.keys())
    print(f"\nValid seeds: {valid_seeds}")

    # ── Compute mean spectrograms per prototype ─────────────────
    # Used for Hungarian matching in input space
    mean_specs = {}  # {seed: (M, C*T*F)}
    for s in valid_seeds:
        means = []
        for k in range(M):
            means.append(all_recon[s][k].mean(axis=0).ravel())
        mean_specs[s] = np.stack(means)  # (M, C*T*F)

    # ── Hungarian matching + MMD ────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-SEED PROTOTYPE STABILITY (input-space MMD)")
    print(f"{'='*70}")

    ref_seed = valid_seeds[0]
    print(f"\nReference seed: {ref_seed}")
    print(f"Prototype stages (ref):", end=" ")
    ref_recon_dir = os.path.join(args.results_dir, f"seed_{ref_seed}", "data_driven")
    ref_stages = [dominant_stage(ref_recon_dir, k) for k in range(M)]
    print(" ".join(f"P{k}={ref_stages[k]}" for k in range(M)))

    # Pairwise analysis
    pairwise_results = []
    all_mmd_values = []

    print(f"\n{'Seed A':<8} {'Seed B':<8} {'Mean MMD':<10} {'Max MMD':<10} "
          f"{'Stage match':<12} {'Mapping'}")
    print("-" * 80)

    for i, s1 in enumerate(valid_seeds):
        for s2 in valid_seeds[i + 1:]:
            # Hungarian matching on mean spectrograms
            cost = cdist(mean_specs[s1], mean_specs[s2], metric="euclidean")
            row_ind, col_ind = linear_sum_assignment(cost)

            # Compute per-prototype MMD for matched pairs
            mmds = []
            stage_matches = 0
            mapping_str = []

            s1_dir = os.path.join(args.results_dir, f"seed_{s1}", "data_driven")
            s2_dir = os.path.join(args.results_dir, f"seed_{s2}", "data_driven")

            for r, c in zip(row_ind, col_ind):
                X = all_recon[s1][r]
                Y = all_recon[s2][c]
                mmd = gaussian_mmd(X, Y)
                mmds.append(mmd)

                stage_r = dominant_stage(s1_dir, r)
                stage_c = dominant_stage(s2_dir, c)
                if stage_r == stage_c:
                    stage_matches += 1
                mapping_str.append(f"{r}→{c}")

            mmds = np.array(mmds)
            all_mmd_values.extend(mmds.tolist())

            pairwise_results.append({
                "seed_a": s1, "seed_b": s2,
                "mean_mmd": float(mmds.mean()),
                "max_mmd": float(mmds.max()),
                "min_mmd": float(mmds.min()),
                "stage_match_rate": stage_matches / M,
                "mapping": list(zip(row_ind.tolist(), col_ind.tolist())),
                "per_proto_mmd": mmds.tolist(),
            })

            print(f"{s1:<8} {s2:<8} {mmds.mean():<10.4f} {mmds.max():<10.4f} "
                  f"{stage_matches}/{M:<10} {','.join(mapping_str[:6])}...")

    all_mmd_values = np.array(all_mmd_values)
    all_mean_mmds = np.array([r["mean_mmd"] for r in pairwise_results])
    all_stage_rates = np.array([r["stage_match_rate"] for r in pairwise_results])

    # ── Intra-seed MMD (baseline) ───────────────────────────────
    print(f"\n{'='*70}")
    print("INTRA-SEED MMD (baseline: split each seed's 256 epochs in half)")
    print(f"{'='*70}")

    intra_mmds = []
    for s in valid_seeds[:3]:  # sample 3 seeds for baseline
        for k in range(M):
            X = all_recon[s][k]
            n = len(X)
            if n < 4:
                continue
            perm = np.random.RandomState(42).permutation(n)
            half = n // 2
            mmd = gaussian_mmd(X[perm[:half]], X[perm[half:2*half]])
            intra_mmds.append(mmd)
    intra_mmds = np.array(intra_mmds)

    print(f"Intra-seed MMD (split-half):  {intra_mmds.mean():.4f} ± {intra_mmds.std():.4f}")
    print(f"Cross-seed MMD:               {all_mmd_values.mean():.4f} ± {all_mmd_values.std():.4f}")
    print(f"Ratio (cross/intra):          {all_mmd_values.mean() / max(intra_mmds.mean(), 1e-8):.2f}x")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Seeds compared:               {len(valid_seeds)}")
    print(f"Pairwise comparisons:         {len(pairwise_results)}")
    print(f"Mean cross-seed MMD:          {all_mean_mmds.mean():.4f} ± {all_mean_mmds.std():.4f}")
    print(f"Mean stage match rate:        {all_stage_rates.mean():.1%} ± {all_stage_rates.std():.1%}")
    print(f"Intra-seed MMD (baseline):    {intra_mmds.mean():.4f} ± {intra_mmds.std():.4f}")

    # Save
    summary = {
        "n_seeds": len(valid_seeds),
        "seeds": valid_seeds,
        "M": M,
        "cross_seed_mmd_mean": float(all_mean_mmds.mean()),
        "cross_seed_mmd_std": float(all_mean_mmds.std()),
        "intra_seed_mmd_mean": float(intra_mmds.mean()),
        "intra_seed_mmd_std": float(intra_mmds.std()),
        "stage_match_rate_mean": float(all_stage_rates.mean()),
        "stage_match_rate_std": float(all_stage_rates.std()),
        "pairwise": pairwise_results,
    }

    out_path = os.path.join(args.results_dir, "seed_reconstruction_stability.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
