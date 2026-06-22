"""Aggregate seed stability results: staging metrics + codebook alignment.

Reads per-seed results and the production model (seed 42) to produce
a summary table and codebook alignment analysis.

Usage:
    python seed_stability_summary.py \
        --results_dir /results/seed_stability/seq \
        --production_codebook /pretrained/protosleepnet-seq-3ch-mixer/vq_kmeans/12/codebook.npy \
        --production_metrics /pretrained/protosleepnet-seq-3ch-mixer/staging_metrics_m12.json
"""
import argparse
import json
import os
import glob

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_seed_results(results_dir):
    """Load metrics and codebooks from all seed_* directories."""
    seeds = {}
    for seed_dir in sorted(glob.glob(os.path.join(results_dir, "seed_*"))):
        seed_name = os.path.basename(seed_dir)
        seed_num = int(seed_name.replace("seed_", ""))

        metrics_path = os.path.join(seed_dir, "staging_metrics.json")
        codebook_path = os.path.join(seed_dir, "codebook_m12.npy")

        if not os.path.exists(metrics_path):
            print(f"WARNING: {metrics_path} missing, skipping seed {seed_num}")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        codebook = None
        if os.path.exists(codebook_path):
            codebook = np.load(codebook_path)

        seeds[seed_num] = {"metrics": metrics, "codebook": codebook}

    return seeds


def hungarian_codebook_alignment(cb1, cb2):
    """Align two codebooks via Hungarian matching on L2 distance.

    Returns:
        matched_distances: (M,) L2 distance per matched pair
        assignment: (M, 2) matched indices
        mean_distance: scalar
    """
    M = cb1.shape[0]
    cost = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            cost[i, j] = np.linalg.norm(cb1[i] - cb2[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_distances = cost[row_ind, col_ind]

    return matched_distances, np.column_stack([row_ind, col_ind]), matched_distances.mean()


def main():
    parser = argparse.ArgumentParser(description="Seed stability summary")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--production_codebook", type=str, default=None,
                        help="Path to seed-42 codebook (M=12)")
    parser.add_argument("--production_metrics", type=str, default=None,
                        help="Path to seed-42 staging metrics JSON")
    args = parser.parse_args()

    seeds = load_seed_results(args.results_dir)
    print(f"Loaded {len(seeds)} seed results: {sorted(seeds.keys())}")

    # Add production model (seed 42) if provided
    if args.production_metrics and os.path.exists(args.production_metrics):
        with open(args.production_metrics) as f:
            prod_metrics = json.load(f)
        prod_codebook = None
        if args.production_codebook and os.path.exists(args.production_codebook):
            prod_codebook = np.load(args.production_codebook)
        seeds[42] = {"metrics": prod_metrics, "codebook": prod_codebook}
        print(f"Added production seed 42")

    if len(seeds) < 2:
        print("ERROR: need at least 2 seeds for comparison")
        return

    # ── Staging stability ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGING STABILITY")
    print(f"{'='*60}")

    metric_keys = ["cohen_kappa", "accuracy", "f1_macro"]
    all_seeds_sorted = sorted(seeds.keys())

    print(f"\n{'Seed':<8} {'Kappa':<10} {'Accuracy':<10} {'F1-macro':<10}")
    print("-" * 40)
    for s in all_seeds_sorted:
        m = seeds[s]["metrics"]
        kappa = m.get("cohen_kappa", m.get("kappa", float("nan")))
        acc = m.get("accuracy", float("nan"))
        f1 = m.get("f1_macro", m.get("f1_score", float("nan")))
        print(f"{s:<8} {kappa:<10.4f} {acc:<10.4f} {f1:<10.4f}")

    print("-" * 40)
    for key in metric_keys:
        values = []
        for s in all_seeds_sorted:
            m = seeds[s]["metrics"]
            v = m.get(key, m.get(key.replace("cohen_", ""), float("nan")))
            values.append(v)
        values = np.array(values)
        print(f"{key:<20} {values.mean():.4f} ± {values.std():.4f}")

    # ── Codebook alignment ──────────────────────────────────────
    codebooks = {s: seeds[s]["codebook"] for s in all_seeds_sorted
                 if seeds[s]["codebook"] is not None}

    if len(codebooks) >= 2:
        print(f"\n{'='*60}")
        print("CODEBOOK ALIGNMENT (Hungarian matching, L2 distance)")
        print(f"{'='*60}")

        cb_seeds = sorted(codebooks.keys())
        n_pairs = len(cb_seeds) * (len(cb_seeds) - 1) // 2
        pairwise_distances = []

        print(f"\n{'Seed A':<8} {'Seed B':<8} {'Mean L2':<10} {'Max L2':<10}")
        print("-" * 40)

        for i, s1 in enumerate(cb_seeds):
            for s2 in cb_seeds[i+1:]:
                dists, assignment, mean_d = hungarian_codebook_alignment(
                    codebooks[s1], codebooks[s2]
                )
                pairwise_distances.append(mean_d)
                print(f"{s1:<8} {s2:<8} {mean_d:<10.4f} {dists.max():<10.4f}")

        pairwise_distances = np.array(pairwise_distances)
        print("-" * 40)
        print(f"Mean pairwise L2: {pairwise_distances.mean():.4f} ± {pairwise_distances.std():.4f}")

        # Compute inter-cluster distances for reference
        for s in cb_seeds[:1]:
            cb = codebooks[s]
            M = cb.shape[0]
            intra_dists = []
            for i in range(M):
                for j in range(i+1, M):
                    intra_dists.append(np.linalg.norm(cb[i] - cb[j]))
            intra_dists = np.array(intra_dists)
            print(f"Reference inter-prototype L2 (seed {s}): {intra_dists.mean():.4f} ± {intra_dists.std():.4f}")
    else:
        print("\nSkipping codebook alignment (need at least 2 codebooks)")

    # ── Save summary ────────────────────────────────────────────
    summary = {
        "n_seeds": len(seeds),
        "seeds": all_seeds_sorted,
        "staging": {},
        "codebook_alignment": {},
    }

    for key in metric_keys:
        values = []
        for s in all_seeds_sorted:
            m = seeds[s]["metrics"]
            values.append(m.get(key, m.get(key.replace("cohen_", ""), float("nan"))))
        summary["staging"][key] = {
            "per_seed": {s: v for s, v in zip(all_seeds_sorted, values)},
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    if len(codebooks) >= 2:
        summary["codebook_alignment"] = {
            "mean_pairwise_l2": float(pairwise_distances.mean()),
            "std_pairwise_l2": float(pairwise_distances.std()),
            "n_pairs": int(n_pairs),
        }

    summary_path = os.path.join(args.results_dir, "seed_stability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
