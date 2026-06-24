"""Cross-seed MMD analysis for one backbone across all M values.

Computes pairwise MMD between data-driven reconstructions of matched
prototypes across seeds. Hungarian matching on mean spectrograms in
input space. Outputs a JSON summary per backbone.

Usage:
    python seed_mmd_analysis.py \
        --results_dir /results/seed_stability/seq \
        --backbone_name PSN --seeds 123 456 789 1024 2048 3141
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
M_VALUES = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]


def gaussian_mmd(X, Y, sigma=None):
    X_flat = X.reshape(X.shape[0], -1).astype(np.float64)
    Y_flat = Y.reshape(Y.shape[0], -1).astype(np.float64)
    XX = cdist(X_flat, X_flat, metric="sqeuclidean")
    YY = cdist(Y_flat, Y_flat, metric="sqeuclidean")
    XY = cdist(X_flat, Y_flat, metric="sqeuclidean")
    if sigma is None:
        all_d = np.concatenate([XX.ravel(), YY.ravel(), XY.ravel()])
        med = np.median(all_d[all_d > 0])
        sigma = np.sqrt(med / 2.0) if med > 0 else 1.0
    gamma = 1.0 / (2.0 * sigma ** 2)
    mmd_sq = np.exp(-gamma * XX).mean() - 2 * np.exp(-gamma * XY).mean() + np.exp(-gamma * YY).mean()
    return float(np.sqrt(max(0, mmd_sq)))


def dominant_stage(labels):
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return "?"
    return STAGE_NAMES[np.bincount(valid, minlength=5).argmax()]


def load_recon(base, seed, M):
    recon_dir = os.path.join(base, f"seed_{seed}", f"data_driven_m{M}")
    if not os.path.isdir(recon_dir) and M == 12:
        recon_dir = os.path.join(base, f"seed_{seed}", "data_driven")
    protos = {}
    for k in range(M):
        for pat in [f"proto_{k:03d}", f"proto_{k}"]:
            ep = os.path.join(recon_dir, pat, "epochs.npy")
            lp = os.path.join(recon_dir, pat, "labels.npy")
            if os.path.exists(ep):
                protos[k] = (np.load(ep), np.load(lp) if os.path.exists(lp) else np.array([]))
                break
    return protos if len(protos) == M else None


def analyze_one_m(base, seeds, M):
    recons = {s: load_recon(base, s, M) for s in seeds}
    recons = {s: r for s, r in recons.items() if r is not None}
    if len(recons) < 2:
        return None

    vs = sorted(recons.keys())
    means = {s: np.stack([recons[s][k][0].mean(0).ravel() for k in range(M)]) for s in vs}

    cross, stages_ok, stages_tot = [], 0, 0
    for i, s1 in enumerate(vs):
        for s2 in vs[i + 1:]:
            cost = cdist(means[s1], means[s2], metric="euclidean")
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                cross.append(gaussian_mmd(recons[s1][r][0], recons[s2][c][0]))
                l1, l2 = recons[s1][r][1], recons[s2][c][1]
                if len(l1) > 0 and len(l2) > 0:
                    if dominant_stage(l1) == dominant_stage(l2):
                        stages_ok += 1
                    stages_tot += 1

    cross = np.array(cross)

    intra = []
    for s in vs[:3]:
        for k in range(M):
            X = recons[s][k][0]
            if len(X) < 4:
                continue
            for rep in range(5):
                p = np.random.RandomState(rep).permutation(len(X))
                h = len(X) // 2
                intra.append(gaussian_mmd(X[p[:h]], X[p[h : 2 * h]]))
    intra = np.array(intra)

    ratio = cross.mean() / max(intra.mean(), 1e-8)
    pooled = np.sqrt((cross.std() ** 2 + intra.std() ** 2) / 2)
    d = (cross.mean() - intra.mean()) / pooled if pooled > 0 else 0
    sr = stages_ok / max(stages_tot, 1) * 100

    return {
        "M": M,
        "cross": float(cross.mean()), "cross_std": float(cross.std()),
        "intra": float(intra.mean()), "intra_std": float(intra.std()),
        "ratio": float(ratio), "d": float(d), "stage": float(sr),
        "n_seeds": len(vs), "n_pairs": len(vs) * (len(vs) - 1) // 2,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-seed MMD analysis")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--backbone_name", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[123, 456, 789, 1024, 2048, 3141])
    parser.add_argument("--m_values", nargs="+", type=int, default=None)
    args = parser.parse_args()

    m_values = args.m_values or M_VALUES

    print(f"=== {args.backbone_name}: Cross-seed MMD for M={m_values} ===")
    print(f"{'M':>5} {'cross MMD':>12} {'intra MMD':>12} {'ratio':>7} {'d':>6} {'stage%':>7}")
    print("-" * 55)

    results = []
    for M in m_values:
        r = analyze_one_m(args.results_dir, args.seeds, M)
        if r is None:
            print(f"{M:>5} SKIP")
            continue
        results.append(r)
        print(f"{M:>5} {r['cross']:>8.4f}±{r['cross_std']:.3f} "
              f"{r['intra']:>8.4f}±{r['intra_std']:.3f} "
              f"{r['ratio']:>6.2f}x {r['d']:>5.2f} {r['stage']:>6.0f}%")

    out_path = os.path.join(args.results_dir, f"mmd_all_m_{args.backbone_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
