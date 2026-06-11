"""Prototype probing with joint embedding + input spectral features.

For each prototype k and each subject:
  - Embedding features: proportion, bout_mean, bout_std, intra_std
  - Spectral features: mean band powers (delta/theta/alpha/sigma/beta)
    per channel (EEG/EOG/EMG) for epochs assigned to prototype k

Tests: embedding-only vs spectral-only vs combined.

Usage:
    python proto_spectral_probe.py \
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \
        --K 30 --C 0.1 --seed 42 --output_dir /out
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

BANDS = {
    "delta": (1, 10),
    "theta": (10, 20),
    "alpha": (20, 31),
    "sigma": (31, 41),
    "beta": (41, 77),
}
N_BANDS = len(BANDS)
N_CHANNELS = 3  # EEG, EOG, EMG


def _group_key(sid):
    if sid.startswith("SC"):
        return sid[:5]
    if sid.startswith("shhs") and "-" in sid:
        return sid.split("-", 1)[1]
    parts = sid.split("-")
    if len(parts) >= 3:
        return parts[-1]
    return sid


def load_subjects(emb_dirs):
    subjects = {}
    for emb_dir in emb_dirs:
        for p in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(p).replace("_embeddings.npy", "")
            if sid in subjects:
                continue
            subjects[sid] = {
                "emb": p,
                "inp": os.path.join(emb_dir, f"{sid}_inputs.npy"),
                "meta": os.path.join(emb_dir, f"{sid}_metadata.json"),
            }
    return subjects


def compute_spectral_profile(inputs, assignments, K):
    """Per-prototype mean relative band powers across channels.

    Args:
        inputs: (N, 3, 29, 129) spectrogram
        assignments: (N,) prototype IDs
        K: number of prototypes

    Returns: (K, N_CHANNELS * N_BANDS) = (K, 15)
    """
    feat_dim = N_CHANNELS * N_BANDS
    profile = np.zeros((K, feat_dim), dtype=np.float32)

    for k in range(K):
        mask = assignments == k
        if mask.sum() == 0:
            continue
        # Mean spectrogram over assigned epochs: (3, 29, 129)
        mean_spec = inputs[mask].mean(axis=0)

        idx = 0
        for ch in range(N_CHANNELS):
            # Convert log-scale to linear, average over time → (129,)
            linear = np.exp(mean_spec[ch])
            mean_spectrum = linear.mean(axis=0)
            total = mean_spectrum.sum()
            if total < 1e-12:
                idx += N_BANDS
                continue
            for band, (lo, hi) in BANDS.items():
                hi = min(hi, len(mean_spectrum))
                profile[k, idx] = mean_spectrum[lo:hi].sum() / total
                idx += 1

    return profile  # (K, 15)


def compute_embedding_profile(emb, assignments, K):
    """Per-prototype embedding features.

    Returns: (K, 4) — proportion, bout_mean, bout_std, intra_std
    """
    N = len(emb)
    profile = np.zeros((K, 4), dtype=np.float32)

    for k in range(K):
        mask = assignments == k
        profile[k, 0] = mask.sum() / N
        if mask.sum() > 1:
            profile[k, 3] = emb[mask].std(axis=0).mean()

    # Bouts
    bouts = {k: [] for k in range(K)}
    cur = assignments[0]
    blen = 1
    for i in range(1, N):
        if assignments[i] == cur:
            blen += 1
        else:
            bouts[cur].append(blen)
            cur = assignments[i]
            blen = 1
    bouts[cur].append(blen)

    for k in range(K):
        if bouts[k]:
            profile[k, 1] = np.mean(bouts[k])
            if len(bouts[k]) > 1:
                profile[k, 2] = np.std(bouts[k])

    return profile  # (K, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--pca", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"Proto-spectral probe: K={args.K}, C={args.C}, pca={args.pca}, seed={args.seed}")

    subjects = load_subjects(args.emb_dirs)

    all_embs, all_inputs, all_labels, all_sids, all_groups = [], [], [], [], []
    for sid, info in subjects.items():
        if not os.path.exists(info["meta"]) or not os.path.exists(info["inp"]):
            continue
        with open(info["meta"]) as f:
            meta = json.load(f)
        group = meta.get("group")
        if group is None:
            continue
        emb = np.load(info["emb"]).astype(np.float32)
        inp = np.load(info["inp"]).astype(np.float32)
        n = min(len(emb), len(inp))
        all_embs.append(emb[:n])
        all_inputs.append(inp[:n])
        all_labels.append(str(group))
        all_sids.append(sid)
        all_groups.append(_group_key(sid))

    unique_labels = sorted(set(all_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in all_labels])
    sids_arr = np.array(all_sids)
    groups_arr = np.array(all_groups)
    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])
    n_subj = len(all_embs)

    print(f"  Subjects: {n_subj}, Classes: {unique_labels}, Dist: {np.bincount(y).tolist()}")

    n_folds = min(args.n_folds, len(unique_groups))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    output_dir = os.path.join(args.output_dir, args.source_name, "diagnosis")
    os.makedirs(output_dir, exist_ok=True)

    # Test three feature sets: emb-only, spectral-only, combined
    feature_sets = {
        "emb": lambda ep, sp: ep,
        "spectral": lambda ep, sp: sp,
        "combined": lambda ep, sp: np.concatenate([ep, sp]),
    }

    all_results = {}

    for fs_name, fs_fn in feature_sets.items():
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(
                cv.split(np.zeros((n_subj, 1)), y, group_ids)):

            # Fit K-Means on train embeddings
            train_embs = [all_embs[i] for i in train_idx]
            centroids = MiniBatchKMeans(
                n_clusters=args.K, random_state=args.seed, batch_size=4096, n_init=3
            ).fit(np.concatenate(train_embs)).cluster_centers_

            # Build features
            X = []
            for i in range(n_subj):
                assignments = cdist(all_embs[i], centroids).argmin(axis=1)
                ep = compute_embedding_profile(all_embs[i], assignments, args.K).flatten()
                sp = compute_spectral_profile(all_inputs[i], assignments, args.K).flatten()
                X.append(fs_fn(ep, sp))
            X = np.array(X, dtype=np.float32)

            if fold_idx == 0 and fs_name == "combined":
                print(f"  Feature dims: emb={args.K*4}, spectral={args.K*15}, combined={args.K*19}")

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if args.pca > 0:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
                nc = min(args.pca, X_tr.shape[0], X_tr.shape[1])
                pca = PCA(n_components=nc, random_state=args.seed)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

            if len(np.unique(y_tr)) < 2:
                continue

            clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            proba = clf.predict_proba(X_te)

            acc = accuracy_score(y_te, pred)
            f1 = f1_score(y_te, pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_te, pred)
            fold_metrics.append({"accuracy": acc, "f1_macro": f1, "kappa": kappa})

        if fold_metrics:
            summary = {}
            for key in fold_metrics[0]:
                vals = [m[key] for m in fold_metrics]
                summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            summary["n_folds"] = len(fold_metrics)
            all_results[fs_name] = summary

            print(f"  {fs_name:<12} MF1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}  "
                  f"Acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}")

    # Save
    report = {
        "K": args.K, "C": args.C, "pca": args.pca, "seed": args.seed,
        "n_subjects": n_subj, "classes": unique_labels,
        "source": args.source_name,
        "results": all_results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
