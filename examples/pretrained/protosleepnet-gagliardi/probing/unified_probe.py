"""Unified probing for disease detection and clinical score regression.

Three modes for epoch-level feature extraction:
  gt        — per-stage mean/std with GT sleep stage labels
  gt_probas — soft weighted mean/std with staging probing probabilities
  gt_free   — K-Means prototypes + embedding profile + spectral band powers

Tasks:
  Classification: diagnosis (HOA vs PD)
  Regression (all subjects): rbdsq, psqi, ess, scopa, purdue
  Regression (PD only): updrs3, ledd

Usage:
    python unified_probe.py \\
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \\
        --mode gt --task diagnosis --C 0.1 --seed 42 --output_dir /out
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

N_STAGES = 5
D = 128
BANDS = {
    "delta": (1, 10), "theta": (10, 20), "alpha": (20, 31),
    "sigma": (31, 41), "beta": (41, 77),
}
N_BANDS = len(BANDS)
N_CHANNELS = 3

TASK_CONFIG = {
    "diagnosis":  {"type": "classification", "field": "group",             "subset": None},
    "rbdsq":      {"type": "regression",     "field": "rbdsq_total",      "subset": None},
    "psqi":       {"type": "regression",     "field": "psqi_total_score", "subset": None},
    "ess":        {"type": "regression",     "field": "ess_total",        "subset": None},
    "scopa":      {"type": "regression",     "field": "scopa_total_score","subset": None},
    "purdue":     {"type": "regression",     "field": "purdue_total",     "subset": None},
    "updrs3":     {"type": "regression",     "field": "updrs3_total",     "subset": "PD"},
    "ledd":       {"type": "regression",     "field": "ledd",             "subset": "PD"},
}


# ── Subject loading ─────────────────────────────────────────────────

def _group_key(sid):
    if sid.startswith("SC"):
        return sid[:5]
    if sid.startswith("shhs") and "-" in sid:
        return sid.split("-", 1)[1]
    parts = sid.split("-")
    return parts[-1] if len(parts) >= 3 else sid


def load_subjects(emb_dirs):
    subjects = {}
    for emb_dir in emb_dirs:
        for p in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(p).replace("_embeddings.npy", "")
            if sid in subjects:
                continue
            subjects[sid] = {
                "emb": p,
                "labels": os.path.join(emb_dir, f"{sid}_labels.npy"),
                "inputs": os.path.join(emb_dir, f"{sid}_inputs.npy"),
                "spectral": os.path.join(emb_dir, f"{sid}_spectral.npy"),
                "meta": os.path.join(emb_dir, f"{sid}_metadata.json"),
            }
    return subjects


def load_staging_predictions(staging_dir):
    preds = {}
    for fold_dir in sorted(glob.glob(os.path.join(staging_dir, "fold_*"))):
        path = os.path.join(fold_dir, "predictions.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for sid, p in data.items():
            if sid not in preds:
                preds[sid] = np.array(p["y_proba"], dtype=np.float32)
    return preds


# ── Feature computation: GT mode ────────────────────────────────────

def compute_features_gt(emb, labels):
    """Per-stage mean/std embeddings with GT labels."""
    valid = labels >= 0
    emb, labels = emb[valid], labels[valid]

    means = np.zeros((N_STAGES, D), dtype=np.float32)
    stds = np.zeros((N_STAGES, D), dtype=np.float32)
    counts = np.zeros(N_STAGES, dtype=np.float32)

    for s in range(N_STAGES):
        mask = labels == s
        counts[s] = mask.sum()
        if counts[s] > 0:
            means[s] = emb[mask].mean(axis=0)
        if counts[s] > 1:
            stds[s] = emb[mask].std(axis=0)

    total = counts.sum()
    proportions = counts / total if total > 0 else counts

    return np.concatenate([means.flatten(), stds.flatten(), proportions])


# ── Feature computation: GT probas mode ─────────────────────────────

def compute_features_gt_probas(emb, proba):
    """Soft weighted mean/std with staging probabilities."""
    N, D_emb = emb.shape
    means = np.zeros((N_STAGES, D_emb), dtype=np.float32)
    stds = np.zeros((N_STAGES, D_emb), dtype=np.float32)

    for s in range(N_STAGES):
        w = proba[:, s]
        w_sum = w.sum()
        if w_sum > 1e-8:
            means[s] = (w[:, None] * emb).sum(axis=0) / w_sum
            diff = emb - means[s]
            stds[s] = np.sqrt((w[:, None] * diff ** 2).sum(axis=0) / w_sum)

    soft_props = proba.mean(axis=0)
    epoch_ent = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    confidence = proba.max(axis=1).mean()

    return np.concatenate([
        means.flatten(), stds.flatten(), soft_props,
        np.array([epoch_ent.mean(), epoch_ent.std(), confidence], dtype=np.float32),
    ])


# ── Feature computation: GT-free mode ──────────────────────────────

def compute_spectral_features(inputs):
    """Band powers from spectrogram (N, 3, 29, 129) → (N, 15)."""
    N, C, T, F = inputs.shape
    feats = np.zeros((N, C * N_BANDS), dtype=np.float32)
    for ch in range(C):
        linear = np.exp(inputs[:, ch, :, :])
        mean_spec = linear.mean(axis=1)
        total = mean_spec.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1e-12)
        for bi, (band, (lo, hi)) in enumerate(BANDS.items()):
            hi = min(hi, F)
            feats[:, ch * N_BANDS + bi] = mean_spec[:, lo:hi].sum(axis=1) / total[:, 0]
    return feats


def compute_features_gt_free(emb, spectral, assignments, K):
    """Per-prototype embedding + spectral profile."""
    N = len(emb)

    # Embedding profile: proportion, bout_mean, bout_std, intra_std
    emb_profile = np.zeros((K, 4), dtype=np.float32)
    for k in range(K):
        mask = assignments == k
        emb_profile[k, 0] = mask.sum() / N
        if mask.sum() > 1:
            emb_profile[k, 3] = emb[mask].std(axis=0).mean()

    bouts = {k: [] for k in range(K)}
    cur, blen = assignments[0], 1
    for i in range(1, N):
        if assignments[i] == cur:
            blen += 1
        else:
            bouts[cur].append(blen)
            cur, blen = assignments[i], 1
    bouts[cur].append(blen)
    for k in range(K):
        if bouts[k]:
            emb_profile[k, 1] = np.mean(bouts[k])
            if len(bouts[k]) > 1:
                emb_profile[k, 2] = np.std(bouts[k])

    # Spectral profile: mean relative band powers per prototype per channel
    spec_profile = np.zeros((K, N_CHANNELS * N_BANDS), dtype=np.float32)
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 0:
            spec_profile[k] = spectral[mask].mean(axis=0)

    return np.concatenate([emb_profile.flatten(), spec_profile.flatten()])


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified probing for disease detection and clinical regression"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--staging_dir", default=None)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--mode", required=True, choices=["gt", "gt_probas", "gt_free"])
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--pca", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    task_cfg = TASK_CONFIG[args.task]
    is_regression = task_cfg["type"] == "regression"

    print(f"Unified probe: mode={args.mode}, task={args.task}, C={args.C}, "
          f"pca={args.pca}, seed={args.seed}")

    # Load staging predictions if needed
    staging = {}
    if args.mode == "gt_probas":
        if not args.staging_dir:
            print("ERROR: --staging_dir required for mode=gt_probas")
            return
        staging = load_staging_predictions(args.staging_dir)
        print(f"  Staging predictions: {len(staging)} subjects")

    # Load subjects
    raw_subjects = load_subjects(args.emb_dirs)
    print(f"  Found {len(raw_subjects)} subjects")

    subjects = []
    for sid, info in raw_subjects.items():
        if not os.path.exists(info["meta"]):
            continue
        with open(info["meta"]) as f:
            meta = json.load(f)

        group = meta.get("group")
        if group is None:
            continue

        # Filter by subset if needed (e.g., PD-only for updrs3)
        if task_cfg["subset"] and str(group) != task_cfg["subset"]:
            continue

        # Get target value
        if args.task == "diagnosis":
            target = str(group)
        else:
            target = meta.get(task_cfg["field"])
            if target is None:
                continue
            target = float(target)

        emb = np.load(info["emb"]).astype(np.float32)
        if len(emb) == 0:
            continue

        subj = {
            "sid": sid, "group": _group_key(sid), "target": target, "emb": emb,
            "meta": meta,
        }

        # Load mode-specific data
        if args.mode == "gt":
            if not os.path.exists(info["labels"]):
                continue
            subj["labels"] = np.load(info["labels"]).astype(np.int64)
            n = min(len(emb), len(subj["labels"]))
            subj["emb"], subj["labels"] = emb[:n], subj["labels"][:n]

        elif args.mode == "gt_probas":
            if sid not in staging:
                continue
            subj["proba"] = staging[sid]
            n = min(len(emb), len(subj["proba"]))
            subj["emb"], subj["proba"] = emb[:n], subj["proba"][:n]

        elif args.mode == "gt_free":
            # Load or compute spectral features
            if os.path.exists(info["spectral"]):
                spectral = np.load(info["spectral"]).astype(np.float32)
            elif os.path.exists(info["inputs"]):
                inp = np.load(info["inputs"]).astype(np.float32)
                n = min(len(emb), len(inp))
                spectral = compute_spectral_features(inp[:n])
                np.save(info["spectral"], spectral)
            else:
                continue
            n = min(len(emb), len(spectral))
            subj["emb"], subj["spectral"] = emb[:n], spectral[:n]

        subjects.append(subj)

    print(f"  Valid subjects: {len(subjects)}")

    if not subjects:
        print("  No valid subjects")
        return

    # Setup targets
    sids = [s["sid"] for s in subjects]
    groups = np.array([s["group"] for s in subjects])
    unique_groups = np.unique(groups)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups])

    if is_regression:
        y = np.array([s["target"] for s in subjects], dtype=np.float64)
        cv = GroupKFold(n_splits=min(args.n_folds, len(unique_groups)))
        print(f"  Target range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")
    else:
        labels_list = [s["target"] for s in subjects]
        unique_labels = sorted(set(labels_list))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_int[l] for l in labels_list])
        print(f"  Classes: {unique_labels}, dist: {np.bincount(y).tolist()}")
        if len(unique_labels) < 2:
            print("  SKIP: only 1 class")
            return
        cv = StratifiedGroupKFold(n_splits=min(args.n_folds, len(unique_groups)),
                                   shuffle=True, random_state=args.seed)

    n_subj = len(subjects)
    output_dir = os.path.join(args.output_dir, args.source_name, args.task)
    os.makedirs(output_dir, exist_ok=True)

    fold_metrics = []
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros((n_subj, 1)), y, group_ids)):

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_assignments[f"fold_{fold_idx}"] = {
            "train": [sids[i] for i in train_idx],
            "test": [sids[i] for i in test_idx],
        }

        # Compute features per mode
        if args.mode == "gt_free":
            # K-Means on train embeddings
            train_embs = [subjects[i]["emb"] for i in train_idx]
            centroids = MiniBatchKMeans(
                n_clusters=args.K, random_state=args.seed, batch_size=4096, n_init=3
            ).fit(np.concatenate(train_embs)).cluster_centers_

        X = []
        for i in range(n_subj):
            s = subjects[i]
            if args.mode == "gt":
                feat = compute_features_gt(s["emb"], s["labels"])
            elif args.mode == "gt_probas":
                feat = compute_features_gt_probas(s["emb"], s["proba"])
            elif args.mode == "gt_free":
                assignments = cdist(s["emb"], centroids).argmin(axis=1)
                feat = compute_features_gt_free(s["emb"], s["spectral"], assignments, args.K)
            X.append(feat)
        X = np.array(X, dtype=np.float32)

        if fold_idx == 0:
            print(f"  Feature dim: {X.shape[1]}")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Optional PCA
        if args.pca > 0:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
            nc = min(args.pca, X_tr.shape[0], X_tr.shape[1])
            pca = PCA(n_components=nc, random_state=args.seed)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        # Train & evaluate
        if is_regression:
            clf = Ridge(alpha=1.0 / args.C)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            mae = mean_absolute_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)
            metrics = {"mae": float(mae), "r2": float(r2)}
            print(f"    Fold {fold_idx}: MAE={mae:.3f}  R2={r2:.3f}")
        else:
            if len(np.unique(y_tr)) < 2:
                continue
            clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_proba = clf.predict_proba(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_te, y_pred)
            metrics = {"accuracy": float(acc), "f1_macro": float(f1), "kappa": float(kappa)}
            print(f"    Fold {fold_idx}: acc={acc:.4f}  f1={f1:.4f}  kappa={kappa:.4f}")

        fold_metrics.append(metrics)

        # Save
        predictions = {}
        for i, idx in enumerate(test_idx):
            entry = {"y_true": float(y_te[i]), "y_pred": float(y_pred[i])}
            if not is_regression:
                entry["y_proba"] = y_proba[i].tolist()
            predictions[sids[idx]] = entry
        with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    if not fold_metrics:
        print("  No valid folds")
        return

    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(fold_metrics)
    summary["n_subjects"] = n_subj
    summary["mode"] = args.mode
    summary["task"] = args.task
    summary["K"] = args.K if args.mode == "gt_free" else None
    summary["C"] = args.C
    summary["pca"] = args.pca
    summary["seed"] = args.seed
    summary["type"] = task_cfg["type"]
    if not is_regression:
        summary["classes"] = unique_labels

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if is_regression:
        print(f"\n  Summary: MAE={summary['mae']['mean']:.3f}+/-{summary['mae']['std']:.3f}  "
              f"R2={summary['r2']['mean']:.3f}+/-{summary['r2']['std']:.3f}")
    else:
        print(f"\n  Summary: f1={summary['f1_macro']['mean']:.4f}+/-{summary['f1_macro']['std']:.4f}  "
              f"acc={summary['accuracy']['mean']:.4f}+/-{summary['accuracy']['std']:.4f}")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
