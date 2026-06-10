"""Predicted-stage conditioned probing for disease discrimination.

Uses staging probing predictions (out-of-fold y_proba) instead of GT labels
to compute stage-conditioned features. Two modes:

  hard — argmax of y_proba → discrete stage labels, then per-stage mean/std
  soft — probability-weighted mean/std pooling per stage + entropy/confidence

Usage:
    python soft_stage_probe.py \\
        --emb_dirs .../alzheimers_AD/all .../alzheimers_HC/all \\
        --staging_dir .../proto-st-3ch-mixer/alzheimers/staging \\
        --source_name alzheimers --task diagnosis \\
        --mode soft --C 0.01 --seed 42 --output_dir /out
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

N_STAGES = 5
D = 128


# ── Subject loading ─────────────────────────────────────────────────

def _group_sleepedf(sid):
    return sid[:5]

def _group_shhs(sid):
    return sid.split("-", 1)[1] if "-" in sid else sid

def _group_wsc(sid):
    parts = sid.split("-")
    return parts[-1] if len(parts) >= 3 else sid

GROUPING = {"SC": _group_sleepedf, "shhs": _group_shhs, "wsc": _group_wsc}

def get_group_key(sid):
    for prefix, fn in GROUPING.items():
        if sid.startswith(prefix):
            return fn(sid)
    return sid


def load_subjects(emb_dirs):
    subjects = {}
    for emb_dir in emb_dirs:
        for emb_path in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(emb_path).replace("_embeddings.npy", "")
            if sid in subjects:
                continue
            subjects[sid] = {
                "emb_path": emb_path,
                "metadata_path": os.path.join(emb_dir, f"{sid}_metadata.json"),
            }
    return subjects


def get_label(meta):
    group = meta.get("group")
    return str(group) if group is not None else None


# ── Staging predictions ─────────────────────────────────────────────

def load_staging_predictions(staging_dir):
    """Merge out-of-fold staging predictions across all folds.

    Returns: {sid: {"y_pred": (N,) int, "y_proba": (N, 5) float}}
    """
    all_preds = {}
    for fold_dir in sorted(glob.glob(os.path.join(staging_dir, "fold_*"))):
        pred_path = os.path.join(fold_dir, "predictions.json")
        if not os.path.exists(pred_path):
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        for sid, p in preds.items():
            if sid in all_preds:
                continue
            all_preds[sid] = {
                "y_pred": np.array(p["y_pred"], dtype=np.int64),
                "y_proba": np.array(p["y_proba"], dtype=np.float32),
            }
    return all_preds


# ── Feature computation ─────────────────────────────────────────────

def compute_hard_stage_features(emb, labels, add_std=True):
    """Per-stage mean/std embeddings using hard predicted labels.

    Same logic as stage_probe.py but with predicted labels.
    """
    stage_means = np.zeros((N_STAGES, D), dtype=np.float32)
    stage_stds = np.zeros((N_STAGES, D), dtype=np.float32)
    stage_counts = np.zeros(N_STAGES, dtype=np.float32)

    for s in range(N_STAGES):
        mask = labels == s
        count = mask.sum()
        stage_counts[s] = count
        if count > 0:
            stage_means[s] = emb[mask].mean(axis=0)
        if count > 1:
            stage_stds[s] = emb[mask].std(axis=0)

    parts = [stage_means.flatten()]  # (640,)
    if add_std:
        parts.append(stage_stds.flatten())  # (640,)

    total = stage_counts.sum()
    proportions = stage_counts / total if total > 0 else stage_counts
    parts.append(proportions)  # (5,)

    return np.concatenate(parts)


def compute_soft_stage_features(emb, proba, add_std=True):
    """Probability-weighted stage features.

    Args:
        emb: (N, D) epoch embeddings
        proba: (N, 5) stage probability vectors
        add_std: include weighted std per stage

    Returns:
        feature vector
    """
    N, D_emb = emb.shape
    parts = []

    # Weighted mean per stage
    stage_means = np.zeros((N_STAGES, D_emb), dtype=np.float32)
    for s in range(N_STAGES):
        w = proba[:, s]  # (N,)
        w_sum = w.sum()
        if w_sum > 1e-8:
            stage_means[s] = (w[:, None] * emb).sum(axis=0) / w_sum
    parts.append(stage_means.flatten())  # (640,)

    # Weighted std per stage
    if add_std:
        stage_stds = np.zeros((N_STAGES, D_emb), dtype=np.float32)
        for s in range(N_STAGES):
            w = proba[:, s]
            w_sum = w.sum()
            if w_sum > 1e-8:
                diff = emb - stage_means[s]
                var_s = (w[:, None] * diff ** 2).sum(axis=0) / w_sum
                stage_stds[s] = np.sqrt(var_s)
        parts.append(stage_stds.flatten())  # (640,)

    # Soft proportions
    soft_props = proba.mean(axis=0)  # (5,)
    parts.append(soft_props)

    # Prediction entropy per epoch
    epoch_entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)  # (N,)
    parts.append(np.array([epoch_entropy.mean(), epoch_entropy.std()],
                           dtype=np.float32))

    # Confidence: mean of max-prob per epoch
    max_probs = proba.max(axis=1)  # (N,)
    parts.append(np.array([max_probs.mean()], dtype=np.float32))

    return np.concatenate(parts)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predicted-stage conditioned probing for disease discrimination"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--staging_dir", type=str, required=True,
                        help="Path to staging probing results (with fold_*/predictions.json)")
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="diagnosis")
    parser.add_argument("--mode", type=str, default="soft", choices=["hard", "soft"])
    parser.add_argument("--add_std", action="store_true", default=True)
    parser.add_argument("--no_std", action="store_true", help="Disable std features")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA components before classification (0=disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    add_std = not args.no_std

    print(f"Soft-stage probe: mode={args.mode}, std={add_std}, C={args.C}, "
          f"pca={args.pca}, seed={args.seed}, source={args.source_name}")

    # Load staging predictions
    staging_preds = load_staging_predictions(args.staging_dir)
    print(f"Loaded staging predictions for {len(staging_preds)} subjects")

    # Load subjects
    subjects = load_subjects(args.emb_dirs)
    print(f"Found {len(subjects)} subjects with embeddings")

    # Build features
    sids, X_all, labels_all, groups = [], [], [], []
    skipped = 0
    for sid, info in subjects.items():
        if sid not in staging_preds:
            skipped += 1
            continue
        if not os.path.exists(info["metadata_path"]):
            skipped += 1
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        label = get_label(meta)
        if label is None:
            skipped += 1
            continue

        emb = np.load(info["emb_path"]).astype(np.float32)
        pred = staging_preds[sid]

        # Align lengths (embeddings and predictions may differ slightly)
        n = min(len(emb), len(pred["y_pred"]))
        emb = emb[:n]
        y_pred = pred["y_pred"][:n]
        y_proba = pred["y_proba"][:n]

        if args.mode == "hard":
            features = compute_hard_stage_features(emb, y_pred, add_std=add_std)
        else:
            features = compute_soft_stage_features(emb, y_proba, add_std=add_std)

        sids.append(sid)
        X_all.append(features)
        labels_all.append(label)
        groups.append(get_group_key(sid))

    if skipped > 0:
        print(f"  Skipped {skipped} subjects (no staging predictions or metadata)")

    X = np.array(X_all)
    unique_labels = sorted(set(labels_all))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels_all])
    sids_arr = np.array(sids)
    groups_arr = np.array(groups)
    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])

    print(f"  Subjects: {len(X)}, Classes: {unique_labels}, "
          f"Dist: {np.bincount(y).tolist()}, Features: {X.shape[1]}")

    if len(unique_labels) < 2:
        print("  SKIP: only 1 class")
        return

    n_folds = min(args.n_folds, len(unique_groups))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    output_dir = os.path.join(args.output_dir, args.source_name, args.task)
    os.makedirs(output_dir, exist_ok=True)

    fold_metrics = []
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, group_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Optional PCA
        if args.pca > 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            n_comp = min(args.pca, X_train.shape[0], X_train.shape[1])
            pca = PCA(n_components=n_comp, random_state=args.seed)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": sids_arr[train_idx].tolist(),
            "test": sids_arr[test_idx].tolist(),
        }

        if len(np.unique(y_train)) < 2:
            print(f"    Fold {fold_idx}: SKIP (single class)")
            continue

        clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}
        fold_metrics.append(metrics)
        print(f"    Fold {fold_idx}: acc={acc:.4f}  f1={f1:.4f}  kappa={kappa:.4f}")

        joblib.dump(clf, os.path.join(fold_dir, "classifier.joblib"))
        predictions = {}
        for i, idx in enumerate(test_idx):
            predictions[sids_arr[idx]] = {
                "y_true": int(y_test[i]),
                "y_pred": int(y_pred[i]),
                "y_proba": y_proba[i].tolist(),
            }
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
    summary["n_subjects"] = len(X)
    summary["C"] = args.C
    summary["seed"] = args.seed
    summary["mode"] = args.mode
    summary["add_std"] = add_std
    summary["pca"] = args.pca
    summary["feat_dim"] = int(X.shape[1])
    summary["classes"] = unique_labels
    summary["type"] = "soft_stage_conditioned"

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
