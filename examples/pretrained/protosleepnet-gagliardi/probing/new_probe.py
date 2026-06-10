"""Subject-wise probing with local K-Means prototypes.

Instead of mean-pooling all epoch embeddings into one vector per subject,
runs K-Means per subject to extract K local centroids, aligns them to a
global reference, and concatenates into a (K*D)-dimensional feature vector.

Pipeline:
  1. Global K-Means on all embeddings → K reference centroids
  2. Per-subject K-Means → K local centroids
  3. Align local to global (nearest-neighbor assignment)
  4. Concatenate aligned local centroids → feature vector
  5. 5-fold StratifiedGroupKFold → LogisticRegression / Ridge

Usage:
    python new_probe.py \\
        --emb_dirs .../alzheimers_AD/all .../alzheimers_HC/all \\
        --source_name alzheimers --task diagnosis \\
        --k 3 --output_dir /out/model_name
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    r2_score,
)

# ── Reuse from probe.py ─────────────────────────────────────────────

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
                "dir": emb_dir,
            }
    return subjects


# ── Task config ─────────────────────────────────────────────────────

TASK_CONFIG = {
    "diagnosis": {"metadata_field": "group", "task_type": "classification"},
    "sex": {"metadata_field": None, "task_type": "classification"},  # nsrr_sex or sex
    "age_regression": {"metadata_field": None, "task_type": "regression"},
}


def get_label(meta, task):
    """Extract label from metadata for a given task."""
    if task == "diagnosis":
        group = meta.get("group")
        if group is None:
            return None
        return str(group)
    elif task == "sex":
        for field in ["nsrr_sex", "sex"]:
            val = meta.get(field)
            if val is not None:
                val_str = str(val).lower()
                if val_str in ("male", "m", "1", "1.0"):
                    return 0
                elif val_str in ("female", "f", "2", "2.0"):
                    return 1
        return None
    elif task == "age_regression":
        for field in ["nsrr_age", "age", "age_msl_testing"]:
            val = meta.get(field)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None
    return None


# ── Core pipeline ───────────────────────────────────────────────────

def align_centroids(local_centroids, global_centroids):
    """Align K local centroids to K global centroids using Hungarian algorithm.

    Args:
        local_centroids: (K, D)
        global_centroids: (K, D)
    Returns:
        aligned: (K, D) — local centroids reordered to match global
    """
    cost = cdist(local_centroids, global_centroids)  # (K, K)
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned = np.zeros_like(local_centroids)
    for r, c in zip(row_ind, col_ind):
        aligned[c] = local_centroids[r]
    return aligned


def build_features(subjects, task, K):
    """Build feature matrix using local K-Means + global alignment.

    Returns: X (N, K*D), labels list, sids list, groups list
    """
    # Step 1: Load all embeddings and collect labels
    all_embs = []  # list of (N_i, D) arrays
    all_sids = []
    all_labels = []
    all_groups = []

    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)

        label = get_label(meta, task)
        if label is None:
            continue

        emb = np.load(info["emb_path"]).astype(np.float32)
        if len(emb) < K:
            continue  # too few epochs for K clusters

        all_embs.append(emb)
        all_sids.append(sid)
        all_labels.append(label)
        all_groups.append(get_group_key(sid))

    if not all_embs:
        return None, None, None, None

    print(f"  Loaded {len(all_embs)} subjects")

    # Step 2: Global K-Means on all embeddings concatenated
    print(f"  Fitting global K-Means (K={K})...")
    all_concat = np.concatenate(all_embs, axis=0)
    global_km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, n_init=3)
    global_km.fit(all_concat)
    global_centroids = global_km.cluster_centers_  # (K, D)
    del all_concat
    print(f"  Global centroids shape: {global_centroids.shape}")

    # Step 3: Per-subject local K-Means + alignment
    D = all_embs[0].shape[1]
    X = np.zeros((len(all_embs), K * D), dtype=np.float32)

    for i, emb in enumerate(all_embs):
        local_km = KMeans(n_clusters=K, random_state=42, n_init=3, max_iter=100)
        local_km.fit(emb)
        local_centroids = local_km.cluster_centers_  # (K, D)

        aligned = align_centroids(local_centroids, global_centroids)
        X[i] = aligned.flatten()

    print(f"  Feature matrix: {X.shape}")
    return X, all_labels, all_sids, all_groups


# ── Probing ─────────────────────────────────────────────────────────

def probe(X, labels, sids, groups, task_type, output_dir, n_folds=5):
    """Run 5-fold CV probing."""
    sids = np.array(sids)
    groups_arr = np.array(groups)
    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])

    is_regression = task_type == "regression"

    if is_regression:
        y = np.array(labels, dtype=np.float64)
        cv = GroupKFold(n_splits=min(n_folds, len(unique_groups)))
    else:
        unique_labels = sorted(set(labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_int[l] for l in labels], dtype=np.int64)
        n_classes = len(unique_labels)
        print(f"  Classes: {unique_labels}, dist: {np.bincount(y, minlength=n_classes).tolist()}")

        if n_classes < 2:
            print("  SKIP: only 1 class")
            return None
        cv = StratifiedGroupKFold(n_splits=min(n_folds, len(unique_groups)),
                                   shuffle=True, random_state=42)

    if len(unique_groups) < 2:
        print("  SKIP: not enough groups")
        return None

    os.makedirs(output_dir, exist_ok=True)
    fold_metrics = []
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, group_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": sids[train_idx].tolist(),
            "test": sids[test_idx].tolist(),
        }

        if is_regression:
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = {"mae": mae, "r2": r2}
            print(f"    Fold {fold_idx}: MAE={mae:.4f}  R2={r2:.4f}")
        else:
            if len(np.unique(y_train)) < 2:
                print(f"    Fold {fold_idx}: SKIP (single class)")
                continue
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_test, y_pred)
            metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}
            print(f"    Fold {fold_idx}: acc={acc:.4f}  f1={f1:.4f}  kappa={kappa:.4f}")

        fold_metrics.append(metrics)

        joblib.dump(clf, os.path.join(fold_dir, "classifier.joblib"))
        predictions = {}
        for i, idx in enumerate(test_idx):
            entry = {"y_true": float(y_test[i]) if is_regression else int(y_test[i]),
                     "y_pred": float(y_pred[i]) if is_regression else int(y_pred[i])}
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
        return None

    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(fold_metrics)
    summary["n_subjects"] = int(len(X))
    summary["n_groups"] = int(len(unique_groups))
    summary["type"] = "subject_wise_local_kmeans"
    if not is_regression:
        summary["classes"] = [str(l) for l in unique_labels]

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if is_regression:
        print(f"\n  Summary: MAE={summary['mae']['mean']:.4f}±{summary['mae']['std']:.4f}")
    else:
        print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
              f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    return summary


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Subject-wise probing with local K-Means prototypes"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--k", type=int, default=3, help="Number of local clusters")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Local K-Means probing: K={args.k}, task={args.task}, source={args.source_name}")

    subjects = load_subjects(args.emb_dirs)
    print(f"Loaded {len(subjects)} subjects from {len(args.emb_dirs)} dir(s)")

    if not subjects:
        print("No subjects found.")
        return

    task_cfg = TASK_CONFIG[args.task]
    task_type = task_cfg["task_type"]

    X, labels, sids, groups = build_features(subjects, args.task, args.k)
    if X is None:
        print("No valid subjects found.")
        return

    task_output_dir = os.path.join(args.output_dir, args.source_name, args.task)
    summary = probe(X, labels, sids, groups, task_type, task_output_dir, args.n_folds)

    print(f"\nResults: {task_output_dir}/")


if __name__ == "__main__":
    main()
