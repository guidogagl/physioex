"""GT-free prototype sequence probing for disease discrimination.

Discovers K global prototypes via K-Means on epoch embeddings (no GT labels),
assigns each epoch to its nearest prototype to form a "prototype hypnogram",
then extracts permanence and transition features from that sequence.

Feature levels:
  compact    — proportions + mean/std bout length + frag + entropy  (3K+2 dim)
  transition — compact + K×K transition probability matrix          (K²+3K+2 dim)
  full       — transition + per-slot mean/std embeddings            (+256K dim)
  local      — transition + local centroid deviations               (+128K dim)

Pipeline per fold:
  1. Fit global K-Means on TRAIN embeddings → K prototypes
  2. Assign every epoch (train+test) to nearest prototype
  3. Extract features from the prototype assignment sequence
  4. Optional PCA (fit on train)
  5. LogisticRegression / Ridge → disease prediction

Usage:
    python proto_sequence_probe.py \\
        --emb_dirs .../alzheimers_AD/all .../alzheimers_HC/all \\
        --source_name alzheimers --task diagnosis \\
        --K 20 --feature_level compact --seed 42 --output_dir /out/model_name
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy as shannon_entropy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


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
                "dir": emb_dir,
            }
    return subjects


# ── Task config ─────────────────────────────────────────────────────

TASK_CONFIG = {
    "diagnosis": {"task_type": "classification"},
    "sex": {"task_type": "classification"},
    "age_regression": {"task_type": "regression"},
}


def get_label(meta, task):
    if task == "diagnosis":
        group = meta.get("group")
        return str(group) if group is not None else None
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


# ── Prototype discovery ─────────────────────────────────────────────

def fit_global_prototypes(embs_list, K, seed):
    """Fit K-Means on concatenated embeddings to discover global prototypes.

    Args:
        embs_list: list of (N_i, D) arrays
        K: number of prototypes
        seed: random seed
    Returns:
        centroids: (K, D)
    """
    all_emb = np.concatenate(embs_list, axis=0)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=3)
    km.fit(all_emb)
    return km.cluster_centers_


def assign_to_prototypes(emb, centroids):
    """Assign each epoch to nearest global prototype.

    Args:
        emb: (N, D)
        centroids: (K, D)
    Returns:
        assignments: (N,) int
    """
    dists = cdist(emb, centroids)  # (N, K)
    return dists.argmin(axis=1)


# ── Feature computation ─────────────────────────────────────────────

def compute_permanence_features(assignments, K):
    """Compute per-prototype permanence features from assignment sequence.

    Returns: (3K,) — proportions (K) + mean bout (K) + std bout (K)
    """
    N = len(assignments)
    proportions = np.zeros(K, dtype=np.float32)
    mean_bouts = np.zeros(K, dtype=np.float32)
    std_bouts = np.zeros(K, dtype=np.float32)

    if N == 0:
        return np.concatenate([proportions, mean_bouts, std_bouts])

    # Proportions
    for p in range(K):
        proportions[p] = (assignments == p).sum() / N

    # Bout lengths via run-length encoding
    bouts = {p: [] for p in range(K)}
    current = assignments[0]
    bout_len = 1
    for i in range(1, N):
        if assignments[i] == current:
            bout_len += 1
        else:
            bouts[current].append(bout_len)
            current = assignments[i]
            bout_len = 1
    bouts[current].append(bout_len)

    for p in range(K):
        if bouts[p]:
            mean_bouts[p] = np.mean(bouts[p])
            if len(bouts[p]) > 1:
                std_bouts[p] = np.std(bouts[p])

    return np.concatenate([proportions, mean_bouts, std_bouts])


def compute_global_stats(assignments, K):
    """Compute fragmentation index and entropy of prototype distribution.

    Returns: (2,) — [frag_index, entropy]
    """
    N = len(assignments)
    if N < 2:
        return np.zeros(2, dtype=np.float32)

    frag_index = float((assignments[1:] != assignments[:-1]).sum()) / N

    proportions = np.zeros(K, dtype=np.float64)
    for p in range(K):
        proportions[p] = (assignments == p).sum()
    proportions = proportions / proportions.sum()
    ent = float(shannon_entropy(proportions + 1e-12))

    return np.array([frag_index, ent], dtype=np.float32)


def compute_transition_matrix(assignments, K):
    """Compute row-normalized transition probability matrix.

    Returns: (K*K,) — flattened transition matrix
    """
    N = len(assignments)
    T = np.zeros((K, K), dtype=np.float32)

    if N < 2:
        return T.flatten()

    for i in range(N - 1):
        T[assignments[i], assignments[i + 1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums

    return T.flatten()


def compute_embedding_features(emb, assignments, K):
    """Compute per-slot mean and std embeddings.

    Returns: (2*K*D,) — mean embeddings (K*D) + std embeddings (K*D)
    """
    D = emb.shape[1]
    means = np.zeros((K, D), dtype=np.float32)
    stds = np.zeros((K, D), dtype=np.float32)

    for p in range(K):
        mask = assignments == p
        count = mask.sum()
        if count > 0:
            means[p] = emb[mask].mean(axis=0)
        if count > 1:
            stds[p] = emb[mask].std(axis=0)

    return np.concatenate([means.flatten(), stds.flatten()])


def compute_local_features(emb, global_centroids, k, K, seed):
    """Compute local centroid deviations from global prototypes.

    For each subject, fits k local centroids and maps each to its nearest
    global prototype. Stores the deviation (local - global) per slot.

    Returns: (K*D,)
    """
    D = emb.shape[1]
    deviations = np.zeros((K, D), dtype=np.float32)

    if len(emb) < k:
        return deviations.flatten()

    local_km = KMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=100)
    local_km.fit(emb)
    local_centroids = local_km.cluster_centers_  # (k, D)

    # Map each local centroid to nearest global prototype
    dists = cdist(local_centroids, global_centroids)  # (k, K)
    best_dist = np.full(K, np.inf)

    for li in range(k):
        gi = dists[li].argmin()
        if dists[li, gi] < best_dist[gi]:
            best_dist[gi] = dists[li, gi]
            deviations[gi] = local_centroids[li] - global_centroids[gi]

    return deviations.flatten()


def compute_subject_features(emb, assignments, global_centroids, K,
                             feature_level, k=5, seed=42):
    """Orchestrate feature computation based on feature level.

    All levels include: permanence (3K) + frag + entropy (2) = 3K+2
    transition+: adds K×K transition matrix
    full: adds per-slot mean/std embeddings
    local: adds local centroid deviations

    Returns: 1-D feature vector
    """
    parts = [
        compute_permanence_features(assignments, K),
        compute_global_stats(assignments, K),
    ]

    if feature_level in ("transition", "full", "local"):
        parts.append(compute_transition_matrix(assignments, K))

    if feature_level == "full":
        parts.append(compute_embedding_features(emb, assignments, K))

    if feature_level == "local":
        parts.append(compute_local_features(emb, global_centroids, k, K, seed))

    return np.concatenate(parts)


# ── Probing ─────────────────────────────────────────────────────────

def probe(all_embs, labels, sids, groups, task_type, K, k, feature_level,
          output_dir, n_folds=5, seed=42, pca=0, C=1.0, fit_all=False):
    """Run 5-fold CV with per-fold prototype discovery."""

    sids_arr = np.array(sids)
    groups_arr = np.array(groups)
    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])
    n_subjects = len(all_embs)

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
                                   shuffle=True, random_state=seed)

    if len(unique_groups) < 2:
        print("  SKIP: not enough groups")
        return None

    # If --fit_all: fit prototypes once on all subjects
    global_centroids_all = None
    if fit_all:
        print(f"  Fitting global K-Means on ALL subjects (K={K})...")
        global_centroids_all = fit_global_prototypes(all_embs, K, seed)

    # Placeholder X for CV split (actual features computed per fold)
    X_placeholder = np.zeros((n_subjects, 1))

    os.makedirs(output_dir, exist_ok=True)
    fold_metrics = []
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_placeholder, y, group_ids)):
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": sids_arr[train_idx].tolist(),
            "test": sids_arr[test_idx].tolist(),
        }

        # Step 1: Global prototypes (per-fold or precomputed)
        if fit_all:
            centroids = global_centroids_all
        else:
            train_embs = [all_embs[i] for i in train_idx]
            centroids = fit_global_prototypes(train_embs, K, seed)

        np.save(os.path.join(fold_dir, "global_centroids.npy"), centroids)

        # Step 2: Assign + compute features for all subjects
        X = np.zeros((n_subjects, 0), dtype=np.float32)
        feat_vectors = []
        for i in range(n_subjects):
            assignments = assign_to_prototypes(all_embs[i], centroids)
            feat = compute_subject_features(all_embs[i], assignments, centroids,
                                            K, feature_level, k, seed)
            feat_vectors.append(feat)
        X = np.array(feat_vectors, dtype=np.float32)

        if fold_idx == 0:
            print(f"  Feature dim: {X.shape[1]} ({feature_level}, K={K})")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Step 3: Optional PCA
        if pca > 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            n_comp = min(pca, X_train.shape[0], X_train.shape[1])
            pca_model = PCA(n_components=n_comp, random_state=seed)
            X_train = pca_model.fit_transform(X_train)
            X_test = pca_model.transform(X_test)
            joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
            joblib.dump(pca_model, os.path.join(fold_dir, "pca.joblib"))

        # Step 4: Classify
        if is_regression:
            clf = Ridge(alpha=1.0 / C)
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
            clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=1)
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
            predictions[sids_arr[idx]] = entry
        with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    if not fold_metrics:
        print("  No valid folds")
        return None

    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(fold_metrics)
    summary["n_subjects"] = n_subjects
    summary["n_groups"] = int(len(unique_groups))
    summary["K"] = K
    summary["feature_level"] = feature_level
    summary["feat_dim"] = int(feat_vectors[0].shape[0])
    summary["pca"] = pca
    summary["C"] = C
    summary["seed"] = seed
    summary["fit_all"] = fit_all
    summary["type"] = "proto_sequence"
    if not is_regression:
        summary["classes"] = [str(l) for l in unique_labels]

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if is_regression:
        print(f"\n  Summary: MAE={summary['mae']['mean']:.4f}+/-{summary['mae']['std']:.4f}")
    else:
        print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}+/-{summary['accuracy']['std']:.4f}, "
              f"f1={summary['f1_macro']['mean']:.4f}+/-{summary['f1_macro']['std']:.4f}")
    return summary


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GT-free prototype sequence probing for disease discrimination"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--K", type=int, default=20,
                        help="Number of global prototypes")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of local prototypes per subject (feature_level=local)")
    parser.add_argument("--feature_level", type=str, default="compact",
                        choices=["compact", "transition", "full", "local"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA components before classification (0=disabled)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="LogReg regularization (lower = stronger)")
    parser.add_argument("--fit_all", action="store_true",
                        help="Fit K-Means on all subjects (faster, slight leakage)")
    args = parser.parse_args()

    print(f"Proto-sequence probe: K={args.K}, k={args.k}, level={args.feature_level}, "
          f"C={args.C}, pca={args.pca}, seed={args.seed}, source={args.source_name}")

    subjects = load_subjects(args.emb_dirs)
    print(f"Loaded {len(subjects)} subjects from {len(args.emb_dirs)} dir(s)")

    if not subjects:
        print("No subjects found.")
        return

    task_type = TASK_CONFIG[args.task]["task_type"]

    # Load all data
    all_embs, all_labels, all_sids, all_groups = [], [], [], []
    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        label = get_label(meta, args.task)
        if label is None:
            continue
        emb = np.load(info["emb_path"]).astype(np.float32)
        if len(emb) == 0:
            continue
        all_embs.append(emb)
        all_sids.append(sid)
        all_labels.append(label)
        all_groups.append(get_group_key(sid))

    if not all_embs:
        print("No valid subjects found.")
        return

    print(f"  Subjects: {len(all_embs)}")

    task_output_dir = os.path.join(args.output_dir, args.source_name, args.task)
    summary = probe(all_embs, all_labels, all_sids, all_groups, task_type,
                    args.K, args.k, args.feature_level, task_output_dir,
                    args.n_folds, args.seed, args.pca, args.C, args.fit_all)

    print(f"\nResults: {task_output_dir}/")


if __name__ == "__main__":
    main()
