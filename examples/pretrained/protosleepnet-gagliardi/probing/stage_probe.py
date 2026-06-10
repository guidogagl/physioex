"""Stage-conditioned subject probing for disease discrimination.

Groups epoch embeddings by sleep stage, computes per-stage mean embeddings,
and concatenates into a 640-dim feature vector (5 stages × 128 dim).
Optionally appends stage proportions (+5 dim).

Usage:
    python stage_probe.py \\
        --emb_dirs .../alzheimers_AD/all .../alzheimers_HC/all \\
        --source_name alzheimers --task diagnosis \\
        --C 0.01 --seed 42 --output_dir /out/model_name
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

N_STAGES = 5  # W=0, N1=1, N2=2, N3=3, REM=4
D = 128       # embedding dim


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
                "labels_path": os.path.join(emb_dir, f"{sid}_labels.npy"),
                "metadata_path": os.path.join(emb_dir, f"{sid}_metadata.json"),
                "dir": emb_dir,
            }
    return subjects


# ── Feature computation ─────────────────────────────────────────────

def compute_stage_features(emb, labels, add_std=False, add_proportions=True, add_fragmentation=False):
    """Compute per-stage mean embeddings + optional std, proportions, fragmentation.

    Args:
        emb: (N, 128) epoch embeddings
        labels: (N,) sleep stage labels (0-4, -1=unscored)
        add_std: if True, append per-stage std embeddings (+640 dim)
        add_proportions: if True, append 5-dim stage proportion vector
        add_fragmentation: if True, append fragmentation features (+7 dim)

    Returns:
        feature vector of variable length depending on flags
    """
    # Filter out unscored epochs
    valid = labels >= 0
    emb = emb[valid]
    labels = labels[valid]

    if len(emb) == 0:
        dim = N_STAGES * D
        if add_std:
            dim += N_STAGES * D
        if add_proportions:
            dim += N_STAGES
        if add_fragmentation:
            dim += 2 + N_STAGES  # frag_index, n_transitions, 5 mean_bout
        return np.zeros(dim, dtype=np.float32)

    # Per-stage mean embeddings
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

    if add_proportions:
        total = stage_counts.sum()
        proportions = stage_counts / total if total > 0 else stage_counts
        parts.append(proportions)  # (5,)

    if add_fragmentation:
        # Fragmentation index + transition count
        transitions = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                transitions += 1
        frag_index = transitions / len(labels) if len(labels) > 0 else 0.0

        # Mean bout length per stage
        bouts = {s: [] for s in range(N_STAGES)}
        current_stage = labels[0]
        bout_len = 1
        for i in range(1, len(labels)):
            if labels[i] == current_stage:
                bout_len += 1
            else:
                bouts[current_stage].append(bout_len)
                current_stage = labels[i]
                bout_len = 1
        bouts[current_stage].append(bout_len)
        mean_bouts = np.array([np.mean(bouts[s]) if bouts[s] else 0.0 for s in range(N_STAGES)],
                              dtype=np.float32)

        parts.append(np.array([frag_index, float(transitions)], dtype=np.float32))
        parts.append(mean_bouts)  # (5,)

    return np.concatenate(parts)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage-conditioned subject probing for disease discrimination"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="diagnosis")
    parser.add_argument("--C", type=float, default=0.01,
                        help="LogReg regularization (lower = stronger)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_proportions", action="store_true",
                        help="Disable stage proportion features")
    parser.add_argument("--add_std", action="store_true",
                        help="Add per-stage std embeddings (+640 dim)")
    parser.add_argument("--add_frag", action="store_true",
                        help="Add fragmentation features (+7 dim)")
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA components before classification (0=disabled)")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    add_proportions = not args.no_proportions

    print(f"Stage probe: C={args.C}, seed={args.seed}, source={args.source_name}, "
          f"std={args.add_std}, frag={args.add_frag}, pca={args.pca}")

    subjects = load_subjects(args.emb_dirs)
    print(f"Loaded {len(subjects)} subjects")

    # Load all data
    sids, X_all, labels_all, groups = [], [], [], []
    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]) or not os.path.exists(info["labels_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        group = meta.get("group")
        if group is None:
            continue

        emb = np.load(info["emb_path"]).astype(np.float32)
        stage_labels = np.load(info["labels_path"]).astype(np.int64)

        # Align lengths
        n = min(len(emb), len(stage_labels))
        emb, stage_labels = emb[:n], stage_labels[:n]

        features = compute_stage_features(emb, stage_labels,
                                          add_std=args.add_std,
                                          add_proportions=add_proportions,
                                          add_fragmentation=args.add_frag)

        sids.append(sid)
        X_all.append(features)
        labels_all.append(str(group))
        groups.append(get_group_key(sid))

    X = np.array(X_all)
    unique_labels = sorted(set(labels_all))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels_all])
    sids = np.array(sids)
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

        # Optional PCA (fit on train, transform both)
        if args.pca > 0:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
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
            "train": sids[train_idx].tolist(),
            "test": sids[test_idx].tolist(),
        }

        if len(np.unique(y_train)) < 2:
            print(f"    Fold {fold_idx}: SKIP (single class)")
            continue

        clf = LogisticRegression(max_iter=1000, C=args.C, solver="lbfgs", n_jobs=-1)
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
            predictions[sids[idx]] = {
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
    summary["feat_dim"] = int(X.shape[1])
    summary["add_proportions"] = add_proportions
    summary["classes"] = unique_labels
    summary["type"] = "stage_conditioned"

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
