"""Discriminative prototype mining for disease discrimination.

Discovers K global prototypes via K-Means, profiles each subject's usage,
tests which prototypes discriminate HC vs PD (Welch t-test + Cohen's d),
characterizes prototypes by sleep stage, and classifies using only
discriminative prototypes.

Usage:
    python discriminative_probes.py \\
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \\
        --staging_dir .../proto-st-3ch-mixer/parkinsons_night/staging \\
        --source_name parkinsons_night --K 30 --C 0.1 --d_threshold 0.4 \\
        --seed 42 --output_dir /out/model_name
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


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
    """Merge out-of-fold staging predictions across all folds."""
    all_preds = {}
    for fold_dir in sorted(glob.glob(os.path.join(staging_dir, "fold_*"))):
        pred_path = os.path.join(fold_dir, "predictions.json")
        if not os.path.exists(pred_path):
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        for sid, p in preds.items():
            if sid not in all_preds:
                all_preds[sid] = {
                    "y_proba": np.array(p["y_proba"], dtype=np.float32),
                }
    return all_preds


# ── Prototype discovery & profiling ─────────────────────────────────

def discover_prototypes(embs_list, K, seed):
    """K-Means on all embeddings → K global centroids."""
    all_emb = np.concatenate(embs_list, axis=0)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=3)
    km.fit(all_emb)
    return km.cluster_centers_


def compute_subject_profile(emb, centroids, K):
    """Compute per-prototype usage profile for one subject.

    Returns dict with:
        proportion: (K,) fraction of epochs per prototype
        bout_mean: (K,) mean bout length per prototype
        bout_std: (K,) std of bout lengths per prototype
        intra_std: (K,) mean of per-dim std within each prototype cluster
        assignments: (N,) prototype assignments
    """
    N = len(emb)
    assignments = cdist(emb, centroids).argmin(axis=1)

    proportion = np.zeros(K, dtype=np.float32)
    bout_mean = np.zeros(K, dtype=np.float32)
    bout_std = np.zeros(K, dtype=np.float32)
    intra_std = np.zeros(K, dtype=np.float32)

    # Proportions
    for k in range(K):
        proportion[k] = (assignments == k).sum() / N

    # Bout lengths
    bouts = {k: [] for k in range(K)}
    current = assignments[0]
    blen = 1
    for i in range(1, N):
        if assignments[i] == current:
            blen += 1
        else:
            bouts[current].append(blen)
            current = assignments[i]
            blen = 1
    bouts[current].append(blen)

    for k in range(K):
        if bouts[k]:
            bout_mean[k] = np.mean(bouts[k])
            if len(bouts[k]) > 1:
                bout_std[k] = np.std(bouts[k])

    # Intra-cluster variability
    for k in range(K):
        mask = assignments == k
        if mask.sum() > 1:
            intra_std[k] = emb[mask].std(axis=0).mean()

    return {
        "proportion": proportion,
        "bout_mean": bout_mean,
        "bout_std": bout_std,
        "intra_std": intra_std,
        "assignments": assignments,
    }


# ── Discrimination testing ──────────────────────────────────────────

def test_discrimination(profiles, labels, K, d_threshold=0.4):
    """Test each prototype for discriminative power between classes.

    Args:
        profiles: list of profile dicts (one per subject)
        labels: list of class labels (str)
        K: number of prototypes
        d_threshold: Cohen's d threshold for "discriminative"

    Returns: list of per-prototype analysis dicts
    """
    classes = sorted(set(labels))
    assert len(classes) == 2

    # Split by class
    idx_0 = [i for i, l in enumerate(labels) if l == classes[0]]
    idx_1 = [i for i, l in enumerate(labels) if l == classes[1]]

    analysis = []
    for k in range(K):
        proto_info = {"index": k}

        best_abs_d = 0
        for feat_name in ["proportion", "bout_mean", "bout_std", "intra_std"]:
            vals_0 = np.array([profiles[i][feat_name][k] for i in idx_0])
            vals_1 = np.array([profiles[i][feat_name][k] for i in idx_1])

            # Welch t-test
            if vals_0.std() == 0 and vals_1.std() == 0:
                t_stat, p_val, d = 0.0, 1.0, 0.0
            else:
                t_stat, p_val = stats.ttest_ind(vals_0, vals_1, equal_var=False)
                pooled_std = np.sqrt((vals_0.std()**2 + vals_1.std()**2) / 2)
                d = (vals_0.mean() - vals_1.mean()) / pooled_std if pooled_std > 0 else 0.0

            proto_info[f"{feat_name}_mean_{classes[0]}"] = float(vals_0.mean())
            proto_info[f"{feat_name}_mean_{classes[1]}"] = float(vals_1.mean())
            proto_info[f"{feat_name}_d"] = float(d)
            proto_info[f"{feat_name}_p"] = float(p_val)

            if abs(d) > best_abs_d:
                best_abs_d = abs(d)

        proto_info["best_abs_d"] = float(best_abs_d)

        # Classify prototype
        prop_0 = np.mean([profiles[i]["proportion"][k] for i in idx_0])
        prop_1 = np.mean([profiles[i]["proportion"][k] for i in idx_1])

        if prop_0 < 0.005 and prop_1 >= 0.005:
            proto_info["type"] = f"class_specific_{classes[1]}"
        elif prop_1 < 0.005 and prop_0 >= 0.005:
            proto_info["type"] = f"class_specific_{classes[0]}"
        elif best_abs_d >= d_threshold:
            proto_info["type"] = "shared_discriminative"
        else:
            proto_info["type"] = "non_discriminative"

        analysis.append(proto_info)

    return analysis


# ── Stage characterization ──────────────────────────────────────────

def characterize_prototypes(centroids, staging_preds, embs, sids, K):
    """Map each prototype to its dominant sleep stage using staging predictions.

    Returns: list of (K,) dicts with stage_distribution and dominant_stage
    """
    # Accumulate stage probabilities per prototype
    stage_accum = np.zeros((K, 5), dtype=np.float64)
    stage_counts = np.zeros(K, dtype=np.float64)

    for i, sid in enumerate(sids):
        if sid not in staging_preds:
            continue
        emb = embs[i]
        proba = staging_preds[sid]["y_proba"]
        n = min(len(emb), len(proba))
        emb, proba = emb[:n], proba[:n]

        assignments = cdist(emb, centroids).argmin(axis=1)
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                stage_accum[k] += proba[mask].sum(axis=0)
                stage_counts[k] += mask.sum()

    characterization = []
    for k in range(K):
        if stage_counts[k] > 0:
            dist = stage_accum[k] / stage_counts[k]
        else:
            dist = np.zeros(5)
        dominant = int(np.argmax(dist))
        characterization.append({
            "stage_distribution": dist.tolist(),
            "dominant_stage": STAGE_NAMES[dominant],
            "dominant_prob": float(dist[dominant]),
            "n_epochs": int(stage_counts[k]),
        })
    return characterization


# ── Feature assembly ────────────────────────────────────────────────

def build_feature_vector(profile, selected_protos, feature_set):
    """Build feature vector using only selected (discriminative) prototypes."""
    parts = []
    for k in selected_protos:
        parts.append(profile["proportion"][k:k+1])
        if feature_set in ("bout", "full"):
            parts.append(profile["bout_mean"][k:k+1])
            parts.append(profile["bout_std"][k:k+1])
        if feature_set == "full":
            parts.append(profile["intra_std"][k:k+1])
    return np.concatenate(parts)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discriminative prototype mining for disease discrimination"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--staging_dir", type=str, default=None,
                        help="Path to staging probing results (optional, for characterization)")
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="diagnosis")
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--d_threshold", type=float, default=0.4)
    parser.add_argument("--feature_set", type=str, default="full",
                        choices=["proportion", "bout", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Discriminative probes: K={args.K}, C={args.C}, d_thresh={args.d_threshold}, "
          f"feat={args.feature_set}, seed={args.seed}, source={args.source_name}")

    # Load subjects
    subjects = load_subjects(args.emb_dirs)
    print(f"Found {len(subjects)} subjects")

    # Load staging predictions (optional)
    staging_preds = {}
    if args.staging_dir and os.path.exists(args.staging_dir):
        staging_preds = load_staging_predictions(args.staging_dir)
        print(f"Loaded staging predictions for {len(staging_preds)} subjects")

    # Load all data
    all_embs, all_labels, all_sids, all_groups = [], [], [], []
    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        label = get_label(meta)
        if label is None:
            continue
        emb = np.load(info["emb_path"]).astype(np.float32)
        if len(emb) == 0:
            continue
        all_embs.append(emb)
        all_labels.append(label)
        all_sids.append(sid)
        all_groups.append(get_group_key(sid))

    if not all_embs:
        print("No valid subjects.")
        return

    unique_labels = sorted(set(all_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in all_labels])
    sids_arr = np.array(all_sids)
    groups_arr = np.array(all_groups)
    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])
    n_subjects = len(all_embs)

    print(f"  Subjects: {n_subjects}, Classes: {unique_labels}, "
          f"Dist: {np.bincount(y).tolist()}")

    if len(unique_labels) < 2:
        print("  SKIP: only 1 class")
        return

    n_folds = min(args.n_folds, len(unique_groups))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    output_dir = os.path.join(args.output_dir, args.source_name, args.task)
    os.makedirs(output_dir, exist_ok=True)

    fold_metrics = []
    fold_assignments = {}
    all_proto_analyses = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(
            np.zeros((n_subjects, 1)), y, group_ids)):

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": sids_arr[train_idx].tolist(),
            "test": sids_arr[test_idx].tolist(),
        }

        y_train, y_test = y[train_idx], y[test_idx]

        # Step 1: Discover prototypes on TRAIN embeddings
        train_embs = [all_embs[i] for i in train_idx]
        centroids = discover_prototypes(train_embs, args.K, args.seed)
        np.save(os.path.join(fold_dir, "prototypes.npy"), centroids)

        # Step 2: Profile ALL subjects
        profiles = []
        for i in range(n_subjects):
            profiles.append(compute_subject_profile(all_embs[i], centroids, args.K))

        # Step 3: Test discrimination on TRAIN subjects only
        train_profiles = [profiles[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        proto_analysis = test_discrimination(train_profiles, train_labels,
                                              args.K, args.d_threshold)

        # Stage characterization
        if staging_preds:
            char = characterize_prototypes(centroids, staging_preds,
                                            all_embs, all_sids, args.K)
            for k in range(args.K):
                proto_analysis[k].update(char[k])

        all_proto_analyses.append(proto_analysis)

        # Select discriminative prototypes
        disc_protos = [pa["index"] for pa in proto_analysis
                       if pa["type"] != "non_discriminative"]

        if fold_idx == 0:
            n_disc = len(disc_protos)
            types = {}
            for pa in proto_analysis:
                t = pa["type"]
                types[t] = types.get(t, 0) + 1
            print(f"  Fold 0: {n_disc}/{args.K} discriminative prototypes, types: {types}")
            if staging_preds:
                for pa in proto_analysis:
                    if pa["type"] != "non_discriminative":
                        stage = pa.get("dominant_stage", "?")
                        prob = pa.get("dominant_prob", 0)
                        d = pa["best_abs_d"]
                        print(f"    proto {pa['index']:>2}: {pa['type']:<25} "
                              f"|d|={d:.3f}  stage={stage} ({prob:.1%})")

        if not disc_protos:
            print(f"    Fold {fold_idx}: no discriminative prototypes, using all")
            disc_protos = list(range(args.K))

        # Step 4: Build features and classify
        X_train = np.array([build_feature_vector(profiles[i], disc_protos, args.feature_set)
                            for i in train_idx])
        X_test = np.array([build_feature_vector(profiles[i], disc_protos, args.feature_set)
                           for i in test_idx])

        if fold_idx == 0:
            print(f"  Feature dim: {X_train.shape[1]} (from {len(disc_protos)} prototypes)")

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

        # Save prototype analysis for this fold
        with open(os.path.join(fold_dir, "prototype_analysis.json"), "w") as f:
            json.dump(proto_analysis, f, indent=2)

    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    if not fold_metrics:
        print("  No valid folds")
        return

    # Summary
    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(fold_metrics)
    summary["n_subjects"] = n_subjects
    summary["K"] = args.K
    summary["C"] = args.C
    summary["d_threshold"] = args.d_threshold
    summary["feature_set"] = args.feature_set
    summary["seed"] = args.seed
    summary["classes"] = unique_labels
    summary["type"] = "discriminative_prototype_mining"

    # Average number of discriminative prototypes across folds
    disc_counts = []
    mean_abs_ds = []
    for pa_list in all_proto_analyses:
        disc = [pa for pa in pa_list if pa["type"] != "non_discriminative"]
        disc_counts.append(len(disc))
        if disc:
            mean_abs_ds.append(np.mean([pa["best_abs_d"] for pa in disc]))
    summary["n_disc_protos_mean"] = float(np.mean(disc_counts))
    summary["n_disc_protos_std"] = float(np.std(disc_counts))
    if mean_abs_ds:
        summary["mean_abs_d"] = float(np.mean(mean_abs_ds))

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    print(f"  Disc protos: {summary['n_disc_protos_mean']:.1f}±{summary['n_disc_protos_std']:.1f} / {args.K}")
    if "mean_abs_d" in summary:
        print(f"  Mean |d| of disc protos: {summary['mean_abs_d']:.3f}")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
