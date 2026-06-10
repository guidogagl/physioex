"""Prototypical disease discrimination via class-conditional K-Means.

Discovers K prototypical patterns per disease class from training epochs,
then classifies test subjects based on their histogram of prototype assignments.

Pipeline per fold:
  1. Split subjects into train/test
  2. K-Means on train class-A epochs → K prototypes for class A
  3. K-Means on train class-B epochs → K prototypes for class B
  4. For each subject: assign epochs to nearest prototype → normalized histogram (2K dim)
  5. LogisticRegression on histograms → disease prediction
  6. Coherence analysis: which prototypes discriminate consistently across train/test

Usage:
    python proto_probe.py \\
        --emb_dirs .../alzheimers_AD/all .../alzheimers_HC/all \\
        --source_name alzheimers --task diagnosis \\
        --k 5 --seed 42 --output_dir /out/model_name
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


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


def get_diagnosis_label(meta):
    group = meta.get("group")
    return str(group) if group is not None else None


# ── Core pipeline ───────────────────────────────────────────────────

def run_fold(subjects_data, train_sids, test_sids, K, seed, fold_dir):
    """Run one fold of prototypical disease probing.

    Args:
        subjects_data: dict sid -> {"emb": np.array, "label": str, "group": str}
        train_sids, test_sids: lists of subject IDs
        K: prototypes per class
        seed: random seed
        fold_dir: output directory for this fold

    Returns: metrics dict
    """
    os.makedirs(fold_dir, exist_ok=True)

    # Separate train epochs by class
    classes = sorted(set(subjects_data[s]["label"] for s in train_sids))
    assert len(classes) == 2, f"Expected 2 classes, got {classes}"

    class_epochs = {c: [] for c in classes}
    for sid in train_sids:
        c = subjects_data[sid]["label"]
        class_epochs[c].append(subjects_data[sid]["emb"])

    # Step 1: K-Means per class on training data
    prototypes = []
    prototype_classes = []
    for c in classes:
        epochs = np.concatenate(class_epochs[c], axis=0)
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=min(4096, len(epochs)), n_init=3)
        km.fit(epochs)
        prototypes.append(km.cluster_centers_)
        prototype_classes.extend([c] * K)

    prototypes = np.concatenate(prototypes, axis=0)  # (2K, D)
    n_protos = len(prototypes)

    # Save prototypes
    np.save(os.path.join(fold_dir, "prototypes.npy"), prototypes)
    with open(os.path.join(fold_dir, "prototype_labels.json"), "w") as f:
        json.dump({"classes": classes, "prototype_class": prototype_classes,
                    "K_per_class": K, "n_prototypes": n_protos}, f, indent=2)

    # Step 2: Compute mean-distance features for all subjects
    def compute_features(emb):
        dists = cdist(emb, prototypes)  # (N_epochs, 2K)
        return dists.mean(axis=0)  # (2K,) mean distance to each prototype

    X_train, y_train = [], []
    for sid in train_sids:
        X_train.append(compute_features(subjects_data[sid]["emb"]))
        y_train.append(classes.index(subjects_data[sid]["label"]))
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test, y_test, test_ids = [], [], []
    for sid in test_sids:
        X_test.append(compute_features(subjects_data[sid]["emb"]))
        y_test.append(classes.index(subjects_data[sid]["label"]))
        test_ids.append(sid)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Save histograms
    np.save(os.path.join(fold_dir, "histograms_train.npy"), X_train)
    np.save(os.path.join(fold_dir, "histograms_test.npy"), X_test)

    # Step 3: Classify
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}

    # Save classifier + predictions
    joblib.dump(clf, os.path.join(fold_dir, "classifier.joblib"))
    predictions = {}
    for i, sid in enumerate(test_ids):
        predictions[sid] = {
            "y_true": int(y_test[i]),
            "y_pred": int(y_pred[i]),
            "y_proba": y_proba[i].tolist(),
        }
    with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f)
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 4: Coherence analysis
    coherence = {"classes": classes, "prototypes": []}
    for j in range(n_protos):
        proto_info = {
            "index": j,
            "class": prototype_classes[j],
        }
        for split_name, sids, X_hist, y_arr in [
            ("train", train_sids, X_train, y_train),
            ("test", test_ids, X_test, y_test),
        ]:
            for ci, c in enumerate(classes):
                mask = y_arr == ci
                if mask.sum() > 0:
                    proto_info[f"{split_name}_{c}_mean"] = float(X_hist[mask, j].mean())
                    proto_info[f"{split_name}_{c}_std"] = float(X_hist[mask, j].std())
        prototypes_info = proto_info
        coherence["prototypes"].append(prototypes_info)

    # Classifier weights (which prototypes are most discriminative)
    if hasattr(clf, "coef_"):
        coherence["classifier_weights"] = clf.coef_[0].tolist()

    with open(os.path.join(fold_dir, "coherence.json"), "w") as f:
        json.dump(coherence, f, indent=2)

    return metrics


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prototypical disease discrimination via class-conditional K-Means"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--source_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="diagnosis")
    parser.add_argument("--k", type=int, default=5, help="Prototypes per class")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Proto probe: K={args.k}, task={args.task}, source={args.source_name}, seed={args.seed}")

    subjects = load_subjects(args.emb_dirs)
    print(f"Loaded {len(subjects)} subjects")

    # Load all data
    subjects_data = {}
    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        label = get_diagnosis_label(meta)
        if label is None:
            continue
        emb = np.load(info["emb_path"]).astype(np.float32)
        subjects_data[sid] = {
            "emb": emb,
            "label": label,
            "group": get_group_key(sid),
        }

    sids = sorted(subjects_data.keys())
    labels = [subjects_data[s]["label"] for s in sids]
    groups = [subjects_data[s]["group"] for s in sids]

    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels])
    unique_groups = list(set(groups))
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups])

    print(f"  Subjects: {len(sids)}, Classes: {unique_labels}, "
          f"Dist: {np.bincount(y).tolist()}")

    n_folds = min(args.n_folds, len(unique_groups))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    output_dir = os.path.join(args.output_dir, args.source_name, args.task)
    os.makedirs(output_dir, exist_ok=True)

    fold_metrics = []
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(sids)), y, group_ids)):
        train_sids = [sids[i] for i in train_idx]
        test_sids = [sids[i] for i in test_idx]

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": train_sids, "test": test_sids,
        }

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        print(f"\n  Fold {fold_idx}: train={len(train_sids)}, test={len(test_sids)}")

        metrics = run_fold(subjects_data, train_sids, test_sids, args.k, args.seed, fold_dir)
        fold_metrics.append(metrics)
        print(f"    acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}  kappa={metrics['kappa']:.4f}")

    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    # Summary
    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(fold_metrics)
    summary["n_subjects"] = len(sids)
    summary["K_per_class"] = args.k
    summary["seed"] = args.seed
    summary["classes"] = unique_labels
    summary["type"] = "prototypical_disease_discrimination"

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
