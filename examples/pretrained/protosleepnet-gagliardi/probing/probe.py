"""Linear probing on pre-extracted epoch embeddings.

Runs 5-fold subject-wise cross-validation with sklearn classifiers.
Auto-discovers available tasks from files in the embedding directory.

Saves per-fold:
  - classifier.joblib          trained sklearn model
  - predictions.json           {subject_id: {y_true, y_pred, y_proba}}
  - fold_assignments.json      {fold_k: {train: [...], test: [...]}}
  - summary.json               aggregated metrics mean +/- std

Output layout::

    {output_dir}/{source_name}/{task_name}/
    ├── fold_assignments.json
    ├── fold_0/
    │   ├── classifier.joblib
    │   └── predictions.json
    ├── ...fold_4/
    └── summary.json

Usage:
    # Event-wise staging on SHHS in-domain (train+valid+test)
    python probe.py --emb_dirs .../train .../valid .../test \\
        --source_name shhs_visit1 --task staging --output_dir /out

    # Subject-wise age regression on MESA OOD
    python probe.py --emb_dirs .../mesa/all \\
        --source_name mesa --task age_regression \\
        --metadata_field nsrr_age --output_dir /out

    # Event-wise arousal on WSC visit1
    python probe.py --emb_dirs .../wsc_visit1/all \\
        --source_name wsc_visit1 --task arousal --output_dir /out

    # List available tasks
    python probe.py --emb_dirs .../train --discover
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    r2_score,
)

# ── Subject grouping ─────────────────────────────────────────────────
# Maps dataset prefixes to grouping functions.
# Returns a group key shared by all recordings of the same person.

def _group_sleepedf(sid):
    # SC4001E0, SC4001E1 → SC400 (same person, 2 nights)
    return sid[:5]

def _group_shhs(sid):
    # shhs1-200001, shhs2-200001 → 200001
    return sid.split("-", 1)[1] if "-" in sid else sid

def _group_wsc(sid):
    # wsc-visit1-10119, wsc-visit5-10119 → 10119
    parts = sid.split("-")
    return parts[-1] if len(parts) >= 3 else sid

def _group_identity(sid):
    return sid

GROUPING = {
    "SC": _group_sleepedf,     # sleepedf subject IDs start with SC
    "shhs": _group_shhs,
    "wsc": _group_wsc,
}

def get_group_key(sid):
    """Auto-detect grouping from subject_id prefix."""
    for prefix, fn in GROUPING.items():
        if sid.startswith(prefix):
            return fn(sid)
    return _group_identity(sid)


# ── Vector quantization ──────────────────────────────────────────────

def quantize_embeddings(emb, codebook):
    """Replace each embedding with its nearest codebook entry.
    Args: emb (N, D), codebook (M, D)
    Returns: quantized (N, D)
    """
    dists = cdist(emb, codebook)  # (N, M)
    indices = dists.argmin(axis=1)  # (N,)
    return codebook[indices]


def medoid_pool(emb, codebook):
    """Pool subject embeddings via medoid: codebook entry with min mean distance.
    Args: emb (N, D) quantized embeddings, codebook (M, D)
    Returns: (D,) codebook entry
    """
    dists = cdist(codebook, emb)  # (M, N)
    return codebook[dists.mean(axis=1).argmin()]


# ── Binning transforms for subject-wise tasks ────────────────────────

BINS = {
    "age_group": ([0, 40, 60, 80, 120], ["<40", "40-60", "60-80", ">80"]),
    "ahi_severity": ([0, 5, 15, 30, 9999], ["Normal", "Mild", "Moderate", "Severe"]),
    "bmi_group": ([0, 18.5, 25, 30, 100], ["Under", "Normal", "Over", "Obese"]),
}


# ── Data loading ─────────────────────────────────────────────────────

def load_subjects(emb_dirs):
    """Load all subjects from one or more embedding directories.

    Returns dict: subject_id -> {emb_path, labels_path, metadata_path,
                                  event_files: {type: path}, dir: str}
    """
    subjects = {}
    for emb_dir in emb_dirs:
        for emb_path in sorted(glob.glob(os.path.join(emb_dir, "*_embeddings.npy"))):
            sid = os.path.basename(emb_path).replace("_embeddings.npy", "")
            if sid in subjects:
                continue  # first occurrence wins
            entry = {
                "emb_path": emb_path,
                "labels_path": os.path.join(emb_dir, f"{sid}_labels.npy"),
                "metadata_path": os.path.join(emb_dir, f"{sid}_metadata.json"),
                "events_summary_path": os.path.join(emb_dir, f"{sid}_events_summary.json"),
                "event_files": {},
                "dir": emb_dir,
            }
            for et in ["arousal", "respiratory", "desaturation", "limb_movement"]:
                p = os.path.join(emb_dir, f"{sid}_{et}.npy")
                if os.path.exists(p):
                    entry["event_files"][et] = p
            subjects[sid] = entry
    return subjects


def discover_tasks(subjects):
    """Auto-discover available tasks from the data.

    Checks multiple subjects to avoid missing tasks when the first subject
    has null values for some fields (e.g. CVD outcomes).
    """
    tasks = []

    # Check first subject for event files
    sample = next(iter(subjects.values()))

    # Event-wise: staging (always available)
    if os.path.exists(sample["labels_path"]):
        tasks.append(("event_wise", "staging", {"label_file": "_labels.npy", "n_classes": 5}))

    # Event-wise: events — check if ANY subject has the event file
    for et in ["arousal", "respiratory", "desaturation", "limb_movement"]:
        if any(et in info["event_files"] for info in subjects.values()):
            tasks.append(("event_wise", et, {"label_file": f"_{et}.npy", "n_classes": 2}))

    # Subject-wise: scan ALL subjects to discover metadata fields with non-null values
    # (CVD outcomes are rare ~2-5%, need full scan to find them)
    all_meta_keys = set()
    for info in subjects.values():
        if os.path.exists(info["metadata_path"]):
            with open(info["metadata_path"]) as f:
                meta = json.load(f)
            for k, v in meta.items():
                if v is not None:
                    all_meta_keys.add(k)

    if all_meta_keys:
        # Known subject-wise tasks
        field_tasks = [
            ("nsrr_sex", "sex", "classification"),
            ("sex", "sex", "classification"),
            ("nsrr_age", "age_regression", "regression"),
            ("age", "age_regression", "regression"),
            ("age_msl_testing", "age_regression", "regression"),
            ("nsrr_age", "age_group", "classification"),
            ("age", "age_group", "classification"),
            ("age_msl_testing", "age_group", "classification"),
            ("nsrr_bmi", "bmi_regression", "regression"),
            ("bmi", "bmi_regression", "regression"),
            ("nsrr_bmi", "bmi_group", "classification"),
            ("bmi", "bmi_group", "classification"),
            ("nsrr_ahi_hp3u", "ahi_severity", "classification"),
            ("ahi", "ahi_severity", "classification"),
            ("nsrr_ahi_hp3u", "ahi_regression", "regression"),
            ("ahi", "ahi_regression", "regression"),
            ("group", "diagnosis", "classification"),
            ("prev_mi", "prev_mi", "classification"),
            ("prev_stk", "prev_stk", "classification"),
            ("any_cvd", "any_cvd", "classification"),
            ("any_chd", "any_chd", "classification"),
            ("afibprevalent", "afib", "classification"),
            ("chf", "chf", "classification"),
            ("stroke", "stroke", "classification"),
        ]
        seen = set()
        for field, name, task_type in field_tasks:
            if field in all_meta_keys and name not in seen:
                seen.add(name)
                tasks.append(("subject_wise", name, {
                    "metadata_field": field,
                    "task_type": task_type,
                }))

    return tasks


def load_event_wise_data(subjects, label_key, codebook=None):
    """Load epoch embeddings + labels for event-wise probing.

    Returns: embeddings (N, D) float32, labels (N,) int64,
             groups (N,) int, subject_ids_per_epoch (N,) object
    """
    all_emb, all_lbl, all_grp, all_sids = [], [], [], []
    skipped = 0

    # Build group mapping upfront
    unique_group_keys = sorted(set(get_group_key(sid) for sid in subjects))
    grp_key_to_int = {g: i for i, g in enumerate(unique_group_keys)}

    for sid, info in subjects.items():
        if label_key == "_labels.npy":
            lbl_path = info["labels_path"]
        else:
            et = label_key[1:].replace(".npy", "")
            lbl_path = info["event_files"].get(et)
            if lbl_path is None:
                skipped += 1
                continue

        if not os.path.exists(lbl_path):
            skipped += 1
            continue

        emb = np.load(info["emb_path"]).astype(np.float32)
        if codebook is not None:
            emb = quantize_embeddings(emb, codebook)
        lbl = np.load(lbl_path)

        # Align lengths
        n = min(len(emb), len(lbl))
        emb, lbl = emb[:n], lbl[:n]

        # Filter invalid labels (staging: -1 = unscored)
        valid = lbl >= 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            skipped += 1
            continue

        all_emb.append(emb[valid])
        all_lbl.append(lbl[valid])
        grp_int = grp_key_to_int[get_group_key(sid)]
        all_grp.append(np.full(n_valid, grp_int, dtype=np.int32))
        all_sids.append(np.full(n_valid, sid, dtype=object))

    if skipped:
        print(f"  Skipped {skipped} subjects (missing labels)")

    return (
        np.concatenate(all_emb, axis=0),
        np.concatenate(all_lbl, axis=0).astype(np.int64),
        np.concatenate(all_grp, axis=0),
        np.concatenate(all_sids, axis=0),
        unique_group_keys,
    )


def load_subject_wise_data(subjects, metadata_field, transform=None, codebook=None):
    """Load mean-pooled embeddings + metadata labels for subject-wise probing.

    Returns: embeddings (N_subj, D), labels (N_subj,), subject_ids list, groups list
    """
    embs, labels, sids, groups = [], [], [], []

    for sid, info in subjects.items():
        if not os.path.exists(info["metadata_path"]):
            continue
        with open(info["metadata_path"]) as f:
            meta = json.load(f)

        val = meta.get(metadata_field)
        if val is None:
            continue

        # Apply transform
        if transform == "bin":
            bins, bin_names = BINS.get(metadata_field, (None, None))
            if bins is None:
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            label = np.digitize(val, bins[1:])  # bin index
            label = min(label, len(bin_names) - 1)
        elif transform == "sex":
            val_str = str(val).lower()
            if val_str in ("male", "m", "1", "1.0"):
                label = 0
            elif val_str in ("female", "f", "2", "2.0"):
                label = 1
            else:
                continue
        elif transform == "classification":
            # Generic: try to convert to int
            try:
                label = int(float(val))
            except (ValueError, TypeError):
                # String label -> use as-is, will be encoded later
                label = str(val)
        elif transform == "regression":
            try:
                label = float(val)
            except (ValueError, TypeError):
                continue
        else:
            label = val

        emb = np.load(info["emb_path"]).astype(np.float32)
        if codebook is not None:
            emb = quantize_embeddings(emb, codebook)
            pooled_emb = medoid_pool(emb, codebook)
        else:
            pooled_emb = emb.mean(axis=0)

        embs.append(pooled_emb)
        labels.append(label)
        sids.append(sid)
        groups.append(get_group_key(sid))

    return np.array(embs), labels, sids, groups


# ── Probing ──────────────────────────────────────────────────────────

def probe_event_wise(subjects, task_name, task_info, output_dir, n_folds=5, max_iter=1000, codebook=None):
    """Run event-wise linear probing."""
    print(f"\n{'='*60}")
    print(f"Event-wise probing: {task_name}")
    print(f"{'='*60}")

    label_key = task_info["label_file"]
    n_classes = task_info["n_classes"]

    X, y, group_ids, epoch_to_subject, unique_group_keys = load_event_wise_data(
        subjects, label_key, codebook=codebook
    )
    actual_classes = len(np.unique(y))
    print(f"  Data: {X.shape[0]} epochs, {len(unique_group_keys)} groups, "
          f"{actual_classes} classes, class dist: {np.bincount(y, minlength=n_classes).tolist()}")
    print(f"  Memory: X={X.nbytes / 1e9:.1f}GB ({X.dtype})")

    if actual_classes < 2:
        print("  SKIP: only 1 class in data")
        return None
    if len(unique_group_keys) < n_folds:
        n_folds = len(unique_group_keys)
        print(f"  Reducing n_folds to {n_folds} (not enough groups)")
    if n_folds < 2:
        print("  SKIP: not enough groups for CV")
        return None

    os.makedirs(output_dir, exist_ok=True)

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_assignments = {}
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, group_ids)):
        print(f"\n  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check train has >= 2 classes
        if len(np.unique(y_train)) < 2:
            print(f"    SKIP fold {fold_idx}: only 1 class in training data")
            continue

        # Fold assignments (subject-level)
        train_subs = sorted(set(epoch_to_subject[train_idx]))
        test_subs = sorted(set(epoch_to_subject[test_idx]))
        fold_assignments[f"fold_{fold_idx}"] = {
            "train": train_subs,
            "test": test_subs,
        }

        # saga for large datasets (fast, converges with enough data), lbfgs for small
        solver = "saga" if len(X_train) > 500_000 else "lbfgs"
        clf = LogisticRegression(max_iter=max_iter, C=1.0, solver=solver, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # Per-subject predictions
        test_subject_ids = epoch_to_subject[test_idx]
        predictions = {}
        for sid in sorted(set(test_subject_ids)):
            mask = test_subject_ids == sid
            predictions[sid] = {
                "y_true": y_test[mask].tolist(),
                "y_pred": y_pred[mask].tolist(),
                "y_proba": y_proba[mask].tolist(),
            }

        # Free test arrays
        del X_train, X_test

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}
        fold_metrics.append(metrics)
        print(f"    acc={acc:.4f}  f1={f1:.4f}  kappa={kappa:.4f}")

        # Save
        joblib.dump(clf, os.path.join(fold_dir, "classifier.joblib"))
        with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Save fold assignments
    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    # Summary
    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = n_folds
    summary["n_epochs"] = int(X.shape[0])
    summary["n_groups"] = int(len(unique_group_keys))
    summary["task"] = task_name
    summary["type"] = "event_wise"

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    return summary


def probe_subject_wise(subjects, task_name, task_info, output_dir, n_folds=5, max_iter=1000, codebook=None):
    """Run subject-wise linear probing."""
    print(f"\n{'='*60}")
    print(f"Subject-wise probing: {task_name}")
    print(f"{'='*60}")

    metadata_field = task_info["metadata_field"]
    task_type = task_info["task_type"]

    # Determine transform
    if task_name in BINS:
        transform = "bin"
    elif task_name in ("sex",):
        transform = "sex"
    elif task_type == "regression":
        transform = "regression"
    else:
        transform = "classification"

    X, labels, sids, groups = load_subject_wise_data(
        subjects, metadata_field, transform=transform, codebook=codebook
    )

    if len(X) == 0:
        print(f"  No subjects with valid {metadata_field} — skipping")
        return None

    # Encode string labels
    is_regression = task_type == "regression"
    if is_regression:
        y = np.array(labels, dtype=np.float64)
    else:
        # Map to int labels
        unique_labels = sorted(set(labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_int[l] for l in labels], dtype=np.int64)
        n_classes = len(unique_labels)

    sids = np.array(sids)
    groups_arr = np.array(groups)
    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])

    print(f"  Data: {len(X)} subjects, {len(unique_groups)} groups")
    if is_regression:
        print(f"  Target: {metadata_field}, mean={y.mean():.2f}, std={y.std():.2f}")
    else:
        print(f"  Classes: {unique_labels}, dist: {np.bincount(y, minlength=n_classes).tolist()}")
        if n_classes < 2:
            print("  SKIP: only 1 class in data")
            return None

    if len(unique_groups) < n_folds:
        n_folds = len(unique_groups)
        print(f"  Reducing n_folds to {n_folds} (not enough groups)")
    if n_folds < 2:
        print("  SKIP: not enough groups for CV")
        return None

    os.makedirs(output_dir, exist_ok=True)

    if is_regression:
        cv = GroupKFold(n_splits=n_folds)
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_assignments = {}
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, group_ids)):
        print(f"\n  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": sids[train_idx].tolist(),
            "test": sids[test_idx].tolist(),
        }

        if is_regression:
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            predictions = {}
            for i, idx in enumerate(test_idx):
                predictions[sids[idx]] = {
                    "y_true": float(y_test[i]),
                    "y_pred": float(y_pred[i]),
                }

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = {"mae": mae, "r2": r2}
            print(f"    MAE={mae:.4f}  R2={r2:.4f}")
        else:
            clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            predictions = {}
            for i, idx in enumerate(test_idx):
                predictions[sids[idx]] = {
                    "y_true": int(y_test[i]),
                    "y_pred": int(y_pred[i]),
                    "y_proba": y_proba[i].tolist(),
                }

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_test, y_pred)
            metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}
            print(f"    acc={acc:.4f}  f1={f1:.4f}  kappa={kappa:.4f}")

        fold_metrics.append(metrics)

        joblib.dump(clf, os.path.join(fold_dir, "classifier.joblib"))
        with open(os.path.join(fold_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Save fold assignments
    with open(os.path.join(output_dir, "fold_assignments.json"), "w") as f:
        json.dump(fold_assignments, f)

    # Summary
    summary = {}
    for key in fold_metrics[0]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = n_folds
    summary["n_subjects"] = int(len(X))
    summary["n_groups"] = int(len(unique_groups))
    summary["task"] = task_name
    summary["type"] = "subject_wise"
    if not is_regression:
        summary["classes"] = [str(l) for l in unique_labels]

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if is_regression:
        print(f"\n  Summary: MAE={summary['mae']['mean']:.4f}±{summary['mae']['std']:.4f}, "
              f"R2={summary['r2']['mean']:.4f}±{summary['r2']['std']:.4f}")
    else:
        print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
              f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    return summary


# ── Source discrimination ─────────────────────────────────────────────

def _dir_label(emb_dir):
    """Extract source label from directory path.

    mass_cohort1/all → mass_cohort1
    train            → train
    """
    base = os.path.basename(emb_dir)
    if base == "all":
        return os.path.basename(os.path.dirname(emb_dir))
    return base


def _source_group_key(sid, subjects):
    """Group key for source discrimination, using metadata base_subject_id if available."""
    info = subjects[sid]
    # Parkinsons: same person has night + nap recordings
    if os.path.exists(info["metadata_path"]):
        with open(info["metadata_path"]) as f:
            meta = json.load(f)
        if "base_subject_id" in meta and meta["base_subject_id"]:
            return str(meta["base_subject_id"])
    return get_group_key(sid)


def probe_source_discrimination(subjects, task_name, output_dir, n_folds=5, max_iter=1000, codebook=None,
                                dir_label_map=None):
    """Classify subjects by their source directory (cohort/visit/site)."""
    print(f"\n{'='*60}")
    print(f"Source discrimination: {task_name}")
    print(f"{'='*60}")

    embs, labels, sids, groups = [], [], [], []
    for sid, info in subjects.items():
        if dir_label_map:
            label = dir_label_map.get(os.path.abspath(info["dir"]))
            if label is None:
                label = _dir_label(info["dir"])
        else:
            label = _dir_label(info["dir"])
        emb = np.load(info["emb_path"]).astype(np.float32)
        if codebook is not None:
            emb = quantize_embeddings(emb, codebook)
            pooled_emb = medoid_pool(emb, codebook)
        else:
            pooled_emb = emb.mean(axis=0)

        embs.append(pooled_emb)
        labels.append(label)
        sids.append(sid)
        groups.append(_source_group_key(sid, subjects))

    X = np.array(embs)
    sids = np.array(sids)
    groups_arr = np.array(groups)

    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels], dtype=np.int64)
    n_classes = len(unique_labels)

    unique_groups = np.unique(groups_arr)
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([group_to_int[g] for g in groups_arr])

    print(f"  Data: {len(X)} subjects, {len(unique_groups)} groups, "
          f"{n_classes} classes: {unique_labels}")
    print(f"  Class dist: {np.bincount(y, minlength=n_classes).tolist()}")

    if n_classes < 2:
        print("  SKIP: only 1 source")
        return None
    if len(unique_groups) < n_folds:
        n_folds = len(unique_groups)
        print(f"  Reducing n_folds to {n_folds}")
    if n_folds < 2:
        print("  SKIP: not enough groups")
        return None

    os.makedirs(output_dir, exist_ok=True)

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_assignments = {}
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, group_ids)):
        print(f"\n  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            print(f"    SKIP fold {fold_idx}: single class in train")
            continue

        fold_assignments[f"fold_{fold_idx}"] = {
            "train": sids[train_idx].tolist(),
            "test": sids[test_idx].tolist(),
        }

        clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        predictions = {}
        for i, idx in enumerate(test_idx):
            predictions[sids[idx]] = {
                "y_true": int(y_test[i]),
                "y_pred": int(y_pred[i]),
                "y_proba": y_proba[i].tolist(),
            }

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        metrics = {"accuracy": acc, "f1_macro": f1, "kappa": kappa}
        fold_metrics.append(metrics)
        print(f"    acc={acc:.4f}  f1={f1:.4f}  kappa={kappa:.4f}")

        joblib.dump(clf, os.path.join(fold_dir, "classifier.joblib"))
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
    summary["n_subjects"] = int(len(X))
    summary["n_groups"] = int(len(unique_groups))
    summary["task"] = task_name
    summary["type"] = "source_discrimination"
    summary["classes"] = unique_labels

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f}, "
          f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
    return summary


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Linear probing on epoch embeddings")
    parser.add_argument("--emb_dirs", nargs="+", required=True,
                        help="Embedding directories (can merge train/valid/test)")
    parser.add_argument("--source_name", type=str, required=True,
                        help="Name for this source (e.g. shhs_visit1, mesa, mass_cohort1)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (staging, arousal, sex, age_regression, ...). "
                             "Use --discover to list available tasks.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for LogisticRegression (default: 1000)")
    parser.add_argument("--discover", action="store_true",
                        help="List available tasks and exit")
    # Override for subject-wise tasks
    parser.add_argument("--metadata_field", type=str, default=None,
                        help="Override metadata field for subject-wise task")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["classification", "regression"],
                        help="Override task type")
    # Source discrimination
    parser.add_argument("--label_from_dir", action="store_true",
                        help="Label subjects by their source directory for "
                             "cohort/visit/site discrimination probing")
    parser.add_argument("--dir_labels", nargs="+", default=None,
                        help="Explicit label per --emb_dirs entry (same order). "
                             "E.g. --emb_dirs train valid test visit2/all "
                             "--dir_labels visit1 visit1 visit1 visit2")
    parser.add_argument("--quantize", type=str, default=None,
                        help="Path to codebook.npy (M, D) for vector quantization")
    args = parser.parse_args()

    codebook = None
    if args.quantize:
        codebook = np.load(args.quantize).astype(np.float32)
        print(f"Quantizing with codebook: {codebook.shape}")

    subjects = load_subjects(args.emb_dirs)
    print(f"Loaded {len(subjects)} subjects from {len(args.emb_dirs)} dir(s)")

    if not subjects:
        print("No subjects found.")
        return

    # Source discrimination mode
    if args.label_from_dir:
        # Build dir -> label mapping
        dir_label_map = None
        if args.dir_labels:
            if len(args.dir_labels) != len(args.emb_dirs):
                print(f"ERROR: --dir_labels ({len(args.dir_labels)}) must match "
                      f"--emb_dirs ({len(args.emb_dirs)})")
                return
            dir_label_map = {
                os.path.abspath(d): l
                for d, l in zip(args.emb_dirs, args.dir_labels)
            }
        task_name = args.task or "source"
        task_output_dir = os.path.join(args.output_dir, args.source_name, task_name)
        probe_source_discrimination(
            subjects, task_name, task_output_dir, args.n_folds,
            max_iter=args.max_iter, codebook=codebook, dir_label_map=dir_label_map,
        )
        print(f"\nResults: {task_output_dir}/")
        return

    available = discover_tasks(subjects)

    if args.discover:
        print(f"\nAvailable tasks ({len(available)}):")
        for task_type, name, info in available:
            print(f"  [{task_type}] {name}  ({info})")
        return

    if args.task is None:
        parser.error("--task is required (use --discover to list available tasks)")

    # Find the requested task
    task_match = None
    for task_type, name, info in available:
        if name == args.task:
            task_match = (task_type, name, info)
            break

    # Allow custom subject-wise task via --metadata_field
    if task_match is None and args.metadata_field:
        task_type = args.task_type or "classification"
        task_match = ("subject_wise", args.task, {
            "metadata_field": args.metadata_field,
            "task_type": task_type,
        })
    elif task_match is None:
        print(f"Task '{args.task}' not found. Available tasks:")
        for t, n, _ in available:
            print(f"  [{t}] {n}")
        return

    task_type, task_name, task_info = task_match
    task_output_dir = os.path.join(args.output_dir, args.source_name, task_name)

    if task_type == "event_wise":
        probe_event_wise(subjects, task_name, task_info, task_output_dir, args.n_folds, args.max_iter, codebook)
    else:
        probe_subject_wise(subjects, task_name, task_info, task_output_dir, args.n_folds, args.max_iter, codebook)

    print(f"\nResults: {task_output_dir}/")


if __name__ == "__main__":
    main()
