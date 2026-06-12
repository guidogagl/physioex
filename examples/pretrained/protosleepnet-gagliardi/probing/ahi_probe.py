"""AHI binary classification with LOSO cross-validation.

Classifies no apnea (AHI < 5) vs apnea (AHI >= 5) using codebook prototype
permanence features. Uses class_weight="balanced" for robustness to
class imbalance (29 vs 57).

Usage:
    python ahi_probe.py \
        --emb_dirs .../parkinsons_night_HOA/all .../parkinsons_night_PD/all \
        --codebook .../codebooks/vq_kmeans/M24.npy \
        --source_name parkinsons_night --output_dir /out
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


AHI_CUTOFF = 5


def load_subjects(emb_dirs, codebook, exclude_mild=False):
    """Load subjects and compute codebook features."""
    K = len(codebook)
    subjects = []

    for emb_dir in emb_dirs:
        for mp in sorted(glob.glob(os.path.join(emb_dir, "*_metadata.json"))):
            sid = os.path.basename(mp).replace("_metadata.json", "")
            with open(mp) as f:
                meta = json.load(f)
            ahi = meta.get("ahi")
            if ahi is None:
                continue

            # Skip mild apnea (5 <= AHI < 15) if requested
            if exclude_mild and 5 <= ahi < 15:
                continue

            emb_path = os.path.join(emb_dir, f"{sid}_embeddings.npy")
            if not os.path.exists(emb_path):
                continue

            emb = np.load(emb_path).astype(np.float32)
            N = len(emb)
            if N == 0:
                continue

            assignments = cdist(emb, codebook).argmin(axis=1)

            proportion = np.zeros(K, dtype=np.float32)
            for k in range(K):
                proportion[k] = (assignments == k).sum() / N

            bout_mean = np.zeros(K, dtype=np.float32)
            bout_std = np.zeros(K, dtype=np.float32)
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
                    bout_mean[k] = np.mean(bouts[k])
                    if len(bouts[k]) > 1:
                        bout_std[k] = np.std(bouts[k])

            features = np.concatenate([proportion, bout_mean, bout_std])
            label = 1 if ahi >= AHI_CUTOFF else 0

            subjects.append({
                "sid": sid,
                "group": meta.get("group", "?"),
                "ahi": float(ahi),
                "label": label,
                "features": features,
            })

    return subjects


def run_loso_fold(args):
    """Run a single LOSO fold."""
    i, X, y, sids, C, use_scaler, penalty = args
    mask = np.ones(len(X), dtype=bool)
    mask[i] = False

    X_train, y_train = X[mask].copy(), y[mask]
    X_test = X[i:i+1].copy()

    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if penalty == "l1":
        clf = LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=5000,
                                  class_weight="balanced")
    else:
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000,
                                  class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)[0]
    y_proba = clf.predict_proba(X_test)[0, 1]
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    return sids[i], int(y[i]), int(y_pred), float(y_proba), float(train_acc)


def main():
    parser = argparse.ArgumentParser(
        description="AHI binary classification (LOSO)"
    )
    parser.add_argument("--emb_dirs", nargs="+", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--scaler", action="store_true")
    parser.add_argument("--exclude_mild", action="store_true",
                        help="Exclude mild apnea (5<=AHI<15), keep only <5 vs >=15")
    parser.add_argument("--penalty", default="l2", choices=["l1", "l2"])
    parser.add_argument("--features", default="all",
                        choices=["all", "prop", "prop_bout", "rem_only"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    codebook = np.load(args.codebook).astype(np.float32)
    K = len(codebook)
    print(f"AHI probe: K={K}, C={args.C}, scaler={args.scaler}, "
          f"penalty={args.penalty}, features={args.features}")

    subjects = load_subjects(args.emb_dirs, codebook, args.exclude_mild)
    print(f"Loaded {len(subjects)} subjects")

    n_pos = sum(1 for s in subjects if s["label"] == 1)
    n_neg = sum(1 for s in subjects if s["label"] == 0)
    print(f"  No apnea (AHI<5): {n_neg}, Apnea (AHI>=5): {n_pos}")

    X_full = np.array([s["features"] for s in subjects], dtype=np.float32)

    if args.features == "prop":
        X = X_full[:, :K]
    elif args.features == "prop_bout":
        X = X_full[:, :2*K]
    elif args.features == "rem_only":
        rem_protos = []
        all_assign, all_labels = [], []
        for s in subjects:
            emb_dir = None
            for d in args.emb_dirs:
                if os.path.exists(os.path.join(d, s["sid"] + "_labels.npy")):
                    emb_dir = d
                    break
            if emb_dir:
                labels = np.load(os.path.join(emb_dir, s["sid"] + "_labels.npy")).astype(np.int64)
                emb = np.load(os.path.join(emb_dir, s["sid"] + "_embeddings.npy")).astype(np.float32)
                n = min(len(emb), len(labels))
                assign = cdist(emb[:n], codebook).argmin(axis=1)
                all_assign.append(assign)
                all_labels.append(labels[:n])
        all_assign = np.concatenate(all_assign)
        all_labels = np.concatenate(all_labels)
        for k in range(K):
            m = all_assign == k
            sl = all_labels[m]
            valid = sl >= 0
            if valid.sum() > 0 and (sl[valid] == 4).sum() / valid.sum() >= 0.30:
                rem_protos.append(k)
        print(f"  REM prototypes: {rem_protos}")
        cols = []
        for k in rem_protos:
            cols.extend([k, K + k, 2 * K + k])
        X = X_full[:, cols]
    else:
        X = X_full

    y = np.array([s["label"] for s in subjects])
    sids = [s["sid"] for s in subjects]
    N = len(subjects)
    print(f"  Feature dim: {X.shape[1]}")

    fold_args = [(i, X, y, sids, args.C, args.scaler, args.penalty) for i in range(N)]

    n_jobs = args.n_jobs if args.n_jobs > 0 else os.cpu_count()
    print(f"  Running {N} LOSO folds on {n_jobs} workers...")

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(run_loso_fold, fold_args))

    predictions = {}
    all_true, all_pred, all_proba, train_accs = [], [], [], []
    for sid, yt, yp, yprob, tacc in results:
        predictions[sid] = {"y_true": yt, "y_pred": yp, "y_proba": yprob}
        all_true.append(yt)
        all_pred.append(yp)
        all_proba.append(yprob)
        train_accs.append(tacc)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred).tolist()
    train_acc_mean = np.mean(train_accs)

    print(f"\n  Results:")
    print(f"    Train accuracy (mean): {train_acc_mean:.4f}")
    print(f"    Test accuracy:  {acc:.4f}")
    print(f"    Test MF1:       {f1:.4f}")
    print(f"    Test Kappa:     {kappa:.4f}")
    print(f"    Confusion matrix: {cm}")
    print(f"    Correct: {int(acc * N)}/{N}")

    output_dir = os.path.join(args.output_dir, args.source_name, "ahi")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    summary = {
        "test": {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "kappa": float(kappa),
            "confusion_matrix": cm,
            "n_correct": int(acc * N),
            "n_total": N,
        },
        "train": {
            "accuracy_mean": float(train_acc_mean),
        },
        "n_subjects": N,
        "n_no_apnea": int(n_neg),
        "n_apnea": int(n_pos),
        "cutoff": AHI_CUTOFF,
        "codebook": args.codebook,
        "K": K,
        "C": args.C,
        "scaler": args.scaler,
        "penalty": args.penalty,
        "features": args.features,
        "class_weight": "balanced",
        "exclude_mild": args.exclude_mild,
        "cv": "LOSO",
        "feature_dim": int(X.shape[1]),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results: {output_dir}/")


if __name__ == "__main__":
    main()
