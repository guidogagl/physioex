"""How well does a seq-to-seq model group similar signals?  KMeans ARI/accuracy on embeddings.

We take a model's contextualized per-epoch embeddings (extracted with ``extract_embeddings`` =
sliding-window *voting average*, see ``physioex/models/embed.py``), run KMeans for several ``k``,
and score the clustering against expert stage labels.

Metrics per k:
  * ARI  (adjusted_rand_score)          — chance-corrected pair agreement (0=random, 1=perfect).
  * AMI  (adjusted_mutual_info_score)   — chance-corrected information overlap.
  * acc_hungarian / kappa_hungarian     — optimal 1-1 cluster->stage matching (bijection);
                                          only at k == n_stages. THIS is the number on the same
                                          scale as model accuracy / Cohen kappa.
  * acc_majority / kappa_majority       — each cluster -> its majority stage (defined for any k;
                                          note: monotonically inflated with k, i.e. ~= purity).
  * silhouette                          — internal cohesion (subsampled when pooled).

Two modes:
  * per_subject : one KMeans per subject (default granularity of a per-recording "vision" operator).
  * pooled      : ONE KMeans over all epochs of all subjects concatenated, global z-score.

Usage:
    python examples/pretrained/_analysis/cluster_ari.py \
        --models sleeptransformer-phan seqsleepnet-phan --dataset_name mass_ss02 \
        --mode pooled --out_dir ./ari_out

    python examples/pretrained/_analysis/cluster_ari.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

K_VALUES_DEFAULT = [2, 3, 4, 5, 6, 7, 8]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
RANDOM_STATE = 42
N_STAGES = 5          # highlighted k / number of AASM stages
SIL_SUBSAMPLE = 6000  # cap silhouette cost on large pooled sets


# --------------------------------------------------------------------------- #
# cluster -> stage mapping + metrics
# --------------------------------------------------------------------------- #
def _map_clusters(clusters: np.ndarray, labels: np.ndarray, n_classes: int, method: str) -> np.ndarray:
    """Map each cluster id to a stage; return predicted-stage array aligned with clusters."""
    k = int(clusters.max()) + 1
    # overlap matrix (clusters x classes)
    C = np.zeros((k, n_classes), dtype=np.int64)
    for c in range(k):
        m = clusters == c
        if m.any():
            C[c] = np.bincount(labels[m], minlength=n_classes)
    if method == "hungarian":
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-C)              # maximize overlap (rectangular OK)
        mapping = {int(r): int(cc) for r, cc in zip(rows, cols)}
        pred = np.array([mapping.get(int(c), int(C[c].argmax())) for c in clusters])
    else:  # majority vote per cluster
        maj = C.argmax(axis=1)
        pred = maj[clusters]
    return pred


def clustering_metrics(labels: np.ndarray, clusters: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import (
        accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
        cohen_kappa_score, f1_score,
    )
    out = {
        "ari": float(adjusted_rand_score(labels, clusters)),
        "ami": float(adjusted_mutual_info_score(labels, clusters)),
    }
    pred_maj = _map_clusters(clusters, labels, n_classes, "majority")
    out["acc_majority"] = float(accuracy_score(labels, pred_maj))
    out["kappa_majority"] = float(cohen_kappa_score(labels, pred_maj))
    out["macro_f1_majority"] = float(f1_score(labels, pred_maj, average="macro"))
    # bijective optimal matching only meaningful when #clusters == #stages
    if int(clusters.max()) + 1 == n_classes:
        pred_h = _map_clusters(clusters, labels, n_classes, "hungarian")
        out["acc_hungarian"] = float(accuracy_score(labels, pred_h))
        out["kappa_hungarian"] = float(cohen_kappa_score(labels, pred_h))
    else:
        out["acc_hungarian"] = None
        out["kappa_hungarian"] = None
    return out


def _silhouette(X: np.ndarray, clusters: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    if len(np.unique(clusters)) < 2:
        return float("nan")
    if X.shape[0] > SIL_SUBSAMPLE:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(X.shape[0], SIL_SUBSAMPLE, replace=False)
        X, clusters = X[idx], clusters[idx]
        if len(np.unique(clusters)) < 2:
            return float("nan")
    try:
        return float(silhouette_score(X, clusters))
    except Exception:
        return float("nan")


def _kmeans(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)


# --------------------------------------------------------------------------- #
# per-subject clustering (also used by the self-test)
# --------------------------------------------------------------------------- #
def cluster_subject(X: np.ndarray, labels: np.ndarray, k_values, n_classes=N_STAGES) -> dict:
    """z-score this subject, KMeans per k, full metrics per k."""
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    res = {"n_epochs": int(X.shape[0]), "n_stages_present": int(len(np.unique(labels)))}
    for k in k_values:
        if X.shape[0] < k:
            res[k] = {"ari": float("nan"), "acc_hungarian": None}
            continue
        cl = _kmeans(Xz, k)
        m = clustering_metrics(labels, cl, n_classes)
        m["silhouette"] = _silhouette(Xz, cl)
        res[k] = m
    return res


# --------------------------------------------------------------------------- #
# data loading (mirrors linear_probe cache layout)
# --------------------------------------------------------------------------- #
def _load_subjects(emb_dir: Path):
    from physioex.models.embed import _load_npy_as_float32
    for subj_dir in sorted(emb_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        emb_path, lbl_path = subj_dir / "embeddings.npy", subj_dir / "labels.npy"
        if emb_path.exists() and lbl_path.exists():
            yield (subj_dir.name,
                   _load_npy_as_float32(emb_path).astype(np.float32),
                   np.load(str(lbl_path)).astype(np.int64))


def _linear_probe_kappa(emb_dir: Path):
    p = emb_dir / "linear_probe_results.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["pooled"]["kappa"])
    except Exception:
        return None


def _load_pooled(emb_dir: Path):
    """Concatenate all scored epochs from all subjects. Returns (X, y, n_subjects)."""
    Xs, ys, n = [], [], 0
    for _sid, emb, lbl in _load_subjects(emb_dir):
        mask = lbl >= 0
        if mask.sum() == 0:
            continue
        Xs.append(emb[mask]); ys.append(lbl[mask]); n += 1
    return np.concatenate(Xs), np.concatenate(ys), n


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def run_pooled(model_name, dataset_name, k_values, cache_dir=None):
    """ONE global KMeans over all epochs of all subjects (global z-score)."""
    from physioex.models import load_embeddings
    emb_dir = Path(load_embeddings(model_name, dataset_name, cache_dir=cache_dir))
    X, y, n_subj = _load_pooled(emb_dir)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)      # GLOBAL z-score (no per-subject norm)
    print(f"  pooled: {n_subj} subjects, {X.shape[0]} epochs, D={X.shape[1]}")

    rows = []
    for k in k_values:
        cl = _kmeans(Xz, k)
        m = clustering_metrics(y, cl, N_STAGES)
        m["silhouette"] = _silhouette(Xz, cl)
        m.update(model=model_name, k=k)
        rows.append(m)
        print(f"    k={k}: ARI={m['ari']:.3f}  AMI={m['ami']:.3f}  "
              f"acc_maj={m['acc_majority']:.3f}  "
              f"acc_hung={m['acc_hungarian'] if m['acc_hungarian'] is None else round(m['acc_hungarian'],3)}  "
              f"κ_hung={m['kappa_hungarian'] if m['kappa_hungarian'] is None else round(m['kappa_hungarian'],3)}")
    summary = {
        "model_name": model_name, "dataset_name": dataset_name, "mode": "pooled",
        "n_subjects": n_subj, "n_epochs": int(X.shape[0]), "k_values": list(k_values),
        "linear_probe_kappa": _linear_probe_kappa(emb_dir),
        "at_k5": next((r for r in rows if r["k"] == N_STAGES), None),
        "best_ari": max(rows, key=lambda r: r["ari"]),
    }
    return rows, summary


def run_per_subject(model_name, dataset_name, k_values, cache_dir=None):
    from physioex.models import load_embeddings
    emb_dir = Path(load_embeddings(model_name, dataset_name, cache_dir=cache_dir))
    rows, per_k = [], {k: {"ari": [], "acc_h": [], "kappa_h": []} for k in k_values}
    n = 0
    for sid, emb, lbl in _load_subjects(emb_dir):
        mask = lbl >= 0
        X, yv = emb[mask], lbl[mask]
        if X.shape[0] < max(k_values) or len(np.unique(yv)) < 2:
            print(f"  [skip] {sid}"); continue
        res = cluster_subject(X, yv, k_values)
        n += 1
        for k in k_values:
            m = res[k]
            rows.append(dict(model=model_name, subject=sid, k=k, n_epochs=res["n_epochs"], **m))
            if np.isfinite(m.get("ari", float("nan"))):
                per_k[k]["ari"].append(m["ari"])
                if m.get("acc_hungarian") is not None:
                    per_k[k]["acc_h"].append(m["acc_hungarian"])
                    per_k[k]["kappa_h"].append(m["kappa_hungarian"])
        r5 = res[N_STAGES]
        print(f"  {sid}: ARI@5={r5['ari']:.3f}  acc_hung@5="
              f"{None if r5['acc_hungarian'] is None else round(r5['acc_hungarian'],3)}")
    summary = {
        "model_name": model_name, "dataset_name": dataset_name, "mode": "per_subject",
        "n_subjects": n, "k_values": list(k_values),
        "linear_probe_kappa": _linear_probe_kappa(emb_dir), "per_k": {},
    }
    for k in k_values:
        a, ah, kh = (np.array(per_k[k][x], float) for x in ("ari", "acc_h", "kappa_h"))
        summary["per_k"][k] = {
            "ari_mean": float(np.nanmean(a)) if a.size else None,
            "ari_std": float(np.nanstd(a)) if a.size else None,
            "acc_hungarian_mean": float(np.nanmean(ah)) if ah.size else None,
            "kappa_hungarian_mean": float(np.nanmean(kh)) if kh.size else None,
        }
    return rows, summary


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _write_csv(rows, path: Path):
    if not rows:
        return
    keys = list({k for r in rows for k in r.keys()})
    order = ["model", "subject", "k", "n_epochs", "ari", "ami", "acc_hungarian",
             "kappa_hungarian", "acc_majority", "kappa_majority", "macro_f1_majority",
             "silhouette"]
    fields = [k for k in order if k in keys] + [k for k in keys if k not in order]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _plot_pooled(rows_by_model, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, (model, rows) in enumerate(rows_by_model.items()):
        ks = [r["k"] for r in rows]
        ax1.plot(ks, [r["ari"] for r in rows], "-o", color=f"C{i}", label=f"{model} ARI")
        ax1.plot(ks, [r["ami"] for r in rows], "--s", color=f"C{i}", alpha=0.6, label=f"{model} AMI")
        accs = [r["acc_hungarian"] if r["acc_hungarian"] is not None else np.nan for r in rows]
        ax2.plot(ks, accs, "-o", color=f"C{i}", label=f"{model} acc(Hungarian)")
        ax2.plot(ks, [r["acc_majority"] for r in rows], ":^", color=f"C{i}", alpha=0.6,
                 label=f"{model} acc(majority)")
    for ax in (ax1, ax2):
        ax.axvline(N_STAGES, color="0.5", ls="--", lw=1); ax.set_xlabel("k"); ax.legend(fontsize=7)
    ax1.set_ylabel("ARI / AMI (chance-corrected)"); ax1.set_title("Pooled — agreement chance-corrected")
    ax2.set_ylabel("accuracy scale [0,1]"); ax2.set_title("Pooled — clustering accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _pooled_table(summaries) -> str:
    lines = ["| model | ARI@5 | AMI@5 | acc(Hung)@5 | κ(Hung)@5 | acc(maj)@5 | best ARI | linear-probe κ |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        a5 = s["at_k5"]; b = s["best_ari"]
        def f(x): return "n/a" if x is None else f"{x:.3f}"
        lines.append(
            f"| {s['model_name']} | {f(a5['ari'])} | {f(a5['ami'])} | {f(a5['acc_hungarian'])} | "
            f"{f(a5['kappa_hungarian'])} | {f(a5['acc_majority'])} | {f(b['ari'])} @k={b['k']} | "
            f"{f(s['linear_probe_kappa'])} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def selftest() -> int:
    rng = np.random.RandomState(0)
    D, per = 128, 200
    centers = rng.randn(N_STAGES, D) * 6.0
    X = np.concatenate([centers[c] + rng.randn(per, D) for c in range(N_STAGES)])
    y = np.concatenate([np.full(per, c) for c in range(N_STAGES)])
    res = cluster_subject(X, y, K_VALUES_DEFAULT)
    r5 = res[N_STAGES]
    print(f"[selftest] 5 separable gaussians -> ARI@5={r5['ari']:.4f} "
          f"acc_hungarian@5={r5['acc_hungarian']:.4f} (expect ~1.0)")
    import tempfile
    rows = [dict(model="t", subject="s", k=k, **res[k]) for k in K_VALUES_DEFAULT]
    with tempfile.TemporaryDirectory() as td:
        _write_csv(rows, Path(td) / "t.csv"); print("[selftest] CSV write OK")
    ok = r5["ari"] > 0.95 and r5["acc_hungarian"] > 0.95
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["sleeptransformer-phan", "seqsleepnet-phan"])
    ap.add_argument("--dataset_name", default="mass_ss02")
    ap.add_argument("--mode", choices=["pooled", "per_subject", "both"], default="both")
    ap.add_argument("--out_dir", default="./ari_out")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--k_values", nargs="+", type=int, default=K_VALUES_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("pooled", "both"):
        summaries, rows_by_model = [], {}
        for model_name in args.models:
            print(f"\n=== POOLED {model_name} / {args.dataset_name} ===")
            rows, summary = run_pooled(model_name, args.dataset_name, args.k_values, args.cache_dir)
            _write_csv(rows, out / f"pooled_{model_name}.csv")
            (out / f"pooled_summary_{model_name}.json").write_text(json.dumps(summary, indent=2))
            summaries.append(summary); rows_by_model[model_name] = rows
        _plot_pooled(rows_by_model, out / "pooled_ari_acc_vs_k.png")
        table = _pooled_table(summaries)
        (out / "pooled_table.md").write_text(table + "\n")
        print("\n" + table)

    if args.mode in ("per_subject", "both"):
        for model_name in args.models:
            print(f"\n=== PER-SUBJECT {model_name} / {args.dataset_name} ===")
            rows, summary = run_per_subject(model_name, args.dataset_name, args.k_values, args.cache_dir)
            _write_csv(rows, out / f"per_subject_{model_name}.csv")
            (out / f"per_subject_summary_{model_name}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOutputs in {out.resolve()}")


if __name__ == "__main__":
    main()
