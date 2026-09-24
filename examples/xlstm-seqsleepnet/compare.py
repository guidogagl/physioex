"""Paired, subject-level comparison of two (or more) runs of run.py.

Loads ``predictions_<mode>.pt`` from run directories that share the same fold /
split, computes per-subject kappa, accuracy, macro-F1, per-class F1 and the
error rate on stage-transition vs non-transition epochs, and reports paired
differences against the first run with subject-bootstrap confidence intervals.
Pooled metrics are printed for historical comparison only.

Usage:
    python compare.py --mode voting  runs/mass_seqsleepnet_L20_f0_s0  runs/mass_x_seqsleepnet_xlstm_bi_L20_f0_s0 ...
    python compare.py --mode voting  --glob 'runs/mass_*_L200_f*_s0'   # many folds: subjects are pooled across folds
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score

CLASSES = ["W", "N1", "N2", "N3", "REM"]


def load_run(run_dir: Path, mode: str):
    """-> dict subject_id -> (pred (n,), target (n,)) restricted to valid epochs."""
    p = torch.load(run_dir / f"predictions_{mode}.pt", map_location="cpu")
    out = {}
    for sid, logits, tgt in zip(p["subject_ids"], p["logits"], p["targets"]):
        pred = logits.argmax(-1).numpy()
        tgt = tgt.numpy()
        out[sid] = (pred, tgt)
    return out


def subject_metrics(pred, tgt, ignore=-1):
    m = tgt != ignore
    p, t = pred[m], tgt[m]
    res = {
        "acc": float((p == t).mean()) if len(t) else np.nan,
        "kappa": cohen_kappa_score(t, p) if len(np.unique(t)) > 1 else np.nan,
        "mf1": f1_score(t, p, average="macro", labels=range(5), zero_division=0),
    }
    for c, name in enumerate(CLASSES):
        res[f"f1_{name}"] = f1_score(t == c, p == c, zero_division=0) if (t == c).any() else np.nan
    # transition epochs: label differs from the previous or next valid epoch
    if len(t) > 2:
        trans = np.zeros_like(t, dtype=bool)
        trans[1:] |= t[1:] != t[:-1]
        trans[:-1] |= t[:-1] != t[1:]
        err = p != t
        res["err_transition"] = float(err[trans].mean()) if trans.any() else np.nan
        res["err_stable"] = float(err[~trans].mean()) if (~trans).any() else np.nan
        res["frac_transition"] = float(trans.mean())
    return res


def bootstrap_ci(values, n=2000, seed=0, ci=0.95):
    v = np.asarray([x for x in values if not np.isnan(x)])
    if len(v) == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(v.mean()), (float(lo), float(hi))


def pooled(run):
    p = np.concatenate([r[0] for r in run.values()])
    t = np.concatenate([r[1] for r in run.values()])
    m = t != -1
    return {"acc": float((p[m] == t[m]).mean()), "kappa": cohen_kappa_score(t[m], p[m]),
            "mf1": f1_score(t[m], p[m], average="macro", labels=range(5), zero_division=0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories; the first is the reference")
    ap.add_argument("--glob", default=None, help="glob of run dirs; grouped by model tag, pooled over folds")
    ap.add_argument("--mode", default="voting")
    ap.add_argument("--metrics", nargs="+", default=["kappa", "acc", "mf1", "f1_N1", "err_transition", "err_stable"])
    ap.add_argument("--json", default=None, help="write the full table here")
    args = ap.parse_args()

    # group run dirs -> label; with --glob, pool folds of the same tag (strip _f<k>)
    groups: dict[str, list[Path]] = defaultdict(list)
    if args.glob:
        for d in sorted(glob.glob(args.glob)):
            d = Path(d)
            tag = "_".join(x for x in d.name.split("_") if not x.startswith("f") or not x[1:].isdigit())
            groups[tag].append(d)
    for d in args.runs:
        groups[Path(d).name].append(Path(d))
    if len(groups) < 1:
        ap.error("no runs given")

    runs = {}
    for label, dirs in groups.items():
        merged = {}
        for d in dirs:
            for sid, v in load_run(d, args.mode).items():
                merged[f"{d.name}::{sid}" if len(dirs) > 1 else sid] = v
        runs[label] = merged

    labels = list(runs)
    ref = labels[0]
    per_subject = {lab: {sid: subject_metrics(*pt) for sid, pt in runs[lab].items()} for lab in labels}

    table = {}
    print(f"mode={args.mode}  reference={ref}  (paired differences vs reference; 95% subject-bootstrap CI)\n")
    header = f"{'run':45s} {'n':>4s} " + " ".join(f"{m:>26s}" for m in args.metrics)
    print(header)
    for lab in labels:
        subs = sorted(per_subject[lab])
        row = {"n_subjects": len(subs), "pooled": pooled(runs[lab])}
        cells = []
        for m in args.metrics:
            vals = [per_subject[lab][s][m] for s in subs]
            mean, (lo, hi) = bootstrap_ci(vals)
            row[m] = {"mean": mean, "ci": [lo, hi]}
            cell = f"{mean:.4f} [{lo:.3f},{hi:.3f}]"
            if lab != ref:
                common = [s for s in subs if s in per_subject[ref]]
                diffs = [per_subject[lab][s][m] - per_subject[ref][s][m] for s in common]
                dmean, (dlo, dhi) = bootstrap_ci(diffs)
                row[f"delta_{m}"] = {"mean": dmean, "ci": [dlo, dhi], "n_paired": len(common)}
                cell += f" Δ{dmean:+.4f}[{dlo:+.3f},{dhi:+.3f}]"
            cells.append(f"{cell:>26s}")
        print(f"{lab[:45]:45s} {len(subs):4d} " + " ".join(cells))
        table[lab] = row
    print("\npooled (historical comparison only):")
    for lab in labels:
        p = table[lab]["pooled"]
        print(f"  {lab[:45]:45s} acc={p['acc']:.4f} kappa={p['kappa']:.4f} mf1={p['mf1']:.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2, default=float))


if __name__ == "__main__":
    main()
