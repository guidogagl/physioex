"""Zoo table of the multimodal framework (xmai-lab/sleep-mechinterp#38, #41-#44).

Reads every member folder <root>/<epoch>-<seq>-s<seed>/ written by member.sbatch and writes <root>/summary.json:

- per member: parameters, minutes, branch test kappas, S2 / S3 test (kappa, acc, macro-F1), the S3 validation kappa
  at the selected checkpoint (pruning metric), F ablation, forgetting; kappa / macro-F1 per cohort of the
  evaluation-only set (12 clinical / pediatric cohorts, primary metric = their mean) and of the population cohorts;
- pruning on MASS validation only: S3 validation kappa below the reference's by more than --prune (0.02), or no
  final model;
- paired comparison with the reference (Z0) per evaluation-only cohort: pooled delta kappa with a bootstrap 95 % CI
  over nights (per-night confusion matrices, same nights and labels for every member), Wilcoxon on per-night kappa,
  Holm over the cohorts; the mean delta over cohorts with its bootstrap CI (cohorts resampled jointly);
  winner rule: better on >= --min-wins cohorts and the CI of the mean delta above 0;
- Friedman test and Nemenyi critical difference on the per-cohort ranks of the members with every cohort.

Seeds of the same (epoch, seq) are averaged in the "pairs" table.

    python examples/pretrained/zoo/analyse.py --root <stage folder> [--ref lstm-gru-s42]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ref", default="lstm-gru-s42")
ap.add_argument("--prune", type=float, default=0.02)
ap.add_argument("--min-wins", type=int, default=9)
ap.add_argument("--n-boot", type=int, default=1000)
args = ap.parse_args()
root = Path(args.root)
NAME = re.compile(r"^([a-z0-9]+)-([a-z0-9]+)-s(\d+)$")


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def selected_val_kappa(run_dir):
    """Validation kappa at the checkpoint the run selected (min loss or max kappa)."""
    run = jload(run_dir / "run.json")
    f = run_dir / "metrics.jsonl"
    if run is None or "final" not in run or not f.exists():
        return None
    recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if not recs:
        return None
    sel = run["final"].get("valid_best", {}).get("select", "loss")
    best = (min(recs, key=lambda r: r["valid"]["loss"]) if sel == "loss" else max(recs, key=lambda r: r["valid"]["kappa"]))
    return float(best["valid"]["kappa"])


def n_params(pt):
    return int(sum(v.numel() for v in torch.load(pt, map_location="cpu").values() if v.dtype.is_floating_point))


def group(run, g):
    out = {}
    d = run / "eval" / g
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        r = json.loads(f.read_text())
        if "models" in r:
            m = next(iter(r["models"].values()))
            out[f.stem] = {"nights": r["nights"], "kappa": m["kappa"], "macro_f1": m["macro_f1"]}
    return out


members = {}
for d in sorted(p for p in root.iterdir() if p.is_dir() and NAME.match(p.name)):
    e, s, seed = NAME.match(d.name).groups()
    s3 = jload(d / "S3_F" / "run.json")
    s2 = jload(d / "S2_F" / "run.json")
    rec = {"epoch": e, "seq": s, "seed": int(seed), "status": "running"}
    rec["branches"] = {m: (jload(d / m / "run.json") or {}).get("final", {}).get("test", {}).get("kappa")
                       for m in ("EEG", "EOG", "EMG")}
    if s2 and "final" in s2:
        rec["S2_test"] = {k: s2["final"]["test"][k] for k in ("kappa", "acc", "macro_f1")}
    if s3 and "final" in s3:
        F = s3["final"]
        rec.update(status="final", S3_test={k: F["test"][k] for k in ("kappa", "acc", "macro_f1")},
                   f1_per_class=F["test"].get("f1_per_class"), val_kappa=selected_val_kappa(d / "S3_F"),
                   ablation_F=F.get("test_ablation", {}).get("F", {}).get("delta"),
                   forgetting=F.get("test_forgetting"), subsets={k: v["kappa"] for k, v in F.get("test_subsets", {}).items()},
                   params=n_params(d / "S3_F" / "best.pt"),
                   minutes=sum((jload(d / x / "run.json") or {}).get("final", {}).get("minutes", 0)
                               for x in ("EEG", "EOG", "EMG", "S2_F", "S3_F")))
    rec["eval_only"] = group(d, "eval_only")
    rec["population"] = group(d, "population")
    rec["indomain"] = group(d, "indomain")
    for g in ("eval_only", "population"):
        if rec[g]:
            rec[f"{g}_mean_kappa"] = float(np.mean([v["kappa"] for v in rec[g].values()]))
    members[d.name] = rec

ref = args.ref
ref_val = members.get(ref, {}).get("val_kappa")
for n, r in members.items():
    if r["status"] != "final":
        r["pruned"] = None
    elif ref_val is None or r.get("val_kappa") is None:
        r["pruned"] = None
    else:
        r["pruned"] = bool(r["val_kappa"] < ref_val - args.prune)


# --- paired comparison with the reference on the evaluation-only cohorts -----------------------------------------
def kappa_cm(c):
    n = c.sum(-1).sum(-1)
    po = np.trace(c, axis1=-2, axis2=-1) / n
    pe = (c.sum(-2) * c.sum(-1)).sum(-1) / n ** 2
    return (po - pe) / (1 - pe)


def night_data(run, ds, name):
    f = root / run / "eval" / "eval_only" / f"{ds}_preds.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {str(s): (z[f"{name}|{s}|y"].astype(int), z[f"{name}|{s}|p"].astype(int)) for s in z["ids"]}


def holm(p):
    p = np.asarray(p, float)
    out, run_max = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run_max = max(run_max, min(1.0, (len(p) - rank) * p[i]))
        out[i] = run_max
    return out


rng = np.random.default_rng(0)
cohorts = sorted(members.get(ref, {}).get("eval_only", {}))
paired = {}
if cohorts:
    ref_n = {ds: night_data(ref, ds, ref) for ds in cohorts}
    W = {ds: None for ds in cohorts}
    for n, r in members.items():
        if n == ref or set(r["eval_only"]) != set(cohorts):
            continue
        rows, boot = {}, []
        for ds in cohorts:
            a, b = ref_n[ds], night_data(n, ds, n)
            ids = sorted(set(a) & set(b))
            ca = np.stack([np.bincount(5 * b[s][0] + b[s][1], minlength=25).reshape(5, 5) for s in ids])  # member
            cr = np.stack([np.bincount(5 * a[s][0] + a[s][1], minlength=25).reshape(5, 5) for s in ids])  # reference
            if W[ds] is None or W[ds].shape[1] != len(ids):
                W[ds] = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                                  for _ in range(args.n_boot)])
            d_boot = kappa_cm(np.einsum("bn,nij->bij", W[ds], ca)) - kappa_cm(np.einsum("bn,nij->bij", W[ds], cr))
            pk = lambda c: np.array([kappa_cm(x) if len(np.unique(np.nonzero(x)[0])) > 1 else np.nan for x in c])
            dn = pk(ca) - pk(cr)
            dn = dn[~np.isnan(dn)]
            rows[ds] = {"nights": len(ids), "delta_kappa": float(kappa_cm(ca.sum(0)) - kappa_cm(cr.sum(0))),
                        "ci95": np.percentile(d_boot, [2.5, 97.5]).tolist(),
                        "wilcoxon_p": float(wilcoxon(dn).pvalue) if np.any(dn != 0) else 1.0}
            boot.append(d_boot)
        for (ds, v), p in zip(rows.items(), holm([v["wilcoxon_p"] for v in rows.values()])):
            v["wilcoxon_p_holm"] = float(p)
        mean_boot = np.mean(boot, 0)
        d = np.array([v["delta_kappa"] for v in rows.values()])
        paired[n] = {"per_cohort": rows, "wins": int((d > 0).sum()), "cohorts": len(d),
                     "mean_delta": float(d.mean()), "mean_delta_ci95": np.percentile(mean_boot, [2.5, 97.5]).tolist(),
                     "ci_excludes_0_better": int(sum(v["ci95"][0] > 0 for v in rows.values())),
                     "ci_excludes_0_worse": int(sum(v["ci95"][1] < 0 for v in rows.values()))}
        paired[n]["beats_reference"] = bool(paired[n]["wins"] >= args.min_wins and paired[n]["mean_delta_ci95"][0] > 0)

# --- Friedman + Nemenyi over the evaluation-only cohorts ---------------------------------------------------------
complete = [n for n, r in members.items() if cohorts and set(r["eval_only"]) == set(cohorts)]
friedman = None
if len(complete) >= 3:
    K = np.array([[members[n]["eval_only"][ds]["kappa"] for n in complete] for ds in cohorts])   # (cohorts, members)
    ranks = np.array([(-row).argsort().argsort() + 1 for row in K], float)
    k, N = len(complete), len(cohorts)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    friedman = {"members": complete, "mean_rank": dict(zip(complete, ranks.mean(0).tolist())),
                "p": float(friedmanchisquare(*K.T).pvalue), "nemenyi_cd": float(cd), "cohorts": N}

# --- seed means per (epoch, seq) -------------------------------------------------------------------------------
pairs = defaultdict(list)
for n, r in members.items():
    if r["status"] == "final":
        pairs[f"{r['epoch']}-{r['seq']}"].append(r)
pair_table = {p: {"seeds": [r["seed"] for r in rs],
                  "S3_test_kappa": float(np.mean([r["S3_test"]["kappa"] for r in rs])),
                  "eval_only_mean_kappa": float(np.nanmean([r.get("eval_only_mean_kappa", np.nan) for r in rs])),
                  "population_mean_kappa": float(np.nanmean([r.get("population_mean_kappa", np.nan) for r in rs]))}
              for p, rs in pairs.items()}

summary = {"root": str(root), "reference": ref, "prune_margin": args.prune, "eval_only_cohorts": cohorts,
           "members": members, "paired_vs_reference": paired, "friedman": friedman, "pairs": pair_table}
(root / "summary.json").write_text(json.dumps(summary, indent=1))

hdr = f"{'member':22s} {'status':7s} {'params':>7s} {'valκ':>6s} {'testκ':>6s} {'evalκ':>6s} {'popκ':>6s} {'wins':>5s} {'meanΔ [CI]':>24s} pruned"
print(hdr)
for n, r in sorted(members.items(), key=lambda x: -(x[1].get("eval_only_mean_kappa") or 0)):
    p = paired.get(n, {})
    ci = p.get("mean_delta_ci95")
    print(f"{n:22s} {r['status']:7s} {r.get('params', 0) / 1e6:6.2f}M {r.get('val_kappa') or float('nan'):6.3f} "
          f"{r.get('S3_test', {}).get('kappa', float('nan')):6.3f} {r.get('eval_only_mean_kappa', float('nan')):6.3f} "
          f"{r.get('population_mean_kappa', float('nan')):6.3f} {p.get('wins', ''):>5} "
          f"{(f'{p['mean_delta']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]' if ci else ''):>24s} {r['pruned']}")
if friedman:
    print(f"Friedman p {friedman['p']:.3g}, Nemenyi CD {friedman['nemenyi_cd']:.2f} over {friedman['cohorts']} cohorts")
print("DONE")
