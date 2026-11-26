"""Out-of-domain summary of the SleepTransformer framework trained on SHHS-1 (xmai-lab/sleep-mechinterp#30).

Reads the per-cohort <ds>.json written by eval_ood.py (first model = reference, ST-S3F) and writes summary.json:
per cohort the pooled kappa / macro-F1 of every model and, for every contrast reference - other, the pooled delta
kappa with its bootstrap 95 % CI and the Wilcoxon p on per-night kappa, Holm-corrected over the cohorts of that
contrast. Counts the cohorts where the reference is better (pooled) and where the CI excludes 0.

    python examples/pretrained/st-shhs/ood_table.py --run <run folder>
"""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
run = Path(args.run)
res = {f.stem: json.loads(f.read_text()) for f in sorted(run.glob("*.json")) if f.stem != "summary"}
res = {k: v for k, v in res.items() if "models" in v}
names = list(next(iter(res.values()))["models"])
ref = names[0]


def holm(p):
    p = np.asarray(p, float)
    out, run_max = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run_max = max(run_max, min(1.0, (len(p) - rank) * p[i]))
        out[i] = run_max
    return out


cohorts = {ds: {"nights": r["nights"], **{n: {"kappa": r["models"][n]["kappa"], "macro_f1": r["models"][n]["macro_f1"]}
                                          for n in names}} for ds, r in res.items()}
contrasts = {}
for n in names[1:]:
    key = f"{ref}-{n}"
    rows = {ds: r["paired"][key] for ds, r in res.items() if key in r["paired"]}
    ph = holm([np.nan_to_num(v["wilcoxon_p"], nan=1.0) for v in rows.values()])
    for (ds, v), p in zip(rows.items(), ph):
        v["wilcoxon_p_holm"] = float(p)
    d = np.array([v["pooled_delta_kappa"] for v in rows.values()])
    contrasts[key] = {
        "cohorts": len(rows), "ref_better_pooled": int((d > 0).sum()),
        "ci_excludes_0_ref_better": int(sum(v["pooled_delta_ci95"][0] > 0 for v in rows.values())),
        "ci_excludes_0_ref_worse": int(sum(v["pooled_delta_ci95"][1] < 0 for v in rows.values())),
        "wilcoxon_holm_lt_05_ref_better": int(sum(v["wilcoxon_p_holm"] < 0.05 and v["pooled_delta_kappa"] > 0
                                                  for v in rows.values())),
        "median_delta": float(np.median(d)), "mean_delta": float(d.mean()), "per_cohort": rows}
summary = {"run": str(run), "reference": ref, "models": names, "cohorts": cohorts, "contrasts": contrasts,
           "mean_kappa": {n: float(np.mean([c[n]["kappa"] for c in cohorts.values()])) for n in names}}
(run / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"{'cohort':22s} {'n':>4s} " + " ".join(f"{n:>8s}" for n in names))
for ds, c in cohorts.items():
    print(f"{ds:22s} {c['nights']:4d} " + " ".join(f"{c[n]['kappa']:8.4f}" for n in names))
for k, c in contrasts.items():
    print(f"{k}: ref better {c['ref_better_pooled']}/{c['cohorts']}, CI>0 {c['ci_excludes_0_ref_better']}, "
          f"CI<0 {c['ci_excludes_0_ref_worse']}, median Δκ {c['median_delta']:+.4f}")
print("DONE")
