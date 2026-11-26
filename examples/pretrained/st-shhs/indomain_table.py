"""In-domain table of the SleepTransformer framework on the filtered SHHS-1 test split (xmai-lab/sleep-mechinterp#28,
#29): clean-cache models scored by eval_ood.py (INDOMAIN=1, shhs_visit1_preds.npz) and the sleeptransformer-phan
reference re-scored on the same nights by eval_ref_filtered.py (preds.npz), paired by subject id.

Contrasts against the first model: pooled delta kappa with a paired bootstrap 95 % CI over nights (each model keeps
its own labels, so the reference pipeline is compared fairly), Wilcoxon on per-night kappa, Holm over the contrasts.
Branch kappas (single modality, pooled) are read from the branch run.json files.

    python examples/pretrained/st-shhs/indomain_table.py --indomain <run>/indomain --ref <run>/ref_filtered \
        --branches <run> --out <run>/indomain/table.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--indomain", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--branches", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-boot", type=int, default=2000)
args = ap.parse_args()


def sid(s):
    s = str(s)
    return s if s.startswith("shhs1-") else f"shhs1-{s}"


z = np.load(Path(args.indomain) / "shhs_visit1_preds.npz")
models = list(dict.fromkeys(k.split("|")[0] for k in z.files if k != "ids"))
nights = {m: {sid(s): (z[f"{m}|{s}|y"], z[f"{m}|{s}|p"]) for s in z["ids"]} for m in models}
r = np.load(Path(args.ref) / "preds.npz")
nights["ref_phan"] = {sid(s): (r[f"{s}|y"], r[f"{s}|p"]) for s in r["ids"]}
names = models + ["ref_phan"]
common = sorted(set.intersection(*(set(nights[n]) for n in names)))
print(f"nights: {', '.join(f'{n} {len(nights[n])}' for n in names)}; common {len(common)}", flush=True)


def pooled(n, idx):
    yt = np.concatenate([nights[n][common[i]][0] for i in idx])
    yp = np.concatenate([nights[n][common[i]][1] for i in idx])
    return yt, yp


allidx = np.arange(len(common))
table = {}
for n in names:
    yt, yp = pooled(n, allidx)
    table[n] = {"kappa": float(cohen_kappa_score(yt, yp)), "acc": float((yt == yp).mean()),
                "macro_f1": float(f1_score(yt, yp, average="macro")),
                "f1_per_class": f1_score(yt, yp, average=None, labels=range(5)).tolist(), "epochs": int(len(yt))}
pn = {n: np.array([cohen_kappa_score(*nights[n][s]) if len(np.unique(np.concatenate(nights[n][s]))) > 1
                   else np.nan for s in common]) for n in names}

def kappa_cm(c):
    n = c.sum()
    po, pe = np.trace(c) / n, (c.sum(0) @ c.sum(1)) / n ** 2
    return (po - pe) / (1 - pe)


cm = {n: np.stack([np.bincount(5 * nights[n][s][0].astype(int) + nights[n][s][1].astype(int), minlength=25)
                   .reshape(5, 5) for s in common]) for n in names}       # per-night confusion matrices
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, len(common), len(common)), minlength=len(common))
              for _ in range(args.n_boot)])                               # bootstrap night weights
kb = {n: np.array([kappa_cm(c) for c in np.einsum("bn,nij->bij", W, cm[n])]) for n in names}
ref = names[0]
contrasts = []
for n in names[1:]:
    d = kb[ref] - kb[n]
    dn = pn[ref] - pn[n]
    ok = ~np.isnan(dn)
    nz = dn[ok]
    contrasts.append({
        "a": ref, "b": n, "pooled_delta_kappa": table[ref]["kappa"] - table[n]["kappa"],
        "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
        "p_boot": float(min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (args.n_boot + 1))),
        "wilcoxon_p": float(wilcoxon(nz).pvalue) if np.any(nz != 0) else 1.0,
        "wins": int((nz > 0).sum()), "ties": int((nz == 0).sum()), "n": int(ok.sum())})
for key in ("p_boot", "wilcoxon_p"):                               # Holm
    order = np.argsort([c[key] for c in contrasts])
    m, run = len(contrasts), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * contrasts[i][key]))
        contrasts[i][f"{key}_holm"] = run

branches = {}
for b in ("EEG", "EOG", "EMG"):
    f = Path(args.branches) / b / "run.json"
    if f.exists():
        branches[b] = json.loads(f.read_text()).get("final", {}).get("test", {}).get("kappa")
out = {"nights": len(common), "models": table, "per_night_kappa_mean": {n: float(np.nanmean(pn[n])) for n in names},
       "contrasts": contrasts, "branches_pooled_kappa": branches, "n_boot": args.n_boot,
       "note": "branches: pooled test kappa from run.json on all test nights of the split, not paired"}
Path(args.out).write_text(json.dumps(out, indent=1))
for n in names:
    print(f"{n}: κ {table[n]['kappa']:.4f} acc {table[n]['acc']:.4f} MF1 {table[n]['macro_f1']:.4f}")
for c in contrasts:
    print(f"{c['a']} - {c['b']}: {c['pooled_delta_kappa']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
          f"wins {c['wins']}/{c['n']} p_boot_holm {c['p_boot_holm']:.3g} wilcoxon_holm {c['wilcoxon_p_holm']:.3g}")
print("branches", branches)
print("DONE")
