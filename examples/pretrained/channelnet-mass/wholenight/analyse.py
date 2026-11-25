"""Analysis of the whole-night fine-tuning runs (xmai-lab/sleep-mechinterp#34).

Reads <run>/<config>_s<seed>/{run.json, preds.npz} and the no-training baseline <run>/S3F_init (S3-F through the same
code: 'vote20' = baseline (a), 'whole' = baseline (c)). Writes <run>/summary.json and <run>/SUMMARY.md:
    per configuration: mean +- sd over seeds (whole night and L 20 voting) of kappa, macro-F1, F1 N1/REM, kappa per
    third of the night, accuracy near (+-2 epochs) / away from stage transitions, occlusion subsets, aux heads;
    seed ensemble (per-epoch majority over seeds) used for the paired contrasts: pooled delta kappa, bootstrap 95 % CI
    over nights, per-night delta, wins, Wilcoxon; Holm over the pre-specified contrasts (bootstrap p);
    hypothesis: configuration with the lowest mean validation loss vs baseline (a).

    python examples/pretrained/channelnet-mass/wholenight/analyse.py --run <run folder>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, f1_score

CONFIGS = ["A-L20", "A-mix", "A-whole", "A-crop", "A-perm", "A-noperm", "A-permrand", "A-noocc", "B-K4", "B-K2",
           "B-nods", "B-untied", "C", "C+A"]
CONTRASTS = [  # (name, (config, mode), (config, mode)); S3F = no-training baseline
    ("1 whole-night inference, no training", ("S3F", "whole"), ("S3F", "vote20")),
    ("2 extra training alone (L20)", ("A-L20", "vote20"), ("S3F", "vote20")),
    ("3a long context vs L20 fine-tuning (voting)", ("A-mix", "whole"), ("A-L20", "vote20")),
    ("3b long context vs L20 fine-tuning (whole)", ("A-mix", "whole"), ("A-L20", "whole")),
    ("4a whole nights only", ("A-whole", "whole"), ("A-mix", "whole")),
    ("4b crops only", ("A-crop", "whole"), ("A-mix", "whole")),
    ("4c permutations only", ("A-perm", "whole"), ("A-mix", "whole")),
    ("4d mix without permutations", ("A-noperm", "whole"), ("A-mix", "whole")),
    ("4e random cuts vs cycle cuts", ("A-permrand", "whole"), ("A-mix", "whole")),
    ("5 no occlusion", ("A-noocc", "whole"), ("A-mix", "whole")),
    ("6a loop K4", ("B-K4", "whole"), ("A-mix", "whole")),
    ("6b K2 vs K4", ("B-K2", "whole"), ("B-K4", "whole")),
    ("6c no deep supervision", ("B-nods", "whole"), ("B-K4", "whole")),
    ("6d untied vs tied", ("B-untied", "whole"), ("B-K4", "whole")),
    ("7a night head alone", ("C", "whole"), ("A-mix", "whole")),
    ("7b night head + sequence vs night head", ("C+A", "whole"), ("C", "whole")),
    ("7c night head + sequence vs sequence", ("C+A", "whole"), ("A-mix", "whole")),
]


def kappa(yt, yp):
    return float(cohen_kappa_score(yt, yp)) if len(np.unique(np.concatenate([yt, yp]))) > 1 else float("nan")


def load_preds(f: Path):
    z = np.load(f)
    out = {}
    for k in z.files:
        if "|" not in k:
            continue
        cond, i, w = k.split("|")
        out.setdefault(cond, {}).setdefault(int(i), {})[w] = z[k]
    return {c: [(d[i]["y"], d[i]["p"]) for i in sorted(d)] for c, d in out.items()}


def metrics(preds):
    yt = np.concatenate([a for a, _ in preds])
    yp = np.concatenate([b for _, b in preds])
    f1 = f1_score(yt, yp, average=None, labels=range(5), zero_division=0)
    thirds = [[], [], []]
    near, far = [], []
    for a, b in preds:
        n = len(a)
        for t in range(3):
            s, e = t * n // 3, (t + 1) * n // 3
            thirds[t].append((a[s:e], b[s:e]))
        tr = np.where(a[1:] != a[:-1])[0] + 1
        m = np.zeros(n, bool)
        for t in tr:
            m[max(0, t - 2):t + 2] = True
        near.append((a[m] == b[m]))
        far.append((a[~m] == b[~m]))
    return {"kappa": kappa(yt, yp), "acc": float((yt == yp).mean()),
            "macro_f1": float(f1_score(yt, yp, average="macro")), "f1": f1.tolist(),
            "kappa_thirds": [kappa(np.concatenate([a for a, _ in th]), np.concatenate([b for _, b in th]))
                             for th in thirds],
            "acc_near_transition": float(np.concatenate(near).mean()),
            "acc_away_transition": float(np.concatenate(far).mean()),
            "per_night_kappa": [kappa(a, b) for a, b in preds]}


def ensemble(seed_preds):
    """Per-epoch majority over seeds (ties -> lowest seed's prediction)."""
    out = []
    for nights in zip(*seed_preds):
        yt = nights[0][0]
        P = np.stack([p for _, p in nights])
        votes = np.stack([(P == c).sum(0) for c in range(5)])
        best = votes.max(0)
        pick = P[0].copy()
        amb = (votes[P[0], np.arange(P.shape[1])] < best)
        pick[amb] = votes.argmax(0)[amb]
        out.append((yt, pick))
    return out


def paired(pa, pb, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pa)
    ka = np.array([kappa(a, b) for a, b in pa])
    kb = np.array([kappa(a, b) for a, b in pb])
    d = ka - kb
    yt = [a for a, _ in pa]
    pooled = lambda P, idx: kappa(np.concatenate([yt[i] for i in idx]), np.concatenate([P[i][1] for i in idx]))
    full = pooled(pa, range(n)) - pooled(pb, range(n))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot.append(pooled(pa, idx) - pooled(pb, idx))
    boot = np.array(boot)
    p_boot = float(min(1.0, 2 * (min((boot <= 0).sum(), (boot >= 0).sum()) + 1) / (n_boot + 1)))   # floor 2/(B+1)
    ok = ~np.isnan(d)
    return {"pooled_delta_kappa": float(full), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_boot": p_boot, "mean_night_delta": float(d[ok].mean()), "wins": int((d[ok] > 0).sum()), "n": int(ok.sum()),
            "wilcoxon_p": float(wilcoxon(d[ok]).pvalue) if np.any(d[ok] != 0) else 1.0, "boot": boot}


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), np.zeros(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    seeds, valid, failed = {}, {}, {}
    for cfg in CONFIGS:
        for d in sorted(run.glob(f"{cfg}_s*")):
            if d.name.rsplit("_s", 1)[0] != cfg:
                continue
            rj = json.loads((d / "run.json").read_text())
            if "final" not in rj or not (d / "preds.npz").exists():
                failed.setdefault(cfg, []).append(d.name)
                continue
            seeds.setdefault(cfg, []).append(load_preds(d / "preds.npz"))
            valid.setdefault(cfg, []).append(rj["final"]["valid_best_loss"])
    base = load_preds(run / "S3F_init" / "preds.npz")

    summary = {"configs": {}, "failed": failed, "baseline": {c: metrics(p) for c, p in base.items()}}
    for d in summary["baseline"].values():
        d.pop("per_night_kappa")
    ens = {"S3F": base}
    for cfg, runs in seeds.items():
        conds = runs[0].keys()
        per_seed = {c: [metrics(r[c]) for r in runs] for c in conds}
        ens[cfg] = {c: ensemble([r[c] for r in runs]) for c in conds}
        row = {"n_seeds": len(runs), "valid_loss": [float(np.mean(valid[cfg])), float(np.std(valid[cfg]))]}
        for c in conds:
            ks = [m["kappa"] for m in per_seed[c]]
            em = metrics(ens[cfg][c])
            em.pop("per_night_kappa")
            row[c] = {"kappa_mean": float(np.mean(ks)), "kappa_sd": float(np.std(ks)),
                      "macro_f1_mean": float(np.mean([m["macro_f1"] for m in per_seed[c]])), "ensemble": em}
        summary["configs"][cfg] = row

    res = []
    for name, (ca, ma), (cb, mb) in CONTRASTS:
        if ca in ens and cb in ens:
            r = paired(ens[ca][ma], ens[cb][mb])
            r.pop("boot")
            res.append({"name": name, "a": f"{ca}/{ma}", "b": f"{cb}/{mb}", **r})
    for r, p, q in zip(res, holm([r["p_boot"] for r in res]), holm([r["wilcoxon_p"] for r in res])):
        r["p_holm"], r["wilcoxon_p_holm"] = p, q
    summary["contrasts"] = res

    sel = min(valid, key=lambda c: np.mean(valid[c]))
    h = paired(ens[sel]["whole"], ens["S3F"]["vote20"])
    m = len(res) + 1
    p_h = min(1.0, h["p_boot"] * m)                        # Bonferroni over contrasts + hypothesis (conservative)
    h.pop("boot")
    summary["hypothesis"] = {"selected_on_validation": sel, **h, "p_corrected": p_h,
                             "supported": bool(h["pooled_delta_kappa"] > 0 and p_h < 0.05)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))

    L = ["# Whole-night fine-tuning on MASS (fold 0 test, 34 nights)", ""]
    b = summary["baseline"]
    L += [f"Baseline S3-F, no training: vote20 κ {b['vote20']['kappa']:.4f}, whole night κ {b['whole']['kappa']:.4f}", ""]
    L += ["| config | seeds | val loss | κ whole (mean ± sd) | κ vote20 | ens κ whole | macro-F1 | F1 N1 | F1 REM | κ thirds | acc near / away transitions |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        if cfg not in summary["configs"]:
            L.append(f"| {cfg} | 0 | {'failed: ' + ', '.join(failed[cfg]) if cfg in failed else 'missing'} |||||||||")
            continue
        r = summary["configs"][cfg]
        e = r["whole"]["ensemble"]
        L.append(f"| {cfg} | {r['n_seeds']} | {r['valid_loss'][0]:.4f} | {r['whole']['kappa_mean']:.4f} ± "
                 f"{r['whole']['kappa_sd']:.4f} | {r['vote20']['kappa_mean']:.4f} | {e['kappa']:.4f} | "
                 f"{e['macro_f1']:.4f} | {e['f1'][1]:.3f} | {e['f1'][4]:.3f} | "
                 f"{' / '.join(f'{k:.3f}' for k in e['kappa_thirds'])} | {e['acc_near_transition']:.3f} / "
                 f"{e['acc_away_transition']:.3f} |")
    L += ["", "| contrast | a | b | Δκ pooled | 95 % CI | night Δ | wins | p boot (Holm) | Wilcoxon p (Holm) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['name']} | {r['a']} | {r['b']} | {r['pooled_delta_kappa']:+.4f} | [{r['ci95'][0]:+.4f}, "
                 f"{r['ci95'][1]:+.4f}] | {r['mean_night_delta']:+.4f} | {r['wins']}/{r['n']} | {r['p_holm']:.3g} | "
                 f"{r['wilcoxon_p_holm']:.3g} |")
    hy = summary["hypothesis"]
    L += ["", f"Hypothesis: selected on validation = **{sel}** (whole night) vs S3-F vote20: Δκ "
          f"{hy['pooled_delta_kappa']:+.4f} [{hy['ci95'][0]:+.4f}, {hy['ci95'][1]:+.4f}], bootstrap p (Bonferroni) {hy['p_corrected']:.3g}, Wilcoxon p {hy['wilcoxon_p']:.3g}"
          f" -> {'SUPPORTED' if hy['supported'] else 'NOT supported'}"]
    (run / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
