"""Paired non-inferiority test of a candidate model against a reference (e.g. a quantised MSAE
against the continuous one), on the same test nights.

For every metric the unit is the test night: ``d_i = candidate_i - reference_i``. The null is
``mean(d) <= -margin`` (the candidate is worse by at least the margin). It is rejected when the
one-sided ``1 - alpha`` lower confidence bound of ``mean(d)`` lies above ``-margin``:

- primary: paired t bound, ``mean - t_{1-alpha, n-1} * sd / sqrt(n)``, with its p value;
- check: percentile bootstrap over nights (``n_boot`` resamples).

The candidate is non-inferior when every metric passes (intersection-union test: each metric at
``alpha``, no multiplicity correction, family-wise error <= alpha). A non-significant difference
is not evidence of equality; only the margin makes the claim.

Metrics come from ``recon_eval.json`` (``per_night``) and ``probe.json`` / ``probe_b<bits>.json``
(:mod:`physioex.models.latent_probe`)::

    python -m physioex.models.noninferiority --ref RUN@8 --cand RUN@4 [--cand OTHER_RUN@8]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

BANDS5 = ("5-8", "9-12", "13-16", "17-30")
# margin per metric (absolute, in the metric's units)
MARGINS = {"wave_r2": 0.01, **{f"logpow5_{b}": 0.02 for b in BANDS5}, "spec_n_r2": 0.013,
           "kappa_NE": 0.01, "spindle_auc": 0.01}


def noninferiority(ref: Sequence[float], cand: Sequence[float], margin: float, alpha: float = 0.05,
                   n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired one-sided test of ``mean(cand - ref) > -margin``."""
    from scipy import stats

    d = np.asarray(cand, float) - np.asarray(ref, float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    if se > 0:
        lb = mean - stats.t.ppf(1 - alpha, n - 1) * se
        p = float(1 - stats.t.cdf((mean + margin) / se, n - 1))
    else:
        lb, p = mean, float(mean <= -margin)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(1)
    lb_boot = float(np.quantile(boots, alpha))
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "margin": margin, "lower_t": float(lb), "p": p,
            "lower_boot": lb_boot, "pass": bool(lb > -margin and lb_boot > -margin)}


def _pair(a: Tuple[List[int], List[float]], b: Tuple[List[int], List[float]]):
    """Align two (nights, values) lists on the common nights."""
    va, vb = dict(zip(*a)), dict(zip(*b))
    common = sorted(set(va) & set(vb))
    return [va[g] for g in common], [vb[g] for g in common]


def arm_metrics(spec: str) -> Dict[str, Tuple[List[int], List[float]]]:
    """``RUN_DIR@BITS`` -> {metric: (nights, values)} from the run's JSON files."""
    run, bits = spec.rsplit("@", 1)
    rec = json.load(open(os.path.join(run, "recon_eval.json")))["model"][f"bits_{bits}"]
    wave = rec.get("NEP_det", rec["NEP"])  # texture models: the deterministic wave carries the phase
    out = {"wave_r2": (wave["per_night"]["night"], wave["per_night"]["r2"])}
    pn = rec["NEP"]["per_night"]  # band power on the full output (with the texture, if any)
    for b in BANDS5:
        out[f"logpow5_{b}"] = (pn["night"], pn["logpow5"][b])
    if "spec_n" in rec:
        out["spec_n_r2"] = (rec["spec_n"]["per_night"]["night"], rec["spec_n"]["per_night"]["r2"])
    for name in (f"probe_b{bits}.json", "probe.json"):
        path = os.path.join(run, name)
        if not os.path.exists(path):
            continue
        pr = json.load(open(path))
        if name == "probe.json" and int(pr.get("latent_bits", 8)) != int(bits):
            continue
        st = pr["staging"]["NE"]
        out["kappa_NE"] = (st["nights"], st["kappa_per_night"])
        sp = pr["spindle_proxy"]
        out["spindle_auc"] = (sp["auc_decoded_NEP_nights"], sp["auc_decoded_NEP_per_night"])
        break
    return out


def compare(ref: str, cand: str, margins: Dict[str, float] = None, alpha: float = 0.05) -> dict:
    margins = margins or MARGINS
    mr, mc = arm_metrics(ref), arm_metrics(cand)
    res = {"ref": ref, "cand": cand, "alpha": alpha, "metrics": {}}
    for k, m in margins.items():
        if k in mr and k in mc:
            r, c = _pair(mr[k], mc[k])
            res["metrics"][k] = {**noninferiority(r, c, m, alpha), "ref_mean": float(np.mean(r)), "cand_mean": float(np.mean(c))}
    res["missing"] = [k for k in margins if k not in res["metrics"]]
    res["noninferior"] = bool(res["metrics"]) and not res["missing"] and all(v["pass"] for v in res["metrics"].values())
    res["p_iut"] = max((v["p"] for v in res["metrics"].values()), default=1.0)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="RUN_DIR@BITS")
    ap.add_argument("--cand", required=True, action="append", help="RUN_DIR@BITS (repeatable)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="JSON with every comparison")
    a = ap.parse_args()
    allres = []
    for c in a.cand:
        r = compare(a.ref, c, alpha=a.alpha)
        allres.append(r)
        print(f"\n{c}  vs  {a.ref}:  {'NON-INFERIOR' if r['noninferior'] else 'not shown'}  (IUT p = {r['p_iut']:.4f})"
              + (f"  missing: {r['missing']}" if r["missing"] else ""))
        print(f"  {'metric':14s} {'ref':>7s} {'cand':>7s} {'diff':>8s} {'lower_t':>8s} {'lower_bs':>8s} {'-margin':>8s} {'p':>8s}  n")
        for k, v in r["metrics"].items():
            print(f"  {k:14s} {v['ref_mean']:7.4f} {v['cand_mean']:7.4f} {v['mean_diff']:+8.4f} {v['lower_t']:+8.4f} "
                  f"{v['lower_boot']:+8.4f} {-v['margin']:+8.4f} {v['p']:8.4f}  {v['n']}{'' if v['pass'] else '  FAIL'}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(allres, f, indent=1)


if __name__ == "__main__":
    main()
