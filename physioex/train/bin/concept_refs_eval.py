"""Validation of the rule-based references of the lexicon (phase C of the plan of 08.11.2026).

A rule detector (``physioex.explain.concept_refs``) becomes a reference of the lexicon only if it agrees
with experts; its rate must also not depend on the site more than the physiology allows.

- **level A, against experts**: event F1 on the dumped token masks (0.5 s), events = runs of tokens,
  one-to-one greedy matching at IoU >= 0.2, precision / recall / F1 pooled over the nights with a
  bootstrap over the nights (95 % CI):
  - ``spindle_rule`` vs the DREAMS scorers (``spindle_vis1`` / ``vis2``), both inside N2/N3;
  - ``k_complex_rule`` vs the DREAMS scorers inside N2, plus a tuning of ``k_mad`` x ``iso_s`` of
    :func:`detect_k_complex` on the uV signal, chosen leave-one-excerpt-out (held-out F1 reported);
  - ``emg_rise``, ``cap_a1``, ``cap_a3`` vs the scored arousals (MASS SS01 experts, NSRR technicians):
    overlap, not identity, is expected.
  Acceptance: precision >= 0.6 and recall >= 0.5 -> ``riferimento``; both >= 0.3 -> ``indicativo``;
  otherwise ``fuori``.
- **level D, site**: events per hour of every rule reference, per group and stage; on the stage where
  the rule fires most, max / min over the groups with >= 1 h of that stage > 3 flags the rule as
  site-dependent.

    python -m physioex.train.bin.concept_refs_eval --dump DUMP   # -> DUMP/refs_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.explain.concept_refs import T, TOK_S, runs

IOU = 0.2
EXPERT_PAIRS = (  # (rule, expert, stages of the comparison)
    ("spindle_rule", "spindle_vis1", (2, 3)), ("spindle_rule", "spindle_vis2", (2, 3)),
    ("k_complex_rule", "k_complex_vis1", (2,)), ("k_complex_rule", "k_complex_vis2", (2,)),
    ("emg_rise", "arousal_ann", (0, 1, 2, 3, 4)), ("cap_a1", "arousal_ann", (1, 2, 3)),
    ("cap_a3", "arousal_ann", (1, 2, 3)))
RULES = ("spindle_rule", "k_complex_rule", "slow_wave", "delta", "vertex", "sawtooth", "cap_a1", "cap_a3",
         "alpha", "lamf", "alpha_rem", "alpha_nrem", "rem_eye", "sem", "blink", "tma", "emg_rise", "rmma",
         "atonia", "artifact_eeg")
KC_GRID = {"k_mad": (4.0, 6.0, 8.0, 10.0, 12.0, 15.0), "iso_s": (0.0, 1.0, 2.0)}


def events(m: np.ndarray):
    """Runs of a token mask as ``(n, 2)`` [start, end) arrays."""
    r = runs(m)
    return np.array(r, np.int64).reshape(-1, 2)


def match(det: np.ndarray, ref: np.ndarray, iou=IOU) -> int:
    """True positives of a one-to-one greedy matching (highest IoU first) of two sorted event lists."""
    pairs = []
    j0 = 0
    for i, (a, b) in enumerate(det):
        while j0 < len(ref) and ref[j0, 1] <= a:
            j0 += 1
        j = j0
        while j < len(ref) and ref[j, 0] < b:
            inter = min(b, ref[j, 1]) - max(a, ref[j, 0])
            u = max(b, ref[j, 1]) - min(a, ref[j, 0])
            if inter > 0 and inter / u >= iou:
                pairs.append((inter / u, i, j))
            j += 1
    used_d, used_r, tp = set(), set(), 0
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_d and j not in used_r:
            used_d.add(i)
            used_r.add(j)
            tp += 1
    return tp


def night_counts(det_m, ref_m, keep=None):
    """``(tp, n_det, n_ref)`` of one night; ``keep``: token mask both sides are restricted to."""
    if keep is not None:
        det_m, ref_m = det_m & keep, ref_m & keep
    d, r = events(det_m), events(ref_m)
    return match(d, r), len(d), len(r)


def prf(c):
    c = np.asarray(c, float).reshape(-1, 3).sum(0)
    p = c[0] / c[1] if c[1] else float("nan")
    r = c[0] / c[2] if c[2] else float("nan")
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def bootstrap(counts, n_boot=1000, seed=0):
    """Precision / recall / F1 pooled over the nights, with 95 % bootstrap CIs over the nights."""
    c = np.asarray(counts, float).reshape(-1, 3)
    p, r, f = prf(c)
    rng = np.random.default_rng(seed)
    bs = np.array([prf(c[rng.integers(0, len(c), len(c))]) for _ in range(n_boot)]) if len(c) > 1 else None
    ci = (lambda k: [float(np.nanquantile(bs[:, k], .025)), float(np.nanquantile(bs[:, k], .975))]) if bs is not None \
        else (lambda k: [float("nan")] * 2)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "ci_precision": ci(0), "ci_recall": ci(1),
            "ci_f1": ci(2), "n_nights": int(len(c)), "n_det": int(c[:, 1].sum()), "n_ref": int(c[:, 2].sum())}


def accept(p, r):
    if not (np.isfinite(p) and np.isfinite(r)):
        return "non valutabile"
    if p >= 0.6 and r >= 0.5:
        return "riferimento"
    return "indicativo" if min(p, r) >= 0.3 else "fuori"


def site_rates(nights, rules=RULES, min_h=1.0, max_ratio=3.0):
    """``nights``: ``[(group, y_tok, {rule: mask})]``. Events per hour per (group, stage); the flag on
    the stage where the rule is most frequent overall."""
    out = {}
    for rule in rules:
        ev = {}  # (group, stage) -> [events, hours]
        for g, y, rf in nights:
            m = rf.get(rule)
            if m is None or len(m) != len(y):
                continue
            starts = np.zeros(len(m), bool)
            st = events(m)[:, 0]
            starts[st] = True
            for s in range(5):
                k = y == s
                e = ev.setdefault((g, s), [0, 0.0])
                e[0] += int(starts[k].sum())
                e[1] += k.sum() * TOK_S / 3600
        if not ev:
            continue
        tot = {s: sum(v[0] for (g, s2), v in ev.items() if s2 == s) /
               max(1e-9, sum(v[1] for (g, s2), v in ev.items() if s2 == s)) for s in range(5)}
        main = max(tot, key=tot.get)
        rates = {g: v[0] / v[1] for (g, s), v in ev.items() if s == main and v[1] >= min_h}
        lo, hi = (min(rates.values()), max(rates.values())) if rates else (0.0, 0.0)
        ratio = hi / lo if lo > 0 else float("inf") if hi > 0 else float("nan")
        out[rule] = {"stage": int(main), "rate_overall": {str(s): float(v) for s, v in tot.items()},
                     "rate_by_group": {g: float(v) for g, v in sorted(rates.items())}, "max_over_min": float(ratio),
                     "site_dependent": bool(ratio > max_ratio)}
    return out


def kc_tuning(excerpts, grid=KC_GRID):
    """``excerpts``: ``[(name, x_uv (n, S), ts, y, fs, {vis1, vis2 token masks})]``. Counts of every
    grid point per excerpt, the leave-one-excerpt-out choice (mean F1 over the two scorers) and its
    held-out precision / recall / F1."""
    from physioex.explain.concept_refs import _per_run, detect_k_complex

    keys = [(k, i) for k in grid["k_mad"] for i in grid["iso_s"]]
    C = {}  # (param, excerpt, scorer) -> counts
    for name, x, ts, y, fs, ex in excerpts:
        keep = np.repeat(y == 2, T) if (y == 2).any() else None
        for k, i in keys:
            det = _per_run(x, ts, fs, lambda s: detect_k_complex(s, fs, k_mad=k, iso_s=i))
            for sc, ref in ex.items():
                C[(k, i), name, sc] = night_counts(det, ref, keep)
    names = [e[0] for e in excerpts]
    scorers = sorted({sc for _, _, sc in C})

    def score(p, ns):
        return np.mean([prf([C[p, n, sc] for n in ns])[2] for sc in scorers])

    held = {sc: [] for sc in scorers}
    picks = {}
    for n in names:
        rest = [m for m in names if m != n]
        best = max(keys, key=lambda p: score(p, rest))
        picks[n] = list(best)
        for sc in scorers:
            held[sc].append(C[best, n, sc])
    full = max(keys, key=lambda p: score(p, names))
    return {"grid": {f"{k}_{i}": {sc: dict(zip(("precision", "recall", "f1"), map(float, prf([C[(k, i), n, sc] for n in names]))))
                                  for sc in scorers} for k, i in keys},
            "loo_picks": picks, "best_all": {"k_mad": full[0], "iso_s": full[1]},
            "heldout": {sc: {**bootstrap(held[sc]), "verdict": accept(*prf(held[sc])[:2])} for sc in scorers}}


def main(argv=None):
    from physioex.train.bin.concept_scan import _group, load_refs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="default DUMP/refs_eval.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    counts = {}  # (rule, expert, group) -> [per-night counts]
    rate_n, kc_ex = [], []
    for key, e in sorted(meta["nights"].items()):
        rp = os.path.join(a.dump, "refs", f"{key}.npz")
        if not os.path.exists(rp):
            continue
        rf = load_refs(rp)
        z = np.load(os.path.join(a.dump, f"{key}.npz"))
        y = np.repeat(z["y"].astype(np.int8), T)
        g = _group(e["unit"])
        rate_n.append((g, y, {r: rf[r] for r in RULES if r in rf}))
        for rule, exp, stages in EXPERT_PAIRS:
            if rule in rf and exp in rf and len(rf[rule]) == len(rf[exp]) == len(y):
                counts.setdefault((rule, exp, g), []).append(night_counts(rf[rule], rf[exp], np.isin(y, stages)))
        if "k_complex_vis1" in rf and "x_eeg" in z.files and "eeg" in sc.get(key, {}):
            o = sc[key]["eeg"]
            x = z["x_eeg"].astype(np.float32) * np.float32(o["scale"]) + np.float32(o["median"])
            ex = {s: rf[f"k_complex_{s}"] for s in ("vis1", "vis2") if f"k_complex_{s}" in rf}
            kc_ex.append((key, x, z["ts"], z["y"].astype(np.int8), meta["fs"]["eeg"], ex))
    print(f"[refs_eval] read {len(rate_n)} nights, {(time.time() - t0) / 60:.1f} min", flush=True)
    expert = {}
    for (rule, exp, g), c in sorted(counts.items()):
        b = bootstrap(c, a.n_boot)
        b["verdict"] = accept(b["precision"], b["recall"])
        expert.setdefault(rule, {})[f"{exp}@{g}"] = b
        print(f"[refs_eval] {rule:15s} vs {exp:15s} {g:10s} P {b['precision']:.2f} R {b['recall']:.2f} "
              f"F1 {b['f1']:.2f} ({b['n_det']}/{b['n_ref']} events, {b['n_nights']} nights) {b['verdict']}", flush=True)
    kc = kc_tuning(kc_ex) if kc_ex else None
    if kc:
        for s, v in kc["heldout"].items():
            print(f"[refs_eval] KC tuned (LOO) vs {s}: P {v['precision']:.2f} R {v['recall']:.2f} F1 {v['f1']:.2f} "
                  f"{v['verdict']}; best on all {kc['best_all']}", flush=True)
    rates = site_rates(rate_n)
    for r, v in rates.items():
        print(f"[refs_eval] rate {r:15s} stage {v['stage']} max/min {v['max_over_min']:.1f}"
              f"{' SITO' if v['site_dependent'] else ''}", flush=True)
    out = a.out or os.path.join(a.dump, "refs_eval.json")
    json.dump({"iou": IOU, "expert": expert, "kc_tuning": kc, "site_rates": rates}, open(out, "w"), indent=1,
              default=float)
    print(f"[refs_eval] -> {out} after {(time.time() - t0) / 60:.1f} min", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
