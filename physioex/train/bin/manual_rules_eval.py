"""Pilot evaluation of the manual-derived rule detectors (M1, xmai-lab/sleep-lens#14).

Runs :func:`physioex.explain.manual_rules.night_m1` on the nights of a dump (``concept_scan`` layout:
``meta.json``, ``scale.json``, ``<key>.npz``, ``refs/<key>.npz``), stores the masks in ``DUMP/m1/<key>.npz``
and scores them with the machinery of :mod:`physioex.train.bin.concept_refs_eval`:

- **against humans** (event F1 at IoU >= 0.2, bootstrap over nights, per group):
  ``arousal_m1`` (and ``arousal_m1 | awakening_m1``) vs ``arousal_ann`` outside W (MASS SS01 experts and
  NSRR technicians are separate groups); ``kc_m1`` vs the two DREAMS scorers in N2, next to the
  inter-scorer agreement (``k_complex_vis1`` vs ``vis2``), which is the ceiling.
- **without humans**: events per hour per group and stage (``site_rates``) for every M1 mask, to compare
  with the ranges the manuals give and to flag site dependence.

Pilot subset: ``--groups`` (default: all) and ``--per-group N`` nights per group (first N keys, sorted).

    python -m physioex.train.bin.manual_rules_eval --dump DUMP [--groups mass_ss1,dreams,shhs]
        [--per-group 20] [--workers 8]                          # -> DUMP/manual_rules_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.explain.concept_refs import T
from physioex.train.bin.concept_refs_eval import accept, bootstrap, night_counts, site_rates

M1_RULES = ("arousal_m1", "awakening_m1", "alpha_burst_m1", "alpha_intrusion_m1", "kc_m1", "kc_arousal_m1",
            "vertex_m1", "sawtooth_m1", "sawtooth_rem_m1", "sem_m1")
HUMAN_PAIRS = (  # (rule, reference, stages of the comparison)
    ("arousal_m1", "arousal_ann", (1, 2, 3, 4)), ("arousal_any_m1", "arousal_ann", (1, 2, 3, 4)),
    ("kc_m1", "k_complex_vis1", (2,)), ("kc_m1", "k_complex_vis2", (2,)),
    ("k_complex_vis1", "k_complex_vis2", (2,)))  # last = inter-scorer ceiling


def _m1_night(job):
    from physioex.explain.manual_rules import night_m1

    key, dump_dir, entry = job
    z = dict(np.load(os.path.join(dump_dir, f"{key}.npz")))
    t0 = time.time()
    r = night_m1(z, entry)
    out = os.path.join(dump_dir, "m1", f"{key}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: np.packbits(v) for k, v in r.items()}, **{f"_len_{k}": len(v) for k, v in r.items()})
    return key, sorted(r), time.time() - t0


def main(argv=None):
    from concurrent.futures import ProcessPoolExecutor

    from physioex.train.bin.concept_scan import _group, load_refs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--groups", default="", help="comma-separated groups (default all)")
    ap.add_argument("--per-group", type=int, default=0, help="nights per group, 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="recompute DUMP/m1 masks")
    ap.add_argument("--out", default=None, help="default DUMP/manual_rules_eval.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    want = set(filter(None, a.groups.split(",")))
    keys, per = [], {}
    for k, v in sorted(meta["nights"].items()):
        g = _group(v["unit"])
        if want and g not in want:
            continue
        if a.per_group and per.get(g, 0) >= a.per_group:
            continue
        per[g] = per.get(g, 0) + 1
        keys.append((k, g))
    print(f"[m1_eval] {len(keys)} nights: {per}", flush=True)
    jobs = []
    for k, _ in keys:
        if not a.force and os.path.exists(os.path.join(a.dump, "m1", f"{k}.npz")):
            continue
        v = meta["nights"][k]
        e = dict(v, fs=meta["fs"], scale={m: [o["median"], o["scale"]] for m, o in sc.get(k, {}).items()})
        jobs.append((k, a.dump, e))
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (key, names, dt) in enumerate(ex.map(_m1_night, jobs, chunksize=1)):
            print(f"[m1] {i + 1}/{len(jobs)} {key} {len(names)} masks {dt:.0f}s", flush=True)
    counts, rate_n = {}, []
    for k, g in keys:
        mp, rp = os.path.join(a.dump, "m1", f"{k}.npz"), os.path.join(a.dump, "refs", f"{k}.npz")
        if not os.path.exists(mp):
            continue
        m = load_refs(mp)
        rf = load_refs(rp) if os.path.exists(rp) else {}
        y = np.repeat(np.load(os.path.join(a.dump, f"{k}.npz"))["y"].astype(np.int8), T)
        if "arousal_m1" in m and "awakening_m1" in m:
            m["arousal_any_m1"] = m["arousal_m1"] | m["awakening_m1"]
        rate_n.append((g, y, {r: m[r] for r in M1_RULES if r in m}))
        src = {**rf, **m}
        for rule, ref, stages in HUMAN_PAIRS:
            if rule in src and ref in src and len(src[rule]) == len(src[ref]) == len(y):
                counts.setdefault((rule, ref, g), []).append(night_counts(src[rule], src[ref], np.isin(y, stages)))
    human = {}
    for (rule, ref, g), c in sorted(counts.items()):
        b = bootstrap(c, a.n_boot)
        b["verdict"] = accept(b["precision"], b["recall"])
        human.setdefault(rule, {})[f"{ref}@{g}"] = b
        print(f"[m1_eval] {rule:16s} vs {ref:15s} {g:10s} P {b['precision']:.2f} R {b['recall']:.2f} "
              f"F1 {b['f1']:.2f} [{b['ci_f1'][0]:.2f}, {b['ci_f1'][1]:.2f}] ({b['n_det']}/{b['n_ref']} events, "
              f"{b['n_nights']} nights) {b['verdict']}", flush=True)
    rates = site_rates(rate_n, rules=M1_RULES)
    for r, v in rates.items():
        ov = " ".join(f"{'WNNNR'[int(s)]}{s}:{x:.1f}" for s, x in v["rate_overall"].items())
        print(f"[m1_eval] rate/h {r:18s} {ov} | max/min {v['max_over_min']:.1f}"
              f"{' SITO' if v['site_dependent'] else ''}", flush=True)
    out = a.out or os.path.join(a.dump, "manual_rules_eval.json")
    json.dump({"nights": per, "human": human, "site_rates": rates}, open(out, "w"), indent=1, default=float)
    print(f"[m1_eval] -> {out} after {(time.time() - t0) / 60:.1f} min", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
