"""D2 of the critical evaluation: is there event content in P beyond the spectrum that a linear word cannot read?
For each reference (DREAMS expert K-complexes and spindles, scorer 1, the K-complex / spindle rules on the pool, the
annotated arousals) a linear probe and a two-layer MLP read the same inputs: the latent (``P`` + ``P_ctx±1``,
with the ICA squares as the lexicon words), its spectral twin at the same context (band powers, ``spec``) and a
richer twin (1 Hz bins + bands, ``specfine``), plus latent and twin together. Expert references: leave one
excerpt out; rule references: leave one site out on the pool. AP with a bootstrap over nights and the paired
difference latent - twin per probe. ``MLP(P) - MLP(spec) >= 0.05`` with ``linear(P) - linear(spec) < 0.05``
= the content is there but not linearly readable; ``MLP(P) - MLP(spec) < 0.05`` = the codec carries no more
than the spectrum for that concept.
``python -m physioex.train.bin.concept_nonlinear --dump DIR --out JSON [--frac 0.15 --n-ss01 10 --hidden 64]``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.train.bin import concept_rationale as RA

REFS = {"k_complex_vis1": "dreams", "spindle_vis1": "dreams", "k_complex_rule": "pool", "spindle_rule": "pool",
        "arousal_ann": "pool"}
INPUTS = {"P": ["P", "P_ctx±1"], "spec": ["spec_tok", "spec_ctx1"],
          "specfine": ["spec_tok", "spec_fine", "spec_ctx1", "spec_fine_ctx1"],
          "P+specfine": ["P", "P_ctx±1", "spec_tok", "spec_fine", "spec_ctx1", "spec_fine_ctx1"]}


def boot_ap(s, r, night, rng, n_boot=200):
    """AP over the tokens of a bootstrap sample of nights: ``(n_boot,)``."""
    ids = np.unique(night)
    by = {i: np.flatnonzero(night == i) for i in ids}
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        ix = np.concatenate([by[i] for i in pick])
        if r[ix].any() and not r[ix].all():
            out[b] = RA.ap(s[ix], r[ix])
    return out


def run(D, refs=REFS, inputs=INPUTS, hidden=64, max_fit=30_000, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for ref, where in refs.items():
        try:
            r, h = RA.ref_mask(D, ref)
        except KeyError:
            res[ref] = {"skipped": "no reference"}
            continue
        sel = h & (D["set"] == where)
        idx = np.flatnonzero(sel)
        if not r[idx].any() or len(idx) < 2 * RA.T:
            res[ref] = {"skipped": "no positives"}
            continue
        key = D["night"][idx] if where == "dreams" else D["gname"][idx]
        groups = sorted(set(key.tolist()))
        row = {"set": where, "n_tokens": int(len(idx)), "n_pos": int(r[idx].sum()), "folds": len(groups),
               "fold_by": "excerpt" if where == "dreams" else "site", "ap": {}, "boot": {}}
        scores = {}
        for name, names in inputs.items():
            X, _, bad = RA.build(D, names, idx)
            if bad or X.shape[1] == 0:
                row["ap"][name] = {"skipped": bad}
                continue
            fin = np.isfinite(X).all(1)
            row["ap"][name] = {"dim": int(X.shape[1])}
            for probe, w in (("linear", 0), ("mlp", hidden)):
                t0 = time.time()
                s = RA.fold_probe(X[fin], [], r[idx][fin], key[fin], groups, max_fit=max_fit, seed=seed, mlp=w)
                full = np.full(len(idx), np.nan)
                full[fin] = s
                ok = np.isfinite(full)
                scores[(name, probe)] = full
                row["ap"][name][probe] = RA.ap(full[ok], r[idx][ok]) if r[idx][ok].any() else None
                row["ap"][name][f"{probe}_sec"] = round(time.time() - t0, 1)
                print(f"[nonlinear] {ref} {name} {probe} AP {row['ap'][name][probe]} ({time.time() - t0:.0f} s)", flush=True)
        night = D["night"][idx]
        rb = {}
        for (name, probe), s in scores.items():
            ok = np.isfinite(s)
            rb[(name, probe)] = boot_ap(np.where(ok, s, -np.inf), r[idx], night, np.random.default_rng(seed), n_boot)
            row["boot"][f"{name}:{probe}"] = {"q05": float(np.nanquantile(rb[(name, probe)], 0.05)),
                                              "q95": float(np.nanquantile(rb[(name, probe)], 0.95))}
        for probe in ("linear", "mlp"):
            for twin in ("spec", "specfine"):
                a, b = rb.get(("P", probe)), rb.get((twin, probe))
                if a is None or b is None:
                    continue
                d = a - b
                row["boot"][f"P-{twin}:{probe}"] = {"mean": float(np.nanmean(d)), "q05": float(np.nanquantile(d, 0.05)),
                                                    "q95": float(np.nanquantile(d, 0.95)),
                                                    "p_gt0": float(np.nanmean(d > 0))}
        a, b = rb.get(("P+specfine", "mlp")), rb.get(("specfine", "mlp"))
        if a is not None and b is not None:
            d = a - b
            row["boot"]["added_by_P:mlp"] = {"mean": float(np.nanmean(d)), "q05": float(np.nanquantile(d, 0.05)),
                                             "q95": float(np.nanquantile(d, 0.95))}
        res[ref] = row
    return res


def verdict(res, margin=0.05):
    """One word per reference: ``readout`` (MLP on P beats the twin by ``margin`` but the linear probe does
    not), ``content`` (both beat it), ``none`` (the MLP does not beat it), ``linear_only`` (odd: linear beats,
    MLP does not)."""
    out = {}
    for ref, row in res.items():
        ap = row.get("ap")
        if not ap or "P" not in ap or "specfine" not in ap or "mlp" not in ap["P"]:
            out[ref] = "skipped"
            continue
        lin = (ap["P"]["linear"] or 0) - (ap["specfine"]["linear"] or 0)
        mlp = (ap["P"]["mlp"] or 0) - (ap["specfine"]["mlp"] or 0)
        out[ref] = ("content" if lin >= margin else "readout") if mlp >= margin else ("linear_only" if lin >= margin else "none")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--n-ss01", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-fit", type=int, default=30_000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--refs", default=",".join(REFS), help="comma list of references")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    RA.SPEC_FINE = True
    refs = {r: REFS.get(r, "pool") for r in a.refs.split(",") if r}
    D, keys, sets = RA.load(a.dump, "eeg", list(refs), frac=a.frac, n_ss01=a.n_ss01, seed=a.seed)
    print(f"[nonlinear] loaded {len(keys)} nights, {len(D['y'])} tokens, {time.time() - t0:.0f} s", flush=True)
    res = run(D, refs, hidden=a.hidden, max_fit=a.max_fit, n_boot=a.n_boot, seed=a.seed)
    out = {"refs": refs, "inputs": INPUTS, "hidden": a.hidden, "max_fit": a.max_fit, "frac": a.frac,
           "nights": {s: sum(v == s for v in sets.values()) for s in sorted(set(sets.values()))},
           "results": res, "verdict": verdict(res), "minutes": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"[nonlinear] wrote {a.out} verdict {out['verdict']} ({out['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
