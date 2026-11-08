"""Short rationale tests of the lexicon terms (plan of 08.11.2026, second part).

For every term an agent workflow wrote a falsifiable hypothesis: in which levels of the encoder (N = state of
the 40 neighbouring epochs, E = epoch, P = 0.5 s token conditioned on N and E) and with which context
(background, isolation, neighbouring tokens) the term lives. A spec per term names the inputs of a few word
variants, the baselines fair for the term and the matching variables; this module tests them on a sample of the
dump before the extended run.

- ``embed`` (CPU): quantized E / N latents of every dumped night from the codes (``MSAEVQ.rvq[s].lookup``),
  ``<dump>/eq/<key>.npz`` with ``Eq_<mod>`` / ``Nq_<mod>`` (the EOG / EMG encoders have no N).
- ``run`` (CPU): per spec and variant a leave-group-out logistic probe on the named inputs (a probe of the
  rationale, not the frozen word), its AP against the baseline probe (paired Wilcoxon over the groups) and the
  **matched test**: inside every (group, stage) the tokens are binned on the matching variables (absolute
  amplitude, amplitude relative to the background, isolation); per bin with >= ``min_bin`` events and
  non-events the normalized AP ``(AP - b) / (1 - b)`` against the bin's base rate; the event-weighted mean per
  group, one-sided Wilcoxon > 0 over the groups. A word that only follows the matched amplitudes scores ~0
  there. A null (references shifted inside the (night, stage) runs, probe refitted) for the best variant.

Inputs (names of the specs): ``P``; ``P_ctx+-k`` (mean of the P of the k tokens before and after, the token
excluded); ``rel_bg`` (log RMS of the token - median log RMS over +-10 s); ``iso`` (max log RMS over +-2 s
without the token - log RMS of the token); ``Eq`` / ``Nq`` (PCA 32 of the quantized epoch / context latent,
fitted label-free on the loaded pool nights); ``ICA_E`` / ``ICA_N`` (FastICA 16 of the same, signed and
absolute); ``codes_E`` / ``codes_N`` / ``codes_P`` (log-lift of the level-0 code and of the level 0+1 prefix,
fitted inside the fold); ``spec_tok`` (band powers + line length of the token); ``abs_amp`` (log RMS of the
token and of +-1 token); ``stage`` (one-hot). Amplitudes are on the per-night robust-scaled signal of the dump.

    python -m physioex.train.bin.concept_rationale embed --dump DUMP --combo-dir RUN
    python -m physioex.train.bin.concept_rationale run --dump DUMP --mod eeg --specs specs.json --tag g1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

T = 60
PCA_DIM, ICA_DIM = 32, 16
POOL_OUT = ("shhs", "mass_ss01", "dreams")
DREAMS_REFS = ("spindle_vis1", "spindle_vis2", "k_complex_vis1", "k_complex_vis2")


# ---------------------------------------------------------------------------- context features
def _runs_tokens(F0, night):
    """Index ranges of every night in the flat token arrays (nights are contiguous blocks)."""
    edges = np.flatnonzero(np.diff(night) != 0) + 1
    return list(zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(night)]])))


def rel_bg(F0, night, half=20):
    """log RMS of the token minus the median log RMS over +-``half`` tokens (+-10 s) of the same night."""
    from scipy.ndimage import median_filter

    out = np.zeros_like(F0)
    for a, b in _runs_tokens(F0, night):
        out[a:b] = F0[a:b] - median_filter(F0[a:b], size=2 * half + 1, mode="nearest")
    return out


def iso(F0, night, half=4):
    """Max log RMS of the +-``half`` neighbouring tokens (+-2 s, the token excluded) minus its own log RMS:
    negative = the token stands out of its surroundings."""
    from scipy.ndimage import maximum_filter

    fp = np.ones(2 * half + 1, bool)
    fp[half] = False
    out = np.zeros_like(F0)
    for a, b in _runs_tokens(F0, night):
        out[a:b] = maximum_filter(F0[a:b], footprint=fp, mode="nearest") - F0[a:b]
    return out


def ctx_mean(P, night, k):
    """Mean of the P of the ``k`` tokens before and after (the token excluded), inside each night."""
    out = np.zeros_like(P)
    for a, b in _runs_tokens(P[:, 0], night):
        c = np.cumsum(np.vstack([np.zeros((1, P.shape[1]), P.dtype), P[a:b]]), 0)
        i = np.arange(b - a)
        lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, b - a)
        out[a:b] = (c[hi] - c[lo] - P[a:b]) / np.maximum(hi - lo - 1, 1)[:, None]
    return out


# ---------------------------------------------------------------------------- probes
def lift_feature(k_tr, r_tr, k_te, prior=20.0):
    """log-lift of the keys (shrunk with ``prior`` pseudo-uses) fitted on train, 0 for unseen keys."""
    b = max(float(r_tr.mean()), 1e-9)
    u, inv = np.unique(k_tr, return_inverse=True)
    ll = np.log((np.bincount(inv, weights=r_tr) + prior * b) / (np.bincount(inv) + prior) / b)
    pos = np.searchsorted(u, k_te).clip(0, len(u) - 1)
    return np.where(u[pos] == k_te, ll[pos], 0.0).astype(np.float32)


def fold_probe(X, codes, r, key, groups, max_fit=50_000, seed=0):
    """Leave-one-group-out logistic regression on ``X`` plus, per code array in ``codes`` (``(n, L)`` int),
    the in-fold log-lift of its level-0 code and of its level 0+1 prefix. Held-out decision values."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    keys = []
    for C in codes:
        k0 = C[:, 0].astype(np.int64)
        keys.append(k0)
        if C.shape[1] > 1:
            keys.append(k0 * (int(C[:, 1].max()) + 1) + C[:, 1].astype(np.int64))
    out = np.full(len(r), np.nan)
    for g in groups:
        tr, te = np.flatnonzero(key != g), np.flatnonzero(key == g)
        if not r[te].any() or not r[tr].any() or r[tr].all():
            continue
        lf = [(lift_feature(k[tr], r[tr], k[tr]), lift_feature(k[tr], r[tr], k[te])) for k in keys]
        Xtr = np.column_stack([X[tr]] + [a for a, _ in lf]) if lf else X[tr]
        Xte = np.column_stack([X[te]] + [b for _, b in lf]) if lf else X[te]
        sub = rng.choice(len(tr), min(max_fit, len(tr)), replace=False)
        mu, sd = Xtr[sub].mean(0), Xtr[sub].std(0) + 1e-6
        m = LogisticRegression(max_iter=300).fit((Xtr[sub] - mu) / sd, r[tr][sub])
        out[te] = m.decision_function((Xte - mu) / sd)
    return out


def ap(s, r):
    from physioex.train.bin.concept_scan import average_precision

    return float(average_precision(s, r))


def per_group(s, r, key, groups):
    return {g: ap(s[key == g], r[key == g]) for g in groups if r[key == g].any() and np.isfinite(s[key == g]).all()}


def matched(s, r, key, y, M, min_bin=10):
    """Event-weighted normalized AP of ``s`` inside bins of the matching variables ``M`` (``(n, m)``), per
    (group, stage): quantile bins per variable (10, 5, 3 for 1, 2, 3 variables). Returns the per-group value
    and the fraction of the events inside usable bins."""
    q = {1: 10, 2: 5}.get(M.shape[1], 3)
    num, den, cov, tot = {}, {}, 0, 0
    for g in np.unique(key):
        for st in np.unique(y):
            m = np.flatnonzero((key == g) & (y == st))
            if not r[m].any():
                continue
            tot += int(r[m].sum())
            b = np.zeros(len(m), np.int64)
            for j in range(M.shape[1]):
                edges = np.unique(np.quantile(M[m, j], np.linspace(0, 1, q + 1)[1:-1]))
                b = b * (q + 1) + np.searchsorted(edges, M[m, j])
            for u in np.unique(b):
                ix = m[b == u]
                npos = int(r[ix].sum())
                if npos < min_bin or len(ix) - npos < min_bin:
                    continue
                base = npos / len(ix)
                v = (ap(s[ix], r[ix]) - base) / (1 - base)
                num[g] = num.get(g, 0.0) + npos * v
                den[g] = den.get(g, 0) + npos
                cov += npos
    pg = {g: num[g] / den[g] for g in num}
    return pg, (cov / tot if tot else 0.0)


def wilcoxon_gt0(d):
    from scipy.stats import wilcoxon

    d = np.asarray([v for v in d if np.isfinite(v)])
    if len(d) < 3 or not np.any(d):
        return 1.0
    return float(wilcoxon(d, alternative="greater").pvalue)


# ---------------------------------------------------------------------------- inputs
def build(D, names):
    """Dense matrix and code arrays of a list of input names; unknown names are returned apart."""
    X, codes, bad = [], [], []
    for n in names:
        mk = re.fullmatch(r"P_ctx(?:\+-|±|\+/-)?(\d+)", n)
        if n == "P":
            X.append(D["P"])
        elif mk:
            X.append(ctx_mean(D["P"], D["night"], int(mk.group(1))))
        elif n in ("rel_bg", "iso"):
            X.append(D[n][:, None])
        elif n in ("Eq", "Nq") and n in D:
            X.append(D[n])
        elif n in ("ICA_E", "ICA_N") and n in D:
            X += [D[n], np.abs(D[n])]
        elif n in ("codes_E", "codes_N", "codes_P") and n in D:
            codes.append(D[n])
        elif n == "spec_tok":
            X.append(D["F"][:, 2:])
        elif n == "abs_amp":
            X.append(D["F"][:, :2])
        elif n == "stage":
            X.append(np.stack([D["y"] == s for s in range(5)], 1).astype(np.float32))
        else:
            bad.append(n)
    X = np.column_stack(X).astype(np.float32) if X else np.zeros((len(D["y"]), 0), np.float32)
    return X, codes, bad


MATCH = {"abs": lambda D: D["F"][:, 0], "rel": lambda D: D["rel_bg"], "iso": lambda D: D["iso"]}


def run_spec(D, spec, n_null=5, seed=0):
    """One term: probes of every variant and of the baseline on the stratum of the spec, matched test, null."""
    from physioex.train.bin.concept_words import shift_within_runs

    rng = np.random.default_rng(seed)
    ref = spec["ref"]
    if ref not in D["refs"]:
        return {"term": spec["term"], "error": f"reference {ref} not in the loaded nights"}
    rv, has = D["refs"][ref]
    dreams = ref in DREAMS_REFS
    sel = np.isin(D["y"], spec["stages"]) & has & ((D["gname"] == "dreams") if dreams else ~np.isin(D["gname"], POOL_OUT))
    idx = np.flatnonzero(sel)
    S = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in D.items() if k != "refs"}
    r = rv[idx]
    key = S["night"].astype(str) if dreams else S["gname"]
    groups = sorted(set(key))
    mv = [MATCH[m](S) for m in spec.get("matched_bins") or ["abs"] if m in MATCH]
    M = np.column_stack(mv)
    out = {"term": spec["term"], "ref": ref, "n_tok": int(len(idx)), "base_rate": float(r.mean()), "groups": len(groups),
           "skipped_inputs": []}
    Xb, cb, bad = build(S, spec["baselines"])
    out["skipped_inputs"] += bad
    sb = fold_probe(Xb, cb, r, key, groups, seed=seed)
    pb = per_group(sb, r, key, groups)
    mb, _ = matched(sb, r, key, S["y"], M)
    out["baseline"] = {"inputs": spec["baselines"], "loo_ap": float(np.mean(list(pb.values()))) if pb else None,
                       "per_group": pb, "matched_mean": float(np.mean(list(mb.values()))) if mb else None}
    base_g = {g: float(r[key == g].mean()) for g in groups if r[key == g].any()}
    vars_, ref_m = [], None
    for inputs in spec["word_variants"]:
        X, cs, bad = build(S, inputs)
        out["skipped_inputs"] += bad
        if X.shape[1] == 0 and not cs:
            continue
        s = fold_probe(X, cs, r, key, groups, seed=seed)
        pg = per_group(s, r, key, groups)
        mg, cov = matched(s, r, key, S["y"], M)
        v = {"inputs": inputs, "loo_ap": float(np.mean(list(pg.values()))) if pg else None, "per_group": pg,
             "nap_vs_base": float(np.mean([(pg[g] - pb[g]) / (1 - base_g[g]) for g in pg if g in pb])) if pb else None,
             "p_vs_base": wilcoxon_gt0([pg[g] - pb[g] for g in pg if g in pb]),
             "matched": {"per_group": mg, "mean": float(np.mean(list(mg.values()))) if mg else None,
                         "p": wilcoxon_gt0(list(mg.values())), "coverage": cov}}
        if inputs == ["P"]:
            ref_m = mg
        vars_.append((v, X, cs))
    for v, _, _ in vars_:
        if ref_m is not None and v["inputs"] != ["P"]:
            mg = v["matched"]["per_group"]
            v["matched"]["p_vs_P"] = wilcoxon_gt0([mg[g] - ref_m[g] for g in mg if g in ref_m])
    out["variants"] = [v for v, _, _ in vars_]
    if vars_ and n_null:
        best = max(vars_, key=lambda t: t[0]["matched"]["mean"] if t[0]["matched"]["mean"] is not None else -9)
        nulls = []
        for i in range(n_null):
            rs = shift_within_runs(r, S["night"], S["y"], rng)
            s = fold_probe(best[1], best[2], rs, key, groups, seed=seed + i + 1)
            mg, _ = matched(s, rs, key, S["y"], M)
            nulls.append(float(np.mean(list(mg.values()))) if mg else 0.0)
        obs = best[0]["matched"]["mean"] or 0.0
        out["null_matched"] = {"inputs": best[0]["inputs"], "values": nulls,
                               "p": (1 + sum(n >= obs for n in nulls)) / (len(nulls) + 1)}
    out["skipped_inputs"] = sorted(set(out["skipped_inputs"]))
    return out


# ---------------------------------------------------------------------------- data
def load(dump, mod, refs_needed, frac=0.1, seed=0, with_eq=True):
    """Token arrays of ``mod`` on a sample of the pool nights (``frac`` per unit, >= 2) plus every
    shhs / MASS SS01 / DREAMS night: P, codes, signal features, context, E / N."""
    from physioex.train.bin.concept_scan import _group, load_refs
    from physioex.train.bin.concept_words import token_features

    meta = json.load(open(os.path.join(dump, "meta.json")))
    fs = meta["fs"][mod]
    rng = np.random.default_rng(seed)
    by_unit = {}
    for k, e in sorted(meta["nights"].items()):
        if mod in e["mods"]:
            by_unit.setdefault(e["unit"], []).append(k)
    keys = []
    for u, ks in by_unit.items():
        g = _group(u)
        keys += ks if g in POOL_OUT else list(rng.choice(ks, max(2, int(round(frac * len(ks)))), replace=False))
    A = {k: [] for k in ("P", "F", "y", "night", "gname", "codes_P", "codes_E", "codes_N", "Eq", "Nq")}
    R = {r: [] for r in refs_needed}
    for i, k in enumerate(sorted(keys)):
        z = np.load(os.path.join(dump, f"{k}.npz"))
        rf = load_refs(os.path.join(dump, "refs", f"{k}.npz"))
        n = len(z["y"])
        A["P"].append(z[f"P_{mod}"].reshape(n * T, -1).astype(np.float32))
        A["F"].append(token_features(z[f"x_{mod}"].astype(np.float32), fs))
        A["y"].append(np.repeat(z["y"].astype(np.int8), T))
        A["night"].append(np.full(n * T, i, np.int32))
        A["gname"].append(np.full(n * T, _group(meta["nights"][k]["unit"]), object))
        cp = z[f"codes_{mod}_P"]
        A["codes_P"].append(cp.reshape(n * T, -1)[:, :2].astype(np.int64))
        for s in ("E", "N"):
            c = f"codes_{mod}_{s}"
            if c in z.files:
                A[f"codes_{s}"].append(np.repeat(z[c][:, :2].astype(np.int64), T, 0))
        eqp = os.path.join(dump, "eq", f"{k}.npz")
        if with_eq and os.path.exists(eqp):
            q = np.load(eqp)
            for s in ("E", "N"):
                if f"{s}q_{mod}" in q.files:
                    A[f"{s}q"].append(np.repeat(q[f"{s}q_{mod}"].astype(np.float32), T, 0))
        for r in refs_needed:
            R[r].append(rf[r] if r in rf and len(rf[r]) == n * T else None)
    D = {k: np.concatenate(v) for k, v in A.items() if v and len(v) == len(keys)}
    D["gname"] = D["gname"].astype(str)
    D["refs"] = {}
    for r, parts in R.items():
        if any(p is not None for p in parts):
            lens = [len(a) for a in A["y"]]
            D["refs"][r] = (np.concatenate([p if p is not None else np.zeros(l, bool) for p, l in zip(parts, lens)]),
                            np.concatenate([np.full(l, p is not None) for p, l in zip(parts, lens)]))
    D["rel_bg"] = rel_bg(D["F"][:, 0], D["night"])
    D["iso"] = iso(D["F"][:, 0], D["night"])
    _latent_views(D)
    return D, sorted(keys)


def _latent_views(D, seed=0):
    """PCA (``Eq`` / ``Nq``) and FastICA (``ICA_E`` / ``ICA_N``) of the epoch latents, fitted label-free on the
    pool epochs (one row per epoch)."""
    from sklearn.decomposition import PCA, FastICA

    pool = ~np.isin(D["gname"], POOL_OUT)
    first = np.zeros(len(D["y"]), bool)
    first[::T] = True
    for s in ("E", "N"):
        if f"{s}q" not in D:
            continue
        Z = D[f"{s}q"]
        fit = Z[first & pool]
        pca = PCA(min(PCA_DIM, Z.shape[1]), random_state=seed).fit(fit)
        ica = FastICA(min(ICA_DIM, Z.shape[1]), random_state=seed, max_iter=500, whiten="unit-variance").fit(fit)
        D[f"ICA_{s}"] = ica.transform(Z).astype(np.float32)
        D[f"{s}q"] = pca.transform(Z).astype(np.float32)


# ---------------------------------------------------------------------------- commands
def embed(a):
    import torch

    from physioex.train.bin.pretrain_combo import load_s2

    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    models = {}
    for m in meta["mods"]:
        _, models[m] = load_s2(os.path.join(a.combo_dir, m, "s2"), m, "cpu")
        models[m].eval()
    t0 = time.time()
    for i, (k, e) in enumerate(sorted(meta["nights"].items())):
        out = os.path.join(a.dump, "eq", f"{k}.npz")
        if os.path.exists(out) and not a.force:
            continue
        z = np.load(os.path.join(a.dump, f"{k}.npz"))
        arr = {}
        with torch.no_grad():
            for m in e["mods"]:
                for s in ("E", "N"):
                    c = f"codes_{m}_{s}"
                    if c in z.files:
                        arr[f"{s}q_{m}"] = models[m].rvq[s].lookup(torch.from_numpy(z[c].astype(np.int64))).numpy().astype(np.float16)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, **arr)
        if i % 25 == 0:
            print(f"[embed] {i + 1}/{len(meta['nights'])} {k} {sorted(arr)} {(time.time() - t0) / 60:.1f} min", flush=True)
    print("DONE", flush=True)


def run(a):
    specs = [s for s in json.load(open(a.specs)) if s.get("mod", a.mod) == a.mod]
    if a.terms:
        specs = [s for s in specs if s["term"] in a.terms.split(",")]
    t0 = time.time()
    D, keys = load(a.dump, a.mod, sorted({s["ref"] for s in specs}), a.frac, a.seed)
    print(f"[rationale] {a.mod}: {len(D['y'])} tokens, {len(keys)} nights, inputs {sorted(k for k in D if k != 'refs')}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    res = []
    for sp in specs:
        r = run_spec(D, sp, a.nulls, a.seed)
        res.append(r)
        if "error" in r:
            print(f"[rationale] {sp['term']}: {r['error']}", flush=True)
            continue
        b = r["baseline"]
        line = " | ".join(f"{'+'.join(v['inputs'])} ap {v['loo_ap']:.3f} m {v['matched']['mean'] if v['matched']['mean'] is not None else float('nan'):.3f}"
                          f" p {v['matched']['p']:.3f}" for v in r["variants"])
        print(f"[rationale] {sp['term']}: base {r['base_rate']:.3f} baseline {b['loo_ap'] or float('nan'):.3f} "
              f"(m {b['matched_mean'] if b['matched_mean'] is not None else float('nan'):.3f}) | {line} | "
              f"null p {r.get('null_matched', {}).get('p', float('nan')):.2f} {(time.time() - t0) / 60:.1f} min", flush=True)
    out = os.path.join(a.dump, f"rationale_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(f"[rationale] -> {out}", flush=True)
    print("DONE", flush=True)


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap_.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("embed")
    e.add_argument("--dump", required=True)
    e.add_argument("--combo-dir", required=True)
    e.add_argument("--force", action="store_true")
    r = sp.add_parser("run")
    r.add_argument("--dump", required=True)
    r.add_argument("--mod", required=True)
    r.add_argument("--specs", required=True)
    r.add_argument("--terms", default="")
    r.add_argument("--tag", required=True)
    r.add_argument("--frac", type=float, default=0.1)
    r.add_argument("--nulls", type=int, default=5)
    r.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args(argv)
    {"embed": embed, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
