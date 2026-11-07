"""Directional words of every lexicon concept (family (b) of the plan of 07.11.2026).

The discrete codes carry every concept but spread it over hundreds of codes (``concept_scan scan``);
a word of the lexicon must be one to three directions. Per modality:

- **bases** (label-free): FastICA of the quantized P tokens (``pq1``, full rank) and of 3 consecutive
  tokens (``pq3``, ``min(3 d_p, 48)``), fitted on the pool nights of the dump. The out-of-pool sets
  (shhs, MASS SS01, DREAMS) never enter the fit; inside the pool leave-one-item-out the basis has seen
  the held-out item without labels (declared in ``LEXICON_CONCETTI.md``).
- **words**: per direction ``z`` (``+``), ``-z`` (``-``), ``|z|``; energies of label-free groups (the 4
  most kurtotic directions, the 10 pairs with the most correlated ``z^2``, all), raw, smoothed over
  +-1 token, and as the log change against the 10 tokens before (``rel``, abrupt onsets such as arousals).
  A word is a continuous score; its threshold is set at freezing.
- **selection**: within the concept's stratum (``concept_scan.TARGETS``) the AP of every word on every
  held-out group; leave-one-group-out choice of the best word (held-out AP, how often the overall best
  word wins); top 3 words with distinct directions; transfer to shhs / MASS SS01 / DREAMS and to the
  expert references.
- **nulls**: the same selection on a random rotation of the whitened coordinates (``--rotations``) and
  on references shifted circularly within the nights (``--shifts``).

    python -m physioex.train.bin.concept_words --dump DUMP --mod eeg --concepts spindle,slow_wave --tag eeg1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.train.bin.concept_scan import EXPERT, TARGETS, _group, average_precision, load_refs

T = 60
FIT_ROWS = 200_000


def ctx3_rows(P):
    """``(n, T, d) -> (n, T, 3d)``: previous, current, next token; contiguous across the epochs of ``P``
    (one night, consecutive dumped epochs), edges repeated."""
    n, t, d = P.shape
    v = P.reshape(-1, d)
    return np.concatenate([np.r_[v[:1], v[:-1]], v, np.r_[v[1:], v[-1:]]], 1).reshape(n, t, 3 * d)


def _nights(dump_dir, mod):
    meta = json.load(open(os.path.join(dump_dir, "meta.json")))
    return [(k, e) for k, e in sorted(meta["nights"].items()) if mod in e["mods"]]


def fit_bases(dump_dir, mod, seed=0, frac=0.1):
    """FastICA bases on the pool nights (label-free); also the energy groups of each basis."""
    from scipy.stats import kurtosis
    from sklearn.decomposition import FastICA

    rng = np.random.default_rng(seed)
    X1, X3 = [], []
    for key, e in _nights(dump_dir, mod):
        if _group(e["unit"]) in ("shhs", "mass_ss01", "dreams"):
            continue
        P = np.load(os.path.join(dump_dir, f"{key}.npz"))[f"P_{mod}"].astype(np.float32)
        idx = rng.random(P.shape[0] * T) < frac
        X1.append(P.reshape(-1, P.shape[-1])[idx])
        X3.append(ctx3_rows(P).reshape(-1, 3 * P.shape[-1])[idx])
    X1, X3 = np.concatenate(X1), np.concatenate(X3)
    out = {}
    for name, X, k in (("pq1", X1, X1.shape[1]), ("pq3", X3, min(X3.shape[1], 48))):
        sub = X[rng.choice(len(X), min(FIT_ROWS, len(X)), replace=False)]
        ica = FastICA(k, whiten="unit-variance", random_state=seed, max_iter=500).fit(sub)
        Z = ica.transform(sub)
        kt = kurtosis(Z, axis=0)
        top = [int(d) for d in np.argsort(-kt)[:4]]
        sq = Z ** 2
        C = np.corrcoef(sq.T)
        iu = np.triu_indices(len(C), 1)
        pairs = [[int(iu[0][a]), int(iu[1][a])] for a in np.argsort(-C[iu])[:10]]
        groups = {"kurt4": top, **{f"d{i}+d{j}": [i, j] for i, j in pairs}, "all": list(range(k))}
        out[name] = {"mean": ica.mean_.astype(np.float32), "unmix": ica.components_.astype(np.float32),
                     "mix": ica.mixing_.astype(np.float32), "groups": groups, "kurtosis": kt.tolist()}
    return out


def transform(P, b, rep):
    X = P if rep == "pq1" else ctx3_rows(P)
    return ((X.reshape(-1, X.shape[-1]) - b["mean"]) @ b["unmix"].T).astype(np.float32)


def load_coords(dump_dir, mod, bases, level_refs, frac=0.25, seed=0):
    """Coordinates of a random ``frac`` of the epochs of every night (whole 60-token epochs, the context
    computed on the whole night first), stage, group, night and the token reference masks."""
    rng = np.random.default_rng(seed + 7)
    Z = {r: [] for r in bases}
    Y, G, NI, groups = [], [], [], []
    R = {r: ([], []) for r in level_refs}
    for i, (key, e) in enumerate(_nights(dump_dir, mod)):
        z = np.load(os.path.join(dump_dir, f"{key}.npz"))
        P = z[f"P_{mod}"].astype(np.float32)
        keep = rng.random(len(P)) < (1.0 if e["unit"] == "dreams" else 2 * frac if e["unit"] == "mass_ss01" else frac)
        if not keep.any():
            continue
        tk = np.repeat(keep, T)
        for r, b in bases.items():
            Z[r].append(transform(P, b, r)[tk])
        y = np.repeat(z["y"].astype(np.int8), T)[tk]
        g = _group(e["unit"])
        if g not in groups:
            groups.append(g)
        Y.append(y)
        G.append(np.full(len(y), groups.index(g), np.int16))
        NI.append(np.full(len(y), i, np.int32))
        rf = load_refs(os.path.join(dump_dir, "refs", f"{key}.npz"))
        for r in level_refs:
            ok = r in rf and len(rf[r]) == len(tk)
            R[r][0].append(rf[r][tk] if ok else np.zeros(len(y), bool))
            R[r][1].append(np.full(len(y), ok))
    return ({r: np.concatenate(v) for r, v in Z.items()}, np.concatenate(Y), np.concatenate(G), np.concatenate(NI),
            groups, {r: (np.concatenate(a), np.concatenate(h)) for r, (a, h) in R.items()})


def _rows_mean3(e):
    e2 = e.reshape(-1, T)
    pad = np.concatenate([e2[:, :1], e2, e2[:, -1:]], 1)
    return ((pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3).reshape(-1)


def _rows_rel(e, base=10):
    """``log e[t] - log median(e[t-base:t])`` inside rows of 60 tokens, at least one token before
    (``vq_lexicon.rel_change`` with ``win=1``; not imported to keep this module free of torch)."""
    from numpy.lib.stride_tricks import sliding_window_view

    L = np.log(np.maximum(e.reshape(-1, T), 1e-6))
    P = np.concatenate([np.full((len(L), base), np.nan), L], 1)
    Wv = sliding_window_view(P, base, axis=1)
    hi = np.maximum(1, np.arange(T))
    return (L - np.nanmedian(Wv[:, hi], -1)).reshape(-1)


def word_names(D, groups, rep):
    """Names of every word of one basis with ``D`` directions."""
    out = [f"{rep}:{f}" for d in range(D) for f in (f"d{d}+", f"d{d}-", f"|d{d}|")]
    return out + [f"{rep}:E[{g}]{k}" for g in groups for k in ("", "~", "rel")]


def word_score(Z, name, groups):
    """Score of one word on coordinates ``Z`` of whole 60-token rows (``rep:...`` already resolved)."""
    w = name.split(":", 1)[1]
    if w.startswith("E["):
        g, kind = w[2:w.index("]")], w[w.index("]") + 1:]
        e = (Z[:, groups[g]] ** 2).sum(1)
        return _rows_mean3(e) if kind == "~" else _rows_rel(e) if kind == "rel" else e
    d = int(w.strip("|+-d"))
    return np.abs(Z[:, d]) if w.startswith("|") else Z[:, d] if w.endswith("+") else -Z[:, d]


def all_words(Zs, bases):
    return [n for rep, Z in Zs.items() for n in word_names(Z.shape[1], bases[rep]["groups"], rep)]


def score_of(Zs, bases, name, idx):
    rep = name.split(":", 1)[0]
    return word_score(Zs[rep][idx], name, bases[rep]["groups"])


def word_dirs(name):
    """Directions of a word (``pq1:d3+`` -> ``("pq1", [3])``; energies: their group)."""
    rep, w = name.split(":", 1)
    if w.startswith("E["):
        return rep, w[2:w.index("]")]
    return rep, int(w.strip("|+-d"))


def ap_matrix(Zs, bases, names, idx, ref, grp, groups):
    """AP of every word on every group that has the reference: ``(W, G)`` and the group ids. ``idx``:
    token indices of whole 60-token rows; ``ref`` / ``grp`` over ``idx``."""
    gs = [g for g in groups if ref[grp == g].any()]
    masks = [grp == g for g in gs]
    A = np.full((len(names), len(gs)), np.nan)
    for i, n in enumerate(names):
        s = score_of(Zs, bases, n, idx)
        for j, m in enumerate(masks):
            A[i, j] = average_precision(s[m], ref[m])
    return A, gs


def loo_choice(A):
    """Leave-one-group-out: best word on the other groups, its AP on the held-out one."""
    held, picks = [], []
    for j in range(A.shape[1]):
        rest = np.delete(A, j, 1)
        w = int(np.nanargmax(np.nanmean(rest, 1))) if A.shape[1] > 1 else int(np.nanargmax(A[:, j]))
        held.append(A[w, j])
        picks.append(w)
    return held, picks


def shift_within_nights(ref, night, rng):
    out = np.empty_like(ref)
    e = np.concatenate([[0], np.flatnonzero(np.diff(night)) + 1, [len(night)]])
    for a, b in zip(e[:-1], e[1:]):
        out[a:b] = np.roll(ref[a:b], rng.integers(1, max(2, b - a)))
    return out


def scan_concept(name, Zs, bases, y, grp_names, night, refs, n_rot=5, n_shift=5, seed=0, max_tok=600_000):
    ref_name, _, stages, _, sel_set = TARGETS[name]
    ref, has = refs[ref_name]
    ext = np.isin(grp_names, ["shhs", "mass_ss01", "dreams"])
    strat = np.isin(y, stages) & has
    fit = strat & ((grp_names == "dreams") if sel_set == "dreams" else ~ext)
    if not ref[fit].any():
        return {"concept": name, "ref": ref_name, "status": "no reference in the selection set"}
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(fit.reshape(-1, T).all(1))  # the stratum and the reference are per epoch: whole rows
    if len(rows) * T > max_tok:
        rows = np.sort(rng.choice(rows, max_tok // T, replace=False))
    idx = (rows[:, None] * T + np.arange(T)).reshape(-1)
    key = grp_names[idx] if sel_set == "pool" else night[idx].astype(str)
    groups = sorted(set(key))
    names = all_words(Zs, bases)
    r = ref[idx]
    A, gs = ap_matrix(Zs, bases, names, idx, r, key, groups)
    mean_ap = np.nanmean(A, 1)
    order = np.argsort(-mean_ap)
    held, picks = loo_choice(A)
    best = int(order[0])
    top, seen = [], set()
    for w in order:
        d = word_dirs(names[w])
        if d in seen:
            continue
        seen.add(d)
        top.append({"word": names[w], "ap": float(mean_ap[w]), "ap_sd_groups": float(np.nanstd(A[w]))})
        if len(top) == 3:
            break
    res = {"concept": name, "ref": ref_name, "stages": list(stages), "selection_set": sel_set,
           "base_rate": float(r.mean()), "n_tok": int(len(idx)), "groups": gs, "top": top,
           "loo_ap": float(np.nanmean(held)), "loo_ap_per_group": dict(zip(gs, map(float, held))),
           "stability": float(np.mean([p == best for p in picks])),
           "loo_words": [names[p] for p in picks]}
    # nulls: random rotations of the whitened coordinates, references shifted within the nights
    from scipy.stats import ortho_group

    rot = []
    for s in range(n_rot):
        Zr = {rep: (Z[idx] @ ortho_group.rvs(Z.shape[1], random_state=seed + 100 + s)).astype(np.float32)
              for rep, Z in Zs.items()}
        A_r = ap_matrix(Zr, bases, names, np.arange(len(idx)), r, key, groups)[0]
        rot.append(float(np.nanmean(loo_choice(A_r)[0])))
    sh = []
    for s in range(n_shift):
        rs = shift_within_nights(r, night[idx], rng)
        sh.append(float(np.nanmean(loo_choice(ap_matrix(Zs, bases, names, idx, rs, key, groups)[0])[0])))
    res["null_rotation"] = {"q95": float(np.quantile(rot, 0.95)) if rot else None, "values": rot}
    res["null_shift"] = {"q95": float(np.quantile(sh, 0.95)) if sh else None, "values": sh}
    # within-stage AP of the best word (mean over the stratum's stages, weighted by the reference)
    ws, wt = 0.0, 0
    for st in stages:
        m = y[idx] == st
        if r[m].any():
            ws += r[m].sum() * average_precision(score_of(Zs, bases, names[best], idx)[m], r[m]) / r[m].mean()
            wt += r[m].sum()
    res["best_ap_over_base_within_stage"] = ws / wt if wt else None
    # transfer of the top words (chosen on the selection set) to the out-of-pool sets and the experts
    tr = {}

    def rows_of(m):
        rw = np.flatnonzero(m.reshape(-1, T).all(1))
        return (rw[:, None] * T + np.arange(T)).reshape(-1)

    for g in ("shhs", "mass_ss01", "dreams"):
        ix = rows_of(strat & (grp_names == g))
        if g == sel_set or not len(ix) or not ref[ix].any():
            continue
        tr[g] = {"base": float(ref[ix].mean()),
                 **{t["word"]: average_precision(score_of(Zs, bases, t["word"], ix), ref[ix]) for t in top}}
    for e in EXPERT.get(name, ()):
        if e not in refs:
            continue
        er, eh = refs[e]
        ix = rows_of(np.isin(y, stages) & eh)
        if len(ix) and er[ix].any():
            tr[f"expert:{e}"] = {"base": float(er[ix].mean()),
                                 **{t["word"]: average_precision(score_of(Zs, bases, t["word"], ix), er[ix]) for t in top}}
    res["transfer"] = tr
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--mod", required=True)
    ap.add_argument("--concepts", default="", help="comma list of token-level TARGETS (empty: fit the bases only)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frac", type=float, default=0.25, help="fraction of the pool epochs loaded")
    ap.add_argument("--rotations", type=int, default=5)
    ap.add_argument("--shifts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    bdir = os.path.join(a.dump, "word_bases")
    os.makedirs(bdir, exist_ok=True)
    bp = os.path.join(bdir, f"{a.mod}.npz")
    if os.path.exists(bp):
        z = np.load(bp, allow_pickle=True)
        bases = z["bases"].item()
    else:
        bases = fit_bases(a.dump, a.mod, a.seed)
        np.savez(bp, bases=np.array(bases, dtype=object))
    print(f"[words] {a.mod} bases {({r: b['unmix'].shape for r, b in bases.items()})}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    names = [n for n in a.concepts.split(",") if n]
    if not names:
        print("DONE", flush=True)
        return
    need = sorted({TARGETS[n][0] for n in names} | {e for n in names for e in EXPERT.get(n, ())})
    Zs, y, G, NI, groups, refs = load_coords(a.dump, a.mod, bases, need, a.frac, a.seed)
    gn = np.array(groups)[G]
    print(f"[words] {a.mod}: {len(y)} tokens, groups {groups}, {(time.time() - t0) / 60:.1f} min", flush=True)
    sp = os.path.join(a.dump, f"words_{a.tag}.json")
    out = json.load(open(sp)) if os.path.exists(sp) else {}
    for n in names:
        r = scan_concept(n, Zs, bases, y, gn, NI, refs, a.rotations, a.shifts, a.seed)
        out[f"{n}/{a.mod}"] = r
        json.dump(out, open(sp, "w"), indent=1)
        print(f"[words] {n}/{a.mod}: top {[(t['word'], round(t['ap'], 3)) for t in r.get('top', [])]} "
              f"loo {r.get('loo_ap', np.nan):.3f} base {r.get('base_rate', np.nan):.3f} "
              f"rot95 {r.get('null_rotation', {}).get('q95')} shift95 {r.get('null_shift', {}).get('q95')} "
              f"stab {r.get('stability')} transfer "
              f"{ {g: {k: round(v, 3) for k, v in t.items()} for g, t in r.get('transfer', {}).items()} } "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
