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

L3 fidelity (GPU): the nights are encoded again (the dump keeps no continuous N / E); on the epochs of the
concept's stratum where its top word fires (threshold: the quantile at the concept's base rate) the P
tokens take donor tokens of the same stage from epochs without the word; the drop of p(expected stage)
in the multi stager is compared with 20 swaps of as many random tokens of the same epochs.

    python -m physioex.train.bin.concept_words fidelity --dump DUMP --splits SPLITS --cache-dir CACHE
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
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "fidelity":
        f = argparse.ArgumentParser(description="L3: stager fidelity of the top word of every concept (GPU)")
        f.add_argument("--dump", required=True)
        f.add_argument("--splits", required=True)
        f.add_argument("--cache-dir", default=None)
        f.add_argument("--per-unit", type=int, default=5)
        f.add_argument("--controls", type=int, default=20)
        f.add_argument("--seed", type=int, default=0)
        return fidelity(f.parse_args(argv[1:]))
    if argv and argv[0] == "freeze":
        f = argparse.ArgumentParser(description="freeze the multimodal lexicon (top word per concept, gates)")
        f.add_argument("--dump", required=True)
        f.add_argument("--out", default=None, help="default: <combo_dir>/lexicon_v2")
        f.add_argument("--frac", type=float, default=0.25)
        f.add_argument("--seed", type=int, default=0)
        return freeze(f.parse_args(argv[1:]))
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



# ---------------------------------------------------------------------------- L3 fidelity (GPU)
#: concept -> (modality, stages the stager should lose when the concept's tokens are swapped out)
FIDELITY = {"spindle": ("eeg", (2,)), "k_complex": ("eeg", (2,)), "slow_wave": ("eeg", (3,)),
            "alpha": ("eeg", (0,)), "rem_eye": ("eog", (4,)), "atonia": ("emg", (4,)),
            "arousal": ("eeg", (0, 1)), "arousal_emg": ("emg", (0, 1))}


def swap_tokens(P, y, fire, pos, rng):
    """Copy of ``P`` (``(n, T, d)``) where the tokens ``pos`` (flat indices) take a random donor token of
    an epoch of the same stage in which the word does not fire (same night)."""
    Q = P.copy().reshape(-1, P.shape[-1])
    ep_fire = fire.reshape(-1, T).any(1)
    yt = np.repeat(y, T)
    donors = {s: np.flatnonzero((yt == s) & ~np.repeat(ep_fire, T)) for s in np.unique(y)}
    for s in np.unique(yt[pos]):
        p = pos[yt[pos] == s]
        d = donors.get(s, np.zeros(0, int))
        if len(d):
            Q[p] = Q[rng.choice(d, len(p))]
    return Q.reshape(P.shape)


def fidelity_night(lat, logp_fn, mods, bases, word, mod, stages, lose, base_rate, n_ctrl, rng):
    """Delta p(lose) on the epochs of ``stages`` where ``word`` fires, swapping its tokens vs swapping the same
    number of random tokens of the same epochs."""
    P = lat[mod]["P"].astype(np.float32)
    y = lat["eeg"]["y"]
    rep = word.split(":", 1)[0]
    s = word_score(transform(P, bases[rep], rep), word, bases[rep]["groups"])
    strat = np.repeat(np.isin(y, stages), T)
    if not strat.any():
        return None
    thr = np.quantile(s[strat], 1 - base_rate)
    fire = (s >= thr) & strat
    pos = np.flatnonzero(fire)
    eps = np.unique(pos // T)
    if len(eps) < 5:
        return None
    p0 = np.exp(logp_fn(lat).astype(np.float32))

    def drop(Pn):
        lat2 = {**lat, mod: {**lat[mod], "P": Pn.astype(np.float16)}}
        p1 = np.exp(logp_fn(lat2).astype(np.float32))
        d = p0[eps][:, list(lose)].sum(1) - p1[eps][:, list(lose)].sum(1)
        return float(d.mean()), float((p0[eps].argmax(1) != p1[eps].argmax(1)).mean())

    real = drop(swap_tokens(P, y, fire, pos, rng))
    ctrl = []
    for _ in range(n_ctrl):
        cpos = np.concatenate([e * T + rng.choice(T, int((pos // T == e).sum()), replace=False) for e in eps])
        ctrl.append(drop(swap_tokens(P, y, fire, cpos, rng)))
    return {"n_epochs": int(len(eps)), "n_tokens": int(len(pos)), "dp": real[0], "flip": real[1],
            "ctrl_dp": [c[0] for c in ctrl], "ctrl_flip": [c[1] for c in ctrl]}


def fidelity(a):
    import torch

    from physioex.data.merged import build_test
    from physioex.models.recon_eval import channel_pipelines
    from physioex.train.bin.concept_scan import _night_arrays, _pick, _stage_logp
    from physioex.train.bin.pretrain_combo import Nights, _cache_dir, load_s2, load_stagers

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    mods = meta["mods"]
    models, args0 = {}, {}
    for m in mods:
        run, models[m] = load_s2(os.path.join(meta["combo_dir"], m, "s2"), m, dev)
        args0[m] = run["args"]
    stagers = load_stagers(os.path.join(meta["combo_dir"], meta["stager"]), mods, dev)
    half_of = {m: args0[m]["seq"] // 2 for m in mods}
    bases = {m: np.load(os.path.join(a.dump, "word_bases", f"{m}.npz"), allow_pickle=True)["bases"].item() for m in mods}
    wj = {m: json.load(open(os.path.join(a.dump, f"words_{m}.json"))) for m in mods}
    plan = {}
    for c, (m, lose) in FIDELITY.items():
        name = c.replace("_emg", "")
        r = wj[m].get(f"{name}/{m}")
        if r and r.get("top"):
            plan[c] = (m, r["top"][0]["word"], TARGETS[name][2], lose, r["base_rate"])
    print(f"[fidelity] words { {c: v[1] for c, v in plan.items()} }", flush=True)
    by_unit = {}
    for k, e in meta["nights"].items():
        if e["unit"] != "dreams" and set(mods) <= set(e["mods"]):
            by_unit.setdefault(e["unit"], []).append(k.split("/", 1)[1])
    units = sorted(by_unit)
    tests = {m: build_test(a.splits, [args0[m]["channel"]], channel_pipelines(args0[m]), args0[m]["seq"],
                           _cache_dir(a.cache_dir, args0[m]), units=units, memmap_cache_size=4,
                           trim_mode="main_period") for m in mods}
    index = {m: {(tests[m].unit_names[d2], s2): d2 for d2, s2, _, _ in tests[m].test} for m in mods}
    rng = np.random.default_rng(a.seed)
    out = {c: {} for c in plan}
    t0 = time.time()
    for unit in units:
        for sid in _pick(by_unit[unit], a.per_unit, a.seed):
            readers = {m: Nights(tests[m].multi._datasets[index[m][unit, sid]], args0[m])
                       for m in mods if (unit, sid) in index[m]}
            if set(readers) != set(mods):
                continue
            _, lat, ts, y = _night_arrays(models, readers, {m: sid for m in mods}, mods, half_of)
            if set(lat) != set(mods) or any(len(lat[m]["y"]) != len(ts) for m in mods):
                continue

            def logp_fn(L):
                return _stage_logp(stagers, L, mods, dev)["multi"]

            for c, (m, word, stages, lose, br) in plan.items():
                r = fidelity_night(lat, logp_fn, mods, bases[m], word, m, stages, lose, br, a.controls, rng)
                if r is not None:
                    out[c][f"{unit}/{sid}"] = r
            print(f"[fidelity] {unit} {sid} {(time.time() - t0) / 60:.1f} min", flush=True)
    summary = {}
    for c, nights in out.items():
        if not nights:
            continue
        dp = np.array([v["dp"] for v in nights.values()])
        cm = np.array([np.mean(v["ctrl_dp"]) for v in nights.values()])
        above = np.mean([v["dp"] > np.quantile(v["ctrl_dp"], 0.95) for v in nights.values()])
        summary[c] = {"word": plan[c][1], "n_nights": len(dp), "dp_mean": float(dp.mean()), "ctrl_dp_mean": float(cm.mean()),
                      "dp_minus_ctrl_median": float(np.median(dp - cm)), "nights_above_ctrl95": float(above),
                      "flip_mean": float(np.mean([v["flip"] for v in nights.values()])),
                      "ctrl_flip_mean": float(np.mean([np.mean(v["ctrl_flip"]) for v in nights.values()]))}
        print(f"[fidelity] {c}: {summary[c]}", flush=True)
    json.dump({"summary": summary, "nights": out}, open(os.path.join(a.dump, "fidelity.json"), "w"), indent=1)
    print("DONE", flush=True)


# ---------------------------------------------------------------------------- freeze (CPU)
#: concepts whose word is read with the experts' events (DREAMS), tolerance +-1 token
EVENT_EXPERT = {"spindle": ("spindle_vis1", "spindle_vis2"), "spindle_expert": ("spindle_vis1", "spindle_vis2"),
                "k_complex": ("k_complex_vis1", "k_complex_vis2")}


def token_runs_iv(m, tol=1):
    """Runs of a token mask (gaps of one token merged) as ``(start, end)`` token intervals widened by ``tol``."""
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    br = np.diff(idx) > 2
    st, en = np.r_[idx[0], idx[1:][br]], np.r_[idx[:-1][br], idx[-1]]
    return [(a - tol, b + 1 + tol) for a, b in zip(st, en)]


def event_f1(pred, ref, tol=1):
    """Event F1 with one-to-one matching by overlap (``vq_event_lexicon2.match_count`` rule)."""
    P, R = token_runs_iv(pred, tol), token_runs_iv(ref, 0)
    tp = 0
    if P and R:
        Pa, Ra = np.asarray(P), np.asarray(R)
        ov = np.minimum(Pa[:, None, 1], Ra[None, :, 1]) - np.maximum(Pa[:, None, 0], Ra[None, :, 0])
        i, j = np.nonzero(ov > 0)
        up, ur = set(), set()
        for k in np.argsort(-ov[i, j], kind="stable"):
            if i[k] in up or j[k] in ur:
                continue
            up.add(i[k]); ur.add(j[k]); tp += 1
    p = tp / len(P) if P else 0.0
    r = tp / len(R) if R else 0.0
    return {"f1": 2 * p * r / (p + r) if p + r else 0.0, "purity": p, "coverage": r, "n_pred": len(P), "n_ref": len(R)}


def _tok_f1(m, r):
    h = np.count_nonzero(m & r)
    return 2 * h / (np.count_nonzero(m) + np.count_nonzero(r)) if m.any() or r.any() else 0.0


def lexicon_word(name, word, bases_m, mod, thr, extra):
    """Our word name (``pq3:E[d6+d7]~``) as a :class:`physioex.models.vq_lexicon.Lexicon` word."""
    rep, w = word.split(":", 1)
    basis = f"{mod}_{rep}"
    out = {"name": name, "event": name, "basis": basis, "word": word}
    if w.startswith("E["):
        g, kind = w[2:w.index("]")], w[w.index("]") + 1:]
        out.update(kind="energy", dirs=[int(d) for d in bases_m[rep]["groups"][g]], side=1, threshold=float(thr),
                   win=3 if kind == "~" else 1, **({"rel": 10} if kind == "rel" else {}))
    else:
        d = int(w.strip("|+-d"))
        if w.startswith("|"):
            out.update(kind="folded", dirs=[d], side=1, threshold=float(thr))
        elif w.endswith("+"):
            out.update(kind="signed", dirs=[d], side=1, threshold=float(thr))
        else:  # score -z >= thr  <=>  z <= -thr
            out.update(kind="signed", dirs=[d], side=-1, threshold=float(-thr))
    out.update(extra)
    return out


def gates(r, fid):
    """Verdict of one concept from the scan (L1, L2) and the fidelity summary (L3)."""
    rot, sh = r["null_rotation"]["q95"], r["null_shift"]["q95"]
    amp = "E[all]" in r["top"][0]["word"]
    l1 = r["loo_ap"] > sh and (amp or r["loo_ap"] > rot)
    l1_margin = r["loo_ap"] - max(sh, rot if not amp else sh)
    l2 = r["stability"] >= 7 / 9
    l3 = None
    if fid is not None:
        l3 = bool(fid["nights_above_ctrl95"] >= 0.9 and fid["dp_minus_ctrl_median"] > 0)
    within = r.get("best_ap_over_base_within_stage") or 0
    if not l1:
        verdict = "bocciata"
    elif r["concept"] in ("atonia",) or within < 1.6:
        verdict = "solo stadio"
    elif amp:
        verdict = "ampiezza"
    elif l1_margin < 0.05 or not l2:
        verdict = "debole"
    else:
        verdict = "valida"
    return {"L1": bool(l1), "L1_margin": float(l1_margin), "L2": bool(l2), "L3": l3, "verdict": verdict}


def freeze(a):
    from physioex.models.vq_lexicon import Lexicon

    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    fp = os.path.join(a.dump, "fidelity.json")
    fid = json.load(open(fp))["summary"] if os.path.exists(fp) else {}
    rng = np.random.default_rng(a.seed)
    lex_bases, lex_words, report = {}, [], {}
    for mod in meta["mods"]:
        wp = os.path.join(a.dump, f"words_{mod}.json")
        if not os.path.exists(wp):
            continue
        wj = json.load(open(wp))
        bases = np.load(os.path.join(a.dump, "word_bases", f"{mod}.npz"), allow_pickle=True)["bases"].item()
        names = [k.split("/")[0] for k, r in wj.items() if r.get("top")]
        need = sorted({TARGETS[n][0] for n in names} | {e for n in names for e in EVENT_EXPERT.get(n, ())}
                      | {e for n in names for e in EXPERT.get(n, ())})
        Zs, y, G, NI, groups, refs = load_coords(a.dump, mod, bases, need, a.frac, a.seed)
        gn = np.array(groups)[G]
        for rep, b in bases.items():
            lex_bases[f"{mod}_{rep}"] = {"rep": rep, "mod": mod, "mean": b["mean"], "unmix": b["unmix"], "mix": b["mix"]}
        for n in names:
            r = wj[f"{n}/{mod}"]
            ref_name, _, stages, _, sel_set = TARGETS[n]
            ref, has = refs[ref_name]
            strat = np.isin(y, stages) & has
            fit = strat & ((gn == "dreams") if sel_set == "dreams" else ~np.isin(gn, ["shhs", "mass_ss01", "dreams"]))
            word = r["top"][0]["word"]
            rows = np.flatnonzero(fit.reshape(-1, T).all(1))
            idx = (rows[:, None] * T + np.arange(T)).reshape(-1)
            sc = score_of(Zs, bases, word, idx)
            rr, key = ref[idx], (gn[idx] if sel_set == "pool" else NI[idx].astype(str))
            q = 1 - r["base_rate"]
            thr = float(np.quantile(sc, q))
            held = []
            for g in sorted(set(key)):
                tr, te = key != g, key == g
                if rr[te].any():
                    held.append(_tok_f1(sc[te] >= np.quantile(sc[tr], q), rr[te]))
            rec = {"word": word, "threshold": thr, "base_rate": r["base_rate"], "token_f1_loo": float(np.mean(held)),
                   "loo_ap": r["loo_ap"], "null_rotation_q95": r["null_rotation"]["q95"],
                   "null_shift_q95": r["null_shift"]["q95"], "stability": r["stability"],
                   "ap_over_base_within_stage": r.get("best_ap_over_base_within_stage"), "transfer_ap": r["transfer"]}
            fkey = n if mod == "eeg" or n != "arousal" else f"{n}_{mod}"
            rec.update(gates(r, fid.get(fkey) if (fkey in FIDELITY and FIDELITY[fkey][0] == mod) else None))
            if fkey in fid and FIDELITY.get(fkey, (None,))[0] == mod:
                rec["fidelity"] = fid[fkey]
            # token F1 at the frozen threshold on the out-of-pool sets; event F1 against the DREAMS experts
            tf = {}
            for g in ("shhs", "mass_ss01", "dreams"):
                m = strat & (gn == g)
                rw = np.flatnonzero(m.reshape(-1, T).all(1))
                ix = (rw[:, None] * T + np.arange(T)).reshape(-1)
                if g != sel_set and len(ix) and ref[ix].any():
                    tf[g] = _tok_f1(score_of(Zs, bases, word, ix) >= thr, ref[ix])
            rec["transfer_token_f1"] = tf
            ev = {}
            for e in EVENT_EXPERT.get(n, ()):
                er, eh = refs[e]
                per = []
                for night in np.unique(NI[(gn == "dreams") & eh]):
                    ix = np.flatnonzero((NI == night) & eh)
                    if not er[ix].any():
                        continue
                    if sel_set == "dreams":  # threshold from the other excerpts
                        oth = np.flatnonzero(fit & (NI != night))
                        t_ = np.quantile(score_of(Zs, bases, word, oth), q)
                    else:
                        t_ = thr
                    per.append(event_f1(score_of(Zs, bases, word, ix) >= t_, er[ix]))
                if per:
                    tp = sum(round(p["coverage"] * p["n_ref"]) for p in per)
                    npred, nref = sum(p["n_pred"] for p in per), sum(p["n_ref"] for p in per)
                    pu, co = (tp / npred if npred else 0.0), (tp / nref if nref else 0.0)
                    ev[e] = {"f1": 2 * pu * co / (pu + co) if pu + co else 0.0, "purity": pu, "coverage": co,
                             "n_pred": npred, "n_ref": nref}
            rec["event_f1_experts"] = ev
            name = n if mod == "eeg" or n not in ("arousal", "resp_end", "plm_arousal") else f"{n}_{mod}"
            report[name] = {"mod": mod, **rec}
            lex_words.append(lexicon_word(name, word, bases, mod, thr,
                                          {"verdict": rec["verdict"], "stages": list(stages), "base_rate": r["base_rate"]}))
            print(f"[freeze] {name}: {rec['verdict']} thr {thr:.3f} tokF1 {rec['token_f1_loo']:.3f} "
                  f"transfer {({k: round(v, 3) for k, v in tf.items()})} events {({k: round(v['f1'], 3) for k, v in ev.items()})}",
                  flush=True)
    lex = Lexicon(lex_bases, lex_words, {"continuous": False, "encoder": meta["combo_dir"], "stager": meta["stager"],
                                         "fitted_on": "pool nights of the concept_scan dump (label-free bases)",
                                         "threshold": "quantile of the pool stratum at the concept base rate",
                                         "event_rules": {w["name"]: {"gap": 1, "min_tok": 1} for w in lex_words}})
    out = a.out or os.path.join(meta["combo_dir"], "lexicon_v2")
    lex.save(out)
    json.dump(report, open(os.path.join(out, "gates.json"), "w"), indent=1)
    # round trip: the saved lexicon fires as the scores of the scan on one pool night (rows, continuous=False)
    lx = Lexicon.load(out)
    key = next(k for k, e in sorted(meta["nights"].items()) if e["unit"] != "dreams" and set(meta["mods"]) <= set(e["mods"]))
    z = np.load(os.path.join(a.dump, f"{key}.npz"))
    pqs = {m: z[f"P_{m}"].astype(np.float32) for m in meta["mods"]}
    fired = lx.fire(pqs, continuous=False)
    bad = []
    for w in lex_words:
        mod, rep = w["basis"].split("_")
        b = np.load(os.path.join(a.dump, "word_bases", f"{mod}.npz"), allow_pickle=True)["bases"].item()
        s_ = word_score(transform(pqs[mod], b[rep], rep), w["word"], b[rep]["groups"])
        thr = w["threshold"] if w["side"] > 0 or w["kind"] != "signed" else -w["threshold"]
        if ((s_ >= thr) != fired[w["name"]]).mean() > 1e-3:
            bad.append(w["name"])
    print(f"[freeze] lexicon -> {out}, {len(lex_words)} words, round trip {'OK' if not bad else 'MISMATCH ' + str(bad)}",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
