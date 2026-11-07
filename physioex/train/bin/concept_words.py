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
- **nulls** (sanity checks): the same selection on a random rotation of the whitened coordinates
  (``--rotations``) and on references shifted circularly inside the (night, stage) runs (``--shifts``,
  default 20, p = (1 + #null >= obs) / (n + 1)), the whole selection redone on each draw.
- **code word** (``--codes``): prefixes of the RVQ codes (level 0, level 0+1) ranked by weight of evidence
  (hits x log-lift); the word is the smallest top-k (k in 5..200, raw or +-1 token) with held-out AP
  >= 90 % of the best, chosen inside the fold. The family IS the evaluated object; k > 50 = "distributed".
  Prefixes in more than 3 families of a modality are state codes and leave every family at freezing.
- **gates** (``gates``), primary test: the definition must beat the paired baselines (P(ref | stage),
  the E[all] energy words, token RMS of the raw signal, logistic amplitude+stage) with a one-sided paired
  Wilcoxon over the held-out groups, BH over the frozen terms (q 0.05); margin in normalized AP
  (AP - b) / (1 - b) >= 0.05; split-half Spearman stability >= 0.7; held-out AP/base inside the stage
  >= 1.6. The code word replaces the ICA word only when it beats it with a paired p < 0.05.
  Retention = AP / AP of a logistic probe on raw band powers of the token (ceiling) is reported, not gated.

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

from physioex.train.bin.concept_scan import (EXPERT, TARGETS, _group, average_precision, code_lift, family,
                                             load_refs, nb_score)

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


#: bands of the raw-signal features (Hz)
BANDS = {"so": (0.5, 2), "delta": (2, 4), "theta": (4, 7), "alpha": (8, 13), "sigma": (11, 16), "beta": (16, 30)}
FEATS = ("rms", "rms3", "line_length") + tuple(BANDS)


def token_features(x, fs):
    """Raw-signal features of every token of one night (``x``: ``(n, 30 fs)``, consecutive epochs): log RMS
    of the token, log RMS over +-1 token, log line length, log band powers (``BANDS``) of the 1.5 s centred
    on the token (Hann). The amplitude baselines and the signal ceiling of the gates."""
    from numpy.lib.stride_tricks import sliding_window_view

    tok = int(round(0.5 * fs))
    v = x.reshape(-1).astype(np.float32)
    t = v.reshape(-1, tok)
    pad = np.concatenate([v[:tok][::-1], v, v[-tok:][::-1]])
    w = sliding_window_view(pad, 3 * tok)[::tok][:len(t)]
    p = np.abs(np.fft.rfft(w * np.hanning(3 * tok).astype(np.float32), axis=1)) ** 2
    f = np.fft.rfftfreq(3 * tok, 1 / fs)
    cols = [np.log(np.sqrt((t ** 2).mean(1)) + 1e-6), np.log(np.sqrt((w ** 2).mean(1)) + 1e-6),
            np.log(np.abs(np.diff(t, axis=1)).mean(1) + 1e-6)]
    cols += [np.log(p[:, (f >= lo) & (f < hi)].sum(1) + 1e-6) for lo, hi in BANDS.values()]
    return np.stack(cols, 1).astype(np.float32)


def load_coords(dump_dir, mod, bases, level_refs, frac=0.25, seed=0, with_codes=False, with_signal=False):
    """Coordinates of a random ``frac`` of the epochs of every night (whole 60-token epochs, the context
    computed on the whole night first), stage, group, night and the token reference masks; with
    ``with_codes`` also the P codes of the same tokens (``(n_tok, L)``); with ``with_signal`` also their
    raw-signal features (:func:`token_features`, last element)."""
    rng = np.random.default_rng(seed + 7)
    Z = {r: [] for r in bases}
    Y, G, NI, groups = [], [], [], []
    R = {r: ([], []) for r in level_refs}
    C, F = [], []
    fs = json.load(open(os.path.join(dump_dir, "meta.json")))["fs"][mod] if with_signal else None
    for i, (key, e) in enumerate(_nights(dump_dir, mod)):
        z = np.load(os.path.join(dump_dir, f"{key}.npz"))
        P = z[f"P_{mod}"].astype(np.float32)
        keep = rng.random(len(P)) < (1.0 if e["unit"] == "dreams" else 2 * frac if e["unit"] == "mass_ss01" else frac)
        if not keep.any():
            continue
        tk = np.repeat(keep, T)
        for r, b in bases.items():
            Z[r].append(transform(P, b, r)[tk])
        if with_codes:
            c = z[f"codes_{mod}_P"]
            C.append(c.reshape(-1, c.shape[-1])[tk].astype(np.int16))
        if with_signal:
            F.append(token_features(z[f"x_{mod}"], fs)[tk])
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
    out = ({r: np.concatenate(v) for r, v in Z.items()}, np.concatenate(Y), np.concatenate(G), np.concatenate(NI),
           groups, {r: (np.concatenate(a), np.concatenate(h)) for r, (a, h) in R.items()})
    out = out + (np.concatenate(C),) if with_codes else out
    return out + (np.concatenate(F),) if with_signal else out


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


def shift_within_runs(ref, night, y, rng):
    """Circular shift of the reference inside every run of consecutive tokens of the same night and stage:
    an event moves in time but stays in its stage (the null of the gates, 08.11.2026)."""
    out = np.empty_like(ref)
    cut = np.flatnonzero((np.diff(night) != 0) | (np.diff(y) != 0)) + 1
    e = np.r_[0, cut, len(ref)]
    for a, b in zip(e[:-1], e[1:]):
        out[a:b] = np.roll(ref[a:b], rng.integers(1, max(2, b - a)))
    return out


def null_p(obs, null):
    """Permutation p-value ``(1 + #(null >= obs)) / (n + 1)``."""
    null = [v for v in null if np.isfinite(v)]
    return float((1 + sum(v >= obs for v in null)) / (len(null) + 1))


def per_group_ap(s, r, key, groups):
    return {g: float(average_precision(s[key == g], r[key == g])) for g in groups if r[key == g].any()}


def paired_p(a, b):
    """One-sided Wilcoxon signed-rank p that the per-group APs ``a`` exceed ``b`` (dicts by group); 1 with
    fewer than 3 common groups or no difference."""
    from scipy.stats import wilcoxon

    gs = [g for g in a if g in b and np.isfinite(a[g]) and np.isfinite(b[g])]
    d = np.array([a[g] - b[g] for g in gs])
    if len(d) < 3 or not np.any(d):
        return 1.0
    return float(wilcoxon(d, alternative="greater").pvalue)


def nap_margin(a, b, base):
    """Mean over the common groups of the normalized-AP difference ``(a - b) / (1 - base)``."""
    gs = [g for g in a if g in b and g in base and np.isfinite(a[g]) and np.isfinite(b[g])]
    return float(np.mean([(a[g] - b[g]) / (1 - base[g]) for g in gs])) if gs else 0.0


def within_stage(hs, r, y, key, stages):
    """Held-out AP / base inside every (group, stage) of the stratum, weighted by the reference."""
    ws, wt = 0.0, 0.0
    ok = np.isfinite(hs)
    for g in np.unique(key):
        for st in stages:
            m = ok & (key == g) & (y == st)
            if r[m].any() and not r[m].all():
                ws += r[m].sum() * average_precision(hs[m], r[m]) / r[m].mean()
                wt += r[m].sum()
    return float(ws / wt) if wt else None


def _sub_rows(n_tok, rng, n_rows=800):
    rows = np.sort(rng.choice(n_tok // T, min(n_rows, n_tok // T), replace=False))
    return (rows[:, None] * T + np.arange(T)).reshape(-1)


def split_half(gs, score_fn, sub, rng, reps=5):
    """Stability on disjoint data: Spearman correlation, on the tokens ``sub``, of the scores of the
    definition fitted on two random disjoint halves of the groups (mean over ``reps`` splits).
    ``score_fn(groups, sub) -> scores``."""
    from scipy.stats import spearmanr

    out = []
    for _ in range(reps):
        p = list(rng.permutation(gs))
        h = len(p) // 2
        if h < 1:
            break
        a, b = score_fn(set(p[:h]), sub), score_fn(set(p[h:]), sub)
        rho = spearmanr(a, b).correlation if np.std(a) > 0 and np.std(b) > 0 else 0.0
        out.append(float(rho) if np.isfinite(rho) else 0.0)
    return float(np.mean(out)) if out else 0.0


# ---------------------------------------------------------------------------- baselines
def _stage_heldout(y, r, key, groups):
    """Leave-one-group-out P(reference | stage)."""
    out = np.full(len(r), np.nan)
    for g in groups:
        tr, te = key != g, key == g
        rate = np.zeros(int(y.max()) + 1)
        for s in np.unique(y[tr]):
            rate[s] = r[tr & (y == s)].mean()
        out[te] = rate[y[te]]
    return out


def _fold_logistic(X, r, key, groups, max_fit=200_000, seed=0):
    """Leave-one-group-out logistic regression on standardized features: held-out decision value."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    out = np.full(len(r), np.nan)
    for g in groups:
        tr, te = np.flatnonzero(key != g), key == g
        if not r[te].any() or not r[tr].any() or r[tr].all():
            continue
        if len(tr) > max_fit:
            tr = rng.choice(tr, max_fit, replace=False)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = LogisticRegression(max_iter=300).fit((X[tr] - mu) / sd, r[tr])
        out[te] = m.decision_function((X[te] - mu) / sd)
    return out


def baselines(Zs, bases, names, idx, r, key, groups, y, A, gs, F=None):
    """Held-out AP per group of the baselines a definition must beat, and of the raw-signal ceiling.

    Rows: ``stadio`` (P(reference | stage) on the other groups); every ICA ``E[all]`` word (total energy of
    the quantized P, no fit); with the raw-signal features ``F`` (:func:`token_features`) the token RMS
    (raw and over +-1 token); ``amp+stadio``, a logistic regression on the stage and those amplitudes
    fitted on the other groups. The baseline of a held-out group is the row chosen on the other groups
    (as the ICA word). Ceiling: logistic regression on all the raw-signal features (reported as the
    retention of the definition, not a gate)."""
    rows = {"stadio": per_group_ap(_stage_heldout(y, r, key, groups), r, key, groups)}
    amp_names = [n for n in names if ":E[all]" in n]
    for n in amp_names:
        j = names.index(n)
        rows[n] = {g: float(A[j, i]) for i, g in enumerate(gs)}
    st = np.stack([y == s for s in np.unique(y)], 1).astype(np.float32)
    amp = [np.log(score_of(Zs, bases, n, idx) + 1e-6) for n in amp_names if not n.endswith("rel")]
    if F is not None:
        rows["rms"] = per_group_ap(F[:, 0], r, key, groups)
        rows["rms3"] = per_group_ap(F[:, 1], r, key, groups)
        amp = [F[:, 0], F[:, 1]] + amp
    if amp:
        rows["amp+stadio"] = per_group_ap(_fold_logistic(np.column_stack([st] + amp), r, key, groups), r, key, groups)
    keys = list(rows)
    M = np.array([[rows[k].get(g, np.nan) for g in gs] for k in keys])
    held, picks = loo_choice(M)
    base = {"per_group": dict(zip(gs, map(float, held))), "loo_ap": float(np.nanmean(held)),
            "picks": [keys[p] for p in picks], "rows_mean": {k: float(np.nanmean(M[i])) for i, k in enumerate(keys)}}
    ceil = None
    if F is not None:
        c = per_group_ap(_fold_logistic(F, r, key, groups), r, key, groups)
        ceil = {"per_group": c, "loo_ap": float(np.nanmean(list(c.values())))}
    return base, ceil


# ---------------------------------------------------------------------------- code words
def code_score(C, lift, win):
    """Naive-Bayes code score (sum of the log-lifts of the P codes of a token) on whole 60-token rows,
    averaged over +-1 token when ``win == 3`` (the ``code`` words of the lexicon of 08.11.2026, kept to
    read it; the gates now use the compact words below)."""
    s = nb_score(C, lift)
    return _rows_mean3(s) if win == 3 else s


#: compact code words
#: sizes tried for the compact code word (number of prefixes)
KS = (5, 10, 20, 50, 100, 200)


def prefix_keys(C, n_codes):
    """Prefix ids of every token: level 0 = first RVQ code; level 1 = first two codes (``c0 * n_codes + c1``)."""
    k0 = C[:, 0].astype(np.int64)
    return [k0] if C.shape[1] < 2 else [k0, k0 * n_codes + C[:, 1].astype(np.int64)]


def prefix_table(K, r, prior=20.0, min_n=50, exclude=None):
    """Enriched prefixes, sorted by weight of evidence (hits x log-lift): ``[(level, prefix, log_lift, woe)]``
    for the prefixes with >= ``min_n`` uses and a lift above 1 (shrunk with ``prior`` pseudo-uses);
    ``exclude``: ``{(level, prefix)}`` left out (state codes)."""
    b = max(float(r.mean()), 1e-9)
    rows = []
    for lvl, k in enumerate(K):
        u, inv = np.unique(k, return_inverse=True)
        n = np.bincount(inv)
        h = np.bincount(inv, weights=r)
        ll = np.log((h + prior * b) / (n + prior) / b)
        ok = (n >= min_n) & (ll > 0)
        if exclude:
            ok &= ~np.isin(u, [p for l2, p in exclude if l2 == lvl])
        rows += [(lvl, int(p), float(l_), float(hh * l_)) for p, l_, hh in zip(u[ok], ll[ok], h[ok])]
    rows.sort(key=lambda t: -t[3])
    return rows


def compact_word(rows, k):
    """The top ``k`` prefixes as ``{level: (sorted prefixes, log-lifts)}``."""
    per = {}
    for lvl, p, ll, _ in rows[:k]:
        per.setdefault(lvl, []).append((p, ll))
    return {l: (np.array([a for a, _ in sorted(v)], np.int64), np.array([b for _, b in sorted(v)], np.float32))
            for l, v in per.items()}


def compact_score(K, word, win):
    """Score of a token: the largest log-lift among its prefixes in the word (0 outside the word), averaged
    over +-1 token with ``win == 3`` (as ``Lexicon`` ``code_prefix`` words)."""
    v = np.zeros(len(K[0]), np.float32)
    for lvl, (keys, w) in word.items():
        if lvl >= len(K) or not len(keys):
            continue
        pos = np.searchsorted(keys, K[lvl]).clip(0, len(keys) - 1)
        v = np.maximum(v, np.where(keys[pos] == K[lvl], w[pos], np.float32(0)))
    return _rows_mean3(v) if win == 3 else v


def compact_heldout(K, r, key, groups, exclude=None, ks=KS, keep_scores=True):
    """Leave-one-group-out compact code word. Per held-out group: the prefix table of the other groups and
    every (k, win) scored on the group; the (k, win) used for the group is chosen on the other groups (the
    smallest k with mean AP >= 90 % of the best, then the better win). Returns the held-out AP per group,
    the held-out token scores, the (k, win) chosen on all the groups and the mean AP of every (k, win)."""
    cfg = [(k, w) for k in ks for w in (1, 3)]
    gs = [g for g in groups if r[key == g].any()]
    A = np.full((len(cfg), len(gs)), np.nan)
    S = {}
    for j, g in enumerate(gs):
        tr, te = key != g, key == g
        if not r[tr].any():
            continue
        rows = prefix_table([k[tr] for k in K], r[tr], exclude=exclude)
        Kte = [k[te] for k in K]
        for i, (k, w) in enumerate(cfg):
            s = compact_score(Kte, compact_word(rows, k), w)
            A[i, j] = average_precision(s, r[te])
            if keep_scores:
                S[i, j] = s

    def pick(cols):
        m = np.nanmean(A[:, cols], 1) if cols else np.full(len(cfg), np.nan)
        if not np.isfinite(m).any():
            return 0
        ok = np.flatnonzero(m >= 0.9 * np.nanmax(m))
        kmin = min(cfg[i][0] for i in ok)
        return max((i for i in ok if cfg[i][0] == kmin), key=lambda i: m[i])

    held, hs = {}, np.full(len(r), np.nan)
    for j, g in enumerate(gs):
        i = pick([c for c in range(len(gs)) if c != j] or [j])
        held[g] = float(A[i, j])
        if keep_scores and (i, j) in S:
            hs[key == g] = S[i, j]
    with np.errstate(all="ignore"):
        curve = {f"{k}/{w}": float(np.nanmean(A[i])) for i, (k, w) in enumerate(cfg)}
    return held, hs, cfg[pick(list(range(len(gs))))], curve


def decode_prefix(lvl, p, n_codes):
    return [int(p)] if lvl == 0 else list(map(int, divmod(int(p), n_codes)))


def nb_per_level(C, r, key, groups, n_codes):
    """Held-out AP of naive Bayes over the first 1..L RVQ levels (where the code information saturates)."""
    out = []
    for l in range(1, C.shape[1] + 1):
        held = []
        for g in groups:
            tr, te = key != g, key == g
            if r[te].any() and r[tr].any():
                held.append(average_precision(nb_score(C[te], code_lift(C[tr], r[tr], n_codes), range(l)), r[te]))
        out.append(float(np.mean(held)) if held else float("nan"))
    return out


def scan_code_word(C, r, key, groups, n_codes, y, stages, night, rng, n_null=20, exclude=None):
    """The concept's compact code word (top-k prefixes of the first two RVQ codes by weight of evidence;
    k and the +-1 token mean chosen inside the folds). Held-out AP per group and token scores, null of
    references shifted inside the (night, stage) runs with the whole selection redone, the word fitted
    on all the selection tokens (the family = the evaluated object), naive-Bayes AP per RVQ level,
    split-half stability, held-out AP/base within the stage."""
    K = prefix_keys(C, n_codes)
    held, hs, (k, win), curve = compact_heldout(K, r, key, groups, exclude)
    obs = float(np.nanmean(list(held.values()))) if held else float("nan")
    nulls = [float(np.nanmean(list(compact_heldout(K, shift_within_runs(r, night, y, rng), key, groups, exclude,
                                                   keep_scores=False)[0].values())))
             for _ in range(n_null)]
    rows = prefix_table(K, r, exclude=exclude)
    fam = [{"level": l, "codes": decode_prefix(l, p, n_codes), "log_lift": round(ll, 4)} for l, p, ll, _ in rows[:k]]
    sub = _sub_rows(len(r), rng)

    def fitted(gset, ix):
        m = np.isin(key, list(gset))
        return compact_score([kk[ix] for kk in K], compact_word(prefix_table([kk[m] for kk in K], r[m], exclude=exclude), k), win)

    return {"k": int(k), "win": int(win), "loo_ap": obs, "per_group": held, "curve": curve,
            "null_shift": {"values": nulls, "q95": float(np.quantile(nulls, 0.95)) if nulls else None,
                           "p": null_p(obs, nulls)},
            "family": fam, "family_size": len(fam), "distributed": bool(k > 50),
            "nb_per_level": nb_per_level(C, r, key, groups, n_codes),
            "stability_split": split_half([g for g in groups if r[key == g].any()], fitted, sub, rng),
            "within": within_stage(hs, r, y, key, stages), "_word": compact_word(rows, k)}


def scan_concept(name, Zs, bases, y, grp_names, night, refs, n_rot=5, n_shift=20, seed=0, max_tok=600_000,
                 codes=None, n_codes=None, F=None):
    """Both definitions of one concept on its selection set (the stratum of ``TARGETS``), their baselines
    and nulls (see the module docstring and ``gates``)."""
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
    r, yi, ni = ref[idx], y[idx], night[idx]
    Fi = F[idx] if F is not None else None
    A, gs = ap_matrix(Zs, bases, names, idx, r, key, groups)
    gbase = {g: float(r[key == g].mean()) for g in gs}
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
    ica_pg = dict(zip(gs, map(float, held)))
    res = {"concept": name, "ref": ref_name, "stages": list(stages), "selection_set": sel_set,
           "base_rate": float(r.mean()), "group_base": gbase, "n_tok": int(len(idx)), "groups": gs, "top": top,
           "loo_ap": float(np.nanmean(held)), "loo_ap_per_group": ica_pg,
           "stability": float(np.mean([p == best for p in picks])),
           "loo_words": [names[p] for p in picks]}
    # baselines (amplitude, stage) and ceiling (raw signal)
    res["baseline"], res["ceiling"] = baselines(Zs, bases, names, idx, r, key, groups, yi, A, gs, Fi)
    bpg = res["baseline"]["per_group"]
    # ICA nulls: random rotations of the whitened coordinates (reported), references shifted inside the
    # (night, stage) runs with the whole word selection redone (the gate)
    from scipy.stats import ortho_group

    rot = []
    for s in range(n_rot):
        Zr = {rep: (Z[idx] @ ortho_group.rvs(Z.shape[1], random_state=seed + 100 + s)).astype(np.float32)
              for rep, Z in Zs.items()}
        A_r = ap_matrix(Zr, bases, names, np.arange(len(idx)), r, key, groups)[0]
        rot.append(float(np.nanmean(loo_choice(A_r)[0])))
    sh = []
    for s in range(n_shift):
        rs = shift_within_runs(r, ni, yi, rng)
        sh.append(float(np.nanmean(loo_choice(ap_matrix(Zs, bases, names, idx, rs, key, groups)[0])[0])))
    res["null_rotation"] = {"q95": float(np.quantile(rot, 0.95)) if rot else None, "values": rot}
    res["null_shift"] = {"q95": float(np.quantile(sh, 0.95)) if sh else None, "values": sh,
                         "p": null_p(res["loo_ap"], sh)}
    # held-out scores of the ICA word (the word chosen without the group) -> within-stage AP/base
    hs = np.full(len(idx), np.nan)
    for j, g in enumerate(gs):
        m = key == g
        hs[m] = score_of(Zs, bases, names[picks[j]], idx[m])
    res["best_ap_over_base_within_stage"] = within_stage(hs, r, yi, key, stages)
    sub = _sub_rows(len(idx), rng)
    col = {g: j for j, g in enumerate(gs)}

    def ica_fitted(gset, ix):
        cols = [col[g] for g in gset if g in col]
        return score_of(Zs, bases, names[int(np.nanargmax(np.nanmean(A[:, cols], 1)))], idx[ix])

    res["ica"] = {"word": names[best], "loo_ap": res["loo_ap"], "per_group": ica_pg, "amp": "E[all]" in names[best],
                  "null_shift": res["null_shift"], "null_rotation": res["null_rotation"],
                  "stability_split": split_half(gs, ica_fitted, sub, rng),
                  "within": res["best_ap_over_base_within_stage"],
                  "p_vs_base": paired_p(ica_pg, bpg), "margin_nap": nap_margin(ica_pg, bpg, gbase)}
    if res["ceiling"]:
        res["ica"]["retention"] = res["loo_ap"] / max(res["ceiling"]["loo_ap"], 1e-9)
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
    if codes is not None:
        cw = scan_code_word(codes[idx], r, key, groups, n_codes, yi, stages, ni, rng, n_shift)
        word, win = cw.pop("_word"), cw["win"]
        cw["p_vs_base"] = paired_p(cw["per_group"], bpg)
        cw["margin_nap"] = nap_margin(cw["per_group"], bpg, gbase)
        if res["ceiling"]:
            cw["retention"] = cw["loo_ap"] / max(res["ceiling"]["loo_ap"], 1e-9)
        res["code_vs_ica_p"] = paired_p(cw["per_group"], ica_pg)
        ctr = {}
        for g in ("shhs", "mass_ss01", "dreams"):
            ix = rows_of(strat & (grp_names == g))
            if g != sel_set and len(ix) and ref[ix].any():
                ctr[g] = {"base": float(ref[ix].mean()),
                          "ap": average_precision(compact_score(prefix_keys(codes[ix], n_codes), word, win), ref[ix])}
        for e in EXPERT.get(name, ()):
            if e in refs:
                er, eh = refs[e]
                ix = rows_of(np.isin(y, stages) & eh)
                if len(ix) and er[ix].any():
                    ctr[f"expert:{e}"] = {"base": float(er[ix].mean()),
                                          "ap": average_precision(compact_score(prefix_keys(codes[ix], n_codes), word, win), er[ix])}
        cw["transfer"] = ctr
        res["code"] = cw
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
    ap.add_argument("--shifts", type=int, default=20)
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
    Zs, y, G, NI, groups, refs, C, F = load_coords(a.dump, a.mod, bases, need, a.frac, a.seed, with_codes=True,
                                                   with_signal=True)
    gn = np.array(groups)[G]
    n_codes = json.load(open(os.path.join(a.dump, "meta.json")))["n_codes"][a.mod]["P"]
    print(f"[words] {a.mod}: {len(y)} tokens, groups {groups}, {(time.time() - t0) / 60:.1f} min", flush=True)
    sp = os.path.join(a.dump, f"words_{a.tag}.json")
    out = json.load(open(sp)) if os.path.exists(sp) else {}
    for n in names:
        r = scan_concept(n, Zs, bases, y, gn, NI, refs, a.rotations, a.shifts, a.seed, codes=C, n_codes=n_codes, F=F)
        out[f"{n}/{a.mod}"] = r
        json.dump(out, open(sp, "w"), indent=1, default=float)
        i_, c_, b_ = r.get("ica", {}), r.get("code", {}), r.get("baseline", {})
        f3 = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"[words] {n}/{a.mod}: ica {i_.get('word')} ap {f3(i_.get('loo_ap'))} p_null {f3(i_.get('null_shift', {}).get('p'))} "
              f"p_base {f3(i_.get('p_vs_base'))} stab {f3(i_.get('stability_split'))} | code k{c_.get('k')}/w{c_.get('win')} "
              f"ap {f3(c_.get('loo_ap'))} p_null {f3(c_.get('null_shift', {}).get('p'))} p_base {f3(c_.get('p_vs_base'))} "
              f"stab {f3(c_.get('stability_split'))} | baseline {f3(b_.get('loo_ap'))} {sorted(set(b_.get('picks', [])))} "
              f"ceiling {f3((r.get('ceiling') or {}).get('loo_ap'))} base {f3(r.get('base_rate'))} "
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


def candidates(r):
    """The two definitions of a concept from its scan, with the fields of :func:`gates`: the ICA word (best
    by mean AP) and the compact code word."""
    out = {}
    i = r.get("ica")
    if i:
        out["ica"] = {"kind": "ica", "word": i["word"], "loo_ap": i["loo_ap"], "per_group": i["per_group"],
                      "null_p": i["null_shift"]["p"], "null_q95": i["null_shift"]["q95"], "amp": i["amp"],
                      "stability": i["stability_split"], "stable": i["stability_split"] >= 0.7, "within": i["within"],
                      "p_vs_base": i["p_vs_base"], "margin_nap": i["margin_nap"], "retention": i.get("retention")}
    c = r.get("code")
    if c and "k" in c:
        out["code"] = {"kind": "code", "word": f"code:top{c['k']}{'~' if c['win'] == 3 else ''}", "k": c["k"],
                       "win": c["win"], "loo_ap": c["loo_ap"], "per_group": c["per_group"],
                       "null_p": c["null_shift"]["p"], "null_q95": c["null_shift"]["q95"], "amp": False,
                       "stability": c["stability_split"], "stable": c["stability_split"] >= 0.7, "within": c["within"],
                       "p_vs_base": c["p_vs_base"], "margin_nap": c["margin_nap"], "retention": c.get("retention"),
                       "family_size": c["family_size"], "distributed": c["distributed"], "family": c["family"],
                       "curve": c.get("curve"), "nb_per_level": c.get("nb_per_level")}
    return out


def choose(cands, p_code_vs_ica):
    """The code word replaces the ICA word only when it is better on the held-out groups (one-sided paired
    test, p < 0.05): on a tie the simpler ICA word stays."""
    if "code" in cands and ("ica" not in cands or (p_code_vs_ica < 0.05
                                                   and cands["code"]["loo_ap"] > cands["ica"]["loo_ap"])):
        return "code"
    return "ica"


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    if not len(p):
        return p
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def gates(concept, c, q=None):
    """Verdict of one definition (08.11.2026, after the review): do the codes determine the term?

    - ``bocciata``: the held-out AP does not beat references shifted inside the (night, stage) runs with
      the whole selection redone (permutation p > 0.05);
    - ``solo stadio``: held-out AP/base inside the stage < 1.6 (or atonia, a tone level);
    - ``ampiezza``: not better than the amplitude / stage baselines (one-sided paired test over the
      held-out groups, BH ``q`` over the frozen terms > 0.05, or normalized-AP margin <= 0), or an ICA
      word on the total energy ``E[all]``;
    - ``debole``: normalized-AP margin over the baseline < 0.05, or split-half stability < 0.7;
    - ``valida`` otherwise.

    Whether the stager uses the term is a property of the stager and is not part of the verdict."""
    q = c["p_vs_base"] if q is None else q
    if c["null_p"] > 0.05:
        v = "bocciata"
    elif concept in ("atonia",) or (c.get("within") or 0) < 1.6:
        v = "solo stadio"
    elif c["amp"] or q > 0.05 or c["margin_nap"] <= 0:
        v = "ampiezza"
    elif c["margin_nap"] < 0.05 or not c["stable"]:
        v = "debole"
    else:
        v = "valida"
    return {"null_p": c["null_p"], "p_vs_base": c["p_vs_base"], "q_vs_base": float(q), "margin_nap": c["margin_nap"],
            "stable": bool(c["stable"]), "verdict": v}


def _family_set(fam, n_codes):
    return {(f["level"], f["codes"][0] if f["level"] == 0 else f["codes"][0] * n_codes + f["codes"][1]) for f in fam}


def freeze(a):
    from collections import Counter

    from physioex.models.vq_lexicon import Lexicon

    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    fp = os.path.join(a.dump, "fidelity.json")
    fid = json.load(open(fp))["summary"] if os.path.exists(fp) else {}
    lex_bases, pend, overlap = {}, [], {}
    for mod in meta["mods"]:
        wp = os.path.join(a.dump, f"words_{mod}.json")
        if not os.path.exists(wp):
            continue
        wj = json.load(open(wp))
        bases = np.load(os.path.join(a.dump, "word_bases", f"{mod}.npz"), allow_pickle=True)["bases"].item()
        names = [k.split("/")[0] for k, r in wj.items() if r.get("top") and k.endswith(f"/{mod}")]
        need = sorted({TARGETS[n][0] for n in names} | {e for n in names for e in EVENT_EXPERT.get(n, ())}
                      | {e for n in names for e in EXPERT.get(n, ())})
        Zs, y, G, NI, groups, refs, C = load_coords(a.dump, mod, bases, need, a.frac, a.seed, with_codes=True)
        n_codes = meta["n_codes"][mod]["P"]
        gn = np.array(groups)[G]
        for rep, b in bases.items():
            lex_bases[f"{mod}_{rep}"] = {"rep": rep, "mod": mod, "mean": b["mean"], "unmix": b["unmix"], "mix": b["mix"]}
        # state codes: prefixes in the code families of more than 3 concepts of the modality (null passed)
        fams = {n: _family_set(wj[f"{n}/{mod}"]["code"]["family"], n_codes) for n in names
                if wj[f"{n}/{mod}"].get("code", {}).get("null_shift", {}).get("p", 1) <= 0.05}
        cnt = Counter(p for s in fams.values() for p in s)
        state = {p for p, k in cnt.items() if k > 3}
        overlap[mod] = {"state_prefixes": len(state),
                        "jaccard": {f"{x}|{z}": round(len(fams[x] & fams[z]) / max(1, len(fams[x] | fams[z])), 3)
                                    for i_, x in enumerate(sorted(fams)) for z in sorted(fams)[i_ + 1:]}}
        for n in names:
            r = wj[f"{n}/{mod}"]
            ref_name, _, stages, _, sel_set = TARGETS[n]
            ref, has = refs[ref_name]
            strat = np.isin(y, stages) & has
            fit = strat & ((gn == "dreams") if sel_set == "dreams" else ~np.isin(gn, ["shhs", "mass_ss01", "dreams"]))
            rows = np.flatnonzero(fit.reshape(-1, T).all(1))
            idx = (rows[:, None] * T + np.arange(T)).reshape(-1)
            rr, key, yi = ref[idx], (gn[idx] if sel_set == "pool" else NI[idx].astype(str)), y[idx]
            cands = candidates(r)
            p_ci = r.get("code_vs_ica_p", 1.0)
            excl = None
            if "code" in cands and state & fams.get(n, set()):
                excl = state  # the state codes leave the event's definition: selection redone without them
                K = prefix_keys(C[idx], n_codes)
                held, hs, (k, win), _ = compact_heldout(K, rr, key, sorted(set(key)), excl)
                bpg = r["baseline"]["per_group"]
                tab = prefix_table(K, rr, exclude=excl)
                cands["code"].update(
                    loo_ap=float(np.nanmean(list(held.values()))), per_group=held, k=int(k), win=int(win),
                    word=f"code:top{k}{'~' if win == 3 else ''}", within=within_stage(hs, rr, yi, key, stages),
                    p_vs_base=paired_p(held, bpg), margin_nap=nap_margin(held, bpg, r["group_base"]),
                    family=[{"level": l, "codes": decode_prefix(l, p, n_codes), "log_lift": round(ll, 4)}
                            for l, p, ll, _ in tab[:k]],
                    family_size=len(tab[:k]), distributed=bool(k > 50), state_removed=len(state & fams[n]))
                p_ci = paired_p(held, cands["ica"]["per_group"]) if "ica" in cands else 0.0
            win_kind = choose(cands, p_ci)
            cw = cands[win_kind]
            word = cw["word"]
            if win_kind == "code":
                ex_ = excl

                def fit_word(ix):
                    return compact_word(prefix_table(prefix_keys(C[ix], n_codes), ref[ix], exclude=ex_), cw["k"])

                word_c = fit_word(idx)

                def score(ix, w_=None):
                    return compact_score(prefix_keys(C[ix], n_codes), word_c if w_ is None else w_, cw["win"])
            else:
                def score(ix, w_=None):
                    return score_of(Zs, bases, word, ix)
            sc = score(idx)
            q = 1 - r["base_rate"]
            thr = float(np.quantile(sc, q))
            if win_kind == "code":
                if thr <= 0:  # most tokens hold no prefix of the word: fire on the word only
                    thr = float(sc[sc > 0].min()) if (sc > 0).any() else 1e-6
                thr -= 1e-6  # discrete scores: the tied values at the threshold fire whatever the summation order
            held_f1 = []
            for g in sorted(set(key)):
                tr, te = key != g, key == g
                if rr[te].any():
                    if win_kind == "code":  # the word of the fold as well
                        wf = fit_word(idx[tr])
                        s_tr = score(idx[tr], wf)
                        t_ = np.quantile(s_tr, q)
                        t_ = t_ if t_ > 0 else (s_tr[s_tr > 0].min() if (s_tr > 0).any() else 1e-6)
                        held_f1.append(_tok_f1(score(idx[te], wf) >= t_, rr[te]))
                    else:
                        held_f1.append(_tok_f1(sc[te] >= np.quantile(sc[tr], q), rr[te]))
            rec = {"word": word, "definition": win_kind, "threshold": thr, "base_rate": r["base_rate"],
                   "token_f1_loo": float(np.mean(held_f1)), "loo_ap": cw["loo_ap"], "per_group": cw["per_group"],
                   "group_base": r["group_base"], "baseline": r["baseline"], "ceiling": r.get("ceiling"),
                   "retention": cw.get("retention"), "null_q95": cw["null_q95"], "stability": cw["stability"],
                   "ap_over_base_within_stage": cw["within"], "code_vs_ica_p": p_ci,
                   "transfer_ap": r["code"]["transfer"] if win_kind == "code" else r["transfer"]}
            for f in ("k", "win", "family_size", "family", "distributed", "curve", "nb_per_level", "state_removed"):
                if f in cw:
                    rec[f] = cw[f]
            fkey = n if mod == "eeg" or n != "arousal" else f"{n}_{mod}"
            if fkey in fid and FIDELITY.get(fkey, (None,))[0] == mod:
                rec["stager_use"] = fid[fkey]  # L3: a property of the stager, not of the lexicon
            # token F1 at the frozen threshold on the out-of-pool sets; event F1 against the DREAMS experts
            tf = {}
            for g in ("shhs", "mass_ss01", "dreams"):
                m = strat & (gn == g)
                rw = np.flatnonzero(m.reshape(-1, T).all(1))
                ix = (rw[:, None] * T + np.arange(T)).reshape(-1)
                if g != sel_set and len(ix) and ref[ix].any():
                    tf[g] = _tok_f1(score(ix) >= thr, ref[ix])
            rec["transfer_token_f1"] = tf
            ev = {}
            for e in EVENT_EXPERT.get(n, ()):
                er, eh = refs[e]
                per = []
                for night in np.unique(NI[(gn == "dreams") & eh]):
                    ix = np.flatnonzero((NI == night) & eh)
                    if not er[ix].any():
                        continue
                    if sel_set == "dreams":  # threshold (and code word) from the other excerpts
                        oth = np.flatnonzero(fit.reshape(-1, T).all(1) & (NI.reshape(-1, T)[:, 0] != night))
                        oth = (oth[:, None] * T + np.arange(T)).reshape(-1)
                        wf = fit_word(oth) if win_kind == "code" else None
                        so = score(oth, wf)
                        t_ = np.quantile(so, q)
                        if win_kind == "code" and t_ <= 0:
                            t_ = so[so > 0].min() if (so > 0).any() else 1e-6
                        per.append(event_f1(score(ix, wf) >= t_, er[ix]))
                    else:
                        per.append(event_f1(score(ix) >= thr, er[ix]))
                if per:
                    tp = sum(round(p["coverage"] * p["n_ref"]) for p in per)
                    npred, nref = sum(p["n_pred"] for p in per), sum(p["n_ref"] for p in per)
                    pu, co = (tp / npred if npred else 0.0), (tp / nref if nref else 0.0)
                    ev[e] = {"f1": 2 * pu * co / (pu + co) if pu + co else 0.0, "purity": pu, "coverage": co,
                             "n_pred": npred, "n_ref": nref}
            rec["event_f1_experts"] = ev
            name = n if mod == "eeg" or n not in ("arousal", "resp_end", "plm_arousal") else f"{n}_{mod}"
            lw = None
            if win_kind == "code":
                cb = f"{mod}_code_{name}"
                lex_bases[cb] = {"rep": "code_prefix", "mod": mod,
                                 **{f"pref_keys{l}": word_c[l][0] for l in word_c},
                                 **{f"pref_w{l}": word_c[l][1] for l in word_c}}
                lw = {"name": name, "event": name, "basis": cb, "word": word, "kind": "code_prefix",
                      "n_codes": int(n_codes), "win": cw["win"], "side": 1, "threshold": thr,
                      "family_size": cw["family_size"]}
            else:
                lw = lexicon_word(name, word, bases, mod, thr, {})
            pend.append((n, name, mod, stages, win_kind, cands, rec, lw))
    # verdicts: BH over the chosen definitions of all the frozen terms
    qs = bh([p[5][p[4]]["p_vs_base"] for p in pend])
    lex_words, report = [], {}
    for (n, name, mod, stages, win_kind, cands, rec, lw), q in zip(pend, qs):
        ver = {k: gates(n, c, q if k == win_kind else None) for k, c in cands.items()}
        rec.update(ver[win_kind])
        rec["alternative"] = {k: {**{f: v for f, v in c.items() if f in ("word", "loo_ap", "null_q95", "stability",
                                                                          "within", "family_size", "retention")},
                                  **ver[k]} for k, c in cands.items() if k != win_kind}
        report[name] = {"mod": mod, **rec}
        lex_words.append({**lw, "verdict": rec["verdict"], "stages": list(stages), "base_rate": rec["base_rate"]})
        print(f"[freeze] {name}: {rec['verdict']} {win_kind} {rec['word']} ap {rec['loo_ap']:.3f} "
              f"base-line {rec['baseline']['loo_ap']:.3f} q {q:.3g} nAP+ {rec['margin_nap']:.3f} "
              f"stab {rec['stability']:.2f} within {rec['ap_over_base_within_stage'] or 0:.2f} "
              f"ret {rec['retention'] or float('nan'):.2f} tokF1 {rec['token_f1_loo']:.3f}", flush=True)
    lex = Lexicon(lex_bases, lex_words, {"continuous": False, "encoder": meta["combo_dir"], "stager": meta["stager"],
                                         "fitted_on": "pool nights of the concept_scan dump (label-free bases)",
                                         "threshold": "quantile of the pool stratum at the concept base rate",
                                         "gates": "concept_words.gates, 08.11.2026 (baselines, BH, split-half)",
                                         "event_rules": {w["name"]: {"gap": 1, "min_tok": 1} for w in lex_words}})
    out = a.out or os.path.join(meta["combo_dir"], "lexicon_v2")
    lex.save(out)
    json.dump(report, open(os.path.join(out, "gates.json"), "w"), indent=1, default=float)
    json.dump(overlap, open(os.path.join(out, "code_overlap.json"), "w"), indent=1)
    # round trip: the saved lexicon fires as the scores of the scan on one pool night (rows, continuous=False)
    lx = Lexicon.load(out)
    key = next(k for k, e in sorted(meta["nights"].items()) if e["unit"] != "dreams" and set(meta["mods"]) <= set(e["mods"]))
    z = np.load(os.path.join(a.dump, f"{key}.npz"))
    pqs = {m: z[f"P_{m}"].astype(np.float32) for m in meta["mods"]}
    cds = {m: z[f"codes_{m}_P"] for m in meta["mods"]}
    fired = lx.fire(pqs, continuous=False, codes=cds)
    bad = []
    for w in lex_words:
        if w["kind"] == "code_prefix":
            c = cds[lx.mod_of(w["basis"])]
            b = lex_bases[w["basis"]]
            wd = {l: (b[f"pref_keys{l}"], b[f"pref_w{l}"]) for l in (0, 1) if f"pref_keys{l}" in b}
            s_ = compact_score(prefix_keys(c.reshape(-1, c.shape[-1]), w["n_codes"]), wd, w["win"])
            if ((s_ >= w["threshold"]) != fired[w["name"]]).mean() > 1e-3:
                bad.append(w["name"])
            continue
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
