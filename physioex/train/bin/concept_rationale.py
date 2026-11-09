"""Short rationale tests of the lexicon terms (plan of 08.11.2026, second part).

For every term an agent workflow (update per group of terms, then a cross-cutting critic) wrote a falsifiable
hypothesis: in which levels of the encoder (N = state of the 40 neighbouring epochs, E = epoch, P = 0.5 s token
conditioned on N and E) and with which context (background, isolation, preceding seconds) the term lives. A
spec per term names the inputs of a few word variants, the baseline inputs fair for the term and the matching
variables. This module tests them on a sample of the dump before the extended run. The probes are rationale
probes (a logistic or ridge on the named inputs), not the frozen words.

Commands:

- ``embed`` (CPU): quantized E / N latents of every dumped night from the codes (``MSAEVQ.rvq[s].lookup``),
  ``<dump>/eq/<key>.npz`` with ``Eq_<mod>`` / ``Nq_<mod>`` (the EOG / EMG encoders have no N).
- ``run`` (CPU): every spec of the modality, ``<dump>/rationale_<tag>.json``.

Token specs (``level: "tok"``):

- leave-one-site-out logistic per variant and for the **baseline** (one probe on all the baseline inputs of the
  spec: the strongest fair signal model, also the ceiling of the transport gate); ``train_ref`` instead fits
  once on the pool with that reference and scores the DREAMS excerpts (selection and test kept apart);
- **matched test**: inside each (site, stage) the tokens are cut into quantile cells of the matching variables
  (5 for one variable, 4 for two, 3 for three); per (night, stage, cell) with >= 3 events and >= 3 non-events
  the normalized AP ``(AP - b) / (1 - b)`` against the cell's base rate, event-weighted per night. A word that
  only follows the matched amplitudes scores ~0;
- inference with the night as unit: sign-flip permutation (2000) of the per-night matched difference, plus the
  sign criterion over sites (>= 2/3 of the sites with a positive mean); the Wilcoxon over sites is descriptive;
- **twins** generated for every variant: E / N inputs replaced by the epoch spectrum (``spec_ep``, plus
  ``stage`` when the stratum has several stages), and every context input replaced by the same context taken
  from another epoch of the same stage and night (``_perm``): the gain of E / N or of the context counts only
  over its twin;
- added factor: for every input the matched difference against the variant without it, when computed;
- ``candidates``: population restricted to the candidate tokens (a reference, or a quantile / minimum of a
  matching variable), the label is the remaining criterion; AUC per night reported (conditional task);
- 5 refitted nulls (references shifted inside the (night, stage) runs) for the best variant.

Epoch specs (``level: "epoch"``): target = fraction of the reference tokens in the epoch (or ``post:<ref>:k``,
the epoch follows an event within k epochs); out-of-fold ridge per variant and baseline on epoch inputs;
partial Spearman of the prediction with the target controlling the epoch RMS, per (night, stage); sign-flip over
nights of the difference with the baseline.

Inputs: ``P``, ``P_ctx±k`` (mean P of the k tokens before and after, token excluded), ``P_back<k>`` (mean P of
the k preceding tokens), ``<ctx>_perm``; ``rel_bg`` (log RMS - median log RMS over +-10 s, the token +-1
excluded), ``rel_back`` (against the 10 s before), ``iso`` (max log RMS over +-2 s without the token - log RMS),
``spec_tok`` (band powers on windows of the definition: SO, delta, alpha 2 s; theta, sigma, beta 1.5 s; line
length), ``spec_back`` (spec_tok - median of the 10 s before), ``spec_ep`` (mean spectrum and RMS of the epoch),
``abs_amp`` (log RMS of the token and of +-1 token, robust-scaled per night), ``abs_uv`` (same in uV from
``scale.json``), ``Eq`` / ``Nq`` (PCA 32 of the quantized latents, label-free on the pool epochs), ``ICA_E`` /
``ICA_N`` (FastICA 16, signed and absolute), ``codes_E`` / ``codes_N`` / ``codes_P`` (in-fold log-lift of the
level-0 code and of the level 0+1 prefix), ``stage``; ``P@<mod>`` / ``rel_back@<mod>`` from another modality
of the same night. Epoch level: ``Eq``, ``Nq``, ``ICA_*``, ``codes_E`` / ``codes_N``, ``P_ep`` (mean P),
``spec_ep``, ``abs_ep``, ``stage``.

Not here (declared): C of the logistic fixed (L2, C = 1), no inner CV; Eq / Nq are 32-d PCA and P is 64-d, so
the twins, not the raw dimensions, carry the comparison.

    python -m physioex.train.bin.concept_rationale embed --dump DUMP --combo-dir RUN
    python -m physioex.train.bin.concept_rationale run --dump DUMP --mod eeg --specs specs.json --tag eeg
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
N_PERM = 2000
SPEC_WIN = {"so": (0.5, 2, 2.0), "delta": (2, 4, 2.0), "theta": (4, 7, 1.5), "alpha": (8, 13, 2.0),
            "sigma": (11, 16, 1.5), "beta": (16, 30, 1.5)}
LATENT = ("Eq", "Nq", "ICA_E", "ICA_N", "codes_E", "codes_N")
N_LATENT = ("Nq", "ICA_N", "codes_N")
NB_EPOCHS = 20  # the N context of the encoder: 40 neighbouring epochs


# ---------------------------------------------------------------------------- context features
def _runs(night):
    """Index ranges of every night in the flat token arrays (nights are contiguous blocks)."""
    edges = np.flatnonzero(np.diff(night) != 0) + 1
    return list(zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(night)]])))


def _filt(a, night, footprint, kind):
    from scipy.ndimage import maximum_filter, median_filter

    f = {"median": median_filter, "max": maximum_filter}[kind]
    out = np.empty_like(a)
    fp = footprint if a.ndim == 1 else footprint[:, None]
    for s, e in _runs(night):
        out[s:e] = f(a[s:e], footprint=fp, mode="nearest")
    return out


def _fp(before, after, skip):
    """Footprint centred on the token: ``before`` / ``after`` tokens, the central ``2 skip + 1`` excluded."""
    h = max(before, after)
    fp = np.zeros(2 * h + 1, bool)
    fp[h - before:h] = True
    fp[h + 1:h + 1 + after] = True
    fp[max(h - skip, 0):h + skip + 1] = False
    return fp


def rel_bg(F0, night, half=20, skip=1):
    """log RMS of the token minus the median log RMS over +-``half`` tokens (+-10 s), the token and its
    +-``skip`` neighbours excluded (the event does not raise its own background)."""
    return F0 - _filt(F0, night, _fp(half, half, skip), "median")


def rel_back(A, night, k=20):
    """``A`` minus its median over the ``k`` preceding tokens (10 s before; arousal-like changes)."""
    return A - _filt(A, night, _fp(k, 0, 0), "median")


def iso(F0, night, half=4):
    """Max log RMS of the +-``half`` neighbouring tokens (+-2 s, token excluded) minus its own log RMS:
    negative = the token stands out of its surroundings (relative isolation)."""
    return _filt(F0, night, _fp(half, half, 0), "max") - F0


def ctx_mean(P, night, k, back=False, fwd=False):
    """Mean of the P of the ``k`` tokens before and after (only before, ``back``; only after, ``fwd``), token
    excluded; with no token on that side (night edge) the token keeps its own P under ``back`` / ``fwd``."""
    out = np.zeros_like(P)
    if fwd:
        for a, b in _runs(night):
            c = np.cumsum(np.vstack([np.zeros((1, P.shape[1]), P.dtype), P[a:b]]), 0)
            i = np.arange(b - a)
            lo, hi = i + 1, np.minimum(i + k + 1, b - a)
            n = hi - lo
            out[a:b] = np.where(n[:, None] > 0, (c[hi] - c[lo]) / np.maximum(n, 1)[:, None], P[a:b])
        return out
    for a, b in _runs(night):
        c = np.cumsum(np.vstack([np.zeros((1, P.shape[1]), P.dtype), P[a:b]]), 0)
        i = np.arange(b - a)
        lo = np.maximum(i - k, 0)
        hi = i if back else np.minimum(i + k + 1, b - a)
        n = hi - lo - (0 if back else 1)
        s = c[hi] - c[lo] - (0 if back else P[a:b])
        out[a:b] = np.where(n[:, None] > 0, s / np.maximum(n, 1)[:, None], P[a:b] if back else 0)
    return out


def epoch_perm(y, night, rng):
    """Token index of the same offset in another epoch of the same stage and night (context twin)."""
    ey, en = y[::T], night[::T]
    pe = np.arange(len(ey))
    for n in np.unique(en):
        for s in np.unique(ey[en == n]):
            ix = np.flatnonzero((en == n) & (ey == s))
            if len(ix) > 1:
                q = ix[rng.permutation(len(ix))]
                pe[q] = np.roll(q, 1)  # one random cycle: never the epoch itself
    tok = np.arange(len(y))
    return pe[tok // T] * T + tok % T


def spec_windows(x, fs):
    """Log band powers on the window of each band (``SPEC_WIN``) centred on every token, plus log line length
    of the token: ``(n * 60, 7)``."""
    from numpy.lib.stride_tricks import sliding_window_view

    tok = int(round(0.5 * fs))
    v = x.reshape(-1).astype(np.float32)
    n = len(v) // tok
    cols = {}
    for wsec in sorted({w for *_, w in SPEC_WIN.values()}):
        wl = int(round(wsec * fs))
        pad = (wl - tok) // 2 + 1
        vp = np.concatenate([v[:pad][::-1], v, v[-(pad + wl):][::-1]])
        w = sliding_window_view(vp, wl)[pad - (wl - tok) // 2::tok][:n]
        p = np.abs(np.fft.rfft((w - w.mean(1, keepdims=True)) * np.hanning(wl).astype(np.float32), axis=1)) ** 2
        f = np.fft.rfftfreq(wl, 1 / fs)
        for b, (lo, hi, ws) in SPEC_WIN.items():
            if ws == wsec:
                cols[b] = np.log(p[:, (f >= lo) & (f < hi)].sum(1) + 1e-6)
    out = [cols[b] for b in SPEC_WIN]
    t = v[: n * tok].reshape(n, tok)
    out.append(np.log(np.abs(np.diff(t, axis=1)).mean(1) + 1e-6))
    return np.stack(out, 1).astype(np.float32)


SPEC_FINE = False  # ``load`` also computes ``spec_fine`` (D1 of the critical evaluation: a richer spectral twin)
FINE_WIN, FINE_HZ = 2.0, (0.5, 30.5)


def spec_fine(x, fs, wsec=FINE_WIN, lo=FINE_HZ[0], hi=FINE_HZ[1]):
    """Log power in 1 Hz bins (``lo``..``hi``) of a ``wsec`` Hann window centred on every token: ``(n * 60, 30)``.
    The spectral twin of P at a comparable width (P has 64 numbers per token)."""
    from numpy.lib.stride_tricks import sliding_window_view

    tok = int(round(0.5 * fs))
    v = x.reshape(-1).astype(np.float32)
    n = len(v) // tok
    wl = int(round(wsec * fs))
    pad = (wl - tok) // 2 + 1
    vp = np.concatenate([v[:pad][::-1], v, v[-(pad + wl):][::-1]])
    w = sliding_window_view(vp, wl)[pad - (wl - tok) // 2::tok][:n]
    p = np.abs(np.fft.rfft((w - w.mean(1, keepdims=True)) * np.hanning(wl).astype(np.float32), axis=1)) ** 2
    f = np.fft.rfftfreq(wl, 1 / fs)
    edges = np.arange(lo, hi + 1e-6, 1.0)
    return np.stack([np.log(p[:, (f >= a) & (f < b)].sum(1) + 1e-6) for a, b in zip(edges[:-1], edges[1:])], 1).astype(np.float32)


INPUT_MAPS = {"spec": {"P": ["spec_tok"], "Eq": ["spec_ep"], "Nq": ["spec_nb"]},
              "specfine": {"P": ["spec_tok", "spec_fine"], "Eq": ["spec_ep"], "Nq": ["spec_nb"]}}


def mapped(D, names):
    """Input names after ``D["_map"]`` (``"spec"`` / ``"specfine"``, D1): every latent of this modality replaced
    by its spectral twin at the same context (``P`` -> token spectrum, ``P_ctx±k`` -> ``spec_ctx{k}``, ``Eq`` ->
    epoch spectrum, ``Nq`` -> neighbourhood summary); other modalities (``P@eog``...) stay latent, declared."""
    mode = D.get("_map")
    if not mode:
        return list(names)
    tab = INPUT_MAPS[mode]
    out = []
    for n in names:
        m = CTX.fullmatch(n)
        if n in tab:
            out += tab[n]
        elif m and not m.group(4):
            k = int(m.group(1) or m.group(2) or m.group(3))
            kind = "ctx" if m.group(1) else ("back" if m.group(2) else "fwd")
            out.append(f"spec_{kind}{k}")
            if mode == "specfine" and kind == "ctx":
                out.append(f"spec_fine_ctx{k}")
        else:
            out.append(n)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------- probes
def lift_feature(k_tr, r_tr, k_te, prior=20.0):
    """log-lift of the keys (shrunk with ``prior`` pseudo-uses) fitted on train, 0 for unseen keys; ``r_tr`` may
    be a fraction (epoch targets)."""
    b = max(float(np.mean(r_tr)), 1e-9)
    u, inv = np.unique(k_tr, return_inverse=True)
    ll = np.log((np.bincount(inv, weights=r_tr) + prior * b + 1e-12) / (np.bincount(inv) + prior) / b)
    pos = np.searchsorted(u, k_te).clip(0, len(u) - 1)
    return np.where(u[pos] == k_te, ll[pos], 0.0).astype(np.float32)


def _code_keys(codes):
    keys = []
    for C in codes:
        k0 = C[:, 0].astype(np.int64)
        keys.append(k0)
        if C.shape[1] > 1:
            keys.append(k0 * (int(C[:, 1].max()) + 1) + C[:, 1].astype(np.int64))
    return keys


def _fit_predict(X, keys, r, tr, te, max_fit, rng, ridge=False, mlp=0):
    """``mlp`` > 0: a two-layer MLP of that width instead of the logistic (D2: non-linear content), balanced by
    subsampling the negatives to at most 4 x the positives before ``max_fit``."""
    from sklearn.linear_model import LogisticRegression, Ridge

    lf = [(lift_feature(k[tr], r[tr], k[tr]), lift_feature(k[tr], r[tr], k[te])) for k in keys]
    Xtr = np.column_stack([X[tr]] + [a for a, _ in lf]) if lf else X[tr]
    Xte = np.column_stack([X[te]] + [b for _, b in lf]) if lf else X[te]
    if mlp:
        pos, neg = np.flatnonzero(r[tr] > 0), np.flatnonzero(r[tr] <= 0)
        neg = rng.choice(neg, min(len(neg), 4 * len(pos)), replace=False)
        cand = np.r_[pos, neg]
        sub = cand[rng.choice(len(cand), min(max_fit, len(cand)), replace=False)]
    else:
        sub = rng.choice(len(tr), min(max_fit, len(tr)), replace=False)
    mu, sd = Xtr[sub].mean(0), Xtr[sub].std(0) + 1e-6
    if ridge:
        return Ridge(alpha=1.0).fit((Xtr[sub] - mu) / sd, r[tr][sub]).predict((Xte - mu) / sd)
    if mlp:
        from sklearn.neural_network import MLPClassifier

        m = MLPClassifier((mlp, mlp), early_stopping=True, max_iter=200, random_state=int(rng.integers(1 << 30)))
        m.fit((Xtr[sub] - mu) / sd, r[tr][sub] > 0)
        return m.predict_proba((Xte - mu) / sd)[:, 1]
    m = LogisticRegression(max_iter=300).fit((Xtr[sub] - mu) / sd, r[tr][sub])
    return m.decision_function((Xte - mu) / sd)


def fold_probe(X, codes, r, key, groups, max_fit=30_000, seed=0, train=None, ridge=False, mlp=0):
    """Held-out scores of a logistic (``ridge``: ridge regression, continuous targets; ``mlp``: two-layer MLP of
    that width) on ``X`` plus the in-fold log-lift of the code arrays. Leave-one-group-out; with a boolean
    ``train`` mask instead one fit on the ``train`` rows that scores the others (transfer)."""
    rng = np.random.default_rng(seed)
    keys = _code_keys(codes)
    out = np.full(len(r), np.nan)
    if train is not None:
        tr, te = np.flatnonzero(train), np.flatnonzero(~train)
        if r[tr].any() and not r[tr].all():
            out[te] = _fit_predict(X, keys, r, tr, te, max_fit, rng, ridge, mlp)
        return out
    for g in groups:
        tr, te = np.flatnonzero(key != g), np.flatnonzero(key == g)
        if ridge:
            if np.ptp(r[tr]) == 0:
                continue
        elif not r[te].any() or not r[tr].any() or r[tr].all():
            continue
        out[te] = _fit_predict(X, keys, r, tr, te, max_fit, rng, ridge, mlp)
    return out


def ap(s, r):
    from physioex.train.bin.concept_scan import average_precision

    return float(average_precision(s, r))


def auc(s, r):
    from scipy.stats import rankdata

    n1 = int(r.sum())
    n0 = len(r) - n1
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[r].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cells(key, y, M):
    """Quantile cell of every token inside its (site, stage): 5 / 4 / 3 quantiles for 1 / 2 / 3 variables."""
    q = {1: 5, 2: 4}.get(M.shape[1], 3)
    b = np.zeros(len(key), np.int64)
    for g in np.unique(key):
        for st in np.unique(y[key == g]):
            m = np.flatnonzero((key == g) & (y == st))
            c = np.zeros(len(m), np.int64)
            for j in range(M.shape[1]):
                edges = np.unique(np.quantile(M[m, j], np.linspace(0, 1, q + 1)[1:-1]))
                c = c * (q + 1) + np.searchsorted(edges, M[m, j])
            b[m] = c
    return b


def matched_nights(s, r, night, y, cell, min_pos=3):
    """Per night: event-weighted normalized AP of ``s`` inside the (stage, cell) groups with >= ``min_pos``
    events and non-events, and the number of events used."""
    out = {}
    ok = np.isfinite(s)
    for n in np.unique(night[r & ok]):
        m = np.flatnonzero((night == n) & ok)
        num = den = 0.0
        comb = y[m].astype(np.int64) * 10_000 + cell[m]
        for u in np.unique(comb):
            ix = m[comb == u]
            npos = int(r[ix].sum())
            if npos < min_pos or len(ix) - npos < min_pos:
                continue
            base = npos / len(ix)
            num += npos * (ap(s[ix], r[ix]) - base) / (1 - base)
            den += npos
        if den:
            out[int(n)] = (num / den, den)
    return out


def signflip_p(d, n_perm=N_PERM, seed=0):
    """One-sided sign-flip permutation p of mean(d) > 0 (units = nights); floor 1 / (n_perm + 1)."""
    d = np.asarray([v for v in d if np.isfinite(v)], float)
    if len(d) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    obs = d.mean()
    null = (rng.choice([-1.0, 1.0], (n_perm, len(d))) * d).mean(1)
    return float((1 + (null >= obs).sum()) / (n_perm + 1))


def wilcoxon_gt0(d):
    from scipy.stats import wilcoxon

    d = np.asarray([v for v in d if np.isfinite(v)])
    if len(d) < 3 or not np.any(d):
        return 1.0
    return float(wilcoxon(d, alternative="greater").pvalue)


def compare(a, b, site_of):
    """Paired per-night difference of two matched dicts: mean, sign-flip p, sites with positive mean."""
    common = sorted(set(a) & set(b))
    d = np.array([a[n][0] - b[n][0] for n in common])
    if not len(d):
        return {"n": 0}
    by = {}
    for n, v in zip(common, d):
        by.setdefault(site_of[n], []).append(v)
    sm = {g: float(np.mean(v)) for g, v in by.items()}
    pos = sum(v > 0 for v in sm.values())
    return {"n": len(d), "mean": float(d.mean()), "p_perm": signflip_p(d), "sites_pos": f"{pos}/{len(sm)}",
            "sign_ok": bool(len(sm) >= 3 and pos >= int(np.ceil(2 * len(sm) / 3))), "per_site": sm,
            "p_wilcoxon_sites": wilcoxon_gt0(list(sm.values()))}


def summary(md, site_of):
    v = [x[0] for x in md.values()]
    by = {}
    for n, (x, _) in md.items():
        by.setdefault(site_of[n], []).append(x)
    return {"mean": float(np.mean(v)) if v else None, "n_nights": len(v), "p_perm": signflip_p(v),
            "per_site": {g: float(np.mean(x)) for g, x in by.items()}}


# ---------------------------------------------------------------------------- inputs
CTX = re.compile(r"(?:P_ctx(?:\+-|±|\+/-)?(\d+)|P_back(\d+)|P_fwd(\d+))(_perm)?")


def token_input(D, name):
    """Full-length array of a token input (cached in ``D["_cache"]``); ``None`` if unknown or missing."""
    if name == "P":
        return D["P"]
    if name in D["_cache"]:
        return D["_cache"][name]
    m = CTX.fullmatch(name)
    out = None
    if m:
        if m.group(4):
            out = token_input(D, name[: -len("_perm")])[D["perm"]]
        else:
            out = ctx_mean(D["P"], D["night"], int(m.group(1) or m.group(2) or m.group(3)),
                           back=m.group(2) is not None, fwd=m.group(3) is not None)
    elif name.startswith("P@") or name.startswith("rel_back@"):
        out = D.get(name)
        if out is not None and out.ndim == 1:
            out = out[:, None]
    elif name in ("rel_bg", "rel_back", "iso"):
        out = D[name][:, None]
    elif name in ("Eq", "Nq", "spec_ep", "spec_back", "spec_nb", "spec_nb_lab") and name in D:
        out = D[name]
    elif re.fullmatch(r"spec_ctx(\d+)", name):
        k = int(name[8:])
        out = (ctx_mean(D["S"], D["night"], k) * 2 * k + D["S"]) / (2 * k + 1)  # token and its +-k, edges approx.
    elif re.fullmatch(r"spec_back(\d+)", name):
        out = ctx_mean(D["S"], D["night"], int(name[9:]), back=True)
    elif re.fullmatch(r"spec_fwd(\d+)", name):
        out = ctx_mean(D["S"], D["night"], int(name[8:]), fwd=True)
    elif name in ("ICA_E", "ICA_N") and name in D:
        out = np.hstack([D[name], np.abs(D[name])])
    elif name == "spec_tok":
        out = D["S"]
    elif name == "spec_fine" and "SF" in D:
        out = D["SF"]
    elif re.fullmatch(r"spec_fine_ctx(\d+)", name) and "SF" in D:
        k = int(name[13:])
        out = (ctx_mean(D["SF"], D["night"], k) * 2 * k + D["SF"]) / (2 * k + 1)
    elif name == "abs_amp":
        out = D["F"][:, :2]
    elif name == "abs_uv":
        out = D["F"][:, :2] + D["logscale"][:, None]
    elif name == "stage":
        out = np.stack([D["y"] == s for s in range(5)], 1).astype(np.float32)
    if out is not None:
        D["_cache"][name] = out
    return out


def _quadratic(name):
    """Latent inputs enter with their element-wise squares: an oscillation flips sign token by token, its
    content is the energy along a direction (the ICA words of the lexicon are energies)."""
    return name == "P" or name.startswith("P@") or name in ("Eq", "Nq", "P_ep") or bool(CTX.fullmatch(name))


def build(D, names, idx):
    X, codes, bad = [], [], []
    for n in mapped(D, names):
        if n in ("codes_E", "codes_N", "codes_P"):
            (codes.append(D[n][idx]) if n in D else bad.append(n))
            continue
        a = token_input(D, n)
        if a is None:
            bad.append(n)
            continue
        X.append(a[idx])
        if _quadratic(n):
            rot = D.get("_icaP") if (n == "P" or CTX.fullmatch(n)) else None
            X.append((rot.transform(a[idx]) if rot is not None else a[idx]) ** 2)
    X = np.column_stack(X).astype(np.float32) if X else np.zeros((len(idx), 0), np.float32)
    return X, codes, bad


MATCH = {"abs": lambda D: D["F"][:, 0], "rel": lambda D: D["rel_bg"], "iso": lambda D: D["iso"],
         "back": lambda D: D["rel_back"], "uv": lambda D: D["F"][:, 0] + D["logscale"], "ll": lambda D: D["S"][:, -1]}


def ref_mask(D, expr):
    """Reference mask and availability: a name, ``y==k``, ``a&b`` (intersection), ``a|b`` (union),
    ``onset:<name>`` (first 2 tokens of every event run, 1 s)."""
    if expr.startswith("y=="):
        return D["y"] == int(expr[3:]), np.ones(len(D["y"]), bool)
    if expr.startswith("onset:"):
        r, h = ref_mask(D, expr[6:])
        same = np.r_[False, D["night"][1:] == D["night"][:-1]]
        first = r & ~(np.r_[False, r[:-1]] & same)
        return first | (r & np.r_[False, first[:-1]] & same), h
    for op, f in (("|", np.logical_or), ("&", np.logical_and)):
        if op in expr:
            parts = [ref_mask(D, p) for p in expr.split(op)]
            r, h = parts[0]
            for r2, h2 in parts[1:]:
                r, h = f(r, r2), h & h2
            return r, h
    if expr not in D["refs"]:
        raise KeyError(expr)
    return D["refs"][expr]


def twins(variants, multi_stage, stage_label=False):
    """Twin variants: latent inputs replaced by the epoch spectrum (+ stage when the stratum has several stages
    and the label is not the stage itself); N inputs replaced by the label-free neighbourhood summary ``spec_nb``
    (mean, sd and mean step of the epoch spectrum over the +-20 epochs, + the epoch spectrum when the variant
    reads no E latent: what N could know without being N; ``spec_nb_lab`` with the stage fractions is too strong,
    the neighbours' stages alone predict the stage better than N); context by its permuted copy, except when the
    label is the stage (the permutation stays inside the stage and keeps the label)."""
    out = []
    for v in variants:
        if any(i in LATENT for i in v):
            t = [i for i in v if i not in LATENT] + ["spec_ep"] + (["stage"] if multi_stage else [])
            out.append((v, "twin_ep", t))
        if any(i in N_LATENT for i in v):  # label-free neighbourhood; + the epoch spectrum when E is not read
            e_in = any(i in LATENT and i not in N_LATENT for i in v)
            out.append((v, "twin_N", [i for i in v if i not in N_LATENT] + ["spec_nb"] + ([] if e_in else ["spec_ep"])))
        if stage_label:  # the permutation stays inside the stage: it keeps the label, no valid context twin
            continue
        if any(CTX.fullmatch(i) and not i.endswith("_perm") for i in v):
            out.append((v, "twin_perm", [i + "_perm" if CTX.fullmatch(i) else i for i in v]))
    return out


# ---------------------------------------------------------------------------- one token spec
def run_tok(D, spec, n_null=5, seed=0):
    from physioex.train.bin.concept_words import shift_within_runs

    rng = np.random.default_rng(seed)
    try:
        rv, has = ref_mask(D, spec["ref"])
    except KeyError as e:
        return {"term": spec["term"], "error": f"reference {e} not in the loaded nights"}
    y = D["yprev"] if spec.get("stage_prev") else D["y"]
    sets = spec.get("sets", ["pool"])
    st_ok = np.isin(y, spec["stages"])
    sel = np.isin(D["set"], sets) & has & st_ok
    for x in spec.get("exclude_sites", []):  # sites whose reference is unusable (point annotations, none)
        sel &= ~np.char.startswith(D["gname"].astype(str), x)
    tr_rows = None
    if spec.get("train_ref"):
        rt, ht = ref_mask(D, spec["train_ref"])
        tr_rows = (D["set"] == "pool") & ht & st_ok
        sel = sel | tr_rows
        rv = np.where(tr_rows, rt, rv)
    cand = spec.get("candidates")
    if cand:
        if "ref" in cand:
            c, hc = ref_mask(D, cand["ref"])
            sel &= c & hc
        else:
            v = MATCH[cand["var"]](D)
            if "min" in cand:
                sel &= v >= cand["min"]
            else:
                thr = np.full(len(v), np.inf)
                for g in np.unique(D["gname"][sel]):
                    for st in spec["stages"]:
                        m = (D["gname"] == g) & (y == st) & sel
                        if m.any():
                            thr[m] = np.quantile(v[m], cand["q"])
                sel &= v >= thr
    idx = np.flatnonzero(sel)
    r = rv[idx].astype(bool)
    yy, night, gn = y[idx], D["night"][idx], D["gname"][idx]
    dreams_eval = sets == ["dreams"]
    train = tr_rows[idx] if tr_rows is not None else None
    key = night.astype(str) if dreams_eval and train is None else gn
    groups = sorted(set(key))
    site_of = {int(n): (str(n) if dreams_eval else g) for n, g in zip(night, gn)}
    M = np.column_stack([MATCH[m](D)[idx] for m in spec.get("matched_bins") or ["abs"]])
    ck = np.zeros(len(idx), int).astype(str) if dreams_eval else gn
    across = bool(spec.get("match_across_stages"))  # the label is the stage: cells pool the stages
    ycell = np.zeros(len(idx), np.int8) if across else yy
    cell = cells(np.where(train, "_train", ck) if train is not None else ck, ycell, M)  # cells of the evaluated tokens
    ev = ~train if train is not None else np.ones(len(idx), bool)
    out = {"term": spec["term"], "mod": spec.get("mod"), "ref": spec["ref"], "gate": spec.get("gate", "content"),
           "n_tok": int(ev.sum()), "base_rate": float(r[ev].mean()) if ev.any() else None, "groups": len(groups),
           "n_nights_with_events": len(set(night[ev & r])), "skipped_inputs": []}
    if not (r & ev).any():
        out["error"] = "no events in the evaluated tokens"
        return out

    def probe(inputs, rr=r, sd=seed):
        X, cs, bad = build(D, inputs, idx)
        out["skipped_inputs"] += bad
        if X.shape[1] == 0 and not cs:
            return None
        return np.where(ev, fold_probe(X, cs, rr, key, groups, seed=sd, train=train), np.nan)

    def score(s, rr=r):
        md = matched_nights(s, rr & ev, night, ycell, cell)
        ok = ev & np.isfinite(s)
        res = {"ap": ap(s[ok], rr[ok]) if rr[ok].any() else None, "matched": summary(md, site_of)}
        if cand:
            au = [auc(s[ok & (night == n)], rr[ok & (night == n)]) for n in np.unique(night[ok])]
            res["auc_nights"] = float(np.nanmean(au)) if np.isfinite(au).any() else None
            res["auc_pooled"] = auc(s[ok], rr[ok])
        return res, md

    sb = probe(spec["baselines"])
    base_res, base_md = score(sb) if sb is not None else ({}, {})
    out["baseline"] = {"inputs": spec["baselines"], **base_res}
    vs = [list(v) for v in spec["word_variants"]]
    ceil_in = spec.get("ceiling") or ceiling_inputs(spec, vs)
    sc = probe(ceil_in)
    ceil_res, _ = score(sc) if sc is not None else ({}, {})
    out["ceiling"] = {"inputs": ceil_in, **ceil_res}
    todo = [(v, None, None) for v in vs] + twins(vs, len(spec["stages"]) > 1 and not spec["ref"].startswith("y=="),
                                                   stage_label=spec["ref"].startswith("y=="))
    done, res, scores = {}, [], {}
    for v, kind, t in todo:
        inputs = t if kind else v
        k = "+".join(inputs)
        if k in done:
            continue
        s = probe(inputs)
        if s is None:
            continue
        rr, md = score(s)
        done[k], scores[k] = md, s
        rr.update({"inputs": inputs, "twin_of": "+".join(v) if kind else None, "kind": kind or "variant"})
        if base_md:
            rr["vs_baseline"] = compare(md, base_md, site_of)
        if base_res.get("ap") and rr["ap"] is not None:
            rr["retention"] = rr["ap"] / base_res["ap"]
        if ceil_res.get("ap") and rr["ap"] is not None:
            rr["retention_ceiling"] = rr["ap"] / ceil_res["ap"]
        res.append(rr)
    for rr in res:
        k = "+".join(rr["inputs"])
        if rr["kind"] == "variant":
            rr["vs_twins"] = {t["kind"]: compare(done[k], done["+".join(t["inputs"])], site_of)
                              for t in res if t["twin_of"] == k}
            rr["added"] = {i: compare(done[k], done[sub], site_of) for i in rr["inputs"]
                           for sub in ["+".join(x for x in rr["inputs"] if x != i)] if sub in done}
    out["variants"] = res
    var_only = [x for x in res if x["kind"] == "variant" and x["matched"]["mean"] is not None]
    if var_only and n_null:
        best = max(var_only, key=lambda x: x["matched"]["mean"])
        nulls = []
        for i in range(n_null):
            rs = r.copy()
            rs[ev] = shift_within_runs(r[ev], night[ev], yy[ev], rng)
            md = matched_nights(probe(best["inputs"], rs, seed + i + 1), rs & ev, night, ycell, cell)
            nulls.append(float(np.mean([x[0] for x in md.values()])) if md else 0.0)
        obs = best["matched"]["mean"]
        out["null_matched"] = {"inputs": best["inputs"], "values": nulls,
                               "p": (1 + sum(n >= obs for n in nulls)) / (len(nulls) + 1)}
        out["_best_scores"] = (idx[ev], scores["+".join(best["inputs"])][ev])
    if dreams_eval:  # excerpts are not sites: no sign criterion
        for rr in res:
            for c in [rr.get("vs_baseline")] + list((rr.get("vs_twins") or {}).values()) + list((rr.get("added") or {}).values()):
                if c:
                    c["sign_ok"] = None
    out["skipped_inputs"] = sorted(set(out["skipped_inputs"]))
    return out


def ceiling_inputs(spec, variants):
    """Signal ceiling with the same context as the words: the baseline inputs plus the spectrum over the widest
    context window of the variants, the epoch spectrum when a variant reads E / N and the neighbourhood summary
    when it reads N."""
    out = list(spec["baselines"])
    flat = {i for v in variants for i in v}
    ks = [int(m.group(1)) for i in flat for m in [CTX.fullmatch(i)] if m and m.group(1)]
    kb = [int(m.group(2)) for i in flat for m in [CTX.fullmatch(i)] if m and m.group(2)]
    kf = [int(m.group(3)) for i in flat for m in [CTX.fullmatch(i)] if m and m.group(3)]
    if ks:
        out.append(f"spec_ctx{max(ks)}")
    if kb:
        out.append(f"spec_back{max(kb)}")
    if kf:
        out.append(f"spec_fwd{max(kf)}")
    if flat & set(LATENT) or any(i.startswith("P@") for i in flat):
        out.append("spec_ep")
    if flat & set(N_LATENT):
        out.append("spec_nb")
    if len(spec["stages"]) > 1 and not spec["ref"].startswith("y==") and "stage" not in out:
        out.append("stage")
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------- one epoch spec
def partial_spearman(p, t, C):
    """Spearman of ``p`` and ``t`` with the ranks of the controls ``C`` regressed out."""
    from scipy.stats import rankdata

    if np.ptp(t) == 0 or np.ptp(p) == 0:
        return np.nan
    Z = np.column_stack([np.ones(len(t))] + [rankdata(c) for c in C.T])
    rp, rt = rankdata(p), rankdata(t)
    ep = rp - Z @ np.linalg.lstsq(Z, rp, rcond=None)[0]
    et = rt - Z @ np.linalg.lstsq(Z, rt, rcond=None)[0]
    d = np.sqrt((ep ** 2).sum() * (et ** 2).sum())
    return float((ep * et).sum() / d) if d > 0 else np.nan


def epoch_target(D, expr):
    """Per epoch target and availability: fraction of a reference (``a``, ``a&b``...) or ``post:<ref>:k`` (an
    event of ``ref`` in one of the ``k`` previous epochs of the same night)."""
    if expr.startswith("post:"):
        _, name, k = expr.split(":")
        r, h = ref_mask(D, name)
        ev = r.reshape(-1, T).any(1)
        en = D["night"][::T]
        t = np.zeros(len(ev))
        for j in range(1, int(k) + 1):
            prev = np.r_[np.zeros(j, bool), ev[:-j] & (en[j:] == en[:-j])]
            t = np.maximum(t, prev)
        return t, h[::T]
    r, h = ref_mask(D, expr)
    return r.reshape(-1, T).mean(1), h[::T]


def epoch_input(D, name):
    if name == "P_ep":
        return D["P"].reshape(-1, T, D["P"].shape[1]).mean(1)
    if name == "abs_ep":
        return D["spec_ep"][::T, -1:]
    if name in ("Eq", "Nq", "spec_ep", "spec_nb", "spec_nb_lab") and name in D:
        return D[name][::T]
    if "@" in name and name in D:  # per-modality epoch latents (Eq@emg, Eq@emg_rel)
        return D[name][::T]
    if name in ("ICA_E", "ICA_N") and name in D:
        return np.hstack([D[name][::T], np.abs(D[name][::T])])
    if name == "stage":
        return np.stack([D["y"][::T] == s for s in range(5)], 1).astype(np.float32)
    return None


def run_epoch(D, spec, n_null=5, seed=0):
    rng = np.random.default_rng(seed)
    try:
        t, has = epoch_target(D, spec["ref"])
    except KeyError as e:
        return {"term": spec["term"], "error": f"reference {e} not in the loaded nights"}
    ey, en, eg, es = D["y"][::T], D["night"][::T], D["gname"][::T], D["set"][::T]
    sel = np.isin(es, spec.get("sets", ["pool"])) & has & np.isin(ey, spec["stages"])
    idx = np.flatnonzero(sel)
    tt, yy, night, key = t[idx], ey[idx], en[idx], eg[idx]
    groups = sorted(set(key))
    site_of = {int(n): g for n, g in zip(night, key)}
    cl = spec.get("controls", ["abs_ep"])
    ctrl = np.column_stack([epoch_input(D, c)[idx] for c in cl]) if cl else np.zeros((len(idx), 0))
    out = {"term": spec["term"], "mod": spec.get("mod"), "ref": spec["ref"], "level": "epoch",
           "n_ep": int(len(idx)), "target_mean": float(tt.mean()) if len(tt) else None, "groups": len(groups),
           "skipped_inputs": []}
    if not len(idx) or np.ptp(tt) == 0:
        out["error"] = "constant or empty target"
        return out

    def probe(inputs, target=tt, sd=seed):
        X, cs = [], []
        for n in inputs:
            if n in ("codes_E", "codes_N"):
                (cs.append(D[n][::T][idx]) if n in D else out["skipped_inputs"].append(n))
                continue
            a = epoch_input(D, n)
            if a is None:
                out["skipped_inputs"].append(n)
                continue
            X.append(a[idx])
            if _quadratic(n):
                X.append(a[idx] ** 2)
        if not X and not cs:
            return None
        X = np.column_stack(X).astype(np.float32) if X else np.zeros((len(idx), 0), np.float32)
        return fold_probe(X, cs, target, key, groups, seed=sd, ridge=True)

    def rho_nights(s, target=tt):
        by = {}
        for n in np.unique(night):
            for st in np.unique(yy[night == n]):
                m = (night == n) & (yy == st) & np.isfinite(s)
                if m.sum() >= 20:
                    v = partial_spearman(s[m], target[m], ctrl[m])
                    if np.isfinite(v):
                        by.setdefault(int(n), []).append((v, int(m.sum())))
        return {n: (float(np.average([a for a, _ in x], weights=[b for _, b in x])), sum(b for _, b in x))
                for n, x in by.items()}

    sb = probe(spec["baselines"])
    bmd = rho_nights(sb) if sb is not None else {}
    out["baseline"] = {"inputs": spec["baselines"], "rho": summary(bmd, site_of)}
    if sb is not None:  # diagnostic: the baseline without the RMS control
        raw = [partial_spearman(sb[m], tt[m], np.zeros((m.sum(), 0))) for n in np.unique(night)
               for m in [(night == n) & np.isfinite(sb)] if m.sum() >= 20]
        out["baseline"]["rho_uncontrolled"] = float(np.nanmean(raw)) if raw else None
    res, done = [], {}
    vs = [list(v) for v in spec["word_variants"]]
    todo = [(v, None, None) for v in vs] + [x for x in twins(vs, len(spec["stages"]) > 1) if x[1] in ("twin_ep", "twin_N")]
    for v, kind, tw in todo:
        inputs = tw if kind else v
        k = "+".join(inputs)
        if k in done:
            continue
        s = probe(inputs)
        if s is None:
            continue
        md = rho_nights(s)
        done[k] = md
        sm = summary(md, site_of)
        bm = out["baseline"]["rho"]["mean"]
        res.append({"inputs": inputs, "kind": kind or "variant", "twin_of": "+".join(v) if kind else None,
                    "rho": sm, "vs_baseline": compare(md, bmd, site_of) if bmd else None,
                    "retention": sm["mean"] / bm if sm["mean"] is not None and bm else None})
    for rr in res:
        k = "+".join(rr["inputs"])
        if rr["kind"] == "variant":
            rr["vs_twins"] = {t["kind"]: compare(done[k], done["+".join(t["inputs"])], site_of)
                              for t in res if t["twin_of"] == k}
    out["variants"] = res
    var_only = [x for x in res if x["kind"] == "variant" and x["rho"]["mean"] is not None]
    if var_only and n_null:
        best = max(var_only, key=lambda x: x["rho"]["mean"])
        nulls = []
        for i in range(n_null):
            ts = tt.copy()
            for n in np.unique(night):
                for st in np.unique(yy[night == n]):
                    m = np.flatnonzero((night == n) & (yy == st))
                    ts[m] = np.roll(tt[m], rng.integers(1, max(2, len(m))))
            md = rho_nights(probe(best["inputs"], ts, seed + i + 1), ts)
            nulls.append(float(np.mean([x[0] for x in md.values()])) if md else 0.0)
        out["null_rho"] = {"inputs": best["inputs"], "values": nulls,
                           "p": (1 + sum(n >= best["rho"]["mean"] for n in nulls)) / (len(nulls) + 1)}
    out["skipped_inputs"] = sorted(set(out["skipped_inputs"]))
    return out


def cross_specificity(D, results, families):
    """nAP of the best word of term X on the reference of term Y (pool tokens where X is scored and Y is
    available); a term passes when its diagonal beats the other cells of its row by >= 0.10."""
    by = {r["term"]: r for r in results if "_best_scores" in r}
    out = {}
    for fam, terms in families.items():
        terms = [t for t in terms if t in by]
        tab = {}
        for x in terms:
            idx, s = by[x]["_best_scores"]
            full = np.full(len(D["y"]), np.nan)
            full[idx] = s
            row = {}
            for yt in terms:
                ry, hy = ref_mask(D, by[yt]["ref"])
                m = np.isfinite(full) & hy & (D["set"] == "pool")
                b = ry[m].mean() if m.any() else 0
                row[yt] = (ap(full[m], ry[m]) - b) / (1 - b) if m.any() and ry[m].any() else None
            tab[x] = row
        margin = {}
        for x, row in tab.items():
            others = [v for k, v in row.items() if k != x and v is not None]
            margin[x] = row[x] - max(others) if row[x] is not None and others else None
        out[fam] = {"nap": tab, "diag_margin": margin}
    return out


# ---------------------------------------------------------------------------- data
def load(dump, mod, refs_needed, extra_mods=(), frac=0.15, min_n=3, n_ss01=10, seed=0, with_shhs=False, opt_mods=()):
    """Token arrays of ``mod`` on a sample of the pool nights (``frac`` per unit, >= ``min_n``), ``n_ss01``
    MASS SS01 nights (expert arousals) and every DREAMS excerpt; shhs is left out. Nights must carry ``mod``
    and the ``extra_mods``; the ``opt_mods`` enter as ``P@<mod>`` where the night has them, NaN elsewhere."""
    from physioex.train.bin.concept_scan import _group, load_refs
    from physioex.train.bin.concept_words import token_features

    meta = json.load(open(os.path.join(dump, "meta.json")))
    scale = json.load(open(os.path.join(dump, "scale.json")))
    fs = meta["fs"][mod]
    rng = np.random.default_rng(seed)
    by_unit = {}
    for k, e in sorted(meta["nights"].items()):
        if mod in e["mods"] and all(m in e["mods"] for m in extra_mods):
            by_unit.setdefault(e["unit"], []).append(k)
    keys, sets = [], {}
    for u, ks in sorted(by_unit.items()):
        if u.startswith("shhs"):
            if not with_shhs:
                continue
            pick, s = ks, "shhs"
            keys += pick
            sets.update({k: s for k in pick})
            continue
        if u == "dreams":
            pick, s = ks, "dreams"
        elif u == "mass_ss01":
            pick, s = [str(k) for k in rng.choice(ks, min(n_ss01, len(ks)), replace=False)], "mass_ss01"
        else:
            nn = min(len(ks), max(min_n, int(round(frac * len(ks)))))
            pick, s = [str(k) for k in rng.choice(ks, nn, replace=False)], "pool"
        keys += pick
        sets.update({k: s for k in pick})
    keys = sorted(keys)
    A = {k: [] for k in ("P", "F", "S", "SF", "y", "night", "gname", "set", "logscale", "codes_P", "codes_E", "codes_N",
                         "Eq", "Nq")}
    X = {f"P@{m}": [] for m in tuple(extra_mods) + tuple(opt_mods)}
    X.update({f"rel_back@{m}": [] for m in extra_mods})
    X.update({f"Eq@{m}": [] for m in opt_mods})
    R = {r: [] for r in refs_needed}
    lens = []
    for i, k in enumerate(keys):
        z = np.load(os.path.join(dump, f"{k}.npz"))
        rf = load_refs(os.path.join(dump, "refs", f"{k}.npz"))
        n = len(z["y"])
        lens.append(n * T)
        x = z[f"x_{mod}"].astype(np.float32)
        A["P"].append(z[f"P_{mod}"].reshape(n * T, -1).astype(np.float32))
        A["F"].append(token_features(x, fs))
        A["S"].append(spec_windows(x, fs))
        if SPEC_FINE:
            A["SF"].append(spec_fine(x, fs))
        A["y"].append(np.repeat(z["y"].astype(np.int8), T))
        A["night"].append(np.full(n * T, i, np.int32))
        A["gname"].append(np.full(n * T, _group(meta["nights"][k]["unit"]), object))
        A["set"].append(np.full(n * T, sets[k], object))
        sc = scale.get(k, {}).get(mod, {}).get("scale")
        A["logscale"].append(np.full(n * T, np.log(sc) if sc else np.nan, np.float32))
        A["codes_P"].append(z[f"codes_{mod}_P"].reshape(n * T, -1)[:, :2].astype(np.int64))
        for s in ("E", "N"):
            c = f"codes_{mod}_{s}"
            if c in z.files:
                A[f"codes_{s}"].append(np.repeat(z[c][:, :2].astype(np.int64), T, 0))
        eqp = os.path.join(dump, "eq", f"{k}.npz")
        if os.path.exists(eqp):
            q = np.load(eqp)
            for s in ("E", "N"):
                if f"{s}q_{mod}" in q.files:
                    A[f"{s}q"].append(np.repeat(q[f"{s}q_{mod}"].astype(np.float32), T, 0))
        for m in extra_mods:
            X[f"P@{m}"].append(z[f"P_{m}"].reshape(n * T, -1).astype(np.float32))
            f0 = token_features(z[f"x_{m}"].astype(np.float32), meta["fs"][m])[:, 0]
            X[f"rel_back@{m}"].append(rel_back(f0, np.zeros(len(f0), np.int32)))
        qm = np.load(eqp) if os.path.exists(eqp) else None
        for m in opt_mods:
            X[f"P@{m}"].append(z[f"P_{m}"].reshape(n * T, -1).astype(np.float32) if f"P_{m}" in z.files else n * T)
            ok = qm is not None and f"Eq_{m}" in qm.files and len(qm[f"Eq_{m}"]) == n
            X[f"Eq@{m}"].append(np.repeat(qm[f"Eq_{m}"].astype(np.float32), T, 0) if ok else n * T)
        for r in refs_needed:
            R[r].append(rf[r] if r in rf and len(rf[r]) == n * T else None)
        if i % 20 == 0:
            print(f"[load] {i + 1}/{len(keys)} {k}", flush=True)
    for key in [f"{a}@{m}" for m in opt_mods for a in ("P", "Eq")]:  # nights without it: NaN rows
        parts = X[key]
        w = next((p.shape[1] for p in parts if not isinstance(p, int)), None)
        X[key] = [] if w is None else [np.full((p, w), np.nan, np.float32) if isinstance(p, int) else p
                                       for p in parts]
    D = {k: np.concatenate(v) for k, v in A.items() if v and len(v) == len(keys)}
    D.update({k: np.concatenate(v) for k, v in X.items() if v})
    D["gname"] = D["gname"].astype(str)
    D["set"] = D["set"].astype(str)
    D["refs"] = {}
    for r, parts in R.items():
        if any(p is not None for p in parts):
            D["refs"][r] = (np.concatenate([p if p is not None else np.zeros(l, bool) for p, l in zip(parts, lens)]),
                            np.concatenate([np.full(l, p is not None) for p, l in zip(parts, lens)]))
    finish(D, seed)
    return D, keys, sets


def finish(D, seed=0):
    """Context features, epoch spectrum, previous-epoch stage, context permutation, latent views."""
    F0 = D["F"][:, 0]
    D["rel_bg"] = rel_bg(F0, D["night"])
    D["rel_back"] = rel_back(F0, D["night"])
    D["iso"] = iso(F0, D["night"])
    D["spec_back"] = rel_back(D["S"], D["night"])
    ep = np.column_stack([D["S"][:, :6].reshape(-1, T, 6).mean(1), F0.reshape(-1, T).mean(1)])
    D["spec_ep"] = np.repeat(ep, T, 0).astype(np.float32)
    en = D["night"][::T]
    st = np.stack([D["y"][::T] == s for s in range(5)], 1).astype(np.float32)
    ep32 = ep.astype(np.float32)
    mu = ctx_mean(ep32, en, NB_EPOCHS)  # +-20 epochs, the epoch excluded
    sd = np.sqrt(np.maximum(ctx_mean(ep32 ** 2, en, NB_EPOCHS) - mu ** 2, 0))
    dif = np.abs(ep32 - np.r_[ep32[:1], ep32[:-1]]) * np.r_[False, en[1:] == en[:-1]][:, None]
    nb = np.column_stack([mu, sd, ctx_mean(dif, en, NB_EPOCHS)])  # label-free: what N could know without labels
    D["spec_nb"] = np.repeat(nb, T, 0).astype(np.float32)
    lab = ctx_mean(np.column_stack([ep32, st]), en, NB_EPOCHS)  # with the stage fractions: a declared ceiling only
    D["spec_nb_lab"] = np.repeat(lab, T, 0).astype(np.float32)
    yp = np.r_[D["y"][:T], D["y"][:-T]]
    same = np.r_[np.zeros(T, bool), D["night"][T:] == D["night"][:-T]]
    D["yprev"] = np.where(same, yp, D["y"])
    D["perm"] = epoch_perm(D["y"], D["night"], np.random.default_rng(seed + 11))
    D["_cache"] = {}
    _latent_views(D, seed)
    _p_ica(D, seed)


def _p_ica(D, seed=0, n_fit=200_000):
    """Label-free FastICA of P on pool tokens: the squares of P (and of its context means) are taken in this
    basis, the one of the lexicon's ICA words (energy along independent directions)."""
    from sklearn.decomposition import FastICA

    pool = np.flatnonzero(D["set"] == "pool")
    if len(pool) < 10 * D["P"].shape[1]:
        return
    rng = np.random.default_rng(seed + 5)
    fit = D["P"][rng.choice(pool, min(n_fit, len(pool)), replace=False)]
    D["_icaP"] = FastICA(D["P"].shape[1], random_state=seed, max_iter=400, whiten="unit-variance").fit(fit)


def _latent_views(D, seed=0):
    """PCA (``Eq`` / ``Nq``) and FastICA (``ICA_E`` / ``ICA_N``) of the epoch latents, fitted label-free on the
    pool epochs (one row per epoch)."""
    from sklearn.decomposition import PCA, FastICA

    pool = D["set"] == "pool"
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


def _ref_names(expr):
    if expr.startswith("post:"):
        return [expr.split(":")[1]]
    expr = expr.replace("onset:", "")
    return [p for p in re.split(r"[&|]", expr) if p and not p.startswith("y==")]


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


def _fmt(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def run(a):
    doc = json.load(open(a.specs))
    specs = [s for s in doc["specs"] if s["mod"] == a.mod and (not a.group or s.get("group") == a.group)]
    if a.terms:
        specs = [s for s in specs if s["term"] in a.terms.split(",")]
    refs = sorted({n for s in specs for e in [s["ref"], s.get("train_ref", ""), (s.get("candidates") or {}).get("ref", "")]
                   if e for n in _ref_names(e)})
    extra = sorted({i.split("@")[1] for s in specs for v in s["word_variants"] + [s["baselines"]] for i in v if "@" in i})
    t0 = time.time()
    shhs = any("shhs" in s.get("sets", []) for s in specs)
    D, keys, sets = load(a.dump, a.mod, refs, extra, a.frac, n_ss01=a.n_ss01, seed=a.seed, with_shhs=shhs)
    print(f"[rationale] {a.mod}: {len(D['y'])} tokens, {len(keys)} nights "
          f"({sum(v == 'pool' for v in sets.values())} pool), refs {sorted(D['refs'])}, extra {extra}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    res = []
    for sp in specs:
        f = run_epoch if sp.get("level") == "epoch" else run_tok
        r = f(D, sp, a.nulls, a.seed)
        res.append(r)
        if "error" in r:
            print(f"[rationale] {sp['term']}: {r['error']}", flush=True)
            continue
        key = "rho" if r.get("level") == "epoch" else "matched"
        b = r["baseline"].get(key) or {}
        parts = []
        for v in r["variants"]:
            c = v.get("vs_baseline") or {}
            parts.append(f"{'+'.join(v['inputs'])}{'(' + v['kind'] + ')' if v['kind'] != 'variant' else ''} "
                         f"{_fmt(v[key]['mean'])} d {_fmt(c.get('mean'))} p {_fmt(c.get('p_perm'))}"
                         + (f" auc {_fmt(v.get('auc_nights'))}" if "auc_nights" in v else ""))
        nl = r.get("null_matched") or r.get("null_rho") or {}
        print(f"[rationale] {sp['term']} ({sp['ref']}): n {r.get('n_tok', r.get('n_ep'))} "
              f"base {_fmt(r.get('base_rate', r.get('target_mean')))} baseline {key} {_fmt(b.get('mean'))} | "
              + " | ".join(parts) + f" | null p {_fmt(nl.get('p'))} | {(time.time() - t0) / 60:.1f} min", flush=True)
    cs = cross_specificity(D, res, doc.get("families", {}))
    for r in res:
        r.pop("_best_scores", None)
    out = os.path.join(a.dump, f"rationale_{a.tag}.json")
    json.dump({"mod": a.mod, "nights": keys, "sets": sets, "results": res, "cross_specificity": cs},
              open(out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
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
    r.add_argument("--group", default="")
    r.add_argument("--terms", default="")
    r.add_argument("--tag", required=True)
    r.add_argument("--frac", type=float, default=0.15)
    r.add_argument("--nulls", type=int, default=5)
    r.add_argument("--n-ss01", type=int, default=10)
    r.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args(argv)
    {"embed": embed, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
