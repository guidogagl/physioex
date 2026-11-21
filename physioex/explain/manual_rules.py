"""Rule detectors compiled from the manual knowledge cards (M1 of xmai-lab/sleep-lens#12, pilot #14).

The cards (``sleep-manuals`` KB, pass B4) give the criteria; this module turns them into detectors on one
central EEG, one EOG and the chin EMG of the v3 cache (uV). Thresholds the manuals do not give are keyword
arguments marked "ours". Sources: [F] Fundamentals of Sleep Medicine 2012 (ch1, ch3, ch4), [K] Kryger atlas
2024 ch13, [Th] Thomas atlas 2023, AASM via [F]. Limits of the single-channel cache: no frontal / occipital
location criteria, no EOG conjugacy, stage gating uses the scored hypnogram.

Concepts (``CONCEPTS_M1``): arousal and awakening, alpha bursts / intrusion and alpha-delta epochs, K complexes
in N2-N3 and those followed by an arousal, vertex sharp waves at sleep onset, sawtooth waves in R and those
before rapid eye movements, slow eye movements in drowsiness and N1.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from physioex.explain.concept_refs import (T, TOK_S, bandpass, contiguous, detect_k_complex, eeg_background,
                                           moving, night_refs, runs, samples_to_tokens)

#: key -> (modality, definition with source)
CONCEPTS_M1 = {
    "arousal_m1": ("eeg+emg", "abrupt shift to theta, alpha or > 16 Hz (not spindles) >= 3 s and < 15 s after "
                              ">= 10 s of sleep; in R with a chin EMG rise >= 1 s [F ch3 p19, K ch13, Th]"),
    "awakening_m1": ("eeg", "as arousal_m1 but >= 15 s [K: arousal is 3 s but less than 15 s]"),
    "alpha_burst_m1": ("eeg", "alpha run < 3 s in N2/N3 outside arousals [F ch4]"),
    "alpha_intrusion_m1": ("eeg", "alpha bursts at more than one per 10 s in NREM, not scored as arousals "
                                  "[F ch4, K]"),
    "alpha_delta_m1_epoch": ("eeg", "N3 epoch with alpha superimposed on delta in > 50 % of the tokens [F ch4]"),
    "kc_m1": ("eeg", "negative sharp wave then positive component, >= 0.5 s, standing out, N2 (also N3) "
                     "[K p441, F ch1]"),
    "kc_arousal_m1": ("eeg", "K complex followed by an arousal starting <= 1 s after its end [AASM via F ch3]"),
    "vertex_m1": ("eeg", "sharp wave < 0.5 s, central, in N1 and at the N1-N2 transition [F ch4, Th p134]"),
    "sawtooth_m1": ("eeg", "2-6 Hz triangular trains, central, in R [F ch1 p7, p36]"),
    "sawtooth_rem_m1": ("eeg+eog", "sawtooth train followed by a rapid eye movement within 2 s "
                                   "[F: frequently precede bursts of REMs]"),
    "sem_m1": ("eog", "conjugate sinusoidal eye movement with initial deflection > 500 ms, in eyes-closed W, "
                      "N1 and the first N2 epoch after N1 [F ch1 p8, Th p134]"),
}

AROUSAL_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (16.0, 30.0)}
SIGMA = (12.0, 16.0)


# ---------------------------------------------------------------------------- band power
def band_power_tokens(sig, fs, bands, win_s=2.0):
    """Absolute power of ``bands`` (``{name: (lo, hi)}``, Hz) per 0.5 s token, Hann window of ``win_s`` centred
    on the token (as ``concept_refs._band_rel``, absolute instead of relative)."""
    from numpy.lib.stride_tricks import sliding_window_view

    w, st = int(round(win_s * fs)), int(round(TOK_S * fs))
    n_tok = len(sig) // st
    xp = np.pad(np.asarray(sig, np.float64), (w // 2 - st // 2, w), mode="reflect")
    seg = sliding_window_view(xp, w)[::st][:n_tok] * np.hanning(w)
    p = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    f = np.fft.rfftfreq(w, 1 / fs)
    return {k: p[:, (f >= lo) & (f < hi)].sum(1) for k, (lo, hi) in bands.items()}


def _median_before(v, n):
    """Median of the ``n`` values before each position (NaN where fewer than ``n`` exist)."""
    from numpy.lib.stride_tricks import sliding_window_view

    import warnings

    vp = np.concatenate([np.full(n, np.nan), v])
    with warnings.catch_warnings():  # the first n positions are all-NaN on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(sliding_window_view(vp, n)[:len(v)], axis=1) if len(v) else v.astype(float)


def _tok_iv(iv, fs):
    tok = int(round(TOK_S * fs))
    return [(a // tok, -(-b // tok)) for a, b in iv]


# ---------------------------------------------------------------------------- arousal
def _emg_rise_inside(emg, fs, a, b, base_s=10.0, factor=2.0, min_s=1.0):  # factor ours
    """True if the chin 0.1 s RMS (10 Hz high-pass) is >= ``factor`` x the median 1 s RMS of the ``base_s``
    before sample ``a`` for >= ``min_s`` inside ``[a, b)`` (the EMG increase of a REM arousal [F ch3 p19])."""
    lo = max(0, a - int(base_s * fs))
    x = bandpass(emg[lo:b], 10, None, fs)
    r01, r1 = moving(x, 0.1 * fs, "rms"), moving(x, fs, "rms")
    base = np.median(r1[:a - lo]) if a > lo else np.inf
    up = r01[a - lo:] >= factor * base
    return any(e - s >= min_s * fs for s, e in runs(up))


def detect_arousal(sig, fs, emg=None, stages_tok=None, k=3.0, base_s=10.0, min_s=3.0, max_s=15.0, gap_s=0.5,
                   emg_min_s=1.0, win_s=2.0):
    """Arousals [AASM via F ch3 p19; K ch13; Th]: "abrupt shift of EEG frequency including alpha, theta and/or
    frequencies greater than 16 Hz (but not spindles) that lasts at least 3 s, with at least 10 s of stable
    sleep preceding the change; in stage R a concurrent increase in submental EMG lasting at least 1 s".

    - shift token: theta (4-8), alpha (8-12) or beta (16-30 Hz) power (``win_s`` Hann window, ours) >= ``k``
      (ours; 3 and 2 s: no event in 30 min of 1/f noise, 2 s bursts rejected, 5 s kept) x the median of that
      band over the ``base_s`` before, and >= the 12-16 Hz power of the token
      (ours: leakage of a spindle into the neighbouring bands is not a shift). 12-16 Hz is never a shift band.
    - event: from a shift token, extended while any band stays >= ``k`` x the baseline frozen at the onset,
      gaps <= ``gap_s`` bridged. Duration = token span minus ``win_s - 0.5`` s (the window widens the run, ours).
    - kept when >= ``min_s``, its onset not in W, the ``base_s`` before contain no W token and no other event;
      onset in R: chin EMG rise >= ``emg_min_s`` inside the event (no EMG: dropped).
    - longer than ``max_s`` -> ``awakening`` [K: 3 s but less than 15 s].

    ``sig``, ``emg``: uV, same samples; ``stages_tok``: stage per 0.5 s token (None: no gating).
    Returns ``{"arousal": [(a, b)], "awakening": [(a, b)]}`` in samples."""
    p = band_power_tokens(sig, fs, {**AROUSAL_BANDS, "sigma": SIGMA}, win_s)
    n = len(p["sigma"])
    nb, gap = int(round(base_s / TOK_S)), int(round(gap_s / TOK_S))
    P = np.stack([p[b] for b in AROUSAL_BANDS])  # (3, n)
    base = np.stack([_median_before(v, nb) for v in P])
    shift = ((P >= k * base) & (P >= p["sigma"])).any(0)
    tok = int(round(TOK_S * fs))
    widen = win_s - TOK_S
    arousal, awakening, last_end = [], [], -10 ** 9
    t = nb
    while t < n:
        if not shift[t]:
            t += 1
            continue
        b0 = base[:, t]
        last, e = t, t + 1
        while e < n and e - last <= gap + 1:
            if ((P[:, e] >= k * b0) & (P[:, e] >= p["sigma"][e])).any():
                last = e
            e += 1
        a_t, b_t = t, last + 1
        t = b_t
        dur = (b_t - a_t) * TOK_S - widen
        if dur < min_s:
            continue
        prev_end, last_end = last_end, b_t  # only events of >= min_s break the stable sleep before the next
        if a_t - prev_end < nb:
            continue
        if stages_tok is not None:
            st = stages_tok[a_t]
            if st == 0 or (stages_tok[a_t - nb:a_t] == 0).any():
                continue
            if st == 4 and (emg is None or not _emg_rise_inside(emg, fs, a_t * tok, b_t * tok, base_s,
                                                                min_s=emg_min_s)):
                continue
        sh = int(round(widen / 2 * fs))
        iv = (a_t * tok + sh, max(a_t * tok + sh + 1, b_t * tok - sh))
        (awakening if dur >= max_s else arousal).append(iv)
    return {"arousal": arousal, "awakening": awakening}


# ---------------------------------------------------------------------------- alpha
def detect_alpha_intrusion(sig, fs, stages_tok, arousal_iv, max_burst_s=3.0, rate_per_10s=1.0, win_s=10.0):
    """Alpha intrusion [F ch4; K]: "intrusions of alpha activity of less than 3 s into NREM sleep at a rate
    greater than one burst per 10 s are not scored as arousals". Alpha token = ``eeg_background`` alpha
    (8-13 Hz dominant). ``alpha_burst``: runs of alpha tokens inside N2/N3 tokens, shorter than ``max_burst_s``,
    not overlapping ``arousal_iv`` (samples). ``alpha_intrusion``: the bursts with more than ``rate_per_10s``
    burst onsets (itself included) in some ``win_s`` window containing their onset, i.e. another onset
    less than ``win_s`` away."""
    alpha, _ = eeg_background(sig, fs)
    n = len(alpha)
    nrem = np.ones(n, bool) if stages_tok is None else np.isin(stages_tok[:n], (2, 3))
    block = np.zeros(n, bool)
    for a, b in _tok_iv(arousal_iv, fs):
        block[a:b] = True
    tok = int(round(TOK_S * fs))
    bursts = [(a, b) for a, b in runs(alpha & nrem) if (b - a) * TOK_S < max_burst_s and not block[a:b].any()]
    on = np.array([a for a, _ in bursts])
    w = int(round(win_s / TOK_S))
    keep = [(a, b) for a, b in bursts if (np.abs(on - a) < w).sum() > rate_per_10s]
    return {"alpha_burst": [(a * tok, b * tok) for a, b in bursts],
            "alpha_intrusion": [(a * tok, b * tok) for a, b in keep]}


def alpha_delta_epochs(sig, fs, stages_ep, share=0.4, min_cover=0.5):
    """Alpha-delta sleep [F ch4: alpha activity superimposed on delta in N3]: N3 epochs where in more than
    ``min_cover`` of the tokens the 8-12 Hz power is >= ``share`` (ours) of the 4-30 Hz power."""
    p = band_power_tokens(sig, fs, {"alpha": (8.0, 12.0), "fast": (4.0, 30.0)})
    ok = p["alpha"] >= share * (p["fast"] + 1e-12)
    n_ep = len(ok) // T
    cover = ok[:n_ep * T].reshape(n_ep, T).mean(1)
    st = np.asarray(stages_ep[:n_ep]) if stages_ep is not None else np.full(n_ep, 3)
    return (cover > min_cover) & (st == 3)


# ---------------------------------------------------------------------------- K complex + arousal
def detect_kc_context(kc_iv, arousal_iv, fs, max_gap_s=1.0):
    """K complexes followed by an arousal [AASM via F ch3: the arousal commences no more than 1 s after the
    termination of the K complex]: arousal onset in ``[kc start, kc end + max_gap_s]`` (samples)."""
    on = np.array(sorted(a for a, _ in arousal_iv))
    g = int(round(max_gap_s * fs))
    out = [(a, b) for a, b in kc_iv if len(on) and ((on >= a) & (on <= b + g)).any()]
    return {"kc_arousal": out}


# ---------------------------------------------------------------------------- per night
def _uv(night, sc, m):
    if f"x_{m}" not in night or m not in sc:
        return None
    med, k = sc[m]
    return night[f"x_{m}"].astype(np.float32) * np.float32(k) + np.float32(med)


def _first_n2_after_n1(y, ts):
    """Epoch mask: N2 epochs whose contiguous previous epoch is N1."""
    out = np.zeros(len(y), bool)
    out[1:] = (y[1:] == 2) & (y[:-1] == 1) & (np.diff(ts) == 1)
    return out


def _adjacent_n1(y, ts):
    """Epoch mask: N2 epochs with a contiguous N1 neighbour (previous or next)."""
    n1 = y == 1
    nb = np.zeros(len(y), bool)
    c = np.diff(ts) == 1
    nb[1:] |= n1[:-1] & c
    nb[:-1] |= n1[1:] & c
    return (y == 2) & nb


def _followed_by(mask, other, within):
    """Runs of ``mask`` with an ``other`` token in the ``within`` tokens after their end."""
    out = np.zeros_like(mask)
    for a, b in runs(mask):
        if other[b:b + within].any():
            out[a:b] = True
    return out


def night_m1(night: Dict[str, np.ndarray], meta: dict) -> Dict[str, np.ndarray]:
    """M1 masks of one dumped night (same inputs as :func:`concept_refs.night_refs`): token masks
    ``(len(ts) * T,)`` for the keys of ``CONCEPTS_M1`` and the epoch mask ``alpha_delta_m1_epoch``
    ``(len(ts),)``. Concepts whose modality has no amplitude scale are left out."""
    ts = np.asarray(night["ts"])
    n = len(ts)
    y = night.get("y")
    y = None if y is None else np.asarray(y).astype(np.int8)
    fs, sc = meta["fs"], meta.get("scale", {})
    refs = night_refs(night, meta)
    stok = None if y is None else np.repeat(y, T)

    def in_stage(*s):
        return np.ones(n * T, bool) if y is None else np.isin(stok, s)

    def ep(mask_ep):
        return np.ones(n * T, bool) if y is None else np.repeat(mask_ep, T)

    out = {}
    eeg, emg = _uv(night, sc, "eeg"), night.get("x_emg")
    if emg is not None:
        emg = _uv(night, sc, "emg") if "emg" in sc else emg.astype(np.float32)  # relative rule: scale-free
    if eeg is not None:
        f = float(fs["eeg"])
        keys = ("arousal_m1", "awakening_m1", "alpha_burst_m1", "alpha_intrusion_m1", "kc_m1", "kc_arousal_m1")
        for kk in keys:
            out[kk] = np.zeros(n * T, bool)
        ad = np.zeros(n, bool)
        for a, b in contiguous(ts):
            s = eeg[a:b].reshape(-1).astype(np.float64)
            m = emg[a:b].reshape(-1).astype(np.float64) if emg is not None and emg.shape == eeg.shape else None
            st = None if stok is None else stok[a * T:b * T]
            ar = detect_arousal(s, f, m, st)
            ai = detect_alpha_intrusion(s, f, st, ar["arousal"] + ar["awakening"])
            kc = detect_k_complex(s, f)
            kca = detect_kc_context(kc, ar["arousal"], f)
            for kk, iv in (("arousal_m1", ar["arousal"]), ("awakening_m1", ar["awakening"]),
                           ("alpha_burst_m1", ai["alpha_burst"]), ("alpha_intrusion_m1", ai["alpha_intrusion"]),
                           ("kc_m1", kc), ("kc_arousal_m1", kca["kc_arousal"])):
                out[kk][a * T:b * T] = samples_to_tokens(iv, len(s), f)[:(b - a) * T]
            ad[a:b] = alpha_delta_epochs(s, f, None if y is None else y[a:b])
        out["kc_m1"] &= in_stage(2, 3)
        out["kc_arousal_m1"] &= in_stage(2, 3)
        out["alpha_delta_m1_epoch"] = ad
    if "vertex" in refs:  # refs gate it to N1/N2
        out["vertex_m1"] = refs["vertex"] & (in_stage(1) | (ep(_adjacent_n1(y, ts)) if y is not None else True))
    if "sawtooth" in refs:  # refs gate it to R
        out["sawtooth_m1"] = refs["sawtooth"]
        if "rem_eye" in refs:
            out["sawtooth_rem_m1"] = _followed_by(refs["sawtooth"], refs["rem_eye"], int(round(2.0 / TOK_S)))
    if "sem" in refs:
        if y is None:
            out["sem_m1"] = refs["sem"]
        else:
            closed_w = (y == 0) & refs["alpha_epoch"] if "alpha_epoch" in refs else np.zeros(n, bool)
            ok = (y == 1) | closed_w | _first_n2_after_n1(y, ts)
            out["sem_m1"] = refs["sem"] & ep(ok)
    return out


__all__ = ["CONCEPTS_M1", "band_power_tokens", "detect_arousal", "detect_alpha_intrusion", "alpha_delta_epochs",
           "detect_kc_context", "night_m1"]
