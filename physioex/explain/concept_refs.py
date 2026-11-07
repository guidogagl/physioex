"""Reference masks of the lexicon concepts (``LEXICON_CONCETTI.md``, plan of 07.11.2026).

Every reference is a boolean mask over the 0.5 s tokens of the dumped epochs (``concept_scan dump``:
the scored epochs ``ts`` of the main sleep period, ``T = 60`` tokens per 30 s epoch); a few concepts
also have an epoch mask. A token is positive when the reference interval overlaps it (as
``vq_event_codes.token_labels``). Signals are the cleaned channels in µV.

Three kinds of reference, kept apart in :data:`CONCEPTS` (``strength``):

- ``expert``: annotations of experts (DREAMS K-complexes / spindles, MASS SS01 arousals);
- ``technician``: clinical annotations (NSRR / WSC arousals, respiratory events, limb movements);
- ``rule``: detectors on the signal written from the AASM definitions (amplitude, frequency,
  duration). They are a weak validation: a low agreement with a rule does not by itself make a
  codebook component wrong.

The spindle and slow-wave detectors follow the published criteria of YASA (Vallat & Walker 2021:
``spindles_detect`` relative sigma power, moving correlation and moving RMS, two of three; ``sw_detect``
negative / positive half-wave durations and amplitudes); they are re-implemented with scipy (YASA is
not in the Sofia environment), so they are "YASA-like", not YASA.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

TOK_S = 0.5
EPOCH_S = 30.0
T = int(EPOCH_S / TOK_S)

#: concept -> (modality, strength, short definition)
CONCEPTS = {
    "spindle": ("eeg", "expert|rule", "11-16 Hz train 0.5-2 s (DREAMS vis1/vis2; YASA-like detector)"),
    "k_complex": ("eeg", "expert", "negative sharp wave then positive, >= 0.5 s (DREAMS vis1/vis2)"),
    "slow_wave": ("eeg", "rule", "0.3-1.5 Hz, negative peak -40..-200 uV, peak-to-peak 75-350 uV (YASA-like)"),
    "vertex": ("eeg", "rule", "sharp negative transient < 0.5 s, >= 4x the epoch's median amplitude"),
    "sawtooth": ("eeg", "rule", "2-6 Hz train >= 1 s, relative power >= 0.4 of 0.5-30 Hz"),
    "alpha": ("eeg", "rule", "8-12 Hz dominant (token); epoch: > 50 % of the tokens"),
    "lamf": ("eeg", "rule", "4-8 Hz dominant at low amplitude (token)"),
    "delta": ("eeg", "rule", "0.5-2 Hz waves >= 75 uV peak-to-peak (token); epoch: >= 20 % of the epoch"),
    "arousal": ("eeg", "expert|technician", "annotated arousal (MASS SS01 experts; NSRR/WSC technicians)"),
    "resp_end": ("eeg", "technician", "end of an apnea/hypopnea, from -2 s to +5 s"),
    "plm_arousal": ("eeg", "technician", "arousal within 2 s of a limb movement"),
    "rem_eye": ("eog", "rule", "deflection >= 50 uV reached in < 0.5 s"),
    "sem": ("eog", "rule", "0.1-2 Hz half-wave of 0.5-2.5 s, >= 25 uV, peak >= 0.4 s after its start"),
    "blink": ("eog", "rule", "half-wave 0.15-0.5 s, >= 100 uV"),
    "atonia": ("emg", "rule", "token RMS <= 1.5x the 5th percentile of the night; epoch: > 50 % of the tokens"),
    "tma": ("emg", "rule", "burst < 0.25 s with RMS >= 4x the epoch's median 1 s RMS"),
    "emg_rise": ("emg", "rule", "3 s RMS >= 2x the median of the 10 s before"),
    "artifact": ("any", "rule|technician", "flat (< 0.5 uV), extreme amplitude, annotated artefact"),
}


# ---------------------------------------------------------------------------- helpers
def _sos(lo, hi, fs, order=4):
    from scipy.signal import butter

    if lo and hi:
        return butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return butter(order, lo or hi, btype="high" if lo else "low", fs=fs, output="sos")


def bandpass(x, lo, hi, fs, order=4):
    from scipy.signal import sosfiltfilt

    hi = min(hi, 0.45 * fs) if hi else hi
    return sosfiltfilt(_sos(lo, hi, fs, order), x)


def runs(m: np.ndarray) -> List[Tuple[int, int]]:
    """Start / end (exclusive) of the True runs of a boolean array."""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def merge_close(iv, gap):
    out = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def moving(x, w, fn):
    """``fn`` (``"rms"`` or ``"mean"``) over a centred window of ``w`` samples."""
    k = np.ones(max(1, int(w))) / max(1, int(w))
    if fn == "rms":
        return np.sqrt(np.convolve(x ** 2, k, mode="same"))
    return np.convolve(x, k, mode="same")


def contiguous(ts: np.ndarray) -> List[Tuple[int, int]]:
    """Runs of consecutive epochs in ``ts`` (positions ``[a, b)`` in the dumped arrays)."""
    br = np.flatnonzero(np.diff(ts) != 1) + 1
    edges = np.concatenate([[0], br, [len(ts)]])
    return list(zip(edges[:-1], edges[1:]))


def samples_to_tokens(iv, n_samples, fs) -> np.ndarray:
    """Token mask (any overlap) of sample intervals ``[a, b)``."""
    tok = int(round(TOK_S * fs))
    m = np.zeros(n_samples // tok, bool)
    for a, b in iv:
        m[a // tok:min(len(m), (b - 1) // tok + 1)] = True
    return m


def events_to_tokens(iv_sec, ts) -> np.ndarray:
    """Token mask of the dumped epochs ``ts`` for intervals in seconds from the recording start."""
    m = np.zeros(len(ts) * T, bool)
    pos = {int(t): i for i, t in enumerate(ts)}
    for a, b in iv_sec:
        ta, tb = int(np.floor(a / TOK_S)), int(np.ceil(b / TOK_S))
        for g in range(ta, max(ta + 1, tb)):
            i = pos.get(g // T)
            if i is not None:
                m[i * T + g % T] = True
    return m


def _per_run(x, ts, fs, fn):
    """Apply a sample-level detector (``fn(signal) -> intervals``) to every contiguous run of epochs
    and return the token mask over all dumped epochs."""
    spe = int(round(EPOCH_S * fs))
    out = np.zeros(len(ts) * T, bool)
    for a, b in contiguous(ts):
        sig = x[a:b].reshape(-1).astype(np.float64)
        out[a * T:b * T] = samples_to_tokens(fn(sig), len(sig), fs)[:(b - a) * T]
        assert len(sig) == (b - a) * spe
    return out


def _band_rel(sig, fs, bands, win_s=2.0, step_s=TOK_S):
    """Relative power of ``bands`` over 0.5-30 Hz in ``win_s`` windows centred on every token."""
    from numpy.lib.stride_tricks import sliding_window_view

    w, st = int(win_s * fs), int(step_s * fs)
    n_tok = len(sig) // st
    xp = np.pad(sig, (w // 2 - st // 2, w), mode="reflect")
    seg = sliding_window_view(xp, w)[::st][:n_tok] * np.hanning(w)
    p = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    f = np.fft.rfftfreq(w, 1 / fs)
    tot = p[:, (f >= 0.5) & (f < 30)].sum(1) + 1e-12
    return {k: p[:, (f >= lo) & (f < hi)].sum(1) / tot for k, (lo, hi) in bands.items()}, np.sqrt((seg ** 2).mean(1))


# ---------------------------------------------------------------------------- EEG detectors
def detect_spindles(sig, fs, min_dur=0.5, max_dur=2.0):
    """YASA-like (``spindles_detect`` defaults): sigma 12-15 Hz; relative sigma power (2 s windows,
    0.2 s step) >= 0.2, moving correlation (0.3 s) of sigma vs broadband >= 0.65, moving RMS (0.3 s)
    of sigma >= mean + 1.5 sd (10-90 % trimmed); a spindle where >= 2 of 3 hold, 0.5-2 s, events
    closer than 0.5 s merged."""
    from scipy.signal import stft

    broad = bandpass(sig, 1, 30, fs)
    sigma = bandpass(sig, 12, 15, fs)
    f, t, Z = stft(broad, fs, nperseg=int(2 * fs), noverlap=int(1.8 * fs), boundary=None, padded=False)
    p = np.abs(Z) ** 2
    rel = p[(f >= 12) & (f <= 15)].sum(0) / (p[(f >= 1) & (f <= 30)].sum(0) + 1e-12)
    rel = np.interp(np.arange(len(sig)) / fs, t, rel)
    w = int(0.3 * fs)
    mb, ms = moving(broad, w, "mean"), moving(sigma, w, "mean")
    cov = moving(broad * sigma, w, "mean") - mb * ms
    sd_b = np.sqrt(np.maximum(moving(broad ** 2, w, "mean") - mb ** 2, 1e-12))
    sd_s = np.sqrt(np.maximum(moving(sigma ** 2, w, "mean") - ms ** 2, 1e-12))
    corr = cov / (sd_b * sd_s)
    rms = moving(sigma, w, "rms")
    lo, hi = np.percentile(rms, [10, 90])
    tr = rms[(rms >= lo) & (rms <= hi)]
    thr = tr.mean() + 1.5 * tr.std()
    m = ((rel >= 0.2).astype(int) + (corr >= 0.65) + (rms >= thr)) >= 2
    iv = merge_close(runs(m), int(0.5 * fs))
    return [(a, b) for a, b in iv if min_dur * fs <= b - a <= max_dur * fs]


def _half_waves(x):
    """Sign runs of ``x``: (start, end, sign)."""
    s = np.sign(x)
    s[s == 0] = 1
    d = np.flatnonzero(np.diff(s) != 0) + 1
    e = np.concatenate([[0], d, [len(x)]])
    return [(a, b, int(s[a])) for a, b in zip(e[:-1], e[1:])]


def detect_slow_waves(sig, fs):
    """YASA-like (``sw_detect`` defaults): 0.3-1.5 Hz; negative half-wave 0.3-1.5 s with minimum in
    [-200, -40] uV, the following positive half-wave 0.1-1 s with maximum in [10, 150] uV,
    peak-to-peak 75-350 uV."""
    x = bandpass(sig, 0.3, 1.5, fs)
    hw = _half_waves(x)
    out = []
    for (a, b, s), (c, d, s2) in zip(hw[:-1], hw[1:]):
        if s >= 0 or s2 <= 0:
            continue
        neg, pos = x[a:b].min(), x[c:d].max()
        if (0.3 * fs <= b - a <= 1.5 * fs and 0.1 * fs <= d - c <= 1.0 * fs and -200 <= neg <= -40
                and 10 <= pos <= 150 and 75 <= pos - neg <= 350):
            out.append((a, d))
    return out


def detect_delta_waves(sig, fs):
    """AASM slow-wave activity: 0.5-2 Hz half-wave pairs with peak-to-peak >= 75 uV."""
    x = bandpass(sig, 0.5, 2.0, fs)
    hw = _half_waves(x)
    out = []
    for (a, b, s), (c, d, _) in zip(hw[:-1], hw[1:]):
        if s < 0 and x[c:d].max() - x[a:b].min() >= 75 and 0.25 * fs <= d - a <= 2.0 * fs:
            out.append((a, d))
    return out


def detect_vertex(sig, fs):
    """Sharp negative transients: 1-10 Hz negative half-wave of 50-250 ms whose minimum is below
    -4x the median absolute amplitude of its 30 s epoch."""
    x = bandpass(sig, 1, 10, fs)
    spe = int(EPOCH_S * fs)
    mad = np.repeat([np.median(np.abs(x[i:i + spe])) + 1e-9 for i in range(0, len(x), spe)], spe)[:len(x)]
    return [(a, b) for a, b, s in _half_waves(x)
            if s < 0 and 0.05 * fs <= b - a <= 0.25 * fs and x[a:b].min() <= -4 * mad[a]]


def detect_sawtooth(sig, fs):
    """2-6 Hz trains: tokens whose 2-6 Hz power is >= 0.4 of 0.5-30 Hz, runs >= 1 s."""
    rel, _ = _band_rel(sig, fs, {"st": (2, 6)})
    tok = int(TOK_S * fs)
    return [(a * tok, b * tok) for a, b in runs(rel["st"] >= 0.4) if b - a >= 2]


def eeg_background(sig, fs):
    """Token masks: ``alpha`` (8-12 Hz relative power larger than 4-8 and 12-30 Hz and >= 0.3) and
    ``lamf`` (4-8 Hz larger than 0.5-4 and 8-12 Hz, RMS below the night's median RMS)."""
    rel, rms = _band_rel(sig, fs, {"d": (0.5, 4), "t": (4, 8), "a": (8, 12), "b": (12, 30)})
    alpha = (rel["a"] >= 0.3) & (rel["a"] > rel["t"]) & (rel["a"] > rel["b"])
    lamf = (rel["t"] > rel["d"]) & (rel["t"] > rel["a"]) & (rms < np.median(rms))
    return alpha, lamf


# ---------------------------------------------------------------------------- EOG / EMG detectors
def detect_eye_movements(sig, fs):
    """Single EOG derivation, 0.3-5 Hz. ``rem_eye``: half-wave reaching >= 50 uV less than 0.5 s after
    its start; ``sem``: 0.1-2 Hz (order 2, to keep the slow deflections) half-wave of 0.5-2.5 s,
    >= 25 uV, peak reached >= 0.4 s after its start; ``blink``: half-wave 0.15-0.5 s, >= 100 uV.
    Without the second EOG channel conjugacy is not checked."""
    x = bandpass(sig, 0.3, 5, fs)
    rem, blink = [], []
    for a, b, _ in _half_waves(x):
        seg = np.abs(x[a:b])
        if len(seg) == 0:
            continue
        pk = int(np.argmax(seg))
        if seg[pk] >= 50 and pk < 0.5 * fs:
            rem.append((a, b))
        if seg[pk] >= 100 and 0.15 * fs <= b - a <= 0.5 * fs:
            blink.append((a, b))
    xs = bandpass(sig, 0.1, 2, fs, order=2)
    sem = [(a, b) for a, b, _ in _half_waves(xs) if 0.5 * fs <= b - a <= 2.5 * fs
           and np.abs(xs[a:b]).max() >= 25 and np.argmax(np.abs(xs[a:b])) >= 0.4 * fs]
    return {"rem_eye": rem, "sem": sem, "blink": blink}


def emg_masks(sig, fs, pct=5, factor=1.5):
    """Chin EMG (10 Hz high-pass). ``atonia``: token RMS <= ``factor`` x the night's ``pct``
    percentile (the lowest level of the night, AASM);
    ``tma``: 0.1 s RMS >= 4x the median 1 s RMS of its epoch, runs < 0.25 s; ``emg_rise``: 3 s RMS
    >= 2x the median 1 s RMS of the 10 s before."""
    x = bandpass(sig, 10, None, fs)
    tok = int(TOK_S * fs)
    n_tok = len(x) // tok
    trms = np.sqrt((x[:n_tok * tok].reshape(n_tok, tok) ** 2).mean(1))
    atonia = trms <= factor * np.percentile(trms, pct)
    r01, r1, r3 = moving(x, 0.1 * fs, "rms"), moving(x, fs, "rms"), moving(x, 3 * fs, "rms")
    spe = int(EPOCH_S * fs)
    base = np.repeat([np.median(r1[i:i + spe]) + 1e-9 for i in range(0, len(x), spe)], spe)[:len(x)]
    tma = [(a, b) for a, b in runs(r01 >= 4 * base) if b - a < 0.25 * fs]
    w10 = int(10 * fs)
    before = np.array([np.median(r1[max(0, i - w10):i]) if i > 0 else np.inf for i in range(0, len(x), tok)])
    rise_tok = r3[::tok][:n_tok] >= 2 * before[:n_tok]
    return atonia, tma, rise_tok


def artifact_mask(sig, fs, extreme=500.0):
    """Tokens flat (std < 0.5 uV) or with |x| >= ``extreme`` uV."""
    tok = int(TOK_S * fs)
    n = len(sig) // tok
    s = sig[:n * tok].reshape(n, tok)
    return (s.std(1) < 0.5) | (np.abs(s).max(1) >= extreme)


# ---------------------------------------------------------------------------- per night
def _events_iv(events, pred):
    return [(e["onset"], e["onset"] + e["dur"]) for e in events if pred(e)]


def night_refs(night: Dict[str, np.ndarray], meta: dict) -> Dict[str, np.ndarray]:
    """All reference masks of one dumped night: ``{name: (n * T,) bool}`` for the token masks and
    ``{name + "_epoch": (n,) bool}`` for the epoch masks. ``night``: the arrays of the dump (``ts``,
    ``x_<mod>``); ``meta``: its ``meta.json`` entry plus ``fs``."""
    ts = night["ts"]
    n = len(ts)
    ev = meta.get("events", [])
    fs = meta["fs"]
    out = {}
    # annotations
    for name, pred in (
            ("spindle_vis1", lambda e: e["type"] == "spindle" and e["concept"] == "vis1"),
            ("spindle_vis2", lambda e: e["type"] == "spindle" and e["concept"] == "vis2"),
            ("k_complex_vis1", lambda e: e["type"] == "k_complex" and e["concept"] == "vis1"),
            ("k_complex_vis2", lambda e: e["type"] == "k_complex" and e["concept"] == "vis2"),
            ("arousal_ann", lambda e: e["type"] == "arousal"),
            ("artifact_ann", lambda e: e["type"] == "artifact" or "artefact" in e["concept"].lower())):
        iv = _events_iv(ev, pred)
        if iv:
            out[name] = events_to_tokens(iv, ts)
    resp = [(e["onset"] + e["dur"] - 2, e["onset"] + e["dur"] + 5) for e in ev
            if e["type"] == "respiratory" and any(k in e["concept"].lower() for k in ("apnea", "hypopnea"))]
    if resp:
        out["resp_end"] = events_to_tokens(resp, ts)
    lm = _events_iv(ev, lambda e: e["type"] == "limb_movement")
    ar = _events_iv(ev, lambda e: e["type"] == "arousal")
    if lm and ar:
        lm_s = np.array(sorted(a for a, _ in lm))
        plm = [(a, b) for a, b in ar if len(lm_s) and np.min(np.abs(lm_s - a)) <= 2.0]
        if plm:
            out["plm_arousal"] = events_to_tokens(plm, ts)
    # EEG
    if "x_eeg" in night:
        x, f = night["x_eeg"].astype(np.float32), fs["eeg"]
        out["spindle_rule"] = _per_run(x, ts, f, lambda s: detect_spindles(s, f))
        out["slow_wave"] = _per_run(x, ts, f, lambda s: detect_slow_waves(s, f))
        out["vertex"] = _per_run(x, ts, f, lambda s: detect_vertex(s, f))
        out["sawtooth"] = _per_run(x, ts, f, lambda s: detect_sawtooth(s, f))
        dw = _per_run(x, ts, f, lambda s: detect_delta_waves(s, f))
        out["delta"] = dw
        out["delta_epoch"] = _epoch_cover(x, ts, f, lambda s: detect_delta_waves(s, f)) >= 0.2
        al, la = np.zeros(n * T, bool), np.zeros(n * T, bool)
        for a, b in contiguous(ts):
            s = x[a:b].reshape(-1).astype(np.float64)
            al_r, la_r = eeg_background(s, f)
            al[a * T:b * T], la[a * T:b * T] = al_r[:(b - a) * T], la_r[:(b - a) * T]
        out["alpha"], out["lamf"] = al, la
        out["alpha_epoch"] = al.reshape(n, T).mean(1) > 0.5
        out["artifact_eeg"] = _tok_fn(x, ts, f, artifact_mask)
    if "x_eog" in night:
        x, f = night["x_eog"].astype(np.float32), fs["eog"]
        for k in ("rem_eye", "sem", "blink"):
            out[k] = _per_run(x, ts, f, lambda s, k=k: detect_eye_movements(s, f)[k])
    if "x_emg" in night:
        x, f = night["x_emg"].astype(np.float32), fs["emg"]
        at, tma, rise = np.zeros(n * T, bool), np.zeros(n * T, bool), np.zeros(n * T, bool)
        for a, b in contiguous(ts):
            s = x[a:b].reshape(-1).astype(np.float64)
            at_r, tma_iv, rise_r = emg_masks(s, f)
            at[a * T:b * T] = at_r[:(b - a) * T]
            tma[a * T:b * T] = samples_to_tokens(tma_iv, len(s), f)[:(b - a) * T]
            rise[a * T:b * T] = rise_r[:(b - a) * T]
        out["atonia"], out["tma"], out["emg_rise"] = at, tma, rise
        out["atonia_epoch"] = at.reshape(n, T).mean(1) > 0.5
    return out


def _epoch_cover(x, ts, fs, fn):
    """Fraction of every epoch covered by the intervals of ``fn``."""
    spe = int(round(EPOCH_S * fs))
    cov = np.zeros(len(ts))
    for a, b in contiguous(ts):
        sig = x[a:b].reshape(-1).astype(np.float64)
        m = np.zeros(len(sig), bool)
        for i, j in fn(sig):
            m[i:j] = True
        cov[a:b] = m.reshape(b - a, spe).mean(1)
    return cov


def _tok_fn(x, ts, fs, fn):
    out = np.zeros(len(ts) * T, bool)
    for a, b in contiguous(ts):
        out[a * T:b * T] = fn(x[a:b].reshape(-1).astype(np.float64), fs)[:(b - a) * T]
    return out
