"""P3 shared library: synthetic PSG generator, parametric class dictionaries (burst / transient /
slow wave), greedy explaining-away inference with NMS, periodicity cap and amplitude cap, masked
background estimators, night gain g. Pure numpy/scipy, CPU. Units: RobustScale (background MAD ~ 1).

Only aggregate numbers leave this module; no raw signal is written or plotted anywhere.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps
from scipy.special import gammaincinv

FS = 100
EPOCH = 30 * FS
CACHE = "/mnt/data/guido/work/cache/samples/v1"
BG_BANDS = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (11.0, 16.0), (16.0, 30.0))  # msvq.BANDS


# ------------------------------------------------------------------ synthetic generator
def _gabor(n, f, d, phase=0.0, fs=FS):
    t = (np.arange(n) - n // 2) / fs
    env = np.exp(-0.5 * (t / (d / 4)) ** 2)
    return env * np.cos(2 * np.pi * f * t + phase)


def one_over_f(n, alpha=1.3, rng=None, f_lo=0.3, fs=FS):
    """Coloured Gaussian background, MAD-normalised to 1, PSD ~ 1/f^alpha above f_lo."""
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = 1.0 / np.maximum(f, f_lo) ** (alpha / 2)
    shape[0] = 0
    x = np.fft.irfft(F * shape, n)
    return x / (np.median(np.abs(x - np.median(x))) / 0.6745)


def kc_shape(n, width=0.12, fs=FS):
    """Biphasic K-complex-like transient (sharp negative, slower positive), zero mean, ~0.8 s."""
    t = (np.arange(n) - n // 2) / fs
    s = -np.exp(-0.5 * (t / width) ** 2) + 0.6 * np.exp(-0.5 * ((t - 0.35) / (2.2 * width)) ** 2)
    s -= s.mean()
    return s / np.abs(s).max()


def sw_shape(n, T=1.5, fs=FS):
    """One full slow-wave cycle (negative half first), zero mean, duration T."""
    t = (np.arange(n) - n // 2) / fs
    s = np.where(np.abs(t) < T / 2, -np.sin(2 * np.pi * (t + T / 2) / T), 0.0)
    return s / np.abs(s).max()


@dataclass
class SynthEpoch:
    x: np.ndarray
    bg: np.ndarray                 # pure background (truth)
    events: list = field(default_factory=list)  # dicts: cls, kind, tau (s), dur, amp, f


def synth_epoch(stage="N2", rng=None, n_spindle=2, n_kc=1, n_sw=0, n_alpha=0, ecg=False,
                spindle_f=(11, 13, 15), sp_amp=(0.8, 2.5), kc_amp=(3, 6), sw_amp=(2, 5), alpha=1.3):
    rng = rng or np.random.default_rng()
    bg = one_over_f(EPOCH, alpha=alpha, rng=rng)
    x = bg.copy()
    ev = []
    n = EPOCH
    busy = np.zeros(n, bool)

    def place(dur, margin=0.25):
        for _ in range(50):
            tau = rng.uniform(0.5 + dur / 2, 29.5 - dur / 2)
            i0, i1 = int((tau - dur / 2 - margin) * FS), int((tau + dur / 2 + margin) * FS)
            if not busy[max(i0, 0):i1].any():
                busy[max(i0, 0):i1] = True
                return tau
        return None

    for _ in range(n_sw):
        T = rng.uniform(1.0, 2.5)
        tau = place(T, 0.05)
        if tau is None:
            continue
        a = rng.uniform(*sw_amp)
        x += a * np.roll(sw_shape(n, T), int(tau * FS) - n // 2)
        ev.append(dict(cls="S", kind="sw", tau=tau, dur=T, amp=a, f=1 / T))
    for _ in range(n_kc):
        tau = place(1.0)
        if tau is None:
            continue
        a = rng.uniform(*kc_amp)
        x += a * np.roll(kc_shape(n), int(tau * FS) - n // 2)
        ev.append(dict(cls="T", kind="kc", tau=tau, dur=0.8, amp=a, f=1.0))
    for _ in range(n_spindle):
        f = float(rng.choice(spindle_f))
        d = rng.uniform(0.6, 1.5)
        tau = place(d)
        if tau is None:
            continue
        a = rng.uniform(*sp_amp)
        x += a * np.roll(_gabor(n, f, d, rng.uniform(0, 2 * np.pi)), int(tau * FS) - n // 2)
        ev.append(dict(cls="B", kind="spindle", tau=tau, dur=d, amp=a, f=f))
    for _ in range(n_alpha):
        d = rng.uniform(2, 5)
        tau = place(d)
        if tau is None:
            continue
        a = rng.uniform(1.0, 2.0)
        x += a * np.roll(_gabor(n, 10.0, d, rng.uniform(0, 2 * np.pi)), int(tau * FS) - n // 2)
        ev.append(dict(cls="B", kind="alpha", tau=tau, dur=d, amp=a, f=10.0))
    if ecg:
        period = rng.uniform(0.85, 1.1)
        t0 = rng.uniform(0, period)
        a = rng.uniform(1.5, 2.5)
        t = (np.arange(n) - n // 2) / FS
        r = -t / 0.04 * np.exp(-0.5 * (t / 0.04) ** 2)
        r /= np.abs(r).max()
        while t0 < 30:
            x += a * np.roll(r, int(t0 * FS) - n // 2)
            ev.append(dict(cls="T", kind="ecg", tau=t0, dur=0.1, amp=a, f=1 / period))
            t0 += period * (1 + 0.02 * rng.standard_normal())
    return SynthEpoch(x=x, bg=bg, events=ev)


def synth_stage(stage, rng, **kw):
    if stage == "N2":
        return synth_epoch("N2", rng, n_spindle=2, n_kc=1, n_sw=1, **kw)
    if stage == "N3":
        return synth_epoch("N3", rng, n_spindle=1, n_kc=1, n_sw=14, **kw)
    if stage == "W":
        return synth_epoch("W", rng, n_spindle=0, n_kc=0, n_sw=0, n_alpha=3, **kw)
    raise ValueError(stage)


# ------------------------------------------------------------------ dictionaries
@dataclass
class Atom:
    cls: str          # "B" burst, "T" transient, "S" slow wave
    name: str
    wave: np.ndarray  # unit norm, centred
    quad: np.ndarray | None  # analytic quadrature (bursts only)
    dur: float
    f: float
    band: tuple | None = None


BURST_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 11.0), "sigma": (11.0, 16.0), "beta": (16.0, 30.0)}


def _unit(w):
    w = w - w.mean()
    return w / (np.linalg.norm(w) + 1e-12)


def burst_atoms(f_step=0.5, durs=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0), alpha_long=(5.0, 8.0), min_cycles=3.0):
    atoms = []
    for bname, (lo, hi) in BURST_BANDS.items():
        step = 1.0 if bname == "beta" else f_step
        for f in np.arange(lo, hi, step):
            ds = list(durs) + (list(alpha_long) if bname == "alpha" else [])
            for d in ds:
                if d * f < min_cycles:
                    continue
                n = int(d * FS) | 1
                n = min(n, EPOCH)
                t = (np.arange(n) - n // 2) / FS
                env = np.hanning(n)
                c = _unit(env * np.cos(2 * np.pi * f * t))
                s = _unit(env * np.sin(2 * np.pi * f * t))
                atoms.append(Atom("B", f"{bname}_{f:.1f}Hz_{d:.2f}s", c, s, d, f, (lo, hi)))
    return atoms


def transient_atoms(widths=(0.05, 0.08, 0.12, 0.18)):
    """Zero-mean, <= 1 s, low-pass (Gaussian derivatives order 1-2 and KC-like biphasic)."""
    atoms = []
    for w in widths:
        n = min(int(8 * w * FS) | 1, FS)  # <= 1 s
        t = (np.arange(n) - n // 2) / FS
        g = np.exp(-0.5 * (t / w) ** 2)
        atoms.append(Atom("T", f"dg1_{w:.2f}", _unit(-t / w * g), None, n / FS, 1 / (4 * w)))
        atoms.append(Atom("T", f"dg2_{w:.2f}", _unit((1 - (t / w) ** 2) * g), None, n / FS, 1 / (4 * w)))
    for w in (0.10, 0.14):
        n = FS - 1  # 0.99 s
        atoms.append(Atom("T", f"kc_{w:.2f}", _unit(kc_shape(n, w)), None, n / FS, 1.0))
    return atoms


def slowwave_atoms(periods=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5)):
    atoms = []
    for T in periods:
        n = int(T * FS) | 1
        atoms.append(Atom("S", f"sw_{T:.2f}", _unit(sw_shape(n, T) * np.hanning(n) ** 0.5), None, T, 1 / T))
    for T in (1.25, 2.0):  # sharp negative then slow positive (asymmetric)
        n = int(T * FS) | 1
        t = (np.arange(n) - n // 2) / FS
        s = -np.exp(-0.5 * ((t + T / 4) / (T / 8)) ** 2) + 0.5 * np.exp(-0.5 * ((t - T / 6) / (T / 4)) ** 2)
        atoms.append(Atom("S", f"swasym_{T:.2f}", _unit(s * np.hanning(n) ** 0.5), None, T, 1 / T))
    return atoms


def dictionaries():
    return burst_atoms() + transient_atoms() + slowwave_atoms()


def perturb(atoms, rng, f_jit=0.3, noise=0.05):
    out = []
    for a in atoms:
        if a.cls == "B":
            f = a.f + rng.uniform(-f_jit, f_jit)
            n = len(a.wave)
            t = (np.arange(n) - n // 2) / FS
            env = np.hanning(n)
            out.append(Atom(a.cls, a.name, _unit(env * np.cos(2 * np.pi * f * t)),
                            _unit(env * np.sin(2 * np.pi * f * t)), a.dur, f, a.band))
        else:
            w = a.wave + noise * rng.standard_normal(len(a.wave)) * np.abs(a.wave).max()
            out.append(Atom(a.cls, a.name, _unit(w), None, a.dur, a.f, a.band))
    return out


# ------------------------------------------------------------------ inference
def _corr_all(r, atoms, nfft):
    """Correlation of r with every atom (centred), same-length output; bursts analytic magnitude."""
    R = np.fft.rfft(r, nfft)
    n = len(r)
    out = np.empty((len(atoms), n))
    phase = np.zeros((len(atoms), n))
    for k, a in enumerate(atoms):
        m = len(a.wave)
        W = np.fft.rfft(a.wave[::-1], nfft)
        c = np.fft.irfft(R * W, nfft)[m // 2:m // 2 + n]
        if a.quad is not None:
            Q = np.fft.rfft(a.quad[::-1], nfft)
            s = np.fft.irfft(R * Q, nfft)[m // 2:m // 2 + n]
            out[k] = np.hypot(c, s)
            phase[k] = np.arctan2(s, c)
        else:
            out[k] = c
    return out, phase


def _local_energy(r, atoms, nfft):
    n = len(r)
    R2 = np.fft.rfft(r * r, nfft)
    out = np.empty((len(atoms), n))
    for k, a in enumerate(atoms):
        m = len(a.wave)
        box = np.ones(m)
        out[k] = np.fft.irfft(R2 * np.fft.rfft(box, nfft), nfft)[m // 2:m // 2 + n]
    return np.sqrt(np.maximum(out, 1e-12))


def _place(atom, tau_i, phase, n):
    w = atom.wave if atom.quad is None else np.cos(phase) * atom.wave - np.sin(phase) * atom.quad
    m = len(w)
    v = np.zeros(n)
    i0 = tau_i - m // 2
    a0, a1 = max(i0, 0), min(i0 + m, n)
    v[a0:a1] = w[a0 - i0:a1 - i0]
    return v


DEFAULT_BUDGET = {"B": 8, "T": 6, "S": 20}
DEFAULT_RHO = {"B": 0.45, "T": 0.45, "S": 0.40}


def infer(x, atoms, budget=None, rho_min=None, amp_cap=5.0, explain=True, nms=True,
          period_cap=4, period_cv=0.12, max_iter=60):
    """Greedy sparse inference with explaining-away (LS refit on the active set), NMS within
    +-dur/2 for the same class, normalised-correlation selection, amplitude cap <= amp_cap x
    epoch RMS, per-atom per-epoch count cap for quasi-periodic onsets (ECG guard).
    Returns list of events (cls, k, tau_s, amp, dur, phase, rho) and the event reconstruction."""
    budget = budget or DEFAULT_BUDGET
    rho_min = rho_min or DEFAULT_RHO
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    r = x.astype(float).copy()
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    cls_of = np.array([a.cls for a in atoms])
    used = {c: 0 for c in budget}
    active = []  # (k, tau_i, phase)
    cols = []
    onsets = {}
    forbid = np.zeros((len(atoms), n), bool)
    for _ in range(max_iter):
        c, ph = _corr_all(r, atoms, nfft)
        e = _local_energy(r, atoms, nfft)
        rho = np.abs(c) / e
        rho[forbid] = -1
        for cc in budget:
            if used[cc] >= budget[cc]:
                rho[cls_of == cc] = -1
        rho[:, :FS // 4] = -1
        rho[:, n - FS // 4:] = -1
        k, ti = np.unravel_index(np.argmax(rho), rho.shape)
        a = atoms[k]
        if rho[k, ti] < rho_min[a.cls]:
            break
        if period_cap and a.cls == "T":
            o = onsets.get(k, [])
            if len(o) >= period_cap:
                forbid[k] = True
                continue
            if len(o) >= 2:
                ioi = np.diff(sorted(o + [ti]))
                if ioi.std() / ioi.mean() < period_cv and len(o) + 1 >= 3:
                    forbid[k] = True
                    continue
        onsets.setdefault(k, []).append(ti)
        active.append((k, ti, ph[k, ti]))
        cols.append(_place(a, ti, ph[k, ti], n))
        used[a.cls] += 1
        if nms:
            half = int(a.dur * FS / 2)
            same = cls_of == a.cls
            forbid[same, max(ti - half, 0):ti + half] = True
        A = np.stack(cols, 1)
        if explain:
            amps, *_ = np.linalg.lstsq(A, x, rcond=None)
        else:
            amps = np.array([cols[j] @ x for j in range(len(cols))])
        cap = np.array([amp_cap * rms / np.abs(atoms[kk].wave).max() for kk, _, _ in active])
        amps = np.clip(amps, -cap, cap)
        r = x - A @ amps
    events = []
    for j, (k, ti, phs) in enumerate(active):
        a = atoms[k]
        events.append(dict(cls=a.cls, k=int(k), name=a.name, tau=ti / FS, amp=float(amps[j]),
                           dur=a.dur, f=a.f, phase=float(phs)))
    recon = x - r
    return events, recon


def topk_per_token(x, atoms, k_per_tok=1, tok=2.0, budget=None):
    """Baseline of the original proposal: top-k per 2-s token on the matched filter, no refit, no NMS."""
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    c, ph = _corr_all(x, atoms, nfft)
    cls_of = np.array([a.cls for a in atoms])
    events, recon = [], np.zeros(n)
    T = int(tok * FS)
    for cc in set(cls_of):
        sub = np.where(cls_of == cc)[0]
        for t0 in range(0, n, T):
            seg = np.abs(c[sub, t0:t0 + T])
            for _ in range(k_per_tok):
                k, ti = np.unravel_index(np.argmax(seg), seg.shape)
                if seg[k, ti] <= 0:
                    break
                kk = sub[k]
                amp = c[kk, t0 + ti]
                v = _place(atoms[kk], t0 + ti, ph[kk, t0 + ti], n)
                recon += amp * v
                events.append(dict(cls=cc, k=int(kk), name=atoms[kk].name, tau=(t0 + ti) / FS,
                                   amp=float(amp), dur=atoms[kk].dur, f=atoms[kk].f))
                seg[k, max(ti - 10, 0):ti + 10] = 0
    return events, recon


def event_mask(events, n, pad=0.5):
    m = np.zeros(n, bool)
    for e in events:
        i0 = int((e["tau"] - e["dur"] / 2 - pad) * FS)
        i1 = int((e["tau"] + e["dur"] / 2 + pad) * FS)
        m[max(i0, 0):min(i1, n)] = True
    return m


# ------------------------------------------------------------------ background estimators
def band_power_windows(x, win=4.0, hop=1.0, bands=BG_BANDS, mask=None):
    """Per-window Hann periodogram band power (mean over bins). ``mask`` True = excluded sample;
    masked windows are periodogram-of-(x*keep) rescaled by the kept fraction; windows with more
    than 60 % masked return nan."""
    W, H = int(win * FS), int(hop * FS)
    n = len(x)
    starts = np.arange(0, n - W + 1, H)
    f = np.fft.rfftfreq(W, 1 / FS)
    hann = np.hanning(W)
    out = np.full((len(starts), len(bands)), np.nan)
    for i, s in enumerate(starts):
        seg = x[s:s + W]
        keep = np.ones(W)
        if mask is not None:
            keep = (~mask[s:s + W]).astype(float)
            if keep.mean() < 0.4:
                continue
        spec = np.abs(np.fft.rfft(seg * keep * hann)) ** 2 / (keep.mean() * (hann ** 2).sum())
        for b, (lo, hi) in enumerate(bands):
            sel = (f >= lo) & (f < hi)
            out[i, b] = spec[sel].mean()
    return out, starts / FS


def q20_gamma_factor(M, q=0.2):
    """Closed form: band power averaged over M periodogram bins of a Gaussian process with level P is
    Gamma(M, P/M); the q-quantile is P * gammaincinv(M, q) / M. Return the correction 1/that."""
    return M / gammaincinv(M, q)


def n_bins(band, win=4.0):
    f = np.fft.rfftfreq(int(win * FS), 1 / FS)
    return int(((f >= band[0]) & (f < band[1])).sum())


# ------------------------------------------------------------------ real data
def subjects(folder):
    d = os.path.join(CACHE, folder)
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, "labels", "*.npy")))


def load_night(folder, sid, chan="EEG"):
    d = os.path.join(CACHE, folder)
    f = glob.glob(os.path.join(d, "signals", sid, f"{chan}_clean_*", "*", "signal.npy"))[0]
    x = np.load(f, mmap_mode="r")
    y = np.load(os.path.join(d, "labels", f"{sid}.npy"))
    return x, y


def night_gain(x, win=4.0, span_min=20, q=0.2):
    """R7: 20th percentile of the log-RMS of 4-s windows inside each sliding 20-min span; one scalar
    per night = median over spans (defined on 30-min extracts as well: a single span)."""
    xe = np.asarray(x, dtype=np.float32)
    W = int(win * FS)
    nw = xe.shape[1] // W
    seg = xe[:, :nw * W].reshape(len(xe), nw, W)
    lr = 0.5 * np.log(np.mean(seg ** 2, -1) + 1e-9)  # (n_epochs, nw)
    per_epoch = lr.reshape(len(xe), -1)
    span = max(1, span_min * 2)  # epochs per 20 min
    vals = []
    for s in range(0, max(len(xe) - span + 1, 1), span // 2):
        vals.append(np.quantile(per_epoch[s:s + span], q))
    return float(np.exp(np.median(vals)))
