"""P1 "classico prima": shared library for the quantitative tests.

Everything here is closed-form or greedy: no learned network. The GPU is used only to batch the
convolutions of the matching pursuit.

Contents
- gpu_setup(): CUDA_VISIBLE_DEVICES=0, memory fraction 0.3 (shared T4).
- synthetic signals: pink/AR background, spindle, K-complex, slow wave, alpha run, ECG-like train.
- dictionaries: parametric Gabor bank per band (bursts), low-pass zero-mean transients, one-cycle slow waves,
  with the structural checks of R3 (cycles >= 3, spectral concentration, energy below 5 Hz).
- omp_conv(): batched convolutional OMP with explaining-away (LS refit on the support), non-maximum
  suppression, per-class budget, selection by normalised correlation, amplitude cap.
- topk_conv(): the matched-filter top-k baseline (no explaining-away).
- background estimators: median / low quantile / masked low quantile on the residual (R1).
- gain g: 20th percentile of the log-RMS of 3 s windows (R7).
- MASS mini-cache loader (aggregate use only).
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

FS = 100
EPOCH = 3000
CACHE = "/mnt/data/guido/work/cache/samples/v1"
# burst bands (R3/R6): non-overlapping, the alpha band is the only representation of alpha
# test_f showed that a 1-4 Hz burst band lies entirely inside the transient region (< 5 Hz) and overlaps the
# slow-wave atoms (coherence 0.92): below 4 Hz only transients and slow waves exist (R3).
BURST_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 11.0), "sigma": (11.0, 16.0), "beta": (16.0, 30.0)}
BURST_BANDS_WITH_DELTA = {"delta_b": (1.0, 4.0), **BURST_BANDS}
POW_BANDS = ((0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 30.0))
# budgets per epoch and per group (R5, R9): bursts per band, transients, slow waves (test_d: N3 needs >= 15-20)
DESIGN_BUDGET = {"sigma": 4, "alpha": 4, "theta": 3, "beta": 2, "transient": 4, "slow_wave": 20}
DESIGN = dict(budget=DESIGN_BUDGET, min_score=0.3, whiten=True)


def gpu_setup(frac: float = 0.3) -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        return torch.device("cuda:0")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- synthetic
def pink_noise(n: int, T: int, rng: np.random.Generator, alpha: float = 1.0, fs: int = FS) -> np.ndarray:
    """1/f^alpha background, unit RMS, no power below 0.3 Hz (as the cleaner's high-pass)."""
    f = np.fft.rfftfreq(T, 1 / fs)
    spec = (rng.standard_normal((n, f.size)) + 1j * rng.standard_normal((n, f.size)))
    shape = np.zeros_like(f)
    shape[f > 0] = f[f > 0] ** (-alpha / 2)
    shape[f < 0.3] = 0
    x = np.fft.irfft(spec * shape, n=T, axis=-1)
    return x / (x.std(-1, keepdims=True) + 1e-9)


def ar_background(n: int, T: int, rng: np.random.Generator, f0: float = 2.0, r: float = 0.95, fs: int = FS) -> np.ndarray:
    """AR(2) resonant at f0 Hz (a delta-rich N3-like background), unit RMS."""
    a1, a2 = 2 * r * np.cos(2 * np.pi * f0 / fs), -r * r
    e = rng.standard_normal((n, T + 500))
    x = np.zeros_like(e)
    for t in range(2, e.shape[1]):
        x[:, t] = a1 * x[:, t - 1] + a2 * x[:, t - 2] + e[:, t]
    x = x[:, 500:]
    return x / (x.std(-1, keepdims=True) + 1e-9)


def hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)


def spindle(f: float, d: float, phase: float = 0.0, fs: int = FS) -> np.ndarray:
    n = int(round(d * fs))
    t = np.arange(n) / fs
    return hann(n) * np.cos(2 * np.pi * f * t + phase)


def kcomplex(fs: int = FS, d: float = 0.9) -> np.ndarray:
    """Biphasic: sharp negative wave then slower positive wave, zero mean, <= 1 s."""
    n = int(round(d * fs))
    t = np.arange(n) / fs
    k = -np.exp(-((t - 0.22) / 0.08) ** 2) + 0.55 * np.exp(-((t - 0.52) / 0.17) ** 2)
    k = k * hann(n) ** 0.3
    return k - k.mean()


def slow_wave(d: float, fs: int = FS) -> np.ndarray:
    """One cycle (negative half then positive half) of duration d in 1-2.5 s, Tukey-windowed, zero mean."""
    n = int(round(d * fs))
    t = np.arange(n) / n
    w = -np.sin(2 * np.pi * t)
    tk = np.ones(n)
    m = int(0.15 * n)
    tk[:m] = hann(2 * m)[:m]
    tk[-m:] = hann(2 * m)[m:]
    w = w * tk
    return w - w.mean()


def ecg_train(T: int, hr_bpm: float, rng: np.random.Generator, fs: int = FS) -> np.ndarray:
    """ECG-like periodic sharp transients (R-peaks, 60-80 ms), as leaked on a mastoid reference."""
    x = np.zeros(T)
    period = 60.0 / hr_bpm * fs
    t0 = rng.uniform(8, period)
    spike = np.exp(-((np.arange(-6, 7)) / 2.0) ** 2)
    spike -= spike.mean()
    while t0 < T - 7:
        i = int(t0)
        x[i - 6:i + 7] += spike
        t0 += period * (1 + 0.03 * rng.standard_normal())
    return x


def plant(x: np.ndarray, atom: np.ndarray, onset: int, amp: float) -> None:
    n = min(len(atom), x.shape[-1] - onset)
    x[onset:onset + n] += amp * atom[:n]


# ----------------------------------------------------------------------------- dictionaries
@dataclass
class Dictionary:
    atoms: np.ndarray                      # (K, L) unit norm, zero-padded to L
    cls: np.ndarray                        # class id per atom (0 burst, 1 transient, 2 slow wave)
    pair: np.ndarray                       # index of the quadrature partner (or -1)
    dur: np.ndarray                        # effective duration in samples (support)
    meta: List[dict] = field(default_factory=list)
    class_names: Tuple[str, ...] = ("burst", "transient", "slow_wave")

    @property
    def group(self) -> np.ndarray:
        """Budget group per atom: the band for bursts, the class name otherwise (R9: budgets per class)."""
        return np.array([m["band"] if m["class"] == "burst" else m["class"] for m in self.meta])

    @property
    def K(self):
        return self.atoms.shape[0]

    @property
    def L(self):
        return self.atoms.shape[1]


def _pad(a: np.ndarray, L: int) -> np.ndarray:
    out = np.zeros(L)
    out[:len(a)] = a
    return out


def spectral_concentration(a: np.ndarray, band: Tuple[float, float], fs: int = FS, nfft: int = 4096) -> float:
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(a, nfft)) ** 2
    m = (f >= band[0]) & (f < band[1])
    return float(p[m].sum() / (p.sum() + 1e-12))


def gabor_bank(bands: Dict[str, Tuple[float, float]] = BURST_BANDS, f_step: float = 1.0,
               durs: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0), min_cycles: float = 3.0,
               min_conc: float = 0.8, fs: int = FS, L: Optional[int] = None) -> Dictionary:
    """Parametric burst family (R3, R6): Hann-windowed cos/sin pairs on a frequency grid inside each band,
    kept only if d*f >= min_cycles and the spectral concentration in the band is >= min_conc."""
    atoms, cls, pair, dur, meta = [], [], [], [], []
    for name, (lo, hi) in bands.items():
        for f in np.arange(lo + f_step / 2, hi, f_step):
            for d in durs:
                if d * f < min_cycles:
                    continue
                c = spindle(f, d, 0.0, fs)
                conc = spectral_concentration(c, (lo, hi), fs)
                if conc < min_conc:
                    continue
                s = spindle(f, d, -np.pi / 2, fs)
                for a, ph in ((c, 0.0), (s, 1.0)):
                    atoms.append(a / np.linalg.norm(a))
                    cls.append(0)
                    dur.append(len(a))
                    meta.append({"class": "burst", "band": name, "f": float(f), "d": float(d), "phase": ph, "conc": conc})
                k = len(atoms)
                pair += [k - 1, k - 2]
    L = L or max(len(a) for a in atoms)
    return Dictionary(np.stack([_pad(a, L) for a in atoms]), np.array(cls), np.array(pair), np.array(dur), meta)


def transient_bank(fs: int = FS, L: Optional[int] = None, scales: Sequence[float] = (0.5, 0.7, 0.9),
                   lowpass_hz: float = 4.0, min_low: float = 0.8) -> Dictionary:
    """Transients (R3): zero mean, <= 1 s, >= 80 % of the energy below 4 Hz (test_f: 5 Hz would admit theta bursts). Family: K-complex template at three
    durations, both polarities of a mono-phasic Gaussian bump (vertex-like) and a first-derivative Gaussian."""
    atoms, meta = [], []
    for d in scales:
        n = int(d * fs)
        t = np.arange(n) / fs
        kc = kcomplex(fs, d)
        g = np.exp(-((t - d / 2) / (d / 6)) ** 2) * hann(n)
        dg = -(t - d / 2) * np.exp(-((t - d / 2) / (d / 6)) ** 2) * hann(n)
        for a, nm in ((kc, "kc"), (-g, "vertex_neg"), (dg, "dgauss")):
            a = a - a.mean()
            a = a / np.linalg.norm(a)
            low = spectral_concentration(a, (0.0, lowpass_hz), fs)
            if low < min_low:
                continue
            atoms.append(a)
            meta.append({"class": "transient", "shape": nm, "d": d, "low4": low})
    L = L or max(len(a) for a in atoms)
    K = len(atoms)
    return Dictionary(np.stack([_pad(a, L) for a in atoms]), np.ones(K, int), -np.ones(K, int),
                      np.array([int(m["d"] * fs) for m in meta]), meta)


def slowwave_bank(fs: int = FS, L: Optional[int] = None, durs: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5)) -> Dictionary:
    """Slow waves (R2): explicit class, one cycle of 1-2.5 s, free amplitude above threshold at read-out."""
    atoms, meta = [], []
    for d in durs:
        a = slow_wave(d, fs)
        atoms.append(a / np.linalg.norm(a))
        meta.append({"class": "slow_wave", "d": d, "f": 1 / d})
    L = L or max(len(a) for a in atoms)
    K = len(atoms)
    return Dictionary(np.stack([_pad(a, L) for a in atoms]), 2 * np.ones(K, int), -np.ones(K, int),
                      np.array([int(d * fs) for d in durs]), meta)


def concat(*ds: Dictionary) -> Dictionary:
    L = max(d.L for d in ds)
    atoms, cls, pair, dur, meta, off = [], [], [], [], [], 0
    for d in ds:
        a = np.zeros((d.K, L))
        a[:, :d.L] = d.atoms
        atoms.append(a)
        cls.append(d.cls)
        pair.append(np.where(d.pair >= 0, d.pair + off, -1))
        dur.append(d.dur)
        meta += d.meta
        off += d.K
    return Dictionary(np.concatenate(atoms), np.concatenate(cls), np.concatenate(pair), np.concatenate(dur), meta)


def full_dictionary(**kw) -> Dictionary:
    return concat(gabor_bank(**kw), transient_bank(), slowwave_bank())


def coherence(D: Dictionary, same_class_only: bool = True) -> np.ndarray:
    """Max |normalised cross-correlation| over all lags between atom pairs (Gram coherence, R6)."""
    A = torch.tensor(D.atoms, dtype=torch.float32)
    K, L = A.shape
    cc = F.conv1d(A[:, None], A[:, None], padding=L - 1).abs().amax(-1).numpy()  # (K, K)
    np.fill_diagonal(cc, 0)
    for i in range(K):
        if D.pair[i] >= 0:
            cc[i, D.pair[i]] = 0
    if same_class_only:
        cc = cc * (D.cls[:, None] == D.cls[None, :])
    return cc


# ----------------------------------------------------------------------------- sparse inference
def whitening_filters(x: torch.Tensor, L: int, q: float = 0.2, fs: int = FS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-epoch whitening weights 1/sqrt(P_bg(f)) on the rfft grid of the epoch (B, T//2+1) and the batch-median
    weights on the rfft grid of the atoms (L//2+1). P_bg(f) = q-quantile over 1 s Hann segments (hop 0.5 s) of
    the periodogram, smoothed over +-1 Hz, floored at 1e-3 of its maximum."""
    B, T = x.shape
    n, h = fs, fs // 2
    w = torch.hann_window(n, device=x.device)
    segs = x.unfold(-1, n, h) * w  # (B, S, n)
    p = torch.fft.rfft(segs, dim=-1).abs() ** 2  # (B, S, n//2+1), 1 Hz grid
    pq = p.quantile(q, dim=1)  # (B, F1)
    pq = F.avg_pool1d(pq[:, None], 3, stride=1, padding=1)[:, 0]
    pq = pq.clamp_min(1e-3 * pq.amax(-1, keepdim=True))
    fT = torch.fft.rfftfreq(T, 1 / fs, device=x.device)
    fL = torch.fft.rfftfreq(L, 1 / fs, device=x.device)
    f1 = torch.fft.rfftfreq(n, 1 / fs, device=x.device)
    def interp(fq):
        idx = torch.bucketize(fq, f1).clamp(1, len(f1) - 1)
        lo, hi = f1[idx - 1], f1[idx]
        t = ((fq - lo) / (hi - lo + 1e-9)).clamp(0, 1)
        return (1 - t)[None] * pq[:, idx - 1] + t[None] * pq[:, idx]
    Wsig = 1 / interp(fT).sqrt()
    Wsig = Wsig / Wsig.mean(-1, keepdim=True)
    Watom = (1 / interp(fL).sqrt()).median(0).values
    return Wsig, Watom


@dataclass
class Events:
    """Flat event list: one row per selected atom placement."""
    b: np.ndarray        # epoch index
    k: np.ndarray        # atom index (for a pair: the cos atom)
    tau: np.ndarray      # onset sample
    amp: np.ndarray      # amplitude in signal units (envelope amplitude for a pair; LS refit)
    cls: np.ndarray
    score: np.ndarray    # normalised correlation at selection


def _conv_scores(r: torch.Tensor, A: torch.Tensor, boxes: torch.Tensor, dur_idx: torch.Tensor,
                 pair: torch.Tensor, L: int):
    """r (B,T) -> corr (B,K,T') with the 'full' convolution (T' = T+L-1), local energy over each atom's own
    support (B,n_dur,T') and the normalised-correlation score (B,K,T'); pairs combine cos and sin."""
    c = F.conv1d(r[:, None], A[:, None], padding=L - 1)  # (B,K,T')
    e = F.conv1d((r * r)[:, None], boxes[:, None], padding=L - 1).clamp_min(1e-6)  # (B,n_dur,T')
    K = A.shape[0]
    idx = torch.where(pair >= 0, pair, torch.arange(K, device=A.device))
    w = (pair >= 0).float()[None, :, None]
    score = c * c
    score = score + score[:, idx] * w
    score = score.div_(e[:, dur_idx]).sqrt_()
    return c, e, score


def omp_conv(x: torch.Tensor, D: Dictionary, budget: Dict[int, int], min_score: float = 0.25,
             amp_cap: float = 5.0, nms_frac: float = 0.5, max_steps: Optional[int] = None,
             topk_only: bool = False, cls_mask: Optional[Sequence[int]] = None, device=None,
             chunk: int = 48, whiten: bool = False) -> Tuple[Events, torch.Tensor]:
    """Batched convolutional OMP (R5). Epoch batches larger than `chunk` are processed in chunks (memory).
    budget keys: class id (int, applies to every group of the class) or group name (str: a burst band,
    "transient", "slow_wave"). whiten=True: selection on the residual pre-whitened by the epoch's robust
    background PSD (matched filter in coloured noise); amplitudes are still fitted in the original domain.

    x: (B, T) float tensor. budget: {class_id: max events per epoch}. Selection by normalised correlation
    (quality of fit, not energy); amplitudes by least squares on the support after each selection
    (explaining-away); non-maximum suppression: same-class atoms within +-nms_frac*dur of a chosen onset are
    excluded; amplitudes capped at amp_cap x RMS of the epoch. topk_only=True reproduces the matched-filter
    baseline (no residual update, amplitude = correlation)."""
    device = device or x.device
    B, T = x.shape
    if B > chunk:
        evs, rs = [], []
        for i in range(0, B, chunk):
            e_, r_ = omp_conv(x[i:i + chunk], D, budget, min_score, amp_cap, nms_frac, max_steps, topk_only, cls_mask, device, chunk, whiten)
            e_.b = e_.b + i
            evs.append(e_); rs.append(r_)
        ev = Events(*[np.concatenate([getattr(e_, f) for e_ in evs]) for f in ("b", "k", "tau", "amp", "cls", "score")])
        return ev, torch.cat(rs)
    A = torch.tensor(D.atoms, dtype=torch.float32, device=device)
    K, L = A.shape
    pair = torch.tensor(D.pair, device=device)
    cls = torch.tensor(D.cls, device=device)
    dur = torch.tensor(D.dur, device=device)
    udur = sorted(set(int(d) for d in D.dur))
    boxes = torch.zeros(len(udur), L, device=device)
    for i, d in enumerate(udur):
        boxes[i, :d] = 1.0
    dur_idx = torch.tensor([udur.index(int(d)) for d in D.dur], device=device)
    rms = x.pow(2).mean(-1).sqrt()
    n_cls = int(cls.max().item()) + 1
    gnames = sorted(set(D.group.tolist()))
    gid = torch.tensor([gnames.index(g) for g in D.group], device=device)
    n_grp = len(gnames)
    bud = torch.zeros(n_grp, dtype=torch.long, device=device)
    for key, v in budget.items():
        if isinstance(key, str):
            if key in gnames:
                bud[gnames.index(key)] = v
        else:
            for gi, g in enumerate(gnames):
                if (D.cls[D.group == g] == key).all():
                    bud[gi] = v
    if cls_mask is not None:
        for gi, g in enumerate(gnames):
            if int(D.cls[D.group == g][0]) not in cls_mask:
                bud[gi] = 0
    # whitening (selection only): robust background PSD per epoch = 20th percentile over 1 s Hann segments
    if whiten:
        Wsig, Watom = whitening_filters(x, L)
        Aw = torch.fft.irfft(torch.fft.rfft(A, dim=-1) * Watom[None], n=L, dim=-1)
        Aw = Aw / Aw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    else:
        Wsig, Aw = None, A
    def wh(v):
        return v if Wsig is None else torch.fft.irfft(torch.fft.rfft(v, dim=-1) * Wsig, n=T, dim=-1)
    max_steps = max_steps or int(bud.sum().item())
    Tp = T + L - 1
    forbid = torch.zeros(B, K, Tp, dtype=torch.bool, device=device)
    # onsets outside [-L+1, T-1] in padded coordinates are meaningless; keep only tau in [0, T - dur)
    tgrid = torch.arange(Tp, device=device)[None, :] - (L - 1)  # tau in signal coordinates
    forbid |= (tgrid[None] < 0) | (tgrid[None] > (T - dur)[None, :, None])
    count = torch.zeros(B, n_grp, dtype=torch.long, device=device)
    active = torch.ones(B, dtype=torch.bool, device=device)
    r = x.clone()
    sel_k = torch.full((B, max_steps), -1, dtype=torch.long, device=device)
    sel_tau = torch.zeros((B, max_steps), dtype=torch.long, device=device)
    sel_score = torch.zeros((B, max_steps), device=device)
    n_sel = torch.zeros(B, dtype=torch.long, device=device)
    # LS design: columns = shifted atoms (and partners) on the support
    cols = torch.zeros(B, T, 2 * max_steps, device=device)
    col_k = torch.zeros(B, 2 * max_steps, dtype=torch.long, device=device)
    amax = A.abs().amax(-1).clamp_min(1e-6)
    ncol = torch.zeros(B, dtype=torch.long, device=device)
    amps = torch.zeros(B, 2 * max_steps, device=device)
    c, e, score = _conv_scores(wh(r), Aw, boxes, dur_idx, pair, L)
    for step in range(max_steps):
        if not topk_only and step > 0:
            c, e, score = _conv_scores(wh(r), Aw, boxes, dur_idx, pair, L)
        s = score.masked_fill(forbid, -1.0)
        # exhausted classes
        exhausted = count >= bud[None]  # (B, n_grp)
        s = s.masked_fill(exhausted[:, gid][:, :, None], -1.0)
        flat = s.reshape(B, -1)
        best, idx = flat.max(-1)
        k = idx // Tp
        tp = idx % Tp
        tau = tp - (L - 1)
        ok = active & (best >= min_score)
        if not ok.any():
            break
        active = ok
        bi = torch.nonzero(ok)[:, 0]
        kk, tt, tpp = k[bi], tau[bi], tp[bi]
        j = n_sel[bi]
        sel_k[bi, j], sel_tau[bi, j], sel_score[bi, j] = kk, tt, best[bi]
        n_sel[bi] += 1
        count[bi, gid[kk]] += 1
        # NMS (R5), vectorised: same-class candidates whose centre lies within +-nms_frac*max(d, d') of the
        # chosen centre are excluded; the chosen (atom, onset) itself is excluded.
        n = len(bi)
        d0 = dur[kk]
        c0 = tpp + d0 // 2
        half = (nms_frac * torch.maximum(d0[:, None], dur[None, :]).float()).long()  # (n,K)
        lo = (c0[:, None] - half - dur[None, :] // 2).clamp_min(0)
        hi = (c0[:, None] + half - dur[None, :] // 2 + 1).clamp_max(Tp)
        same = cls[None, :] == cls[kk][:, None]
        tg = torch.arange(Tp, device=device)[None, None, :]
        forbid[bi] = forbid[bi] | (same[:, :, None] & (tg >= lo[:, :, None]) & (tg < hi[:, :, None]))
        forbid[bi, kk, tpp] = True
        # add the columns (atom and its quadrature partner) shifted to tau, vectorised
        col = ncol[bi]
        jj = torch.arange(L, device=device)[None, :]
        pos = tt[:, None] + jj
        valid = pos < T
        rows = bi[:, None].expand(-1, L)
        cols[rows[valid], pos[valid], col[:, None].expand(-1, L)[valid]] = A[kk][valid]
        col_k[bi, col] = kk
        has = pair[kk] >= 0
        if has.any():
            v2 = valid & has[:, None]
            cols[rows[v2], pos[v2], (col + 1)[:, None].expand(-1, L)[v2]] = A[pair[kk]][v2]
            col_k[bi[has], col[has] + 1] = pair[kk[has]]
        ncol[bi] += 1 + has.long()
        if topk_only:
            # baseline amplitude = projection of x on the (unit-norm, original-domain) column, no residual update
            amps[bi, col] = (cols[bi, :, col] * x[bi]).sum(-1)
            if has.any():
                amps[bi[has], col[has] + 1] = (cols[bi[has], :, col[has] + 1] * x[bi[has]]).sum(-1)
            continue
        # LS refit on the support (explaining-away), batched over the active epochs
        m = int(ncol[bi].max().item())
        Xb = cols[bi, :, :m]
        G = torch.bmm(Xb.transpose(1, 2), Xb) + 1e-4 * torch.eye(m, device=device)[None]
        sol = torch.linalg.solve(G, torch.bmm(Xb.transpose(1, 2), x[bi][:, :, None]))[:, :, 0]
        # cap: peak of the decoded waveform (coef x max|atom|) <= amp_cap x RMS of the epoch
        cap = amp_cap * rms[bi][:, None] / amax[col_k[bi, :m]]
        sol = torch.maximum(torch.minimum(sol, cap), -cap)
        amps[bi, :m] = sol
        r[bi] = x[bi] - torch.bmm(Xb, sol[:, :, None])[:, :, 0]
    # gather events (vectorised): column offset of each selection = cumulative width (1, or 2 for a pair)
    valid = sel_k >= 0
    kk_all = sel_k.clamp_min(0)
    hp = (pair[kk_all] >= 0) & valid
    width = valid.long() + hp.long()
    start = torch.cumsum(width, 1) - width
    a1 = amps.gather(1, start)
    a2 = amps.gather(1, (start + 1).clamp_max(amps.shape[1] - 1))
    amp_all = torch.where(hp, torch.sqrt(a1 * a1 + a2 * a2), a1)
    bb = torch.arange(B, device=device)[:, None].expand_as(sel_k)
    v = valid
    ev_b, ev_k, ev_tau = bb[v].cpu().numpy(), kk_all[v].cpu().numpy(), sel_tau[v].cpu().numpy()
    ev_amp, ev_cls, ev_score = amp_all[v].cpu().numpy(), cls[kk_all[v]].cpu().numpy(), sel_score[v].cpu().numpy()
    if topk_only:
        # the baseline's reconstruction uses the correlation amplitudes
        r = x - torch.bmm(cols, amps[:, :, None])[:, :, 0]
    ev = Events(np.array(ev_b), np.array(ev_k), np.array(ev_tau), np.array(ev_amp), np.array(ev_cls), np.array(ev_score))
    return ev, r


def reconstruct(ev: Events, D: Dictionary, B: int, T: int, amps_by_pair: Optional[np.ndarray] = None,
                cls_only: Optional[Sequence[int]] = None) -> np.ndarray:
    """Sum of selected atoms per class (envelope amplitude x cos atom; for the energy accounting the
    partner phase is irrelevant because cos/sin pairs have equal norm)."""
    y = np.zeros((B, T))
    for i in range(len(ev.b)):
        if cls_only is not None and ev.cls[i] not in cls_only:
            continue
        a = D.atoms[ev.k[i], :D.dur[ev.k[i]]]
        plant(y[ev.b[i]], a, ev.tau[i], ev.amp[i])
    return y



def refine_bursts(x: torch.Tensor, r: torch.Tensor, ev: Events, D: Dictionary, df: float = 0.5, step: float = 0.1,
                  fs: int = FS) -> np.ndarray:
    """R6 (f continuous): for each burst event re-fit the frequency on a fine grid f0 +- df (step) at the same
    onset and duration, on the residual plus the event's own contribution. Returns the refined f per event
    (nan for non-burst events). Amplitudes are then re-estimated by projection on the refined pair."""
    out = np.full(len(ev.b), np.nan)
    A = torch.tensor(D.atoms, dtype=torch.float32, device=x.device)
    offs = np.arange(-df, df + 1e-9, step)
    for i in range(len(ev.b)):
        if ev.cls[i] != 0:
            continue
        m = D.meta[ev.k[i]]
        d = int(D.dur[ev.k[i]])
        b_, t_ = int(ev.b[i]), int(ev.tau[i])
        a_cos = A[ev.k[i], :d]
        a_sin = A[D.pair[ev.k[i]], :d]
        seg = r[b_, t_:t_ + d]
        # add back the event (its LS amplitude is a projection on the pair => reconstruct via projection)
        seg = seg + a_cos * (x[b_, t_:t_ + d] - r[b_, t_:t_ + d]) @ a_cos + a_sin * (x[b_, t_:t_ + d] - r[b_, t_:t_ + d]) @ a_sin
        best, bf = -1.0, m["f"]
        for o in offs:
            f = m["f"] + o
            c = torch.tensor(spindle(f, d / fs, 0.0, fs), dtype=torch.float32, device=x.device)
            s_ = torch.tensor(spindle(f, d / fs, -np.pi / 2, fs), dtype=torch.float32, device=x.device)
            c, s_ = c / c.norm(), s_ / s_.norm()
            sc = float(((seg @ c) ** 2 + (seg @ s_) ** 2).sqrt() / (seg.norm() + 1e-6))
            if sc > best:
                best, bf = sc, f
        out[i] = bf
    return out


def null_thresholds(D: Dictionary, bg: torch.Tensor, budget: Dict[int, int], q: float = 0.95, **kw) -> Dict[int, float]:
    """Per-class selection threshold fixed from a null (R13, prior in closed form): the q-quantile over null
    epochs of the best normalised-correlation score of that class on event-free background."""
    ev, _ = omp_conv(bg, D, budget, min_score=0.0, **kw)
    lab = np.array([D.group[k] if isinstance(next(iter(budget)), str) else D.cls[k] for k in ev.k])
    thr = {}
    for c in budget:
        best = np.zeros(bg.shape[0])
        for i in np.nonzero(lab == c)[0]:
            best[ev.b[i]] = max(best[ev.b[i]], ev.score[i])
        thr[c] = float(np.quantile(best, q))
    return thr


# ----------------------------------------------------------------------------- background and gain
def seg_band_logpow(x: np.ndarray, bands=POW_BANDS, seg: float = 1.0, hop: float = 0.5, fs: int = FS) -> np.ndarray:
    """(B, T) -> (B, n_seg, n_bands) log10 band power of Hann segments of `seg` s with hop `hop` s."""
    n, h = int(seg * fs), int(hop * fs)
    B, T = x.shape
    starts = np.arange(0, T - n + 1, h)
    w = hann(n)
    segs = np.stack([x[:, s:s + n] * w for s in starts], 1)  # (B, S, n)
    p = np.abs(np.fft.rfft(segs, axis=-1)) ** 2 / (w ** 2).sum()
    f = np.fft.rfftfreq(n, 1 / fs)
    out = np.stack([p[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands], -1)
    return np.log10(out + 1e-10), starts


def background_estimate(x: np.ndarray, mask: Optional[np.ndarray] = None, win: float = 4.0, hop: float = 1.0,
                        q: float = 0.2, how: str = "quantile", seg: float = 1.0, seg_hop: float = 0.5,
                        bands=POW_BANDS, fs: int = FS, min_clean: float = 0.5):
    """Background log band power per window (B, n_win, n_bands).

    how='median': median over the 1 s segments of the window (the proposal).
    how='quantile': q-quantile over the segments (R1).
    mask (B, T) bool, True on event supports: segments with >= min_clean of masked samples are dropped
    before the quantile (R1, masked); if a window has no clean segment the quantile over all is used."""
    lp, starts = seg_band_logpow(x, bands, seg, seg_hop, fs)
    B, S, nb = lp.shape
    n = int(seg * fs)
    if mask is not None:
        frac = np.stack([mask[:, s:s + n].mean(-1) for s in starts], 1)  # (B, S)
        clean = frac < min_clean
    else:
        clean = np.ones((B, S), bool)
    centers = starts + n / 2
    wstarts = np.arange(0, x.shape[1] - int(win * fs) + 1, int(hop * fs))
    out = np.zeros((B, len(wstarts), nb))
    for wi, ws in enumerate(wstarts):
        inwin = (centers >= ws) & (centers < ws + win * fs)
        for b in range(B):
            sel = inwin & clean[b]
            if sel.sum() == 0:
                sel = inwin
            v = lp[b, sel]
            out[b, wi] = np.median(v, 0) if how == "median" else np.quantile(v, q, axis=0)
    return out, wstarts


def gain_g(x_night: np.ndarray, win: float = 3.0, q: float = 0.2, fs: int = FS) -> float:
    """R7: 20th percentile of the log-RMS of 3 s windows over the record (a 20 min sliding version
    is the same statistic computed per 20 min block; the record here is <= 20 min of epochs)."""
    n = int(win * fs)
    v = x_night.reshape(-1)
    m = len(v) // n
    rms = np.sqrt((v[:m * n].reshape(m, n) ** 2).mean(-1))
    return float(10 ** np.quantile(np.log10(rms + 1e-9), q))


# ----------------------------------------------------------------------------- data
def list_subjects(ds: str) -> List[str]:
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{CACHE}/{ds}/labels/*.npy"))


def load_subject(ds: str, sid: str, chan: str = "EEG") -> Tuple[np.ndarray, np.ndarray]:
    p = glob.glob(f"{CACHE}/{ds}/signals/{sid}/{chan}_clean_*/*/signal.npy")[0]
    x = np.load(p, mmap_mode="r")
    y = np.load(f"{CACHE}/{ds}/labels/{sid}.npy")
    return x, y


def sample_epochs(ds: str, sids: Sequence[str], stage: Optional[int], n_per: int, rng: np.random.Generator,
                  chan: str = "EEG") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X (N,T) float32, subject index (N,), stage (N,)). Aggregate use only."""
    X, S, Y = [], [], []
    for si, sid in enumerate(sids):
        x, y = load_subject(ds, sid, chan)
        idx = np.nonzero(y == stage)[0] if stage is not None else np.nonzero(y >= 0)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(n_per, len(idx)), replace=False)
        X.append(np.asarray(x[np.sort(pick)], dtype=np.float32))
        S += [si] * len(pick)
        Y += list(y[np.sort(pick)])
    return np.concatenate(X), np.array(S), np.array(Y)


def ap_score(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, score))


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self) -> float:
        now = time.time()
        d = now - self.t
        self.t = now
        return d


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
