"""P2 "structured neural" design: shared library for the quantitative tests.

Synthetic PSG-like data with known ground truth, analytic (cos/sin) Gabor dictionaries, a
convolutional sparse coder (FISTA with group soft-threshold on the analytic pairs, non-maximum
suppression on the onset, per-class budget, least-squares refit of the amplitudes on the support,
amplitude cap) and the background estimators (median / low quantile / masked low quantile).
Units: the mini-cache is RobustScale (median/MAD) so a background RMS of ~1 is the natural scale
and every threshold below is dimensionless (R4/R18 of the critique).
"""
from __future__ import annotations

import glob
import json
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

FS = 100
EPOCH = 30 * FS
CACHE = "/mnt/data/guido/work/cache/samples/v1"
BURST_BANDS = {"theta": (6.0, 8.0), "alpha": (8.0, 12.0), "sigma": (11.0, 16.0), "beta": (16.0, 30.0)}  # lower edge 6 Hz: <= 20 % energy below 5 Hz with >= 3 cycles (R3)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_setup():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.3, 0)


# ------------------------------------------------------------------ synthetic data
def one_over_f(n, rng, alpha=1.0, fs=FS, lo=0.3):
    """Unit-RMS 1/f^alpha noise, high-passed at ``lo`` Hz (the cleaner v3 removes the drift)."""
    f = np.fft.rfftfreq(n, 1 / fs)
    amp = np.where(f > 0, 1.0 / np.maximum(f, lo) ** (alpha / 2), 0.0)
    spec = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-9)


def ar2(n, rng, f0=1.5, r=0.97, fs=FS):
    """Unit-RMS AR(2) with a resonance at ``f0`` (delta-rich background for the N3-like case)."""
    th = 2 * math.pi * f0 / fs
    a1, a2 = 2 * r * math.cos(th), -r * r
    e = rng.standard_normal(n + 500)
    x = np.zeros(n + 500)
    for t in range(2, n + 500):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    x = x[500:]
    return x / (x.std() + 1e-9)


def gabor(f, d, fs=FS, phase=0.0):
    """Unit-norm cos-phase Gabor of frequency ``f`` and duration ``d`` (Gaussian envelope, sigma = d/4)."""
    n = int(round(d * fs))
    t = (np.arange(n) - (n - 1) / 2) / fs
    env = np.exp(-0.5 * (t / (d / 4)) ** 2)
    w = env * np.cos(2 * math.pi * f * t + phase)
    return w / (np.linalg.norm(w) + 1e-12)


def kc_shape(fs=FS, d=0.9, ratio=0.5):
    """Zero-mean biphasic K-complex-like transient (sharp negative, slower positive), ~d s."""
    n = int(round(d * fs))
    t = np.arange(n) / fs
    neg = -np.exp(-0.5 * ((t - 0.25 * d) / (0.09 * d)) ** 2)
    pos = ratio * np.exp(-0.5 * ((t - 0.62 * d) / (0.16 * d)) ** 2)
    w = neg + pos
    w -= w.mean()
    w *= np.hanning(n)
    return w / (np.linalg.norm(w) + 1e-12)


def slow_wave_shape(f=0.8, fs=FS, asym=0.6):
    """Zero-mean one-cycle slow wave (1/f s long), negative half shorter than the positive one."""
    n = int(round(fs / f))
    t = np.linspace(0, 1, n)
    w = np.where(t < asym, -np.sin(math.pi * t / asym), np.sin(math.pi * (t - asym) / (1 - asym)))
    w = w * np.hanning(n) ** 0.3
    w -= w.mean()
    return w / (np.linalg.norm(w) + 1e-12)


def ecg_shape(fs=FS):
    n = int(0.12 * fs)
    t = np.arange(n) - n // 2
    w = np.exp(-0.5 * (t / 1.2) ** 2) - 0.35 * np.exp(-0.5 * ((t - 3) / 2.5) ** 2) - 0.1 * np.exp(-0.5 * ((t + 3) / 2.5) ** 2)
    w -= w.mean()
    return w / (np.linalg.norm(w) + 1e-12)


def place(x, w, onset, amp):
    """Add ``amp * w`` at sample ``onset`` (clipped at the ends)."""
    n = len(w)
    a, b = max(0, onset), min(len(x), onset + n)
    if b > a:
        x[a:b] += amp * w[a - onset:b - onset]
    return x


def synth_epoch(rng, kind="N2", bg="1/f", n_spindles=2, n_kc=1, n_sw=0, alpha_run=False, ecg=False,
                spindle_peak=1.5, kc_peak=4.0, sw_peak=4.0, alpha_peak=1.0, ecg_peak=0.6, sp_freqs=(11, 13, 15)):
    """Returns ``(x, bg, events)``; ``events`` = list of (cls, onset_sample, dur_samples, f, peak)."""
    x_bg = one_over_f(EPOCH, rng) if bg == "1/f" else ar2(EPOCH, rng)
    x = x_bg.copy()
    ev = []
    busy = np.zeros(EPOCH, bool)

    def free_slot(n, tries=50):
        for _ in range(tries):
            o = int(rng.integers(50, EPOCH - n - 50))
            if not busy[o:o + n].any():
                busy[max(0, o - 30):o + n + 30] = True
                return o
        return None

    for _ in range(n_sw):  # slow waves first: they are the dense class
        f = float(rng.uniform(0.6, 1.3))
        w = slow_wave_shape(f)
        o = free_slot(len(w))
        if o is None:
            break
        pk = sw_peak * float(rng.uniform(0.7, 1.4))
        place(x, w, o, pk / np.abs(w).max())
        ev.append(("sw", o, len(w), f, pk))
    for _ in range(n_kc):
        d = float(rng.uniform(0.7, 1.0))
        w = kc_shape(d=d)
        o = free_slot(len(w))
        if o is None:
            break
        pk = kc_peak * float(rng.uniform(0.6, 1.4))
        place(x, w, o, pk / np.abs(w).max())
        ev.append(("kc", o, len(w), 0.0, pk))
    for _ in range(n_spindles):
        f = float(rng.choice(sp_freqs)) + float(rng.uniform(-0.3, 0.3))
        d = float(rng.uniform(0.5, 1.5))
        w = gabor(f, d, phase=float(rng.uniform(0, 2 * math.pi)))
        o = free_slot(len(w))
        if o is None:
            break
        pk = spindle_peak * float(rng.uniform(0.6, 1.4))
        place(x, w, o, pk / np.abs(w).max())
        ev.append(("spindle", o, len(w), f, pk))
    if alpha_run:
        f = float(rng.uniform(8.5, 11.5))
        d = float(rng.uniform(3, 8))
        w = gabor(f, d)
        w = w * np.hanning(len(w)) ** -0.5 if False else w
        o = free_slot(len(w))
        if o is not None:
            pk = alpha_peak * float(rng.uniform(0.7, 1.3))
            place(x, w, o, pk / np.abs(w).max())
            ev.append(("alpha", o, len(w), f, pk))
    if ecg:
        w = ecg_shape()
        rr = float(rng.uniform(0.8, 1.1))
        o = int(rng.integers(0, int(rr * FS)))
        while o < EPOCH - len(w):
            place(x, w, o, ecg_peak / np.abs(w).max())
            ev.append(("ecg", o, len(w), 0.0, ecg_peak))
            o += int(rr * FS * float(rng.uniform(0.97, 1.03)))
    return x.astype(np.float32), x_bg.astype(np.float32), ev


# ------------------------------------------------------------------ dictionaries
def burst_dictionary(bands=BURST_BANDS, f_step=1.0, cycles=(3, 5, 8, 12), d_max=3.0, fs=FS):
    """Analytic Gabor atoms per band: (cos, sin) pairs, ``>= 3`` cycles (R3), f on a 1 Hz grid.
    Returns kernels ``(2K, L)`` zero-padded to a common length and the atom table."""
    atoms, table = [], []
    for name, (lo, hi) in bands.items():
        for f in np.arange(lo, hi + 1e-6, f_step):
            for c in cycles:
                d = c / f
                if d > d_max or d < 0.3:
                    continue
                atoms.append((gabor(f, d, fs, 0.0), gabor(f, d, fs, -math.pi / 2)))
                table.append({"cls": "burst", "band": name, "f": float(f), "d": float(d), "cycles": c})
    L = max(len(a[0]) for a in atoms)
    K = np.zeros((2 * len(atoms), L), np.float32)
    for i, (c, s) in enumerate(atoms):
        o = (L - len(c)) // 2
        K[2 * i, o:o + len(c)] = c
        K[2 * i + 1, o:o + len(s)] = s
    return K, table


def transient_dictionary(fs=FS):
    """Fixed low-pass transient atoms <= 1 s, zero mean, >= 80 % energy below 5 Hz (R3):
    biphasic KC-like shapes with 3 durations x 2 polarity ratios + derivative-of-Gaussian bumps."""
    atoms, table = [], []
    for d in (0.6, 0.8, 1.0):
        for ratio in (0.3, 0.7):
            atoms.append(kc_shape(fs, d, ratio))
            table.append({"cls": "transient", "d": d, "shape": f"kc{ratio}"})
    for d in (0.4, 0.7):
        n = int(d * fs)
        t = (np.arange(n) - n / 2) / fs
        w = -t * np.exp(-0.5 * (t / (d / 6)) ** 2)
        w -= w.mean()
        atoms.append(w / np.linalg.norm(w))
        table.append({"cls": "transient", "d": d, "shape": "dog"})
    L = max(len(a) for a in atoms)
    K = np.zeros((len(atoms), L), np.float32)
    for i, a in enumerate(atoms):
        o = (L - len(a)) // 2
        K[i, o:o + len(a)] = a
    return K, table


def slowwave_dictionary(freqs=(0.5, 0.65, 0.8, 1.0), asyms=(0.4, 0.6), fs=FS):
    """Slow-wave class: one-cycle zero-mean waves of 1-2.5 s (R2)."""
    atoms, table = [], []
    for f in freqs:
        for a in asyms:
            atoms.append(slow_wave_shape(f, fs, a))
            table.append({"cls": "sw", "f": f, "d": 1 / f, "asym": a})
    L = max(len(a) for a in atoms)
    K = np.zeros((len(atoms), L), np.float32)
    for i, a in enumerate(atoms):
        o = (L - len(a)) // 2
        K[i, o:o + len(a)] = a
    return K, table


def atom_energy_below(w, hz=5.0, fs=FS):
    spec = np.abs(np.fft.rfft(w, 8 * len(w))) ** 2
    f = np.fft.rfftfreq(8 * len(w), 1 / fs)
    return float(spec[f < hz].sum() / (spec.sum() + 1e-12))


def atom_duration(w, frac=0.9, fs=FS):
    e = np.cumsum(w ** 2) / (np.sum(w ** 2) + 1e-12)
    a, b = np.searchsorted(e, (1 - frac) / 2), np.searchsorted(e, 1 - (1 - frac) / 2)
    return (b - a) / fs


# ------------------------------------------------------------------ convolutional sparse coding
class CSC:
    """Block-coordinate FISTA for ``x ~ sum_c conv(z_c, D_c)`` (one step per class with its own
    Lipschitz constant, so a coherent slow class does not slow the burst class), group
    soft-threshold on the analytic pairs with a per-class price, then (R5): selection by
    normalized correlation on the explained-away residual, non-maximum suppression within ±d/2,
    per-class budget, joint least-squares refit of the amplitudes on the support (ridge 1e-3) and
    amplitude cap ``<= cap x RMS`` of the epoch."""

    def __init__(self, dicts, prices, budgets, groups, nms_frac=0.5, n_iter=40, cap=5.0, nc_min=0.0,
                 device=DEV):
        self.device = device
        self.D = [torch.tensor(K, device=device)[:, None] for K in dicts]
        self.L = [K.shape[1] for K in dicts]
        self.prices = list(prices)
        self.budgets = list(budgets)
        self.groups = list(groups)
        self.nms_frac = nms_frac
        self.n_iter = n_iter
        self.cap = cap
        self.nc_min = nc_min
        self.bands = None  # per class, per atom (lo, hi) for the whitened nc; set by build()
        self.steps = None

    def _conv(self, z, c):  # synthesis: (B, K, T) -> (B, T); atom k centred at its onset index
        L = self.L[c]
        return F.conv_transpose1d(z, self.D[c])[:, 0, L // 2:L // 2 + z.shape[-1]]

    def _corr(self, r, c):  # analysis (exact adjoint of _conv): (B, T) -> (B, K, T)
        L = self.L[c]
        return F.conv1d(F.pad(r[:, None], (L // 2, L - 1 - L // 2)), self.D[c])

    def synth(self, zs):
        return sum(self._conv(z, c) for c, z in enumerate(zs))

    def _lipschitz(self, c, T):
        z = torch.randn(1, self.D[c].shape[0], T, device=self.device)
        for _ in range(20):
            z = self._corr(self._conv(z, c), c)
            z = z / (z.norm() + 1e-12)
        return float(self._corr(self._conv(z, c), c).norm().item())

    def _prox(self, z, c, t):
        lam = self.prices[c] * t
        if self.groups[c] == 2:
            B, K2, T = z.shape
            zz = z.reshape(B, K2 // 2, 2, T)
            mag = zz.norm(dim=2, keepdim=True)
            return (zz * (1 - lam / (mag + 1e-9)).clamp_min(0)).reshape(B, K2, T)
        return torch.sign(z) * (z.abs() - lam).clamp_min(0)

    def fista(self, x, n_iter=None):
        """``x`` (B, T) float tensor -> dense codes per class (block-coordinate FISTA)."""
        B, T = x.shape
        if self.steps is None:
            self.steps = [1.0 / (self._lipschitz(c, T) * 1.02) for c in range(len(self.D))]
        zs = [torch.zeros(B, D.shape[0], T, device=self.device) for D in self.D]
        ys = [z.clone() for z in zs]
        tk = 1.0
        for _ in range(n_iter or self.n_iter):
            new = []
            for c in range(len(self.D)):
                r = x - self.synth([*new, *ys[len(new):]])
                new.append(self._prox(ys[c] + self.steps[c] * self._corr(r, c), c, self.steps[c]))
            tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
            ys = [n + (tk - 1) / tk1 * (n - z) for n, z in zip(new, zs)]
            zs, tk = new, tk1
        return zs

    def magnitude(self, z, c):
        if self.groups[c] == 2:
            B, K2, T = z.shape
            return z.reshape(B, K2 // 2, 2, T).norm(dim=2)
        return z.abs()

    def _bandfilter(self, r, lo, hi):
        """Ideal band-pass by FFT mask (B, T) -> (B, T)."""
        T = r.shape[-1]
        f = torch.fft.rfftfreq(T, 1 / FS).to(r.device)
        m = ((f >= lo) & (f < hi)).float()
        return torch.fft.irfft(torch.fft.rfft(r, dim=-1) * m, n=T, dim=-1)

    @staticmethod
    def _context_q20(loc, ctx=600, hop=10):
        """20th percentile of ``loc`` (B, T) over a sliding context of ``ctx`` samples (decimated by
        ``hop``): the in-band background level under the candidate, as the masked q20 of the model (R1)."""
        B, T = loc.shape
        d = loc[:, ::hop]
        n = ctx // hop
        u = F.pad(d[:, None], (n // 2, n // 2), mode="replicate")[:, 0].unfold(1, n + 1, 1)  # (B, Td, n+1)
        q = u.kthvalue(max(1, int(0.2 * (n + 1))), dim=-1).values
        return q.repeat_interleave(hop, 1)[:, :T]

    def normalized_corr(self, r, c, atom_dur):
        """Score of a candidate (B, K, T). Classes B and T: in-band SNR = |<r_b, phi_k>| / sqrt(q20 of
        the in-band local energy over the +-3 s context), i.e. the event energy against the stationary
        background level of the same band (A7-like relative power, R1); unbounded, calibrated on a null.
        Class S (index 2): broadband normalized correlation (shape) in [0, 1], the amplitude rule gates."""
        K = self.D[c].shape[0] // self.groups[c]
        bands = self.bands[c] if getattr(self, "bands", None) is not None else [(0.0, FS / 2)] * K
        cache, out = {}, None
        for k in range(K):
            b = bands[k] if c != 2 else (0.0, FS / 2)
            if b not in cache:
                rb = self._bandfilter(r, *b) if b[1] < FS / 2 else r
                e2 = rb.pow(2)[:, None]
                cache[b] = (self.magnitude(self._corr(rb, c), c), e2, 1e-4 * e2.mean())
            corr, e2, floor = cache[b]
            if out is None:
                out = torch.zeros_like(corr)
            w = (int(atom_dur[c][k] * FS) + 4) | 1
            loc = F.avg_pool1d(e2, w, stride=1, padding=w // 2)[:, 0, :corr.shape[-1]] * w
            if c == 2:
                den = loc
            else:  # context q20, floored at a quarter of the epoch median (a burst is judged against its band's floor)
                den = torch.maximum(self._context_q20(loc), 0.25 * loc.median(-1, keepdim=True).values)
            out[:, k] = corr[:, k] / (den + floor * w).sqrt()
        return out

    def _ls_refit(self, x, events, T):
        """Joint least-squares refit of the amplitudes of ``events`` (per sample), ridge 1e-3, cap."""
        rms = x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-9
        xhat = torch.zeros_like(x)
        out = []
        for b in range(len(events)):
            ev = events[b]
            if not ev:
                out.append([])
                continue
            cols = []
            for (c, k, tau, v) in ev:
                g, L = self.groups[c], self.L[c]
                for j in range(g):
                    col = torch.zeros(T + 2 * L, device=self.device)
                    s0 = tau - L // 2 + L
                    col[s0:s0 + L] = self.D[c][g * k + j, 0]
                    cols.append(col[L:L + T])
            A = torch.stack(cols, 1)
            G = A.T @ A + 1e-3 * torch.eye(A.shape[1], device=self.device)
            coef = torch.linalg.solve(G, A.T @ x[b])
            capv = self.cap * rms[b, 0] * math.sqrt(T)
            res, j = [], 0
            for (c, k, tau, v) in ev:
                g = self.groups[c]
                m = coef[j:j + g].norm().item()
                if m > capv:
                    coef[j:j + g] = coef[j:j + g] * (capv / m)
                res.append((c, k, tau, float(coef[j:j + g].norm().item()), v, [float(u) for u in coef[j:j + g]]))
                j += g
            xhat[b] = A @ coef
            out.append(res)
        return out, xhat

    def _candidates(self, r_c, z, c, atom_dur, score):
        """Score, per-atom NMS, greedy cross-atom NMS and budget for one class on its residual."""
        B, T = r_c.shape
        mag = self.magnitude(z, c)
        if score == "nc":
            sc = self.normalized_corr(r_c, c, atom_dur)
        else:
            sc = mag / (r_c.pow(2).mean(-1, keepdim=True).sqrt() + 1e-9)[:, :, None]
        sc = sc * ((mag > 0) & (sc > self.nc_min))
        kept = torch.zeros_like(sc, dtype=torch.bool)
        for k in range(sc.shape[1]):
            w = max(3, int(atom_dur[c][k] * FS * self.nms_frac) | 1)
            mp = F.max_pool1d(sc[:, k:k + 1], w, stride=1, padding=w // 2)[:, 0, :T]
            kept[:, k] = (sc[:, k] >= mp) & (sc[:, k] > 0)
        sc = sc * kept
        flat = sc.reshape(B, -1)
        val, idx = flat.topk(min(4 * self.budgets[c], flat.shape[1]), dim=-1)
        events = []
        for b in range(B):
            acc = []
            for v, i in zip(val[b].tolist(), idx[b].tolist()):
                if v <= 0 or len(acc) >= self.budgets[c]:
                    break
                k, tau = i // T, i % T
                if all(abs(tau - t2) > max(atom_dur[c][k], atom_dur[c][k2]) * FS * self.nms_frac for (_, k2, t2, _) in acc):
                    acc.append((c, k, tau, float(v)))
            events.append(acc)
        return events

    def select(self, x, zs, table, atom_dur, score="nc", order=(2, 1, 0), sw_peak_min=None):
        """R1/R5: class-sequential explaining-away (slow waves first, then transients, then bursts): each
        class is scored on x minus the least-squares synthesis of the classes already selected (the
        dense FISTA codes only provide the candidate supports), its events pass NMS and budget, its
        amplitudes are refit; the slow-wave events below the amplitude rule are dropped before the
        residual is passed on. A final joint LS refit over all classes gives the amplitudes."""
        B, T = x.shape
        sw_peak_min = SW_PEAK_MIN if sw_peak_min is None else sw_peak_min
        selected = [[] for _ in range(B)]
        explained = torch.zeros_like(x)
        pending = set(range(len(zs)))
        for c in order:
            pending.discard(c)
            r_c = x - explained  # each class sees x minus what the classes before it explained (R1)
            ev_c = self._candidates(r_c, zs[c], c, atom_dur, score)
            ref, xh = self._ls_refit(r_c, ev_c, T)
            if c == 2:  # amplitude rule of the slow-wave class before passing the residual on
                g = self.groups[c]
                peaks = self.D[c][::g, 0].abs().amax(-1)
                keep = [[e for e in ev if e[3] * float(peaks[e[1]]) >= sw_peak_min] for ev in ref]
                ref, xh = self._ls_refit(r_c, [[(e[0], e[1], e[2], e[4]) for e in ev] for ev in keep], T)
            explained = explained + xh
            for b in range(B):
                selected[b] += [(e[0], e[1], e[2], e[4]) for e in ref[b]]
        return self._ls_refit(x, selected, T)

    def run(self, x, table, atom_dur, **kw):
        zs = self.fista(x)
        return self.select(x, zs, table, atom_dur, **kw)


def topk_per_token(csc, x, atom_dur, k=2, token=2.0):
    """Baseline of the original proposal: matched-filter maps, top-k per 2 s token, no explaining-away."""
    B, T = x.shape
    events = [[] for _ in range(B)]
    xhat = torch.zeros_like(x)
    rms = x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-9
    n_tok = int(T / (token * FS))
    for c in range(len(csc.D)):
        m = csc.magnitude(csc._corr(x, c), c) / rms[:, :, None]  # (B,K,T)
        for b in range(B):
            for t0 in range(n_tok):
                seg = m[b, :, t0 * int(token * FS):(t0 + 1) * int(token * FS)]
                val, idx = seg.flatten().topk(k)
                for v, i in zip(val.tolist(), idx.tolist()):
                    events[b].append((c, i // seg.shape[1], t0 * int(token * FS) + i % seg.shape[1], v))
    for b in range(B):
        for (c, k_, tau, v) in events[b]:
            g = csc.groups[c]
            e = torch.zeros(1, csc.D[c].shape[0], T, device=csc.device)
            a = csc._corr(x[b:b + 1], c)[0, g * k_:g * k_ + g, tau]
            for j in range(g):
                e[0, g * k_ + j, tau] = a[j]
            xhat[b] += csc._conv(e, c)[0]
    return events, xhat


# ------------------------------------------------------------------ background estimators
def seg_band_power(x, bands, seg=1.0, hop=0.5, fs=FS):
    """``x`` (B, T) numpy -> (B, n_seg, n_bands) band power per ``seg`` s segment, hop ``hop`` s."""
    n, h = int(seg * fs), int(hop * fs)
    starts = np.arange(0, x.shape[1] - n + 1, h)
    win = np.hanning(n)
    segs = np.stack([x[:, s:s + n] * win for s in starts], 1)  # (B, S, n)
    spec = np.abs(np.fft.rfft(segs, axis=-1)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    out = np.stack([spec[..., (f >= a) & (f < b)].sum(-1) for a, b in bands], -1)
    return out, starts


def background_estimate(x, bands, mask=None, est="q20", seg=1.0, hop=0.5, ctx=6.0, fs=FS):
    """Per-segment background band power (B, S, nb). ``est`` in {median, q20, q25}; ``mask`` (B, T)
    bool marks the event supports: masked segments are excluded from the quantile (R1). The
    quantile is taken over the ``ctx`` s context centred on the segment (R10: window 3-6 s)."""
    P, starts = seg_band_power(x, bands, seg, hop, fs)
    n = int(seg * fs)
    if mask is not None:
        segm = np.stack([mask[:, s:s + n].mean(-1) for s in starts], 1) > 0.3  # (B,S) segment is masked
    else:
        segm = np.zeros(P.shape[:2], bool)
    q = {"median": 50, "q20": 20, "q25": 25}[est]
    half = int(ctx / hop / 2)
    out = np.zeros_like(P)
    S = P.shape[1]
    for s in range(S):
        a, b = max(0, s - half), min(S, s + half + 1)
        for i in range(P.shape[0]):
            keep = ~segm[i, a:b]
            vals = P[i, a:b][keep] if keep.sum() >= 2 else P[i, a:b]
            out[i, s] = np.percentile(vals, q, axis=0)
    return out, starts, P


# ------------------------------------------------------------------ real data
def load_subjects(site="mass_ss01", n=None, ch="EEG"):
    subs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{CACHE}/{site}/labels/*.npy"))
    if n:
        subs = subs[:n]
    out = []
    for s in subs:
        p = glob.glob(f"{CACHE}/{site}/signals/{s}/{ch}_clean_*/*/signal.npy")
        if not p:
            continue
        x = np.load(p[0], mmap_mode="r")
        y = np.load(f"{CACHE}/{site}/labels/{s}.npy")
        n_min = min(len(x), len(y))  # one SS01 night has 1208 epochs of signal and 1207 labels
        out.append((s, x[:n_min], y[:n_min]))
    return out


def pick_epochs(subjects, stage, per_subject, rng):
    """Random epochs of a stage from each subject -> (X (n, T) float32, subject index array)."""
    X, sid = [], []
    for i, (s, x, y) in enumerate(subjects):
        idx = np.where(y == stage)[0]
        if len(idx) == 0:
            continue
        sel = rng.choice(idx, min(per_subject, len(idx)), replace=False)
        X.append(np.asarray(x[np.sort(sel)], np.float32))
        sid += [i] * len(sel)
    return np.concatenate(X), np.array(sid)


def ap_score(scores, labels):
    """Average precision from arrays of scores and binary labels (sklearn-free)."""
    o = np.argsort(-scores)
    l = labels[o]
    if l.sum() == 0:
        return 0.0
    tp = np.cumsum(l)
    prec = tp / (np.arange(len(l)) + 1)
    return float((prec * l).sum() / l.sum())


def match_events(pred, truth, tol):
    """Greedy matching of predicted onsets (sorted by score desc) to true onsets within ``tol``
    samples: returns labels per prediction (1 hit) and the number of missed truths."""
    used = np.zeros(len(truth), bool)
    lab = np.zeros(len(pred), int)
    for i, p in enumerate(pred):
        d = np.abs(truth - p)
        d[used] = 10 ** 9
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= tol:
            used[j] = True
            lab[i] = 1
    return lab, int((~used).sum())


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ------------------------------------------------------------------ model assembly and null calibration
CLS = ("burst", "transient", "sw")


def build(prices=(1.0, 1.5, 2.0), budgets=(10, 10, 20), n_iter=40, cycles=(3, 5, 8, 12), bands=BURST_BANDS,
          cap=5.0, device=DEV):
    """Default P2 inference for the tests: three fixed dictionaries (the learned model initialises
    its parametric atoms on the same grid). Returns ``(csc, tables, durs)``."""
    KB, tb = burst_dictionary(bands=bands, cycles=cycles)
    KT, tt = transient_dictionary()
    KS, ts = slowwave_dictionary()
    csc = CSC([KB, KT, KS], prices=list(prices), budgets=list(budgets), groups=[2, 1, 1], n_iter=n_iter,
              cap=cap, device=device)
    tables = [tb, tt, ts]
    durs = [[a["d"] for a in t] for t in tables]
    csc.bands = [[tuple(bands[a["band"]]) for a in tb], [(0.0, 5.0)] * len(tt), [(0.0, 2.0)] * len(ts)]
    return csc, tables, durs


def infer(csc, durs, X, batch=32, **kw):
    """numpy (n, T) -> (events per epoch, xhat numpy)."""
    evs, xh = [], []
    for i in range(0, len(X), batch):
        x = torch.tensor(np.asarray(X[i:i + batch], np.float32), device=csc.device)
        e, h = csc.run(x, None, durs, **kw)
        evs += e
        xh.append(h.cpu().numpy())
    return evs, np.concatenate(xh)


def calibrate_nc(csc, durs, rng, n=192, q=99.0, bg="1/f"):
    """Null of the normalized correlation: event-free synthetic background, the ``q``-th percentile
    of the selected candidates' nc per class = closed-form prior for the detection gate (one false
    alarm per 100 candidates). Returns thresholds per class."""
    X = np.stack([synth_epoch(rng, bg=bg, n_spindles=0, n_kc=0, n_sw=0)[0] for _ in range(n)])
    evs, _ = infer(csc, durs, X)
    return _thresholds(csc, evs, q)


SW_PEAK_MIN = 3.0  # ~75 uV at a night scale of 25 uV (rule_masks convention): the AASM slow-wave amplitude
SW_NC_MIN = 0.5    # shape gate of the slow-wave class (fixed, not from a null: see PROGETTO.md section 3)


def _thresholds(csc, evs, q):
    """Per class and per atom: ``q``-th percentile of the score of the null candidates of that atom
    (atoms with fewer than 20 null candidates inherit the class percentile). Class S: fixed shape gate."""
    thr = []
    for c in range(len(csc.D)):
        K = csc.D[c].shape[0] // csc.groups[c]
        allv = np.array([e[4] for ev in evs for e in ev if e[0] == c])
        base = float(np.percentile(allv, q)) if len(allv) else 1.0
        row = np.full(K, base)
        for k in range(K):
            v = np.array([e[4] for ev in evs for e in ev if e[0] == c and e[1] == k])
            if len(v) >= 20:
                row[k] = float(np.percentile(v, q))
        thr.append(row if c != 2 else np.full(K, SW_NC_MIN))
    return thr


def gate(evs, thr, csc=None, sw_peak_min=SW_PEAK_MIN):
    """Class B and T: nc >= null-calibrated threshold. Class S (index 2): nc >= SW_NC_MIN and decoded
    peak amplitude >= ``sw_peak_min`` (absolute amplitude rule, R2); needs ``csc`` for the atom peaks."""
    out = []
    for ev in evs:
        keep = []
        for e in ev:
            c, k, tau, a, v, coef = e
            if c == 2 and csc is not None:
                g = csc.groups[c]
                peak = a * float(csc.D[c][g * k, 0].abs().max().item())
                if v >= SW_NC_MIN and peak >= sw_peak_min:
                    keep.append(e)
            elif v >= float(np.asarray(thr[c]).reshape(-1)[k] if np.ndim(thr[c]) else thr[c]):
                keep.append(e)
        out.append(keep)
    return out


def resynth(csc, evs, T=EPOCH, classes=None):
    """Synthesis of the (gated) event lists -> numpy (n, T); ``classes`` restricts to those classes."""
    out = np.zeros((len(evs), T), np.float32)
    for b, ev in enumerate(evs):
        for (c, k, tau, a, v, coef) in ev:
            if classes is not None and c not in classes:
                continue
            g, L = csc.groups[c], csc.L[c]
            for j in range(g):
                w = csc.D[c][g * k + j, 0].cpu().numpy() * coef[j]
                s0 = tau - L // 2
                a0, b0 = max(0, s0), min(T, s0 + L)
                if b0 > a0:
                    out[b, a0:b0] += w[a0 - s0:b0 - s0]
    return out


def support_mask(evs, durs, T=EPOCH, pad=0.0):
    m = np.zeros((len(evs), T), bool)
    for b, ev in enumerate(evs):
        for (c, k, tau, a, v, coef) in ev:
            h = int((durs[c][k] / 2 + pad) * FS)
            m[b, max(0, tau - h):min(T, tau + h + 1)] = True
    return m


def bandpass(x, lo, hi, fs=FS):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def surrogate(X, rng):
    """Phase-randomised surrogate of every epoch (same amplitude spectrum, random phases):
    a spectrum-matched null for the nc gate that needs no labels and exists on any extract."""
    S = np.fft.rfft(X, axis=-1)
    ph = np.exp(2j * np.pi * rng.uniform(0, 1, S.shape))
    ph[:, 0] = 1
    return np.fft.irfft(np.abs(S) * ph, n=X.shape[-1], axis=-1).astype(np.float32)


def calibrate_nc_surrogate(csc, durs, X, rng, q=99.0, n_max=192):
    """nc gate per class from the phase-randomised surrogates of the data themselves."""
    Xs = surrogate(X[:n_max], rng)
    evs, _ = infer(csc, durs, Xs)
    return _thresholds(csc, evs, q)
