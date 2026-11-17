"""Class dictionaries of psg_gen (SPEC section 2, ``dicts.py``).

Three atom families, on the shared convention (SPEC section 0):

* **bursts** (class 0): analytic Gabor atoms, a cos/sin pair per (band, frequency, duration);
  f on a ``f_step`` Hz grid inside each burst band, duration ``d = cycles / f`` from the cycles
  grid with ``d >= max(0.5, 3 / lo)`` (AASM 0.5 s floor, 3 cycles at the band edge) and
  ``d <= d_max`` (alpha also 5 and 8 s), kept only if the spectral concentration in the band
  is >= 0.8 (R3, R6);
* **transients** (class 1): zero-mean biphasic K-complex-like shapes (sharp negative lobe,
  slower positive lobe) at three scales, <= 1 s, >= 80 % of the energy below 4 Hz (a 5 Hz gate
  would admit theta bursts);
* **slow waves** (class 2): one zero-mean cycle of 1-2.5 s, low-pass 2 Hz (R2).

Every atom is unit L2 norm and centred: the centre of its 90 % energy support sits at index
``L // 2`` of the zero-padded length ``L``. :func:`random_dictionary` is the null column (R13):
same classes, durations and bands, random band-limited waveforms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch
import torch.nn.functional as F

from physioex.models.msvq import BANDS as BG_BANDS

FS = 100
EPOCH = 3000
CLASSES = ("burst", "transient", "slowwave")
CLS_BURST, CLS_TRANSIENT, CLS_SLOWWAVE = 0, 1, 2
BURST_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0), "alpha": (8.0, 11.0), "sigma": (11.0, 16.0), "beta": (16.0, 30.0),
}
BURST_BAND_NAMES = tuple(BURST_BANDS)


@dataclass
class Dictionary:
    """Unit-norm atoms centred at ``L // 2`` and zero-padded to the common length ``L``."""

    atoms: torch.Tensor        # (K, L) float32, unit L2 norm
    quad: torch.Tensor         # (K, L) quadrature (sin) partner for bursts, zeros otherwise
    cls: torch.Tensor          # (K,) long in {CLS_BURST, CLS_TRANSIENT, CLS_SLOWWAVE}
    band: torch.Tensor         # (K,) long, index in BURST_BANDS order for bursts, -1 otherwise
    freq: torch.Tensor         # (K,) float, centre frequency in Hz for bursts, nan otherwise
    dur: torch.Tensor          # (K,) float, duration in seconds (construction; the 90 %
    # energy support of the analytic atoms, shared by the cos/sin pair)
    family: torch.Tensor       # (K,) long, family id: bursts share one per (band, freq) across
    # durations; the other atoms have one family each
    names: list[str]           # K human-readable names, e.g. "sigma_13Hz_5cyc", "kc_0.7s"

    def to(self, device) -> "Dictionary":
        return replace(self, atoms=self.atoms.to(device), quad=self.quad.to(device),
                       cls=self.cls.to(device), band=self.band.to(device),
                       freq=self.freq.to(device), dur=self.dur.to(device),
                       family=self.family.to(device), names=list(self.names))

    def __len__(self) -> int:
        return self.atoms.shape[0]

    @property
    def n_family(self) -> int:
        return int(self.family.max().item()) + 1 if len(self) else 0


# ----------------------------------------------------------------------------- geometry
def _hann(n: int) -> torch.Tensor:
    if n < 3:
        return torch.ones(n)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(n) / (n - 1))


def _energy_center(a: torch.Tensor) -> float:
    """Index of the centre of the 90 % energy support of ``a``."""
    e = a.square()
    tot = e.sum()
    if tot < 1e-12:
        return (a.numel() - 1) / 2.0
    cs = e.cumsum(0) / tot
    i0 = int(torch.searchsorted(cs, torch.tensor(0.05)).item())
    i1 = int(torch.searchsorted(cs, torch.tensor(0.95)).item())
    return 0.5 * (i0 + i1)


def _center(a: torch.Tensor, L: int) -> torch.Tensor:
    """Place the centre of the 90 % energy support of ``a`` at index ``L // 2`` of length L."""
    out = torch.zeros(L)
    sh = L // 2 - int(round(_energy_center(a)))
    idx = torch.arange(a.numel()) + sh
    m = (idx >= 0) & (idx < L)
    out[idx[m]] = a[m]
    return out


def _finalize(atoms: list[torch.Tensor], quads: Sequence[torch.Tensor | None], cls: list[int],
              band: list[int], freq: list[float], durs: list[float], family: list[int],
              names: list[str], fs: int) -> Dictionary:
    """Centre, unit-normalise and pad a list of natural-length atoms into a Dictionary.
    ``durs`` are the construction durations in seconds."""
    if not atoms:
        raise ValueError("empty atom list")
    L = max(a.numel() for a in atoms)
    A = torch.stack([_center(a, L) for a in atoms])
    A = A / A.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Q = torch.stack([_center(q, L) if q is not None else torch.zeros(L) for q in quads])
    Q = Q / Q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return Dictionary(A, Q, torch.tensor(cls), torch.tensor(band), torch.tensor(freq),
                      torch.tensor(durs), torch.tensor(family), names)


# ----------------------------------------------------------------------------- shape generators
def _gabor(f: float, d: float, phase: float, fs: int) -> torch.Tensor:
    """Burst atom of frequency f and duration d: cos(2 pi f t + phase) under a Hann envelope."""
    n = max(int(round(d * fs)), 4)
    t = (torch.arange(n) - (n - 1) / 2.0) / fs
    return _hann(n) * torch.cos(2 * math.pi * f * t + phase)


def _kc(d: float, fs: int) -> torch.Tensor:
    """Biphasic K-complex-like transient of duration d: sharp negative lobe (0.25 d, sigma
    0.15 d), slower positive lobe (0.62 d, sigma 0.25 d); the lobes stay below ~4 Hz so the
    shape does not mimic a low-frequency burst."""
    n = max(int(round(d * fs)), 4)
    t = torch.arange(n) / fs
    w = -torch.exp(-0.5 * ((t - 0.25 * d) / (0.15 * d)) ** 2) \
        + 0.6 * torch.exp(-0.5 * ((t - 0.62 * d) / (0.25 * d)) ** 2)
    w = w * _hann(n)
    return w - w.mean()


def _slow_wave(d: float, fs: int) -> torch.Tensor:
    """One zero-mean cycle of duration d (negative half then positive half), tukey-windowed."""
    n = max(int(round(d * fs)), 4)
    w = -torch.sin(2 * math.pi * torch.arange(n) / n)
    m = max(int(0.15 * n), 1)
    h = _hann(2 * m)
    w = w * torch.ones(n)
    w[:m] *= h[:m]
    w[-m:] *= h[m:]
    return w - w.mean()


# ----------------------------------------------------------------------------- banks
def gabor_bank(bands: dict[str, tuple[float, float]] = BURST_BANDS, f_step: float = 1.0,
               cycles: Sequence[int] = (3, 5, 8, 12), d_max: float = 3.0,
               alpha_long: Sequence[float] = (5.0, 8.0), fs: int = FS) -> Dictionary:
    """Analytic burst bank: cos/sin Gabor pairs, >= 3 cycles, >= 0.8 concentration in the band,
    duration >= max(0.5 s, 3 / lo) (the AASM 0.5 s floor and the 3-cycle duration at the band
    edge; shorter bursts are transient-like and would bleed into the transient class)."""
    atoms, quads, band, freq, durs, family, names = [], [], [], [], [], [], []
    fam: dict[tuple[str, float], int] = {}
    for bnd, (lo, hi) in bands.items():
        bi = list(bands).index(bnd)
        dmin = max(0.5, 3.0 / lo)
        for f in torch.arange(lo, hi + 1e-6, f_step).tolist():
            ds = [c / f for c in cycles if dmin <= c / f <= d_max]
            if bnd == "alpha":
                ds += [d for d in alpha_long if dmin <= d]
            if not ds:
                continue
            fid = None
            for d in sorted(set(ds)):
                c = _gabor(f, d, 0.0, fs)
                if spectral_concentration(c, (lo, hi), fs) < 0.8:
                    continue
                if fid is None:
                    fid = fam.setdefault((bnd, float(f)), len(fam))
                s = _gabor(f, d, -math.pi / 2, fs)
                atoms += [c, s]
                quads += [s, c]
                band += [bi, bi]
                freq += [float(f), float(f)]
                durs += [d, d]
                family += [fid, fid]
                tag = f"{d:g}s" if bnd == "alpha" and d in alpha_long else f"{int(round(d * f))}cyc"
                names += [f"{bnd}_{f:g}Hz_{tag}"] * 2
    return _finalize(atoms, quads, [CLS_BURST] * len(atoms), band, freq, durs, family, names, fs)


def transient_bank(scales: Sequence[float] = (0.5, 0.7, 0.9), fs: int = FS) -> Dictionary:
    """Fixed transient bank: K-complex-like shapes at the given scales, kept if >= 0.8 of the
    energy is below 4 Hz (a 5 Hz gate would admit theta bursts)."""
    atoms, durs, names = [], [], []
    for d in scales:
        if energy_below(_kc(d, fs), 4.0, fs) < 0.8:
            continue
        atoms.append(_kc(d, fs))
        durs.append(float(d))
        names.append(f"kc_{d:g}s")
    return _finalize(atoms, [None] * len(atoms), [CLS_TRANSIENT] * len(atoms),
                     [-1] * len(atoms), [float("nan")] * len(atoms),
                     durs, list(range(len(atoms))), names, fs)


def slowwave_bank(periods: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5), fs: int = FS) -> Dictionary:
    """Slow-wave bank: one zero-mean cycle per period (1-2.5 s), low-pass 2 Hz by construction."""
    atoms, durs, names = [], [], []
    for d in periods:
        atoms.append(_slow_wave(d, fs))
        durs.append(float(d))
        names.append(f"sw_{d:g}s")
    return _finalize(atoms, [None] * len(atoms), [CLS_SLOWWAVE] * len(atoms),
                     [-1] * len(atoms), [float("nan")] * len(atoms),
                     durs, list(range(len(atoms))), names, fs)


def concat(*ds: Dictionary) -> Dictionary:
    """Concatenate dictionaries, re-padded to the max length and re-indexed on contiguous
    families (in the order of appearance)."""
    if not ds:
        raise ValueError("concat needs at least one dictionary")
    L = max(d.atoms.shape[1] for d in ds)
    next_fam = 0
    atoms, quads, cls, band, freq, durs, fams, names = [], [], [], [], [], [], [], []
    for d in ds:
        ids = d.family.tolist()
        local: dict[int, int] = {}
        for f in ids:
            if f not in local:
                local[f] = next_fam
                next_fam += 1
        atoms.append(torch.stack([_center(d.atoms[k], L) for k in range(len(d))]))
        quads.append(torch.stack([_center(d.quad[k], L) for k in range(len(d))]))
        cls.append(d.cls)
        band.append(d.band)
        freq.append(d.freq)
        durs.append(d.dur)
        fams.append(torch.tensor([local[f] for f in ids]))
        names += list(d.names)
    return Dictionary(torch.cat(atoms), torch.cat(quads), torch.cat(cls), torch.cat(band),
                      torch.cat(freq), torch.cat(durs), torch.cat(fams), names)


def full_dictionary(fs: int = FS, **kw) -> Dictionary:
    """slowwave + transient + gabor (``kw`` goes to :func:`gabor_bank`)."""
    return concat(slowwave_bank(fs=fs), transient_bank(fs=fs), gabor_bank(fs=fs, **kw))


def random_dictionary(like: Dictionary, seed: int, fs: int = FS) -> Dictionary:
    """Null column (R13): same classes, durations and bands as ``like`` with random band-limited
    waveforms (bursts in their band, transients below 5 Hz, slow waves below 2 Hz); the burst
    quadrature is the exact -90 degree partner."""
    gen = torch.Generator().manual_seed(seed)
    L = like.atoms.shape[1]
    atoms, quads = [], []
    for k in range(len(like)):
        c = int(like.cls[k])
        if c == CLS_BURST:
            lo, hi = BURST_BANDS[BURST_BAND_NAMES[int(like.band[k])]]
        elif c == CLS_TRANSIENT:
            lo, hi = 0.0, 5.0
        else:
            lo, hi = 0.0, 2.0
        n = max(int(round(float(like.dur[k]) * fs)), 4)
        m = 4 * n
        f = torch.fft.rfftfreq(m, 1 / fs)
        mask = ((f >= lo) & (f < hi)).float()
        mask[0] = 0.0
        W = torch.fft.rfft(torch.randn(m, generator=gen)) * mask
        a = torch.fft.irfft(W, n=m)[:n] * _hann(n)
        q = torch.fft.irfft(W * torch.tensor(-1j), n=m)[:n] * _hann(n) if c == CLS_BURST \
            else torch.zeros(n)
        atoms.append(a)
        quads.append(q)
    return _finalize(atoms, quads, like.cls.tolist(), like.band.tolist(), like.freq.tolist(),
                     [float(v) for v in like.dur.tolist()], like.family.tolist(),
                     list(like.names), fs)


# ----------------------------------------------------------------------------- analysis
def spectral_concentration(atom: torch.Tensor, band: tuple[float, float], fs: int = FS,
                           nfft: int = 4096) -> float:
    """Fraction of the |FFT|^2 energy of ``atom`` inside ``band``."""
    p = torch.fft.rfft(atom, n=nfft).abs() ** 2
    f = torch.fft.rfftfreq(nfft, 1 / fs)
    m = (f >= band[0]) & (f < band[1])
    return float(p[m].sum() / (p.sum() + 1e-12))


def energy_below(atom: torch.Tensor, hz: float, fs: int = FS):
    """Fraction of the |FFT|^2 energy of ``atom`` below ``hz`` Hz (one atom -> float,
    (K, L) -> (K,) tensor)."""
    p = torch.fft.rfft(atom, n=8 * atom.shape[-1]).abs() ** 2
    f = torch.fft.rfftfreq(8 * atom.shape[-1], 1 / fs)
    out = p[..., f < hz].sum(-1) / (p.sum(-1) + 1e-12)
    return float(out) if out.ndim == 0 else out


def _energy_below_vec(atoms: torch.Tensor, hz: float, fs: int = FS) -> torch.Tensor:
    n = atoms.shape[1]
    p = torch.fft.rfft(atoms, n=8 * n).abs() ** 2
    f = torch.fft.rfftfreq(8 * n, 1 / fs)
    return p[:, f < hz].sum(-1) / (p.sum(-1) + 1e-12)


def coherence(d: Dictionary, same_class_only: bool = True) -> torch.Tensor:
    """(K, K) max |normalised cross-correlation| over all lags between atom pairs (unit-norm
    atoms, so the max of the raw cross-correlation is the normalised one); the quadrature
    partners of the Gabor pairs are zeroed out (orthogonal by construction)."""
    A = d.atoms
    K, L = A.shape
    cc = F.conv1d(A[:, None], A[:, None], padding=L - 1).abs().amax(-1)
    cc.fill_diagonal_(0.0)
    for i in range(K):
        if d.cls[i] != CLS_BURST:
            continue
        for j in range(i + 1, K):
            if d.cls[j] != CLS_BURST or d.band[i] != d.band[j]:
                continue
            if not (torch.isclose(d.freq[i], d.freq[j]) and torch.isclose(d.dur[i], d.dur[j])):
                continue
            if torch.allclose(d.quad[i], d.atoms[j], atol=1e-5) \
                    or torch.allclose(d.quad[j], d.atoms[i], atol=1e-5):
                cc[i, j] = cc[j, i] = 0.0
    if same_class_only:
        cc = cc * (d.cls[:, None] == d.cls[None, :])
    return cc


def overlap_report(d: Dictionary, fs: int = FS) -> dict:
    """Class-constraint report: band separation (energy below 5 Hz) and the cross-class Gram
    coherence of :func:`coherence`, plus the minimum burst cycles (freq x duration)."""
    cc = coherence(d, same_class_only=False)
    bi, ti, si = d.cls == CLS_BURST, d.cls == CLS_TRANSIENT, d.cls == CLS_SLOWWAVE
    eb = _energy_below_vec(d.atoms, 5.0, fs)
    return {
        "burst_energy_below_5hz": float(eb[bi].max()),
        "transient_energy_below_5hz": float(eb[ti].min()),
        "coh_burst_transient": float(cc[bi][:, ti].max()),
        "coh_burst_slowwave": float(cc[bi][:, si].max()),
        "coh_transient_slowwave": float(cc[ti][:, si].max()),
        "min_cycles": float((d.freq[bi] * d.dur[bi]).min()),
    }


# ----------------------------------------------------------------------------- learning
def _ramp(f: torch.Tensor, fc: float, w: float = 0.5) -> torch.Tensor:
    """Raised-cosine 0 -> 1 over ``fc +- w/2``."""
    t = ((f - fc) / w + 1.0).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * t)


def _band_filter(a: torch.Tensor, lo: float, hi: float, fs: int) -> torch.Tensor:
    """Zero-phase band limit of ``a`` with 0.5 Hz raised-cosine transitions (4x oversampled DFT)."""
    m = 4 * a.numel()
    f = torch.fft.rfftfreq(m, 1 / fs)
    wgt = _ramp(f, lo + 0.5) * (1.0 - _ramp(f, hi - 0.5))
    wgt[0] = 0.0 if lo > 0 else 1.0
    return torch.fft.irfft(torch.fft.rfft(a, n=m) * wgt, n=m)[: a.numel()]


def project_constraints(d: Dictionary, fs: int = FS) -> Dictionary:
    """Re-impose the class constraints on (learned) atoms: zero mean, band limits (bursts in
    their band, transients low-pass 5 Hz, slow waves low-pass 2 Hz), max duration (transients
    <= 1 s, slow waves <= 2.5 s), unit norm, re-centred at ``L // 2``. Metadata is unchanged;
    an atom that collapses to zero is kept as it was."""
    max_s = {CLS_TRANSIENT: 1.0, CLS_SLOWWAVE: 2.5}
    L = d.atoms.shape[1]
    out = []
    for k in range(len(d)):
        a = d.atoms[k].clone()
        c = int(d.cls[k])
        if c == CLS_BURST:
            lo, hi = BURST_BANDS[BURST_BAND_NAMES[int(d.band[k])]]
        elif c == CLS_SLOWWAVE:
            lo, hi = 0.0, 2.0
        else:
            lo, hi = 0.0, 5.0
        a = _band_filter(a, lo, hi, fs)
        cmax = max_s.get(c)
        if cmax is not None:
            ctr = _energy_center(a)
            a = a * (torch.arange(a.numel()) - ctr).abs().le(cmax * fs / 2)
        a = a - a.mean()
        if a.norm() < 1e-8:
            a = d.atoms[k]
        else:
            a = a / a.norm()
        out.append(_center(a, L))
    return replace(d, atoms=torch.stack(out))
