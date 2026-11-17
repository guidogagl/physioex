"""Synthetic epoch generators for the psg_gen test suite (SPEC.md section 3).

Conventions: epochs are 30 s at 100 Hz (``T = 3000``) in RobustScale-like units where the
background floor has RMS 1 and a planted slow wave has peak-to-peak 3-5. Atoms and segments
are centred at ``len // 2`` and peak-normalised (max |.| == 1), so ``plant(x, atom, onset, amp)``
adds ``amp * atom`` around ``onset`` (seconds). Seeded torch generators throughout: no call
shares randomness with another.
"""
from __future__ import annotations

import torch

FS = 100
T = 3000


class _Gen:
    """Seeded torch generator with the two numpy-style draws used below."""

    def __init__(self, seed: int):
        self.g = torch.Generator().manual_seed(int(seed))

    def uniform(self, lo: float, hi: float, size=()):
        return lo + (hi - lo) * torch.rand(size, generator=self.g)

    def standard_normal(self, size):
        return torch.randn(size, generator=self.g)


def _gen(seed: int) -> _Gen:
    return _Gen(seed)


def _peak_normalize(a: torch.Tensor) -> torch.Tensor:
    m = a.abs().max()
    return a / m if m > 0 else a


def random_onsets(n: int, seed: int, lo: float = 2.0, hi: float = 28.0, min_gap: float = 2.0) -> list[float]:
    """``n`` onsets (seconds) in [lo, hi] that are pairwise >= ``min_gap`` apart, seeded."""
    gen = _gen(seed)
    out: list[float] = []
    while len(out) < n:
        t0 = float(gen.uniform(lo, hi))
        if all(abs(t0 - o) >= min_gap for o in out):
            out.append(t0)
    return out


def pink_background(T=T, fs=FS, seed: int = 0, scale: float = 1.0, fmin: float = 0.5, fmax: float = 30.0) -> torch.Tensor:
    """1/f (one-over-f) background: random-phase spectrum with PSD ~ 1/f on [fmin, fmax],
    normalised to RMS ``scale``."""
    gen = _gen(seed)
    freqs = torch.linspace(0.0, fs / 2, T // 2 + 1)
    psd = torch.zeros_like(freqs)
    m = (freqs >= fmin) & (freqs <= fmax)
    psd[m] = 1.0 / freqs[m]
    amp = (psd * T / fs).sqrt()
    phase = 2 * torch.pi * gen.uniform(0.0, 1.0, (T // 2 + 1,))
    x = torch.fft.irfft(amp * torch.exp(1j * phase), n=T)
    return x * (scale / x.std())


def ar2_delta(T=T, fs=FS, f0: float = 1.5, zeta: float = 0.5, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    """AR(2) delta background: real AR(2) process resonating at ``f0`` Hz (damping ``zeta``),
    normalised to RMS ``scale``."""
    gen = _gen(seed)
    w = 2.0 * torch.pi * f0 / fs
    r = torch.exp(-zeta * w)
    a1, a2 = 2.0 * r * torch.cos(w), -r * r
    w_in = gen.standard_normal(T)
    x = torch.zeros(T)
    for t in range(2, T):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + w_in[t]  # poles r*exp(+-i w)
    return x * (scale / x.std())


def gabor_spindle(f: float = 13.0, cycles: int = 8, fs=FS) -> torch.Tensor:
    """Gabor burst: ``cycles`` cycles of ``f`` under a Gaussian envelope (sigma = 1/4 of the
    duration), centred, peak-normalised."""
    n = int(round(cycles * fs / f))
    t = (torch.arange(n) - n // 2) / fs
    sigma = cycles / f / 4.0
    return _peak_normalize(torch.exp(-(t ** 2) / (2 * sigma ** 2)) * torch.cos(2 * torch.pi * f * t))


def k_complex(scale: float = 0.7, fs=FS) -> torch.Tensor:
    """K-complex-like transient of ``scale`` seconds (0.5-0.9): negative lobe then positive
    rebound, zero-mean, peak-normalised; >= 0.8 of the energy below 5 Hz."""
    n = int(round(scale * fs))
    t = torch.arange(n) / fs
    a = (-1.0 * torch.exp(-((t - 0.35 * scale) ** 2) / (2 * (0.22 * scale) ** 2))
         + 0.8 * torch.exp(-((t - 0.78 * scale) ** 2) / (2 * (0.18 * scale) ** 2)))
    return _peak_normalize(a - a.mean())


def slow_wave(period: float = 1.5, fs=FS) -> torch.Tensor:
    """One cycle of a sine with period ``period`` seconds (1.0-2.5), raised-cosine windowed,
    zero-mean, peak-normalised (peak-to-peak of a planted wave = 2 * amp)."""
    n = int(round(period * fs))
    t = torch.arange(n) / fs
    win = 0.5 * (1.0 - torch.cos(2 * torch.pi * torch.arange(n) / max(n - 1, 1)))
    a = torch.sin(2 * torch.pi * t / period) * win
    return _peak_normalize(a - a.mean())


def alpha_run(f: float = 10.0, duration: float = 1.5, fs=FS, seed: int = 0) -> torch.Tensor:
    """Alpha run: ``duration`` seconds of ``f`` Hz with a slow 0.25 Hz amplitude modulation,
    raised-cosine ends, peak-normalised."""
    n = int(round(duration * fs))
    t = torch.arange(n) / fs
    win = 0.5 * (1.0 - torch.cos(2 * torch.pi * torch.arange(n) / max(n - 1, 1)))
    mod = 0.7 + 0.3 * torch.sin(2 * torch.pi * 0.25 * t + _gen(seed).uniform(0.0, 6.28))
    return _peak_normalize(win * mod * torch.cos(2 * torch.pi * f * t))


def ecg_train(bpm: float = 60.0, T=T, fs=FS, amp: float = 1.0) -> torch.Tensor:
    """QRS-like transient train at ``bpm`` beats per minute (period 60/bpm s): a train of
    narrow zero-mean spikes, whole-train peak-normalised, multiplied by ``amp``."""
    period = 60.0 / bpm
    n = int(round(period * fs))
    t = torch.arange(n) / fs
    spike = (-0.35 * torch.exp(-((t - 0.145) ** 2) / (2 * 0.012 ** 2))
             + 1.0 * torch.exp(-((t - 0.105) ** 2) / (2 * 0.016 ** 2))
             - 0.5 * torch.exp(-((t - 0.16) ** 2) / (2 * 0.014 ** 2)))
    spike -= spike.mean()
    out = torch.zeros(T)
    t0 = 0.5                                              # seconds; QRS ~0.1 s into each beat
    while int(round(t0 * fs)) + n <= T:
        i = int(round(t0 * fs))
        out[i:i + n] += spike
        t0 += period
    return amp * _peak_normalize(out)


def plant(x: torch.Tensor, atom: torch.Tensor, onset: float, amp: float = 1.0) -> torch.Tensor:
    """Add ``amp * atom`` to the 1-D epoch ``x``, centred at ``onset`` (seconds). In-place;
    returns ``x``. The part of the atom outside the epoch is dropped."""
    c = int(round(onset * FS))
    h = len(atom) // 2
    lo, hi = c - h, c - h + len(atom)
    a0, a1 = max(lo, 0), min(hi, x.numel())
    if a1 > a0:
        x[a0:a1] += amp * atom[a0 - lo:a1 - lo]
    return x


def support_span(onset: float, duration: float, pad: float = 0.2) -> tuple[int, int]:
    """Sample span of an atom of ``duration`` seconds centred at ``onset`` (seconds),
    dilated by ``pad`` seconds on each side, clipped to the epoch."""
    lo = int((onset - duration / 2 - pad) * FS)
    hi = int((onset + duration / 2 + pad) * FS)
    return max(lo, 0), min(hi, T)


def epochs(kind: str = "pink", B: int = 8, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    """(B, T) background epochs with one seeded realisation per row; ``kind`` is "pink"
    (1/f floor) or "ar2" (delta resonance)."""
    out = torch.empty(B, T)
    for b in range(B):
        if kind == "pink":
            out[b] = pink_background(seed=seed + b, scale=scale)
        elif kind == "ar2":
            out[b] = ar2_delta(seed=seed + b, scale=scale)
        else:
            raise ValueError(kind)
    return out
