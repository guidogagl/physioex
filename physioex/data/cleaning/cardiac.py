"""Cardiac field removal: R-peak detection and peak-aligned template subtraction."""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt

# Plausible RR range (s): 30-180 bpm.
RR_MIN, RR_MAX = 0.33, 2.0


def _band(x, fs, lo, hi, order=3):
    hi = min(hi, 0.45 * fs)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def detect_r_peaks(
    ecg: np.ndarray, fs: float, min_quality: float = 0.8
) -> Tuple[Optional[np.ndarray], float]:
    """R-peak times in seconds, or None when the signal does not carry a usable heartbeat.

    The QRS band (5-20 Hz) is taken with its polarity (sign of the larger tail,
    so inverted leads work); peaks at least RR_MIN apart above half the 99th
    percentile in 10 s blocks (and above a quarter of the night median of that
    percentile) are kept. Quality is the share of RR intervals
    within [RR_MIN, RR_MAX] that differ from the local median RR by less than
    20%; below ``min_quality`` the detection is rejected. This makes the
    function usable on surrogate channels (chin, mastoid) when no ECG is
    recorded: it returns None rather than noise.
    """
    x = _band(np.asarray(ecg, dtype=np.float64), fs, 5.0, 20.0)
    if np.percentile(x, 99.5) < -np.percentile(x, 0.5):
        x = -x
    block = int(10 * fs)
    starts = range(0, len(x), block)
    p99 = np.array([np.percentile(x[s : s + block], 99) for s in starts])
    floor = 0.25 * np.median(
        p99
    )  # blocks without a heartbeat do not lower the bar to the noise
    thr = np.repeat(np.maximum(0.5 * p99, floor), block)[: len(x)]
    pk, _ = find_peaks(x, distance=max(1, int(RR_MIN * fs)))
    pk = pk[x[pk] > thr[pk]]
    if len(pk) < 10:
        return None, 0.0
    rr = np.diff(pk) / fs
    med = median_filter(rr, size=15, mode="nearest")
    ok = (rr >= RR_MIN) & (rr <= RR_MAX) & (np.abs(rr - med) < 0.2 * med)
    quality = float(ok.mean())
    if quality < min_quality:
        return None, quality
    return pk / fs, quality


def peak_aligned_mean(
    x: np.ndarray, fs: float, peaks_s: np.ndarray, pre: float = 0.2, post: float = 0.4
) -> np.ndarray:
    """Average of x around the R peaks: what is time-locked to the heartbeat."""
    a, b = int(round(pre * fs)), int(round(post * fs))
    idx = np.round(np.asarray(peaks_s) * fs).astype(int)
    idx = idx[(idx - a >= 0) & (idx + b < len(x))]
    if len(idx) == 0:
        return np.zeros(a + b)
    return x[idx[:, None] + np.arange(-a, b)[None, :]].mean(0)


def subtract_cardiac(
    x: np.ndarray,
    fs: float,
    peaks_s: np.ndarray,
    window_s: float = 120.0,
    pre: float = 0.2,
    post: float = 0.4,
) -> np.ndarray:
    """Subtract from x, around every R peak, the mean of the peak-aligned segments
    within +-window_s/2 (moving template: amplitude and shape change with posture).

    Segments that overlap at high heart rate are subtracted once per peak; the
    template is tapered (Tukey, 10%) so no step is introduced at its edges.
    """
    x = np.asarray(x, dtype=np.float64)
    a, b = int(round(pre * fs)), int(round(post * fs))
    idx = np.round(np.asarray(peaks_s) * fs).astype(int)
    idx = idx[(idx - a >= 0) & (idx + b < len(x))]
    if len(idx) == 0:
        return x.copy()
    seg = x[idx[:, None] + np.arange(-a, b)[None, :]]
    cs = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, 0)])
    t = idx / fs
    lo = np.searchsorted(t, t - window_s / 2)
    hi = np.searchsorted(t, t + window_s / 2, side="right")
    tmpl = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    n = a + b
    taper = np.ones(n)
    k = max(1, int(0.1 * n))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    taper[:k], taper[-k:] = ramp, ramp[::-1]
    out = x.copy()
    for i, s in enumerate(idx):
        out[s - a : s + b] -= taper * tmpl[i]
    return out


def cardiac_residual(
    x: np.ndarray, fs: float, peaks_s: np.ndarray, pre: float = 0.2, post: float = 0.4
) -> float:
    """Peak-to-peak amplitude (signal units) of the heartbeat-locked average."""
    m = peak_aligned_mean(x, fs, peaks_s, pre, post)
    return float(m.max() - m.min())
