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
    ecg: np.ndarray,
    fs: float,
    min_quality: float = 0.8,
    block_s: float = 300.0,
    min_coverage: float = 0.1,
) -> Tuple[Optional[np.ndarray], float]:
    """R-peak times in seconds, or None when the signal does not carry a usable heartbeat.

    The QRS band (5-20 Hz) is taken with its polarity (sign of the larger tail,
    so inverted leads work); peaks at least RR_MIN apart above half the 99th
    percentile in 10 s blocks (and above a quarter of the night median of that
    percentile) are kept. Quality is judged in blocks of ``block_s``: the share
    of RR intervals within [RR_MIN, RR_MAX] that differ from the local median
    RR by less than 20%. Only the peaks of blocks with quality >= ``min_quality``
    are returned, so a lead that is flat or detached for part of the night keeps
    its good hours. The second value is the share of the night covered by good
    blocks; below ``min_coverage`` the detection is rejected. This makes the
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
    nb = int(np.ceil(len(x) / (block_s * fs)))
    blk = (pk[1:] / (block_s * fs)).astype(int)
    good = np.zeros(nb, bool)
    for i in range(nb):
        m = blk == i
        # a block needs enough beats (>= 30 bpm on average) and regular intervals
        good[i] = m.sum() >= block_s / 2.0 and ok[m].mean() >= min_quality
    coverage = float(good.mean())
    if coverage < min_coverage:
        return None, coverage
    keep = good[(pk / (block_s * fs)).astype(int)]
    return pk[keep] / fs, coverage


def merge_peaks(
    primary: np.ndarray, secondary: np.ndarray, block_s: float = 300.0
) -> np.ndarray:
    """Peaks of ``primary``, plus those of ``secondary`` in the blocks where ``primary`` has none."""
    have = set((np.asarray(primary) // block_s).astype(int).tolist())
    extra = [t for t in secondary if int(t // block_s) not in have]
    return np.sort(np.concatenate([primary, np.asarray(extra, dtype=float)]))


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
    ``post`` covers the QRS (0.4 s) or also the pulse wave (0.7 s); it is cut so that
    pre + post stays below 0.9 of the median RR interval.

    Segments that overlap at high heart rate are subtracted once per peak; the
    template is tapered (Tukey, 10%) so no step is introduced at its edges.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(peaks_s) > 2:  # the window never reaches the next beat (pre + post below 0.9 of the median RR)
        post = min(post, 0.9 * float(np.median(np.diff(peaks_s))) - pre)
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
