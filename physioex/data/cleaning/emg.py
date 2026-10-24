"""Clean EMG (chin, legs): bipolar derivation in, muscle only out, at the native sampling rate."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

from physioex.data.cleaning.cardiac import cardiac_residual, subtract_cardiac

# AASM EMG band. The top is capped at 0.45 fs for low-rate recordings.
EMG_BAND = (10.0, 100.0)


def notch_mains(
    x: np.ndarray, fs: float, mains: float, quality: float = 30.0
) -> np.ndarray:
    """Zero-phase notch at the mains frequency and its harmonics below Nyquist."""
    for f0 in np.arange(mains, fs / 2 - 1, mains):
        b, a = iirnotch(f0, quality, fs)
        x = filtfilt(b, a, x)
    return x


def rms_envelope(
    x: np.ndarray, fs: float, win_s: float = 1.0, out_fs: float = 10.0
) -> np.ndarray:
    """Moving RMS over win_s, sampled at out_fs (the envelope scorers read for REM atonia)."""
    w = max(1, int(round(win_s * fs)))
    p = np.convolve(np.asarray(x, dtype=np.float64) ** 2, np.ones(w) / w, mode="same")
    step = fs / out_fs
    idx = np.round(np.arange(0, len(x) / step) * step).astype(int)
    return np.sqrt(np.maximum(p[idx[idx < len(x)]], 0.0))


def _blank(
    x: np.ndarray, fs: float, peaks_s: np.ndarray, half_s: float
) -> Tuple[np.ndarray, float]:
    """Replace +-half_s around each R peak with a linear bridge; returns the blanked share."""
    x = x.copy()
    h = max(1, int(round(half_s * fs)))
    mask = np.zeros(len(x), bool)
    for s in np.round(np.asarray(peaks_s) * fs).astype(int):
        lo, hi = max(0, s - h), min(len(x) - 1, s + h)
        if hi - lo < 2:
            continue
        x[lo : hi + 1] = np.linspace(x[lo], x[hi], hi - lo + 1)
        mask[lo : hi + 1] = True
    return x, float(mask.mean())


def clean_emg(
    x: np.ndarray,
    fs: float,
    mains: float,
    peaks_s: Optional[np.ndarray] = None,
    band: Tuple[float, float] = EMG_BAND,
    blank_s: float = 0.05,
    blank_above: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Muscle-only EMG from a bipolar derivation.

    1. band-pass ``band`` (top capped at 0.45 fs): the high-pass removes delta,
       eye movements, K-complexes and drift;
    2. mains notch with harmonics;
    3. cardiac template subtraction at the native rate (when R peaks are given);
    4. if the heartbeat-locked residual is still above ``blank_above`` times the
       signal RMS, blank +-``blank_s`` around each QRS.

    Returns the clean signal and diagnostics (cardiac residual before and after,
    as a share of the RMS; blanked share).
    """
    x = np.asarray(x, dtype=np.float64)
    hi = min(band[1], 0.45 * fs)
    diag: Dict[str, float] = {"fs": float(fs), "band_hi": float(hi)}
    if hi <= band[0] + 1:
        raise ValueError(f"EMG at {fs} Hz has no band above {band[0]} Hz")
    sos = butter(4, [band[0], hi], btype="band", fs=fs, output="sos")
    y = notch_mains(sosfiltfilt(sos, x), fs, mains)
    if peaks_s is not None and len(peaks_s):
        rms = float(np.sqrt(np.mean(y**2))) or 1.0
        diag["cardiac_before"] = cardiac_residual(y, fs, peaks_s, 0.1, 0.1) / rms
        y = subtract_cardiac(y, fs, peaks_s, pre=0.1, post=0.15)
        diag["cardiac_after_template"] = (
            cardiac_residual(y, fs, peaks_s, 0.1, 0.1) / rms
        )
        if diag["cardiac_after_template"] > blank_above:
            y, diag["blanked_share"] = _blank(y, fs, peaks_s, blank_s)
        diag["cardiac_after"] = cardiac_residual(y, fs, peaks_s, 0.1, 0.1) / rms
    return y, diag
