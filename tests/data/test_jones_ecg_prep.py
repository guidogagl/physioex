"""Tests for the Jones et al. (2024) ECG preprocessing step.

Synthetic data only (no network): ``archetypes_path=None`` with the
archetype download monkeypatched to fail, so the median-of-beats
template fallback is exercised.  One ``@pytest.mark.hf`` test downloads
``archetypes.mat`` and runs the full step end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from physioex.data.presets import get_preset
from physioex.data.steps.jones_ecg import (
    JONES_ECG_FS_OUT,
    JonesECGPreprocess,
)

RNG_SEED = 2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthesize_ecg(fs: float, seconds: float = 600.0,
                    hr: float = 70.0, line_hz: float = 60.0) -> np.ndarray:
    """QRS-like Gaussian spikes + baseline wander + line noise + white noise."""
    rng = np.random.default_rng(RNG_SEED)
    n = int(round(seconds * fs))
    t = np.arange(n) / fs

    # R-peaks (slight jitter)
    rr = 60.0 / hr
    r0 = np.arange(0.0, seconds, rr)
    r_t = r0 + rng.uniform(-0.03, 0.03, size=r0.size)
    r_t = r_t[(r_t >= 0) & (r_t < seconds)]
    r_idx = np.round(r_t * fs).astype(int)

    x = np.zeros(n)
    width = 0.015 * fs  # ~15 ms Gaussian QRS
    for r in r_idx:
        d = np.abs(np.arange(n) - r)
        x += 1.0 * np.exp(-0.5 * (d / width) ** 2)
        d = np.abs(np.arange(n) - (r + 0.004 * fs))
        x -= 0.2 * np.exp(-0.5 * (d / (width / 2)) ** 2)  # S-wave-ish
    x += 0.1 * np.sin(2 * np.pi * 0.2 * t)          # baseline wander
    x += 0.1 * np.sin(2 * np.pi * line_hz * t)      # mains
    x += 0.01 * rng.standard_normal(n)              # white noise
    return x


def _band_power_db(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Total power in [lo, hi] Hz, in dB of the signal's total power."""
    N = len(x)
    X = np.abs(np.fft.rfft(x)) ** 2
    X[1:-1] *= 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    m = (freqs >= lo) & (freqs <= hi)
    total = X.sum()
    if total <= 0 or not m.any():
        return -np.inf
    return 10.0 * np.log10(X[m].sum() / total)


def _run_step(x: np.ndarray, fs: float, monkeypatch=None,
              **kwargs) -> np.ndarray:
    """Compile and run JonesECGPreprocess on ``x`` at ``fs``.

    When ``monkeypatch`` is given, the archetype download is forced to
    fail (fallback template is used); otherwise the download registry is
    left as-is.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(
            "physioex.data.steps.jones_ecg._load_archetypes",
            lambda path: None,
        )
    step = JonesECGPreprocess(archetypes_path=None, **kwargs)
    cs = step.compile(fs)
    assert cs.fs_out == JONES_ECG_FS_OUT
    return cs.apply(x)


# ---------------------------------------------------------------------------
# Main synthetic-signal checks across sampling rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fs", [200.0, 250.0, 256.0, 512.0])
def test_synthetic_ecg_full_check(fs, monkeypatch):
    x = _synthesize_ecg(fs)
    y = _run_step(x, fs, monkeypatch)

    # shape / dtype / range
    assert y.dtype == np.float32
    assert abs(len(y) - round(len(x) * JONES_ECG_FS_OUT / fs)) <= 1
    assert np.all(np.abs(y) <= 1.0)

    # R-peak amplitude after scaling: the factor is 2 x P90 of per-beat
    # maxima, so the median of |y| at the R peaks should sit in
    # [0.3, 0.7] (P50 < P90 <= median-ish, and < the max of 1.0).
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(np.abs(y), distance=int(0.3 * JONES_ECG_FS_OUT))
    # keep only peaks near the expected R times
    t = np.arange(len(y)) / JONES_ECG_FS_OUT
    r_t = np.arange(0.0, len(x) / fs, 60.0 / 70.0)
    keep = np.array(
        [min(abs(r - tt) for tt in r_t) < 0.15 for tt in t[peaks]], dtype=bool
    )
    if keep.sum() >= 5:
        med = np.median(np.abs(y[peaks[keep]]) + 0.05)
        assert 0.3 <= med <= 0.7, (
            f"median R amplitude {med:.3f} not in [0.3, 0.7]"
        )

    # 60 Hz power reduced by >= 20 dB
    p60_in = _band_power_db(x, fs, 59.0, 61.0)
    p60_out = _band_power_db(y, JONES_ECG_FS_OUT, 59.0, 61.0)
    assert p60_in - p60_out >= 20.0, f"60 Hz reduced by {p60_in - p60_out:.1f} dB"

    # baseline wander power reduced by >= 20 dB
    bw_in = _band_power_db(x, fs, 0.05, 0.5)
    bw_out = _band_power_db(y, JONES_ECG_FS_OUT, 0.05, 0.5)
    assert bw_in - bw_out >= 20.0, (
        f"baseline wander reduced by {bw_in - bw_out:.1f} dB"
    )


# ---------------------------------------------------------------------------
# Clip handling
# ---------------------------------------------------------------------------


def test_flat_clipped_stretch_becomes_zero(monkeypatch):
    fs = 256.0
    x = _synthesize_ecg(fs, seconds=60.0)
    mx = x.max()
    a, b = int(30 * fs), int(60 * fs)  # a 30 s flat-clipped stretch at the max
    x[a:b] = mx
    y = _run_step(x, fs, monkeypatch)
    # after resampling the stretch maps to roughly [a*256/fs, b*256/fs]
    ya = int(round(a * JONES_ECG_FS_OUT / fs))
    yb = int(round(b * JONES_ECG_FS_OUT / fs))
    stretch = y[ya + 10 : yb - 10]
    assert len(stretch) > 0
    assert np.all(np.abs(stretch) < 0.1), (
        f"clipped stretch not removed (max |y| = {np.abs(stretch).max():.3f})"
    )


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_all_zero_input_returns_zeros(monkeypatch):
    x = np.zeros(1000, dtype=np.float64)
    y = _run_step(x, 250.0, monkeypatch)
    assert y.dtype == np.float32
    assert np.all(y == 0.0)


# ---------------------------------------------------------------------------
# spec / cache hash
# ---------------------------------------------------------------------------


def test_spec_stable_includes_line_freq_excludes_archetypes_path(monkeypatch):
    s1 = JonesECGPreprocess().spec()
    s2 = JonesECGPreprocess(archetypes_path="/tmp/foo.mat").spec()
    assert s1 == s2
    assert "line_freq=60" in s1
    assert "archetypes_path" not in s1
    assert JonesECGPreprocess(line_freq=50.0).spec() != s1
    assert JonesECGPreprocess(find_noise=False).spec() != s1


# ---------------------------------------------------------------------------
# Preset
# ---------------------------------------------------------------------------


def test_preset_jones_ecg_compiles(monkeypatch):
    preset = get_preset("jones_ecg")
    compiled = preset.compile(256.0)
    assert compiled.fs_out == 256.0
    # 30 s epoch -> 7680 samples
    x = _synthesize_ecg(256.0, seconds=30.0)
    y = compiled(x)
    assert abs(len(y) - 7680) <= 1
    assert y.dtype == np.float32


# ---------------------------------------------------------------------------
# Full pipeline with real archetypes (network)
# ---------------------------------------------------------------------------


@pytest.mark.hf
def test_archetypes_download_path():
    import os

    if not os.environ.get("PHYSIOEX_TEST_HF"):
        pytest.skip("PHYSIOEX_TEST_HF not set")
    x = _synthesize_ecg(250.0, seconds=120.0)
    y = _run_step(x, 250.0)  # real download, cached after first run
    assert y.dtype == np.float32
    assert np.all(np.abs(y) <= 1.0)
    assert abs(len(y) - round(len(x) * JONES_ECG_FS_OUT / 250.0)) <= 1
