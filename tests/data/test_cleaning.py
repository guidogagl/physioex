"""Night-level cleaning (physioex.data.cleaning) on semi-synthetic signals with known mixing."""

import numpy as np
import pytest

from physioex.data.cleaning import (
    DEFAULT_BANDS,
    band_regression,
    epoch_mask,
    fft_band,
    sobi,
    split_ocular,
)

FS = 100.0


def _ar(n, a, seed):
    """AR(1) noise: a sets the spectrum, so sources with different a are separable by SOBI"""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _corr(a, b):
    return abs(np.corrcoef(a, b)[0, 1])


def test_fft_bands_sum_back_to_band_union():
    x = _ar(6000, 0.9, 0)
    parts = sum(fft_band(x, lo, hi, FS) for lo, hi in DEFAULT_BANDS)
    np.testing.assert_allclose(parts, fft_band(x, DEFAULT_BANDS[0][0], DEFAULT_BANDS[-1][1], FS), atol=1e-9)


def test_epoch_mask():
    m = epoch_mask([0, 2, 3, 4], [2, 3], n=4 * 3000, fs=FS)
    assert m.sum() == 2 * 3000
    assert not m[:3000].any() and m[3000:9000].all() and not m[9000:].any()


def test_band_regression_removes_known_leak_and_keeps_eyes():
    n = 30000
    brain = {"c3": _ar(n, 0.95, 1), "c4": _ar(n, 0.8, 2)}
    eye = np.zeros(n)
    eye[n // 2:] = _ar(n // 2, 0.99, 3)  # eyes move only in the second half
    eog = {"e": eye + 0.5 * brain["c3"] - 0.2 * brain["c4"]}
    # still eyes: what the EOG shares with the EEG is brain. The fit stops short of the eye onset,
    # where the brick-wall bands smear the step of the eye signal backwards in time.
    fit = np.arange(n) < int(0.4 * n)
    clean, coef = band_regression(eog, brain, fit, FS)
    for lo, _ in DEFAULT_BANDS:
        np.testing.assert_allclose(coef[("e", lo)], [0.5, -0.2], atol=0.01)
    lo, hi = DEFAULT_BANDS[0][0], DEFAULT_BANDS[-1][1]
    target = fft_band(eye, lo, hi, FS)
    assert _corr(clean["e"], target) > 0.999
    assert _corr(clean["e"][: n // 2 - 1000], brain["c3"][: n // 2 - 1000]) < 0.02


def test_band_regression_empty_fit_raises():
    with pytest.raises(ValueError):
        band_regression({"e": np.ones(100)}, {"c": np.ones(100)}, np.zeros(100, bool), FS)


def test_sobi_recovers_sources():
    n = 20000
    S = np.stack([_ar(n, a, 10 + i) for i, a in enumerate((0.99, 0.6, -0.5))])
    A = np.array([[1.0, 0.5, 0.2], [0.3, 1.0, 0.4], [0.6, 0.2, 1.0]])
    U, A_hat = sobi(A @ S, lags=list(range(1, 51, 5)))
    est = U @ (A @ S)
    for s in S:
        assert max(_corr(s, e) for e in est) > 0.98
    np.testing.assert_allclose(A_hat @ U, np.eye(3), atol=1e-8)


def test_split_ocular_reconstructs_input_and_finds_eyes():
    n = 20000
    eye = _ar(n, 0.995, 20)
    b1, b2, b3 = _ar(n, 0.7, 21), _ar(n, -0.3, 22), _ar(n, 0.2, 23)
    eog = {"E1": 3 * eye + 0.4 * b1, "E2": -2.5 * eye + 0.3 * b2}
    eeg = {"F3": 0.3 * eye + b1 + 0.2 * b2, "C3": b2 + 0.3 * b3, "C4": b3 + 0.2 * b1}
    r = split_ocular(eog, eeg)  # 5 channels, 4 sources: rank-deficient on purpose
    assert r.mixing.shape == (5, 4)
    for k, x in eog.items():
        np.testing.assert_allclose(r.eog_ocular[k] + r.eog_cerebral[k], x - x.mean(), atol=1e-8)
    assert r.ocular.sum() >= 1
    assert _corr(r.eog_ocular["E1"], eye) > 0.95
    assert _corr(r.eeg_cerebral["C3"], eye) < 0.1
