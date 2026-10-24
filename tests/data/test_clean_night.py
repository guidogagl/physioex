"""NightCleaner, cardiac and EMG cleaning on a semi-synthetic night with known sources."""

import numpy as np
import pytest

from physioex.data.cleaning import CleanMontage
from physioex.data.cleaning.cardiac import (
    cardiac_residual,
    detect_r_peaks,
    subtract_cardiac,
)
from physioex.data.cleaning.emg import clean_emg, rms_envelope
from physioex.data.cleaning.night import NightCleaner, resample
from physioex.data.cleaning.regression import epoch_mask, fft_band

FS = 200.0
HYP = np.array([0] * 8 + [2] * 8 + [3] * 8 + [4] * 8 + [2] * 4)  # W, N2, N3, REM, N2


def _ar(n, a, rng):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x / x.std()


def _gate(stages, n, fs=FS, gain=1.0, base=0.0):
    g = np.full(n, base)
    g[epoch_mask(HYP, stages, n, fs)] = gain
    return g


@pytest.fixture(scope="module")
def night():
    rng = np.random.default_rng(0)
    n = int(len(HYP) * 30 * FS)
    t = np.arange(n) / FS
    beats = np.cumsum(0.9 + 0.05 * rng.standard_normal(int(n / FS / 0.9) + 10))
    beats = beats[beats < t[-1] - 1]
    qrs = np.zeros(n)
    for b in beats:
        qrs += np.exp(-0.5 * ((t - b) / 0.012) ** 2) - 0.3 * np.exp(
            -0.5 * ((t - b - 0.25) / 0.06) ** 2
        )
    delta = _ar(n, 0.995, rng) * _gate([3], n, gain=3.0, base=1.0)
    alpha = np.sin(2 * np.pi * 10 * t) * _gate([0], n, gain=1.0, base=0.2) + 0.3 * _ar(
        n, 0.9, rng
    )
    eye = _ar(n, 0.999, rng) * _gate([0, 4], n, gain=4.0, base=0.0)
    muscle = (
        rng.standard_normal(n)
        * _gate([0], n, gain=3.0, base=1.0)
        * _gate([4], n, gain=0.3, base=1.0)
    )
    noise = lambda: 0.2 * rng.standard_normal(n)  # noqa: E731
    sig = {
        "eeg": (20 * (delta + 0.6 * alpha + 0.15 * eye) + 3 * qrs + noise(), FS),
        "eog": (20 * (eye + 0.4 * delta) + 5 * qrs + noise(), FS),
        "chin": (5 * muscle + 8 * qrs + noise(), FS),
        "ecg": (500 * qrs + noise(), FS),
    }
    extra = {
        "C3": (
            20 * (0.9 * delta + 0.5 * alpha + 0.1 * eye + 0.3 * _ar(n, 0.5, rng))
            + 3 * qrs,
            FS,
        ),
        "F3": (
            20 * (delta + 0.3 * alpha + 0.4 * eye + 0.3 * _ar(n, 0.7, rng)) + 3 * qrs,
            FS,
        ),
        "O1": (20 * (0.4 * delta + alpha + 0.3 * _ar(n, 0.3, rng)) + 3 * qrs, FS),
    }
    return sig, extra, dict(delta=delta, eye=eye, beats=beats, qrs=qrs, muscle=muscle)


def _corr(a, b, mask):
    return np.corrcoef(a[mask], b[mask])[0, 1]


def test_r_peaks_and_template(night):
    sig, _, src = night
    peaks, q = detect_r_peaks(*sig["ecg"])
    assert q > 0.95 and abs(len(peaks) - len(src["beats"])) <= 2
    x = 3 * src["qrs"] + 0.2 * np.random.default_rng(2).standard_normal(len(src["qrs"]))
    y = subtract_cardiac(x, FS, peaks)
    assert cardiac_residual(y, FS, peaks) < 0.1 * cardiac_residual(x, FS, peaks)
    # no heartbeat in a noise channel: detection refuses
    assert (
        detect_r_peaks(np.random.default_rng(1).standard_normal(len(x)), FS)[0] is None
    )


def test_clean_emg(night):
    sig, _, src = night
    peaks, _ = detect_r_peaks(*sig["ecg"])
    y, d = clean_emg(*sig["chin"], mains=50.0, peaks_s=peaks)
    assert d["cardiac_after"] < 0.1 < d["cardiac_before"]
    env = rms_envelope(y, FS, 1.0, 1.0)
    n = len(env)
    w, n2, rem = (epoch_mask(HYP, [s], n, 1.0) for s in (0, 2, 4))
    assert env[rem].mean() < env[n2].mean() < env[w].mean()


@pytest.mark.parametrize("level", ["rich", "poor"])
def test_night_cleaner(night, level):
    sig, extra, src = night
    m = CleanMontage(eeg=["x"], eog=["x"], extra_eeg=["C3", "F3", "O1"], level=level)
    out = NightCleaner(m).clean(sig, extra if level == "rich" else {}, HYP, level=level)
    assert out.level == level
    assert set(out.outputs) >= {"EEG_clean", "EOG_clean", "EMG_clean", "EMG_ENV_clean"}
    fs = 100.0
    eog, eeg = out.outputs["EOG_clean"][0], out.outputs["EEG_clean"][0]
    n = len(eog)
    delta = fft_band(resample(src["delta"], FS, fs)[:n], 0.3, 4, fs)
    eye = fft_band(resample(src["eye"], FS, fs)[:n], 0.3, 4, fs)
    n3, rem = epoch_mask(HYP, [3], n, fs), epoch_mask(HYP, [4], n, fs)
    eog_lo, eeg_lo = fft_band(eog, 0.3, 4, fs), fft_band(eeg, 0.3, 4, fs)
    assert abs(_corr(eog_lo, delta, n3)) < 0.15  # brain out of the EOG
    assert _corr(eog_lo, eye, rem) > 0.9  # eyes kept
    assert abs(_corr(eeg_lo, eye, rem)) < 0.15  # eyes out of the EEG
    assert _corr(eeg_lo, delta, n3) > 0.9  # delta kept in the EEG
    assert out.diagnostics["cardiac"]["source"] == "ecg"
    assert out.diagnostics["cardiac"]["residual_uV"]["eeg"] < 1.0
    if level == "rich":
        assert out.diagnostics["sobi"]["n_ocular"] >= 1


def test_spec_changes_with_parameters():
    m = CleanMontage(eeg=["x"], eog=["y"])
    assert NightCleaner(m).spec() == NightCleaner(m).spec()
    assert NightCleaner(m).spec() != NightCleaner(m, activity_ratio=2.0).spec()
    assert (
        NightCleaner(m).spec()
        != NightCleaner(CleanMontage(eeg=["x"], eog=["y"], mains=60.0)).spec()
    )


def test_missing_eog():
    out = NightCleaner(CleanMontage(eeg=["x"], eog=["y"])).clean(
        {"eeg": (np.zeros(6000), 100.0)}, {}, [2, 2]
    )
    assert out.level == "missing" and not out.outputs
