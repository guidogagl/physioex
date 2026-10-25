"""NightCleaner, cardiac and EMG cleaning on a semi-synthetic night with known sources."""

import numpy as np
import pytest

from physioex.data.cleaning import CleanMontage
from physioex.data.cleaning.cardiac import (
    cardiac_residual,
    detect_r_peaks,
    peak_aligned_mean,
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
    out = NightCleaner(m).clean(sig, extra if level == "rich" else {}, level=level)
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


def test_states_follow_the_night(night):
    """Label-free masks find the stages they stand for (stages used only to check)."""
    sig, extra, _ = night
    cl = NightCleaner(CleanMontage(eeg=["x"], eog=["x"]))
    eeg, eog = (cl._prep(*sig[k], None) for k in ("eeg", "eog"))
    st = cl.states(eeg, eog)
    n = len(st["quiet"])
    stage = lambda s: epoch_mask(HYP, s, n, 100.0)  # noqa: E731
    assert not (st["quiet"] & stage([0, 4])).any()  # still eyes: never W/REM
    assert (st["active"] & stage([0, 4])).sum() == st["active"].sum()
    assert (st["deep"] & stage([3])).sum() > 0.8 * st["deep"].sum()


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
        {"eeg": (np.zeros(6000), 100.0)}, {}
    )
    assert out.level == "missing" and not out.outputs


def test_window_crops_and_realigns(night):
    sig, extra, _ = night
    m = CleanMontage(eeg=["x"], eog=["x"])
    full = NightCleaner(m).clean(sig, {}, level="poor")
    win = NightCleaner(m).clean(sig, {}, level="poor", window=(240.0, 960.0))
    for name, (y, fs) in win.outputs.items():
        assert len(y) == len(full.outputs[name][0])  # still aligned with the recording
        a, b = int(240 * fs), int(960 * fs)
        assert not y[:a].any() and not y[b + 1 :].any() and y[a:b].any()
    assert win.diagnostics["window_s"] == [240.0, 960.0]


@pytest.mark.parametrize("gain", [1.2, np.inf])
def test_regression_guard_dead_eeg(night, gain):
    """A dead EEG electrode (noise and heartbeat only) must not inflate the EOG or the EEG through steps 4-5."""
    sig, _, _ = night
    rng = np.random.default_rng(5)
    sig = dict(sig, eeg=(0.3 * rng.standard_normal(len(sig["eeg"][0])), FS))
    cl = NightCleaner(CleanMontage(eeg=["x"], eog=["x"], level="poor"), max_regression_gain=gain)
    out = cl.clean(sig, {}, level="poor")
    p = lambda x: float(np.mean(x**2))  # noqa: E731
    ratio = {k: p(out.outputs[f"{k.upper()}_clean"][0]) / p(cl._prep(*sig[k], None)) for k in ("eog", "eeg")}
    if gain == 1.2:
        assert ratio["eog"] < 1.2 and ratio["eeg"] < 1.2, (ratio, out.diagnostics)
    else:
        print("no guard:", ratio)


def test_muscle_wideband_removes_injected_muscle(night):
    """Muscle leaked into 6 EEG channels (broadband, bursts that follow the chin) goes out of 16-35 Hz; delta stays."""
    sig, _, src = night
    rng = np.random.default_rng(5)
    n = len(src["delta"])
    t = np.arange(n) / FS
    alpha = np.sin(2 * np.pi * 10 * t) * _gate([0], n, gain=1.0, base=0.2)
    spindle = np.sin(2 * np.pi * 13 * t) * _gate([2], n, gain=0.5, base=0.0)
    burst = fft_band(src["muscle"], 20.0, 90.0, FS)  # same envelope as the chin muscle, its own noise
    chans = {}
    for k in ("eeg", "C3", "F3", "F4", "O1", "P3"):
        w = rng.uniform(0.3, 1.0, 4)
        brain = 20 * (w[0] * src["delta"] + w[1] * alpha + w[2] * spindle + 0.1 * w[3] * src["eye"])
        chans[k] = brain + 8 * rng.uniform(0.5, 1.0) * burst + rng.standard_normal(n) + 3 * src["qrs"]
    sig = {**sig, "eeg": (chans.pop("eeg"), FS)}
    extra = {k: (x, FS) for k, x in chans.items()}
    m = CleanMontage(eeg=["x"], eog=["x"], extra_eeg=list(extra), level="rich")
    outs = {mode: NightCleaner(m, muscle_mode=mode).clean(sig, extra, level="rich") for mode in ("off", "wideband")}
    fs = 100.0
    n = len(outs["off"].outputs["EEG_clean"][0])
    mus = fft_band(resample(burst, FS, fs)[:n], 20.0, 35.0, fs)
    delta = fft_band(resample(src["delta"], FS, fs)[:n], 0.3, 4, fs)
    every = np.ones(n, bool)
    hi = {k: fft_band(o.outputs["EEG_clean"][0][:n], 20.0, 35.0, fs) for k, o in outs.items()}
    d = outs["wideband"].diagnostics["muscle"]
    assert d["n_removed"] >= 1, str(d)
    assert np.var(hi["wideband"]) < 0.5 * np.var(hi["off"])
    assert abs(_corr(hi["wideband"], mus, every)) < 0.5 * abs(_corr(hi["off"], mus, every))
    lo = lambda o: fft_band(o.outputs["EEG_clean"][0][:n], 0.3, 4, fs)  # noqa: E731
    n3 = epoch_mask(HYP, [3], n, fs)
    assert _corr(lo(outs["wideband"]), delta, n3) > 0.9
    diff = lo(outs["wideband"]) - lo(outs["off"])  # only 16-35 Hz is subtracted: the slow band moves by resampling leakage only
    assert np.sqrt(np.mean(diff**2)) < 0.01 * np.sqrt(np.mean(lo(outs["off"]) ** 2))


def test_cardiac_window_covers_pulse_wave():
    """A pulse wave 0.5 s after the R peak stays with the 0.4 s window and goes with 0.7 s; the window stops before the next beat."""
    rng = np.random.default_rng(3)
    n = int(600 * FS)
    t = np.arange(n) / FS
    beats = np.cumsum(0.95 + 0.03 * rng.standard_normal(700))
    beats = beats[beats < t[-1] - 1]
    x = np.zeros(n)
    for b in beats:
        x += 10 * np.exp(-0.5 * ((t - b) / 0.012) ** 2) + 4 * np.exp(-0.5 * ((t - b - 0.5) / 0.08) ** 2)
    x += 0.5 * rng.standard_normal(n)
    pulse = lambda y: np.ptp(peak_aligned_mean(y, FS, beats, 0.2, 0.8)[int(0.5 * FS) : int(0.8 * FS)])  # noqa: E731
    short, long = subtract_cardiac(x, FS, beats, post=0.4), subtract_cardiac(x, FS, beats, post=0.7)
    assert pulse(long) < 0.2 * pulse(short)
    # at 120 bpm the window is cut below the RR interval: no beat is subtracted twice
    fast = np.arange(1.0, 590.0, 0.5)
    y = np.zeros(n)
    for b in fast:
        y += 10 * np.exp(-0.5 * ((t - b) / 0.012) ** 2)
    assert cardiac_residual(subtract_cardiac(y, FS, fast, post=0.7), FS, fast) < 0.05 * cardiac_residual(y, FS, fast)
