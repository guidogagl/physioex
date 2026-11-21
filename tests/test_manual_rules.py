"""Synthetic tests of the manual-derived rule detectors (physioex.explain.manual_rules)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from physioex.explain import manual_rules as M  # noqa: E402
from physioex.explain.concept_refs import T  # noqa: E402

FS = 128


def pink(n, uv=20.0, seed=0):
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std() * uv


def burst(x, start_s, dur_s, hz, uv):
    x = x.copy()
    a, b = int(start_s * FS), int((start_s + dur_s) * FS)
    t = np.arange(b - a) / FS
    x[a:b] += uv * np.sin(2 * np.pi * hz * t)
    return x


def overlaps(iv, a_s, b_s):
    return any(a < b_s * FS and b > a_s * FS for a, b in iv)


@pytest.fixture
def bg():
    return pink(40 * FS)


def test_arousal_alpha_burst(bg):
    r = M.detect_arousal(burst(bg, 20, 5, 10, 60), FS)
    assert len(r["arousal"]) == 1 and overlaps(r["arousal"], 20, 25)
    assert not r["awakening"]


def test_arousal_too_short(bg):
    r = M.detect_arousal(burst(bg, 20, 2, 10, 60), FS)
    assert not r["arousal"] and not r["awakening"]


def test_awakening_long():
    x = burst(pink(60 * FS), 20, 20, 10, 60)
    r = M.detect_arousal(x, FS)
    assert not r["arousal"] and len(r["awakening"]) == 1


def test_arousal_needs_10s_sleep(bg):
    r = M.detect_arousal(burst(bg, 5, 5, 10, 60), FS)
    assert not r["arousal"]


def test_arousal_not_spindle_band(bg):
    r = M.detect_arousal(burst(bg, 20, 5, 14, 60), FS)
    assert not r["arousal"] and not r["awakening"]


def test_arousal_in_rem_needs_emg(bg):
    x = burst(bg, 20, 5, 10, 60)
    st = np.full(len(x) // (FS // 2), 4)
    flat = np.random.default_rng(1).normal(0, 2, len(x))
    assert not M.detect_arousal(x, FS, flat, st)["arousal"]
    rise = flat.copy()
    rise[21 * FS:23 * FS] *= 4
    assert len(M.detect_arousal(x, FS, rise, st)["arousal"]) == 1
    assert not M.detect_arousal(x, FS, None, st)["arousal"]


def test_arousal_dropped_in_wake(bg):
    x = burst(bg, 20, 5, 10, 60)
    st = np.full(len(x) // (FS // 2), 2)
    st[30:] = 0  # W from 15 s
    assert not M.detect_arousal(x, FS, None, st)["arousal"]


def test_alpha_intrusion_train():
    x = pink(40 * FS, 10)
    for s in (10, 14, 18):
        x = burst(x, s, 1, 10, 40)
    st = np.full(len(x) // (FS // 2), 2)
    r = M.detect_alpha_intrusion(x, FS, st, [])
    assert r["alpha_intrusion"] and len(r["alpha_burst"]) >= 3


def test_alpha_isolated_burst():
    x = burst(pink(40 * FS, 10), 20, 1, 10, 40)
    st = np.full(len(x) // (FS // 2), 2)
    r = M.detect_alpha_intrusion(x, FS, st, [])
    assert r["alpha_burst"] and not r["alpha_intrusion"]


def test_alpha_burst_inside_arousal_dropped():
    x = burst(pink(40 * FS, 10), 20, 1, 10, 40)
    st = np.full(len(x) // (FS // 2), 2)
    assert not M.detect_alpha_intrusion(x, FS, st, [(19 * FS, 23 * FS)])["alpha_burst"]


def test_kc_context():
    kc = [(10 * FS, 11 * FS)]
    assert M.detect_kc_context(kc, [(int(11.5 * FS), 15 * FS)], FS)["kc_arousal"] == kc
    assert M.detect_kc_context(kc, [(14 * FS, 18 * FS)], FS)["kc_arousal"] == []


def test_alpha_delta_epochs():
    n = 2 * 30 * FS
    t = np.arange(n) / FS
    x = 80 * np.sin(2 * np.pi * 1 * t) + 25 * np.sin(2 * np.pi * 10 * t)
    ad = M.alpha_delta_epochs(x, FS, np.array([3, 2]))
    assert ad.tolist() == [True, False]


def test_night_m1_keys():
    rng = np.random.default_rng(0)
    n = 4
    night = {"ts": np.arange(n), "y": np.array([1, 2, 2, 4]),
             **{f"x_{m}": rng.normal(0, 1, (n, 30 * FS)).astype(np.float32) for m in ("eeg", "eog", "emg")}}
    meta = {"fs": {"eeg": float(FS), "eog": float(FS), "emg": float(FS)},
            "scale": {"eeg": [0.0, 20.0], "eog": [0.0, 20.0]}, "events": []}
    out = M.night_m1(night, meta)
    for k in M.CONCEPTS_M1:
        assert k in out, k
        assert len(out[k]) == (n if k.endswith("_epoch") else n * T), k
        assert out[k].dtype == bool


def test_night_m1_vertex_gating():
    """vertex_m1 only in N1 and in N2 epochs next to N1."""
    rng = np.random.default_rng(0)
    n = 5
    night = {"ts": np.arange(n), "y": np.array([2, 2, 1, 2, 2]),
             **{f"x_{m}": rng.normal(0, 1, (n, 30 * FS)).astype(np.float32) for m in ("eeg", "eog", "emg")}}
    meta = {"fs": {"eeg": float(FS), "eog": float(FS), "emg": float(FS)},
            "scale": {"eeg": [0.0, 20.0], "eog": [0.0, 20.0]}, "events": []}
    v = M.night_m1(night, meta)["vertex_m1"].reshape(n, T)
    assert not v[0].any() and not v[4].any()
