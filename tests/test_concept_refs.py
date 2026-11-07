"""Reference detectors of the lexicon concepts on synthetic signals (07.11.2026): each finds what it
must and rejects the contrary."""
import numpy as np

from physioex.explain import concept_refs as R

FS = 100.0


def _bg(sec, amp=5.0, seed=0):
    return np.random.default_rng(seed).normal(0, amp, int(sec * FS))


def _pink(sec, rms=15.0, seed=0):
    n = int(sec * FS)
    w = np.fft.rfft(np.random.default_rng(seed).normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / FS)
    w[1:] /= np.sqrt(f[1:])
    w[0] = 0
    y = np.fft.irfft(w, n)
    return y / y.std() * rms


def _covers(iv, a_s, b_s):
    return any(a <= b_s * FS and b >= a_s * FS for a, b in iv)


def test_spindle_detected_and_not_in_noise():
    x = _pink(30)
    t = np.arange(int(1.0 * FS)) / FS
    x[1000:1100] += 30 * np.sin(2 * np.pi * 13 * t) * np.hanning(100)
    assert _covers(R.detect_spindles(x, FS), 10.2, 10.8)
    # 1/f background without spindles: rare detections (YASA-like criteria are relative)
    assert sum(len(R.detect_spindles(_pink(30, seed=s), FS)) for s in range(10)) <= 10


def test_slow_wave_amplitude_rule():
    x = _bg(30, amp=2)
    t = np.arange(int(1.0 * FS)) / FS
    big = -60 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, 120 uV peak to peak
    x[1000:1100] += big
    assert _covers(R.detect_slow_waves(x, FS), 10.2, 10.8)
    y = _bg(30, amp=2)
    y[1000:1100] += big / 4  # 30 uV peak to peak: not a slow wave
    assert not R.detect_slow_waves(y, FS)


def test_eye_movement_speed():
    x = _bg(30, amp=1)
    fast = np.concatenate([np.linspace(0, 80, 15), np.linspace(80, 0, 60)])  # 80 uV in 0.15 s
    x[500:575] += fast
    slow = 60 * np.sin(np.pi * np.arange(150) / 150)  # 60 uV in 0.75 s, 1.5 s half-wave
    x[2000:2150] += slow
    d = R.detect_eye_movements(x, FS)
    assert _covers(d["rem_eye"], 5.0, 5.7)
    assert not _covers(d["rem_eye"], 20.2, 21.3)
    assert _covers(d["sem"], 20.2, 21.3)


def test_emg_atonia_and_tma():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 10, int(60 * FS))
    x[3000:4500] *= 0.05  # 15 s of atonia
    x[3500:3515] += rng.normal(0, 200, 15)  # 0.15 s burst on the low tone
    at, tma, _, _ = R.emg_masks(x, FS)
    assert at[62:88].mean() > 0.9 and at[:50].mean() < 0.1
    assert R.samples_to_tokens(tma, len(x), FS)[70]


def test_events_to_tokens_on_dumped_epochs():
    ts = np.array([10, 11, 13])  # epoch 12 not dumped
    m = R.events_to_tokens([(10 * 30 + 1.0, 10 * 30 + 2.0), (12 * 30 + 5, 12 * 30 + 6)], ts)
    assert m.reshape(3, R.T)[0, 2:4].all() and m.sum() == 2


def test_night_refs_shapes():
    rng = np.random.default_rng(4)
    ts = np.array([0, 1, 2, 5, 6])
    night = {"ts": ts, "x_eeg": rng.normal(0, 10, (5, 3000)), "x_eog": rng.normal(0, 10, (5, 3000)),
             "x_emg": rng.normal(0, 10, (5, 3000))}
    meta = {"fs": {"eeg": 100.0, "eog": 100.0, "emg": 100.0}, "scale": {m: [0.0, 1.0] for m in ("eeg", "eog", "emg")},
            "events": [{"type": "arousal", "concept": "ASDA arousal", "onset": 31.0, "dur": 5.0}]}
    r = R.night_refs(night, meta)
    for k, v in r.items():
        assert v.dtype == bool and len(v) in (5, 5 * R.T), k
    assert r["arousal_ann"].reshape(5, R.T)[1, 2:12].all()


def test_vertex_sharp_vs_slow_and_small():
    x = _pink(30, rms=5)
    t = np.arange(20) / FS
    sharp = np.concatenate([-80 * np.sin(np.pi * t / 0.2), 50 * np.sin(np.pi * np.arange(20) / 20)])  # 0.2 s down, rebound
    x[1000:1040] += sharp
    assert _covers(R.detect_vertex(x, FS), 10.0, 10.2)
    y = _pink(30, rms=5)
    y[1000:1040] += sharp / 8  # 10 uV: not a vertex
    assert not _covers(R.detect_vertex(y, FS), 10.0, 10.4)


def test_sawtooth_train_vs_background():
    x = _pink(30, rms=5)
    t = np.arange(300) / FS
    x[1000:1300] += 20 * np.sign(np.sin(2 * np.pi * 3 * t)) * np.abs(np.sin(2 * np.pi * 3 * t)) ** 0.3  # 3 s at 3 Hz, 40 uV ptp
    assert _covers(R.detect_sawtooth(x, FS), 10.5, 12.5)
    assert not R.detect_sawtooth(_pink(30, rms=5, seed=1), FS)


def test_night_refs_without_scale_skips_amplitude_rules():
    rng = np.random.default_rng(5)
    night = {"ts": np.arange(4), "x_eeg": rng.normal(0, 1.5, (4, 3000)), "x_eog": rng.normal(0, 1.5, (4, 3000))}
    r = R.night_refs(night, {"fs": {"eeg": 100.0, "eog": 100.0}})
    assert "spindle_rule" in r and "alpha" in r
    assert not {"slow_wave", "delta", "vertex", "sawtooth", "artifact_eeg", "rem_eye", "sem"} & set(r)


def test_eye_movement_rejected_when_also_on_eeg():
    x = _bg(30, amp=1)
    fast = np.concatenate([np.linspace(0, 80, 15), np.linspace(80, 0, 60)])
    x[500:575] += fast
    eeg = _bg(30, amp=1, seed=1)
    assert _covers(R.detect_eye_movements(x, FS, ref=eeg)["rem_eye"], 5.0, 5.7)
    eeg[500:575] += fast * 0.8  # frontal delta: same wave on EEG and EOG
    assert not _covers(R.detect_eye_movements(x, FS, ref=eeg)["rem_eye"], 5.0, 5.7)


def test_stage_gates():
    rng = np.random.default_rng(6)
    night = {"ts": np.arange(4), "y": np.array([0, 2, 4, 2]), "x_eeg": rng.normal(0, 1.5, (4, 3000))}
    r = R.night_refs(night, {"fs": {"eeg": 100.0}, "scale": {"eeg": [0.0, 10.0]}})
    assert not r["sawtooth"].reshape(4, R.T)[[0, 1, 3]].any()
    assert not r["spindle_rule"].reshape(4, R.T)[[0, 2]].any()


def test_k_complex_isolated_vs_train():
    x = _pink(30, rms=5)
    t = np.arange(int(0.8 * FS)) / FS
    kc = -60 * np.sin(2 * np.pi * t / 0.8)  # 0.4 s negative, 0.4 s positive: 0.8 s biphasic
    x[1000:1080] += kc
    assert _covers(R.detect_k_complex(x, FS), 10.0, 10.8)
    y = _pink(30, rms=5)
    for k in range(6):  # a train of slow waves is not a K-complex
        y[1000 + 80 * k:1080 + 80 * k] += kc
    assert not _covers(R.detect_k_complex(y, FS), 11.6, 13.0)


def test_cap_a1_and_a3():
    x = _pink(120, rms=5)
    t = np.arange(int(6 * FS)) / FS
    x[6000:6600] += 40 * np.sin(2 * np.pi * 1.5 * t)  # 6 s delta burst
    x[9000:9600] += 20 * np.sin(2 * np.pi * 20 * t)  # 6 s fast burst
    d = R.detect_cap_a(x, FS)
    assert _covers(d["a1"], 61, 65) and not _covers(d["a1"], 91, 95)
    assert _covers(d["a3"], 91, 95)


def test_rmma_series_vs_single_burst():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 5, int(60 * FS))
    for k in range(4):  # 4 bursts of 0.5 s, 1 s apart
        a = 2000 + 100 * k
        x[a:a + 50] += rng.normal(0, 60, 50)
    x[4500:4550] += rng.normal(0, 60, 50)  # isolated burst
    rm = R.emg_masks(x, FS)[3]
    assert _covers(rm, 20.0, 23.5) and not _covers(rm, 45.0, 45.5)


def test_emg_rise_needs_one_second():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 5, int(60 * FS))
    x[3000:3300] = rng.normal(0, 30, 300)  # 3 s rise
    x[4500:4560] = rng.normal(0, 30, 60)  # 0.6 s
    rise = R.emg_masks(x, FS)[2]
    assert rise[60:66].any() and not rise[89:93].any()
