"""
Unit tests for the preset preprocessing pipelines.

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_presets.py
"""

import sys

from physioex.data.pipeline import PreprocessingPipeline
from physioex.data.steps import (
    Resample, BandpassFilter, HighPassFilter, NotchFilter,
    XSleepNetSpectrogram, Identity,
)
from physioex.data.presets import get_preset, available_presets, PRESETS

passed, failed = 0, 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


# ---------------------------------------------------------------------------
# Test 1: raw preset returns pipeline with 2 steps (Bandpass then Resample).
# Order matches legacy preprocessing (filter at native fs, then resample).
# ---------------------------------------------------------------------------
def test_raw_preset_steps():
    try:
        p = get_preset("raw")
        assert isinstance(p, PreprocessingPipeline), f"Expected PreprocessingPipeline, got {type(p)}"
        assert len(p.steps) == 2, f"Expected 2 steps, got {len(p.steps)}"
        assert isinstance(p.steps[0], BandpassFilter), f"Step 0: expected BandpassFilter, got {type(p.steps[0])}"
        assert isinstance(p.steps[1], Resample), f"Step 1: expected Resample, got {type(p.steps[1])}"
        report("1. raw preset has BandpassFilter + Resample", True)
    except Exception as exc:
        report("1. raw preset has BandpassFilter + Resample", False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: seqsleepnet preset has 3 steps ending in spectrogram.
# Order: BandpassFilter -> Resample -> XSleepNetSpectrogram.
# ---------------------------------------------------------------------------
def test_seqsleepnet_preset_steps():
    try:
        p = get_preset("seqsleepnet")
        assert isinstance(p, PreprocessingPipeline), f"Expected PreprocessingPipeline, got {type(p)}"
        assert len(p.steps) == 3, f"Expected 3 steps, got {len(p.steps)}"
        assert isinstance(p.steps[0], BandpassFilter), f"Step 0: expected BandpassFilter, got {type(p.steps[0])}"
        assert isinstance(p.steps[1], Resample), f"Step 1: expected Resample, got {type(p.steps[1])}"
        assert isinstance(p.steps[2], XSleepNetSpectrogram), (
            f"Step 2: expected XSleepNetSpectrogram, got {type(p.steps[2])}"
        )
        report("2. seqsleepnet preset has 3 steps ending in XSleepNetSpectrogram", True)
    except Exception as exc:
        report("2. seqsleepnet preset has 3 steps ending in XSleepNetSpectrogram", False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: xsleepnet_mouse preset has noverlap=188
# ---------------------------------------------------------------------------
def test_xsleepnet_mouse_noverlap():
    try:
        p = get_preset("xsleepnet_mouse")
        assert isinstance(p, PreprocessingPipeline), f"Expected PreprocessingPipeline, got {type(p)}"
        assert len(p.steps) == 3, f"Expected 3 steps, got {len(p.steps)}"
        spec_step = p.steps[2]
        assert isinstance(spec_step, XSleepNetSpectrogram), (
            f"Step 2: expected XSleepNetSpectrogram, got {type(spec_step)}"
        )
        assert spec_step.noverlap == 188, f"Expected noverlap=188, got {spec_step.noverlap}"
        report("3. xsleepnet_mouse preset has noverlap=188", True)
    except Exception as exc:
        report("3. xsleepnet_mouse preset has noverlap=188", False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: identity preset is a single Identity step
# ---------------------------------------------------------------------------
def test_identity_preset():
    try:
        p = get_preset("identity")
        assert isinstance(p, PreprocessingPipeline), f"Expected PreprocessingPipeline, got {type(p)}"
        assert len(p.steps) == 1, f"Expected 1 step, got {len(p.steps)}"
        assert isinstance(p.steps[0], Identity), f"Step 0: expected Identity, got {type(p.steps[0])}"
        report("4. identity preset is a single Identity step", True)
    except Exception as exc:
        report("4. identity preset is a single Identity step", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: unknown preset raises ValueError with available names
# ---------------------------------------------------------------------------
def test_unknown_preset_raises():
    try:
        raised = False
        msg = ""
        try:
            get_preset("unknown")
        except ValueError as exc:
            raised = True
            msg = str(exc)
        assert raised, "Expected ValueError for unknown preset"
        assert "unknown" in msg.lower() or "'unknown'" in msg, (
            f"Error message should mention 'unknown': {msg}"
        )
        # Check that available preset names are listed
        for name in ("raw", "seqsleepnet", "identity"):
            assert name in msg, f"Error message should list available preset '{name}': {msg}"
        report("5. unknown preset raises ValueError with available names", True)
    except Exception as exc:
        report("5. unknown preset raises ValueError with available names", False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: same preset called twice has identical hash (determinism)
# ---------------------------------------------------------------------------
def test_hash_determinism():
    try:
        p1 = get_preset("raw")
        p2 = get_preset("raw")
        assert p1.hash() == p2.hash(), f"Hashes differ: {p1.hash()} != {p2.hash()}"

        p3 = get_preset("seqsleepnet")
        p4 = get_preset("seqsleepnet")
        assert p3.hash() == p4.hash(), f"Hashes differ: {p3.hash()} != {p4.hash()}"
        report("6. same preset called twice has identical hash", True)
    except Exception as exc:
        report("6. same preset called twice has identical hash", False, str(exc))


# ---------------------------------------------------------------------------
# Test 7: semantically-distinct presets have different hashes.
# Note: 'raw', 'eeg', 'eog' are intentionally equal (same bandpass 0.3-40 +
# resample 100) so they SHOULD share a hash -> the cache reuses the entry.
# Only check distinctness between presets with genuinely different steps.
# ---------------------------------------------------------------------------
def test_hash_uniqueness():
    try:
        # These presets have genuinely different step specs and must differ:
        distinct_names = [
            "raw", "seqsleepnet", "xsleepnet_mouse", "identity",
            "emg", "ecg",
        ]
        hashes = {n: get_preset(n).hash() for n in distinct_names}
        assert len(set(hashes.values())) == len(hashes), (
            f"Hash collision among distinct presets: {hashes}"
        )
        # eeg & eog use the same bandpass 0.3-40 as raw -> same hash (cache reuse).
        eeg_h = get_preset("eeg").hash()
        eog_h = get_preset("eog").hash()
        raw_h = get_preset("raw").hash()
        assert eeg_h == eog_h == raw_h, (
            f"eeg/eog/raw should share hash for cache reuse: "
            f"eeg={eeg_h} eog={eog_h} raw={raw_h}"
        )
        # emg now uses HighPassFilter(10) -> MUST differ from eeg/raw
        emg_h = get_preset("emg").hash()
        assert emg_h != eeg_h, "emg (HP >10Hz) must differ from eeg (BP 0.3-40)"
        report("7. distinct presets have distinct hashes; equivalent presets share (cache reuse)", True)
    except Exception as exc:
        report("7. distinct presets have distinct hashes; equivalent presets share (cache reuse)", False, str(exc))


# ---------------------------------------------------------------------------
# Test 8: overrides work (target_fs=256 on raw). Resample is now step 1.
# ---------------------------------------------------------------------------
def test_overrides():
    try:
        p = get_preset("raw", target_fs=256.0)
        assert isinstance(p, PreprocessingPipeline), f"Expected PreprocessingPipeline, got {type(p)}"
        assert len(p.steps) == 2, f"Expected 2 steps, got {len(p.steps)}"
        resample_step = p.steps[1]  # Resample is now at index 1
        assert isinstance(resample_step, Resample), f"Step 1: expected Resample, got {type(resample_step)}"
        assert resample_step.target_fs == 256.0, (
            f"Expected target_fs=256.0, got {resample_step.target_fs}"
        )
        # Verify it differs from default
        p_default = get_preset("raw")
        assert p_default.steps[1].target_fs == 100.0, (
            f"Default target_fs should be 100.0, got {p_default.steps[1].target_fs}"
        )
        assert p.hash() != p_default.hash(), "Override should produce different hash"
        report("8. overrides work: get_preset('raw', target_fs=256)", True)
    except Exception as exc:
        report("8. overrides work: get_preset('raw', target_fs=256)", False, str(exc))


# ---------------------------------------------------------------------------
# Test 9: available_presets() returns sorted list with expected entries
# ---------------------------------------------------------------------------
def test_available_presets():
    try:
        ap = available_presets()
        assert isinstance(ap, list), f"Expected list, got {type(ap)}"
        assert ap == sorted(ap), f"List should be sorted: {ap}"
        for name in ("raw", "seqsleepnet", "xsleepnet_mouse", "identity",
                      "eeg", "eog", "emg", "ecg",
                      "time_domain", "time_frequency"):
            assert name in ap, f"Expected '{name}' in available_presets(), got {ap}"
        report("9. available_presets() returns sorted list with expected entries", True)
    except Exception as exc:
        report("9. available_presets() returns sorted list with expected entries", False, str(exc))


# ---------------------------------------------------------------------------
# Test 10: per-modality single-pipeline presets (eeg/eog/emg/ecg)
# ---------------------------------------------------------------------------
def test_modality_presets():
    try:
        eeg = get_preset("eeg")
        eog = get_preset("eog")
        emg = get_preset("emg")
        ecg = get_preset("ecg")
        for p, name in ((eeg, "eeg"), (eog, "eog"), (emg, "emg"), (ecg, "ecg")):
            assert isinstance(p, PreprocessingPipeline), (
                f"{name}: expected PreprocessingPipeline, got {type(p)}"
            )
        # EEG/EOG: bandpass 0.3-40 Hz (matches Phan)
        assert isinstance(eeg.steps[0], BandpassFilter)
        assert eeg.steps[0].low == 0.3 and eeg.steps[0].high == 40.0
        assert isinstance(eog.steps[0], BandpassFilter)
        assert eog.steps[0].low == 0.3 and eog.steps[0].high == 40.0
        # EMG: high-pass 10 Hz (matches Phan/AASM -- NOT bandpass 0.3-10)
        assert isinstance(emg.steps[0], HighPassFilter), (
            f"EMG step 0 should be HighPassFilter, got {type(emg.steps[0])}"
        )
        assert emg.steps[0].cutoff == 10.0
        # ECG: bandpass 0.5-40 Hz
        assert isinstance(ecg.steps[0], BandpassFilter)
        assert ecg.steps[0].low == 0.5 and ecg.steps[0].high == 40.0
        # All end with Resample to 100 Hz
        for p in (eeg, eog, emg, ecg):
            assert isinstance(p.steps[-1], Resample)
            assert p.steps[-1].target_fs == 100.0
        report("10. per-modality presets: EEG/EOG BP 0.3-40, EMG HP >10, ECG BP 0.5-40", True)
    except Exception as exc:
        report("10. per-modality presets: EEG/EOG BP 0.3-40, EMG HP >10, ECG BP 0.5-40", False, str(exc))


# ---------------------------------------------------------------------------
# Test 11: modality presets with notch_freq injection
# ---------------------------------------------------------------------------
def test_modality_preset_with_notch():
    try:
        eeg_no_notch = get_preset("eeg", notch_freq=None)
        eeg_50 = get_preset("eeg", notch_freq=50)
        assert not any(isinstance(s, NotchFilter) for s in eeg_no_notch.steps)
        notches = [s for s in eeg_50.steps if isinstance(s, NotchFilter)]
        assert len(notches) == 1, f"expected 1 notch, got {len(notches)}"
        assert notches[0].freq == 50.0, f"notch freq = {notches[0].freq}"
        # Different hash
        assert eeg_no_notch.hash() != eeg_50.hash()
        report("11. modality presets support notch_freq override", True)
    except Exception as exc:
        report("11. modality presets support notch_freq override", False, str(exc))


# ---------------------------------------------------------------------------
# Test 12: time_domain bundle preset -- returns dict with per-modality pipes
# ---------------------------------------------------------------------------
def test_time_domain_bundle():
    try:
        bundle = get_preset("time_domain")
        assert isinstance(bundle, dict), f"expected dict, got {type(bundle)}"
        for key in ("EEG", "EOG", "EMG", "ECG"):
            assert key in bundle, f"missing modality {key}"
            assert isinstance(bundle[key], PreprocessingPipeline)
        # Each modality ends with Resample
        for key, pipe in bundle.items():
            assert isinstance(pipe.steps[-1], Resample), f"{key} last step not Resample"
        # EEG: bandpass 0.3-40
        eeg_bp = [s for s in bundle["EEG"].steps if isinstance(s, BandpassFilter)][0]
        assert eeg_bp.high == 40.0 and eeg_bp.low == 0.3
        # EMG: high-pass >10 Hz (not bandpass!)
        emg_hp = [s for s in bundle["EMG"].steps if isinstance(s, HighPassFilter)]
        assert len(emg_hp) == 1, f"EMG should have 1 HighPassFilter, got {len(emg_hp)}"
        assert emg_hp[0].cutoff == 10.0
        # Different hashes between modalities
        hashes = {k: v.hash() for k, v in bundle.items()}
        # EEG and EOG use same band -> same hash
        assert hashes["EEG"] == hashes["EOG"]
        # EMG and ECG must differ from EEG (different filter types/params)
        assert hashes["EMG"] != hashes["EEG"]
        assert hashes["ECG"] != hashes["EEG"]
        report("12. time_domain bundle: EEG/EOG BP, EMG HP, ECG BP", True)
    except Exception as exc:
        report("12. time_domain bundle: EEG/EOG BP, EMG HP, ECG BP", False, str(exc))


# ---------------------------------------------------------------------------
# Test 13: time_frequency bundle = time_domain + spectrogram per modality
# ---------------------------------------------------------------------------
def test_time_frequency_bundle():
    try:
        bundle = get_preset("time_frequency")
        td = get_preset("time_domain")
        assert isinstance(bundle, dict)
        for key in ("EEG", "EOG", "EMG", "ECG"):
            pipe = bundle[key]
            # Same leading steps as time_domain, plus one XSleepNetSpectrogram at end
            assert len(pipe.steps) == len(td[key].steps) + 1
            assert isinstance(pipe.steps[-1], XSleepNetSpectrogram)
            # Leading steps identical hashes (spec comparison)
            for i, (a, b) in enumerate(zip(pipe.steps, td[key].steps)):
                assert a.spec() == b.spec(), f"{key} step {i} mismatch"
        report("13. time_frequency bundle is time_domain + spectrogram", True)
    except Exception as exc:
        report("13. time_frequency bundle is time_domain + spectrogram", False, str(exc))


# ---------------------------------------------------------------------------
# Test 14: time_frequency preset uses clip_db=-25 by default
# ---------------------------------------------------------------------------
def test_time_frequency_clip_db():
    try:
        bundle = get_preset("time_frequency")
        for key, pipe in bundle.items():
            spec_steps = [s for s in pipe.steps if isinstance(s, XSleepNetSpectrogram)]
            assert len(spec_steps) == 1, f"{key}: expected 1 XSleepNetSpectrogram"
            s = spec_steps[0]
            assert s.clip_db == -25.0, (
                f"{key}: expected clip_db=-25.0, got {s.clip_db}"
            )
        # Also verify the clipping works on actual data
        import numpy as np
        p = bundle["EEG"]
        compiled = p.compile(fs_in=100.0)
        # 1D signal: 3000 samples = 1 epoch at 100 Hz
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(3000).astype(np.float32) * 0.01  # very small signal
        out = compiled(sig)
        # After log-scaling, small-signal bins should be clipped at -25 dB
        assert out.min() >= -25.0 - 0.01, (
            f"Spectrogram min {out.min():.1f} below clip_db=-25"
        )
        report("14. time_frequency preset uses clip_db=-25 dB", True)
    except Exception as exc:
        report("14. time_frequency preset uses clip_db=-25 dB", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running preset pipeline tests")
    print("=" * 60)

    test_raw_preset_steps()
    test_seqsleepnet_preset_steps()
    test_xsleepnet_mouse_noverlap()
    test_identity_preset()
    test_unknown_preset_raises()
    test_hash_determinism()
    test_hash_uniqueness()
    test_overrides()
    test_available_presets()
    test_modality_presets()
    test_modality_preset_with_notch()
    test_time_domain_bundle()
    test_time_frequency_bundle()
    test_time_frequency_clip_db()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
