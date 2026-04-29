"""
Unit tests for the preprocessing pipeline and step implementations.

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_pipeline.py
"""

import sys
import pickle

import numpy as np
from physioex.data.pipeline import PreprocessingStep, CompiledStep, PreprocessingPipeline, CompiledPipeline
from physioex.data.steps.identity import Identity
from physioex.data.steps.bandpass import BandpassFilter
from physioex.data.steps.notch import NotchFilter
from physioex.data.steps.resample import Resample
from physioex.data.steps.normalize import ZScoreNormalize
from physioex.data.steps.spectrogram import XSleepNetSpectrogram

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
# Test 1: Hash determinism
# ---------------------------------------------------------------------------
def test_hash_determinism():
    try:
        p1 = PreprocessingPipeline([BandpassFilter(0.3, 40)])
        p2 = PreprocessingPipeline([BandpassFilter(0.3, 40)])
        assert p1.hash() == p2.hash(), f"{p1.hash()} != {p2.hash()}"
        report("1. Hash determinism", True)
    except Exception as exc:
        report("1. Hash determinism", False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: Hash sensitivity
# ---------------------------------------------------------------------------
def test_hash_sensitivity():
    try:
        p1 = PreprocessingPipeline([BandpassFilter(0.3, 40)])
        p2 = PreprocessingPipeline([BandpassFilter(0.5, 40)])
        assert p1.hash() != p2.hash(), "Hashes should differ for different params"
        report("2. Hash sensitivity", True)
    except Exception as exc:
        report("2. Hash sensitivity", False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: Float normalization
# ---------------------------------------------------------------------------
def test_float_normalization():
    try:
        s1 = BandpassFilter(low=0.3).spec()
        s2 = BandpassFilter(low=0.30000000000000004).spec()
        assert s1 == s2, f"Specs differ:\n  {s1}\n  {s2}"
        report("3. Float normalization", True)
    except Exception as exc:
        report("3. Float normalization", False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: Argument order invariance
# ---------------------------------------------------------------------------
def test_argument_order_invariance():
    try:
        # Constructor kwarg order shouldn't affect spec (sorted keys)
        s1 = BandpassFilter(low=0.3, high=40.0, order=5).spec()
        s2 = BandpassFilter(order=5, high=40.0, low=0.3).spec()
        assert s1 == s2, f"Specs differ:\n  {s1}\n  {s2}"
        report("4. Argument order invariance", True)
    except Exception as exc:
        report("4. Argument order invariance", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: Identity pipeline
# ---------------------------------------------------------------------------
def test_identity_pipeline():
    try:
        pipe = PreprocessingPipeline([Identity()])
        compiled = pipe.compile(fs_in=256)
        x = np.random.randn(100).astype(np.float32)
        y = compiled(x)
        assert np.array_equal(x, y), "Identity pipeline should return input unchanged"
        report("5. Identity pipeline", True)
    except Exception as exc:
        report("5. Identity pipeline", False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: BandpassFilter compile
# ---------------------------------------------------------------------------
def test_bandpass_filter():
    try:
        fs = 256.0
        t = np.arange(0, 2, 1.0 / fs)
        # Create a 60Hz sinusoid (outside 0.3-40Hz band)
        signal_60hz = np.sin(2 * np.pi * 60 * t).astype(np.float32)
        energy_in = np.sum(signal_60hz ** 2)

        bp = BandpassFilter(low=0.3, high=40.0)
        cs = bp.compile(fs)
        filtered = cs.apply(signal_60hz)
        energy_out = np.sum(filtered ** 2)

        assert energy_out < energy_in / 2, (
            f"60Hz energy not sufficiently attenuated: in={energy_in:.2f}, out={energy_out:.2f}"
        )
        report("6. BandpassFilter attenuates 60Hz", True)
    except Exception as exc:
        report("6. BandpassFilter attenuates 60Hz", False, str(exc))


# ---------------------------------------------------------------------------
# Test 7: Resample compile
# ---------------------------------------------------------------------------
def test_resample():
    try:
        fs_in = 256.0
        target_fs = 100.0
        n_samples = 2560  # 10 seconds at 256Hz
        signal = np.random.randn(n_samples).astype(np.float32)

        rs = Resample(target_fs=target_fs)
        cs = rs.compile(fs_in)
        out = cs.apply(signal)

        expected_len = int(round(n_samples * target_fs / fs_in))
        assert out.shape[-1] == expected_len, (
            f"Expected length {expected_len}, got {out.shape[-1]}"
        )
        report("7. Resample output length", True)
    except Exception as exc:
        report("7. Resample output length", False, str(exc))


# ---------------------------------------------------------------------------
# Test 8: Resample compile with fs_in == target_fs (fast path)
# ---------------------------------------------------------------------------
def test_resample_noop():
    try:
        signal = np.random.randn(1000).astype(np.float64)
        rs = Resample(target_fs=100.0)
        cs = rs.compile(100.0)
        out = cs.apply(signal)
        # Should be equal (up to dtype cast)
        assert np.allclose(signal, out, atol=1e-6), "Resample no-op should preserve values"
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
        report("8. Resample no-op fast path", True)
    except Exception as exc:
        report("8. Resample no-op fast path", False, str(exc))


# ---------------------------------------------------------------------------
# Test 9: Pipeline fs propagation
# ---------------------------------------------------------------------------
def test_pipeline_fs_propagation():
    try:
        pipe = PreprocessingPipeline([Resample(100), BandpassFilter(0.3, 40)])
        compiled = pipe.compile(fs_in=256)
        assert compiled.fs_out == 100, f"Expected fs_out=100, got {compiled.fs_out}"
        report("9. Pipeline fs propagation", True)
    except Exception as exc:
        report("9. Pipeline fs propagation", False, str(exc))


# ---------------------------------------------------------------------------
# Test 10: Pipeline with spectrogram (fs_out=0 does not propagate)
# ---------------------------------------------------------------------------
def test_pipeline_spectrogram_fs():
    try:
        pipe = PreprocessingPipeline([Resample(100), XSleepNetSpectrogram()])
        compiled = pipe.compile(fs_in=256)
        # Spectrogram returns fs_out=0; pipeline should keep last non-zero fs (100)
        assert compiled.fs_out == 100, f"Expected fs_out=100, got {compiled.fs_out}"
        report("10. Pipeline with spectrogram preserves last non-zero fs", True)
    except Exception as exc:
        report("10. Pipeline with spectrogram preserves last non-zero fs", False, str(exc))


# ---------------------------------------------------------------------------
# Test 11: Empty pipeline
# ---------------------------------------------------------------------------
def test_empty_pipeline():
    try:
        pipe = PreprocessingPipeline([])
        h = pipe.hash()
        assert isinstance(h, str) and len(h) == 16, f"Invalid hash: {h!r}"
        compiled = pipe.compile(fs_in=256)
        x = np.random.randn(100).astype(np.float32)
        y = compiled(x)
        assert np.array_equal(x, y), "Empty pipeline should be a no-op"
        report("11. Empty pipeline", True)
    except Exception as exc:
        report("11. Empty pipeline", False, str(exc))


# ---------------------------------------------------------------------------
# Test 12: ZScoreNormalize
# ---------------------------------------------------------------------------
def test_zscore_normalize():
    try:
        np.random.seed(42)
        signal = np.random.randn(10000).astype(np.float32) * 5 + 3
        zn = ZScoreNormalize()
        cs = zn.compile(fs_in=256)
        out = cs.apply(signal)
        assert abs(out.mean()) < 0.01, f"Mean should be ~0, got {out.mean()}"
        assert abs(out.std() - 1.0) < 0.01, f"Std should be ~1, got {out.std()}"
        report("12. ZScoreNormalize", True)
    except Exception as exc:
        report("12. ZScoreNormalize", False, str(exc))


# ---------------------------------------------------------------------------
# Test 13: XSleepNetSpectrogram on 1D input
# ---------------------------------------------------------------------------
def test_spectrogram_1d():
    try:
        fs = 100.0
        duration = 30  # 30 seconds (one epoch)
        n_samples = int(fs * duration)
        signal = np.random.randn(n_samples).astype(np.float32)

        spec_step = XSleepNetSpectrogram(nperseg=200, noverlap=100, nfft=256)
        cs = spec_step.compile(fs)
        out = cs.apply(signal)

        expected_F = 256 // 2 + 1  # nfft//2 + 1 = 129
        assert out.ndim == 2, f"Expected 2D output, got {out.ndim}D"
        assert out.shape[1] == expected_F, f"Expected F={expected_F}, got shape {out.shape}"
        report("13. XSleepNetSpectrogram 1D input", True, f"shape={out.shape}")
    except Exception as exc:
        report("13. XSleepNetSpectrogram 1D input", False, str(exc))


# ---------------------------------------------------------------------------
# Test 14: XSleepNetSpectrogram on 2D input
# ---------------------------------------------------------------------------
def test_spectrogram_2d():
    try:
        fs = 100.0
        n_epochs = 5
        samples_per_epoch = 3000  # 30 seconds at 100Hz
        signal = np.random.randn(n_epochs, samples_per_epoch).astype(np.float32)

        spec_step = XSleepNetSpectrogram(nperseg=200, noverlap=100, nfft=256)
        cs = spec_step.compile(fs)
        out = cs.apply(signal)

        expected_F = 256 // 2 + 1
        assert out.ndim == 3, f"Expected 3D output, got {out.ndim}D"
        assert out.shape[0] == n_epochs, f"Expected {n_epochs} epochs, got {out.shape[0]}"
        assert out.shape[2] == expected_F, f"Expected F={expected_F}, got shape {out.shape}"
        report("14. XSleepNetSpectrogram 2D input", True, f"shape={out.shape}")
    except Exception as exc:
        report("14. XSleepNetSpectrogram 2D input", False, str(exc))


# ---------------------------------------------------------------------------
# Test 15: Picklability of step classes
# ---------------------------------------------------------------------------
def test_picklability():
    try:
        steps = [
            Identity(),
            BandpassFilter(0.3, 40),
            NotchFilter(50, 30),
            Resample(100),
            ZScoreNormalize(),
            XSleepNetSpectrogram(),
        ]
        for step in steps:
            data = pickle.dumps(step)
            restored = pickle.loads(data)
            assert step.spec() == restored.spec(), (
                f"Spec mismatch after pickle for {type(step).__name__}: "
                f"{step.spec()} != {restored.spec()}"
            )
        report("15. Picklability of all step classes", True)
    except Exception as exc:
        report("15. Picklability of all step classes", False, str(exc))


# ---------------------------------------------------------------------------
# Test 16: CompiledPipeline callable chains steps
# ---------------------------------------------------------------------------
def test_compiled_pipeline_callable():
    try:
        fs_in = 256.0
        t = np.arange(0, 2, 1.0 / fs_in)
        # Mix of 5Hz (in-band) and 60Hz (out-of-band)
        signal = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 60 * t)).astype(np.float32)

        pipe = PreprocessingPipeline([
            BandpassFilter(0.3, 40),
            Resample(100),
        ])
        compiled = pipe.compile(fs_in)
        out = compiled(signal)

        # After bandpass 0.3-40Hz, the 60Hz component should be attenuated.
        # After resample from 256 -> 100, length should change.
        expected_len = int(round(len(t) * 100 / fs_in))
        assert out.shape[-1] == expected_len, (
            f"Expected length {expected_len}, got {out.shape[-1]}"
        )
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
        report("16. CompiledPipeline callable chains steps", True, f"out_len={out.shape[-1]}")
    except Exception as exc:
        report("16. CompiledPipeline callable chains steps", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running preprocessing pipeline tests")
    print("=" * 60)

    test_hash_determinism()
    test_hash_sensitivity()
    test_float_normalization()
    test_argument_order_invariance()
    test_identity_pipeline()
    test_bandpass_filter()
    test_resample()
    test_resample_noop()
    test_pipeline_fs_propagation()
    test_pipeline_spectrogram_fs()
    test_empty_pipeline()
    test_zscore_normalize()
    test_spectrogram_1d()
    test_spectrogram_2d()
    test_picklability()
    test_compiled_pipeline_callable()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
