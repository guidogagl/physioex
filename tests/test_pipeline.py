"""Unit tests for the preprocessing pipeline and step implementations."""
import pickle

import numpy as np

from physioex.data.pipeline import PreprocessingPipeline
from physioex.data.steps.bandpass import BandpassFilter
from physioex.data.steps.identity import Identity
from physioex.data.steps.normalize import ZScoreNormalize
from physioex.data.steps.notch import NotchFilter
from physioex.data.steps.resample import Resample
from physioex.data.steps.spectrogram import XSleepNetSpectrogram


def test_hash_determinism():
    p1 = PreprocessingPipeline([BandpassFilter(0.3, 40)])
    p2 = PreprocessingPipeline([BandpassFilter(0.3, 40)])
    assert p1.hash() == p2.hash()


def test_hash_sensitivity():
    p1 = PreprocessingPipeline([BandpassFilter(0.3, 40)])
    p2 = PreprocessingPipeline([BandpassFilter(0.5, 40)])
    assert p1.hash() != p2.hash()


def test_float_normalization():
    assert BandpassFilter(low=0.3).spec() == BandpassFilter(low=0.30000000000000004).spec()


def test_argument_order_invariance():
    s1 = BandpassFilter(low=0.3, high=40.0, order=5).spec()
    s2 = BandpassFilter(order=5, high=40.0, low=0.3).spec()
    assert s1 == s2


def test_identity_pipeline():
    compiled = PreprocessingPipeline([Identity()]).compile(fs_in=256)
    x = np.random.randn(100).astype(np.float32)
    assert np.array_equal(x, compiled(x))


def test_bandpass_filter():
    fs = 256.0
    t = np.arange(0, 2, 1.0 / fs)
    signal_60hz = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    energy_in = np.sum(signal_60hz ** 2)
    cs = BandpassFilter(low=0.3, high=40.0).compile(fs)
    energy_out = np.sum(cs.apply(signal_60hz) ** 2)
    assert energy_out < energy_in / 2, f"in={energy_in:.2f}, out={energy_out:.2f}"


def test_resample():
    fs_in, target_fs, n = 256.0, 100.0, 2560
    signal = np.random.randn(n).astype(np.float32)
    out = Resample(target_fs=target_fs).compile(fs_in).apply(signal)
    assert out.shape[-1] == int(round(n * target_fs / fs_in))


def test_resample_noop():
    signal = np.random.randn(1000).astype(np.float64)
    out = Resample(target_fs=100.0).compile(100.0).apply(signal)
    assert np.allclose(signal, out, atol=1e-6)
    assert out.dtype == np.float32


def test_pipeline_fs_propagation():
    compiled = PreprocessingPipeline([Resample(100), BandpassFilter(0.3, 40)]).compile(fs_in=256)
    assert compiled.fs_out == 100


def test_pipeline_spectrogram_fs():
    compiled = PreprocessingPipeline([Resample(100), XSleepNetSpectrogram()]).compile(fs_in=256)
    # Spectrogram returns fs_out=0; pipeline keeps the last non-zero fs (100).
    assert compiled.fs_out == 100


def test_empty_pipeline():
    pipe = PreprocessingPipeline([])
    h = pipe.hash()
    assert isinstance(h, str) and len(h) == 16
    x = np.random.randn(100).astype(np.float32)
    assert np.array_equal(x, pipe.compile(fs_in=256)(x))


def test_zscore_normalize():
    np.random.seed(42)
    signal = np.random.randn(10000).astype(np.float32) * 5 + 3
    out = ZScoreNormalize().compile(fs_in=256).apply(signal)
    assert abs(out.mean()) < 0.01
    assert abs(out.std() - 1.0) < 0.01


def test_spectrogram_1d():
    fs = 100.0
    signal = np.random.randn(int(fs * 30)).astype(np.float32)
    out = XSleepNetSpectrogram(nperseg=200, noverlap=100, nfft=256).compile(fs).apply(signal)
    assert out.ndim == 2
    assert out.shape[1] == 256 // 2 + 1


def test_spectrogram_2d():
    fs = 100.0
    signal = np.random.randn(5, 3000).astype(np.float32)
    out = XSleepNetSpectrogram(nperseg=200, noverlap=100, nfft=256).compile(fs).apply(signal)
    assert out.ndim == 3
    assert out.shape[0] == 5
    assert out.shape[2] == 256 // 2 + 1


def test_picklability():
    steps = [
        Identity(), BandpassFilter(0.3, 40), NotchFilter(50, 30),
        Resample(100), ZScoreNormalize(), XSleepNetSpectrogram(),
    ]
    for step in steps:
        restored = pickle.loads(pickle.dumps(step))
        assert step.spec() == restored.spec(), type(step).__name__


def test_compiled_pipeline_callable():
    fs_in = 256.0
    t = np.arange(0, 2, 1.0 / fs_in)
    signal = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    out = PreprocessingPipeline([BandpassFilter(0.3, 40), Resample(100)]).compile(fs_in)(signal)
    assert out.shape[-1] == int(round(len(t) * 100 / fs_in))
    assert out.dtype == np.float32
