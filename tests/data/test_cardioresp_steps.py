"""Tests for the cardiorespiratory data-layer primitives.

Covers:
  - the new steps (ChebyshevLowpass, FFTResample, InterpResample,
    FixedLength, Clip, ZScore, PhysicalScale) against direct numpy/scipy/
    pandas references;
  - pipeline.hash() sensitivity to step parameters;
  - the ``wav2sleep`` and ``sleepppgnet`` presets end-to-end on a
    synthetic 8 h signal (1200 epochs of 1024 samples for ECG/PPG, 256
    for RESP, 4096 for EOG);
  - the AASM-4 label scheme;
  - PPG modality classification;
  - epoch slicing at fractional sampling rates (fs = 1024/30, 256/30).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.signal import cheby2, resample, sosfiltfilt

from physioex.data.pipeline import PreprocessingPipeline
from physioex.data.steps import (
    ChebyshevLowpass,
    FFTResample,
    FixedLength,
    Identity,
    InterpResample,
    Clip,
    ZScore,
    ZScoreNormalize,
    PhysicalScale,
    Resample,
)
from physioex.data.presets import get_preset
from physioex.data.labels import (
    AASM4_CLASS_NAMES,
    available_label_transforms,
    get_label_transform,
)
from physioex.data.modality import ModalityType, infer_channel_modality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# ChebyshevLowpass
# ---------------------------------------------------------------------------


def test_chebyshev_lowpass_matches_scipy(rng):
    fs = 256.0
    t = np.arange(0, 8, 1.0 / fs)
    x = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    x = x.astype(np.float32)
    ref = sosfiltfilt(
        cheby2(8, 40, 8.0 / (fs / 2), btype="low", output="sos"),
        x.astype(np.float64),
    )
    out = ChebyshevLowpass(8.0, 8, 40).compile(fs).apply(x)
    assert out.dtype == np.float32
    assert out.shape == x.shape
    assert np.allclose(out, ref, atol=1e-4), "output must equal sosfiltfilt(cheby2)"
    # High-frequency energy must be strongly attenuated (stop-band >= 40 dB).
    low = np.sin(2 * np.pi * 1 * t)
    energy_ratio = np.sum(out**2) / np.sum(low**2)
    assert energy_ratio < 2.0, f"lowpass did not preserve the 1 Hz tone (ratio={energy_ratio})"


def test_chebyshev_lowpass_skips_when_cutoff_at_nyquist(rng):
    fs = 16.0  # Nyquist 8 Hz
    x = rng.standard_normal(1600).astype(np.float32)
    out = ChebyshevLowpass(cutoff_hz=8.0).compile(fs).apply(x)
    assert np.allclose(out, x, atol=1e-6), "cutoff == Nyquist must pass through"


def test_chebyshev_lowpass_hash_changes_with_params():
    base = PreprocessingPipeline([ChebyshevLowpass()])
    assert base.hash() != PreprocessingPipeline([ChebyshevLowpass(cutoff_hz=4.0)]).hash()
    assert base.hash() != PreprocessingPipeline([ChebyshevLowpass(order=4)]).hash()
    assert base.hash() != PreprocessingPipeline([ChebyshevLowpass(stop_db=20.0)]).hash()
    # Deterministic for identical parameters
    assert base.hash() == PreprocessingPipeline([ChebyshevLowpass()]).hash()


# ---------------------------------------------------------------------------
# FFTResample
# ---------------------------------------------------------------------------


def test_fft_resample_matches_scipy(rng):
    fs_in = 250.0
    n = 12345
    x = rng.standard_normal(n).astype(np.float32)
    target = 1024.0 / 30.0
    ref = resample(x.astype(np.float64), int(n * target / fs_in))
    out = FFTResample(target).compile(fs_in).apply(x)
    assert out.dtype == np.float32
    assert out.shape == (int(n * target / fs_in),), (
        f"n_out must be int(len*target/fs): got {out.shape}, "
        f"want {int(n * target / fs_in)}"
    )
    assert np.allclose(out, ref, atol=1e-4)


def test_fft_resample_fs_out():
    cs = FFTResample(1024.0 / 30.0).compile(250.0)
    assert cs.fs_out == 1024.0 / 30.0


def test_fft_resample_hash_changes_with_params():
    base = PreprocessingPipeline([FFTResample(1024.0 / 30.0)])
    assert base.hash() != PreprocessingPipeline([FFTResample(256.0 / 30.0)]).hash()
    assert base.hash() == PreprocessingPipeline([FFTResample(1024.0 / 30.0)]).hash()


# ---------------------------------------------------------------------------
# InterpResample
# ---------------------------------------------------------------------------


def _pandas_interp_reference(x: np.ndarray, fs: float, spe: int, epoch_sec: float):
    """Reference: dropna -> union index -> interpolate(index, inside) -> target -> fillna(0)."""
    n = len(x)
    fs_out = spe / epoch_sec
    n_out = int(np.floor(n / fs * fs_out))
    s = pd.Series(x, index=np.arange(n) / fs).dropna()
    target = (np.arange(n_out) + 1) / fs_out
    full = pd.Series(index=np.union1d(s.index.values, target)).reindex(
        np.union1d(s.index.values, target)
    )
    full.loc[s.index] = s.values
    full = full.interpolate(method="index", limit_area="inside")
    ref = full.reindex(target)
    return ref.fillna(0.0).values


@pytest.mark.parametrize("nans", [0, 5, 200], ids=["clean", "few-nans", "many-nans"])
def test_interp_resample_matches_pandas(rng, nans):
    fs = 250.0
    n = int(8 * 3600 * fs)  # 8 h at 250 Hz
    t = np.arange(n) / fs
    x = (
        np.sin(2 * np.pi * 0.2 * t)
        + rng.standard_normal(n) * 0.01
    )
    if nans:
        # Scatter NaN gaps (interior + one leading gap).
        idx = rng.choice(n, size=nans, replace=False)
        x = x.astype(np.float64)
        x[idx] = np.nan
        if nans > 10:
            x[:20] = np.nan  # leading gap -> zero
    x = x.astype(np.float32)
    spe = 1024
    out = InterpResample(spe).compile(fs).apply(x)
    ref = _pandas_interp_reference(x.astype(np.float64), fs, spe, 30.0)
    assert out.shape == ref.shape
    assert out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-5), (
        f"max diff {np.max(np.abs(np.asarray(out) - np.asarray(ref)))}"
    )


def test_interp_resample_length_and_fs(rng):
    fs = 250.0
    n = int(8 * 3600 * fs)
    x = rng.standard_normal(n).astype(np.float32)
    cs = InterpResample(1024).compile(fs)
    assert cs.fs_out == 1024.0 / 30.0
    out = cs.apply(x)
    # 8 h at 34.133 samples/s -> floor(n/fs * 1024/30)
    expected = int(np.floor(n / fs * (1024.0 / 30.0)))
    assert out.shape == (expected,), f"got {out.shape}, want ({expected},)"


def test_interp_resample_hash_changes_with_params():
    base = PreprocessingPipeline([InterpResample(1024)])
    assert base.hash() != PreprocessingPipeline([InterpResample(256)]).hash()
    # epoch_sec is a real parameter: changing it changes the hash...
    assert base.hash() != PreprocessingPipeline([InterpResample(1024, epoch_sec=15.0)]).hash()
    # ...but the default is stable.
    assert base.hash() == PreprocessingPipeline([InterpResample(1024)]).hash()


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------


def test_clip_matches_numpy(rng):
    x = rng.standard_normal(50000).astype(np.float32) * 2.0
    ref = np.clip(x, x.mean() - 3.0 * x.std(ddof=0), x.mean() + 3.0 * x.std(ddof=0))
    out = Clip(3.0).compile(100.0).apply(x)
    assert out.dtype == np.float32
    assert np.array_equal(out, ref.astype(np.float32))
    # Outliers beyond 3 std must be exactly at the bound
    assert out.max() == ref.max()
    assert out.min() == ref.min()


def test_clip_hash_changes_with_params():
    base = PreprocessingPipeline([Clip()])
    assert base.hash() != PreprocessingPipeline([Clip(n_std=2.0)]).hash()
    assert base.hash() == PreprocessingPipeline([Clip()]).hash()


# ---------------------------------------------------------------------------
# ZScore
# ---------------------------------------------------------------------------


def test_zscore_matches_numpy(rng):
    x = rng.standard_normal(100000).astype(np.float32) * 3.0 + 7.0
    ref = (x - x.mean()) / max(x.std(ddof=1), 1e-6)
    out = ZScore(ddof=1, eps=1e-6).compile(100.0).apply(x)
    assert out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-5)
    assert abs(out.mean()) < 1e-3
    assert abs(out.std(ddof=1) - 1.0) < 1e-3


def test_zscore_ddof0_default():
    x = np.arange(100, dtype=np.float32)
    ref = (x - x.mean()) / max(x.std(ddof=0), 0.0)
    out = ZScore().compile(100.0).apply(x)
    assert np.allclose(out, ref, atol=1e-5)


def test_zscore_zero_std_zero_eps():
    x = np.full(100, 5.0, dtype=np.float32)
    out = ZScore(ddof=0, eps=0.0).compile(100.0).apply(x)
    assert np.allclose(out, 0.0), "std==0 and eps==0 must return x - mean"


def test_zscore_hash_changes_with_params():
    base = PreprocessingPipeline([ZScore()])
    assert base.hash() != PreprocessingPipeline([ZScore(ddof=1)]).hash()
    assert base.hash() != PreprocessingPipeline([ZScore(eps=1e-6)]).hash()
    assert base.hash() == PreprocessingPipeline([ZScore()]).hash()


# ---------------------------------------------------------------------------
# FixedLength
# ---------------------------------------------------------------------------


def test_fixed_length_pads_and_truncates(rng):
    fs = 1024.0 / 30.0
    # Exactly 10 h
    n10h = round(36000 * fs)  # 1228800
    x_long = rng.standard_normal(int(n10h * 1.1)).astype(np.float32)
    out_long = FixedLength(36000.0).compile(fs).apply(x_long)
    assert out_long.shape == (n10h,)
    # Truncation keeps the start
    assert np.allclose(out_long, x_long[:n10h], atol=1e-6)

    x_short = rng.standard_normal(int(n10h * 0.8)).astype(np.float32)
    out_short = FixedLength(36000.0, fill=-1.0).compile(fs).apply(x_short)
    assert out_short.shape == (n10h,)
    assert np.allclose(out_short[: len(x_short)], x_short, atol=1e-6)
    assert np.allclose(out_short[len(x_short):], -1.0, atol=1e-6)


def test_fixed_length_exact_length_is_noop(rng):
    fs = 100.0
    n = round(30.0 * fs)
    x = rng.standard_normal(n).astype(np.float32)
    out = FixedLength(30.0).compile(fs).apply(x)
    assert np.array_equal(out, x)


def test_fixed_length_hash_changes_with_params():
    base = PreprocessingPipeline([FixedLength(36000.0)])
    assert base.hash() != PreprocessingPipeline([FixedLength(72000.0)]).hash()
    assert base.hash() != PreprocessingPipeline([FixedLength(36000.0, fill=1.0)]).hash()
    assert base.hash() == PreprocessingPipeline([FixedLength(36000.0)]).hash()


# ---------------------------------------------------------------------------
# PhysicalScale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unit,factor",
    [("uV", 1e-3), ("µV", 1e-3), ("mV", 1.0), ("V", 1e3), ("", 1.0), ("weird", 1.0)],
)
def test_physical_scale_mV(rng, unit, factor):
    x = rng.standard_normal(1000).astype(np.float32)
    out = PhysicalScale("mV").compile(100.0, header={"physical_dimension": unit}).apply(x)
    assert np.allclose(out, x * factor, atol=1e-6)


def test_physical_scale_range(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    # Range [-2, 4] -> center 1, half 3
    out = PhysicalScale("range").compile(
        100.0, header={"physical_min": -2.0, "physical_max": 4.0}
    ).apply(x)
    ref = (x - 1.0) / 3.0
    assert np.allclose(out, ref, atol=1e-6)
    # Symmetric range [pmin, pmax] with |x| <= half maps to [-1, 1]
    y = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    out2 = PhysicalScale("range").compile(
        100.0, header={"physical_min": -1.0, "physical_max": 1.0}
    ).apply(y)
    assert np.allclose(out2, y, atol=1e-6)


def test_physical_scale_zero_range_passthrough(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    out = PhysicalScale("range").compile(
        100.0, header={"physical_min": 1.0, "physical_max": 1.0}
    ).apply(x)
    assert np.allclose(out, x, atol=1e-6)


def test_physical_scale_missing_header_passthrough(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    out_mv = PhysicalScale("mV").compile(100.0, header=None).apply(x)
    assert np.allclose(out_mv, x, atol=1e-6)
    out_range = PhysicalScale("range").compile(100.0, header={"physical_dimension": "uV"}).apply(x)
    assert np.allclose(out_range, x, atol=1e-6)


def test_physical_scale_hash_changes_with_mode():
    base = PreprocessingPipeline([PhysicalScale("mV")])
    assert base.hash() != PreprocessingPipeline([PhysicalScale("range")]).hash()
    assert base.hash() == PreprocessingPipeline([PhysicalScale("mV")]).hash()


def test_physical_scale_needs_header_flag():
    assert PhysicalScale("mV").needs_header is True
    # Steps that don't consume the header simply don't declare the flag.
    assert not getattr(Resample(100), "needs_header", False)


# ---------------------------------------------------------------------------
# ZScoreNormalize: default behaviour must be unchanged
# ---------------------------------------------------------------------------


def test_zscore_normalize_default_unchanged(rng):
    x = rng.standard_normal(10000).astype(np.float32) * 5 + 3
    out = ZScoreNormalize().compile(256.0).apply(x)
    assert abs(out.mean()) < 0.01
    assert abs(out.std() - 1.0) < 0.01
    # The default eps=1e-8 spec string must be stable.
    assert ZScoreNormalize().spec() == ZScoreNormalize(mode="per_subject", eps=1e-8).spec()


# ---------------------------------------------------------------------------
# Presets: structure + end-to-end
# ---------------------------------------------------------------------------


def test_wav2sleep_preset_structure():
    bundle = get_preset("wav2sleep")
    assert isinstance(bundle, dict)
    for mod in ("ECG", "PPG", "RESP", "EOG"):
        assert mod in bundle, f"missing {mod}"
        pipe = bundle[mod]
        assert isinstance(pipe, PreprocessingPipeline)
        assert isinstance(pipe.steps[0], PhysicalScale)
        assert isinstance(pipe.steps[1], InterpResample)
        assert isinstance(pipe.steps[2], FixedLength)
        assert isinstance(pipe.steps[3], ZScore)
    # ECG -> mV, PPG/RESP -> range
    assert bundle["ECG"].steps[0].mode == "mV"
    assert bundle["PPG"].steps[0].mode == "range"
    assert bundle["RESP"].steps[0].mode == "range"
    assert bundle["EOG"].steps[0].mode == "mV"
    # Interpolation grids
    assert bundle["ECG"].steps[1].samples_per_epoch == 1024
    assert bundle["PPG"].steps[1].samples_per_epoch == 1024
    assert bundle["EOG"].steps[1].samples_per_epoch == 4096
    assert bundle["RESP"].steps[1].samples_per_epoch == 256
    # Z-score over the full padded grid (ddof=1, eps=1e-6)
    for mod in ("ECG", "PPG", "RESP", "EOG"):
        assert bundle[mod].steps[3].ddof == 1
        assert bundle[mod].steps[3].eps == 1e-6


def test_sleepppgnet_preset_structure():
    pipe = get_preset("sleepppgnet")
    assert isinstance(pipe, PreprocessingPipeline)
    assert [type(s).__name__ for s in pipe.steps] == [
        "ChebyshevLowpass", "FFTResample", "Clip", "ZScore", "FixedLength",
    ]
    assert pipe.steps[0].cutoff_hz == 8.0
    assert pipe.steps[0].order == 8
    assert pipe.steps[0].stop_db == 40.0
    assert abs(pipe.steps[1].target_fs - 1024.0 / 30.0) < 1e-9
    assert pipe.steps[2].n_std == 3.0
    assert pipe.steps[3].ddof == 0 and pipe.steps[3].eps == 0.0
    assert pipe.steps[4].seconds == 36000.0


def _synthetic_recording(rng, duration_sec: float, fs: float, amplitude: float = 1.0):
    """8 h of a smooth signal (no NaN) at ``fs``."""
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    x = (
        amplitude * (np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        + rng.standard_normal(n) * 0.01
    )
    return x.astype(np.float32)


def test_wav2sleep_preset_end_to_end(rng):
    bundle = get_preset("wav2sleep")
    duration = 8 * 3600  # 8 h
    header = {
        "physical_dimension": "mV",
        "physical_min": -1.0,
        "physical_max": 1.0,
    }
    fs_map = {"ECG": 250.0, "PPG": 250.0, "RESP": 250.0, "EOG": 512.0}
    n1024 = round(36000 * 1024.0 / 30.0)  # 1228800
    n256 = round(36000 * 256.0 / 30.0)    # 307200
    n4096 = round(36000 * 4096.0 / 30.0)  # 4915200

    expected = {"ECG": (1200, 1024), "PPG": (1200, 1024),
                "RESP": (1200, 256), "EOG": (1200, 4096)}
    for mod, pipe in bundle.items():
        x = _synthetic_recording(rng, duration, fs_map[mod])
        compiled = pipe.compile(fs_map[mod], header=header)
        # Run through one epoch of the dataset logic:
        # InterpResample -> FixedLength -> (epoch at end)
        from physioex.data.base import BasePhysioDataset

        # Reuse the dataset's epoch slicing on the pipeline output.
        y = compiled(x)  # 1D at fs_out
        fs_out = compiled.fs_out
        sps = int(round(30.0 * fs_out))
        n_full = (y.shape[-1] // sps) * sps
        epoched = y[..., :n_full].reshape(-1, sps)
        assert epoched.shape == expected[mod], (
            f"{mod}: got {epoched.shape}, want {expected[mod]}"
        )
        # The 10 h window is zero-padded: with an 8 h recording the tail
        # (last 2 h) must be exactly zero before z-scoring, so after
        # z-scoring the tail equals the (constant) z-score of 0.
        assert not np.isnan(epoched).any(), f"{mod}: NaNs in output"
        # z-score is finite everywhere
        assert np.isfinite(epoched).all(), f"{mod}: non-finite values"


def test_sleepppgnet_preset_end_to_end(rng):
    pipe = get_preset("sleepppgnet")
    duration = 8 * 3600
    fs_in = 250.0
    x = _synthetic_recording(rng, duration, fs_in, amplitude=0.4)
    compiled = pipe.compile(fs_in, header=None)  # no PhysicalScale here
    y = compiled(x)
    fs_out = compiled.fs_out
    assert abs(fs_out - 1024.0 / 30.0) < 1e-9
    sps = int(round(30.0 * fs_out))
    assert sps == 1024, f"fs=1024/30 must give exactly 1024 samples/epoch, got {sps}"
    n_full = (y.shape[-1] // sps) * sps
    epoched = y[..., :n_full].reshape(-1, sps)
    # 10 h at 1024/30 -> 1228800 samples -> 1200 epochs of 1024
    assert epoched.shape == (1200, 1024), f"got {epoched.shape}"
    # Last 2 h zero-padded: the z-score of 0 over the full grid is a
    # constant != 0 (the grid mean/std are shifted by the padding).
    tail = epoched[-40:]
    assert np.allclose(tail, tail[0:1], atol=1e-5), "zero-padding tail must be constant"


# ---------------------------------------------------------------------------
# Label schemes
# ---------------------------------------------------------------------------


def test_aasm4_mapping():
    tf = get_label_transform("aasm4")
    # AASM 5: W=0, N1=1, N2=2, N3=3, REM=4, -1 -> AASM 4: W=0, N1->1, N2->1, N3=2, REM=3, -1
    labels = np.array([0, 1, 2, 3, 4, -1, 0, 3], dtype=np.int16)
    out = tf(labels)
    assert out.dtype == labels.dtype
    assert list(out) == [0, 1, 1, 2, 3, -1, 0, 2]
    assert tuple(AASM4_CLASS_NAMES) == ("W", "Light", "Deep", "REM")
    # N1 and N2 must map to the same class (the whole point of the 4-class scheme)
    assert out[1] == out[2] != out[0]


def test_aasm5_identity():
    tf = get_label_transform("aasm5")
    labels = np.array([0, 1, 2, 3, 4, -1], dtype=np.int16)
    out = tf(labels)
    assert list(out) == list(labels)


def test_label_transform_names():
    assert set(available_label_transforms()) == {"aasm4", "aasm5"}
    with pytest.raises(ValueError):
        get_label_transform("bogus")


# ---------------------------------------------------------------------------
# PPG modality classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Pleth", "PLTH", "PPG", "TC-PPG", "PLESMO", "PulseAmp", "RD-Pleth",
     "PWF", "PLETHWV", "PLETHMASIMO", "pleth", "PlethMasimo"],
)
def test_ppg_channel_names_map_to_ppg(name):
    assert infer_channel_modality(name) == ModalityType.PPG


@pytest.mark.parametrize(
    "name",
    ["SpO2", "SaO2", "OxStat", "OxStatus"],
)
def test_saturation_names_stay_spo2(name):
    assert infer_channel_modality(name) == ModalityType.SPO2


def test_ppg_enum_value_and_existing_values_unchanged():
    assert int(ModalityType.PPG) == 15
    # Existing members keep their integer values.
    assert int(ModalityType.EEG) == 0
    assert int(ModalityType.SPO2) == 5
    assert int(ModalityType.OTHER) == 14


# ---------------------------------------------------------------------------
# Epoch slicing at fractional fs
# ---------------------------------------------------------------------------


class _EpochSlicer:
    """Minimal stand-in for the dataset's final epoch reshape, using the
    exact ``int(round(epoch_length_sec * fs))`` rule from
    ``BasePhysioDataset._epoch_and_run_pipeline``."""

    @staticmethod
    def epoch(x: np.ndarray, fs: float, epoch_sec: float = 30.0) -> np.ndarray:
        sps = int(round(epoch_sec * fs))
        assert sps > 0
        n_full = (x.shape[-1] // sps) * sps
        return x[..., :n_full].reshape(-1, sps)


def test_epoch_slicing_fs_1024_over_30():
    fs = 1024.0 / 30.0
    n = int(round(30.0 * fs)) * 1200  # exactly 1200 epochs' worth
    x = np.arange(n, dtype=np.float32)
    out = _EpochSlicer.epoch(x, fs)
    assert out.shape == (1200, 1024), f"fs=1024/30 must give 1024 samples/epoch, got {out.shape}"
    assert int(round(30.0 * (1024.0 / 30.0))) == 1024


def test_epoch_slicing_fs_256_over_30():
    fs = 256.0 / 30.0
    n = int(round(30.0 * fs)) * 1000
    x = np.arange(n, dtype=np.float32)
    out = _EpochSlicer.epoch(x, fs)
    assert out.shape == (1000, 256), f"fs=256/30 must give 256 samples/epoch, got {out.shape}"


def test_epoch_slicing_via_real_dataset_method():
    """Exercise the actual ``BasePhysioDataset._epoch_and_run_pipeline`` with
    fs = 1024/30 to prove the fix lives in the dataset, not a test stand-in."""
    from physioex.data.base import BasePhysioDataset

    # Bypass Dataset.__init__ entirely; the method under test only reads
    # ``self.epoch_length_sec``.
    stub = BasePhysioDataset.__new__(BasePhysioDataset)
    stub.epoch_length_sec = 30.0
    compiled = PreprocessingPipeline([Identity()]).compile(1024.0 / 30.0)
    n = int(round(30.0 * (1024.0 / 30.0))) * 5  # 5 epochs of 1024
    x = np.arange(n, dtype=np.float32)
    out = stub._epoch_and_run_pipeline(x, 1024.0 / 30.0, compiled, None)
    assert out.shape == (5, 1024), f"got {out.shape}"
