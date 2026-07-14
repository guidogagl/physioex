"""Hardening tests: label sanitization, signal-label alignment, safe slicing.

These tests validate that BasePhysioDataset:
  1. Always produces labels in AASM 5-class {0..4} or -1 (never other values).
  2. Maps dataset-specific stage aliases (e.g. N4 -> N3) as AASM requires.
  3. Never drops physical epochs: missing labels get -1, truncated signals
     get zero-fill. Labels and signals are always the same length.
  4. Handles label-signal length mismatches via _safe_slice.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pyedflib

from physioex.data.base import (
    BasePhysioDataset,
    SubjectSpec,
    AASM_VALID_LABELS,
)

from tests.test_raw_dataset_integration import (
    write_fake_edf,
    write_fake_annotations_edf,
)

def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


# ---------------------------------------------------------------------------
# _sanitize_labels
# ---------------------------------------------------------------------------

def test_sanitize_passes_valid_labels():
    try:
        labels = np.array([-1, 0, 1, 2, 3, 4, 0, 2], dtype=np.int16)
        out = BasePhysioDataset._sanitize_labels(labels)
        assert np.array_equal(out, labels), f"expected unchanged, got {out}"
        report("_sanitize_labels: valid labels pass through", True)
    except Exception as exc:
        report("_sanitize_labels: valid labels pass through", False, str(exc))


def test_sanitize_out_of_range_becomes_minus_one():
    try:
        labels = np.array([0, 5, 6, 9, -2, 3, 100], dtype=np.int16)
        out = BasePhysioDataset._sanitize_labels(labels)
        expected = np.array([0, -1, -1, -1, -1, 3, -1], dtype=np.int16)
        assert np.array_equal(out, expected), f"got {out}"
        report("_sanitize_labels: out-of-range -> -1", True)
    except Exception as exc:
        report("_sanitize_labels: out-of-range -> -1", False, str(exc))


def test_sanitize_empty_array():
    try:
        labels = np.array([], dtype=np.int16)
        out = BasePhysioDataset._sanitize_labels(labels)
        assert out.shape == (0,), f"got {out.shape}"
        report("_sanitize_labels: empty array", True)
    except Exception as exc:
        report("_sanitize_labels: empty array", False, str(exc))


def test_sanitize_preserves_dtype():
    try:
        labels = np.array([0, 5, -1], dtype=np.int16)
        out = BasePhysioDataset._sanitize_labels(labels)
        assert out.dtype == np.int16, f"got dtype={out.dtype}"
        report("_sanitize_labels: preserves int16 dtype", True)
    except Exception as exc:
        report("_sanitize_labels: preserves int16 dtype", False, str(exc))


def test_sanitize_accepts_larger_int_dtype():
    try:
        labels = np.array([0, 5, -1, 2, 9], dtype=np.int64)
        out = BasePhysioDataset._sanitize_labels(labels)
        assert out.dtype == np.int16  # coerced to int16
        assert np.array_equal(out, np.array([0, -1, -1, 2, -1], dtype=np.int16))
        report("_sanitize_labels: int64 input coerced", True)
    except Exception as exc:
        report("_sanitize_labels: int64 input coerced", False, str(exc))


# ---------------------------------------------------------------------------
# _safe_slice
# ---------------------------------------------------------------------------

def test_safe_slice_in_range():
    try:
        arr = np.arange(10)
        out = BasePhysioDataset._safe_slice(arr, 2, 7, fill_value=-1)
        assert np.array_equal(out, np.arange(2, 7)), f"got {out}"
        report("_safe_slice: in-range no padding", True)
    except Exception as exc:
        report("_safe_slice: in-range no padding", False, str(exc))


def test_safe_slice_past_end_pads():
    try:
        arr = np.array([10, 20, 30, 40, 50])
        # request [3:8], source has only 5 elements -> 2 taken, 3 padded
        out = BasePhysioDataset._safe_slice(arr, 3, 8, fill_value=-1)
        assert out.shape == (5,)
        expected = np.array([40, 50, -1, -1, -1], dtype=arr.dtype)
        assert np.array_equal(out, expected), f"got {out}"
        report("_safe_slice: past-end pads with fill_value", True)
    except Exception as exc:
        report("_safe_slice: past-end pads with fill_value", False, str(exc))


def test_safe_slice_2d():
    try:
        arr = np.zeros((3, 4))
        out = BasePhysioDataset._safe_slice(arr, 1, 5, fill_value=-1)
        assert out.shape == (4, 4), f"got {out.shape}"
        # Rows 0-1 are arr[1:3] (zeros), rows 2-3 are pad (-1)
        assert (out[:2] == 0).all()
        assert (out[2:] == -1).all()
        report("_safe_slice: 2D trailing dims preserved", True)
    except Exception as exc:
        report("_safe_slice: 2D trailing dims preserved", False, str(exc))


def test_safe_slice_length_zero_raises():
    try:
        try:
            BasePhysioDataset._safe_slice(np.arange(5), 3, 3, fill_value=0)
            report("_safe_slice: start==end raises", False, "no error raised")
        except ValueError:
            report("_safe_slice: start==end raises", True)
    except Exception as exc:
        report("_safe_slice: start==end raises", False, str(exc))


# ---------------------------------------------------------------------------
# Integration: label-signal alignment with synthetic corrupt EDF
# ---------------------------------------------------------------------------

class _FakeEDFDataset(BasePhysioDataset):
    """Minimal dataset for controlled label/signal alignment testing."""

    DATASET_NAME = "hardening_fake"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0
    CHANNEL_PREFERENCES = {
        "EEG": [("C4", "M2"), "C4-M2", "EEG"],
        "EOG": ["EOG"],
        "EMG": ["EMG"],
    }

    def __init__(self, root, subject_id, labels_override=None, **kwargs):
        self._subj = subject_id
        self._labels_override = labels_override
        super().__init__(root=root, **kwargs)

    def _list_subjects(self):
        root = Path(self.root)
        return [SubjectSpec(
            subject_id=self._subj,
            edf_path=root / f"{self._subj}.edf",
            label_path=root / f"{self._subj}_scoring.edf",
        )]

    def _read_subject_labels(self, spec):
        if self._labels_override is not None:
            return np.asarray(self._labels_override, dtype=np.int16)
        # Default: parse the annotation EDF we wrote
        with pyedflib.EdfReader(str(spec.label_path)) as f:
            onsets, durations, stages = f.readAnnotations()
        if len(stages) == 0:
            return np.array([], dtype=np.int16)
        smap = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
        total = max(float(o) + float(d) for o, d in zip(onsets, durations))
        n = int(total // self.epoch_length_sec)
        out = np.full(n, -1, dtype=np.int16)
        for o, d, s in zip(onsets, durations, stages):
            i0 = int(round(float(o) / self.epoch_length_sec))
            i1 = int(round((float(o) + float(d)) / self.epoch_length_sec))
            if i1 > n:
                i1 = n
            out[i0:i1] = smap.get(str(s).strip(), -1)
        return out


def test_labels_shorter_than_signal_padded_with_minus_one():
    """Signal has 10 epochs (300s at 30s/epoch) but labels cover only 5."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(d / "H1.edf", n_channels=3, duration_sec=300,
                       channel_names=["C4-M2", "EOG", "EMG"])
        # Only 5 scored epochs
        write_fake_annotations_edf(
            d / "H1_scoring.edf", stages=["W", "N1", "N2", "N3", "R"],
        )
        try:
            ds = _FakeEDFDataset(
                root=str(d), subject_id="H1",
                channels=["EEG", "EOG", "EMG"],
                pipelines="raw", sequence_length=10, cache_dir=c,
            )
            assert len(ds) == 1, f"expected 1 sample, got {len(ds)}"
            item = ds[0]
            labels = item["labels"].numpy()
            assert labels.shape == (10,), f"got {labels.shape}"
            # First 5 = scored stages, last 5 = -1
            assert tuple(labels[:5]) == (0, 1, 2, 3, 4), f"got {labels[:5]}"
            assert (labels[5:] == -1).all(), f"got {labels[5:]}"
            # Signal tensors still have 10 epochs
            for ch in item["channel_order"]:
                assert item["signals"][ch].shape[0] == 10
            report("integration: labels padded with -1 when shorter than signal", True)
        except Exception as exc:
            report("integration: labels padded with -1 when shorter than signal",
                   False, str(exc))


def test_labels_longer_than_signal_handled():
    """Labels cover more epochs than the signal does: safe slice pads signal with 0."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        # Short EDF: 150s = 5 epochs only
        write_fake_edf(d / "H2.edf", n_channels=3, duration_sec=150,
                       channel_names=["C4-M2", "EOG", "EMG"])
        # But label array claims 10 epochs
        write_fake_annotations_edf(
            d / "H2_scoring.edf",
            stages=["W", "N1", "N2", "N3", "R", "W", "N1", "N2", "N3", "R"],
        )
        try:
            ds = _FakeEDFDataset(
                root=str(d), subject_id="H2",
                channels=["EEG"],
                pipelines="raw", sequence_length=10, cache_dir=c,
            )
            assert len(ds) == 1
            item = ds[0]
            assert item["labels"].shape == (10,)
            for ch in item["channel_order"]:
                assert item["signals"][ch].shape[0] == 10
            # Trailing 5 signal positions = pad = zero (float32 cast of 0.0)
            for ch in item["channel_order"]:
                tail = item["signals"][ch][5:].numpy()
                assert np.allclose(tail, 0.0), f"tail not zero-padded: {tail.mean()}"
            report("integration: signal zero-padded when shorter than labels", True)
        except Exception as exc:
            report("integration: signal zero-padded when shorter than labels",
                   False, str(exc))


def test_invalid_labels_replaced_with_minus_one():
    """Parser emits out-of-range stages (e.g. 5, 99) -> sanitized to -1."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(d / "H3.edf", n_channels=3, duration_sec=150,
                       channel_names=["C4-M2", "EOG", "EMG"])
        write_fake_annotations_edf(
            d / "H3_scoring.edf", stages=["W"] * 5,
        )
        # Inject invalid label values via override
        override = np.array([0, 5, 2, 99, -7], dtype=np.int16)
        try:
            ds = _FakeEDFDataset(
                root=str(d), subject_id="H3",
                channels=["EEG"],
                pipelines="raw", sequence_length=5, cache_dir=c,
                labels_override=override,
            )
            item = ds[0]
            labels = item["labels"].numpy()
            expected = np.array([0, -1, 2, -1, -1], dtype=np.int64)
            assert np.array_equal(labels, expected), f"got {labels}, expected {expected}"
            report("integration: out-of-range labels sanitized to -1", True)
        except Exception as exc:
            report("integration: out-of-range labels sanitized to -1",
                   False, str(exc))


def test_no_epochs_dropped_when_half_unscored():
    """Every physical epoch is indexed; unscored ones just have label=-1."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        # 600s = 20 epochs
        write_fake_edf(d / "H4.edf", n_channels=3, duration_sec=600,
                       channel_names=["C4-M2", "EOG", "EMG"])
        # Only 10 scored
        write_fake_annotations_edf(
            d / "H4_scoring.edf", stages=["W", "N1", "N2", "N3", "R"] * 2,
        )
        try:
            ds = _FakeEDFDataset(
                root=str(d), subject_id="H4",
                channels=["EEG"],
                pipelines="raw", sequence_length=20, cache_dir=c,
            )
            # Sequence mode with L=20 means exactly one flat index covering all 20
            assert len(ds) == 1
            item = ds[0]
            assert item["labels"].shape == (20,)
            # First 10 scored, last 10 = -1 (not dropped)
            scored = item["labels"].numpy()[:10]
            unscored = item["labels"].numpy()[10:]
            assert set(scored.tolist()) <= set(AASM_VALID_LABELS)
            assert (unscored == -1).all()
            report("integration: no epoch dropped; unscored get -1", True)
        except Exception as exc:
            report("integration: no epoch dropped; unscored get -1", False, str(exc))


def test_full_recording_mode_alignment():
    """In sequence_length=-1 mode, labels and signals agree on full-night length."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        # 900s = 30 epochs; only 15 scored
        write_fake_edf(d / "H5.edf", n_channels=3, duration_sec=900,
                       channel_names=["C4-M2", "EOG", "EMG"])
        write_fake_annotations_edf(
            d / "H5_scoring.edf", stages=["W", "N1", "N2"] * 5,
        )
        try:
            ds = _FakeEDFDataset(
                root=str(d), subject_id="H5",
                channels=["EEG"],
                pipelines="raw", sequence_length=-1, cache_dir=c,
            )
            item = ds[0]
            expected_n = 30
            assert item["labels"].shape[0] == expected_n
            for ch in item["channel_order"]:
                assert item["signals"][ch].shape[0] == expected_n
            # First 15 scored, last 15 = -1
            assert (item["labels"][:15] >= 0).all()
            assert (item["labels"][15:] == -1).all()
            report("integration: sequence_length=-1 full-night alignment", True)
        except Exception as exc:
            report("integration: sequence_length=-1 full-night alignment",
                   False, str(exc))


# ---------------------------------------------------------------------------
# SleepEDF stage map: R&K (W,1,2,3,4,R) -> AASM 5-class (W,N1,N2,N3,REM)
# ---------------------------------------------------------------------------

def test_sleepedf_stage_map_is_aasm_5class():
    try:
        from physioex.data.datasets.sleepedf import SLEEPEDF_STAGE_MAP
        values = set(SLEEPEDF_STAGE_MAP.values())
        assert values <= set(AASM_VALID_LABELS), (
            f"SleepEDF stage map emits non-AASM values: {values - set(AASM_VALID_LABELS)}"
        )
        # N4 must map to N3 (AASM collapses the two)
        assert SLEEPEDF_STAGE_MAP["Sleep stage 4"] == 3, (
            "SleepEDF 'Sleep stage 4' must map to N3 (AASM convention)"
        )
        # Verify all canonical R&K names are present
        for name in ("Sleep stage W", "Sleep stage 1", "Sleep stage 2",
                     "Sleep stage 3", "Sleep stage 4", "Sleep stage R",
                     "Sleep stage ?", "Movement time"):
            assert name in SLEEPEDF_STAGE_MAP, f"missing: {name}"
        report("SleepEDF stage map: R&K -> AASM 5-class", True)
    except Exception as exc:
        report("SleepEDF stage map: R&K -> AASM 5-class", False, str(exc))


def test_all_dataset_stage_maps_in_aasm_range():
    """For each dataset module that exports a stage map, values must be AASM-valid."""
    from physioex.data.readers.annotations import NSRR_STAGE_MAP
    from physioex.data.datasets.sleepedf import SLEEPEDF_STAGE_MAP
    from physioex.data.datasets.dcsm import DCSM_STAGE_MAP
    from physioex.data.datasets.hmc import HMC_STAGE_MAP

    maps = {
        "NSRR_STAGE_MAP": NSRR_STAGE_MAP,
        "SLEEPEDF_STAGE_MAP": SLEEPEDF_STAGE_MAP,
        "DCSM_STAGE_MAP": DCSM_STAGE_MAP,
        "HMC_STAGE_MAP": HMC_STAGE_MAP,
    }

    allgood = True
    for name, m in maps.items():
        vals = set(m.values())
        bad = vals - set(AASM_VALID_LABELS)
        if bad:
            allgood = False
            print(f"  !! {name} emits non-AASM values: {bad}")
    report("all dataset stage maps emit AASM-valid values", allgood)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

