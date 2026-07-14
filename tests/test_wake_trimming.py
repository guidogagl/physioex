"""Tests for pre-/post-sleep wake trimming.

Verifies that ``BasePhysioDataset._trim_excess_wake`` keeps up to 30 minutes
(60 epochs at 30s) of wake before the first non-wake epoch and after the last
non-wake epoch, marking everything beyond that as -1.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec

from tests.factories.edf import (
    write_fake_edf,
    write_fake_annotations_edf,
)

def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


# ---------------------------------------------------------------------------
# Unit tests on _trim_excess_wake (pure function, default keep_minutes=30)
# ---------------------------------------------------------------------------

KEEP = 60  # 30 min / 30s epoch = 60 epochs kept at each boundary


def test_trim_empty_noop():
    out = BasePhysioDataset._trim_excess_wake(np.array([], dtype=np.int16))
    report("trim: empty array no-op", out.shape == (0,))


def test_trim_short_wake_noop():
    """If pre/post wake is shorter than 60 epochs, nothing is trimmed."""
    labels = np.concatenate(
        [
            np.full(50, 0, dtype=np.int16),  # 50 wake (< 60 keep)
            np.full(10, 2, dtype=np.int16),  # 10 N2
            np.full(30, 0, dtype=np.int16),  # 30 wake (< 60 keep)
        ]
    )
    out = BasePhysioDataset._trim_excess_wake(labels)
    report("trim: short wake untouched", np.array_equal(out, labels))


def test_trim_long_evening_wake():
    """100 wake before sleep -> keep 60, mark first 40 as -1."""
    labels = np.concatenate(
        [
            np.full(100, 0, dtype=np.int16),  # 100 evening wake
            np.full(10, 2, dtype=np.int16),  # 10 N2
        ]
    )
    out = BasePhysioDataset._trim_excess_wake(labels)
    # first_sleep = 100, keep from max(0, 100-60)=40
    assert (out[:40] == -1).all(), f"leading not trimmed: {out[:40]}"
    assert (out[40:100] == 0).all(), f"kept wake changed: {out[40:100]}"
    assert (out[100:] == 2).all(), f"sleep changed: {out[100:]}"
    report("trim: long evening wake", True)


def test_trim_long_morning_wake():
    """100 wake after sleep -> keep 60, mark last 40 as -1."""
    labels = np.concatenate(
        [
            np.full(10, 2, dtype=np.int16),  # 10 N2
            np.full(100, 0, dtype=np.int16),  # 100 morning wake
        ]
    )
    out = BasePhysioDataset._trim_excess_wake(labels)
    # last_sleep = 9, trim_end = min(110, 9+1+60) = 70
    assert (out[:10] == 2).all(), f"sleep changed: {out[:10]}"
    assert (out[10:70] == 0).all(), f"kept wake changed: {out[10:70]}"
    assert (out[70:] == -1).all(), f"trailing not trimmed: {out[70:]}"
    report("trim: long morning wake", True)


def test_trim_both_sides():
    """Long wake on both sides -> trim both independently."""
    labels = np.concatenate(
        [
            np.full(80, 0, dtype=np.int16),  # 80 evening wake
            np.full(5, 2, dtype=np.int16),  # 5 N2
            np.full(90, 0, dtype=np.int16),  # 90 morning wake
        ]
    )
    out = BasePhysioDataset._trim_excess_wake(labels)
    # first_sleep=80, trim_start=max(0,80-60)=20 -> [0:20] = -1
    # last_sleep=84, trim_end=min(175, 84+1+60)=145 -> [145:175] = -1
    assert (out[:20] == -1).all(), f"evening not trimmed: {out[:20]}"
    assert (out[20:80] == 0).all(), f"kept evening wake: {out[20:80]}"
    assert (out[80:85] == 2).all(), f"sleep changed: {out[80:85]}"
    assert (out[85:145] == 0).all(), f"kept morning wake: {out[85:145]}"
    assert (out[145:] == -1).all(), f"morning not trimmed: {out[145:]}"
    report("trim: both sides", True)


def test_trim_all_wake_noop():
    """All wake, no sleep epochs -> nothing to anchor, no trim."""
    labels = np.full(200, 0, dtype=np.int16)
    out = BasePhysioDataset._trim_excess_wake(labels)
    report("trim: all wake no-op", np.array_equal(out, labels))


def test_trim_preserves_mid_wake():
    """Wake inside the sleep period (fragmented awakenings) is never trimmed."""
    labels = np.concatenate(
        [
            np.full(80, 0, dtype=np.int16),  # 80 evening wake
            np.full(5, 2, dtype=np.int16),  # N2
            np.full(50, 0, dtype=np.int16),  # 50 mid-sleep wake
            np.full(5, 3, dtype=np.int16),  # N3
            np.full(10, 0, dtype=np.int16),  # 10 morning wake (< 60 keep)
        ]
    )
    out = BasePhysioDataset._trim_excess_wake(labels)
    # first_sleep=80, trim_start=20 -> [0:20]=-1
    # last_sleep=139, trim_end=min(150,140+60)=150 -> no trailing trim
    assert (out[:20] == -1).all(), f"evening trim: {out[:20]}"
    assert (out[20:80] == 0).all(), f"kept evening: {out[20:80]}"
    assert (out[85:135] == 0).all(), f"mid-sleep wake changed: {out[85:135]}"
    report("trim: mid-sleep wake preserved", True)


def test_trim_preserves_existing_neg1():
    """Existing -1 labels are not counted as wake and stay -1."""
    labels = np.array(
        [-1, -1, 0, 0, 0, 2, 2, 2, 0, 0, 0],
        dtype=np.int16,
    )
    out = BasePhysioDataset._trim_excess_wake(labels)
    # first_sleep=5, 5-60<0 -> no leading trim
    # last_sleep=7, 7+1+60>11 -> no trailing trim
    # All short -> no trimming at all
    assert out[0] == -1 and out[1] == -1, "existing -1 preserved"
    assert (out[2:5] == 0).all(), "short pre-wake kept"
    report("trim: preserves pre-existing -1 labels", True)


def test_trim_custom_keep_minutes():
    """Custom keep_minutes parameter."""
    labels = np.concatenate(
        [
            np.full(20, 0, dtype=np.int16),  # 20 evening wake
            np.full(5, 2, dtype=np.int16),  # 5 N2
        ]
    )
    # keep_minutes=5 -> keep 10 epochs (5min * 60s / 30s)
    out = BasePhysioDataset._trim_excess_wake(labels, keep_minutes=5.0)
    # first_sleep=20, trim_start=max(0,20-10)=10
    assert (out[:10] == -1).all(), f"custom trim: {out[:10]}"
    assert (out[10:20] == 0).all(), f"custom kept: {out[10:20]}"
    report("trim: custom keep_minutes=5", True)


def test_trim_idempotent():
    """Trimming twice gives the same result."""
    labels = np.concatenate(
        [
            np.full(100, 0, dtype=np.int16),
            np.full(5, 2, dtype=np.int16),
            np.full(100, 0, dtype=np.int16),
        ]
    )
    once = BasePhysioDataset._trim_excess_wake(labels)
    twice = BasePhysioDataset._trim_excess_wake(once)
    report("trim: idempotent", np.array_equal(once, twice))


# ---------------------------------------------------------------------------
# Integration: BasePhysioDataset with trim_excess_wake flag
# ---------------------------------------------------------------------------


class _FakeDS(BasePhysioDataset):
    DATASET_NAME = "wake_trim_test"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0
    CHANNEL_PREFERENCES = {
        "EEG": [("C4", "M2"), "C4-M2", "EEG"],
        "EOG": ["EOG"],
        "EMG": ["EMG"],
    }

    def __init__(self, root, subj, override_labels, **kwargs):
        self._subj = subj
        self._override = np.asarray(override_labels, dtype=np.int16)
        super().__init__(root=root, **kwargs)

    def _list_subjects(self):
        return [
            SubjectSpec(
                subject_id=self._subj,
                edf_path=Path(self.root) / f"{self._subj}.edf",
                label_path=Path(self.root) / f"{self._subj}_scoring.edf",
            )
        ]

    def _read_subject_labels(self, spec):
        return self._override


def test_dataset_default_trims():
    """Default trim_excess_wake=True trims excess pre/post wake."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        n_epochs = 200
        write_fake_edf(
            d / "T1.edf",
            n_channels=3,
            duration_sec=n_epochs * 30,
            channel_names=["C4-M2", "EOG", "EMG"],
        )
        write_fake_annotations_edf(d / "T1_scoring.edf", stages=["W"] * n_epochs)

        # 100 wake + 10 N2 + 90 wake
        lbls = np.concatenate(
            [
                np.full(100, 0, dtype=np.int16),
                np.full(10, 2, dtype=np.int16),
                np.full(90, 0, dtype=np.int16),
            ]
        )
        ds = _FakeDS(
            root=str(d),
            subj="T1",
            override_labels=lbls,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=-1,
            cache_dir=c,
            trim_excess_wake=True,
        )
        item = ds[0]
        labels = item["labels"].numpy()
        # first_sleep=100, trim_start=40 -> [0:40]=-1
        # last_sleep=109, trim_end=170 -> [170:200]=-1
        assert (labels[:40] == -1).all(), f"evening not trimmed: {labels[:40]}"
        assert (labels[40:100] == 0).all(), f"kept evening: {labels[40:100]}"
        assert (labels[100:110] == 2).all(), f"sleep: {labels[100:110]}"
        assert (labels[110:170] == 0).all(), f"kept morning: {labels[110:170]}"
        assert (labels[170:] == -1).all(), f"morning not trimmed: {labels[170:]}"
        report("integration: trim_excess_wake=True applies 30min window", True)


def test_dataset_flag_disabled():
    """trim_excess_wake=False leaves labels untouched."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(
            d / "T2.edf",
            n_channels=3,
            duration_sec=30 * 30,
            channel_names=["C4-M2", "EOG", "EMG"],
        )
        write_fake_annotations_edf(d / "T2_scoring.edf", stages=["W"] * 30)

        lbls = np.concatenate(
            [
                np.full(15, 0, dtype=np.int16),
                np.full(5, 2, dtype=np.int16),
                np.full(10, 0, dtype=np.int16),
            ]
        )
        ds = _FakeDS(
            root=str(d),
            subj="T2",
            override_labels=lbls,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=30,
            cache_dir=c,
            trim_excess_wake=False,
        )
        item = ds[0]
        labels = item["labels"].numpy()
        assert np.array_equal(labels.astype(np.int16), lbls)
        report("integration: trim_excess_wake=False leaves labels untouched", True)


def test_dataset_length_preserved():
    """After trim, the number of indexed epochs must NOT change (no drop)."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        n_epochs = 200
        write_fake_edf(
            d / "T3.edf",
            n_channels=3,
            duration_sec=n_epochs * 30,
            channel_names=["C4-M2", "EOG", "EMG"],
        )
        write_fake_annotations_edf(d / "T3_scoring.edf", stages=["W"] * n_epochs)

        lbls = np.concatenate(
            [
                np.full(100, 0, dtype=np.int16),
                np.full(5, 2, dtype=np.int16),
                np.full(95, 0, dtype=np.int16),
            ]
        )
        ds = _FakeDS(
            root=str(d),
            subj="T3",
            override_labels=lbls,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=-1,
            cache_dir=c,
            trim_excess_wake=True,
        )
        item = ds[0]
        assert item["labels"].shape[0] == n_epochs
        for ch in item["channel_order"]:
            assert item["signals"][ch].shape[0] == n_epochs
        assert (item["labels"] == -1).sum().item() > 0
        report("integration: trim preserves total epoch count", True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

