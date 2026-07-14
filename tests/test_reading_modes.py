"""Tests for the two reading modes: fixed-length sequences vs full recordings."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pyedflib

from physioex.data.base import BasePhysioDataset, SubjectSpec

# Reuse the fake EDF helpers by importing them from the integration test
from tests.factories.edf import (
    write_fake_edf, write_fake_annotations_edf, FakeEDFDataset,
)

def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


def test_sequence_mode_length():
    """sequence_length=L -> __len__ = sum over subjects of (n_epochs - L + 1)."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        # 20 epochs (600s at 30s per epoch)
        write_fake_edf(d / "S1.edf", duration_sec=600, channel_names=["C4-M2", "EOG", "EMG"],
                       n_channels=3)
        write_fake_annotations_edf(d / "S1_sleepscoring.edf", stages=["W"] * 20)

        ds = FakeEDFDataset(
            root=str(d), subject_id="S1",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw", sequence_length=5, cache_dir=c,
        )
        assert len(ds) == 20 - 5 + 1, f"expected 16, got {len(ds)}"
        report("sequence_length=5, 20 epochs -> __len__==16", True)


def test_recording_mode_length():
    """sequence_length=-1 -> __len__ = n_subjects."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(d / "S2.edf", duration_sec=300, channel_names=["C4-M2", "EOG", "EMG"],
                       n_channels=3)
        write_fake_annotations_edf(d / "S2_sleepscoring.edf", stages=["W"] * 10)

        ds = FakeEDFDataset(
            root=str(d), subject_id="S2",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw", sequence_length=-1, cache_dir=c,
        )
        assert len(ds) == 1, f"expected 1 (one subject), got {len(ds)}"
        report("sequence_length=-1, 1 subject -> __len__==1", True)


def test_sequence_mode_shape():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(d / "S3.edf", duration_sec=600, channel_names=["C4-M2", "EOG", "EMG"],
                       n_channels=3)
        write_fake_annotations_edf(d / "S3_sleepscoring.edf", stages=["W"] * 20)

        ds = FakeEDFDataset(
            root=str(d), subject_id="S3",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw", sequence_length=7, cache_dir=c,
        )
        item = ds[0]
        # signals: each channel has shape (seq_len, samples_per_epoch)
        for key in item["channel_order"]:
            assert item["signals"][key].shape[0] == 7, f"channel {key} shape[0]={item['signals'][key].shape[0]}"
        assert item["labels"].shape == (7,)
        assert item["epoch_indices"].shape == (7,)
        report("sequence mode item shapes", True)


def test_recording_mode_shape():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(d / "S4.edf", duration_sec=900, channel_names=["C4-M2", "EOG", "EMG"],
                       n_channels=3)
        write_fake_annotations_edf(d / "S4_sleepscoring.edf", stages=["W"] * 30)

        ds = FakeEDFDataset(
            root=str(d), subject_id="S4",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw", sequence_length=-1, cache_dir=c,
        )
        item = ds[0]
        # full recording -> shape[0] == n_epochs (30)
        for key in item["channel_order"]:
            assert item["signals"][key].shape[0] == 30, f"channel {key} shape[0]={item['signals'][key].shape[0]}"
        assert item["labels"].shape == (30,)
        assert item["recording_length"] == 30
        report("recording mode item shapes", True)


def test_different_subjects_different_lengths_recording_mode():
    """In recording mode, each subject yields its own length (no padding in the Dataset)."""
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as c:
        d = Path(d)
        write_fake_edf(d / "SA.edf", duration_sec=600, channel_names=["C4-M2", "EOG", "EMG"],
                       n_channels=3)
        write_fake_annotations_edf(d / "SA_sleepscoring.edf", stages=["W"] * 20)

        # Two separate datasets, pointing at the same root but different subject_ids,
        # both in recording mode:
        for sid, duration in [("SA", 600)]:
            ds = FakeEDFDataset(
                root=str(d), subject_id=sid,
                channels=["EEG", "EOG", "EMG"],
                pipelines="raw", sequence_length=-1, cache_dir=c,
            )
            item = ds[0]
            expected = int(duration // 30)
            for key in item["channel_order"]:
                assert item["signals"][key].shape[0] == expected
        report("recording mode yields subject-specific length", True)


