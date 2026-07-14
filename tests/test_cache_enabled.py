"""Tests for the ``cache_enabled`` parameter on BasePhysioDataset.

Verifies that when ``cache_enabled=False``, no disk I/O occurs for headers,
labels, or signals, while data is still returned correctly.  Reuses the
``FakeEDFDataset`` helper from ``test_raw_dataset_integration.py``.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Reuse helpers from the existing integration test
from tests.factories.edf import (
    FakeEDFDataset,
    write_fake_edf,
    write_fake_annotations_edf,
)

def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


# -------------------------------------------------------------------
# 1. cache_enabled=True (default): cache files appear on disk
# -------------------------------------------------------------------

def test_cache_enabled_true_writes_to_disk():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB01.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB01_sleepscoring.edf",
                                    stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB01",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
            cache_enabled=True,  # explicit default
        )

        _ = ds[0]

        cache_root = Path(cache_dir) / "v1" / "fake_integration_test"
        header_file = cache_root / "headers" / "SUB01.json"
        labels_file = cache_root / "labels" / "SUB01.npy"
        signals_dir = cache_root / "signals" / "SUB01"

        ok = header_file.exists() and labels_file.exists() and signals_dir.exists()
        report("cache_enabled=True: cache files written to disk", ok,
               f"header={header_file.exists()}, labels={labels_file.exists()}, signals={signals_dir.exists()}")


# -------------------------------------------------------------------
# 2. cache_enabled=False: NO cache files on disk, data still returned
# -------------------------------------------------------------------

def test_cache_enabled_false_no_disk_io():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB02.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB02_sleepscoring.edf",
                                    stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB02",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
            cache_enabled=False,
        )

        item = ds[0]

        # Verify NO cache files on disk
        cache_root = Path(cache_dir) / "v1" / "fake_integration_test"
        header_file = cache_root / "headers" / "SUB02.json"
        labels_file = cache_root / "labels" / "SUB02.npy"
        signals_dir = cache_root / "signals" / "SUB02"

        no_header = not header_file.exists()
        no_labels = not labels_file.exists()
        no_signals = not signals_dir.exists()

        ok_no_disk = no_header and no_labels and no_signals
        report("cache_enabled=False: no cache files on disk", ok_no_disk,
               f"no_header={no_header}, no_labels={no_labels}, no_signals={no_signals}")

        # Verify data is still returned correctly
        ok_data = (
            "signals" in item
            and "labels" in item
            and item["labels"].shape == (5,)
            and len(item["channel_order"]) == 3
        )
        report("cache_enabled=False: data returned correctly", ok_data)


# -------------------------------------------------------------------
# 3. cache_enabled=False repeated access: still works
# -------------------------------------------------------------------

def test_cache_enabled_false_repeated_access():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB03.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB03_sleepscoring.edf",
                                    stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB03",
            channels=["EEG"],
            pipelines="raw",
            sequence_length=3,
            cache_dir=cache_dir,
            cache_enabled=False,
        )

        item1 = ds[0]
        item2 = ds[0]  # second access -- labels from in-memory cache, signal re-computed
        item3 = ds[1]  # different index

        # All accesses should produce valid results
        ok = (
            item1["labels"].shape == (3,)
            and item2["labels"].shape == (3,)
            and item3["labels"].shape == (3,)
        )
        # Labels should match across repeated access of same index (in-memory cache)
        labels_match = torch.equal(item1["labels"], item2["labels"])

        report("cache_enabled=False: repeated access works", ok and labels_match,
               f"shapes_ok={ok}, labels_match={labels_match}")


# -------------------------------------------------------------------
# 4. Both modes produce same data (within float tolerance)
# -------------------------------------------------------------------

def test_both_modes_produce_same_data():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir_on, \
         tempfile.TemporaryDirectory() as cache_dir_off:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB04.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB04_sleepscoring.edf",
                                    stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds_on = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB04",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=str(cache_dir_on),
            cache_enabled=True,
        )

        ds_off = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB04",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=str(cache_dir_off),
            cache_enabled=False,
        )

        item_on = ds_on[0]
        item_off = ds_off[0]

        # Labels must be identical
        labels_eq = torch.equal(item_on["labels"], item_off["labels"])

        # Signals must be close (cache may use bfloat16 vs float32, so allow tolerance)
        signals_close = True
        for key in item_on["channel_order"]:
            sig_on = item_on["signals"][key].float()
            sig_off = item_off["signals"][key].float()
            if not torch.allclose(sig_on, sig_off, atol=1e-2, rtol=1e-2):
                signals_close = False
                max_diff = (sig_on - sig_off).abs().max().item()
                report(f"signal mismatch on channel {key}", False,
                       f"max_diff={max_diff}")

        # Channel order must match
        order_eq = item_on["channel_order"] == item_off["channel_order"]

        ok = labels_eq and signals_close and order_eq
        report("cache_enabled=True and False produce same data", ok,
               f"labels_eq={labels_eq}, signals_close={signals_close}, order_eq={order_eq}")


# ----- runner -----

