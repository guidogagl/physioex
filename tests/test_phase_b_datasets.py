"""Phase B integration tests: SleepEDF and DCSM dataset modules.

Creates synthetic fake EDF/annotation files on-the-fly and verifies discovery,
channel resolution, label parsing, and end-to-end dict return for both datasets.

Run standalone:
    python test/tests/test_phase_b_datasets.py
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import pyedflib

from physioex.data.datasets import get_dataset, available_datasets
from physioex.data.datasets.sleepedf import SleepEDFDataset, SLEEPEDF_STAGE_MAP
from physioex.data.datasets.dcsm import DCSMDataset, DCSM_STAGE_MAP
from tests.test_raw_dataset_integration import write_fake_edf, write_fake_annotations_edf


passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


# ===================================================================
# Helpers
# ===================================================================

def make_sleepedf_subject(
    root: Path,
    psg_name: str,
    hyp_name: str,
    stages: list,
    epoch_sec: float = 30.0,
    channel_names=None,
    fs_per_channel=None,
):
    """Create a fake SleepEDF PSG + hypnogram pair in ``root``."""
    if channel_names is None:
        channel_names = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
    if fs_per_channel is None:
        fs_per_channel = [100, 100, 100]
    n_channels = len(channel_names)
    duration_sec = len(stages) * epoch_sec

    write_fake_edf(
        root / psg_name,
        n_channels=n_channels,
        duration_sec=duration_sec,
        fs_per_channel=fs_per_channel,
        channel_names=channel_names,
    )
    write_fake_annotations_edf(
        root / hyp_name,
        stages=stages,
        epoch_sec=epoch_sec,
    )


def make_dcsm_subject(
    root: Path,
    subj_name: str,
    hyp_rows: list,
    channel_names=None,
    fs_per_channel=None,
    duration_sec: float = None,
):
    """Create a fake DCSM subject directory with psg.edf + hypnogram.ids."""
    if channel_names is None:
        channel_names = ["C3-M2", "E1-M2", "E2-M2", "CHIN", "ECG-II"]
    if fs_per_channel is None:
        fs_per_channel = [256] * len(channel_names)
    # Infer duration from hypnogram if not given
    if duration_sec is None:
        max_end = max(int(r.split()[0]) + int(r.split()[1]) for r in hyp_rows)
        duration_sec = float(max_end)

    subj_dir = root / subj_name
    subj_dir.mkdir(parents=True, exist_ok=True)

    write_fake_edf(
        subj_dir / "psg.edf",
        n_channels=len(channel_names),
        duration_sec=duration_sec,
        fs_per_channel=fs_per_channel,
        channel_names=channel_names,
    )

    with open(subj_dir / "hypnogram.ids", "w") as f:
        for row in hyp_rows:
            f.write(row + "\n")


# ===================================================================
# SleepEDF tests
# ===================================================================

def test_sleepedf_registry():
    """1. get_dataset('sleepedf') returns SleepEDFDataset."""
    cls = get_dataset("sleepedf")
    ok = cls is SleepEDFDataset
    report("SleepEDF registry registration", ok, f"got {cls}")


def test_sleepedf_in_available():
    """Verify 'sleepedf' appears in available_datasets()."""
    ok = "sleepedf" in available_datasets()
    report("SleepEDF in available_datasets()", ok)


def test_sleepedf_empty_root():
    """2. Empty root returns no subjects (no crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = SleepEDFDataset(
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("SleepEDF empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


def test_sleepedf_subject_discovery():
    """3. Discovers a single subject from synthetic PSG + hypnogram."""
    stages = ["Sleep stage W"] * 5 + ["Sleep stage 2"] * 5  # 10 epochs = 300s
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_sleepedf_subject(
            data_dir,
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            stages=stages,
        )
        ds = SleepEDFDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = len(ds._subjects) == 1
        detail = f"subjects={[s.subject_id for s in ds._subjects]}"
        report("SleepEDF discovers 1 subject", ok, detail)


def test_sleepedf_subject_id():
    """Verify subject_id is derived from PSG stem (e.g. 'SC4001E0')."""
    stages = ["Sleep stage W"] * 10
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_sleepedf_subject(
            data_dir,
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            stages=stages,
        )
        ds = SleepEDFDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = ds._subjects[0].subject_id == "SC4001E0"
        report("SleepEDF subject_id == 'SC4001E0'", ok,
               f"got {ds._subjects[0].subject_id!r}")


def test_sleepedf_channel_resolution():
    """4. EEG request resolves to 'EEG Fpz-Cz'."""
    stages = ["Sleep stage W"] * 10
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_sleepedf_subject(
            data_dir,
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            stages=stages,
        )
        ds = SleepEDFDataset(
            root=str(data_dir),
            channels=["EEG", "EOG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        resolved = ds._resolved[ds._subjects[0].subject_id]
        eeg_physical = resolved[0].physical
        ok = eeg_physical == "EEG Fpz-Cz"
        report("SleepEDF EEG resolves to 'EEG Fpz-Cz'", ok, f"got {eeg_physical!r}")


def test_sleepedf_label_parsing():
    """5. Labels from annotations match expected length + stage values."""
    stages = (
        ["Sleep stage W"] * 3
        + ["Sleep stage 1"] * 2
        + ["Sleep stage 2"] * 2
        + ["Sleep stage 3"] * 1
        + ["Sleep stage R"] * 2
    )  # 10 epochs
    expected = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_sleepedf_subject(
            data_dir,
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            stages=stages,
        )
        ds = SleepEDFDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        ok_len = len(labels) == 10
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report(
            "SleepEDF label parsing (length + values)",
            ok,
            f"len={len(labels)} expected=10, vals={list(labels)}",
        )


def test_sleepedf_end_to_end():
    """6. ds[0] returns a valid dict with expected keys and shapes."""
    stages = ["Sleep stage W"] * 5 + ["Sleep stage R"] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_sleepedf_subject(
            data_dir,
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            stages=stages,
        )
        ds = SleepEDFDataset(
            root=str(data_dir),
            channels=["EEG", "EOG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        item = ds[0]
        checks = [
            "signals" in item,
            "labels" in item,
            "subject" in item,
            "channel_order" in item,
            torch.is_tensor(item["labels"]),
            item["labels"].shape[0] == 5,
            len(item["channel_order"]) == 2,
            item["subject"]["dataset"] == "sleepedf",
        ]
        ok = all(checks)
        report("SleepEDF end-to-end ds[0] dict", ok, f"checks={checks}")


def test_sleepedf_two_subjects():
    """7. Two PSG/hyp pairs -> two distinct subjects discovered."""
    stages = ["Sleep stage W"] * 10
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_sleepedf_subject(
            data_dir,
            "SC4001E0-PSG.edf",
            "SC4001EC-Hypnogram.edf",
            stages=stages,
        )
        make_sleepedf_subject(
            data_dir,
            "SC4002E0-PSG.edf",
            "SC4002EH-Hypnogram.edf",
            stages=stages,
        )
        ds = SleepEDFDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ids = [s.subject_id for s in ds._subjects]
        ok = len(ids) == 2 and ids[0] != ids[1]
        report("SleepEDF discovers 2 subjects with distinct IDs", ok, f"ids={ids}")


# ===================================================================
# DCSM tests
# ===================================================================

def test_dcsm_registry():
    """8. get_dataset('dcsm') returns DCSMDataset."""
    cls = get_dataset("dcsm")
    ok = cls is DCSMDataset
    report("DCSM registry registration", ok, f"got {cls}")


def test_dcsm_in_available():
    """Verify 'dcsm' appears in available_datasets()."""
    ok = "dcsm" in available_datasets()
    report("DCSM in available_datasets()", ok)


def test_dcsm_empty_root():
    """9. Empty root returns no subjects."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = DCSMDataset(
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("DCSM empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


def test_dcsm_subject_discovery():
    """10. Discovers subject from per-subject directory layout."""
    hyp_rows = [
        "0 30 W",
        "30 30 N1",
        "60 60 N2",
        "120 30 N3",
        "150 30 REM",
        "180 30 W",
        "210 30 N1",
        "240 30 N2",
        "270 30 N3",
        "300 30 REM",  # total = 330s = 11 epochs
    ]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_dcsm_subject(data_dir, "subject_001", hyp_rows, duration_sec=330)
        ds = DCSMDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = len(ds._subjects) == 1
        detail = f"subjects={[s.subject_id for s in ds._subjects]}"
        report("DCSM discovers 1 subject", ok, detail)


def test_dcsm_subject_id():
    """Verify DCSM subject_id is the directory name."""
    hyp_rows = ["0 30 W", "30 30 N1", "60 60 N2", "120 30 N3", "150 150 W"]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_dcsm_subject(data_dir, "my_subject_42", hyp_rows)
        ds = DCSMDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = ds._subjects[0].subject_id == "my_subject_42"
        report("DCSM subject_id == dir name", ok,
               f"got {ds._subjects[0].subject_id!r}")


def test_dcsm_channel_resolution():
    """11. EEG request resolves to 'C3-M2'."""
    hyp_rows = ["0 30 W", "30 30 N1", "60 60 N2", "120 30 N3", "150 150 W"]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_dcsm_subject(data_dir, "subject_001", hyp_rows)
        ds = DCSMDataset(
            root=str(data_dir),
            channels=["EEG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        resolved = ds._resolved[ds._subjects[0].subject_id]
        eeg_physical = resolved[0].physical
        ok = eeg_physical == "C3-M2"
        report("DCSM EEG resolves to 'C3-M2'", ok, f"got {eeg_physical!r}")


def test_dcsm_label_parsing():
    """12. Labels from hypnogram.ids match expected values."""
    # 0-30 W, 30-60 N1, 60-120 N2, 120-150 N3 => 5 epochs
    hyp_rows = [
        "0 30 W",
        "30 30 N1",
        "60 60 N2",
        "120 30 N3",
    ]
    expected = [0, 1, 2, 2, 3]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_dcsm_subject(data_dir, "subject_001", hyp_rows, duration_sec=150)
        ds = DCSMDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        ok_len = len(labels) == 5
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report(
            "DCSM label parsing (length + values)",
            ok,
            f"len={len(labels)} expected=5, vals={list(labels)}",
        )


def test_dcsm_end_to_end():
    """13. ds[0] returns dict with expected keys."""
    hyp_rows = [
        "0 30 W",
        "30 30 N1",
        "60 60 N2",
        "120 30 N3",
        "150 30 REM",
        "180 30 W",
        "210 30 N1",
        "240 30 N2",
        "270 30 N3",
        "300 30 REM",
    ]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_dcsm_subject(data_dir, "subject_001", hyp_rows, duration_sec=330)
        ds = DCSMDataset(
            root=str(data_dir),
            channels=["EEG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        item = ds[0]
        checks = [
            "signals" in item,
            "labels" in item,
            "subject" in item,
            "channel_order" in item,
            torch.is_tensor(item["labels"]),
            item["labels"].shape[0] == 5,
            len(item["channel_order"]) == 2,
            item["subject"]["dataset"] == "dcsm",
        ]
        ok = all(checks)
        report("DCSM end-to-end ds[0] dict", ok, f"checks={checks}")


def test_dcsm_multiple_subjects():
    """Discover multiple DCSM subjects."""
    hyp_rows = ["0 30 W", "30 30 N1", "60 60 N2", "120 30 N3", "150 150 W"]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_dcsm_subject(data_dir, "subject_001", hyp_rows)
        make_dcsm_subject(data_dir, "subject_002", hyp_rows)
        make_dcsm_subject(data_dir, "subject_003", hyp_rows)
        ds = DCSMDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ids = sorted(s.subject_id for s in ds._subjects)
        ok = len(ids) == 3 and len(set(ids)) == 3
        report("DCSM discovers 3 subjects", ok, f"ids={ids}")


# ===================================================================
# Real-data smoke tests (guarded)
# ===================================================================

def run_real_data_tests():
    """Only runs if PHYSIOEX_TEST_REAL_DATA=1."""
    print("\n--- Real-data smoke tests ---")

    # SleepEDF real data
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = SleepEDFDataset(
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n > 50
            report("SleepEDF real data: >50 subjects", ok, f"got {n}")

            item = ds[0]
            ok = (
                "signals" in item
                and "labels" in item
                and torch.is_tensor(item["labels"])
            )
            report("SleepEDF real data: ds[0] returns valid dict", ok)

            # Check cache files appeared
            cache_root = Path(cache_dir) / "v1" / "sleepedf"
            ok = cache_root.exists()
            report("SleepEDF real data: cache dir created", ok)
    except Exception as e:
        report("SleepEDF real data", False, f"SKIP: {e}")

    # DCSM -- data is zipped only, so expect 0 subjects or skip
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = DCSMDataset(
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            # If data is extracted, expect many subjects; otherwise 0 is fine
            if n > 0:
                item = ds[0]
                ok = "signals" in item and "labels" in item
                report("DCSM real data: ds[0] returns valid dict", ok)
            else:
                report("DCSM real data", True, "SKIP: no extracted data (0 subjects)")
    except Exception as e:
        report("DCSM real data", False, f"SKIP: {e}")


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase B dataset integration tests")
    print("=" * 60)

    # SleepEDF synthetic tests
    print("\n--- SleepEDF (synthetic) ---")
    test_sleepedf_registry()
    test_sleepedf_in_available()
    test_sleepedf_empty_root()
    test_sleepedf_subject_discovery()
    test_sleepedf_subject_id()
    test_sleepedf_channel_resolution()
    test_sleepedf_label_parsing()
    test_sleepedf_end_to_end()
    test_sleepedf_two_subjects()

    # DCSM synthetic tests
    print("\n--- DCSM (synthetic) ---")
    test_dcsm_registry()
    test_dcsm_in_available()
    test_dcsm_empty_root()
    test_dcsm_subject_discovery()
    test_dcsm_subject_id()
    test_dcsm_channel_resolution()
    test_dcsm_label_parsing()
    test_dcsm_end_to_end()
    test_dcsm_multiple_subjects()

    # Real-data tests (optional, guarded by env var)
    if os.environ.get("PHYSIOEX_TEST_REAL_DATA", "") == "1":
        run_real_data_tests()
    else:
        print("\n--- Real-data smoke tests SKIPPED (set PHYSIOEX_TEST_REAL_DATA=1) ---")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
