"""WSC (Wisconsin Sleep Cohort) dataset tests.

Synthetic tests create fake EDF + .stg.txt files on-the-fly and verify
discovery, label parsing, visit filtering, and end-to-end dict return.

Real-data smoke tests are guarded by PHYSIOEX_TEST_REAL_DATA=1.

Run standalone:
    python test/tests/test_wsc_dataset.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from physioex.data.datasets import get_dataset, available_datasets
from physioex.data.datasets.wsc import WSCDataset, _STG_MAP
from tests.factories.edf import write_fake_edf


def report(name, ok, detail=""):
    """Thin assert shim: fail the test with a descriptive message."""
    assert ok, f"{name}{(' -- ' + detail) if detail else ''}"


# ===================================================================
# Helpers
# ===================================================================

def make_wsc_subject(
    root: Path,
    visit: int,
    subject_num: str,
    stg_lines: list,
    channel_names=None,
    duration_sec: float = 150.0,
):
    """Create a fake WSC subject: EDF + .stg.txt annotation file.

    Args:
        root: directory that will contain ``polysomnography/``.
        visit: visit number (1-5).
        subject_num: e.g. "12345".
        stg_lines: list of tab-separated lines for the .stg.txt body
                   (excluding the header).
        channel_names: EDF channel labels.
        duration_sec: EDF duration in seconds.
    """
    if channel_names is None:
        channel_names = ["C3_M2", "E1", "E2", "chin"]
    poly_dir = root / "polysomnography"
    poly_dir.mkdir(parents=True, exist_ok=True)

    base = f"wsc-visit{visit}-{subject_num}-nsrr"
    edf_path = poly_dir / f"{base}.edf"
    stg_path = poly_dir / f"{base}.stg.txt"

    write_fake_edf(
        edf_path,
        n_channels=len(channel_names),
        duration_sec=duration_sec,
        channel_names=channel_names,
    )

    with open(stg_path, "w") as f:
        f.write("Epoch\tUser-Defined Stage\tCAST-Defined Stage\n")
        for line in stg_lines:
            f.write(line + "\n")


# ===================================================================
# 1. Registry
# ===================================================================

def test_registry():
    """get_dataset('wsc') returns WSCDataset."""
    cls = get_dataset("wsc")
    ok = cls is WSCDataset
    report("WSC registry registration", ok, f"got {cls}")


def test_in_available():
    """'wsc' appears in available_datasets()."""
    ok = "wsc" in available_datasets()
    report("WSC in available_datasets()", ok)


# ===================================================================
# 2. Valid visits (no error)
# ===================================================================

def test_valid_visits():
    """WSCDataset(visit=1) and visit=5 do not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        for v in (1, 5):
            try:
                ds = WSCDataset(
                    visit=v,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, "cache"),
                )
                report(f"WSC valid visit={v} no error", True)
            except Exception as e:
                report(f"WSC valid visit={v} no error", False, str(e))


# ===================================================================
# 3. Invalid visits raise ValueError
# ===================================================================

def test_invalid_visits():
    """WSCDataset(visit=0) and visit=6 raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        for v in (0, 6):
            try:
                ds = WSCDataset(
                    visit=v,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, "cache"),
                )
                report(f"WSC invalid visit={v} raises ValueError", False, "no error raised")
            except ValueError:
                report(f"WSC invalid visit={v} raises ValueError", True)
            except Exception as e:
                report(f"WSC invalid visit={v} raises ValueError", False, f"wrong error: {e}")


# ===================================================================
# 4. Dataset name per visit
# ===================================================================

def test_dataset_name_per_visit():
    """WSCDataset(visit=2).DATASET_NAME == 'wsc_visit2'."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = WSCDataset(
            visit=2,
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = ds.DATASET_NAME == "wsc_visit2"
        report("WSC DATASET_NAME == 'wsc_visit2'", ok, f"got {ds.DATASET_NAME!r}")


# ===================================================================
# 5. Empty root -> 0 subjects
# ===================================================================

def test_empty_root():
    """Empty root returns 0 subjects (no crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = WSCDataset(
            visit=1,
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("WSC empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


# ===================================================================
# 6. Subject discovery
# ===================================================================

def test_subject_discovery():
    """Create fake WSC files; verify 1 subject discovered with '12345' in ID."""
    stg_lines = [
        "1\t0\t0",
        "2\t2\t2",
        "3\t2\t2",
        "4\t3\t3",
        "5\t5\t5",
    ]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_wsc_subject(data_dir, visit=1, subject_num="12345", stg_lines=stg_lines)
        ds = WSCDataset(
            visit=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        n = len(ds._subjects)
        ok_count = n == 1
        ok_id = n > 0 and "12345" in ds._subjects[0].subject_id
        ok = ok_count and ok_id
        detail = f"n_subjects={n}"
        if n > 0:
            detail += f", subject_id={ds._subjects[0].subject_id!r}"
        report("WSC discovers 1 subject with '12345' in ID", ok, detail)


# ===================================================================
# 7. STG label parsing
# ===================================================================

def test_stg_label_parsing():
    """Verify .stg.txt labels match expected AASM mapping."""
    # 0->W(0), 2->N2(2), 2->N2(2), 3->N3(3), 5->REM(4)
    stg_lines = [
        "1\t0\t0",
        "2\t2\t2",
        "3\t2\t2",
        "4\t3\t3",
        "5\t5\t5",
    ]
    expected = [0, 2, 2, 3, 4]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_wsc_subject(data_dir, visit=1, subject_num="12345", stg_lines=stg_lines)
        ds = WSCDataset(
            visit=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        ok_len = len(labels) == 5
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report(
            "WSC STG label parsing (length + values)",
            ok,
            f"len={len(labels)} expected=5, vals={list(labels)} expected={expected}",
        )


# ===================================================================
# 8. Visit filtering
# ===================================================================

def test_visit_filtering():
    """Files for visit1 and visit2 in same dir; visit=1 only sees visit1."""
    stg_lines = ["1\t0\t0", "2\t2\t2", "3\t2\t2", "4\t3\t3", "5\t5\t5"]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_wsc_subject(data_dir, visit=1, subject_num="11111", stg_lines=stg_lines)
        make_wsc_subject(data_dir, visit=2, subject_num="22222", stg_lines=stg_lines)

        ds1 = WSCDataset(
            visit=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        ids1 = [s.subject_id for s in ds1._subjects]

        ds2 = WSCDataset(
            visit=2,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "v2"),
        )
        ids2 = [s.subject_id for s in ds2._subjects]

        ok_v1 = len(ids1) == 1 and "11111" in ids1[0] and "visit1" in ids1[0]
        ok_v2 = len(ids2) == 1 and "22222" in ids2[0] and "visit2" in ids2[0]
        ok = ok_v1 and ok_v2
        report(
            "WSC visit filtering (visit1 vs visit2)",
            ok,
            f"visit1_ids={ids1}, visit2_ids={ids2}",
        )


# ===================================================================
# 9. End-to-end ds[0]
# ===================================================================

def test_end_to_end():
    """ds[0] returns a dict with expected keys."""
    stg_lines = ["1\t0\t0", "2\t2\t2", "3\t2\t2", "4\t3\t3", "5\t5\t5"]
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_wsc_subject(data_dir, visit=1, subject_num="12345", stg_lines=stg_lines)
        ds = WSCDataset(
            visit=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=3,
            cache_dir=cache_dir,
        )
        item = ds[0]
        checks = [
            "signals" in item,
            "labels" in item,
            "subject" in item,
            "channel_order" in item,
            torch.is_tensor(item["labels"]),
            item["labels"].shape[0] == 3,
        ]
        ok = all(checks)
        report("WSC end-to-end ds[0] dict", ok, f"checks={checks}")


# ===================================================================
# Real-data smoke tests (guarded)
# ===================================================================

@pytest.mark.real_data
def test_real_data_smoke():
    """Only runs if PHYSIOEX_TEST_REAL_DATA=1."""
    print("\n--- WSC real-data smoke tests ---")

    # Visit 1
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = WSCDataset(
                visit=1,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n > 500
            report("WSC visit 1 real data: >500 subjects", ok, f"got {n}")
    except Exception as e:
        report("WSC visit 1 real data", False, f"SKIP: {e}")

    # Visit 4
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = WSCDataset(
                visit=4,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n > 50
            report("WSC visit 4 real data: >50 subjects", ok, f"got {n}")
    except Exception as e:
        report("WSC visit 4 real data", False, f"SKIP: {e}")

    # Visit 5
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = WSCDataset(
                visit=5,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n >= 1
            report("WSC visit 5 real data: >=1 subjects", ok, f"got {n}")
    except Exception as e:
        report("WSC visit 5 real data", False, f"SKIP: {e}")


# ===================================================================
# Runner
# ===================================================================

