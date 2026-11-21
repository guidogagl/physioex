"""Tests for the cardiorespiratory dataset support.

Covers: registry entries for CFS/CHAT, session ids, the wav2sleep named
splits (local filtering + census), subject metadata (harmonized CSV + EDF
startdatetime), and ECG/PPG/ABD/THX channel preference resolution.
"""
from __future__ import annotations

import datetime
import logging
import tempfile
from pathlib import Path

import pytest

from physioex.data.base import SubjectSpec
from physioex.data.datasets import REGISTRY, available_datasets, get_dataset
from physioex.data.datasets._splits import load_wav2sleep_split
from physioex.data.datasets.chat import CHATDataset
from physioex.data.datasets.cfs import CFSDataset
from physioex.data.datasets.ccshs import CCSHSDataset
from physioex.data.readers.edf import resolve_channels
from tests.factories.edf import write_fake_edf


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _xml(events, rec_dur=300.0):
    head = (
        "<ScoredEvent><EventType/><EventConcept>Recording Start Time</EventConcept>"
        f"<Start>0</Start><Duration>{rec_dur}</Duration>"
        "<ClockTime>00.00.00 20.33.32</ClockTime></ScoredEvent>"
    )
    return (
        "<?xml version='1.0'?><PSGAnnotation><EpochLength>30</EpochLength><ScoredEvents>"
        + head
        + "".join(events)
        + "</ScoredEvents></PSGAnnotation>"
    )


def _ev(concept, start, dur, etype="Stages|Stages"):
    return (
        f"<ScoredEvent><EventType>{etype}</EventType><EventConcept>{concept}"
        f"</EventConcept><Start>{start}</Start><Duration>{dur}</Duration></ScoredEvent>"
    )


def _write_nsrr_subject(root: Path, stem: str, n_epochs: int = 10):
    """Create <stem>.edf + <stem>-nsrr.xml under an NSRR polysomnography layout."""
    edf_dir = root / "polysomnography" / "edfs"
    xml_dir = root / "polysomnography" / "annotations-events-nsrr"
    edf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    write_fake_edf(
        edf_dir / f"{stem}.edf",
        n_channels=1,
        duration_sec=30 * n_epochs,
        channel_names=["EEG"],
    )
    (xml_dir / f"{stem}-nsrr.xml").write_text(
        _xml([_ev("Wake|0", 30 * i, 30) for i in range(n_epochs)], 30.0 * n_epochs)
    )


def _write_wsc_subject(root: Path, subject_num: str, n_epochs: int = 10, visit: int = 1):
    """Create a WSC subject: wsc-visit{n}-{num}-nsrr.edf + .stg.txt in polysomnography/."""
    poly_dir = root / "polysomnography"
    poly_dir.mkdir(parents=True, exist_ok=True)
    base = f"wsc-visit{visit}-{subject_num}-nsrr"
    write_fake_edf(
        poly_dir / f"{base}.edf",
        n_channels=1,
        duration_sec=30 * n_epochs,
        channel_names=["C3_M2"],
    )
    with open(poly_dir / f"{base}.stg.txt", "w") as f:
        f.write("Epoch\tUser-Defined Stage\tCAST-Defined Stage\n")
        for i in range(n_epochs):
            f.write(f"{i + 1}\t2\t2\n")


def _write_harmonized_csv(root: Path, name: str, rows):
    """rows: list of (nsrrid, age, sex)."""
    d = root / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["nsrrid,nsrr_age,nsrr_sex"]
    for nsrrid, age, sex in rows:
        lines.append(f"{nsrrid},{age},{sex}")
    (d / name).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


def test_registry_cfs_chat():
    assert REGISTRY["cfs"] is CFSDataset
    assert REGISTRY["chat"] is CHATDataset
    assert get_dataset("cfs") is CFSDataset
    assert get_dataset("chat") is CHATDataset
    assert "cfs" in available_datasets()
    assert "chat" in available_datasets()


def test_registry_cardioresp_modalities_present():
    from physioex.data.datasets.shhs import SHHSDataset
    from physioex.data.datasets.mesa import MESADataset
    from physioex.data.datasets.mros import MrOSDataset
    from physioex.data.datasets.wsc import WSCDataset

    for cls in (
        CFSDataset,
        CHATDataset,
        CCSHSDataset,
        SHHSDataset,
        MESADataset,
        MrOSDataset,
        WSCDataset,
    ):
        prefs = cls.CHANNEL_PREFERENCES
        for key in ("ECG", "PPG", "ABD", "THX"):
            assert key in prefs, f"{cls.__name__} missing {key} preferences"
            assert isinstance(prefs[key], list) and prefs[key], (
                f"{cls.__name__}.{key} preferences empty"
            )


# ---------------------------------------------------------------------------
# 2. Session ids
# ---------------------------------------------------------------------------


def test_session_id_identity_on_nsrr_stems():
    ds = object.__new__(CFSDataset)
    ds.DATASET_NAME = "cfs"
    # string form
    assert ds.session_id("cfs-visit5-800002") == "cfs-visit5-800002"
    # SubjectSpec form
    spec = SubjectSpec(subject_id="cfs-visit5-800002", edf_path=Path("x.edf"))
    assert ds.session_id(spec) == "cfs-visit5-800002"


def test_session_id_wsc_appends_nsrr():
    from physioex.data.datasets.wsc import WSCDataset

    ds = object.__new__(WSCDataset)
    assert ds.session_id("wsc-visit1-10119") == "wsc-visit1-10119-nsrr"
    spec = SubjectSpec(subject_id="wsc-visit1-10119", edf_path=Path("x.edf"))
    assert ds.session_id(spec) == "wsc-visit1-10119-nsrr"


def test_chat_nsrrid_extraction():
    ds = object.__new__(CHATDataset)
    assert ds._nsrrid_of("chat-baseline-300002") == "300002"
    assert ds._nsrrid_of("chat-followup-300002") == "300002"
    assert ds._nsrrid_of("chat-baseline-nonrandomized-300006") == "300006"


# ---------------------------------------------------------------------------
# 3. wav2sleep splits
# ---------------------------------------------------------------------------


def test_load_wav2sleep_split_files():
    train = load_wav2sleep_split("cfs", "train")
    val = load_wav2sleep_split("cfs", "val")
    test = load_wav2sleep_split("cfs", "test")
    assert all(s.startswith("cfs-visit5-") for s in train + val + test)
    # MrOS is test-only: train/val are empty lists
    assert load_wav2sleep_split("mros", "train") == []
    assert load_wav2sleep_split("mros", "val") == []
    assert load_wav2sleep_split("mros", "test")
    # census has no train list
    assert load_wav2sleep_split("cfs", "train", census=True) == []
    assert load_wav2sleep_split("cfs", "val", census=True)


def test_wav2sleep_split_filters_to_local_subjects(tmp_path, cache_dir):
    # Only two of the ids from the real split files exist locally.
    _write_nsrr_subject(tmp_path, "cfs-visit5-800002")
    _write_nsrr_subject(tmp_path, "cfs-visit5-800010")
    _write_nsrr_subject(tmp_path, "cfs-visit5-999999")  # not in any split file
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    train, val, test = ds.get_splits(fold=0, split="wav2sleep")
    assert "cfs-visit5-800002" in test  # it is in the real test list
    assert "cfs-visit5-800010" in train  # it is in the real train list
    # everything else in the lists is dropped (not local)
    assert set(train + val + test) <= set(ds.get_subjects())
    assert len(train) == 1 and len(val) == 0 and len(test) == 1


def test_wav2sleep_split_missing_logged_once(tmp_path, caplog):
    _write_nsrr_subject(tmp_path, "cfs-visit5-800002")
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    with caplog.at_level(logging.INFO, logger="physioex.data"):
        ds.get_splits(fold=0, split="wav2sleep")
    info_msgs = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "not present locally" in r.getMessage()
    ]
    # one message per non-empty part (train/val/test) that had missing ids
    assert len(info_msgs) >= 1


def test_wav2sleep_census_keeps_only_this_dataset(tmp_path):
    # The census lists mix ccshs/cfs/chat/mesa/wsc ids; only the ones that
    # start with this dataset's prefix survive. 800011 is in the census val list.
    _write_nsrr_subject(tmp_path, "cfs-visit5-800011")
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    train, val, test = ds.get_splits(fold=0, split="wav2sleep-census")
    assert train == []  # census has no train list
    assert set(val + test) <= {"cfs-visit5-800011"}
    # 800011 is in the census val list -> must be present
    assert "cfs-visit5-800011" in val


def test_wsc_split_returns_subject_ids_without_nsrr_suffix(tmp_path):
    from physioex.data.datasets.wsc import WSCDataset

    # 10191 and 10226 are in the real wav2sleep wsc train list; 99999 is local-only.
    _write_wsc_subject(tmp_path, "10191")
    _write_wsc_subject(tmp_path, "10226")
    _write_wsc_subject(tmp_path, "99999")
    with tempfile.TemporaryDirectory() as cache:
        ds = WSCDataset(
            root=str(tmp_path), visit=1, channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    train, val, test = ds.get_splits(fold=0, split="wav2sleep")
    # Returned ids are physioex subject_ids (no trailing -nsrr)
    assert all(not i.endswith("-nsrr") for i in train + val + test)
    assert set(train + val + test) <= set(ds.get_subjects())
    assert "wsc-visit1-10191" in train
    assert "wsc-visit1-10226" in train


def test_unknown_split_raises(tmp_path):
    _write_nsrr_subject(tmp_path, "cfs-visit5-800002")
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    with pytest.raises(ValueError, match="Unknown split"):
        ds.get_splits(fold=0, split="nope")


# ---------------------------------------------------------------------------
# 4. Subject metadata
# ---------------------------------------------------------------------------


def test_subject_metadata_from_harmonized_csv(tmp_path):
    _write_nsrr_subject(tmp_path, "cfs-visit5-800002")
    _write_nsrr_subject(tmp_path, "cfs-visit5-800010")
    _write_harmonized_csv(
        tmp_path,
        "cfs-harmonized-dataset-0.2.0.csv",
        [(800002, 55, "M"), (800010, 63, "F")],
    )
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    meta = ds.subject_metadata("cfs-visit5-800002")
    assert meta["age"] == 55.0
    assert meta["sex"] == "M"
    meta2 = ds.subject_metadata("cfs-visit5-800010")
    assert meta2["age"] == 63.0
    assert meta2["sex"] == "F"
    # subject not in the CSV -> None, never raises
    meta3 = ds.subject_metadata("cfs-visit5-999999")
    assert meta3["age"] is None and meta3["sex"] is None


def test_subject_metadata_missing_csv(tmp_path):
    _write_nsrr_subject(tmp_path, "cfs-visit5-800002")
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    meta = ds.subject_metadata("cfs-visit5-800002")
    assert meta["age"] is None and meta["sex"] is None
    # keys are always present
    assert set(meta) == {"age", "sex", "recording_start"}


def test_subject_metadata_recording_start(tmp_path):
    _write_nsrr_subject(tmp_path, "cfs-visit5-800002")
    _write_harmonized_csv(
        tmp_path, "cfs-harmonized-dataset-0.2.0.csv", [(800002, 55, "M")]
    )
    with tempfile.TemporaryDirectory() as cache:
        ds = CFSDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    meta = ds.subject_metadata("cfs-visit5-800002")
    # The fake EDF has a startdatetime (set by pyedflib on write);
    # recording_start must be a datetime.time or None (not raise).
    assert meta["recording_start"] is None or isinstance(
        meta["recording_start"], datetime.time
    )


def test_chat_metadata_row_key(tmp_path):
    _write_nsrr_subject(tmp_path, "chat-baseline-300002")
    _write_nsrr_subject(tmp_path, "chat-baseline-nonrandomized-300006")
    _write_harmonized_csv(
        tmp_path,
        "chat-harmonized-dataset-0.2.0.csv",
        [(300002, 41, "F"), (300006, 44, "M")],
    )
    with tempfile.TemporaryDirectory() as cache:
        ds = CHATDataset(
            root=str(tmp_path), channels=["EEG"], pipelines="raw",
            sequence_length=2, cache_dir=cache,
        )
    assert ds.subject_metadata("chat-baseline-300002")["sex"] == "F"
    assert ds.subject_metadata("chat-baseline-nonrandomized-300006")["age"] == 44.0


# ---------------------------------------------------------------------------
# 5. Channel preference resolution (ECG / PPG / ABD / THX)
# ---------------------------------------------------------------------------


def _resolve(prefs, requests, available):
    fs = {a: 256.0 for a in available}
    return [r.physical for r in resolve_channels(requests, available, fs, prefs)]


def test_ecg_prefers_canonical_names():
    prefs = CFSDataset.CHANNEL_PREFERENCES
    # "ECG" is first and present
    assert _resolve(prefs, ["ECG"], ["ECG", "EKG", "ECG1"]) == ["ECG"]
    # "ECG" absent -> "EKG"
    assert _resolve(prefs, ["ECG"], ["EKG"]) == ["EKG"]
    # "ECG L-ECG R" is a specific channel name (matches case-insensitively)
    assert _resolve(prefs, ["ECG"], ["ECG L-ECG R"]) == ["ECG L-ECG R"]


def test_ppg_prefers_pleth_variants():
    prefs = CFSDataset.CHANNEL_PREFERENCES
    assert _resolve(prefs, ["PPG"], ["Pleth", "PlethWV"]) == ["Pleth"]
    assert _resolve(prefs, ["PPG"], ["PlethWV"]) == ["PlethWV"]
    # "PLETHMASIMO" (uppercase channel name) matches case-insensitively to the
    # preference entry "PLETHMASIMO"
    res = _resolve(prefs, ["PPG"], ["PLETHMASIMO"])
    assert res[0].upper() == "PLETHMASIMO"


def test_abd_and_thx_resolve_distinctly():
    prefs = CFSDataset.CHANNEL_PREFERENCES
    # "ABDO RES" and "THOR RES" are distinct channel names
    res = _resolve(prefs, ["ABD", "THX"], ["ABDO RES", "THOR RES", "ECG"])
    assert res == ["ABDO RES", "THOR RES"]
    # canonical names also work
    assert _resolve(prefs, ["ABD"], ["ABD", "THX"]) == ["ABD"]
    assert _resolve(prefs, ["THX"], ["ABD", "THX"]) == ["THX"]


def test_abd_thx_fall_back_to_variants():
    prefs = CFSDataset.CHANNEL_PREFERENCES
    # "Abdo" (capital A) matches case-insensitively
    assert _resolve(prefs, ["ABD"], ["Abdo", "Thor"]) == ["Abdo"]
    assert _resolve(prefs, ["THX"], ["Abdo", "Thor"]) == ["Thor"]


def test_ccshs_cardioresp_preferences():
    prefs = CCSHSDataset.CHANNEL_PREFERENCES
    assert "ECG" in prefs and "PPG" in prefs and "ABD" in prefs and "THX" in prefs
    res = _resolve(prefs, ["ABD", "THX", "PPG"], ["ABDO RES", "THOR RES", "Pleth"])
    assert res == ["ABDO RES", "THOR RES", "Pleth"]


def test_chat_cardioresp_preferences():
    prefs = CHATDataset.CHANNEL_PREFERENCES
    res = _resolve(prefs, ["ECG", "ABD", "THX"], ["ECG L", "ABDO RES", "THOR RES"])
    assert res == ["ECG L", "ABDO RES", "THOR RES"]


def test_sleepppgnet_replica_split_file():
    from physioex.data.datasets._splits import load_sleepppgnet_replica_split

    ids = load_sleepppgnet_replica_split("mesa", "test")
    assert len(ids) == 204 and len(set(ids)) == 204
    assert all(i.startswith("mesa-sleep-") for i in ids)
    assert load_sleepppgnet_replica_split("mesa", "train") == []
    assert load_sleepppgnet_replica_split("shhs", "test") == []


def test_harmonized_age_sex_falls_back_to_plain_columns():
    # CCSHS ships ccshs-trec-dataset with ``age`` / ``male`` instead of the
    # harmonized nsrr_age / nsrr_sex columns.
    from physioex.data.datasets._nsrr import _NSRRBaseDataset

    f = _NSRRBaseDataset._harmonized_age_sex
    assert f({"nsrr_age": 50.0, "nsrr_sex": "female"}) == (50.0, "F")
    assert f({"age": 17.0, "male": 1}) == (17.0, "M")
    assert f({"age": "16", "male": "0"}) == (16.0, "F")
    assert f({}) == (None, None)
