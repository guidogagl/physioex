"""Unit tests for the CCSHS reader (synthetic NSRR XML, no raw data)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from physioex.data.datasets import REGISTRY
from physioex.data.datasets.ccshs import CCSHSDataset, parse_ccshs_xml
from physioex.data.readers.edf import resolve_channels
from physioex.models.foundation_datasets import DATASET_CONFIGS


def _ev(concept, start, dur, etype="Stages|Stages"):
    return (f"<ScoredEvent><EventType>{etype}</EventType><EventConcept>{concept}"
            f"</EventConcept><Start>{start}</Start><Duration>{dur}</Duration></ScoredEvent>")


def _xml(events, rec_dur=300.0):
    head = ("<ScoredEvent><EventType/><EventConcept>Recording Start Time</EventConcept>"
            f"<Start>0</Start><Duration>{rec_dur}</Duration>"
            "<ClockTime>00.00.00 20.33.32</ClockTime></ScoredEvent>")
    return ("<?xml version='1.0'?><PSGAnnotation><EpochLength>30</EpochLength><ScoredEvents>"
            + head + "".join(events) + "</ScoredEvents></PSGAnnotation>")


def test_registered():
    assert REGISTRY["ccshs"] is CCSHSDataset
    assert CCSHSDataset.DATASET_NAME == "ccshs"
    assert "ccshs" in DATASET_CONFIGS


def test_label_mapping_and_onset_indexing(tmp_path):
    # Events deliberately out of order: indexing must use onset, not row order.
    events = [
        _ev("REM sleep|5", 240, 60),
        _ev("Wake|0", 0, 30),
        _ev("Stage 1 sleep|1", 30, 30),
        _ev("Stage 3 sleep|3", 120, 30),
        _ev("Stage 4 sleep|4", 150, 30),
        _ev("Movement|6", 180, 30),
        _ev("Stage 2 sleep|2", 60, 60),
        # non-stage event must be ignored
        _ev("Hypopnea|Hypopnea", 10, 12, etype="Respiratory|Respiratory"),
    ]
    p = tmp_path / "s-nsrr.xml"
    p.write_text(_xml(events))
    out = parse_ccshs_xml(p)
    # 0 1 2 2 3 3 -1 (unlabelled epoch 210-240 -> -1) 4 4
    assert out["labels"].tolist() == [0, 1, 2, 2, 3, 3, -1, -1, 4, 4]
    assert out["duration_sec"] == 300.0
    assert out["clock_time"] == "00.00.00 20.33.32"


def test_length_is_floor_of_recording_duration(tmp_path):
    p = tmp_path / "s-nsrr.xml"
    p.write_text(_xml([_ev("Wake|0", 0, 30)], rec_dur=95.0))
    assert len(parse_ccshs_xml(p)["labels"]) == 3


def test_unknown_stage_raises(tmp_path):
    p = tmp_path / "s-nsrr.xml"
    p.write_text(_xml([_ev("Stage 9 sleep|9", 0, 30)]))
    with pytest.raises(ValueError):
        parse_ccshs_xml(p)


def test_subject_pairing_and_metadata(tmp_path):
    (tmp_path / "polysomnography" / "edfs").mkdir(parents=True)
    (tmp_path / "polysomnography" / "annotations-events-nsrr").mkdir(parents=True)
    for sid in ("ccshs-trec-1800001", "ccshs-trec-1800005"):
        (tmp_path / "polysomnography" / "edfs" / f"{sid}.edf").write_bytes(b"")
    (tmp_path / "polysomnography" / "annotations-events-nsrr"
     / "ccshs-trec-1800001-nsrr.xml").write_text(_xml([_ev("Wake|0", 0, 30)]))
    ds = object.__new__(CCSHSDataset)
    ds.root = str(tmp_path)
    ds._exclude = frozenset()
    ds._metadata = {"1800001": {"age": 10}}
    specs = ds._list_subjects()
    assert [s.subject_id for s in specs] == ["ccshs-trec-1800001"]  # EDF without XML skipped
    assert specs[0].external_meta["age"] == 10


def test_channel_resolution_pairs():
    avail = ["C3", "C4", "A1", "A2", "LOC", "ROC", "EMG1", "EMG2", "EMG3"]
    fs = {a: 128.0 for a in avail}
    res = resolve_channels(["EEG", "EEG", "EOG", "EMG"], avail, fs,
                           CCSHSDataset.CHANNEL_PREFERENCES)
    assert [r.physical for r in res] == [("C3", "A2"), ("C4", "A1"),
                                          ("LOC", "ROC"), ("EMG1", "EMG2")]


# ── exclusion list, metadata glob, end-to-end through BasePhysioDataset ─────────────────────────

STAGES = [0, 0, 1, 2, 2, 3, 3, 2, 4, 4, 2, 0]  # 12 epochs = 360 s
CH = ["C3", "C4", "A1", "A2", "LOC", "ROC", "EMG1", "EMG2"]
FS = [128, 128, 128, 128, 128, 128, 256, 256]  # EEG/EOG 128 Hz, chin EMG 256 Hz as in the real EDFs
_CONCEPT = {0: "Wake|0", 1: "Stage 1 sleep|1", 2: "Stage 2 sleep|2", 3: "Stage 3 sleep|3", 4: "REM sleep|5"}


def _make_subject(root: Path, nsrrid: int, stages=STAGES):
    from tests.factories.edf import write_fake_edf

    edf_dir = root / "polysomnography" / "edfs"
    xml_dir = root / "polysomnography" / "annotations-events-nsrr"
    edf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ccshs-trec-{nsrrid}"
    write_fake_edf(edf_dir / f"{stem}.edf", n_channels=len(CH), duration_sec=30 * len(stages),
                   fs_per_channel=FS, channel_names=CH)
    evs = [_ev(_CONCEPT[s], 30 * i, 30) for i, s in enumerate(stages)]
    (xml_dir / f"{stem}-nsrr.xml").write_text(_xml(evs, rec_dur=30.0 * len(stages)))
    return stem


def _write_csv(root: Path, name: str, ids):
    d = root / "datasets"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("nsrrid,age\n" + "".join(f"{i},{10 + k}\n" for k, i in enumerate(ids)))


def test_excluded_subjects_default_list():
    assert len(CCSHSDataset.EXCLUDED_SUBJECTS) == 11
    assert "ccshs-trec-1800064" in CCSHSDataset.EXCLUDED_SUBJECTS
    assert "ccshs-trec-1800489" in CCSHSDataset.EXCLUDED_SUBJECTS
    assert all(s.startswith("ccshs-trec-18") for s in CCSHSDataset.EXCLUDED_SUBJECTS)


def test_exclusion_and_metadata_glob(tmp_path):
    for i in (1800001, 1800064, 1800005):
        _make_subject(tmp_path, i)
    _write_csv(tmp_path, "ccshs-trec-dataset-0.9.1.csv", [1800001, 1800005])  # newer version than 0.8.0
    with tempfile.TemporaryDirectory() as cache:
        ds = CCSHSDataset(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="raw",
                          sequence_length=4, cache_dir=cache)
        assert sorted(ds.get_subjects()) == ["ccshs-trec-1800001", "ccshs-trec-1800005"]  # 1800064 excluded
        assert ds._metadata["1800005"]["age"] == 11
        keep_all = CCSHSDataset(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="raw",
                                sequence_length=4, cache_dir=cache, exclude_subjects=[])
        assert len(keep_all.get_subjects()) == 3
        only = CCSHSDataset(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="raw",
                            sequence_length=4, cache_dir=cache, exclude_subjects=["ccshs-trec-1800001"])
        assert "ccshs-trec-1800001" not in only.get_subjects()


def test_end_to_end_spectrogram_and_labels(tmp_path):
    _make_subject(tmp_path, 1800001)
    with tempfile.TemporaryDirectory() as cache:
        ds = CCSHSDataset(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="seqsleepnet",
                          sequence_length=4, cache_dir=cache, exclude_subjects=[])
        res = ds._resolved["ccshs-trec-1800001"]
        assert [r.physical for r in res] == [("C3", "A2"), ("LOC", "ROC"), ("EMG1", "EMG2")]
        item = ds[0]
        order = item["channel_order"]
        assert len(order) == 3
        for name in order:
            assert tuple(item["signals"][name].shape) == (4, 29, 129)  # 4 epochs x log-spectrogram
        assert item["labels"].tolist() == STAGES[:4]  # N1 stays 1, S3/S4 -> 3, REM -> 4
        last = ds[len(ds) - 1]
        assert last["labels"].tolist() == STAGES[-4:]
        assert item["subject"]["dataset"] == "ccshs"


def test_unscored_movement_epochs_are_minus_one(tmp_path):
    stages = [0, 2, 2, 6, 2, 2, 0, 0]  # 6 = Movement|6 in the XML
    from tests.factories.edf import write_fake_edf

    edf_dir = tmp_path / "polysomnography" / "edfs"
    xml_dir = tmp_path / "polysomnography" / "annotations-events-nsrr"
    edf_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)
    write_fake_edf(edf_dir / "ccshs-trec-1800002.edf", n_channels=len(CH), duration_sec=30 * len(stages),
                   fs_per_channel=FS, channel_names=CH)
    conc = {**_CONCEPT, 6: "Movement|6"}
    (xml_dir / "ccshs-trec-1800002-nsrr.xml").write_text(
        _xml([_ev(conc[s], 30 * i, 30) for i, s in enumerate(stages)], rec_dur=30.0 * len(stages)))
    with tempfile.TemporaryDirectory() as cache:
        ds = CCSHSDataset(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="seqsleepnet",
                          sequence_length=4, cache_dir=cache, exclude_subjects=[])
        labs = np.concatenate([ds[i]["labels"].numpy() for i in (0, len(ds) - 1)])
        assert -1 in labs.tolist()
