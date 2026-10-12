"""Unit tests for the CCSHS reader (synthetic NSRR XML, no raw data)."""
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
