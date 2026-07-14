"""Unit tests for physioex.data.readers.annotations parsers.

Covers the four public parsers on synthetic inputs (no real data):
  - parse_nsrr_xml           (stage labels, N4->N3, unscored, empty)
  - parse_tsv_annotations    (default + custom stage maps, bad rows, empty)
  - parse_nsrr_xml_events    (category mapping + extra SpO2 fields)
  - parse_stages_csv         (dur=0 / dur=30 / block variants + events + clamp)
"""
import numpy as np
import pytest

from physioex.data.events import SleepEvent
from physioex.data.readers.annotations import (
    parse_nsrr_xml,
    parse_tsv_annotations,
    parse_nsrr_xml_events,
    parse_stages_csv,
    NSRR_STAGE_MAP,
    STAGES_STAGE_MAP,
)


# ---------------------------------------------------------------------------
# NSRR XML stage parser
# ---------------------------------------------------------------------------

def _write_nsrr_xml(path, events):
    """events: list of (concept, start, duration)."""
    lines = ['<?xml version="1.0"?>', "<PSGAnnotation>", "  <ScoredEvents>"]
    for concept, start, dur in events:
        lines += [
            "    <ScoredEvent>",
            "      <EventType>Stages|Stages</EventType>",
            f"      <EventConcept>{concept}</EventConcept>",
            f"      <Start>{start}</Start>",
            f"      <Duration>{dur}</Duration>",
            "    </ScoredEvent>",
        ]
    lines += ["  </ScoredEvents>", "</PSGAnnotation>"]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_parse_nsrr_xml_basic(tmp_path):
    xml = tmp_path / "a.xml"
    _write_nsrr_xml(
        xml,
        [
            ("Wake|0", 0, 60),          # epochs 0,1
            ("Stage 2 sleep|2", 60, 60),  # epochs 2,3
            ("REM sleep|5", 120, 30),   # epoch 4
        ],
    )
    labels = parse_nsrr_xml(xml)
    assert labels.dtype == np.int16
    assert labels.tolist() == [0, 0, 2, 2, 4]


def test_parse_nsrr_xml_n4_maps_to_n3(tmp_path):
    xml = tmp_path / "b.xml"
    _write_nsrr_xml(xml, [("Stage 4 sleep|4", 0, 30)])
    assert parse_nsrr_xml(xml).tolist() == [3]


def test_parse_nsrr_xml_unscored_and_gaps(tmp_path):
    xml = tmp_path / "c.xml"
    # A gap epoch (1) stays -1; explicit Unscored also -1.
    _write_nsrr_xml(xml, [("Wake|0", 0, 30), ("Unscored|9", 60, 30)])
    assert parse_nsrr_xml(xml).tolist() == [0, -1, -1]


def test_parse_nsrr_xml_empty(tmp_path):
    xml = tmp_path / "empty.xml"
    _write_nsrr_xml(xml, [])
    out = parse_nsrr_xml(xml)
    assert out.size == 0 and out.dtype == np.int16


def test_parse_nsrr_xml_custom_epoch_length(tmp_path):
    xml = tmp_path / "d.xml"
    _write_nsrr_xml(xml, [("Wake|0", 0, 60)])
    # 20s epochs -> 3 epochs over 60s
    assert parse_nsrr_xml(xml, epoch_length_sec=20.0).tolist() == [0, 0, 0]


def test_nsrr_stage_map_contract():
    assert NSRR_STAGE_MAP["Stage 4 sleep|4"] == 3  # N4 collapses to N3
    assert NSRR_STAGE_MAP["Unscored|9"] == -1


# ---------------------------------------------------------------------------
# TSV parser
# ---------------------------------------------------------------------------

def _write_tsv(path, rows, header="start\tend\tstage"):
    lines = [header] + ["\t".join(str(c) for c in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_tsv_default_map(tmp_path):
    tsv = tmp_path / "a.tsv"
    _write_tsv(tsv, [(0, 30, "W"), (30, 60, "N2"), (60, 90, "REM")])
    assert parse_tsv_annotations(tsv).tolist() == [0, 2, 4]


def test_parse_tsv_unknown_stage_is_unscored(tmp_path):
    tsv = tmp_path / "b.tsv"
    _write_tsv(tsv, [(0, 30, "W"), (30, 60, "???")])
    assert parse_tsv_annotations(tsv).tolist() == [0, -1]


def test_parse_tsv_bad_rows_skipped(tmp_path):
    tsv = tmp_path / "c.tsv"
    # second row has a non-numeric start -> skipped
    _write_tsv(tsv, [(0, 30, "W"), ("oops", 60, "N2"), (60, 90, "N3")])
    out = parse_tsv_annotations(tsv)
    assert out[0] == 0
    assert out[2] == 3  # third row still placed at epoch 2


def test_parse_tsv_empty(tmp_path):
    tsv = tmp_path / "d.tsv"
    _write_tsv(tsv, [])
    out = parse_tsv_annotations(tsv)
    assert out.size == 0 and out.dtype == np.int16


def test_parse_tsv_custom_columns_and_map(tmp_path):
    tsv = tmp_path / "e.tsv"
    _write_tsv(tsv, [(0, 30, "asleep")], header="onset\toffset\tlabel")
    out = parse_tsv_annotations(
        tsv,
        start_col="onset",
        end_col="offset",
        stage_col="label",
        stage_map={"asleep": 2},
    )
    assert out.tolist() == [2]


# ---------------------------------------------------------------------------
# NSRR XML event parser
# ---------------------------------------------------------------------------

def _write_nsrr_events_xml(path, events):
    """events: list of (etype, concept, start, dur, extra_dict)."""
    lines = ['<?xml version="1.0"?>', "<PSGAnnotation>", "  <ScoredEvents>"]
    for etype, concept, start, dur, extra in events:
        lines += [
            "    <ScoredEvent>",
            f"      <EventType>{etype}</EventType>",
            f"      <EventConcept>{concept}</EventConcept>",
            f"      <Start>{start}</Start>",
            f"      <Duration>{dur}</Duration>",
        ]
        for k, v in (extra or {}).items():
            lines.append(f"      <{k}>{v}</{k}>")
        lines.append("    </ScoredEvent>")
    lines += ["  </ScoredEvents>", "</PSGAnnotation>"]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_parse_nsrr_xml_events_categories(tmp_path):
    xml = tmp_path / "ev.xml"
    _write_nsrr_events_xml(
        xml,
        [
            ("Stages|Stages", "Wake|0", 0, 30, None),   # skipped (stage)
            ("Respiratory|Respiratory", "Hypopnea|Hypopnea", 100, 15,
             {"SpO2Nadir": 88.0, "SpO2Baseline": 95.0}),
            ("Arousals|Arousals", "ASDA arousal|Arousal", 200, 3, None),
            ("Respiratory|Respiratory", "Obstructive apnea desaturation", 300, 20, None),
        ],
    )
    events = parse_nsrr_xml_events(xml)
    assert all(isinstance(e, SleepEvent) for e in events)
    # stage event excluded
    assert len(events) == 3
    by_type = {e.concept: e for e in events}

    hyp = by_type["Hypopnea"]
    assert hyp.type == "respiratory"
    assert hyp.onset_sec == 100.0 and hyp.duration_sec == 15.0
    assert hyp.extra["spO2Nadir"] == 88.0
    assert hyp.extra["spO2Baseline"] == 95.0

    assert by_type["ASDA arousal"].type == "arousal"
    # "desaturation" keyword in concept overrides category
    desat = [e for e in events if "desaturation" in e.concept.lower()][0]
    assert desat.type == "desaturation"


def test_parse_nsrr_xml_events_empty(tmp_path):
    xml = tmp_path / "none.xml"
    _write_nsrr_events_xml(xml, [("Stages|Stages", "Wake|0", 0, 30, None)])
    assert parse_nsrr_xml_events(xml) == []


# ---------------------------------------------------------------------------
# STAGES CSV parser
# ---------------------------------------------------------------------------

def _write_stages_csv(path, rows):
    """rows: list of (time_str, duration, event)."""
    lines = ["Start Time,Duration (seconds),Event"]
    for t, d, e in rows:
        lines.append(f"{t},{d},{e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_stages_csv_dur30(tmp_path):
    csv_path = tmp_path / "dur30.csv"
    _write_stages_csv(
        csv_path,
        [
            ("22:00:00", 30, "Wake"),
            ("22:00:30", 30, "Stage2"),
            ("22:01:00", 30, "REM"),
        ],
    )
    labels, events = parse_stages_csv(csv_path)
    assert labels.tolist() == [0, 2, 4]
    assert events == []


def test_parse_stages_csv_blocks(tmp_path):
    csv_path = tmp_path / "blocks.csv"
    # A 90s Stage2 block == 3 epochs, then a 30s Wake.
    _write_stages_csv(
        csv_path,
        [("23:00:00", 90, "Stage2"), ("23:01:30", 30, "Wake")],
    )
    labels, _ = parse_stages_csv(csv_path)
    assert labels.tolist() == [2, 2, 2, 0]


def test_parse_stages_csv_dur0(tmp_path):
    csv_path = tmp_path / "dur0.csv"
    # dur=0 encoding: one epoch per timestamp, 30s apart.
    _write_stages_csv(
        csv_path,
        [("01:00:00", 0, "Wake"), ("01:00:30", 0, "Stage1"), ("01:01:00", 0, "Stage3")],
    )
    labels, _ = parse_stages_csv(csv_path)
    assert labels.tolist() == [0, 1, 3]


def test_parse_stages_csv_events_separated(tmp_path):
    csv_path = tmp_path / "ev.csv"
    _write_stages_csv(
        csv_path,
        [
            ("00:00:00", 30, "Wake"),
            ("00:00:05", 12, "Hypopnea"),      # respiratory event
            ("00:00:30", 30, "Stage2"),
            ("00:00:40", 8, "Arousal"),        # arousal event
        ],
    )
    labels, events = parse_stages_csv(csv_path)
    assert labels.tolist() == [0, 2]
    cats = sorted(e.type for e in events)
    assert cats == ["arousal", "respiratory"]


def test_parse_stages_csv_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    _write_stages_csv(csv_path, [])
    labels, events = parse_stages_csv(csv_path)
    assert labels.size == 0 and events == []


def test_parse_stages_csv_anomalous_duration_clamped(tmp_path):
    csv_path = tmp_path / "anom.csv"
    # 2592000s (30 days) must be clamped to one epoch, not blow up the array.
    _write_stages_csv(
        csv_path,
        [("00:00:00", 30, "Wake"), ("00:00:30", 2592000, "Stage2")],
    )
    labels, _ = parse_stages_csv(csv_path)
    assert labels.tolist() == [0, 2]


def test_stages_stage_map_contract():
    assert STAGES_STAGE_MAP["Wake"] == 0
    assert STAGES_STAGE_MAP["REM"] == 4
    assert STAGES_STAGE_MAP["MT"] == -1
