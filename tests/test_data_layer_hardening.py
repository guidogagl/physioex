"""Data-layer correctness fixes: subjects without channels, label cache revision, ``artifact`` events."""
from __future__ import annotations

import csv
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from physioex.data.base import BasePhysioDataset, SubjectSpec
from physioex.data.collate import dict_collate_fn
from physioex.data.readers.annotations import parse_nsrr_xml_events, parse_stages_csv
from tests.factories.edf import write_fake_edf

pytestmark = pytest.mark.unit

N_EPOCHS = 12


class _Fake(BasePhysioDataset):
    """Two-subject dataset over fake EDFs; subclasses set ``LABELS_REVISION`` and ``label_value``."""

    DATASET_NAME = "hardening_fake"
    CHANNEL_PREFERENCES = {"EEG": ["C4-M2"], "EOG": ["EOG"], "EMG": ["EMG"]}
    label_value = 0
    reads = 0
    fail_reads = False

    def _list_subjects(self):
        root = Path(self.root)
        return [SubjectSpec(subject_id=p.stem, edf_path=p, label_path=None) for p in sorted(root.glob("*.edf"))]

    def _read_subject_labels(self, spec):
        type(self).reads += 1
        if type(self).fail_reads:
            raise FileNotFoundError("raw annotations are not available")
        return np.full(N_EPOCHS, type(self).label_value, dtype=np.int16)


def _subject(root: Path, name: str, channels):
    write_fake_edf(root / f"{name}.edf", n_channels=len(channels), duration_sec=30 * N_EPOCHS,
                   fs_per_channel=[100] * len(channels), channel_names=channels)


# ── 1. subjects without any requested channel ──────────────────────────────────────────────────

def test_subject_without_any_requested_channel_gets_zeros_of_the_right_shape(tmp_path):
    _subject(tmp_path, "a_ok", ["C4-M2", "EOG", "EMG"])
    _subject(tmp_path, "b_nothing", ["FOO", "BAR"])          # none of EEG/EOG/EMG resolvable
    _subject(tmp_path, "c_partial", ["C4-M2", "FOO"])         # EEG only: zero-filled
    with tempfile.TemporaryDirectory() as cache, pytest.warns(UserWarning, match="b_nothing"):
        ds = _Fake(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="seqsleepnet",
                   sequence_length=4, cache_dir=cache)
    assert ds.get_subjects() == ["a_ok", "b_nothing", "c_partial"]    # never dropped
    ranges = {sid: (s, e) for sid, s, e in ds._subject_ranges}
    items = [ds[ranges[sid][0]] for sid in ("a_ok", "b_nothing", "c_partial")]
    for it in items:
        for name in it["channel_order"]:
            assert tuple(it["signals"][name].shape) == (4, 29, 129)  # same shape for every subject
    assert all((items[1]["signals"][n] == 0).all() for n in items[1]["channel_order"])
    batch = dict_collate_fn(items)                                    # used to raise in torch.stack
    for name in batch["channel_order"]:
        assert tuple(batch["signals"][name].shape) == (3, 4, 29, 129)


# ── 2. label cache revision ────────────────────────────────────────────────────────────────────

def _make(root, cache, rev, value, fail=False):
    cls = type("FakeRev", (_Fake,), {"LABELS_REVISION": rev, "label_value": value, "reads": 0, "fail_reads": fail})
    ds = cls(root=str(root), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache)
    return cls, ds


def _meta_path(ds, sid):
    return ds.cache.labels_meta_path(ds.DATASET_NAME, sid)


def test_old_revision_cache_is_recomputed_and_rewritten(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    _subject(root, "s1", ["C4-M2"])
    with tempfile.TemporaryDirectory() as cache:
        cls1, ds1 = _make(root, cache, rev=1, value=0)
        assert ds1._load_or_compute_labels(ds1._subjects[0]).tolist() == [0] * N_EPOCHS
        assert json.loads(_meta_path(ds1, "s1").read_text())["labels_revision"] == 1
        # the reader is fixed: revision 2 now returns other labels; the stale cache must not be served
        cls2, ds2 = _make(root, cache, rev=2, value=3)
        assert ds2._load_or_compute_labels(ds2._subjects[0]).tolist() == [3] * N_EPOCHS
        assert cls2.reads == 1
        assert json.loads(_meta_path(ds2, "s1").read_text())["labels_revision"] == 2
        # and the next read comes from the cache
        _, ds3 = _make(root, cache, rev=2, value=9)
        assert ds3._load_or_compute_labels(ds3._subjects[0]).tolist() == [3] * N_EPOCHS


def test_cache_without_the_key_counts_as_revision_one(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    _subject(root, "s1", ["C4-M2"])
    with tempfile.TemporaryDirectory() as cache:
        _, ds = _make(root, cache, rev=1, value=0)
        ds._load_or_compute_labels(ds._subjects[0])
        mp = _meta_path(ds, "s1")
        meta = json.loads(mp.read_text()); meta.pop("labels_revision"); mp.write_text(json.dumps(meta))
        cls1, ds1 = _make(root, cache, rev=1, value=3)         # same revision: cache stays valid
        assert ds1._load_or_compute_labels(ds1._subjects[0]).tolist() == [0] * N_EPOCHS and cls1.reads == 0
        cls2, ds2 = _make(root, cache, rev=2, value=3)         # newer reader: recomputed
        assert ds2._load_or_compute_labels(ds2._subjects[0]).tolist() == [3] * N_EPOCHS and cls2.reads == 1


def test_stale_cache_is_used_with_a_warning_when_recompute_is_impossible(tmp_path, caplog):
    root = tmp_path / "raw"; root.mkdir()
    _subject(root, "s1", ["C4-M2"])
    with tempfile.TemporaryDirectory() as cache:
        _, ds = _make(root, cache, rev=1, value=0)
        ds._load_or_compute_labels(ds._subjects[0])
        _, ds2 = _make(root, cache, rev=2, value=3, fail=True)
        with caplog.at_level(logging.WARNING, logger="physioex.data"):
            out = ds2._load_or_compute_labels(ds2._subjects[0])
        assert out.tolist() == [0] * N_EPOCHS
        assert "stale" in caplog.text


def test_label_revisions_of_the_fixed_readers():
    from physioex.data.datasets.alzheimers import AlzheimersDataset
    from physioex.data.datasets.stages import STAGESDataset

    assert BasePhysioDataset.LABELS_REVISION == 1
    assert STAGESDataset.LABELS_REVISION >= 2 and AlzheimersDataset.LABELS_REVISION >= 2


# ── 2b. cached headers without start_sec ───────────────────────────────────────────────────────

def _header_path(ds, sid):
    return ds.cache.header_path(ds.DATASET_NAME, sid)


def test_header_cache_without_start_sec_is_refreshed(tmp_path):
    root = tmp_path / "raw"; root.mkdir()
    _subject(root, "s1", ["C4-M2"])
    with tempfile.TemporaryDirectory() as cache:
        _, ds = _make(root, cache, rev=1, value=0)
        hp = _header_path(ds, "s1")
        meta = json.loads(hp.read_text()); assert "start_sec" in meta
        meta.pop("start_sec"); hp.write_text(json.dumps(meta))      # a header written by an older version
        _, ds2 = _make(root, cache, rev=1, value=0)
        assert "start_sec" in json.loads(hp.read_text())            # re-probed and rewritten
        assert hasattr(ds2._headers["s1"], "start_sec")


def test_stale_header_is_used_with_a_warning_when_it_cannot_be_refreshed(tmp_path, caplog):
    root = tmp_path / "raw"; root.mkdir()
    _subject(root, "s1", ["C4-M2"])
    with tempfile.TemporaryDirectory() as cache:
        _, ds = _make(root, cache, rev=1, value=0)
        hp = _header_path(ds, "s1")
        meta = json.loads(hp.read_text()); meta.pop("start_sec"); hp.write_text(json.dumps(meta))

        class _NoRaw(_Fake):
            def _read_edf_header(self, spec):
                raise OSError("raw file not available")

        with caplog.at_level(logging.WARNING, logger="physioex.data"):
            ds3 = _NoRaw(root=str(root), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache)
        assert ds3.get_subjects() == ["s1"] and "predates" in caplog.text


# ── 3. artifact events ─────────────────────────────────────────────────────────────────────────

def _nsrr_xml(path):
    def ev(etype, concept, start, dur):
        return (f"<ScoredEvent><EventType>{etype}</EventType><EventConcept>{concept}</EventConcept>"
                f"<Start>{start}</Start><Duration>{dur}</Duration></ScoredEvent>")
    body = "".join([
        ev("Respiratory|Respiratory", "Hypopnea|Hypopnea", 10, 12),
        ev("Respiratory|Respiratory", "SpO2 artifact|SpO2 artifact", 40, 20),
        ev("Respiratory|Respiratory", "SpO2 desaturation|SpO2 desaturation", 70, 15),
        ev("Arousals|Arousals", "ASDA arousal|Arousal (ARO SPONT)", 100, 5),
    ])
    Path(path).write_text("<PSGAnnotation><ScoredEvents>" + body + "</ScoredEvents></PSGAnnotation>")


def test_nsrr_spo2_artifact_is_its_own_type(tmp_path):
    p = tmp_path / "x-nsrr.xml"
    _nsrr_xml(p)
    types = {e.concept: e.type for e in parse_nsrr_xml_events(p)}
    assert types["SpO2 artifact"] == "artifact"
    assert types["Hypopnea"] == "respiratory"
    assert types["SpO2 desaturation"] == "desaturation"
    assert types["ASDA arousal"] == "arousal"


def test_stages_spo2_artifact_is_its_own_type(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start Time", "Duration (seconds)", "Event"])
        for i, st in enumerate(["Wake", "Stage1", "Stage2"]):
            w.writerow([f"00:{i // 2:02d}:{(i % 2) * 30:02d}.000", "30", st])
        w.writerow(["00:00:10.000", "20", "SpO2 artifact"])
        w.writerow(["00:00:40.000", "20", "Desaturation"])
    _, events = parse_stages_csv(p)
    by = {e.concept: e.type for e in events}
    assert by["SpO2 artifact"] == "artifact"
    assert by["Desaturation"] == "desaturation"
