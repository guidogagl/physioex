"""Tests for the per-epoch event metadata system.

Covers:
  - SleepEvent creation and attributes
  - event_to_dict / events_to_dicts / dicts_to_events serialization round-trip
  - map_events_to_epochs: basic, spanning, boundary, zero-duration, empty, beyond
  - BasePhysioDataset integration: events key, collate_fn, get_subjects, caching

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_events.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch

from physioex.data.events import (
    SleepEvent,
    event_to_dict,
    events_to_dicts,
    dicts_to_events,
    map_events_to_epochs,
)
from physioex.data.collate import dict_collate_fn
from physioex.data.base import BasePhysioDataset, SubjectSpec

passed, failed = 0, 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


# ---------------------------------------------------------------------------
# 1. SleepEvent creation
# ---------------------------------------------------------------------------
def test_sleep_event_creation():
    try:
        ev = SleepEvent(
            type="respiratory",
            concept="Hypopnea",
            onset_sec=120.5,
            duration_sec=15.0,
            extra={"desaturation": 4},
        )
        assert ev.type == "respiratory", f"type: {ev.type!r}"
        assert ev.concept == "Hypopnea", f"concept: {ev.concept!r}"
        assert ev.onset_sec == 120.5, f"onset_sec: {ev.onset_sec}"
        assert ev.duration_sec == 15.0, f"duration_sec: {ev.duration_sec}"
        assert ev.extra == {"desaturation": 4}, f"extra: {ev.extra}"
        report("1. SleepEvent creation with all fields", True)
    except Exception as exc:
        report("1. SleepEvent creation with all fields", False, str(exc))


def test_sleep_event_defaults():
    try:
        ev = SleepEvent(type="arousal", concept="ASDA arousal",
                        onset_sec=0.0, duration_sec=3.0)
        assert ev.extra == {}, f"default extra should be empty dict, got {ev.extra!r}"
        report("1b. SleepEvent default extra field", True)
    except Exception as exc:
        report("1b. SleepEvent default extra field", False, str(exc))


# ---------------------------------------------------------------------------
# 2. event_to_dict / events_to_dicts
# ---------------------------------------------------------------------------
def test_event_to_dict():
    try:
        ev = SleepEvent(
            type="limb_movement", concept="PLM (Left)",
            onset_sec=45.0, duration_sec=2.0,
            extra={"severity": "mild"},
        )
        d = event_to_dict(ev)
        assert d["type"] == "limb_movement", f"type: {d['type']!r}"
        assert d["concept"] == "PLM (Left)", f"concept: {d['concept']!r}"
        assert d["onset_sec"] == 45.0, f"onset_sec: {d['onset_sec']}"
        assert d["duration_sec"] == 2.0, f"duration_sec: {d['duration_sec']}"
        assert d["extra"] == {"severity": "mild"}, f"extra: {d.get('extra')}"
        report("2a. event_to_dict with extra field", True)
    except Exception as exc:
        report("2a. event_to_dict with extra field", False, str(exc))


def test_event_to_dict_no_extra():
    try:
        ev = SleepEvent(type="arousal", concept="Spontaneous",
                        onset_sec=10.0, duration_sec=5.0)
        d = event_to_dict(ev)
        assert "extra" not in d, f"extra should not be in dict for empty extra, got {d}"
        report("2b. event_to_dict omits empty extra", True)
    except Exception as exc:
        report("2b. event_to_dict omits empty extra", False, str(exc))


def test_events_to_dicts():
    try:
        events = [
            SleepEvent(type="respiratory", concept="Apnea",
                       onset_sec=60.0, duration_sec=10.0),
            SleepEvent(type="arousal", concept="RERA",
                       onset_sec=90.0, duration_sec=3.0,
                       extra={"associated_event": "Hypopnea"}),
        ]
        dicts = events_to_dicts(events)
        assert len(dicts) == 2
        assert dicts[0]["concept"] == "Apnea"
        assert dicts[1]["extra"]["associated_event"] == "Hypopnea"
        report("2c. events_to_dicts", True)
    except Exception as exc:
        report("2c. events_to_dicts", False, str(exc))


# ---------------------------------------------------------------------------
# 3. dicts_to_events round-trip
# ---------------------------------------------------------------------------
def test_round_trip():
    try:
        original = [
            SleepEvent(type="respiratory", concept="Hypopnea",
                       onset_sec=120.0, duration_sec=15.0,
                       extra={"SpO2_nadir": 88}),
            SleepEvent(type="arousal", concept="ASDA arousal",
                       onset_sec=135.0, duration_sec=3.0),
            SleepEvent(type="desaturation", concept="SpO2 desat",
                       onset_sec=125.0, duration_sec=20.0,
                       extra={"nadir": 85, "baseline": 95}),
        ]
        dicts = events_to_dicts(original)
        restored = dicts_to_events(dicts)
        assert len(restored) == len(original), f"length: {len(restored)} vs {len(original)}"
        for orig, rest in zip(original, restored):
            assert orig.type == rest.type, f"type: {orig.type} vs {rest.type}"
            assert orig.concept == rest.concept, f"concept: {orig.concept} vs {rest.concept}"
            assert orig.onset_sec == rest.onset_sec, f"onset: {orig.onset_sec} vs {rest.onset_sec}"
            assert orig.duration_sec == rest.duration_sec, f"dur: {orig.duration_sec} vs {rest.duration_sec}"
            assert orig.extra == rest.extra, f"extra: {orig.extra} vs {rest.extra}"
        report("3. dicts_to_events round-trip", True)
    except Exception as exc:
        report("3. dicts_to_events round-trip", False, str(exc))


# ---------------------------------------------------------------------------
# 4. map_events_to_epochs -- basic (within one epoch)
# ---------------------------------------------------------------------------
def test_map_basic():
    try:
        # Epoch 0: [0, 30), Epoch 1: [30, 60), ...
        # Event at onset=45, dur=10 -> [45, 55) lies entirely within epoch 1
        ev = SleepEvent(type="respiratory", concept="Hypopnea",
                        onset_sec=45.0, duration_sec=10.0)
        result = map_events_to_epochs([ev], n_epochs=4, epoch_length_sec=30.0)
        assert len(result) == 4, f"expected 4 epoch lists, got {len(result)}"
        assert len(result[0]) == 0, f"epoch 0 should be empty, got {result[0]}"
        assert len(result[1]) == 1, f"epoch 1 should have 1 event, got {len(result[1])}"
        assert result[1][0]["concept"] == "Hypopnea"
        assert len(result[2]) == 0
        assert len(result[3]) == 0
        report("4. map_events_to_epochs -- basic (within epoch 1)", True)
    except Exception as exc:
        report("4. map_events_to_epochs -- basic (within epoch 1)", False, str(exc))


# ---------------------------------------------------------------------------
# 5. map_events_to_epochs -- spanning two epochs
# ---------------------------------------------------------------------------
def test_map_spanning():
    try:
        # Event at onset=25, dur=15 -> [25, 40) spans epoch 0 [0,30) and epoch 1 [30,60)
        ev = SleepEvent(type="arousal", concept="Arousal",
                        onset_sec=25.0, duration_sec=15.0)
        result = map_events_to_epochs([ev], n_epochs=3, epoch_length_sec=30.0)
        assert len(result[0]) == 1, f"epoch 0 should have event, got {len(result[0])}"
        assert len(result[1]) == 1, f"epoch 1 should have event, got {len(result[1])}"
        assert len(result[2]) == 0, f"epoch 2 should be empty, got {len(result[2])}"
        report("5. map_events_to_epochs -- spanning epoch 0 and 1", True)
    except Exception as exc:
        report("5. map_events_to_epochs -- spanning epoch 0 and 1", False, str(exc))


# ---------------------------------------------------------------------------
# 6. map_events_to_epochs -- boundary (event starts at epoch boundary)
# ---------------------------------------------------------------------------
def test_map_boundary():
    try:
        # Event at onset=30.0, dur=5 -> [30.0, 35.0) lies in epoch 1 [30,60), NOT epoch 0
        ev = SleepEvent(type="respiratory", concept="Apnea",
                        onset_sec=30.0, duration_sec=5.0)
        result = map_events_to_epochs([ev], n_epochs=3, epoch_length_sec=30.0)
        assert len(result[0]) == 0, f"epoch 0 should be empty (boundary), got {len(result[0])}"
        assert len(result[1]) == 1, f"epoch 1 should have 1 event, got {len(result[1])}"
        report("6. map_events_to_epochs -- boundary (onset=30.0)", True)
    except Exception as exc:
        report("6. map_events_to_epochs -- boundary (onset=30.0)", False, str(exc))


# ---------------------------------------------------------------------------
# 7. map_events_to_epochs -- zero duration
# ---------------------------------------------------------------------------
def test_map_zero_duration():
    try:
        # Event at onset=15, dur=0 -> treated as instantaneous at t=15, epoch 0 [0,30)
        ev = SleepEvent(type="other", concept="Marker",
                        onset_sec=15.0, duration_sec=0.0)
        result = map_events_to_epochs([ev], n_epochs=3, epoch_length_sec=30.0)
        assert len(result[0]) == 1, f"epoch 0 should have zero-dur event, got {len(result[0])}"
        assert len(result[1]) == 0
        report("7. map_events_to_epochs -- zero duration (onset=15)", True)
    except Exception as exc:
        report("7. map_events_to_epochs -- zero duration (onset=15)", False, str(exc))


# ---------------------------------------------------------------------------
# 8. map_events_to_epochs -- empty event list
# ---------------------------------------------------------------------------
def test_map_empty():
    try:
        result = map_events_to_epochs([], n_epochs=5, epoch_length_sec=30.0)
        assert len(result) == 5, f"expected 5 epoch lists, got {len(result)}"
        for i, ep in enumerate(result):
            assert ep == [], f"epoch {i} should be empty, got {ep}"
        report("8. map_events_to_epochs -- empty event list", True)
    except Exception as exc:
        report("8. map_events_to_epochs -- empty event list", False, str(exc))


# ---------------------------------------------------------------------------
# 9. map_events_to_epochs -- event beyond n_epochs
# ---------------------------------------------------------------------------
def test_map_beyond():
    try:
        # Event at onset=1000 with n_epochs=10 (max time = 300 sec)
        ev = SleepEvent(type="respiratory", concept="Late",
                        onset_sec=1000.0, duration_sec=5.0)
        result = map_events_to_epochs([ev], n_epochs=10, epoch_length_sec=30.0)
        # Should not crash; event simply doesn't appear in any epoch
        assert len(result) == 10
        total_events = sum(len(ep) for ep in result)
        assert total_events == 0, f"expected 0 events (beyond range), got {total_events}"
        report("9. map_events_to_epochs -- event beyond n_epochs (no crash)", True)
    except Exception as exc:
        report("9. map_events_to_epochs -- event beyond n_epochs", False, str(exc))


# ---------------------------------------------------------------------------
# 10. BasePhysioDataset events key in __getitem__
# ---------------------------------------------------------------------------
def _write_fake_edf(path, n_channels=2, duration_sec=300, channel_names=None, fs=100, seed=0):
    """Minimal EDF writer for integration tests."""
    import pyedflib
    if channel_names is None:
        channel_names = ["C4-M2", "EOG"][:n_channels]
    rng = np.random.default_rng(seed)
    n_records = int(duration_sec)
    signals = []
    headers = []
    for i in range(n_channels):
        t = np.arange(n_records * fs) / fs
        x = rng.standard_normal(n_records * fs).astype(np.float64) * 20.0
        signals.append(x)
        headers.append({
            "label": channel_names[i],
            "dimension": "uV",
            "sample_frequency": fs,
            "physical_min": -300.0,
            "physical_max": 300.0,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        })
    writer = pyedflib.EdfWriter(str(path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(headers)
        writer.writeSamples(signals)
    finally:
        writer.close()


def _write_fake_annotations_edf(path, stages, epoch_sec=30.0):
    import pyedflib
    total_duration = int(len(stages) * epoch_sec)
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders([{
            "label": "dummy",
            "dimension": "uV",
            "sample_frequency": 1,
            "physical_min": -1.0, "physical_max": 1.0,
            "digital_min": -32768, "digital_max": 32767,
            "transducer": "", "prefilter": "",
        }])
        writer.writeSamples([np.zeros(total_duration, dtype=np.float64)])
        for i, s in enumerate(stages):
            writer.writeAnnotation(i * epoch_sec, epoch_sec, s)
    finally:
        writer.close()


class FakeEventDataset(BasePhysioDataset):
    """Test subclass that returns synthetic events."""
    DATASET_NAME = "fake_events_test"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0
    CHANNEL_PREFERENCES = {
        "EEG": [("C4", "M2"), "C4-M2", "EEG"],
        "EOG": ["EOG"],
    }

    def __init__(self, root, subject_id, events=None, **kwargs):
        self._fake_subject_id = subject_id
        self._fake_events = events or []
        super().__init__(root=root, **kwargs)

    def _list_subjects(self):
        import pyedflib
        root = Path(self.root)
        return [SubjectSpec(
            subject_id=self._fake_subject_id,
            edf_path=root / f"{self._fake_subject_id}.edf",
            label_path=root / f"{self._fake_subject_id}_sleepscoring.edf",
        )]

    def _read_subject_labels(self, spec):
        import pyedflib
        with pyedflib.EdfReader(str(spec.label_path)) as f:
            onsets, durations, stages = f.readAnnotations()
        if len(stages) == 0:
            return np.array([], dtype=np.int16)
        stage_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
        total = max(float(o) + float(d) for o, d in zip(onsets, durations))
        n_epochs = int(total // self.epoch_length_sec)
        labels = np.full(n_epochs, -1, dtype=np.int16)
        for onset, duration, stage_str in zip(onsets, durations, stages):
            i0 = int(round(float(onset) / self.epoch_length_sec))
            i1 = int(round((float(onset) + float(duration)) / self.epoch_length_sec))
            if i1 > n_epochs:
                i1 = n_epochs
            labels[i0:i1] = stage_map.get(str(stage_str).strip(), -1)
        return labels

    def _read_subject_events(self, spec):
        return list(self._fake_events)


def test_getitem_events_key():
    try:
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
            data_dir = Path(data_dir)
            # 300s = 10 epochs of 30s
            _write_fake_edf(data_dir / "SUB01.edf", duration_sec=300)
            _write_fake_annotations_edf(
                data_dir / "SUB01_sleepscoring.edf",
                stages=["W", "N1", "N2", "N3", "R", "W", "N1", "N2", "N3", "R"],
            )

            events = [
                SleepEvent(type="respiratory", concept="Hypopnea",
                           onset_sec=45.0, duration_sec=10.0),
                SleepEvent(type="arousal", concept="ASDA arousal",
                           onset_sec=150.0, duration_sec=3.0),
            ]

            ds = FakeEventDataset(
                root=str(data_dir), subject_id="SUB01",
                events=events,
                channels=["EEG", "EOG"],
                pipelines="raw",
                sequence_length=5,
                cache_dir=cache_dir,
            )

            item = ds[0]
            assert "events" in item, "item should have 'events' key"
            assert isinstance(item["events"], list), f"events should be list, got {type(item['events'])}"
            assert len(item["events"]) == 5, f"events length should match sequence_length=5, got {len(item['events'])}"
            # Epoch 1 (index 1 in the sequence starting at 0) should have the hypopnea
            assert len(item["events"][1]) == 1, f"epoch 1 should have 1 event, got {len(item['events'][1])}"
            assert item["events"][1][0]["concept"] == "Hypopnea"
            report("10. BasePhysioDataset __getitem__ returns events key", True)
    except Exception as exc:
        report("10. BasePhysioDataset __getitem__ returns events key", False, str(exc))


# ---------------------------------------------------------------------------
# 11. Events in collate_fn
# ---------------------------------------------------------------------------
def test_events_in_collate():
    try:
        # Build two fake batch items with events
        item1 = {
            "signals": {"EEG": torch.randn(3, 3000)},
            "channel_order": ["EEG"],
            "labels": torch.randint(0, 5, (3,)),
            "events": [
                [{"type": "respiratory", "concept": "Apnea", "onset_sec": 5, "duration_sec": 10}],
                [],
                [{"type": "arousal", "concept": "ASDA", "onset_sec": 65, "duration_sec": 3}],
            ],
        }
        item2 = {
            "signals": {"EEG": torch.randn(3, 3000)},
            "channel_order": ["EEG"],
            "labels": torch.randint(0, 5, (3,)),
            "events": [[], [], []],
        }

        out = dict_collate_fn([item1, item2])
        assert "events" in out, "collated batch should have 'events'"
        assert isinstance(out["events"], list), f"events should be list, got {type(out['events'])}"
        assert len(out["events"]) == 2, f"batch size 2, events length: {len(out['events'])}"
        # Each entry is the per-epoch event list for that sample
        assert len(out["events"][0]) == 3, f"sample 0 should have 3 epoch lists"
        assert len(out["events"][0][0]) == 1, f"sample 0, epoch 0 should have 1 event"
        assert len(out["events"][1][0]) == 0, f"sample 1, epoch 0 should be empty"
        report("11. Events preserved in dict_collate_fn", True)
    except Exception as exc:
        report("11. Events preserved in dict_collate_fn", False, str(exc))


# ---------------------------------------------------------------------------
# 12. get_subjects / get_subject_metadata API
# ---------------------------------------------------------------------------
def test_get_subjects_and_metadata():
    try:
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
            data_dir = Path(data_dir)
            _write_fake_edf(data_dir / "SUB01.edf", duration_sec=300)
            _write_fake_annotations_edf(
                data_dir / "SUB01_sleepscoring.edf",
                stages=["W"] * 10,
            )

            ds = FakeEventDataset(
                root=str(data_dir), subject_id="SUB01",
                channels=["EEG"],
                pipelines="raw",
                sequence_length=3,
                cache_dir=cache_dir,
            )

            subjects = ds.get_subjects()
            assert subjects == ["SUB01"], f"get_subjects: {subjects}"

            meta = ds.get_subject_metadata("SUB01")
            assert meta["id"] == "SUB01", f"metadata id: {meta.get('id')}"
            assert meta["dataset"] == "fake_events_test", f"dataset: {meta.get('dataset')}"
            report("12. get_subjects / get_subject_metadata API", True)
    except Exception as exc:
        report("12. get_subjects / get_subject_metadata API", False, str(exc))


# ---------------------------------------------------------------------------
# 13. Events caching to disk
# ---------------------------------------------------------------------------
def test_events_caching():
    try:
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
            data_dir = Path(data_dir)
            _write_fake_edf(data_dir / "SUB01.edf", duration_sec=300)
            _write_fake_annotations_edf(
                data_dir / "SUB01_sleepscoring.edf",
                stages=["W"] * 10,
            )

            events = [
                SleepEvent(type="respiratory", concept="Hypopnea",
                           onset_sec=45.0, duration_sec=10.0),
            ]

            ds = FakeEventDataset(
                root=str(data_dir), subject_id="SUB01",
                events=events,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=3,
                cache_dir=cache_dir,
                cache_enabled=True,
            )

            # Access an item to trigger event loading + caching
            _ = ds[0]

            # Verify events JSON was written to cache
            from physioex.data.cache import ChannelCache
            cache = ChannelCache(cache_dir)
            events_file = cache.events_path("fake_events_test", "SUB01")
            assert events_file.exists(), f"events cache file not found: {events_file}"

            # Verify content is valid JSON with the event
            import json
            with open(events_file) as f:
                cached_data = json.load(f)
            assert len(cached_data) == 1, f"expected 1 cached event, got {len(cached_data)}"
            assert cached_data[0]["concept"] == "Hypopnea"

            # Create a second dataset instance with same cache -- should NOT call _read_subject_events
            call_count = {"n": 0}
            ds2 = FakeEventDataset(
                root=str(data_dir), subject_id="SUB01",
                events=events,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=3,
                cache_dir=cache_dir,
                cache_enabled=True,
            )
            orig_read = ds2._read_subject_events
            def counting_read(spec):
                call_count["n"] += 1
                return orig_read(spec)
            ds2._read_subject_events = counting_read

            _ = ds2[0]
            assert call_count["n"] == 0, (
                f"expected 0 calls to _read_subject_events on cache hit, got {call_count['n']}"
            )
            report("13. Events cached to disk and reloaded", True)
    except Exception as exc:
        report("13. Events cached to disk and reloaded", False, str(exc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running per-epoch event metadata tests")
    print("=" * 60)

    test_sleep_event_creation()
    test_sleep_event_defaults()
    test_event_to_dict()
    test_event_to_dict_no_extra()
    test_events_to_dicts()
    test_round_trip()
    test_map_basic()
    test_map_spanning()
    test_map_boundary()
    test_map_zero_duration()
    test_map_empty()
    test_map_beyond()
    test_getitem_events_key()
    test_events_in_collate()
    test_get_subjects_and_metadata()
    test_events_caching()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
