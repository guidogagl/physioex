"""Tests for the STAGES dataset module.

Unit tests use synthetic CSV data and require no real data files.
Real-data smoke tests are guarded behind PHYSIOEX_TEST_REAL_DATA=1.

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_stages_dataset.py
"""

import os
import sys
import tempfile
import csv
from pathlib import Path

passed, failed, skipped = 0, 0, 0

REAL_DATA = os.environ.get("PHYSIOEX_TEST_REAL_DATA", "0") == "1"


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


def skip(name, reason=""):
    global skipped
    skipped += 1
    suffix = f" -- {reason}" if reason else ""
    print(f"[SKIP] {name}{suffix}")


# ---------------------------------------------------------------------------
# Try to import the STAGES module -- it may still be under construction
# ---------------------------------------------------------------------------
try:
    from physioex.data.datasets.stages import STAGESDataset, STAGES_STAGE_MAP, SITES as STAGES_SITES
    from physioex.data.readers.annotations import parse_stages_csv
    STAGES_AVAILABLE = True
except ImportError as e:
    STAGES_AVAILABLE = False
    _import_err = str(e)


def require_stages(test_name):
    """Decorator/guard: skip test if stages module is not available yet."""
    if not STAGES_AVAILABLE:
        skip(test_name, f"stages module not importable: {_import_err}")
        return False
    return True


# ---------------------------------------------------------------------------
# Helper: write synthetic STAGES CSV
# ---------------------------------------------------------------------------
def write_stages_csv(path, rows, header=("Start Time", "Duration (seconds)", "Event")):
    """Write a STAGES-format CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 1. parse_stages_csv -- dur=0 format (BOGN-style)
# ---------------------------------------------------------------------------
def test_parse_dur0():
    if not require_stages("1. parse_stages_csv dur=0 format"):
        return
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Duration (seconds)", "Event"])
            # 5 epochs: W, N1, N2, N3, REM
            # dur=0 format: each line is one 30s epoch, duration column is 0
            for i, stage in enumerate(["Wake", "Stage1", "Stage2", "Stage3", "REM"]):
                t = i * 30
                h, m, s = t // 3600, (t % 3600) // 60, t % 60
                time_str = f"{h:02d}:{m:02d}:{s:02d}.000"
                writer.writerow([time_str, "0", stage])

        try:
            labels, events = parse_stages_csv(path)
            assert len(labels) == 5, f"expected 5 labels, got {len(labels)}"
            assert labels[0] == 0, f"epoch 0 should be Wake(0), got {labels[0]}"
            assert labels[1] == 1, f"epoch 1 should be N1(1), got {labels[1]}"
            assert labels[2] == 2, f"epoch 2 should be N2(2), got {labels[2]}"
            assert labels[3] == 3, f"epoch 3 should be N3(3), got {labels[3]}"
            assert labels[4] == 4, f"epoch 4 should be REM(4), got {labels[4]}"
            report("1. parse_stages_csv dur=0 format", True)
        finally:
            os.unlink(path)
    except Exception as exc:
        report("1. parse_stages_csv dur=0 format", False, str(exc))


# ---------------------------------------------------------------------------
# 2. parse_stages_csv -- dur=30 format
# ---------------------------------------------------------------------------
def test_parse_dur30():
    if not require_stages("2. parse_stages_csv dur=30 format"):
        return
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Duration (seconds)", "Event"])
            for i, stage in enumerate(["Wake", "Stage1", "Stage2"]):
                t = i * 30
                h, m, s = t // 3600, (t % 3600) // 60, t % 60
                time_str = f"{h:02d}:{m:02d}:{s:02d}.000"
                writer.writerow([time_str, "30", stage])

        try:
            labels, events = parse_stages_csv(path)
            assert len(labels) == 3, f"expected 3 labels, got {len(labels)}"
            assert labels[0] == 0
            assert labels[1] == 1
            assert labels[2] == 2
            report("2. parse_stages_csv dur=30 format", True)
        finally:
            os.unlink(path)
    except Exception as exc:
        report("2. parse_stages_csv dur=30 format", False, str(exc))


# ---------------------------------------------------------------------------
# 3. parse_stages_csv -- block format (variable durations)
# ---------------------------------------------------------------------------
def test_parse_block():
    if not require_stages("3. parse_stages_csv block format"):
        return
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Duration (seconds)", "Event"])
            # Block of 90s of Wake = 3 epochs
            writer.writerow(["00:00:00.000", "90", "Wake"])
            # Block of 60s of N2 = 2 epochs
            writer.writerow(["00:01:30.000", "60", "Stage2"])

        try:
            labels, events = parse_stages_csv(path)
            assert len(labels) == 5, f"expected 5 labels, got {len(labels)}"
            assert all(l == 0 for l in labels[:3]), f"first 3 should be Wake, got {labels[:3]}"
            assert all(l == 2 for l in labels[3:5]), f"last 2 should be N2, got {labels[3:5]}"
            report("3. parse_stages_csv block format", True)
        finally:
            os.unlink(path)
    except Exception as exc:
        report("3. parse_stages_csv block format", False, str(exc))


# ---------------------------------------------------------------------------
# 4. parse_stages_csv -- events extraction
# ---------------------------------------------------------------------------
def test_parse_events():
    if not require_stages("4. parse_stages_csv events extraction"):
        return
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Duration (seconds)", "Event"])
            # Mix of stages and events
            writer.writerow(["00:00:00.000", "0", "Wake"])
            writer.writerow(["00:00:30.000", "0", "Stage2"])
            writer.writerow(["00:00:15.000", "10", "Hypopnea"])
            writer.writerow(["00:00:40.000", "5", "Arousal"])
            writer.writerow(["00:01:00.000", "0", "Stage3"])

        try:
            labels, events = parse_stages_csv(path)
            assert len(labels) == 3, f"expected 3 stage labels, got {len(labels)}"
            assert labels[0] == 0, f"epoch 0: Wake expected, got {labels[0]}"
            assert labels[1] == 2, f"epoch 1: N2 expected, got {labels[1]}"
            assert labels[2] == 3, f"epoch 2: N3 expected, got {labels[2]}"

            # Events should include the non-stage entries
            assert len(events) >= 2, f"expected at least 2 events, got {len(events)}"
            concepts = [e.concept for e in events]
            assert "Hypopnea" in concepts, f"Hypopnea not found in events: {concepts}"
            assert "Arousal" in concepts, f"Arousal not found in events: {concepts}"
            report("4. parse_stages_csv events extraction", True)
        finally:
            os.unlink(path)
    except Exception as exc:
        report("4. parse_stages_csv events extraction", False, str(exc))


# ---------------------------------------------------------------------------
# 5. parse_stages_csv -- midnight crossing
# ---------------------------------------------------------------------------
def test_parse_midnight():
    if not require_stages("5. parse_stages_csv midnight crossing"):
        return
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Duration (seconds)", "Event"])
            # 23:59:30 -> 00:00:00 -> 00:00:30
            writer.writerow(["23:59:30.000", "0", "Wake"])
            writer.writerow(["00:00:00.000", "0", "Stage1"])
            writer.writerow(["00:00:30.000", "0", "Stage2"])

        try:
            labels, events = parse_stages_csv(path)
            assert len(labels) == 3, f"expected 3 labels with midnight crossing, got {len(labels)}"
            assert labels[0] == 0, f"epoch 0 should be Wake, got {labels[0]}"
            assert labels[1] == 1, f"epoch 1 should be N1, got {labels[1]}"
            assert labels[2] == 2, f"epoch 2 should be N2, got {labels[2]}"
            report("5. parse_stages_csv midnight crossing", True)
        finally:
            os.unlink(path)
    except Exception as exc:
        report("5. parse_stages_csv midnight crossing", False, str(exc))


# ---------------------------------------------------------------------------
# 6. parse_stages_csv -- anomalous duration clamping
# ---------------------------------------------------------------------------
def test_parse_anomalous_duration():
    if not require_stages("6. parse_stages_csv anomalous duration clamping"):
        return
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["Start Time", "Duration (seconds)", "Event"])
            # Anomalous: duration > 43200 should be treated as single epoch
            writer.writerow(["00:00:00.000", "86400", "Wake"])
            writer.writerow(["00:00:30.000", "0", "Stage2"])

        try:
            labels, events = parse_stages_csv(path)
            # The anomalous 86400s duration should be clamped to 1 epoch, not expand to 2880 epochs
            assert len(labels) == 2, f"expected 2 labels (anomalous dur clamped), got {len(labels)}"
            assert labels[0] == 0, f"epoch 0 should be Wake, got {labels[0]}"
            assert labels[1] == 2, f"epoch 1 should be N2, got {labels[1]}"
            report("6. parse_stages_csv anomalous duration clamping", True)
        finally:
            os.unlink(path)
    except Exception as exc:
        report("6. parse_stages_csv anomalous duration clamping", False, str(exc))


# ---------------------------------------------------------------------------
# 7. Stage map
# ---------------------------------------------------------------------------
def test_stage_map():
    if not require_stages("7. STAGES stage map"):
        return
    try:
        expected = {
            "Wake": 0, "Stage1": 1, "Stage2": 2, "Stage3": 3, "REM": 4,
        }
        for name, val in expected.items():
            assert STAGES_STAGE_MAP.get(name) == val, (
                f"stage map {name}: expected {val}, got {STAGES_STAGE_MAP.get(name)}"
            )
        # Unknown / Movement should map to -1
        for unknown in ["UnknownStage", "MT"]:
            mapped = STAGES_STAGE_MAP.get(unknown, -1)
            assert mapped == -1, f"{unknown} should map to -1, got {mapped}"
        report("7. STAGES stage map values", True)
    except Exception as exc:
        report("7. STAGES stage map values", False, str(exc))


# ---------------------------------------------------------------------------
# 8. CHANNEL_PREFERENCES valid
# ---------------------------------------------------------------------------
def test_channel_preferences():
    if not require_stages("8. STAGESDataset CHANNEL_PREFERENCES"):
        return
    try:
        prefs = STAGESDataset.CHANNEL_PREFERENCES
        assert isinstance(prefs, dict), f"CHANNEL_PREFERENCES is {type(prefs)}"
        # Should cover at least EEG and EOG modalities
        assert "EEG" in prefs, f"EEG not in CHANNEL_PREFERENCES"
        assert "EOG" in prefs, f"EOG not in CHANNEL_PREFERENCES"
        # Preference list should be non-empty
        assert len(prefs["EEG"]) > 0, "EEG preference list is empty"
        assert len(prefs["EOG"]) > 0, "EOG preference list is empty"
        # Should contain entries for multiple naming conventions
        eeg_str = " ".join(str(x) for x in prefs["EEG"])
        assert len(prefs["EEG"]) >= 4, (
            f"EEG preferences should have at least 4 entries for naming conventions, "
            f"got {len(prefs['EEG'])}"
        )
        report("8. STAGESDataset CHANNEL_PREFERENCES valid", True)
    except Exception as exc:
        report("8. STAGESDataset CHANNEL_PREFERENCES valid", False, str(exc))


# ---------------------------------------------------------------------------
# 9. SITES list
# ---------------------------------------------------------------------------
def test_sites_list():
    if not require_stages("9. STAGESDataset SITES list"):
        return
    try:
        sites = STAGES_SITES
        assert isinstance(sites, (list, tuple)), f"SITES is {type(sites)}"
        assert len(sites) == 13, f"expected 13 STAGES sites, got {len(sites)}"
        # Verify a few known sites
        sites_set = set(sites)
        for expected_site in ["BOGN", "GSDV", "MSMI"]:
            assert expected_site in sites_set, f"{expected_site} not in SITES: {sites}"
        report("9. STAGESDataset SITES list (13 sites)", True)
    except Exception as exc:
        report("9. STAGESDataset SITES list", False, str(exc))


# ---------------------------------------------------------------------------
# 10. Dynamic DATASET_NAME
# ---------------------------------------------------------------------------
def test_dynamic_name():
    if not require_stages("10. STAGESDataset dynamic DATASET_NAME"):
        return
    try:
        # The default DATASET_NAME should include "stages"
        default_name = STAGESDataset.DATASET_NAME
        assert "stages" in default_name.lower(), (
            f"DATASET_NAME should contain 'stages', got {default_name!r}"
        )
        report("10. STAGESDataset DATASET_NAME contains 'stages'", True)
    except Exception as exc:
        report("10. STAGESDataset dynamic DATASET_NAME", False, str(exc))


# ===========================================================================
# Real-data smoke tests (guarded behind PHYSIOEX_TEST_REAL_DATA=1)
# ===========================================================================

def test_real_gsdv_instantiation():
    """11. Instantiate STAGESDataset with GSDV site, verify subject discovery."""
    if not REAL_DATA:
        skip("11. GSDV instantiation", "PHYSIOEX_TEST_REAL_DATA not set")
        return
    if not require_stages("11. GSDV instantiation"):
        return
    try:
        ds = STAGESDataset(site="GSDV", pipelines="time_domain")
        n = ds.get_n_subjects()
        assert n > 0, f"expected > 0 subjects, got {n}"
        subjects = ds.get_subjects()
        assert len(subjects) == n
        report("11. GSDV instantiation + subject discovery", True, f"{n} subjects")
    except Exception as exc:
        report("11. GSDV instantiation + subject discovery", False, str(exc))


def test_real_gsdv_load_item():
    """12. Load item[0] from GSDV, verify dict structure."""
    if not REAL_DATA:
        skip("12. GSDV load item[0]", "PHYSIOEX_TEST_REAL_DATA not set")
        return
    if not require_stages("12. GSDV load item[0]"):
        return
    try:
        ds = STAGESDataset(site="GSDV", pipelines="time_domain",
                           channels=["EEG", "EOG"], sequence_length=5)
        item = ds[0]
        assert "signals" in item, "missing 'signals'"
        assert "labels" in item, "missing 'labels'"
        assert "events" in item, "missing 'events'"
        assert "channel_order" in item, "missing 'channel_order'"
        assert "subject" in item, "missing 'subject'"
        assert item["labels"].shape[0] == 5, f"labels shape: {item['labels'].shape}"
        assert len(item["events"]) == 5, f"events length: {len(item['events'])}"
        report("12. GSDV load item[0] dict structure", True)
    except Exception as exc:
        report("12. GSDV load item[0] dict structure", False, str(exc))


def test_real_gsdv_events_populated():
    """13. Verify GSDV events are populated (respiratory events expected)."""
    if not REAL_DATA:
        skip("13. GSDV events populated", "PHYSIOEX_TEST_REAL_DATA not set")
        return
    if not require_stages("13. GSDV events populated"):
        return
    try:
        ds = STAGESDataset(site="GSDV", pipelines="time_domain",
                           channels=["EEG"], sequence_length=21)
        subjects = ds.get_subjects()
        # Check first few subjects for events
        found_events = False
        for sid in subjects[:5]:
            evts = ds.get_subject_events(sid)
            if evts:
                found_events = True
                break
        assert found_events, "No events found in any of the first 5 GSDV subjects"
        report("13. GSDV events populated", True, f"subject {sid} has {len(evts)} events")
    except Exception as exc:
        report("13. GSDV events populated", False, str(exc))


def test_real_gsdv_subject_metadata():
    """14. Verify get_subject_metadata returns expected fields."""
    if not REAL_DATA:
        skip("14. GSDV subject metadata", "PHYSIOEX_TEST_REAL_DATA not set")
        return
    if not require_stages("14. GSDV subject metadata"):
        return
    try:
        ds = STAGESDataset(site="GSDV", pipelines="time_domain",
                           channels=["EEG"], sequence_length=5)
        subjects = ds.get_subjects()
        meta = ds.get_subject_metadata(subjects[0])
        assert "id" in meta, f"missing 'id' in metadata"
        assert "dataset" in meta, f"missing 'dataset' in metadata"
        assert "stages" in meta["dataset"].lower(), (
            f"dataset should contain 'stages', got {meta['dataset']!r}"
        )
        report("14. GSDV subject metadata", True, f"keys: {list(meta.keys())}")
    except Exception as exc:
        report("14. GSDV subject metadata", False, str(exc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running STAGES dataset tests")
    print("=" * 60)

    # Unit tests (synthetic data)
    test_parse_dur0()
    test_parse_dur30()
    test_parse_block()
    test_parse_events()
    test_parse_midnight()
    test_parse_anomalous_duration()
    test_stage_map()
    test_channel_preferences()
    test_sites_list()
    test_dynamic_name()

    # Real-data smoke tests
    test_real_gsdv_instantiation()
    test_real_gsdv_load_item()
    test_real_gsdv_events_populated()
    test_real_gsdv_subject_metadata()

    print("=" * 60)
    total = passed + failed
    msg = f"Results: {passed} passed, {failed} failed"
    if skipped > 0:
        msg += f", {skipped} skipped"
    msg += f" out of {total + skipped}"
    print(msg)
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
