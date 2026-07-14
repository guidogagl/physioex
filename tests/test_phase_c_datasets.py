"""Phase C integration tests: MESA, MrOS, and HomePAP (NSRR-family) dataset modules.

Creates synthetic fake EDF + NSRR XML files on-the-fly and verifies discovery,
channel resolution, label parsing, and end-to-end dict return for all three.

Real-data smoke tests are guarded by ``PHYSIOEX_TEST_REAL_DATA=1``.

Run standalone:
    python test/tests/test_phase_c_datasets.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from physioex.data.datasets import get_dataset, available_datasets
from physioex.data.datasets.mesa import MESADataset
from physioex.data.datasets.mros import MrOSDataset
from physioex.data.datasets.hpap import HPAPDataset
from physioex.data.readers.annotations import NSRR_STAGE_MAP
from tests.test_raw_dataset_integration import write_fake_edf


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

# NSRR EventConcept strings matching NSRR_STAGE_MAP keys.
NSRR_STAGE_CONCEPTS = {
    0: "Wake|0",
    1: "Stage 1 sleep|1",
    2: "Stage 2 sleep|2",
    3: "Stage 3 sleep|3",
    4: "REM sleep|5",
    -1: "Unscored|9",
}


def write_fake_nsrr_xml(
    path: Path,
    stages_per_epoch: list,
    epoch_sec: float = 30.0,
):
    """Write a minimal NSRR-format XML annotation file.

    ``stages_per_epoch`` is a list of ints (0-4, or -1 for unscored),
    one per epoch.  Adjacent epochs with the same stage are merged into
    a single ScoredEvent for compactness (mirrors real NSRR XMLs).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        "<PSGAnnotation>",
        "  <ScoredEvents>",
    ]

    # Merge consecutive identical stages into runs
    if stages_per_epoch:
        runs = []
        cur_stage = stages_per_epoch[0]
        cur_start = 0
        for i in range(1, len(stages_per_epoch)):
            if stages_per_epoch[i] != cur_stage:
                runs.append((cur_start, i - cur_start, cur_stage))
                cur_stage = stages_per_epoch[i]
                cur_start = i
            # else continue accumulating
        runs.append((cur_start, len(stages_per_epoch) - cur_start, cur_stage))

        for start_epoch, n_epochs, stage in runs:
            concept = NSRR_STAGE_CONCEPTS.get(stage, "Unscored|9")
            start_sec = start_epoch * epoch_sec
            duration_sec = n_epochs * epoch_sec
            lines.append("    <ScoredEvent>")
            lines.append("      <EventType>Stages|Stages</EventType>")
            lines.append(f"      <EventConcept>{concept}</EventConcept>")
            lines.append(f"      <Start>{start_sec}</Start>")
            lines.append(f"      <Duration>{duration_sec}</Duration>")
            lines.append("    </ScoredEvent>")

    lines.append("  </ScoredEvents>")
    lines.append("</PSGAnnotation>")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_nsrr_subject(
    edf_dir: Path,
    xml_dir: Path,
    stem: str,
    stages: list,
    epoch_sec: float = 30.0,
    channel_names=None,
    fs_per_channel=None,
):
    """Create a paired fake EDF + NSRR XML for a single subject.

    ``stages`` is a list of int per-epoch labels (0-4 / -1).
    """
    if channel_names is None:
        channel_names = ["EEG1", "EOG-L", "EMG"]
    if fs_per_channel is None:
        fs_per_channel = [256] * len(channel_names)

    duration_sec = len(stages) * epoch_sec
    edf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    write_fake_edf(
        edf_dir / f"{stem}.edf",
        n_channels=len(channel_names),
        duration_sec=duration_sec,
        fs_per_channel=fs_per_channel,
        channel_names=channel_names,
    )
    write_fake_nsrr_xml(
        xml_dir / f"{stem}-nsrr.xml",
        stages_per_epoch=stages,
        epoch_sec=epoch_sec,
    )


# ===================================================================
# MESA synthetic tests
# ===================================================================

def test_mesa_registry():
    """1. get_dataset('mesa') returns MESADataset."""
    cls = get_dataset("mesa")
    ok = cls is MESADataset
    report("MESA registry registration", ok, f"got {cls}")


def test_mesa_in_available():
    """Verify 'mesa' appears in available_datasets()."""
    ok = "mesa" in available_datasets()
    report("MESA in available_datasets()", ok)


def test_mesa_empty_root():
    """2. Empty root -> 0 subjects, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = MESADataset(
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("MESA empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


def test_mesa_subject_discovery():
    """3. Discovers a single subject from synthetic EDF + NSRR XML."""
    stages = [0] * 5 + [2] * 5  # 10 epochs
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
            "mesa-sleep-0001",
            stages=stages,
        )
        ds = MESADataset(
            root=str(root),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = len(ds._subjects) == 1
        detail = f"subjects={[s.subject_id for s in ds._subjects]}"
        report("MESA discovers 1 subject", ok, detail)


def test_mesa_subject_id():
    """4. Subject ID equals EDF stem."""
    stages = [0] * 10
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
            "mesa-sleep-0001",
            stages=stages,
        )
        ds = MESADataset(
            root=str(root),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = ds._subjects[0].subject_id == "mesa-sleep-0001"
        report(
            "MESA subject_id == 'mesa-sleep-0001'",
            ok,
            f"got {ds._subjects[0].subject_id!r}",
        )


def test_mesa_missing_xml_skipped():
    """5. Missing XML -> subject not discovered."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        edf_dir = root / "polysomnography" / "edfs"
        edf_dir.mkdir(parents=True)
        xml_dir = root / "polysomnography" / "annotations-events-nsrr"
        xml_dir.mkdir(parents=True)
        # Write EDF but NO XML
        write_fake_edf(
            edf_dir / "mesa-sleep-9999.edf",
            n_channels=3, duration_sec=300,
            channel_names=["EEG1", "EOG-L", "EMG"],
            fs_per_channel=[256, 256, 256],
        )
        ds = MESADataset(
            root=str(root),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        ok = len(ds._subjects) == 0
        report("MESA missing XML -> subject skipped", ok, f"got {len(ds._subjects)}")


def test_mesa_label_parsing():
    """6. Label parsing matches stages written in XML."""
    stages = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4]  # 10 epochs
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
            "mesa-sleep-0001",
            stages=stages,
        )
        ds = MESADataset(
            root=str(root),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        ok_len = len(labels) == 10
        ok_vals = list(labels) == stages
        ok = ok_len and ok_vals
        report(
            "MESA label parsing (length + values)",
            ok,
            f"len={len(labels)} expected=10, vals={list(labels)}",
        )


def test_mesa_end_to_end():
    """7. ds[0] returns a valid dict."""
    stages = [0] * 5 + [4] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
            "mesa-sleep-0001",
            stages=stages,
        )
        ds = MESADataset(
            root=str(root),
            channels=["EEG"],
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
            len(item["channel_order"]) == 1,
            item["subject"]["dataset"] == "mesa",
        ]
        ok = all(checks)
        report("MESA end-to-end ds[0] dict", ok, f"checks={checks}")


# ===================================================================
# MrOS synthetic tests
# ===================================================================

def test_mros_registry():
    """8. get_dataset('mros') returns MrOSDataset."""
    cls = get_dataset("mros")
    ok = cls is MrOSDataset
    report("MrOS registry registration", ok, f"got {cls}")


def test_mros_empty_root():
    """9. Empty root -> 0 subjects."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = MrOSDataset(
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("MrOS empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


def test_mros_subject_discovery():
    """10. Subject discovery with a fake EDF+XML pair."""
    stages = [0] * 5 + [2] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
            "mros-visit1-aa0001",
            stages=stages,
            channel_names=["EEG", "EOG-L", "EMG"],
        )
        ds = MrOSDataset(
            root=str(root),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )
        ok = len(ds._subjects) == 1
        detail = f"subjects={[s.subject_id for s in ds._subjects]}"
        report("MrOS discovers 1 subject", ok, detail)


# ===================================================================
# HPAP synthetic tests
# ===================================================================

def test_hpap_registry():
    """11. get_dataset('hpap') returns HPAPDataset."""
    cls = get_dataset("hpap")
    ok = cls is HPAPDataset
    report("HPAP registry registration", ok, f"got {cls}")


def test_hpap_empty_root():
    """12. Empty root -> 0 subjects."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = HPAPDataset(
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("HPAP empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


def test_hpap_subset_enumeration():
    """13. Subset enumeration: lab-full only -> discovered by 'all' and 'lab-full', not 'home'."""
    stages = [0] * 5 + [2] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        # Create only lab/full files
        make_nsrr_subject(
            root / "polysomnography" / "edfs" / "lab" / "full",
            root / "polysomnography" / "annotations-events-nsrr" / "lab" / "full",
            "homepap-lab-full-1600001",
            stages=stages,
            channel_names=["EEG", "LOC", "EMG"],
        )
        # subset="all" should find it
        ds_all = HPAPDataset(
            root=str(root), subset="all",
            channels=["EEG"], pipelines="raw", sequence_length=5,
            cache_dir=os.path.join(cache_dir, "all"),
        )
        ok_all = len(ds_all._subjects) == 1

        # subset="lab-full" should also find it
        ds_lab = HPAPDataset(
            root=str(root), subset="lab-full",
            channels=["EEG"], pipelines="raw", sequence_length=5,
            cache_dir=os.path.join(cache_dir, "lab"),
        )
        ok_lab = len(ds_lab._subjects) == 1

        # subset="home" should find 0
        ds_home = HPAPDataset(
            root=str(root), subset="home",
            channels=["EEG"], pipelines="raw", sequence_length=5,
            cache_dir=os.path.join(cache_dir, "home"),
        )
        ok_home = len(ds_home._subjects) == 0

        ok = ok_all and ok_lab and ok_home
        report(
            "HPAP subset enumeration (all=1, lab-full=1, home=0)",
            ok,
            f"all={len(ds_all._subjects)}, lab-full={len(ds_lab._subjects)}, home={len(ds_home._subjects)}",
        )


def test_hpap_invalid_subset():
    """14. Invalid subset raises ValueError."""
    ok = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            HPAPDataset(
                root=tmp, subset="nonexistent",
                channels=["EEG"], pipelines="raw", sequence_length=1,
                cache_dir=os.path.join(tmp, "cache"),
            )
        except ValueError:
            ok = True
        except Exception as e:
            report("HPAP invalid subset -> ValueError", False, f"got {type(e).__name__}: {e}")
            return
    report("HPAP invalid subset -> ValueError", ok)


def test_hpap_mixed_subsets():
    """15. Files in both lab/full and home -> 'all' finds both, 'lab-full' finds 1."""
    stages = [0] * 5 + [2] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        # lab/full
        make_nsrr_subject(
            root / "polysomnography" / "edfs" / "lab" / "full",
            root / "polysomnography" / "annotations-events-nsrr" / "lab" / "full",
            "homepap-lab-full-1600001",
            stages=stages,
            channel_names=["EEG", "LOC", "EMG"],
        )
        # home
        make_nsrr_subject(
            root / "polysomnography" / "edfs" / "home",
            root / "polysomnography" / "annotations-events-nsrr" / "home",
            "homepap-home-1600005",
            stages=stages,
            channel_names=["EEG", "LOC", "EMG"],
        )
        ds_all = HPAPDataset(
            root=str(root), subset="all",
            channels=["EEG"], pipelines="raw", sequence_length=5,
            cache_dir=os.path.join(cache_dir, "all"),
        )
        ds_lab = HPAPDataset(
            root=str(root), subset="lab-full",
            channels=["EEG"], pipelines="raw", sequence_length=5,
            cache_dir=os.path.join(cache_dir, "lab"),
        )
        ok_all = len(ds_all._subjects) == 2
        ok_lab = len(ds_lab._subjects) == 1
        ok = ok_all and ok_lab
        report(
            "HPAP mixed subsets (all=2, lab-full=1)",
            ok,
            f"all={len(ds_all._subjects)}, lab-full={len(ds_lab._subjects)}",
        )


def test_hpap_subject_ids_no_collision():
    """16. Subject IDs from different subsets don't collide."""
    stages = [0] * 5 + [2] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs" / "lab" / "full",
            root / "polysomnography" / "annotations-events-nsrr" / "lab" / "full",
            "homepap-lab-full-1600001",
            stages=stages,
            channel_names=["EEG", "LOC", "EMG"],
        )
        make_nsrr_subject(
            root / "polysomnography" / "edfs" / "home",
            root / "polysomnography" / "annotations-events-nsrr" / "home",
            "homepap-home-1600005",
            stages=stages,
            channel_names=["EEG", "LOC", "EMG"],
        )
        ds = HPAPDataset(
            root=str(root), subset="all",
            channels=["EEG"], pipelines="raw", sequence_length=5,
            cache_dir=cache_dir,
        )
        ids = [s.subject_id for s in ds._subjects]
        ok = len(ids) == 2 and len(set(ids)) == 2
        report("HPAP subject IDs are distinct across subsets", ok, f"ids={ids}")


def test_hpap_end_to_end():
    """17. ds[0] returns dict."""
    stages = [0] * 5 + [4] * 5
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        root = Path(data_dir)
        make_nsrr_subject(
            root / "polysomnography" / "edfs" / "lab" / "full",
            root / "polysomnography" / "annotations-events-nsrr" / "lab" / "full",
            "homepap-lab-full-1600001",
            stages=stages,
            channel_names=["EEG", "LOC", "EMG"],
        )
        ds = HPAPDataset(
            root=str(root), subset="all",
            channels=["EEG"], pipelines="raw", sequence_length=5,
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
            len(item["channel_order"]) == 1,
            item["subject"]["dataset"] == "hpap",
        ]
        ok = all(checks)
        report("HPAP end-to-end ds[0] dict", ok, f"checks={checks}")


# ===================================================================
# Real-data smoke tests (guarded)
# ===================================================================

def run_real_data_tests():
    """Only runs if PHYSIOEX_TEST_REAL_DATA=1."""
    print("\n--- Real-data smoke tests ---")

    # MESA real data
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = MESADataset(
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n > 100
            report("MESA real data: >100 subjects", ok, f"got {n}")
            if n > 0:
                item = ds[0]
                ok = (
                    "signals" in item
                    and "labels" in item
                    and torch.is_tensor(item["labels"])
                )
                report("MESA real data: ds[0] returns valid dict", ok)
    except Exception as e:
        report("MESA real data", False, f"SKIP: {e}")

    # HPAP real data
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = HPAPDataset(
                subset="all",
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n > 100
            report("HPAP real data: >100 subjects", ok, f"got {n}")
            if n > 0:
                item = ds[0]
                ok = (
                    "signals" in item
                    and "labels" in item
                    and torch.is_tensor(item["labels"])
                )
                report("HPAP real data: ds[0] returns valid dict", ok)
    except Exception as e:
        report("HPAP real data", False, f"SKIP: {e}")

    # MrOS real data (expected empty -- directory exists but no EDFs)
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = MrOSDataset(
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = True  # 0 subjects expected; just verify no crash
            report("MrOS real data: no crash (empty dir)", ok, f"got {n} subjects")
    except Exception as e:
        report("MrOS real data", False, f"SKIP: {e}")


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase C dataset integration tests (NSRR family)")
    print("=" * 60)

    # MESA synthetic
    print("\n--- MESA (synthetic) ---")
    test_mesa_registry()
    test_mesa_in_available()
    test_mesa_empty_root()
    test_mesa_subject_discovery()
    test_mesa_subject_id()
    test_mesa_missing_xml_skipped()
    test_mesa_label_parsing()
    test_mesa_end_to_end()

    # MrOS synthetic
    print("\n--- MrOS (synthetic) ---")
    test_mros_registry()
    test_mros_empty_root()
    test_mros_subject_discovery()

    # HPAP synthetic
    print("\n--- HPAP (synthetic) ---")
    test_hpap_registry()
    test_hpap_empty_root()
    test_hpap_subset_enumeration()
    test_hpap_invalid_subset()
    test_hpap_mixed_subsets()
    test_hpap_subject_ids_no_collision()
    test_hpap_end_to_end()

    # Real-data tests (optional, guarded by env var)
    if os.environ.get("PHYSIOEX_TEST_REAL_DATA", "") == "1":
        run_real_data_tests()
    else:
        print(
            "\n--- Real-data smoke tests SKIPPED "
            "(set PHYSIOEX_TEST_REAL_DATA=1) ---"
        )

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
