"""MASS (Montreal Archive of Sleep Studies) dataset tests.

Synthetic tests create fake EDF + SAF/annotation files on-the-fly and verify
discovery, label parsing, cohort filtering, and end-to-end dict return.

Real-data smoke tests are guarded by PHYSIOEX_TEST_REAL_DATA=1.

Run standalone:
    python test/tests/test_mass_dataset.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from physioex.data.datasets import get_dataset, available_datasets
from physioex.data.datasets.mass import MASSDataset, MASS_STAGE_MAP
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

def make_mass_subject(
    root: Path,
    cohort: int,
    subject_num: str,
    annotation_type: str = "saf",
    stages: list = None,
    channel_names=None,
    duration_sec: float = 150.0,
    epoch_sec: float = 30.0,
):
    """Create a fake MASS subject: EDF + annotation file.

    Args:
        root: directory that will contain ``SS{cohort:02d}/``.
        cohort: cohort number (1-5).
        subject_num: e.g. "0001".
        annotation_type: "saf" for SAF text, "edf_annotations" for EDF+,
                         "edf_base" for _Base.edf.
        stages: list of stage strings (e.g. ["Sleep stage W", ...]).
        channel_names: EDF channel labels.
        duration_sec: EDF duration in seconds.
        epoch_sec: epoch length for annotation spacing.
    """
    if channel_names is None:
        channel_names = ["EEG C3-CLE", "EOG Left Horiz", "EMG Chin1", "ECG I"]
    if stages is None:
        stages = ["Sleep stage W", "Sleep stage 2", "Sleep stage 3",
                  "Sleep stage R", "Sleep stage W"]

    cohort_dir = root / f"SS{cohort:02d}"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = cohort_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    subject_id = f"01-{cohort:02d}-{subject_num}"
    edf_path = cohort_dir / f"{subject_id} PSG.edf"

    write_fake_edf(
        edf_path,
        n_channels=len(channel_names),
        duration_sec=duration_sec,
        channel_names=channel_names,
    )

    if annotation_type == "saf":
        saf_path = ann_dir / f"{subject_id}_saf.txt"
        _write_saf(saf_path, stages, epoch_sec=epoch_sec)
    elif annotation_type == "edf_annotations":
        ann_path = ann_dir / f"{subject_id}_Annotations.edf"
        write_fake_annotations_edf(ann_path, stages, epoch_sec=epoch_sec)
    elif annotation_type == "edf_base":
        ann_path = ann_dir / f"{subject_id}_Base.edf"
        write_fake_annotations_edf(ann_path, stages, epoch_sec=epoch_sec)
    else:
        raise ValueError(f"Unknown annotation_type: {annotation_type}")


def _write_saf(path: Path, stages: list, epoch_sec: float = 30.0):
    """Write a fake SAF text file with the given stage strings."""
    with open(path, "w", encoding="latin1") as f:
        for i, stage in enumerate(stages):
            onset = i * epoch_sec
            # Format: onset\x15duration\x14stage_name\x14\x00
            line = f"{onset}\x15{epoch_sec}\x14{stage}\x14\x00\n"
            f.write(line)


# ===================================================================
# 1. Registry
# ===================================================================

def test_registry():
    """get_dataset('mass') returns MASSDataset."""
    cls = get_dataset("mass")
    ok = cls is MASSDataset
    report("MASS registry registration", ok, f"got {cls}")


def test_in_available():
    """'mass' appears in available_datasets()."""
    ok = "mass" in available_datasets()
    report("MASS in available_datasets()", ok)


# ===================================================================
# 2. Valid cohorts (no error)
# ===================================================================

def test_valid_cohorts():
    """MASSDataset(cohort=1..5) does not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        for c in (1, 2, 3, 4, 5):
            try:
                ds = MASSDataset(
                    cohort=c,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, f"cache_c{c}"),
                )
                report(f"MASS valid cohort={c} no error", True)
            except Exception as e:
                report(f"MASS valid cohort={c} no error", False, str(e))


# ===================================================================
# 3. Invalid cohorts raise ValueError
# ===================================================================

def test_invalid_cohorts():
    """MASSDataset(cohort=0) and cohort=6 raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        for c in (0, 6):
            try:
                ds = MASSDataset(
                    cohort=c,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, f"cache_c{c}"),
                )
                report(f"MASS invalid cohort={c} raises ValueError", False,
                       "no error raised")
            except ValueError:
                report(f"MASS invalid cohort={c} raises ValueError", True)
            except Exception as e:
                report(f"MASS invalid cohort={c} raises ValueError", False,
                       f"wrong error: {e}")


# ===================================================================
# 4. Dataset name per cohort
# ===================================================================

def test_dataset_name_per_cohort():
    """MASSDataset(cohort=1).DATASET_NAME == 'mass_ss01'."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = MASSDataset(
            cohort=1,
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = ds.DATASET_NAME == "mass_ss01"
        report("MASS DATASET_NAME == 'mass_ss01'", ok, f"got {ds.DATASET_NAME!r}")


# ===================================================================
# 5. Cohort-specific epoch length
# ===================================================================

def test_epoch_length_per_cohort():
    """SS01 uses 30s epochs; SS02/SS04/SS05 use 20s epochs."""
    expected = {1: 30.0, 2: 20.0, 3: 30.0, 4: 20.0, 5: 20.0}
    with tempfile.TemporaryDirectory() as tmp:
        for c, exp in expected.items():
            ds = MASSDataset(
                cohort=c,
                root=tmp,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=1,
                cache_dir=os.path.join(tmp, f"cache_c{c}"),
            )
            ok = ds.epoch_length_sec == exp
            report(
                f"MASS cohort={c} epoch_length_sec == {exp}",
                ok,
                f"got {ds.epoch_length_sec}",
            )


# ===================================================================
# 6. Empty root -> 0 subjects
# ===================================================================

def test_empty_root():
    """Empty root returns 0 subjects (no crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = MASSDataset(
            cohort=1,
            root=tmp,
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(tmp, "cache"),
        )
        ok = len(ds._subjects) == 0
        report("MASS empty root -> 0 subjects", ok, f"got {len(ds._subjects)}")


# ===================================================================
# 7. Subject discovery (synthetic)
# ===================================================================

def test_subject_discovery():
    """Create fake MASS SS01 files; verify 1 subject discovered."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=1, subject_num="0001",
                          annotation_type="edf_annotations")
        ds = MASSDataset(
            cohort=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        n = len(ds._subjects)
        ok_count = n == 1
        ok_id = n > 0 and "01-01-0001" in ds._subjects[0].subject_id
        ok = ok_count and ok_id
        detail = f"n_subjects={n}"
        if n > 0:
            detail += f", subject_id={ds._subjects[0].subject_id!r}"
        report("MASS discovers 1 subject with '01-01-0001' in ID", ok, detail)


# ===================================================================
# 8. EDF annotation parsing
# ===================================================================

def test_annotation_edf_parsing():
    """Verify EDF+ annotation labels match expected AASM mapping."""
    stages = ["Sleep stage W", "Sleep stage 2", "Sleep stage 3",
              "Sleep stage R", "Sleep stage W"]
    expected = [0, 2, 3, 4, 0]
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=1, subject_num="0001",
                          annotation_type="edf_annotations",
                          stages=stages)
        ds = MASSDataset(
            cohort=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        ok_len = len(labels) == len(expected)
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report(
            "MASS EDF annotation parsing (length + values)",
            ok,
            f"len={len(labels)} expected={len(expected)}, "
            f"vals={list(labels)} expected={expected}",
        )


# ===================================================================
# 9. SAF text parsing
# ===================================================================

def test_saf_parsing():
    """Verify SAF text labels match expected AASM mapping.

    For 20s cohorts (SS04), first and last epochs are dropped (Phan ±5s
    padding convention), so [W,N1,N2,N4,R] → after drop → [N1,N2,N4].
    """
    stages = ["Sleep stage W", "Sleep stage 1", "Sleep stage 2",
              "Sleep stage 4", "Sleep stage R"]
    expected = [1, 2, 3]  # after drop first(W) and last(R), N4→3
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=4, subject_num="0001",
                          annotation_type="saf",
                          stages=stages,
                          epoch_sec=20.0)
        ds = MASSDataset(
            cohort=4,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        ok_len = len(labels) == len(expected)
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report(
            "MASS SAF parsing (length + values)",
            ok,
            f"len={len(labels)} expected={len(expected)}, "
            f"vals={list(labels)} expected={expected}",
        )


# ===================================================================
# 10. SS02 uses _Base.edf annotation files
# ===================================================================

def test_ss02_base_edf():
    """SS02 subjects use _Base.edf annotation files.

    SS02 is a 20s cohort, so first/last epochs are dropped (Phan convention).
    Use 5 stages so after drop we have 3 testable labels.
    """
    stages = ["Sleep stage W", "Sleep stage 1", "Sleep stage 2",
              "Sleep stage 3", "Sleep stage R"]
    expected = [1, 2, 3]  # after drop first(W) and last(R)
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=2, subject_num="0001",
                          annotation_type="edf_base",
                          stages=stages,
                          epoch_sec=20.0,
                          duration_sec=120.0)  # 120s = 6 x 20s (5 stages + margin)
        ds = MASSDataset(
            cohort=2,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        n = len(ds._subjects)
        ok_found = n == 1
        labels = ds._read_subject_labels(ds._subjects[0]) if n > 0 else []
        ok_vals = list(labels) == expected if n > 0 else False
        ok = ok_found and ok_vals
        report(
            "MASS SS02 _Base.edf annotation parsing",
            ok,
            f"n_subjects={n}, vals={list(labels)}, expected={expected}",
        )


# ===================================================================
# 10b. Expert micro-events: spindles / K-complexes
# ===================================================================

def test_event_categorization():
    """_categorize_mass_event maps spindle/K-complex + physiological events."""
    from physioex.data.datasets.mass import _categorize_mass_event

    cases = {
        "Sleep spindle": "spindle",
        "SpindleE1": "spindle",
        "K-complex": "k_complex",
        "KComplex E2": "k_complex",
        "Micro-arousal": "arousal",
        "Obstructive Apnea": "respiratory",
        "PLM": "limb_movement",
        "Desaturation": "desaturation",
        "Something else": "other",
    }
    bad = {k: _categorize_mass_event(k) for k, v in cases.items()
           if _categorize_mass_event(k) != v}
    report("MASS event categorization (spindle/k_complex/...)", not bad, str(bad))


def test_spindle_event_discovery():
    """A ``_Spindles.edf`` file is discovered and its events parsed as spindle/k_complex."""
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=2, subject_num="0001",
                          annotation_type="edf_base",
                          stages=["Sleep stage W", "Sleep stage 2",
                                  "Sleep stage 2", "Sleep stage 3",
                                  "Sleep stage R"],
                          epoch_sec=20.0, duration_sec=120.0)
        # Add an expert spindle/K-complex annotation file next to the staging file.
        subject_id = "01-02-0001"
        ann_dir = data_dir / "SS02" / "annotations"
        write_fake_annotations_edf(
            ann_dir / f"{subject_id}_Spindles.edf",
            ["Sleep spindle", "Sleep spindle", "K-complex"],
            epoch_sec=20.0,
        )
        ds = MASSDataset(
            cohort=2,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
            cache_enabled=False,
        )
        discovered = ds._event_paths.get(subject_id, [])
        events = ds.get_subject_events(subject_id)
        types = sorted({e.type for e in events})
        ok = (len(discovered) == 1
              and "spindle" in types
              and "k_complex" in types)
        report(
            "MASS spindle/K-complex event discovery",
            ok,
            f"discovered={len(discovered)} types={types}",
        )


# ===================================================================
# 11. Cohort isolation
# ===================================================================

def test_cohort_isolation():
    """Subjects for cohort 1 and cohort 3 in same root; cohort=1 only sees SS01."""
    stages = ["Sleep stage W", "Sleep stage 2", "Sleep stage R",
              "Sleep stage 3", "Sleep stage W"]
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=1, subject_num="0001",
                          annotation_type="edf_annotations", stages=stages)
        make_mass_subject(data_dir, cohort=3, subject_num="0001",
                          annotation_type="edf_annotations", stages=stages)

        ds1 = MASSDataset(
            cohort=1,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "c1"),
        )
        ds3 = MASSDataset(
            cohort=3,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "c3"),
        )
        ids1 = [s.subject_id for s in ds1._subjects]
        ids3 = [s.subject_id for s in ds3._subjects]

        ok_c1 = len(ids1) == 1 and "01-01-" in ids1[0]
        ok_c3 = len(ids3) == 1 and "01-03-" in ids3[0]
        ok = ok_c1 and ok_c3
        report(
            "MASS cohort isolation (c1 vs c3)",
            ok,
            f"cohort1_ids={ids1}, cohort3_ids={ids3}",
        )


# ===================================================================
# 12. End-to-end ds[0]
# ===================================================================

def test_end_to_end():
    """ds[0] returns a dict with expected keys."""
    stages = ["Sleep stage W", "Sleep stage 2", "Sleep stage 3",
              "Sleep stage R", "Sleep stage W"]
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_mass_subject(data_dir, cohort=1, subject_num="0001",
                          annotation_type="edf_annotations", stages=stages)
        ds = MASSDataset(
            cohort=1,
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
        report("MASS end-to-end ds[0] dict", ok, f"checks={checks}")


# ===================================================================
# Real-data smoke tests (guarded)
# ===================================================================

def run_real_data_tests():
    """Only runs if PHYSIOEX_TEST_REAL_DATA=1."""
    print("\n--- MASS real-data smoke tests ---")

    cohort_expected_min = {
        1: 40,   # SS01: 53 subjects
        2: 15,   # SS02: 19 subjects
        3: 50,   # SS03: 62 subjects
        4: 30,   # SS04: 40 subjects
        5: 20,   # SS05: 26 subjects
    }

    for cohort, min_subj in cohort_expected_min.items():
        try:
            with tempfile.TemporaryDirectory() as cache_dir:
                ds = MASSDataset(
                    cohort=cohort,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=21,
                    cache_dir=cache_dir,
                )
                n = len(ds._subjects)
                ok = n >= min_subj
                report(
                    f"MASS SS{cohort:02d} real data: >={min_subj} subjects",
                    ok,
                    f"got {n}",
                )
        except Exception as e:
            report(f"MASS SS{cohort:02d} real data", False, f"SKIP: {e}")

    # SS02: epoch_length_sec == 20.0
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = MASSDataset(
                cohort=2,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            ok = ds.epoch_length_sec == 20.0
            report("MASS SS02 epoch_length_sec == 20.0", ok,
                   f"got {ds.epoch_length_sec}")
    except Exception as e:
        report("MASS SS02 epoch_length_sec", False, f"SKIP: {e}")

    # SS02: verify _Base.edf annotations are found
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = MASSDataset(
                cohort=2,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            if len(ds._subjects) > 0:
                spec = ds._subjects[0]
                ok = str(spec.label_path).endswith("_Base.edf")
                report("MASS SS02 uses _Base.edf annotations", ok,
                       f"label_path={spec.label_path}")
            else:
                report("MASS SS02 uses _Base.edf annotations", False,
                       "no subjects found")
    except Exception as e:
        report("MASS SS02 _Base.edf", False, f"SKIP: {e}")

    # SS04: verify _saf.txt annotations are found
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = MASSDataset(
                cohort=4,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            if len(ds._subjects) > 0:
                spec = ds._subjects[0]
                ok = str(spec.label_path).endswith("_saf.txt")
                report("MASS SS04 uses _saf.txt annotations", ok,
                       f"label_path={spec.label_path}")
            else:
                report("MASS SS04 uses _saf.txt annotations", False,
                       "no subjects found")
    except Exception as e:
        report("MASS SS04 _saf.txt", False, f"SKIP: {e}")

    # All cohorts: probe available channels
    for cohort in (1, 2, 3, 4, 5):
        try:
            with tempfile.TemporaryDirectory() as cache_dir:
                ds = MASSDataset(
                    cohort=cohort,
                    channels=["EEG", "EOG", "EMG"],
                    pipelines="raw",
                    sequence_length=21,
                    cache_dir=cache_dir,
                )
                channels = ds.available_channels()
                eeg_chans = [c for c in channels if c.startswith("EEG")]
                eog_chans = [c for c in channels if c.startswith("EOG")]
                emg_chans = [c for c in channels if c.startswith("EMG")]
                ok = len(eeg_chans) > 0 and len(eog_chans) > 0
                report(
                    f"MASS SS{cohort:02d} channels probe",
                    ok,
                    f"EEG={eeg_chans[:3]}, EOG={eog_chans[:2]}, "
                    f"EMG={emg_chans[:2]}",
                )
        except Exception as e:
            report(f"MASS SS{cohort:02d} channels", False, f"SKIP: {e}")


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MASS dataset tests")
    print("=" * 60)

    print("\n--- MASS (synthetic) ---")
    test_registry()
    test_in_available()
    test_valid_cohorts()
    test_invalid_cohorts()
    test_dataset_name_per_cohort()
    test_epoch_length_per_cohort()
    test_empty_root()
    test_subject_discovery()
    test_annotation_edf_parsing()
    test_saf_parsing()
    test_ss02_base_edf()
    test_event_categorization()
    test_spindle_event_discovery()
    test_cohort_isolation()
    test_end_to_end()

    # Real-data tests (optional)
    if os.environ.get("PHYSIOEX_TEST_REAL_DATA", "") == "1":
        run_real_data_tests()
    else:
        print("\n--- MASS real-data smoke tests SKIPPED "
              "(set PHYSIOEX_TEST_REAL_DATA=1) ---")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
