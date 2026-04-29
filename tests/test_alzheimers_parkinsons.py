"""Alzheimer's and Parkinson's dataset tests.

Synthetic tests create fake EDF + TSV files on-the-fly and verify discovery,
label parsing, subset/group filtering, and end-to-end dict return.

Real-data smoke tests are guarded by PHYSIOEX_TEST_REAL_DATA=1.

Run standalone:
    python test/tests/test_alzheimers_parkinsons.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from physioex.data.datasets import get_dataset, available_datasets
from physioex.data.datasets.alzheimers import AlzheimersDataset, _STAGE_MAP as ALZ_STAGE_MAP
from physioex.data.datasets.parkinsons import ParkinsonsDataset, _STAGE_MAP as PD_STAGE_MAP
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

def make_alzheimers_subject(
    root: Path,
    subj_id: str,
    stages: list = None,
    channel_names=None,
    duration_sec: float = 150.0,
):
    """Create a fake Alzheimer's subject: EDF + TSV with comment headers."""
    if channel_names is None:
        channel_names = ["EEG C4-REF", "EEG C3-REF", "EEG EOG1-REF",
                         "EEG EOG2-REF", "EMG1", "ECG"]
    if stages is None:
        stages = ["Wake", "S1", "S2", "S3", "REM"]

    data_dir = root / "Data" / subj_id
    data_dir.mkdir(parents=True, exist_ok=True)

    edf_path = data_dir / f"{subj_id}_r1.edf"
    write_fake_edf(
        edf_path,
        n_channels=len(channel_names),
        duration_sec=duration_sec,
        channel_names=channel_names,
        fs_per_channel=[200] * len(channel_names),
    )

    tsv_path = data_dir / f"{subj_id}_r1_a1.tsv"
    _write_alzheimers_tsv(tsv_path, stages, subj_id=subj_id)


def _write_alzheimers_tsv(path: Path, stages: list, subj_id: str = "AD008"):
    """Write a fake Alzheimer's TSV with comment header lines."""
    with open(path, "w") as f:
        f.write(f"# Source: Alzheimer disease Database UZ Leuven\n")
        f.write(f"# Subject id: {subj_id}\n")
        f.write(f"# Annotation_file: {subj_id}_ann.EDF\n")
        f.write(f"# PSG_file: {subj_id}_combined.edf\n")
        f.write(f"# Offset wrt orignal PSG: 900\n")
        f.write(f"#Start & end of sleep annotations: 40830.02- 78564.14 \n")
        f.write(f"\n")  # blank line after comments
        onset = 40860
        for stage in stages:
            end = onset + 30
            f.write(f"{onset}\t{end}\t{stage}\t\n")
            onset = end


def make_parkinsons_subject(
    root: Path,
    subj_id: str,
    is_nap: bool = False,
    stages: list = None,
    channel_names=None,
    duration_sec: float = 150.0,
):
    """Create a fake Parkinson's subject: EDF + TSV (no header)."""
    if channel_names is None:
        channel_names = ["EEG C3-A2", "EEG C4-A1", "EOG Left",
                         "EOG right", "EMG Chin", "ECG V1"]
    if stages is None:
        stages = ["Wake", "S1", "S2", "S3", "REM"]

    dir_name = f"{subj_id}_nap" if is_nap else subj_id
    data_dir = root / "Data" / dir_name
    data_dir.mkdir(parents=True, exist_ok=True)

    edf_path = data_dir / f"{dir_name}_r1.edf"
    write_fake_edf(
        edf_path,
        n_channels=len(channel_names),
        duration_sec=duration_sec,
        channel_names=channel_names,
        fs_per_channel=[500] * len(channel_names),
    )

    tsv_path = data_dir / f"{dir_name}_r1_a1.tsv"
    _write_parkinsons_tsv(tsv_path, stages)


def _write_parkinsons_tsv(path: Path, stages: list):
    """Write a fake Parkinson's TSV (no header, just onset/end/stage)."""
    with open(path, "w") as f:
        onset = 0
        for stage in stages:
            end = onset + 30
            f.write(f"{onset}\t{end}\t{stage}\n")
            onset = end


def write_demographics_csv(root: Path, subjects: dict):
    """Write a fake Target_sleep_demographic.csv.

    Args:
        subjects: dict mapping subject_id -> group (e.g. {"Sub_0001": "HOA"}).
    """
    csv_path = root / "Target_sleep_demographic.csv"
    with open(csv_path, "w") as f:
        f.write("record_id,group\n")
        for sid, group in subjects.items():
            f.write(f"{sid},{group}\n")


# ===================================================================
# ALZHEIMER'S TESTS
# ===================================================================

# 1. Registry
def test_alz_registry():
    cls = get_dataset("alzheimers")
    ok = cls is AlzheimersDataset
    report("ALZ registry: get_dataset('alzheimers')", ok, f"got {cls}")


def test_alz_in_available():
    ok = "alzheimers" in available_datasets()
    report("ALZ in available_datasets()", ok)


# 2. Valid subsets
def test_alz_valid_subsets():
    with tempfile.TemporaryDirectory() as tmp:
        for subset in (None, "AD", "HC"):
            try:
                ds = AlzheimersDataset(
                    subset=subset,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, f"cache_{subset}"),
                )
                report(f"ALZ valid subset={subset!r} no error", True)
            except Exception as e:
                report(f"ALZ valid subset={subset!r} no error", False, str(e))


# 3. Invalid subset
def test_alz_invalid_subset():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ds = AlzheimersDataset(
                subset="INVALID",
                root=tmp,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=1,
                cache_dir=os.path.join(tmp, "cache"),
            )
            report("ALZ invalid subset raises ValueError", False, "no error raised")
        except ValueError:
            report("ALZ invalid subset raises ValueError", True)
        except Exception as e:
            report("ALZ invalid subset raises ValueError", False, f"wrong error: {e}")


# 4. Subject discovery with fake files
def test_alz_subject_discovery():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_alzheimers_subject(data_dir, "AD001")
        make_alzheimers_subject(data_dir, "AD002")
        make_alzheimers_subject(data_dir, "HC001")

        ds = AlzheimersDataset(
            subset=None,
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        n = len(ds._subjects)
        ok = n == 3
        ids = [s.subject_id for s in ds._subjects]
        report("ALZ discovers 3 subjects (2 AD + 1 HC)", ok,
               f"n={n}, ids={ids}")


# 5. Subset filtering
def test_alz_subset_filtering():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_alzheimers_subject(data_dir, "AD001")
        make_alzheimers_subject(data_dir, "AD002")
        make_alzheimers_subject(data_dir, "HC001")
        make_alzheimers_subject(data_dir, "HC002")

        ds_ad = AlzheimersDataset(
            subset="AD",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "ad"),
        )
        ds_hc = AlzheimersDataset(
            subset="HC",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "hc"),
        )
        ad_ids = [s.subject_id for s in ds_ad._subjects]
        hc_ids = [s.subject_id for s in ds_hc._subjects]
        ok_ad = len(ad_ids) == 2 and all(i.startswith("AD") for i in ad_ids)
        ok_hc = len(hc_ids) == 2 and all(i.startswith("HC") for i in hc_ids)
        ok = ok_ad and ok_hc
        report("ALZ subset filtering (AD=2, HC=2)", ok,
               f"AD={ad_ids}, HC={hc_ids}")


# 6. TSV parsing with comment headers
def test_alz_tsv_parsing():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        stages = ["Wake", "S1", "S2", "S3", "REM"]
        make_alzheimers_subject(data_dir, "AD001", stages=stages)

        ds = AlzheimersDataset(
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        expected = [0, 1, 2, 3, 4]
        ok_len = len(labels) == len(expected)
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report("ALZ TSV parsing (comment headers + stages)", ok,
               f"len={len(labels)}, vals={list(labels)}, expected={expected}")


# 7. Stage mapping
def test_alz_stage_map():
    expected = {"Wake": 0, "S1": 1, "S2": 2, "S3": 3, "REM": 4}
    ok = ALZ_STAGE_MAP == expected
    report("ALZ stage map correct", ok, f"got {ALZ_STAGE_MAP}")


# 8. End-to-end ds[0]
def test_alz_end_to_end():
    stages = ["Wake", "S1", "S2", "S3", "REM"]
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        make_alzheimers_subject(data_dir, "AD001", stages=stages)

        ds = AlzheimersDataset(
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
        report("ALZ end-to-end ds[0] dict", ok, f"checks={checks}")


# 8b. Dataset name dynamic
def test_alz_dataset_name():
    with tempfile.TemporaryDirectory() as tmp:
        ds_none = AlzheimersDataset(
            subset=None, root=tmp, channels=["EEG"],
            pipelines="raw", sequence_length=1,
            cache_dir=os.path.join(tmp, "c0"),
        )
        ds_ad = AlzheimersDataset(
            subset="AD", root=tmp, channels=["EEG"],
            pipelines="raw", sequence_length=1,
            cache_dir=os.path.join(tmp, "c1"),
        )
        ds_hc = AlzheimersDataset(
            subset="HC", root=tmp, channels=["EEG"],
            pipelines="raw", sequence_length=1,
            cache_dir=os.path.join(tmp, "c2"),
        )
        ok = (ds_none.DATASET_NAME == "alzheimers"
              and ds_ad.DATASET_NAME == "alzheimers_ad"
              and ds_hc.DATASET_NAME == "alzheimers_hc")
        report("ALZ dynamic DATASET_NAME", ok,
               f"none={ds_none.DATASET_NAME!r}, AD={ds_ad.DATASET_NAME!r}, "
               f"HC={ds_hc.DATASET_NAME!r}")


# ===================================================================
# PARKINSON'S TESTS
# ===================================================================

# 9. Registry
def test_pd_registry():
    cls = get_dataset("parkinsons")
    ok = cls is ParkinsonsDataset
    report("PD registry: get_dataset('parkinsons')", ok, f"got {cls}")


def test_pd_in_available():
    ok = "parkinsons" in available_datasets()
    report("PD in available_datasets()", ok)


# 10. Valid recording types
def test_pd_valid_recording():
    with tempfile.TemporaryDirectory() as tmp:
        for rec in ("night", "nap", "all"):
            try:
                ds = ParkinsonsDataset(
                    recording=rec,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, f"cache_{rec}"),
                )
                report(f"PD valid recording={rec!r} no error", True)
            except Exception as e:
                report(f"PD valid recording={rec!r} no error", False, str(e))


# 11. Invalid recording
def test_pd_invalid_recording():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ds = ParkinsonsDataset(
                recording="INVALID",
                root=tmp,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=1,
                cache_dir=os.path.join(tmp, "cache"),
            )
            report("PD invalid recording raises ValueError", False,
                   "no error raised")
        except ValueError:
            report("PD invalid recording raises ValueError", True)
        except Exception as e:
            report("PD invalid recording raises ValueError", False,
                   f"wrong error: {e}")


# 12. Valid groups
def test_pd_valid_groups():
    with tempfile.TemporaryDirectory() as tmp:
        write_demographics_csv(Path(tmp), {})
        for group in (None, "HOA", "PD"):
            try:
                ds = ParkinsonsDataset(
                    recording="night",
                    group=group,
                    root=tmp,
                    channels=["EEG"],
                    pipelines="raw",
                    sequence_length=1,
                    cache_dir=os.path.join(tmp, f"cache_{group}"),
                )
                report(f"PD valid group={group!r} no error", True)
            except Exception as e:
                report(f"PD valid group={group!r} no error", False, str(e))


# 12b. Invalid group
def test_pd_invalid_group():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ds = ParkinsonsDataset(
                recording="night",
                group="INVALID",
                root=tmp,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=1,
                cache_dir=os.path.join(tmp, "cache"),
            )
            report("PD invalid group raises ValueError", False,
                   "no error raised")
        except ValueError:
            report("PD invalid group raises ValueError", True)
        except Exception as e:
            report("PD invalid group raises ValueError", False,
                   f"wrong error: {e}")


# 13. Subject discovery: night vs nap
def test_pd_subject_discovery():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        demos = {"Sub_0001": "HOA", "Sub_0002": "PD", "Sub_0003": "HOA"}
        write_demographics_csv(data_dir, demos)

        # Create night + nap for Sub_0001, only night for Sub_0002,
        # only nap for Sub_0003
        make_parkinsons_subject(data_dir, "Sub_0001", is_nap=False)
        make_parkinsons_subject(data_dir, "Sub_0001", is_nap=True)
        make_parkinsons_subject(data_dir, "Sub_0002", is_nap=False)
        make_parkinsons_subject(data_dir, "Sub_0003", is_nap=True)

        # Night only
        ds_night = ParkinsonsDataset(
            recording="night",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "night"),
        )
        night_ids = [s.subject_id for s in ds_night._subjects]

        # Nap only
        ds_nap = ParkinsonsDataset(
            recording="nap",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "nap"),
        )
        nap_ids = [s.subject_id for s in ds_nap._subjects]

        # All
        ds_all = ParkinsonsDataset(
            recording="all",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "all"),
        )
        all_ids = [s.subject_id for s in ds_all._subjects]

        ok_night = sorted(night_ids) == ["Sub_0001", "Sub_0002"]
        ok_nap = sorted(nap_ids) == ["Sub_0001_nap", "Sub_0003_nap"]
        ok_all = len(all_ids) == 4
        ok = ok_night and ok_nap and ok_all
        report("PD subject discovery (night/nap/all)", ok,
               f"night={night_ids}, nap={nap_ids}, all={all_ids}")


# 14. Group filtering
def test_pd_group_filtering():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        demos = {"Sub_0001": "HOA", "Sub_0002": "PD", "Sub_0003": "HOA"}
        write_demographics_csv(data_dir, demos)

        make_parkinsons_subject(data_dir, "Sub_0001", is_nap=False)
        make_parkinsons_subject(data_dir, "Sub_0002", is_nap=False)
        make_parkinsons_subject(data_dir, "Sub_0003", is_nap=False)

        ds_hoa = ParkinsonsDataset(
            recording="night",
            group="HOA",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "hoa"),
        )
        ds_pd = ParkinsonsDataset(
            recording="night",
            group="PD",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=os.path.join(cache_dir, "pd"),
        )
        hoa_ids = [s.subject_id for s in ds_hoa._subjects]
        pd_ids = [s.subject_id for s in ds_pd._subjects]
        ok_hoa = sorted(hoa_ids) == ["Sub_0001", "Sub_0003"]
        ok_pd = pd_ids == ["Sub_0002"]
        ok = ok_hoa and ok_pd
        report("PD group filtering (HOA=2, PD=1)", ok,
               f"HOA={hoa_ids}, PD={pd_ids}")


# 15. TSV parsing with Unscorable and LIGHTS_OFF
def test_pd_tsv_parsing():
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        demos = {"Sub_0001": "HOA"}
        write_demographics_csv(data_dir, demos)

        stages = ["LIGHTS_OFF", "Wake", "S1", "S2", "S3", "REM", "Unscorable"]
        make_parkinsons_subject(data_dir, "Sub_0001", stages=stages,
                                duration_sec=210.0)

        ds = ParkinsonsDataset(
            recording="night",
            root=str(data_dir),
            channels=["EEG"],
            pipelines="raw",
            sequence_length=1,
            cache_dir=cache_dir,
        )
        spec = ds._subjects[0]
        labels = ds._read_subject_labels(spec)
        expected = [-1, 0, 1, 2, 3, 4, -1]
        ok_len = len(labels) == len(expected)
        ok_vals = list(labels) == expected
        ok = ok_len and ok_vals
        report("PD TSV parsing (LIGHTS_OFF=-1, Unscorable=-1)", ok,
               f"len={len(labels)}, vals={list(labels)}, expected={expected}")


# 16. End-to-end ds[0]
def test_pd_end_to_end():
    stages = ["Wake", "S1", "S2", "S3", "REM"]
    with tempfile.TemporaryDirectory() as data_dir, \
         tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        demos = {"Sub_0001": "HOA"}
        write_demographics_csv(data_dir, demos)
        make_parkinsons_subject(data_dir, "Sub_0001", stages=stages)

        ds = ParkinsonsDataset(
            recording="night",
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
        report("PD end-to-end ds[0] dict", ok, f"checks={checks}")


# 16b. Stage map
def test_pd_stage_map():
    expected = {"Wake": 0, "S1": 1, "S2": 2, "S3": 3, "REM": 4,
                "Unscorable": -1, "LIGHTS_OFF": -1}
    ok = PD_STAGE_MAP == expected
    report("PD stage map correct", ok, f"got {PD_STAGE_MAP}")


# 16c. Dataset name dynamic
def test_pd_dataset_name():
    with tempfile.TemporaryDirectory() as tmp:
        write_demographics_csv(Path(tmp), {})
        ds1 = ParkinsonsDataset(
            recording="night", group=None, root=tmp, channels=["EEG"],
            pipelines="raw", sequence_length=1,
            cache_dir=os.path.join(tmp, "c1"),
        )
        ds2 = ParkinsonsDataset(
            recording="nap", group="PD", root=tmp, channels=["EEG"],
            pipelines="raw", sequence_length=1,
            cache_dir=os.path.join(tmp, "c2"),
        )
        ds3 = ParkinsonsDataset(
            recording="all", group="HOA", root=tmp, channels=["EEG"],
            pipelines="raw", sequence_length=1,
            cache_dir=os.path.join(tmp, "c3"),
        )
        ok = (ds1.DATASET_NAME == "parkinsons_night"
              and ds2.DATASET_NAME == "parkinsons_nap_pd"
              and ds3.DATASET_NAME == "parkinsons_all_hoa")
        report("PD dynamic DATASET_NAME", ok,
               f"night={ds1.DATASET_NAME!r}, nap_pd={ds2.DATASET_NAME!r}, "
               f"all_hoa={ds3.DATASET_NAME!r}")


# ===================================================================
# Real-data smoke tests (guarded)
# ===================================================================

def run_real_data_tests():
    print("\n--- Alzheimer's real-data smoke tests ---")

    # 17. Alzheimer's: ~69 subjects total
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = AlzheimersDataset(
                subset=None,
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            # Allow for some corrupt subjects (original code removes AD014)
            ok = n >= 60
            report(f"ALZ real data: >=60 subjects", ok, f"got {n}")
    except Exception as e:
        report("ALZ real data: >=60 subjects", False, f"SKIP: {e}")

    # 17b. Alzheimer's subset=AD gives ~37
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = AlzheimersDataset(
                subset="AD",
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n >= 30
            report(f"ALZ real data subset=AD: >=30 subjects", ok, f"got {n}")
    except Exception as e:
        report("ALZ real data subset=AD", False, f"SKIP: {e}")

    print("\n--- Parkinson's real-data smoke tests ---")

    # 18. Parkinson's night: >80 subjects
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = ParkinsonsDataset(
                recording="night",
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n >= 80
            report(f"PD real data night: >=80 subjects", ok, f"got {n}")
    except Exception as e:
        report("PD real data night", False, f"SKIP: {e}")

    # 19. Parkinson's nap: >60 subjects
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = ParkinsonsDataset(
                recording="nap",
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n >= 60
            report(f"PD real data nap: >=60 subjects", ok, f"got {n}")
    except Exception as e:
        report("PD real data nap", False, f"SKIP: {e}")

    # 20. Parkinson's group=PD: >40 subjects
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            ds = ParkinsonsDataset(
                recording="night",
                group="PD",
                channels=["EEG"],
                pipelines="raw",
                sequence_length=21,
                cache_dir=cache_dir,
            )
            n = len(ds._subjects)
            ok = n >= 40
            report(f"PD real data group=PD: >=40 subjects", ok, f"got {n}")
    except Exception as e:
        report("PD real data group=PD", False, f"SKIP: {e}")

    # Probe available channels
    for name, kwargs in [
        ("ALZ", {"subset": None}),
        ("PD night", {"recording": "night"}),
        ("PD nap", {"recording": "nap"}),
    ]:
        try:
            with tempfile.TemporaryDirectory() as cache_dir:
                if "subset" in kwargs:
                    ds = AlzheimersDataset(
                        channels=["EEG", "EOG", "EMG"],
                        pipelines="raw",
                        sequence_length=21,
                        cache_dir=cache_dir,
                        **kwargs,
                    )
                else:
                    ds = ParkinsonsDataset(
                        channels=["EEG", "EOG", "EMG"],
                        pipelines="raw",
                        sequence_length=21,
                        cache_dir=cache_dir,
                        **kwargs,
                    )
                channels = ds.available_channels()
                eeg = [c for c in channels if "EEG" in c or "C3" in c or "C4" in c]
                eog = [c for c in channels if "EOG" in c or "eog" in c.lower()]
                emg = [c for c in channels if "EMG" in c or "emg" in c.lower()]
                ok = len(eeg) > 0
                report(f"{name} channels probe", ok,
                       f"EEG={eeg[:3]}, EOG={eog[:2]}, EMG={emg[:2]}")
        except Exception as e:
            report(f"{name} channels probe", False, f"SKIP: {e}")


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alzheimer's and Parkinson's dataset tests")
    print("=" * 60)

    print("\n--- Alzheimer's (synthetic) ---")
    test_alz_registry()
    test_alz_in_available()
    test_alz_valid_subsets()
    test_alz_invalid_subset()
    test_alz_subject_discovery()
    test_alz_subset_filtering()
    test_alz_tsv_parsing()
    test_alz_stage_map()
    test_alz_end_to_end()
    test_alz_dataset_name()

    print("\n--- Parkinson's (synthetic) ---")
    test_pd_registry()
    test_pd_in_available()
    test_pd_valid_recording()
    test_pd_invalid_recording()
    test_pd_valid_groups()
    test_pd_invalid_group()
    test_pd_subject_discovery()
    test_pd_group_filtering()
    test_pd_tsv_parsing()
    test_pd_stage_map()
    test_pd_end_to_end()
    test_pd_dataset_name()

    # Real-data tests (optional)
    if os.environ.get("PHYSIOEX_TEST_REAL_DATA", "") == "1":
        run_real_data_tests()
    else:
        print("\n--- Real-data smoke tests SKIPPED "
              "(set PHYSIOEX_TEST_REAL_DATA=1) ---")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
