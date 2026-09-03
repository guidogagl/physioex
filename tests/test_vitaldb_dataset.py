"""VitalDB dataset tests: ``.vital`` reading and case-level target broadcast.

Synthetic ``.vital`` files are built with the real VitalRecorder writer, so the
container, the header probe and the waveform read are all exercised end to end.
Skipped when the optional ``vitaldb`` dependency is absent.

Real-data smoke test is guarded by ``PHYSIOEX_TEST_REAL_DATA=1`` and
``PHYSIOEX_DATA`` pointing at a tree with ``vitaldb/raw``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("vitaldb", reason="VitalDB reading requires the 'datasets' extra")

from physioex.data.datasets import REGISTRY, get_dataset  # noqa: E402
from physioex.data.datasets.vitaldb import EEG1, EEG2, VitalDBDataset  # noqa: E402
from physioex.data.readers.vital import (  # noqa: E402
    list_vital_tracks,
    probe_vital_header,
)
from tests.factories.vital import (  # noqa: E402
    build_fake_vitaldb,
    write_fake_meta,
    write_fake_vital,
)

EPOCH = 120.0
DURATION = 600.0  # 5 epochs of 120 s


@pytest.fixture
def vitaldb_root(tmp_path) -> Path:
    return build_fake_vitaldb(tmp_path / "vitaldb", n_cases=3, duration_sec=DURATION)


def make_dataset(root, cache_dir, **kwargs) -> VitalDBDataset:
    kwargs.setdefault("sequence_length", 1)
    kwargs.setdefault("epoch_length_sec", EPOCH)
    return VitalDBDataset(root=str(root), cache_dir=str(cache_dir), **kwargs)


# ===================================================================
# Reader
# ===================================================================


def test_header_lists_only_waveform_tracks(vitaldb_root):
    header = probe_vital_header(vitaldb_root / "raw" / "1.vital")
    assert header.available_channels == sorted([EEG1, EEG2])
    assert header.channel_fs[EEG1] == 128.0
    assert header.channel_units[EEG1] == "uV"
    assert header.duration_sec == pytest.approx(DURATION)
    # numeric tracks exist in the file but are not epochable channels
    assert "BIS/BIS" not in header.available_channels
    assert "BIS/BIS" in list_vital_tracks(vitaldb_root / "raw" / "1.vital")


def test_header_carries_external_patient_meta(vitaldb_root):
    header = probe_vital_header(
        vitaldb_root / "raw" / "1.vital", patient_meta={"age": "71", "sex": "F"}
    )
    assert header.patient_meta["age"] == "71"


def test_read_channel_returns_native_rate_signal(vitaldb_root, tmp_path):
    ds = make_dataset(vitaldb_root, tmp_path / "cache", cache_enabled=False)
    spec = ds._subjects[0]
    resolved = ds._resolved[spec.subject_id][0]
    sig, fs = ds._read_subject_channel(spec, resolved)
    assert fs == 128.0
    assert sig.dtype == np.float32
    assert np.isfinite(sig).all()  # gaps are zero-filled, never NaN
    assert sig.shape[0] == pytest.approx(DURATION * fs, rel=0.01)


def test_millivolt_track_is_scaled_to_microvolts(tmp_path):
    root = tmp_path / "vitaldb"
    (root / "raw").mkdir(parents=True)
    write_fake_vital(root / "raw" / "1.vital", duration_sec=DURATION, unit="mV", seed=1)
    write_fake_meta(root, [{"caseid": "1"}])
    ds = make_dataset(root, tmp_path / "cache", cache_enabled=False)
    spec = ds._subjects[0]
    sig, _ = ds._read_subject_channel(spec, ds._resolved[spec.subject_id][0])
    # the factory writes ~20 uV noise; read as mV it must come back ~1000x larger
    assert np.abs(sig).mean() > 1_000


# ===================================================================
# Discovery
# ===================================================================


def test_registry_exposes_vitaldb():
    assert REGISTRY["vitaldb"] is VitalDBDataset
    assert get_dataset("vitaldb") is VitalDBDataset


def test_lists_every_general_anaesthesia_case(vitaldb_root, tmp_path):
    ds = make_dataset(vitaldb_root, tmp_path / "cache")
    assert ds.get_subjects() == ["1", "2", "3"]


def test_filters_on_anaesthesia_type(tmp_path):
    root = tmp_path / "vitaldb"
    (root / "raw").mkdir(parents=True)
    for cid in ("1", "2"):
        write_fake_vital(root / "raw" / f"{cid}.vital", duration_sec=DURATION)
    write_fake_meta(
        root,
        [{"caseid": "1", "ane_type": "General"}, {"caseid": "2", "ane_type": "Spinal"}],
    )
    assert make_dataset(root, tmp_path / "c1").get_subjects() == ["1"]
    assert make_dataset(root, tmp_path / "c2", ane_type=None).get_subjects() == ["1", "2"]


def test_case_without_the_required_track_is_dropped(tmp_path):
    root = tmp_path / "vitaldb"
    (root / "raw").mkdir(parents=True)
    for cid in ("1", "2"):
        write_fake_vital(root / "raw" / f"{cid}.vital", duration_sec=DURATION)
    write_fake_meta(
        root,
        [{"caseid": "1"}, {"caseid": "2"}],
        tracks_per_case={"1": [EEG1, EEG2], "2": ["Solar8000/HR"]},
    )
    assert make_dataset(root, tmp_path / "cache").get_subjects() == ["1"]


def test_missing_clinical_table_is_a_clear_error(tmp_path):
    root = tmp_path / "vitaldb"
    (root / "raw").mkdir(parents=True)
    write_fake_vital(root / "raw" / "1.vital", duration_sec=DURATION)
    with pytest.raises(FileNotFoundError, match="cases.csv"):
        make_dataset(root, tmp_path / "cache")


def test_external_meta_carries_subjectid_for_grouped_splits(vitaldb_root, tmp_path):
    ds = make_dataset(vitaldb_root, tmp_path / "cache")
    # the factory makes cases 1 and 2 the same patient: splitting on caseid leaks
    patients = {s: ds.get_subject_metadata(s)["subjectid"] for s in ds.get_subjects()}
    assert patients["1"] == patients["2"] != patients["3"]


# ===================================================================
# Case-level target
# ===================================================================


def test_target_none_labels_everything_unscored(vitaldb_root, tmp_path):
    ds = make_dataset(vitaldb_root, tmp_path / "cache")
    labels = ds._get_labels(ds._subjects[0])
    assert labels.shape[0] == int(DURATION // EPOCH)
    assert (labels == -1).all()


def test_target_callable_broadcasts_over_epochs(vitaldb_root, tmp_path):
    icu = lambda row: int(float(row["icu_days"] or 0) > 0)  # noqa: E731
    ds = make_dataset(vitaldb_root, tmp_path / "cache", target=icu)
    by_case = {s: np.unique(ds._get_labels(spec)) for s, spec in
               zip(ds.get_subjects(), ds._subjects)}
    assert by_case["1"].tolist() == [1]  # icu_days = 3
    assert by_case["2"].tolist() == [0]  # icu_days = 0
    assert ds._get_labels(ds._subjects[0]).shape[0] == int(DURATION // EPOCH)


def test_target_column_name_is_read_from_the_clinical_table(vitaldb_root, tmp_path):
    ds = make_dataset(vitaldb_root, tmp_path / "cache", target="death_inhosp")
    assert (ds._get_labels(ds._subjects[0]) == 0).all()


def test_blank_target_cell_becomes_unscored(tmp_path):
    root = tmp_path / "vitaldb"
    (root / "raw").mkdir(parents=True)
    write_fake_vital(root / "raw" / "1.vital", duration_sec=DURATION)
    write_fake_meta(root, [{"caseid": "1", "icu_days": ""}])
    ds = make_dataset(root, tmp_path / "cache", target="icu_days")
    assert (ds._get_labels(ds._subjects[0]) == -1).all()


def test_non_binary_target_is_rejected_not_silently_coerced(vitaldb_root, tmp_path):
    # _sanitize_labels would map 7 to -1 without a word; the dataset must refuse.
    with pytest.raises(ValueError, match="outside"):
        make_dataset(vitaldb_root, tmp_path / "cache", target=lambda row: 7)


def test_labels_are_not_cached_across_targets(vitaldb_root, tmp_path):
    """The on-disk label cache is keyed on (dataset, subject) only.

    Reusing it would return the first target's labels for the second endpoint.
    """
    cache = tmp_path / "cache"
    ds_a = make_dataset(vitaldb_root, cache, target=lambda row: 1, cache_enabled=True)
    assert (ds_a._get_labels(ds_a._subjects[0]) == 1).all()
    ds_b = make_dataset(vitaldb_root, cache, target=lambda row: 0, cache_enabled=True)
    assert (ds_b._get_labels(ds_b._subjects[0]) == 0).all()


# ===================================================================
# End to end
# ===================================================================


def test_item_shape_and_contents(vitaldb_root, tmp_path):
    ds = make_dataset(
        vitaldb_root,
        tmp_path / "cache",
        sequence_length=2,
        target=lambda row: int(float(row["icu_days"] or 0) > 0),
        channels=["EEG"],
        pipelines="raw",
    )
    item = ds[0]
    assert torch.is_tensor(item["labels"])
    assert item["labels"].shape[0] == 2
    assert item["subject"]["dataset"] == "vitaldb"
    assert len(item["channel_order"]) == 1
    signal = item[item["channel_order"][0]] if item["channel_order"][0] in item else None
    if signal is not None:
        assert signal.shape[0] == 2  # one row per epoch in the sequence


def test_two_channels_resolve_to_both_eeg_leads(vitaldb_root, tmp_path):
    ds = make_dataset(
        vitaldb_root, tmp_path / "cache", channels=[EEG1, EEG2], cache_enabled=False
    )
    resolved = ds._resolved[ds._subjects[0].subject_id]
    assert [r.physical for r in resolved] == [EEG1, EEG2]


@pytest.mark.skipif(
    os.environ.get("PHYSIOEX_TEST_REAL_DATA") != "1",
    reason="set PHYSIOEX_TEST_REAL_DATA=1 to run against the downloaded VitalDB",
)
def test_real_vitaldb_smoke():
    ds = VitalDBDataset(
        target=lambda row: int(float(row["icu_days"] or 0) > 0),
        sequence_length=1,
        epoch_length_sec=EPOCH,
    )
    assert ds.get_n_subjects() > 5_000
    item = ds[0]
    assert torch.is_tensor(item["labels"])
