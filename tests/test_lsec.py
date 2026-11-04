"""Unit tests for the LSEC reader (synthetic EDF and UTF-16 annotation files, no real data)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from physioex.data.datasets import REGISTRY
from physioex.data.datasets.lsec import (
    LeuvenSleepEpilepsyChildrenDataset as LSEC,
    _edf_header_fields,
    parse_lsec_labels,
    shift_and_cut,
)
from physioex.data.readers.edf import resolve_channels
from physioex.models.foundation_datasets import DATASET_CONFIGS

CH = ["Fp1", "C3", "C4", "A1", "A2", "LOC", "ROC", "EMG Chin", "EMG Li", "EMG Re", "ECG", "CPU"]
FS = [256] * len(CH)
NAME = {0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
N_EPOCHS = 20  # EDF length in epochs


def _csv(path: Path, rows, crlf=True):
    """rows: [(index, stage_name)] -> UTF-16 (BOM) file, blank lines between rows like the real files."""
    nl = "\r\n" if crlf else "\n"
    path.write_bytes(("".join(f"{i}\t{s}{nl}{nl}" for i, s in rows)).encode("utf-16"))


def _subject(root: Path, folder: str, stem: str, rows, n_epochs=N_EPOCHS, csv_suffix="_annotations.csv"):
    from tests.factories.edf import write_fake_edf

    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    write_fake_edf(d / f"{stem}.EDF", n_channels=len(CH), duration_sec=30 * n_epochs, fs_per_channel=FS, channel_names=CH)
    _csv(d / f"{stem}{csv_suffix}", rows)


def _rows(start, stages):
    return [(start + i, NAME[s]) for i, s in enumerate(stages)]


STAGES = [0, 2, 2, 3, 3, 2, 4, 4, 2, 0]  # 10 scored epochs


def test_registered():
    assert REGISTRY["lsec"] is LSEC
    assert LSEC.DATASET_NAME == "lsec"
    assert "lsec" in DATASET_CONFIGS
    assert DATASET_CONFIGS["lsec"].class_name == "LeuvenSleepEpilepsyChildrenDataset"


def test_parse_labels_utf16_index_not_from_zero(tmp_path):
    p = tmp_path / "x_annotations.csv"
    _csv(p, [(5, "WAKE"), (6, "N1"), (7, "N2"), (8, "N3"), (9, "REM"), (11, "N2")])
    lab = parse_lsec_labels(p)
    assert lab.dtype == np.int16
    assert lab.tolist() == [-1] * 5 + [0, 1, 2, 3, 4, -1, 2]  # index 10 has no line -> -1
    _csv(p, [(0, "WAKE"), (1, "N2")], crlf=False)
    assert parse_lsec_labels(p).tolist() == [0, 2]


def test_parse_labels_errors(tmp_path):
    p = tmp_path / "x.csv"
    _csv(p, [(0, "N4")])
    with pytest.raises(ValueError, match="unknown stage"):
        parse_lsec_labels(p)
    p.write_bytes("0\tN1\textra\n".encode("utf-16"))
    with pytest.raises(ValueError, match="expected"):
        parse_lsec_labels(p)
    _csv(p, [])
    assert parse_lsec_labels(p).size == 0


def test_shift_and_cut():
    lab = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    assert shift_and_cut(lab, 0, 5).tolist() == [0, 1, 2, 3, 4]
    assert shift_and_cut(lab, 0, 3).tolist() == [0, 1, 2]  # past the end of the signal: dropped
    assert shift_and_cut(lab, 0, 8).tolist() == [0, 1, 2, 3, 4, -1, -1, -1]
    assert shift_and_cut(lab, 2, 6).tolist() == [-1, -1, 0, 1, 2, 3]  # signal epoch = index + offset
    assert shift_and_cut(lab, -2, 5).tolist() == [2, 3, 4, -1, -1]
    assert shift_and_cut(lab, -9, 5).tolist() == [-1] * 5
    assert shift_and_cut(lab, 9, 5).tolist() == [-1] * 5
    assert shift_and_cut(np.array([], dtype=np.int16), 3, 4).tolist() == [-1] * 4


def test_header_fields(tmp_path):
    h = bytearray(b" " * 256)
    h[0:1] = b"0"
    pat = b"##0123456789abcdef F 2010-03-01"
    h[8 : 8 + len(pat)] = pat
    h[168:176] = b"01.03.22"
    p = tmp_path / "x.EDF"
    p.write_bytes(bytes(h))
    f = _edf_header_fields(p)
    assert f["sex"] == "F" and f["patient_code"] == "##0123456789abcdef"
    assert f["birthdate"] == "2010-03-01" and f["age_years"] == pytest.approx(12.0, abs=0.01)
    # a header without the fields never raises
    q = tmp_path / "y.EDF"
    q.write_bytes(b" " * 256)
    assert _edf_header_fields(q)["age_years"] is None


def test_listing_groups_typo_and_ignored_files(tmp_path):
    _subject(tmp_path, "DRE", "DRE_1", _rows(3, STAGES))
    _subject(tmp_path, "WCE", "WCE_2", _rows(3, STAGES))
    _subject(tmp_path, "Patienten zonder epilepsie", "Control_30", _rows(3, STAGES), csv_suffix="_annontations.csv")
    _subject(tmp_path, "Patienten zonder epilepsie", "Control_31", _rows(3, STAGES))
    # ignored: derived exports, no annotation file, sidecar, extra scorings
    d = tmp_path / "DRE"
    for junk in ("DRE_24_CONTINUOUS.edf", "DRE_24_SD.edf", "DRE_1.EDF.fst", "DRE_1.EDF.ftg"):
        (d / junk).write_bytes(b"")
    from tests.factories.edf import write_fake_edf

    write_fake_edf(d / "DRE_9.EDF", n_channels=len(CH), duration_sec=60, fs_per_channel=FS, channel_names=CH)  # no CSV
    _csv(tmp_path / "Patienten zonder epilepsie" / "Control_8_annotations_2.csv", _rows(0, STAGES))
    with tempfile.TemporaryDirectory() as cache:
        ds = LSEC(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="raw", sequence_length=4, cache_dir=cache)
        assert sorted(ds.get_subjects()) == ["Control_30", "Control_31", "DRE_1", "WCE_2"]
        groups = {sid: ds.get_subject_metadata(sid)["group"] for sid in ds.get_subjects()}
        assert groups == {"Control_30": "Control", "Control_31": "Control", "DRE_1": "DRE", "WCE_2": "WCE"}
        only = LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache, subset="WCE")
        assert only.get_subjects() == ["WCE_2"]
        ex = LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache,
                  exclude_subjects=["DRE_1"])
        assert "DRE_1" not in ex.get_subjects()
        with pytest.raises(ValueError):
            LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache, subset="XYZ")


def test_channel_resolution_and_derivation(tmp_path):
    avail = CH + ["Rec Cntr"]
    fs = {a: 256.0 for a in avail}
    res = resolve_channels(["EEG", "EOG", "EMG"], avail, fs, LSEC.CHANNEL_PREFERENCES)
    assert [r.physical for r in res] == [("C3", "A2"), ("LOC", "ROC"), "EMG Chin"]
    _subject(tmp_path, "DRE", "DRE_1", _rows(3, STAGES))
    with tempfile.TemporaryDirectory() as cache:
        c3 = LSEC(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="raw", sequence_length=4, cache_dir=cache)
        c4 = LSEC(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="raw", sequence_length=4, cache_dir=cache,
                  eeg_derivation="C4-A1")
        assert [r.physical for r in c3._resolved["DRE_1"]] == [("C3", "A2"), ("LOC", "ROC"), "EMG Chin"]
        assert [r.physical for r in c4._resolved["DRE_1"]] == [("C4", "A1"), ("LOC", "ROC"), "EMG Chin"]
        both = LSEC(root=str(tmp_path), channels=["EEG", "EEG"], pipelines="raw", sequence_length=4, cache_dir=cache)
        assert [r.physical for r in both._resolved["DRE_1"]] == [("C3", "A2"), ("C4", "A1")]
        with pytest.raises(ValueError):
            LSEC(root=str(tmp_path), eeg_derivation="Cz-A1")


def test_leg_emg_is_never_chin():
    assert LSEC.CHANNEL_PREFERENCES["EMG"] == ["EMG Chin"]
    res = resolve_channels(["EMG"], ["EMG Li", "EMG Re"], {"EMG Li": 256.0, "EMG Re": 256.0},
                           LSEC.CHANNEL_PREFERENCES, allow_missing=True)
    assert res == [None]


def test_labels_index_mode_cut_at_edf_end(tmp_path):
    # index 15..24 on a 20-epoch EDF: epochs 20..24 are past the end of the signal
    _subject(tmp_path, "DRE", "DRE_1", _rows(15, STAGES))
    with tempfile.TemporaryDirectory() as cache:
        ds = LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache)
        lab = ds._get_labels(ds._subjects[0])
        assert len(lab) == N_EPOCHS
        assert lab[:15].tolist() == [-1] * 15
        assert lab[15:].tolist() == STAGES[:5]


def test_labels_audited_mode_offsets(tmp_path):
    _subject(tmp_path, "DRE", "DRE_1", _rows(15, STAGES))
    _subject(tmp_path, "WCE", "WCE_2", _rows(2, STAGES))
    off = tmp_path / "offsets.csv"
    off.write_text("subject_id,offset_epochs\nDRE_1,-5\nWCE_2,0\n")
    with tempfile.TemporaryDirectory() as cache:
        ds = LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache,
                  label_offset="audited", label_offsets_path=str(off))
        d = {s.subject_id: ds._get_labels(s) for s in ds._subjects}
        assert d["DRE_1"][10:20].tolist() == STAGES  # shifted back by 5: now fully inside the signal
        assert d["WCE_2"][2:12].tolist() == STAGES
        # same cache directory, other mode: the raw labels on disk are reused, only the runtime step differs
        raw = LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache)
        assert raw._get_labels(raw._subjects[0])[15:].tolist() == STAGES[:5]
        assert raw._get_labels(raw._subjects[1])[2:12].tolist() == STAGES


def test_audited_mode_requires_offsets(tmp_path):
    with pytest.raises(ValueError, match="label_offsets_path"):
        LSEC(root=str(tmp_path), label_offset="audited")
    with pytest.raises(ValueError):
        LSEC(root=str(tmp_path), label_offset="bogus")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="subject_id"):
        LSEC(root=str(tmp_path), label_offset="audited", label_offsets_path=str(bad))
    _subject(tmp_path, "DRE", "DRE_1", _rows(3, STAGES))
    ok = tmp_path / "ok.csv"
    ok.write_text("subject_id,offset_epochs\nDRE_7,0\n")
    with tempfile.TemporaryDirectory() as cache:
        # a recording without an offset is an error in audited mode (never a silent 0)
        with pytest.raises(KeyError):
            LSEC(root=str(tmp_path), channels=["EEG"], pipelines="raw", sequence_length=4, cache_dir=cache,
                 label_offset="audited", label_offsets_path=str(ok))


def test_end_to_end_spectrogram(tmp_path):
    _subject(tmp_path, "DRE", "DRE_1", _rows(3, STAGES))
    with tempfile.TemporaryDirectory() as cache:
        ds = LSEC(root=str(tmp_path), channels=["EEG", "EOG", "EMG"], pipelines="seqsleepnet", sequence_length=4,
                  cache_dir=cache)
        item = ds[0]
        assert len(item["channel_order"]) == 3
        for name in item["channel_order"]:
            assert tuple(item["signals"][name].shape) == (4, 29, 129)
        last = ds[len(ds) - 1]
        assert last["labels"].shape[0] == 4
        assert item["subject"]["dataset"] == "lsec"
        assert item["subject"]["group"] == "DRE"
