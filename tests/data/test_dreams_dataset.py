"""DREAMS micro-event databases: discovery, 5 s hypnogram, expert events, tolerant EDF, clean."""
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from physioex.data.datasets import get_dataset
from physioex.data.datasets.dreams import DREAMSDataset, hypnogram_to_epochs
from physioex.data.readers.annotations import parse_onset_duration_txt
from physioex.data.readers.edf import read_edf_lenient
from tests.factories.edf import write_fake_edf

KC_CH = ["FP1-A2", "CZ-A1", "EMG1", "EOG1", "EOG2", "O1-A2", "FP2-A1", "O2-A1", "CZ2-A1", "EMG2", "ECG", "PCPAP"]
N_EP = 4  # 2 min excerpt


def _break_pcpap(path: Path):
    """Make the last channel non-compliant as in DREAMS: physical min = physical max."""
    b = bytearray(path.read_bytes())
    ns = int(b[252:256])
    off = 256 + ns * (16 + 80 + 8)
    i = ns - 1
    for base in (off, off + 8 * ns):  # physical min, physical max
        b[base + 8 * i:base + 8 * (i + 1)] = b"0       "
    path.write_bytes(bytes(b))


@pytest.fixture
def root(tmp_path):
    kc = tmp_path / "dreams" / "DatabaseKcomplexes"
    sp = tmp_path / "dreams" / "DatabaseSpindles"
    kc.mkdir(parents=True)
    sp.mkdir(parents=True)
    for d, names, fs in ((kc, KC_CH, 200), (sp, ["O1-A1", "C3-A1", "FP1-A1", "EMG1", "EOG2-A1", "EOG1-A1"], 100)):
        write_fake_edf(d / "excerpt1.edf", n_channels=len(names), duration_sec=N_EP * 30,
                       fs_per_channel=[fs] * len(names), channel_names=names)
    _break_pcpap(kc / "excerpt1.edf")
    # epoch 0: W W W W S1 S1 -> W; epoch 1: S2x3 S3x3 -> unscored; epoch 2: S3x2 S4x4 -> N3; epoch 3: REM
    steps = [5] * 4 + [3] * 2 + [2] * 3 + [1] * 3 + [1] * 2 + [0] * 4 + [4] * 6
    (kc / "Hypnogram_excerpt1.txt").write_text("[Hypnogram]\n" + "\n".join(map(str, steps)) + "\n")
    (kc / "Visual_scoring1_excerpt1.txt").write_text("[vis1_Kcomplexes/CZ-A1]\n   50.4113\t    0.6544\n  101.7100\t    1.1305\n")
    (kc / "Visual_scoring2_excerpt1.txt").write_text("[vis2_Kcomplexes/CZ-A1]\n    1.1100\t    0.7400\n")
    (sp / "Visual_scoring1_excerpt1.txt").write_text("[vis1_Spindles/C3-A1]\n   12.0\t    0.72\n")
    return tmp_path / "dreams"


def _ds(root, tmp_path, **kw):
    kw.setdefault("channels", ["EEG"])
    return DREAMSDataset(root=str(root), pipelines="raw", sequence_length=1, cache_dir=str(tmp_path / "cache"), **kw)


def test_registry():
    assert get_dataset("dreams") is DREAMSDataset


def test_hypnogram_majority():
    steps = np.array([5] * 4 + [3] * 2 + [2] * 3 + [1] * 3 + [1] * 2 + [0] * 4 + [4] * 6 + [-1] * 6)
    np.testing.assert_array_equal(hypnogram_to_epochs(steps), [0, -1, 3, 4, -1])


def test_onset_duration_parser(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("[vis1_Kcomplexes/CZ-A1]\n   50.4113\t    0.6544\n\n  101.71  1.1305\n")
    assert parse_onset_duration_txt(p) == ("vis1_Kcomplexes/CZ-A1", [(50.4113, 0.6544), (101.71, 1.1305)])


def test_lenient_reader_matches_pyedflib(tmp_path):
    p = tmp_path / "x.edf"
    write_fake_edf(p, n_channels=3, duration_sec=60, fs_per_channel=[200, 100, 200], channel_names=["A", "B", "C"])
    e = read_edf_lenient(p)
    with pyedflib.EdfReader(str(p)) as f:
        for i in range(3):
            np.testing.assert_allclose(e["signals"][i], f.readSignal(i), atol=1e-2)
    assert e["fs"][:3] == [200.0, 100.0, 200.0]


def test_discovery_subsets_labels_events(root, tmp_path):
    ds = _ds(root, tmp_path)
    ids = [s.subject_id for s in ds._subjects]
    assert ids == ["kc_excerpt1", "sp_excerpt1"]
    assert [s.subject_id for s in _ds(root, tmp_path, subset="spindles")._subjects] == ["sp_excerpt1"]
    kc = ds._subjects[0]
    np.testing.assert_array_equal(ds._read_subject_labels(kc), [0, -1, 3, 4])
    # no hypnogram: every epoch unscored
    assert (ds._read_subject_labels(ds._subjects[1]) == -1).all()
    ev = ds._read_subject_events(kc)
    assert [(e.type, e.concept, round(e.onset_sec, 2)) for e in ev] == [
        ("k_complex", "vis1", 50.41), ("k_complex", "vis1", 101.71), ("k_complex", "vis2", 1.11)]
    assert ev[0].extra["channel"] == "CZ-A1"
    assert ds._read_subject_events(ds._subjects[1])[0].type == "spindle"


def test_noncompliant_edf_reads_in_uv(root, tmp_path):
    ds = _ds(root, tmp_path, subset="kcomplexes", channels=["EEG", "EOG", "EMG"])
    it = ds[0]
    x = {k: it["signals"][k].numpy().reshape(-1) for k in it["channel_order"]}
    assert len(ds) == N_EP
    for v in x.values():
        assert np.isfinite(v).all() and 5 < v.std() < 100  # fake signal: ~20 µV noise


def test_clean_channels_rich(root, tmp_path):
    ds = _ds(root, tmp_path, subset="kcomplexes", channels=["EEG_clean", "EOG_clean", "EMG_clean"])
    it = ds[0]
    assert all(np.abs(it["signals"][k].numpy()).sum() > 0 for k in it["channel_order"])
    diag = ds.get_clean_diagnostics("kc_excerpt1")
    assert diag["level"] == "rich"
    assert diag["resolved"]["eog"] == "EOG1 - EOG2"
