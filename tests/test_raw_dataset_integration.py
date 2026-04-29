"""End-to-end integration test for BasePhysioDataset using synthetic fake EDF.

Creates a tiny EDF on-the-fly using pyedflib.EdfWriter, instantiates a trivial
BasePhysioDataset subclass, and verifies the full lazy-load + cache + dict-return
pipeline works.
"""
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import pyedflib

from physioex.data.base import BasePhysioDataset, SubjectSpec
from physioex.data.pipeline import PreprocessingPipeline
from physioex.data.steps import Identity, BandpassFilter, Resample, XSleepNetSpectrogram
from physioex.data.presets import get_preset


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


# -------------------------------------------------------------------
# Helpers: build a tiny fake EDF
# -------------------------------------------------------------------

def write_fake_edf(path: Path, n_channels: int = 4, duration_sec: float = 300.0,
                    fs_per_channel=None, channel_names=None, seed: int = 0) -> None:
    """Create a tiny fake EDF file at ``path`` with random signals.

    Args:
        n_channels: number of channels to write.
        duration_sec: total duration (must be an integer number of 30s epochs for
                     clean label alignment).
        fs_per_channel: list of sample rates, one per channel. Default: all 100Hz.
        channel_names: list of channel labels. Default: ["C4-M2","C3-M1","EOG","EMG"].
    """
    if fs_per_channel is None:
        fs_per_channel = [100] * n_channels
    if channel_names is None:
        defaults = ["C4-M2", "C3-M1", "EOG", "EMG"]
        channel_names = defaults[:n_channels]

    if len(fs_per_channel) != n_channels or len(channel_names) != n_channels:
        raise ValueError("fs_per_channel and channel_names must have length n_channels")

    rng = np.random.default_rng(seed)
    duration_sec = int(duration_sec)
    # pyedflib wants "N data records" and each record has a set number of samples per channel.
    # Simplest: 1 record per second.
    n_records = duration_sec
    signals = []
    headers = []
    for i in range(n_channels):
        fs = fs_per_channel[i]
        # Generate a random signal + a low-freq component for filter testability
        t = np.arange(n_records * fs) / fs
        x = (rng.standard_normal(n_records * fs).astype(np.float64) * 20.0
             + 10.0 * np.sin(2 * np.pi * 1.0 * t))  # add 1Hz component
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
        writer.setPatientCode("SUB01")
        writer.setPatientName("TestPatient")
        writer.writeSamples(signals)
    finally:
        writer.close()


def write_fake_annotations_edf(path: Path, stages: list, epoch_sec: float = 30.0):
    """Write an EDF with only annotations -- used as a sleepscoring sidecar."""
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
        # Must write at least one sample covering the total duration
        writer.writeSamples([np.zeros(total_duration, dtype=np.float64)])
        # Write annotations
        for i, s in enumerate(stages):
            writer.writeAnnotation(i * epoch_sec, epoch_sec, s)
    finally:
        writer.close()


# -------------------------------------------------------------------
# Trivial subclass that uses the fake files
# -------------------------------------------------------------------

class FakeEDFDataset(BasePhysioDataset):
    DATASET_NAME = "fake_integration_test"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0
    CHANNEL_PREFERENCES = {
        "EEG": [("C4", "M2"), "C4-M2", "C3-M1", "EEG"],
        "EOG": ["EOG"],
        "EMG": ["EMG"],
    }

    def __init__(self, root, subject_id: str, **kwargs):
        self._fake_subject_id = subject_id
        super().__init__(root=root, **kwargs)

    def _list_subjects(self):
        root = Path(self.root)
        return [SubjectSpec(
            subject_id=self._fake_subject_id,
            edf_path=root / f"{self._fake_subject_id}.edf",
            label_path=root / f"{self._fake_subject_id}_sleepscoring.edf",
        )]

    def _read_subject_labels(self, spec):
        with pyedflib.EdfReader(str(spec.label_path)) as f:
            onsets, durations, stages = f.readAnnotations()
        if len(stages) == 0:
            return np.array([], dtype=np.int16)
        # Map stage strings to integers (use HMC-like mapping)
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


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_first_access_triggers_cache():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        # 300s = 10 epochs
        write_fake_edf(data_dir / "SUB01.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB01_sleepscoring.edf",
                                    stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB01",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw",
            sequence_length=5,
            cache_dir=cache_dir,
        )

        # Access index 0
        item = ds[0]

        # Verify cache files appeared
        cache_root = Path(cache_dir) / "v1" / "fake_integration_test"
        header_file = cache_root / "headers" / "SUB01.json"
        labels_file = cache_root / "labels" / "SUB01.npy"
        signals_dir = cache_root / "signals" / "SUB01"

        assert header_file.exists(), f"header cache missing: {header_file}"
        assert labels_file.exists(), f"labels cache missing: {labels_file}"
        assert signals_dir.exists() and any(signals_dir.iterdir()), "signal cache missing"

        # Verify dict format
        assert "signals" in item and "labels" in item and "subject" in item
        assert "channel_order" in item and len(item["channel_order"]) == 3
        assert item["labels"].shape == (5,)
        report("first access triggers cache + dict format correct", True)


def test_second_access_reads_from_cache():
    """After first access, pipeline should NOT be re-invoked on second access."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB02.edf", duration_sec=300)
        write_fake_annotations_edf(data_dir / "SUB02_sleepscoring.edf",
                                    stages=["W"] * 10)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB02",
            channels=["EEG"],
            pipelines="raw",
            sequence_length=3,
            cache_dir=cache_dir,
        )

        _ = ds[0]  # triggers cache

        # Instrument: count compile() calls on a new dataset with same cache
        ds2 = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB02",
            channels=["EEG"],
            pipelines="raw",
            sequence_length=3,
            cache_dir=cache_dir,
        )

        # Count how many times _read_subject_channel is called (should be 0 when cache hits)
        call_count = {"n": 0}
        orig = ds2._read_subject_channel
        def wrapped(spec, resolved):
            call_count["n"] += 1
            return orig(spec, resolved)
        ds2._read_subject_channel = wrapped

        _ = ds2[0]
        _ = ds2[1]
        assert call_count["n"] == 0, f"expected 0 EDF reads on cache hit, got {call_count['n']}"
        report("second access uses cache (no re-read)", True)


def test_different_pipelines_coexist():
    """Two pipelines on the same subject produce separate cache entries."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB03.edf", duration_sec=300)
        write_fake_annotations_edf(data_dir / "SUB03_sleepscoring.edf",
                                    stages=["W"] * 10)

        ds_raw = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB03",
            channels=["EEG"], pipelines="raw", sequence_length=3, cache_dir=cache_dir,
        )
        _ = ds_raw[0]

        ds_ss = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB03",
            channels=["EEG"], pipelines="seqsleepnet", sequence_length=3, cache_dir=cache_dir,
        )
        _ = ds_ss[0]

        # Under v1/fake_integration_test/signals/SUB03/{channel}/ there must be 2 subdirs
        ch_dir = Path(cache_dir) / "v1" / "fake_integration_test" / "signals" / "SUB03"
        # one channel subdir
        assert ch_dir.exists()
        channel_subdirs = list(ch_dir.iterdir())
        assert len(channel_subdirs) == 1, f"expected 1 channel dir, got {channel_subdirs}"
        pipeline_dirs = list(channel_subdirs[0].iterdir())
        assert len(pipeline_dirs) == 2, f"expected 2 pipeline dirs, got {pipeline_dirs}"
        report("raw + seqsleepnet caches coexist", True)


def test_labels_shared_across_pipelines():
    """Labels are cached once per subject, shared across all pipelines."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB04.edf", duration_sec=300)
        write_fake_annotations_edf(data_dir / "SUB04_sleepscoring.edf",
                                    stages=["W"] * 10)

        for pipeline_name in ["raw", "seqsleepnet"]:
            ds = FakeEDFDataset(
                root=str(data_dir), subject_id="SUB04",
                channels=["EEG"], pipelines=pipeline_name,
                sequence_length=3, cache_dir=cache_dir,
            )
            _ = ds[0]

        labels_dir = Path(cache_dir) / "v1" / "fake_integration_test" / "labels"
        label_files = list(labels_dir.glob("SUB04*"))
        # one .npy + one .meta.json = 2 files
        assert len(label_files) == 2, f"expected 2 label files, got {label_files}"
        report("labels cached once, shared across pipelines", True)


def test_two_eeg_channels():
    """Requesting ['EEG', 'EEG'] yields two distinct physical channels."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB05.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB05_sleepscoring.edf",
                                    stages=["W"] * 10)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB05",
            channels=["EEG", "EEG"],
            pipelines="raw", sequence_length=3, cache_dir=cache_dir,
        )
        item = ds[0]
        assert len(item["channel_order"]) == 2
        ordered = item["channel_order"]
        assert ordered[0] != ordered[1], f"expected distinct channels, got {ordered}"
        report("['EEG', 'EEG'] -> two distinct physical channels", True)


def test_mixed_specific_and_generic():
    """Requesting ['C4-M2', 'EOG'] works (specific + generic)."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB06.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB06_sleepscoring.edf",
                                    stages=["W"] * 10)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB06",
            channels=["C4-M2", "EOG"],
            pipelines="raw", sequence_length=3, cache_dir=cache_dir,
        )
        item = ds[0]
        assert "C4-M2" in item["channel_order"]
        assert any("EOG" in c for c in item["channel_order"]), f"EOG not found in {item['channel_order']}"
        report("mixed specific + generic channels", True)


def test_header_probe_cache():
    """Probe JSON is written on first probe()."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB07.edf", duration_sec=60,
                       channel_names=["C4-M2", "EOG", "EMG"], n_channels=3)
        write_fake_annotations_edf(data_dir / "SUB07_sleepscoring.edf", stages=["W", "W"])

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB07",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw", sequence_length=1, cache_dir=cache_dir,
        )
        info = ds.probe()
        assert "available_channels" in info
        assert "channel_fs" in info
        header_json = Path(cache_dir) / "v1" / "fake_integration_test" / "headers" / "SUB07.json"
        assert header_json.exists(), f"header JSON not cached: {header_json}"
        report("probe() populates header cache", True)


def test_dict_schema():
    """Verify full dict schema."""
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB08.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB08_sleepscoring.edf",
                                    stages=["W", "N1", "N2", "N3", "R", "W", "N1", "N2", "N3", "R"])

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB08",
            channels=["EEG", "EOG", "EMG"],
            pipelines="raw", sequence_length=5, cache_dir=cache_dir,
        )
        item = ds[0]
        # Shape checks
        assert "signals" in item and isinstance(item["signals"], dict)
        assert "channel_order" in item and isinstance(item["channel_order"], list)
        assert "channel_info" in item and isinstance(item["channel_info"], dict)
        assert "labels" in item and torch.is_tensor(item["labels"])
        assert item["labels"].shape[0] == 5
        assert "subject" in item and "id" in item["subject"]
        assert item["subject"]["dataset"] == "fake_integration_test"
        assert "epoch_indices" in item
        assert item["epoch_indices"].shape == (5,)
        # Each signal must be a tensor, and channel_info has the right keys
        for key in item["channel_order"]:
            assert key in item["signals"]
            assert torch.is_tensor(item["signals"][key])
            info = item["channel_info"][key]
            assert "fs_in" in info and "pipeline_hash" in info and "modality" in info
        report("dict return schema complete", True)


# ----- runner -----

if __name__ == "__main__":
    print("=" * 60)
    print("Raw dataset integration tests")
    print("=" * 60)
    test_first_access_triggers_cache()
    test_second_access_reads_from_cache()
    test_different_pipelines_coexist()
    test_labels_shared_across_pipelines()
    test_two_eeg_channels()
    test_mixed_specific_and_generic()
    test_header_probe_cache()
    test_dict_schema()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
