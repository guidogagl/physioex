"""End-to-end integration tests for BasePhysioDataset using synthetic fake EDF.

The synthetic generators live in ``tests/factories/edf.py`` (single source of
truth); they are re-exported here for the test modules that still import
``from tests.test_raw_dataset_integration import ...``.
"""
import tempfile
from pathlib import Path

import torch

# Re-export the synthetic factories (canonical home: tests.factories.edf).
from tests.factories.edf import (  # noqa: F401
    FakeEDFDataset,
    write_fake_annotations_edf,
    write_fake_edf,
)


def test_first_access_triggers_cache():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB01.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB01_sleepscoring.edf",
                                   stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB01",
            channels=["EEG", "EOG", "EMG"], pipelines="raw",
            sequence_length=5, cache_dir=cache_dir,
        )
        item = ds[0]

        cache_root = Path(cache_dir) / "v1" / "fake_integration_test"
        assert (cache_root / "headers" / "SUB01.json").exists()
        assert (cache_root / "labels" / "SUB01.npy").exists()
        signals_dir = cache_root / "signals" / "SUB01"
        assert signals_dir.exists() and any(signals_dir.iterdir())

        assert {"signals", "labels", "subject", "channel_order"} <= set(item)
        assert len(item["channel_order"]) == 3
        assert item["labels"].shape == (5,)


def test_second_access_reads_from_cache():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB02.edf", duration_sec=300)
        write_fake_annotations_edf(data_dir / "SUB02_sleepscoring.edf", stages=["W"] * 10)

        FakeEDFDataset(
            root=str(data_dir), subject_id="SUB02", channels=["EEG"],
            pipelines="raw", sequence_length=3, cache_dir=cache_dir,
        )[0]  # triggers cache

        ds2 = FakeEDFDataset(
            root=str(data_dir), subject_id="SUB02", channels=["EEG"],
            pipelines="raw", sequence_length=3, cache_dir=cache_dir,
        )
        call_count = {"n": 0}
        orig = ds2._read_subject_channel

        def wrapped(spec, resolved):
            call_count["n"] += 1
            return orig(spec, resolved)

        ds2._read_subject_channel = wrapped
        _ = ds2[0]
        _ = ds2[1]
        assert call_count["n"] == 0, f"expected 0 EDF reads on cache hit, got {call_count['n']}"


def test_different_pipelines_coexist():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB03.edf", duration_sec=300)
        write_fake_annotations_edf(data_dir / "SUB03_sleepscoring.edf", stages=["W"] * 10)

        FakeEDFDataset(root=str(data_dir), subject_id="SUB03", channels=["EEG"],
                       pipelines="raw", sequence_length=3, cache_dir=cache_dir)[0]
        FakeEDFDataset(root=str(data_dir), subject_id="SUB03", channels=["EEG"],
                       pipelines="seqsleepnet", sequence_length=3, cache_dir=cache_dir)[0]

        ch_dir = Path(cache_dir) / "v1" / "fake_integration_test" / "signals" / "SUB03"
        channel_subdirs = list(ch_dir.iterdir())
        assert len(channel_subdirs) == 1
        assert len(list(channel_subdirs[0].iterdir())) == 2


def test_labels_shared_across_pipelines():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB04.edf", duration_sec=300)
        write_fake_annotations_edf(data_dir / "SUB04_sleepscoring.edf", stages=["W"] * 10)

        for pipeline_name in ["raw", "seqsleepnet"]:
            FakeEDFDataset(root=str(data_dir), subject_id="SUB04", channels=["EEG"],
                           pipelines=pipeline_name, sequence_length=3, cache_dir=cache_dir)[0]

        labels_dir = Path(cache_dir) / "v1" / "fake_integration_test" / "labels"
        assert len(list(labels_dir.glob("SUB04*"))) == 2  # .npy + .meta.json


def test_two_eeg_channels():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB05.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB05_sleepscoring.edf", stages=["W"] * 10)

        ds = FakeEDFDataset(root=str(data_dir), subject_id="SUB05", channels=["EEG", "EEG"],
                            pipelines="raw", sequence_length=3, cache_dir=cache_dir)
        ordered = ds[0]["channel_order"]
        assert len(ordered) == 2
        assert ordered[0] != ordered[1], f"expected distinct channels, got {ordered}"


def test_mixed_specific_and_generic():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB06.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB06_sleepscoring.edf", stages=["W"] * 10)

        ds = FakeEDFDataset(root=str(data_dir), subject_id="SUB06", channels=["C4-M2", "EOG"],
                            pipelines="raw", sequence_length=3, cache_dir=cache_dir)
        order = ds[0]["channel_order"]
        # Channel keys are modality-based (e.g. "EEG_0"/"EOG_0"), not physical names.
        assert len(order) == 2
        assert any("EEG" in c for c in order), order
        assert any("EOG" in c for c in order), order


def test_header_probe_cache():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB07.edf", duration_sec=60,
                       channel_names=["C4-M2", "EOG", "EMG"], n_channels=3)
        write_fake_annotations_edf(data_dir / "SUB07_sleepscoring.edf", stages=["W", "W"])

        ds = FakeEDFDataset(root=str(data_dir), subject_id="SUB07",
                            channels=["EEG", "EOG", "EMG"], pipelines="raw",
                            sequence_length=1, cache_dir=cache_dir)
        info = ds.probe()
        assert "available_channels" in info and "channel_fs" in info
        header_json = Path(cache_dir) / "v1" / "fake_integration_test" / "headers" / "SUB07.json"
        assert header_json.exists()


def test_dict_schema():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as cache_dir:
        data_dir = Path(data_dir)
        write_fake_edf(data_dir / "SUB08.edf", duration_sec=300,
                       channel_names=["C4-M2", "C3-M1", "EOG", "EMG"])
        write_fake_annotations_edf(data_dir / "SUB08_sleepscoring.edf",
                                   stages=["W", "N1", "N2", "N3", "R"] * 2)

        ds = FakeEDFDataset(root=str(data_dir), subject_id="SUB08",
                            channels=["EEG", "EOG", "EMG"], pipelines="raw",
                            sequence_length=5, cache_dir=cache_dir)
        item = ds[0]
        assert isinstance(item["signals"], dict)
        assert isinstance(item["channel_order"], list)
        assert isinstance(item["channel_info"], dict)
        assert torch.is_tensor(item["labels"]) and item["labels"].shape[0] == 5
        assert item["subject"]["dataset"] == "fake_integration_test"
        assert item["epoch_indices"].shape == (5,)
        for key in item["channel_order"]:
            assert key in item["signals"]
            assert torch.is_tensor(item["signals"][key])
            info = item["channel_info"][key]
            assert {"fs_in", "pipeline_hash", "modality"} <= set(info)
