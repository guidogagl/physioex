"""Clean channels as virtual channels of a dataset: resolution, cache, metadata, no regression."""

import numpy as np
import pytest

from physioex.data.cleaning import CleanMontage
from physioex.data.pipeline import PreprocessingPipeline
from physioex.data.steps import Resample
from tests.factories.edf import (
    FakeEDFDataset,
    write_fake_annotations_edf,
    write_fake_edf,
)

STAGES = ["W"] * 4 + ["N2"] * 6 + ["N3"] * 4 + ["R"] * 4 + ["N2"] * 2
LABELS = ["C4-M1", "C3-M2", "E1", "E2", "Chin1", "Chin2", "ECG"]


class CleanFake(FakeEDFDataset):
    DATASET_NAME = "fake_clean_channels"
    CHANNEL_PREFERENCES = {"EEG": ["C4-M1"], "EOG": [("E1", "E2")], "EMG": [("Chin1", "Chin2")]}
    CLEAN_MONTAGE = CleanMontage(
        eeg=["C4-M1"],
        eog=[("E1", "E2")],
        chin=[("Chin1", "Chin2")],
        ecg=["ECG"],
        extra={"C3": ["C3-M2"]},
        extra_eeg=["C3"],
    )


class PlainFake(CleanFake):
    CLEAN_MONTAGE = None


PIPES = {"__default__": "raw", "EMG_ENV_clean": PreprocessingPipeline([Resample(100.0)])}


@pytest.fixture
def make(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_fake_edf(
        data / "sub01.edf",
        n_channels=len(LABELS),
        duration_sec=len(STAGES) * 30,
        fs_per_channel=[200] * len(LABELS),
        channel_names=LABELS,
    )
    write_fake_annotations_edf(data / "sub01_sleepscoring.edf", STAGES)

    def _make(cls, channels, cache="cache", pipelines="raw"):
        return cls(
            root=str(data),
            subject_id="sub01",
            channels=channels,
            pipelines=pipelines,
            sequence_length=1,
            cache_dir=str(tmp_path / cache),
        )

    return _make


def _signals(ds):
    return np.stack([np.asarray(ds[i]["signals"]) for i in range(len(ds))])


def test_classic_channels_unchanged(make):
    """A recipe on the class changes nothing for the channels asked as before."""
    chans = ["EEG", "EOG", "EMG"]
    a, b = make(CleanFake, chans, "c1"), make(PlainFake, chans, "c2")
    np.testing.assert_array_equal(_signals(a), _signals(b))
    assert "clean_level" not in a.get_subject_metadata("sub01")


def test_clean_channels(make):
    chans = ["EEG_clean", "EOG_clean", "EMG_ENV_clean", "LEG_L_clean", "EEG"]
    ds = make(CleanFake, chans, pipelines=PIPES)
    x = _signals(ds)
    assert x.shape[0] == len(STAGES) and x.shape[-1] == 3000
    sig = x.reshape(len(STAGES), len(chans), -1) if x.ndim > 3 else x
    assert np.abs(sig[:, 0]).sum() > 0 and np.abs(sig[:, 1]).sum() > 0
    assert np.abs(sig[:, 3]).sum() == 0  # no leg in the recording: zero-filled
    # requested order kept: the classic EEG is not the clean one
    assert not np.allclose(sig[:, 0], sig[:, 4])
    assert ds.get_subject_metadata("sub01")["clean_level"] == "poor"
    diag = ds.get_clean_diagnostics("sub01")
    assert diag["level"] == "poor" and diag["resolved"]["eog"] == "E1-E2"

    # a second dataset on the same cache reads the cleaned night back, identical
    again = make(CleanFake, chans, pipelines=PIPES)
    np.testing.assert_allclose(_signals(again), x, atol=1e-3)


def test_no_recipe_zero_fills(make):
    ds = make(PlainFake, ["EEG_clean", "EEG"])
    x = _signals(ds)
    assert np.abs(x[:, 0]).sum() == 0 and np.abs(x[:, 1]).sum() > 0
