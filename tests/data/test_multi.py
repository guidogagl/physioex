"""Unit/integration tests for physioex.data.multi.MultiDataset.

Covers flat indexing, negative/out-of-range handling, dataset_idx injection,
cross-dataset ``split`` coordination, the convenience accessors, the
sequence-length invariant, repr, and ``close``.

Builds two single-subject ``FakeEDFDataset`` instances via the shared EDF
factory (no real data required).
"""
from pathlib import Path

import numpy as np
import pytest

from physioex.data.multi import MultiDataset
from tests.factories.edf import (
    write_fake_edf,
    write_fake_annotations_edf,
    FakeEDFDataset,
)


def _make_ds(root: Path, cache: Path, sid: str, n_epochs: int, seq_len: int = 5):
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    write_fake_edf(
        root / f"{sid}.edf",
        duration_sec=n_epochs * 30,
        channel_names=["C4-M2", "EOG", "EMG"],
        n_channels=3,
    )
    write_fake_annotations_edf(
        root / f"{sid}_sleepscoring.edf", stages=["W"] * n_epochs
    )
    return FakeEDFDataset(
        root=str(root),
        subject_id=sid,
        channels=["EEG", "EOG", "EMG"],
        pipelines="raw",
        sequence_length=seq_len,
        cache_dir=str(cache),
    )


@pytest.fixture
def two_datasets(tmp_path):
    da = _make_ds(tmp_path / "a", tmp_path / "ca", "A1", n_epochs=20, seq_len=5)
    db = _make_ds(tmp_path / "b", tmp_path / "cb", "B1", n_epochs=15, seq_len=5)
    return da, db


# ---------------------------------------------------------------------------
# 1. Construction guards
# ---------------------------------------------------------------------------

def test_empty_datasets_rejected():
    with pytest.raises(ValueError):
        MultiDataset([])


def test_sequence_length_must_match(tmp_path):
    da = _make_ds(tmp_path / "a", tmp_path / "ca", "A1", n_epochs=20, seq_len=5)
    db = _make_ds(tmp_path / "b", tmp_path / "cb", "B1", n_epochs=20, seq_len=7)
    with pytest.raises(AssertionError):
        MultiDataset([da, db])


# ---------------------------------------------------------------------------
# 2. Flat indexing / length
# ---------------------------------------------------------------------------

def test_len_is_sum_of_children(two_datasets):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)
    assert len(md) == len(da) + len(db)
    assert md.sequence_length == 5


def test_flat_getitem_resolves_to_right_dataset(two_datasets):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)

    # First index -> dataset 0, last index -> dataset 1
    first = md[0]
    last = md[len(md) - 1]
    assert first["subject"]["dataset_idx"] == 0
    assert last["subject"]["dataset_idx"] == 1
    # An index just past dataset A's length maps into dataset B.
    boundary = md[len(da)]
    assert boundary["subject"]["dataset_idx"] == 1


def test_negative_index(two_datasets):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)
    assert md[-1]["subject"]["dataset_idx"] == md[len(md) - 1]["subject"]["dataset_idx"]


@pytest.mark.parametrize("bad", [-1000, 10_000])
def test_out_of_range_raises(two_datasets, bad):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)
    with pytest.raises(IndexError):
        _ = md[bad]


# ---------------------------------------------------------------------------
# 3. split() coordination
# ---------------------------------------------------------------------------

def test_split_shifts_train_and_retags_subjects(two_datasets):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)

    train, valid, test = md.split(fold=0)

    assert isinstance(train, np.ndarray)
    # Train indices, if any, must live inside the flat index space.
    if train.size:
        assert train.min() >= 0
        assert train.max() < len(md)

    # Every valid/test tuple must be re-tagged with a real constituent index.
    for ds_idx, sid in list(valid) + list(test):
        assert ds_idx in (0, 1)
        assert isinstance(sid, str)

    # Every valid/test subject is a known (ds_idx, sid) pair from the roster.
    tagged = md.get_subjects_with_dataset_idx()
    assert set(valid) | set(test) <= set(tagged)


# ---------------------------------------------------------------------------
# 4. Convenience accessors
# ---------------------------------------------------------------------------

def test_accessors(two_datasets):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)

    assert md.get_n_subjects() == da.get_n_subjects() + db.get_n_subjects()
    assert md.get_subjects() == da.get_subjects() + db.get_subjects()

    tagged = md.get_subjects_with_dataset_idx()
    assert (0, "A1") in tagged
    assert (1, "B1") in tagged

    # datasets property returns a copy (mutating it does not affect internals)
    lst = md.datasets
    lst.clear()
    assert len(md.datasets) == 2

    # available_channels merges children's counts
    merged = md.available_channels()
    assert isinstance(merged, dict)
    assert merged  # non-empty


def test_repr_and_close(two_datasets):
    da, db = two_datasets
    md = MultiDataset([da, db], memmap_cache_size=0)
    r = repr(md)
    assert "MultiDataset" in r and "total=" in r
    # close() must not raise even when children have no memmap cache open.
    md.close()
