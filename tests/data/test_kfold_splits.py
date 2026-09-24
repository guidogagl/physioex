"""Subject-level k-fold split utilities (no data on disk needed)."""
from types import SimpleNamespace

import pytest

from physioex.data.splits import assign_kfold_splits, kfold_subject_splits


def _ids(n, prefix="s"):
    return [f"{prefix}{i:03d}" for i in range(n)]


def test_phan_mass_protocol_20_fold_180_10_10():
    ids = _ids(200)
    seen_test = []
    for fold in range(20):
        train, valid, test = kfold_subject_splits(ids, n_folds=20, fold=fold, n_valid=10)
        assert (len(train), len(valid), len(test)) == (180, 10, 10)
        assert not (set(train) & set(valid)) and not (set(train) & set(test)) and not (set(valid) & set(test))
        assert set(train) | set(valid) | set(test) == set(ids)
        seen_test.extend(test)
    assert sorted(seen_test) == sorted(ids)  # every subject tested exactly once


def test_loso_with_grouped_nights_keeps_groups_together():
    # 20 subjects x 2 nights, one subject with a single night (SleepEDF-SC 2013: 39 recordings)
    ids = [f"SC4{s:02d}{n}E0" for s in range(20) for n in (1, 2)][:-1]
    group = lambda sid: sid[:5]  # noqa: E731
    all_test = []
    for fold in range(20):
        train, valid, test = kfold_subject_splits(ids, n_folds=20, fold=fold, n_valid=4, group_fn=group)
        assert len({group(s) for s in test}) == 1
        assert len({group(s) for s in valid}) == 4
        for part in (train, valid, test):
            keys = {group(s) for s in part}
            # all nights of every group in the part are present
            assert all(s in part for s in ids if group(s) in keys)
        all_test.extend(test)
    assert sorted(all_test) == sorted(ids)


def test_fold_is_deterministic_and_seed_changes_it():
    ids = _ids(30)
    a = kfold_subject_splits(ids, 5, 2, n_valid=3, seed=1)
    b = kfold_subject_splits(ids, 5, 2, n_valid=3, seed=1)
    c = kfold_subject_splits(ids, 5, 2, n_valid=3, seed=2)
    assert a == b and a != c


def test_invalid_arguments():
    ids = _ids(10)
    with pytest.raises(ValueError):
        kfold_subject_splits(ids, 5, 5)
    with pytest.raises(ValueError):
        kfold_subject_splits(ids, 11, 0)
    with pytest.raises(ValueError):
        kfold_subject_splits(ids, 10, 0, n_valid=9)


class _FakeDataset:
    def __init__(self, ids):
        self._subjects = [SimpleNamespace(subject_id=s) for s in ids]
        self._split_fn = None

    def set_split_fn(self, fn):
        self._split_fn = fn


def test_assign_kfold_splits_is_joint_across_datasets():
    cohorts = [_FakeDataset(_ids(53, "a")), _FakeDataset(_ids(19, "b")), _FakeDataset(_ids(128, "c"))]
    assign_kfold_splits(cohorts, n_folds=20, n_valid=10)
    for fold in range(20):
        parts = [ds._split_fn(fold) for ds in cohorts]
        train = sum((p[0] for p in parts), [])
        valid = sum((p[1] for p in parts), [])
        test = sum((p[2] for p in parts), [])
        assert (len(train), len(valid), len(test)) == (180, 10, 10)
        for ds, (tr, va, te) in zip(cohorts, parts):
            own = {s.subject_id for s in ds._subjects}
            assert set(tr) | set(va) | set(te) == own


def test_assign_kfold_splits_tolerates_id_collisions():
    cohorts = [_FakeDataset(_ids(10)), _FakeDataset(_ids(10))]  # identical ids
    assign_kfold_splits(cohorts, n_folds=5, n_valid=2)
    tr0, va0, te0 = cohorts[0]._split_fn(0)
    tr1, va1, te1 = cohorts[1]._split_fn(0)
    assert len(tr0) + len(va0) + len(te0) == 10 and len(tr1) + len(va1) + len(te1) == 10
    assert len(te0) + len(te1) == 4
