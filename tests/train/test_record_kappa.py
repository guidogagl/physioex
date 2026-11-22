"""Pooled and per-record Cohen's kappa against scikit-learn."""
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from physioex.train.metrics import (
    kappa_from_confusion,
    median_record_kappa,
    per_record_kappa,
    pooled_kappa,
    record_confusion_matrix,
)


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    trues, preds = [], []
    for n in (50, 80, 120):
        t = rng.integers(0, 4, n)
        p = np.where(rng.random(n) < 0.7, t, rng.integers(0, 4, n))
        t[:3] = -1  # unscored epochs
        trues.append(t)
        preds.append(p)
    return trues, preds


def test_per_record_matches_sklearn(records):
    trues, preds = records
    got = per_record_kappa(trues, preds, 4)
    want = [cohen_kappa_score(t[t >= 0], p[t >= 0]) for t, p in zip(trues, preds)]
    np.testing.assert_allclose(got, want, atol=1e-12)
    assert median_record_kappa(trues, preds, 4) == pytest.approx(np.median(want))


def test_pooled_matches_sklearn_on_concatenation(records):
    trues, preds = records
    t = np.concatenate(trues)
    p = np.concatenate(preds)
    assert pooled_kappa(trues, preds, 4) == pytest.approx(
        cohen_kappa_score(t[t >= 0], p[t >= 0])
    )


def test_confusion_drops_ignored_and_checks_range():
    cm = record_confusion_matrix([0, 1, -1, 2], [0, 2, 3, 2], 3)
    assert cm.sum() == 3 and cm[1, 2] == 1
    with pytest.raises(ValueError):
        record_confusion_matrix([0, 5], [0, 0], 3)


def test_undefined_kappa_is_nan_and_skipped_by_median():
    assert np.isnan(kappa_from_confusion(np.zeros((4, 4))))
    trues = [np.zeros(10, int), np.array([0, 1, 0, 1])]
    preds = [np.zeros(10, int), np.array([0, 1, 0, 1])]
    k = per_record_kappa(trues, preds, 2)
    assert np.isnan(k[0]) and k[1] == pytest.approx(1.0)
    assert median_record_kappa(trues, preds, 2) == pytest.approx(1.0)
