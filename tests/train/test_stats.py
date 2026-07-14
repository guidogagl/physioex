"""Unit tests for physioex.train.stats (per-subject aggregation + CM figure)."""
import math

import numpy as np
import pytest
import torch

from physioex.train import stats as S


def _perfect_subject(n_classes=5, per_class=4):
    """Build (preds, targets) where argmax(preds) == targets exactly."""
    targets = torch.arange(n_classes).repeat_interleave(per_class)
    preds = torch.nn.functional.one_hot(targets, n_classes).float() * 10.0
    return preds, targets


# ---------------------------------------------------------------------------
# per_subject_metrics
# ---------------------------------------------------------------------------

def test_per_subject_metrics_keys_and_perfect_scores():
    preds, targets = _perfect_subject()
    out = S.per_subject_metrics([preds, preds], [targets, targets], n_classes=5)

    # scalar metric keys present, one value per subject
    for name in S.SCALAR_METRICS:
        assert name in out and len(out[name]) == 2
    # per-class keys use "{metric}/{class_name}"
    assert "f1/N2" in out and "precision/W" in out and "recall/REM" in out

    # perfect predictions -> accuracy 1.0 for every subject
    assert all(abs(a - 1.0) < 1e-6 for a in out["accuracy"])
    assert all(abs(f - 1.0) < 1e-6 for f in out["f1/N2"])


def test_per_subject_metrics_respects_ignore_index():
    preds, targets = _perfect_subject()
    targets = targets.clone()
    targets[:2] = -1  # mark two epochs unscored
    out = S.per_subject_metrics([preds], [targets], n_classes=5)
    # ignored epochs excluded -> still perfect on the scored ones
    assert abs(out["accuracy"][0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_single_subject_no_ci():
    per = {"accuracy": [0.8]}
    agg = S.aggregate(per)
    a = agg["accuracy"]
    assert a["n"] == 1
    assert a["mean"] == pytest.approx(0.8)
    assert a["std"] == 0.0
    assert a["ci_low"] == a["ci_high"] == pytest.approx(0.8)


def test_aggregate_normal_ci_matches_formula():
    vals = [0.7, 0.8, 0.9, 0.6, 0.85]
    agg = S.aggregate({"accuracy": vals}, ci_method="normal")["accuracy"]
    arr = np.asarray(vals)
    mean = arr.mean()
    std = arr.std(ddof=1)
    half = 1.959963984540054 * std / math.sqrt(len(vals))
    assert agg["mean"] == pytest.approx(mean)
    assert agg["std"] == pytest.approx(std)
    assert agg["ci_low"] == pytest.approx(mean - half)
    assert agg["ci_high"] == pytest.approx(mean + half)


def test_aggregate_bootstrap_is_deterministic_and_bracketed():
    vals = [0.5, 0.6, 0.7, 0.8, 0.9]
    a1 = S.aggregate({"m": vals}, ci_method="bootstrap", seed=42)["m"]
    a2 = S.aggregate({"m": vals}, ci_method="bootstrap", seed=42)["m"]
    assert a1 == a2  # same seed -> identical
    assert a1["ci_low"] <= a1["mean"] <= a1["ci_high"]


def test_aggregate_skips_empty_metric():
    agg = S.aggregate({"empty": [], "ok": [0.5, 0.6]})
    assert "empty" not in agg
    assert "ok" in agg


# ---------------------------------------------------------------------------
# aggregated_scalars
# ---------------------------------------------------------------------------

def test_aggregated_scalars_flattening():
    agg = S.aggregate({"accuracy": [0.7, 0.8, 0.9]}, ci_method="normal")
    flat = S.aggregated_scalars(agg)
    assert "eval/accuracy/mean" in flat
    assert "eval/accuracy/std" in flat
    assert "eval/accuracy/ci_low" in flat
    assert "eval/accuracy/ci_high" in flat
    assert flat["eval/accuracy/mean"] == pytest.approx(agg["accuracy"]["mean"])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_class_names():
    assert S._class_names(5) == ["W", "N1", "N2", "N3", "REM"]
    assert S._class_names(3) == ["C0", "C1", "C2"]


def test_z_for_95_percent():
    assert S._z_for(0.95) == pytest.approx(1.959963984540054, abs=1e-4)


# ---------------------------------------------------------------------------
# confusion-matrix figures (matplotlib Agg)
# ---------------------------------------------------------------------------

def test_figure_from_cm_returns_figure():
    import matplotlib
    cm = np.array([[0.9, 0.1], [0.2, 0.8]])
    fig = S.figure_from_cm(cm, class_names=["A", "B"])
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_confusion_matrix_figure_from_preds():
    import matplotlib
    preds, targets = _perfect_subject(n_classes=5, per_class=3)
    fig = S.confusion_matrix_figure(preds, targets, n_classes=5)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)
