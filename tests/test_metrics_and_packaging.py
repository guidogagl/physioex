"""Regression metrics + packaging-hygiene tests.

- regression metrics (mse/mae/r2, ignore_index masking, metric dicts);
- no hard-coded author absolute paths in shipped code / examples;
- pyproject dependency layout (core vs optional extras).
"""
import math
from pathlib import Path

import pytest
import torch

from physioex.train.metrics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    mae_score,
    mse_score,
    r2_score,
)

_REPO = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ImportError:  # py<3.11
    import tomli as tomllib


# --------------------------------------------------------------------------
# Regression metrics
# --------------------------------------------------------------------------

def test_mse_perfect():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mse_score(t.clone(), t, ignore_index=None) == 0.0


def test_mse_constant_offset():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(mse_score(t + 2.0, t, ignore_index=None), 4.0, rel_tol=1e-5)


def test_mae_constant_offset():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(mae_score(t + 2.0, t, ignore_index=None), 2.0, rel_tol=1e-5)


def test_r2_perfect():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(r2_score(t.clone(), t, ignore_index=None), 1.0, rel_tol=1e-5)


def test_ignore_index_masking():
    t = torch.tensor([1.0, -1.0, 3.0, -1.0, 5.0])
    o = torch.tensor([1.0, 999.0, 3.0, 999.0, 5.0])
    assert mse_score(o, t, ignore_index=-1) == 0.0
    assert mae_score(o, t, ignore_index=-1) == 0.0


def test_all_masked_returns_zero():
    t = torch.tensor([-1.0, -1.0, -1.0])
    o = torch.tensor([10.0, 20.0, 30.0])
    assert mse_score(o, t, ignore_index=-1) == 0.0
    assert mae_score(o, t, ignore_index=-1) == 0.0
    assert r2_score(o, t, ignore_index=-1) == 0.0


def test_metric_dicts_keys():
    assert set(REGRESSION_METRICS.keys()) == {"mse", "mae", "r2"}
    assert set(CLASSIFICATION_METRICS.keys()) == {
        "accuracy", "f1_score", "precision", "recall",
        "cohen_kappa", "confusion_matrix", "support",
    }


# --------------------------------------------------------------------------
# Packaging hygiene
# --------------------------------------------------------------------------

@pytest.mark.parametrize("subdir", ["examples", "physioex"])
def test_no_hardcoded_author_paths(subdir):
    """No machine-specific author absolute paths in shipped code / examples."""
    offenders = []
    for py in (_REPO / subdir).rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for needle in ("/home/dev/physioex", "/home/dev/sleep-data"):
            if needle in text:
                offenders.append(f"{py.relative_to(_REPO)}: {needle}")
    assert not offenders, "hard-coded author paths:\n" + "\n".join(offenders)


@pytest.fixture(scope="module")
def pyproject():
    with open(_REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _has(dep_list, pkg):
    return any(pkg == d or d.startswith(pkg) for d in dep_list)


@pytest.mark.parametrize("pkg", ["torch", "rich", "psutil", "torchinfo", "einops", "tqdm", "captum"])
def test_core_deps_present(pyproject, pkg):
    assert _has(pyproject["project"]["dependencies"], pkg)


@pytest.mark.parametrize("pkg", ["seaborn", "boto3", "botocore", "wfdb", "psg_utils"])
def test_removed_deps_absent_from_core(pyproject, pkg):
    """Deps dropped during the cleanup must not reappear in core dependencies."""
    assert not _has(pyproject["project"]["dependencies"], pkg)


@pytest.mark.parametrize(
    "extra,pkg",
    [
        ("gpu-monitor", "nvitop"),
        ("legacy", "torchmetrics"),
        ("legacy", "lightning"),
        ("legacy", "pytorch_lightning"),
        ("foundation", "transformers"),
        ("datasets", "mne"),
    ],
)
def test_optional_extras_layout(pyproject, extra, pkg):
    extras = pyproject["project"]["optional-dependencies"]
    assert extra in extras, f"missing extra [{extra}]"
    assert _has(extras[extra], pkg), f"{pkg} not in [{extra}]"
