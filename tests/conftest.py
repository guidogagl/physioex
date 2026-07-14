"""Shared pytest fixtures and marker gating for the PhysioEx test suite.

Markers (registered in pyproject.toml):
- ``unit`` / ``integration`` — categorical, no gating.
- ``real_data`` — needs datasets on disk; skipped unless PHYSIOEX_TEST_REAL_DATA=1.
- ``gpu``       — needs CUDA; skipped unless a GPU is available.
- ``hf``        — needs HuggingFace network/checkpoints; skipped unless PHYSIOEX_TEST_HF=1.
- ``slow``      — long-running; runs by default, deselect with ``-m "not slow"``.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Marker gating
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip gated tests unless their environment is available."""
    real_data = os.environ.get("PHYSIOEX_TEST_REAL_DATA", "") == "1"
    hf = os.environ.get("PHYSIOEX_TEST_HF", "") == "1"
    gpu = _cuda_available()

    skip_real = pytest.mark.skip(reason="needs real data (set PHYSIOEX_TEST_REAL_DATA=1)")
    skip_hf = pytest.mark.skip(reason="needs HuggingFace access (set PHYSIOEX_TEST_HF=1)")
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU")

    for item in items:
        if "real_data" in item.keywords and not real_data:
            item.add_marker(skip_real)
        if "hf" in item.keywords and not hf:
            item.add_marker(skip_hf)
        if "gpu" in item.keywords and not gpu:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    """A fresh temporary data root (for synthetic dataset trees)."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def cache_dir(tmp_path):
    """A fresh temporary ChannelCache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Synthetic-data factory fixtures (thin wrappers over tests.factories)
# ---------------------------------------------------------------------------

@pytest.fixture
def edf_factory():
    """Return the low-level EDF writers (path-based)."""
    from tests.factories.edf import write_fake_edf, write_fake_annotations_edf

    return {"edf": write_fake_edf, "annotations": write_fake_annotations_edf}


@pytest.fixture
def fake_edf_dataset(data_dir, cache_dir):
    """Factory: build a ready FakeEDFDataset over one synthetic subject.

    Usage: ``ds = fake_edf_dataset(stages=["W","N1","N2"], channels=["EEG"])``.
    """
    from tests.factories.edf import (
        FakeEDFDataset,
        write_fake_annotations_edf,
        write_fake_edf,
    )

    def _make(
        subject_id="sub01",
        stages=None,
        channels=None,
        n_channels=4,
        duration_sec=None,
        pipelines="raw",
        sequence_length=1,
        **kwargs,
    ):
        stages = stages or ["W", "N1", "N2", "N3", "R"]
        channels = channels or ["EEG", "EOG", "EMG"]
        if duration_sec is None:
            duration_sec = len(stages) * 30
        write_fake_edf(
            data_dir / f"{subject_id}.edf",
            n_channels=n_channels,
            duration_sec=duration_sec,
        )
        write_fake_annotations_edf(
            data_dir / f"{subject_id}_sleepscoring.edf", stages
        )
        return FakeEDFDataset(
            root=str(data_dir),
            subject_id=subject_id,
            channels=channels,
            pipelines=pipelines,
            sequence_length=sequence_length,
            cache_dir=str(cache_dir),
            **kwargs,
        )

    return _make
