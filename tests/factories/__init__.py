"""Synthetic-data factories for the PhysioEx test suite.

Reusable, side-effect-free generators of fake EDF recordings, annotation
sidecars and per-dataset subject trees, plus a trivial concrete dataset. These
are the building blocks behind the shared pytest fixtures in ``tests/conftest.py``.
"""
from tests.factories.edf import (
    FakeEDFDataset,
    write_fake_annotations_edf,
    write_fake_edf,
)

__all__ = ["write_fake_edf", "write_fake_annotations_edf", "FakeEDFDataset"]
