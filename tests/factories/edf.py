"""Synthetic EDF/annotation generators and a trivial dataset subclass.

Canonical home for the fake-data primitives historically defined in
``tests/test_raw_dataset_integration.py`` (kept re-exported there for
backward compatibility). Used by conftest fixtures and by the dataset tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib

from physioex.data.base import BasePhysioDataset, SubjectSpec


def write_fake_edf(
    path: Path,
    n_channels: int = 4,
    duration_sec: float = 300.0,
    fs_per_channel=None,
    channel_names=None,
    seed: int = 0,
) -> None:
    """Create a tiny fake EDF+ file at ``path`` with random signals.

    Each channel is Gaussian noise (20 uV) plus a 10 uV 1 Hz sine (so
    frequency-selective preprocessing is testable). ``duration_sec`` should be an
    integer number of 30 s epochs for clean label alignment; 1 record per second.
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
    n_records = duration_sec
    signals = []
    headers = []
    for i in range(n_channels):
        fs = fs_per_channel[i]
        t = np.arange(n_records * fs) / fs
        x = (
            rng.standard_normal(n_records * fs).astype(np.float64) * 20.0
            + 10.0 * np.sin(2 * np.pi * 1.0 * t)
        )
        signals.append(x)
        headers.append(
            {
                "label": channel_names[i],
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": -300.0,
                "physical_max": 300.0,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        )

    writer = pyedflib.EdfWriter(str(path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(headers)
        writer.setPatientCode("SUB01")
        writer.setPatientName("TestPatient")
        writer.writeSamples(signals)
    finally:
        writer.close()


def write_fake_annotations_edf(path: Path, stages: list, epoch_sec: float = 30.0) -> None:
    """Write an annotation-only EDF+ sidecar (one annotation per stage)."""
    total_duration = int(len(stages) * epoch_sec)
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(
            [
                {
                    "label": "dummy",
                    "dimension": "uV",
                    "sample_frequency": 1,
                    "physical_min": -1.0,
                    "physical_max": 1.0,
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
            ]
        )
        writer.writeSamples([np.zeros(total_duration, dtype=np.float64)])
        for i, s in enumerate(stages):
            writer.writeAnnotation(i * epoch_sec, epoch_sec, s)
    finally:
        writer.close()


class FakeEDFDataset(BasePhysioDataset):
    """Minimal concrete ``BasePhysioDataset`` over a single fake subject."""

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
        return [
            SubjectSpec(
                subject_id=self._fake_subject_id,
                edf_path=root / f"{self._fake_subject_id}.edf",
                label_path=root / f"{self._fake_subject_id}_sleepscoring.edf",
            )
        ]

    def _read_subject_labels(self, spec):
        with pyedflib.EdfReader(str(spec.label_path)) as f:
            onsets, durations, stages = f.readAnnotations()
        if len(stages) == 0:
            return np.array([], dtype=np.int16)
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


__all__ = ["write_fake_edf", "write_fake_annotations_edf", "FakeEDFDataset"]
