"""HMC (Haaglanden Medical Centre) sleep dataset, PhysioNet.

Each subject has:
  - {subject_id}.edf   -> polysomnography signals
  - {subject_id}_sleepscoring.edf   -> EDF with embedded sleep stage annotations
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyedflib

from physioex.data.base import BasePhysioDataset, SubjectSpec


# Maps the stage strings from HMC sleepscoring EDF annotations to AASM 5-class.
# Values come from physioex/preprocess/hmc.py (stages_map) adapted.
HMC_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
    # Aliases sometimes seen
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "REM": 4,
    # Wake / unknown
    "Sleep stage ?": -1,
}


class HMCDataset(BasePhysioDataset):
    """HMC sleep dataset."""

    DATASET_NAME = "hmc"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    # Preference lists for generic modality requests. Reuses inherited defaults
    # and adds HMC-specific single-channel fallbacks.
    # Real HMC EDF channels: "EEG C4-M1", "EEG C3-M2", "EEG F4-M1",
    # "EEG O2-M1", "EOG E1-M2", "EOG E2-M2", "EMG chin", "ECG"
    CHANNEL_PREFERENCES = {
        "EEG": [
            "EEG C4-M1",
            "EEG C3-M2",
            "EEG F4-M1",
            "EEG O2-M1",
            ("C4", "M1"),
            ("C3", "M2"),
            ("C4", "A1"),
            ("C3", "A2"),
            "C4-M1",
            "C3-M2",
            "C4-M2",
            "C3-M1",
            "EEG(sec)",
            "EEG",
            "EEG1",
        ],
        "EOG": [
            "EOG E1-M2",
            "EOG E2-M2",
            ("E1", "M2"),
            ("E2", "M1"),
            ("EOG(L)", "EOG(R)"),
            "EOG",
        ],
        "EMG": [
            "EMG chin",
            "EMG Chin",
            ("LCHIN", "CCHIN"),
            ("EMG1", "EMG2"),
            "EMG",
        ],
        "ECG": ["ECG", "ECG1", "ECG2", "EKG"],
    }

    def __init__(
        self,
        root: str = "/home/dev/sleep-data/raw-sleep/hmc/physionet.org/files/hmc-sleep-staging/1.1/recordings",
        **kwargs,
    ):
        super().__init__(root=root, **kwargs)

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        if not root.exists():
            return []

        subjects: List[SubjectSpec] = []
        for psg_edf in sorted(root.rglob("*.edf")):
            # Skip annotation files themselves
            if psg_edf.stem.endswith("_sleepscoring"):
                continue
            score_edf = psg_edf.with_name(psg_edf.stem + "_sleepscoring.edf")
            if not score_edf.exists():
                continue
            subjects.append(
                SubjectSpec(
                    subject_id=psg_edf.stem,
                    edf_path=psg_edf,
                    label_path=score_edf,
                    external_meta={},
                )
            )
        return subjects

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse HMC's embedded sleepscoring annotations -> per-epoch labels.

        The sleepscoring EDF contains onsets (seconds), durations (seconds),
        and stage strings. Each epoch is 30 seconds.
        """
        with pyedflib.EdfReader(str(spec.label_path)) as f:
            # pyedflib returns 3-tuple: (onsets, durations, labels)
            onsets, durations, stages = f.readAnnotations()

        if len(stages) == 0:
            return np.array([], dtype=np.int16)

        # Compute total span and per-epoch label array
        total = max(float(o) + float(d) for o, d in zip(onsets, durations))
        n_epochs = int(total // self.epoch_length_sec)
        labels = np.full(n_epochs, -1, dtype=np.int16)
        for onset, duration, stage_str in zip(onsets, durations, stages):
            i0 = int(round(float(onset) / self.epoch_length_sec))
            i1 = int(round((float(onset) + float(duration)) / self.epoch_length_sec))
            if i1 > n_epochs:
                i1 = n_epochs
            stage = HMC_STAGE_MAP.get(str(stage_str).strip(), -1)
            labels[i0:i1] = stage
        return labels
