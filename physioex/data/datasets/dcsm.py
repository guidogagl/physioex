"""DCSM (Danish Center for Sleep Medicine) dataset.

Expected post-extraction layout::

    <root>/
        <subject_uuid>/
            psg.edf
            hypnogram.ids
        <subject_uuid>/
            ...

The ``dcsm_dataset.zip`` archive extracts the subject directories under
``data/sleep/DCSM/``, so the default root points at that nested path
(``<parent>/dcsm/extracted/data/sleep/DCSM``). Users with a custom layout
can override the ``root`` argument.

Each ``hypnogram.ids`` is a whitespace-separated file with three columns
(no header): ``start_sec  duration_sec  stage_name``.

Channel layout (256 Hz native): C3-M2, C4-M1, E1-M2, E2-M2, CHIN, ECG-II.

Stage names: W, N1, N2, N3, REM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root


DCSM_STAGE_MAP: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    # Aliases sometimes seen
    "Wake": 0,
    "R": 4,
}


class DCSMDataset(BasePhysioDataset):
    """Danish Center for Sleep Medicine dataset."""

    DATASET_NAME = "dcsm"
    DATASET_SUBDIR = "DCSM"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": ["C3-M2", "C4-M1", ("C3", "M2"), ("C4", "M1"), "EEG"],
        "EOG": [("E1", "M2"), ("E2", "M2"), "E1-M2", "E2-M2", "EOG"],
        "EMG": ["CHIN", "EMG Chin", "EMG"],
        "ECG": ["ECG-II", "ECG", "EKG"],
    }

    def __init__(
        self,
        root: Optional[str] = None,
        **kwargs,
    ):
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        if not root.exists():
            return []

        subjects: List[SubjectSpec] = []
        for subj_dir in sorted(root.iterdir()):
            if not subj_dir.is_dir():
                continue
            psg = subj_dir / "psg.edf"
            hyp = subj_dir / "hypnogram.ids"
            if not psg.exists() or not hyp.exists():
                continue
            subjects.append(
                SubjectSpec(
                    subject_id=subj_dir.name,
                    edf_path=psg,
                    label_path=hyp,
                )
            )
        return subjects

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse ``hypnogram.ids``: three columns (``start_sec``, ``duration_sec``,
        ``stage_name``) separated by commas or whitespace (no header).
        """
        import pandas as pd

        # Real DCSM files are comma-separated; synthetic tests sometimes use
        # whitespace. Accept both by splitting on commas or whitespace runs.
        df = pd.read_csv(spec.label_path, header=None, sep=r"[,\s]+", engine="python")
        if df.shape[1] < 3:
            return np.array([], dtype=np.int16)

        starts = df.iloc[:, 0].astype(int).values
        durations = df.iloc[:, 1].astype(int).values
        stage_names = df.iloc[:, 2].astype(str).values

        if len(starts) == 0:
            return np.array([], dtype=np.int16)

        total_sec = int((starts + durations).max())
        n_epochs = int(total_sec // self.epoch_length_sec)
        labels = np.full(n_epochs, -1, dtype=np.int16)

        for start_s, dur_s, stage_str in zip(starts, durations, stage_names):
            stage = DCSM_STAGE_MAP.get(stage_str.strip(), -1)
            i0 = int(round(float(start_s) / self.epoch_length_sec))
            i1 = int(round((float(start_s) + float(dur_s)) / self.epoch_length_sec))
            if i1 > n_epochs:
                i1 = n_epochs
            labels[i0:i1] = stage

        return labels
