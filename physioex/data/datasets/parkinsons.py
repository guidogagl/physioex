"""Parkinson's disease dataset (UZ Leuven).

87 subjects (40 HOA + 48 PD), with up to 157 recordings: 86 night + 71 nap.
Night recordings have 35-36 channels @ 500 Hz; nap recordings have 15
channels @ 500 Hz.  Bipolar EEG montage (``EEG C3-A2``, ``EEG C4-A1``),
single EOG channels (``EOG Left``, ``EOG right``), chin EMG, and ECG.

Annotations are TSV files with NO header -- clean tab-separated
``onset\\tend\\tstage`` rows.  Stage vocabulary: Wake, S1, S2, S3, REM,
Unscorable, LIGHTS_OFF.

Two subsetting dimensions:
  - ``recording``: ``"night"``, ``"nap"``, or ``"all"``
  - ``group``: ``"HOA"``, ``"PD"``, or ``None`` (all)

Demographics are loaded from ``Target_sleep_demographic.csv`` in the dataset
root for group filtering and full subject metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root

logger = logging.getLogger("physioex.data")


# AASM 5-class mapping from TSV stage strings.
_STAGE_MAP: Dict[str, int] = {
    "Wake": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "REM": 4,
    "Unscorable": -1,
    "LIGHTS_OFF": -1,
}


class ParkinsonsDataset(BasePhysioDataset):
    """Parkinson's disease sleep dataset (UZ Leuven).

    Args:
        recording: ``"night"`` for overnight PSG, ``"nap"`` for daytime nap,
            or ``"all"`` for both.
        group: ``"HOA"`` for healthy older adults, ``"PD"`` for Parkinson's
            disease, or ``None`` for all subjects.
        root: path to the ``Parkinson_data`` directory.
    """

    DATASET_NAME = "parkinsons"
    DATASET_SUBDIR = "Parkinson_data"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    # Channel preference lists -- bipolar montage.
    # Two hardware variants: ~46 subjects have "EOG Left"/"EOG right"/"EMG Chin",
    # ~40 subjects have "EOG1"/"EOG2"/"EMG1".
    CHANNEL_PREFERENCES = {
        "EEG": [
            "EEG C3-A2",
            "EEG C4-A1",
            "EEG C3-REF",
            "EEG C4-REF",
            ("C3", "A2"),
            ("C4", "A1"),
        ],
        "EOG": [
            "EOG Left",
            "EOG right",
            ("EOG Left", "EOG right"),
            ("EOG1", "EOG2"),  # numbered variant (~40 subjects)
            "EOG1",
            "EOG2",
            "EOG LOC-A2",
            "EOG ROC-A1",
            ("EOG LOC", "EOG ROC"),
        ],
        "EMG": [
            "EMG Chin",
            "EMG Chin1",
            "EMG chin",
            "EMG1",  # numbered variant (~40 subjects)
        ],
        "ECG": [
            "ECG V1",
            "ECG",
            "ECG1",
        ],
    }

    def __init__(
        self,
        recording: str = "night",
        group: Optional[str] = None,
        root: Optional[str] = None,
        **kwargs,
    ):
        if recording not in ("night", "nap", "all"):
            raise ValueError(
                f"recording must be 'night', 'nap', or 'all'; got {recording!r}"
            )
        if group is not None and group not in ("HOA", "PD"):
            raise ValueError(f"group must be 'HOA', 'PD', or None; got {group!r}")
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        self.recording = recording
        self.group = group

        # Dynamic dataset name for cache separation.
        name = f"parkinsons_{recording}"
        if group:
            name += f"_{group.lower()}"
        self.DATASET_NAME = name

        # Load full demographics CSV for group filtering and metadata.
        self._demographics, self._full_metadata = self._load_demographics(root)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Demographics & metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _load_demographics(root: str):
        """Load demographics CSV for group filtering and full metadata.

        Returns:
            demographics: dict ``{"Sub_0001": "HOA", ...}`` for group filtering.
            full_metadata: dict ``{"Sub_0001": {all 62 columns}, ...}``
        """
        csv_path = Path(root) / "Target_sleep_demographic.csv"
        if not csv_path.exists():
            logger.warning("Parkinsons metadata CSV not found: %s", csv_path)
            return {}, {}
        import pandas as pd

        df = pd.read_csv(csv_path)
        demographics: Dict[str, str] = dict(
            zip(df.iloc[:, 0].astype(str), df["group"].astype(str))
        )
        full_metadata: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(row["record_id"])
            full_metadata[key] = {
                col: (None if pd.isna(val) else val) for col, val in row.items()
            }
        return demographics, full_metadata

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        data_dir = Path(self.root) / "Data"
        if not data_dir.exists():
            return []

        subjects: List[SubjectSpec] = []
        for subj_dir in sorted(data_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            dir_name = subj_dir.name  # "Sub_0001" or "Sub_0001_nap"
            is_nap = dir_name.endswith("_nap")
            base_id = dir_name.replace("_nap", "") if is_nap else dir_name

            # Filter by recording type.
            if self.recording == "night" and is_nap:
                continue
            if self.recording == "nap" and not is_nap:
                continue

            # Filter by group (from demographics CSV).
            if self.group is not None:
                subj_group = self._demographics.get(base_id)
                if subj_group != self.group:
                    continue

            # Locate EDF + TSV.
            edf = subj_dir / f"{dir_name}_r1.edf"
            tsv = subj_dir / f"{dir_name}_r1_a1.tsv"
            if not edf.exists() or not tsv.exists():
                continue

            group = self._demographics.get(base_id, "unknown")
            meta: Dict[str, Any] = {
                "group": group,
                "recording": "nap" if is_nap else "night",
                "base_subject_id": base_id,
            }
            # Add all demographic/clinical columns from the CSV.
            if base_id in self._full_metadata:
                meta.update(self._full_metadata[base_id])
            subjects.append(
                SubjectSpec(
                    subject_id=dir_name,
                    edf_path=edf,
                    label_path=tsv,
                    external_meta=meta,
                )
            )
        return subjects

    # ------------------------------------------------------------------
    # Coherent splits (night + nap share the same subject-level split)
    # ------------------------------------------------------------------

    def get_splits(self, fold: int = 0):
        """Override to ensure coherent splits across night and nap recordings.

        The split is done on the GLOBAL set of base subject IDs (all folders
        in ``Data/``, both night and nap), so the random shuffle always
        operates on the same list regardless of the ``recording`` filter.
        This guarantees that if ``Sub_0001`` is in the test set, both
        ``Sub_0001`` (night) and ``Sub_0001_nap`` (nap) end up in test —
        preventing data leakage across recording types.
        """
        import random

        # Always enumerate ALL base subject IDs from the filesystem, regardless
        # of which recording type or group this instance is configured for.
        data_dir = Path(self.root) / "Data"
        all_base_ids = sorted(
            set(
                d.name.replace("_nap", "")
                for d in data_dir.iterdir()
                if d.is_dir() and d.name.startswith("Sub_")
            )
        )

        rng = random.Random(42 + int(fold))
        rng.shuffle(all_base_ids)

        n = len(all_base_ids)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)
        train_base = set(all_base_ids[:n_train])
        valid_base = set(all_base_ids[n_train : n_train + n_valid])
        test_base = set(all_base_ids[n_train + n_valid :])

        # Map back to actual subject_ids in this dataset instance
        train_ids = [
            s.subject_id
            for s in self._subjects
            if s.external_meta.get("base_subject_id", s.subject_id) in train_base
        ]
        valid_ids = [
            s.subject_id
            for s in self._subjects
            if s.external_meta.get("base_subject_id", s.subject_id) in valid_base
        ]
        test_ids = [
            s.subject_id
            for s in self._subjects
            if s.external_meta.get("base_subject_id", s.subject_id) in test_base
        ]

        return train_ids, valid_ids, test_ids

    # ------------------------------------------------------------------
    # Label reading
    # ------------------------------------------------------------------

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse Parkinson's TSV annotation file.

        Format: NO header lines -- clean tab-separated ``onset\\tend\\tstage``
        rows. Onset and end are in seconds; each row spans one 30s epoch.
        ``Unscorable`` and ``LIGHTS_OFF`` map to -1 (unscored).
        """
        stages: List[int] = []
        with open(spec.label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                stage_str = parts[2].strip()
                stages.append(_STAGE_MAP.get(stage_str, -1))
        return np.array(stages, dtype=np.int16)
