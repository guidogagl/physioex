"""Alzheimer's disease dataset (UZ Leuven).

69 subjects: 37 AD + 32 HC (healthy controls), prefix-encoded (AD* vs HC*).
Each subject has one night recording with average-reference EEG montage,
differential EOG, chin EMG, and ECG -- all at 200 Hz, 30s epochs.

Annotations are TSV files with 7-8 comment header lines (starting with ``#``),
followed by tab-separated ``onset\\tend\\tstage`` rows.
Stage vocabulary: Wake, S1, S2, S3, REM.

The ``subset`` parameter allows selecting only AD or HC subjects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root


# AASM 5-class mapping from TSV stage strings.
_STAGE_MAP: Dict[str, int] = {
    "Wake": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "REM": 4,
}


class AlzheimersDataset(BasePhysioDataset):
    """Alzheimer's disease sleep dataset (UZ Leuven).

    Args:
        subset: ``"AD"`` for Alzheimer's patients only, ``"HC"`` for healthy
            controls only, or ``None`` for all subjects.
        root: path to the ``AlzheimerData`` directory.
    """

    DATASET_NAME = "alzheimers"
    DATASET_SUBDIR = "AlzheimerData"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    # Channel preference lists -- average reference montage.
    # Two hardware variants: 33 subjects have EMG1/EMG2, 36 have EMG Chin.
    CHANNEL_PREFERENCES = {
        "EEG": [
            "EEG C4-REF",
            "EEG C3-REF",
            "EEG F4-REF",
            "EEG F3-REF",
            "EEG O2-REF",
            "EEG O1-REF",
            "EEG Fz-REF",
            "EEG Pz-REF",
        ],
        "EOG": [
            ("EEG EOG1-REF", "EEG EOG2-REF"),
        ],
        "EMG": [
            "EMG Chin",  # 36 subjects (newer hardware)
            "EMG1",
            "EMG2",
            ("EMG1", "EMG2"),  # 33 subjects (older hardware)
        ],
        "ECG": [
            "ECG V1",
            "ECG",
        ],
    }

    def __init__(
        self,
        subset: Optional[str] = None,
        root: Optional[str] = None,
        **kwargs,
    ):
        if subset is not None and subset not in ("AD", "HC"):
            raise ValueError(f"subset must be 'AD', 'HC', or None; got {subset!r}")
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        # Store before super().__init__ which calls _list_subjects.
        self._subset_filter = subset
        if subset:
            self.DATASET_NAME = f"alzheimers_{subset.lower()}"
        super().__init__(root=root, **kwargs)

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
            subj_id = subj_dir.name

            # Filter by subset prefix.
            if self._subset_filter == "AD" and not subj_id.startswith("AD"):
                continue
            if self._subset_filter == "HC" and not subj_id.startswith("HC"):
                continue

            edf = subj_dir / f"{subj_id}_r1.edf"
            tsv = subj_dir / f"{subj_id}_r1_a1.tsv"
            if not edf.exists() or not tsv.exists():
                continue

            subjects.append(
                SubjectSpec(
                    subject_id=subj_id,
                    edf_path=edf,
                    label_path=tsv,
                    external_meta={
                        "group": "AD" if subj_id.startswith("AD") else "HC",
                    },
                )
            )
        return subjects

    # ------------------------------------------------------------------
    # Label reading
    # ------------------------------------------------------------------

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse Alzheimer's TSV annotation file.

        Format: comment lines starting with ``#`` (7-8 lines), an optional
        blank line, then tab-separated ``onset\\tend\\tstage`` rows.  Onset
        and end are in seconds; each row spans one 30s epoch.
        """
        stages: List[int] = []
        with open(spec.label_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("onset"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                stage_str = parts[2].strip()
                stages.append(_STAGE_MAP.get(stage_str, -1))
        return np.array(stages, dtype=np.int16)
