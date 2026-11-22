"""MESA (Multi-Ethnic Study of Atherosclerosis) sleep dataset, NSRR format.

Layout on disk::

    <root>/
        polysomnography/
            edfs/
                mesa-sleep-0001.edf
                ...
            annotations-events-nsrr/
                mesa-sleep-0001-nsrr.xml
                ...
        datasets/
            mesa-sleep-harmonized-dataset-0.8.0.csv

Subject ID = EDF stem (e.g. ``mesa-sleep-0001``).
Annotations are NSRR XML parsed by ``parse_nsrr_xml``.
Subject metadata is loaded from the NSRR harmonized CSV keyed by ``nsrrid``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from physioex.data.base import SubjectSpec, get_data_root
from physioex.data.datasets._nsrr import _NSRRBaseDataset

logger = logging.getLogger("physioex.data")


class MESADataset(_NSRRBaseDataset):
    """MESA sleep dataset (~2000 subjects)."""

    DATASET_NAME = "mesa"
    DATASET_SUBDIR = "mesa"
    DATASET_PREFIX = "mesa-sleep-"
    HARMONIZED_CSV_GLOB = "mesa-sleep-harmonized-dataset-*.csv"
    CSV_ID_COLUMN = "nsrrid"

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [
            "EEG1",
            "EEG2",
            "EEG3",
            "EEG C4-M1",
            "EEG C3-M2",
            "EEG Cz-Oz",
            ("C4", "M1"),
            ("C3", "M2"),
            "C4-M1",
            "C3-M2",
            "EEG",
        ],
        "EOG": [
            "EOG-L",
            "EOG-R",
            "EOG(L)",
            "EOG(R)",
            ("EOG-L", "EOG-R"),
            "EOG",
        ],
        "EMG": [
            "EMG",
            ("LCHIN", "RCHIN"),
        ],
        "ECG": [
            "ECG",
            "EKG",
            "ECG1",
            "ECG L",
            "ECGL",
            "ECG L-ECG R",
        ],
        "PPG": [
            "Pleth",
            "PlethWV",
            "PWF",
            "PlethMasimo",
            "PletMasimo",
            "PlethMasino",
            "PLETHMASIMO",
        ],
        "ABD": [
            "ABD",
            "Abdo",
            "ABDO RES",
            "ABDO EFFORT",
            "Abdominal",
            "abdomen",
        ],
        "THX": [
            "THX",
            "Thor",
            "THOR RES",
            "THOR EFFORT",
            "Thoracic",
            "Chest",
            "thorax",
            "CHEST",
        ],
    }

    def __init__(
        self,
        root: Optional[str] = None,
        **kwargs,
    ):
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        self._metadata = self._load_metadata(root)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @classmethod
    def _load_metadata(cls, root: str) -> Dict[str, Dict[str, Any]]:
        """Load harmonized CSV into a dict keyed by nsrrid (as string)."""
        return cls.load_harmonized_csv(root)

    def _row_key_of(self, subject_id: str) -> Optional[str]:
        # subject_id is "mesa-sleep-0001" -> nsrrid "1" (leading zeros stripped)
        return str(int(self._nsrrid_of(subject_id)))

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        specs = self._pair_edfs_with_xml(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
        )
        # Enrich with metadata from harmonized CSV.
        for spec in specs:
            row = self._subject_row(spec.subject_id)
            if row is not None:
                spec.external_meta.update(row)
        return specs
