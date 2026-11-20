"""MrOS (Osteoporotic Fractures in Men Study) sleep dataset, NSRR format.

Layout on disk::

    <root>/
        polysomnography/
            edfs/
                visit1/
                    mros-visit1-aa0001.edf
                    ...
            annotations-events-nsrr/
                visit1/
                    mros-visit1-aa0001-nsrr.xml
                    ...
        datasets/
            mros-visit1-harmonized-0.6.0.csv

Subject ID = EDF stem (e.g. ``mros-visit1-aa0001``).
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


class MrOSDataset(_NSRRBaseDataset):
    """MrOS sleep dataset (visit 1)."""

    DATASET_NAME = "mros"
    DATASET_SUBDIR = "mros"
    SESSION_PREFIX = "mros-visit1-"
    DATASET_PREFIX = "mros-visit1-"
    HARMONIZED_CSV_GLOB = "mros-visit1-harmonized-*.csv"
    CSV_ID_COLUMN = "nsrrid"

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [
            ("C4", "A1"),
            ("C3", "A2"),
            ("C4", "M1"),
            ("C3", "M2"),
            "EEG C4-M1",
            "EEG C3-M2",
            "EEG2",
            "EEG3",
            "EEG",
        ],
        "EOG": ["LOC", "ROC", ("LOC", "ROC"), "EOG-L", "EOG-R", ("EOG-L", "EOG-R")],
        "EMG": [
            "EMG",
            ("L Chin", "R Chin"),
            "L Chin",
            "R Chin",
            ("LCHIN", "RCHIN"),
            "LCHIN",
            "RCHIN",
        ],
        "ECG": [
            "ECG L",
            "ECG R",
            ("ECG L", "ECG R"),
            "ECG",
            "EKG",
            "ECG1",
            "ECGL",
            "ECG L-ECG R",
            "ECG2",
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
        """Load harmonized CSV into a dict keyed by nsrrid (uppercase, e.g. 'AA0001')."""
        meta = cls.load_harmonized_csv(root)
        return {k.upper(): v for k, v in meta.items()}

    def _row_key_of(self, subject_id: str) -> Optional[str]:
        # subject_id is "mros-visit1-aa0001" -> nsrrid "AA0001"
        return self._nsrrid_of(subject_id).upper()

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        # MrOS EDFs and annotations are organised in visit subdirectories.
        # Try visit1 first (primary); fall back to flat layout for compat.
        edf_dir = root / "polysomnography" / "edfs" / "visit1"
        xml_dir = root / "polysomnography" / "annotations-events-nsrr" / "visit1"
        if not edf_dir.exists():
            # Fall back to flat layout (no visit subdirs)
            edf_dir = root / "polysomnography" / "edfs"
            xml_dir = root / "polysomnography" / "annotations-events-nsrr"
        specs = self._pair_edfs_with_xml(edf_dir, xml_dir)
        # Enrich with metadata from harmonized CSV.
        for spec in specs:
            row = self._subject_row(spec.subject_id)
            if row is not None:
                spec.external_meta.update(row)
        return specs
