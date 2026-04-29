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
from typing import Any, Dict, List

from physioex.data.base import SubjectSpec
from physioex.data.datasets._nsrr import _NSRRBaseDataset

logger = logging.getLogger("physioex.data")


class MrOSDataset(_NSRRBaseDataset):
    """MrOS sleep dataset (visit 1)."""

    DATASET_NAME = "mros"

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
        "ECG": ["ECG L", "ECG R", ("ECG L", "ECG R"), "ECG", "ECG1", "ECG2"],
    }

    def __init__(
        self,
        root: str = "/home/dev/sleep-data/raw-sleep/mros",
        **kwargs,
    ):
        self._metadata = self._load_metadata(root)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(root: str) -> Dict[str, Dict[str, Any]]:
        """Load harmonized CSV into a dict keyed by nsrrid (uppercase, e.g. 'AA0001')."""
        csv_path = Path(root) / "datasets" / "mros-visit1-harmonized-0.6.0.csv"
        if not csv_path.exists():
            logger.warning("MrOS metadata CSV not found: %s", csv_path)
            return {}
        import pandas as pd

        df = pd.read_csv(csv_path)
        meta: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(row["nsrrid"]).upper()
            meta[key] = {
                col: (None if pd.isna(val) else val) for col, val in row.items()
            }
        return meta

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
        # subject_id is "mros-visit1-aa0001" -> nsrrid "AA0001"
        for spec in specs:
            # Extract site+number portion after "mros-visit1-"
            parts = spec.subject_id.split("-", 2)
            if len(parts) >= 3:
                nsrrid = parts[2].upper()  # "aa0001" -> "AA0001"
                if nsrrid in self._metadata:
                    spec.external_meta.update(self._metadata[nsrrid])
        return specs
