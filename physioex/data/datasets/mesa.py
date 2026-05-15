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
        "ECG": ["ECG", "EKG", "ECG1"],
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

    @staticmethod
    def _load_metadata(root: str) -> Dict[str, Dict[str, Any]]:
        """Load harmonized CSV into a dict keyed by nsrrid (as string)."""
        csv_path = Path(root) / "datasets" / "mesa-sleep-harmonized-dataset-0.8.0.csv"
        if not csv_path.exists():
            logger.warning("MESA metadata CSV not found: %s", csv_path)
            return {}
        import pandas as pd

        df = pd.read_csv(csv_path)
        meta: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(int(row["nsrrid"]))
            meta[key] = {
                col: (None if pd.isna(val) else val) for col, val in row.items()
            }
        return meta

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        specs = self._pair_edfs_with_xml(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
        )
        # Enrich with metadata from harmonized CSV.
        # subject_id is "mesa-sleep-0001" -> numeric id "1"
        for spec in specs:
            num_id = spec.subject_id.replace("mesa-sleep-", "").lstrip("0") or "0"
            if num_id in self._metadata:
                spec.external_meta.update(self._metadata[num_id])
        return specs
