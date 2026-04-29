"""SHHS (Sleep Heart Health Study) sleep dataset, NSRR format.

Contains two visits selectable via the ``visit`` parameter:
  - visit 1 (SHHS-1): ~5793 subjects
  - visit 2 (SHHS-2): ~2651 subjects

Layout on disk::

    <root>/
        polysomnography/
            edfs/
                shhs1/
                    shhs1-200001.edf
                    ...
                shhs2/
                    shhs2-200077.edf
                    ...
            annotations-events-nsrr/
                shhs1/
                    shhs1-200001-nsrr.xml
                    ...
                shhs2/
                    shhs2-200077-nsrr.xml
                    ...
        datasets/
            shhs-harmonized-dataset-0.21.0.csv

Subject ID = EDF stem (e.g. ``shhs1-200001``).
Annotations are NSRR XML parsed by ``parse_nsrr_xml``.
Subject metadata is loaded from the NSRR harmonized CSV keyed by
``(nsrrid, visitnumber)``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from physioex.data.base import SubjectSpec
from physioex.data.datasets._nsrr import _NSRRBaseDataset

logger = logging.getLogger("physioex.data")

# Map visit parameter to subdirectory name
_VISIT_DIRS = {
    1: "shhs1",
    2: "shhs2",
}


class SHHSDataset(_NSRRBaseDataset):
    """SHHS sleep dataset (visit 1 or visit 2).

    Parameters
    ----------
    root : str
        Path to the top-level SHHS directory (contains ``polysomnography/``
        and ``datasets/``).
    visit : int
        Which visit to load: ``1`` for SHHS-1 (~5793 subjects) or ``2``
        for SHHS-2 (~2651 subjects).  Default is ``1``.
    **kwargs
        Forwarded to :class:`_NSRRBaseDataset` / :class:`BasePhysioDataset`.
    """

    DATASET_NAME = "shhs_visit1"  # overridden dynamically in __init__

    # Channels present across both visits.
    #   EEG:  "EEG" = C4/A1 primary, "EEG(sec)" = C3/A2 secondary
    #   EOG:  "EOG(L)" and "EOG(R)"
    #   EMG:  "EMG" (chin)
    #   ECG:  "ECG"
    # Sample rates vary: SHHS-1 mostly 125 Hz, SHHS-2 mostly 125/128 Hz.
    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [
            "EEG",
            "EEG(sec)",
        ],
        "EOG": [
            "EOG(L)",
            "EOG(R)",
            ("EOG(L)", "EOG(R)"),
        ],
        "EMG": [
            "EMG",
        ],
        "ECG": [
            "ECG",
        ],
    }

    def __init__(
        self,
        root: str = "/home/dev/sleep-data/raw-sleep/shhs",
        visit: int = 1,
        **kwargs,
    ):
        if visit not in _VISIT_DIRS:
            raise ValueError(
                f"Unknown visit {visit!r}. Available: {sorted(_VISIT_DIRS)}"
            )
        self._visit = visit
        self.DATASET_NAME = f"shhs_visit{visit}"
        self._metadata = self._load_metadata(root, visit)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(root: str, visit: int) -> Dict[str, Dict[str, Any]]:
        """Load harmonized CSV into a dict keyed by nsrrid (as string).

        The harmonized CSV contains rows for both visits.  We filter to
        the requested ``visitnumber`` so that metadata lookups are
        unambiguous.
        """
        csv_path = Path(root) / "datasets" / "shhs-harmonized-dataset-0.21.0.csv"
        if not csv_path.exists():
            logger.warning("SHHS metadata CSV not found: %s", csv_path)
            return {}
        import pandas as pd

        df = pd.read_csv(csv_path)
        df = df[df["visitnumber"] == visit]
        meta: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(int(row["nsrrid"]))
            meta[key] = {
                col: (None if pd.isna(val) else val) for col, val in row.items()
            }
        return meta

    # ------------------------------------------------------------------
    # Fixed benchmark splits (Phan / SleepTransformer)
    # ------------------------------------------------------------------

    def get_splits(self, fold: int = 0):
        """Return fixed train/valid/test split from Phan's SleepTransformer.

        For visit 1, fold 0 uses the benchmark split from
        ``_splits/shhs.json`` (train=4065, valid=100, test=1639).
        For other visits or folds, falls back to random 70/15/15.
        """
        if self._visit == 1 and fold == 0:
            split_path = Path(__file__).parent / "_splits" / "shhs.json"
            if split_path.exists():
                import json

                with open(split_path) as f:
                    splits = json.load(f)
                fold_key = f"fold_{fold}"
                if fold_key in splits:
                    s = splits[fold_key]
                    # Filter to subjects actually present in this dataset instance
                    present = {spec.subject_id for spec in self._subjects}
                    train = [sid for sid in s["train"] if sid in present]
                    valid = [sid for sid in s["valid"] if sid in present]
                    test = [sid for sid in s["test"] if sid in present]
                    return train, valid, test
        # Fallback to default random split
        return super().get_splits(fold=fold)

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        visit_dir = _VISIT_DIRS[self._visit]

        specs = self._pair_edfs_with_xml(
            root / "polysomnography" / "edfs" / visit_dir,
            root / "polysomnography" / "annotations-events-nsrr" / visit_dir,
        )

        # Enrich with metadata from harmonized CSV.
        # subject_id is e.g. "shhs1-200001" -> nsrrid "200001"
        for spec in specs:
            parts = spec.subject_id.split("-", 1)
            if len(parts) == 2:
                nsrrid = parts[1]
                if nsrrid in self._metadata:
                    spec.external_meta.update(self._metadata[nsrrid])

        return specs
