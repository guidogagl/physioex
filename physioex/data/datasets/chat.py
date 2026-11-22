"""CHAT (Childhood Adenotonsillectomy Trial) sleep dataset, NSRR format.

CHAT has three session groups: baseline, followup and the nonrandomized
baseline subset, all with one recording per subject::

    <root>/
        polysomnography/
            edfs/
                chat-baseline-300002.edf
                chat-followup-300002.edf
                chat-baseline-nonrandomized-300006.edf
                ...
            annotations-events-nsrr/
                chat-baseline-300002-nsrr.xml
                ...
        datasets/
            chat-harmonized-dataset-0.2.0.csv

Session id = EDF stem (e.g. ``chat-baseline-300002``).  Annotations are NSRR
XML parsed by ``parse_nsrr_xml``.  Subject metadata is loaded from the NSRR
harmonized CSV keyed by ``nsrrid``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from physioex.data.base import get_data_root
from physioex.data.datasets._nsrr import _NSRRBaseDataset

logger = logging.getLogger("physioex.data")


class CHATDataset(_NSRRBaseDataset):
    """CHAT sleep dataset (baseline + followup + nonrandomized baseline)."""

    DATASET_NAME = "chat"
    DATASET_SUBDIR = "chat"
    #: Session stems as distributed by NSRR: chat-baseline-*, chat-followup-*
    #: and chat-baseline-nonrandomized-*. The session id equals the
    #: subject_id; ``_nsrrid_of`` recovers the nsrrid for the CSV lookup by
    #: stripping the longest matching stem.
    DATASET_PREFIX = "chat-"
    #: Longer stem variants, in order of priority for ``_nsrrid_of``.
    SESSION_STEM_PRIORITY = (
        "chat-baseline-nonrandomized-",
        "chat-baseline-",
        "chat-followup-",
    )
    HARMONIZED_CSV_GLOB = "chat-harmonized-dataset-*.csv"
    CSV_ID_COLUMN = "nsrrid"

    #: Alternate channel names as distributed by NSRR (wav2sleep order):
    #: ECG, PPG (Masimo plethysmograph) and the ABD/THX respiratory-effort
    #: belts. ``ABD`` and ``THX`` are usable as channel requests.
    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": ["EEG", "EEG1", "EEG2"],
        "EOG": [
            "EOG",
            "EOG(L)",
            "EOG(R)",
            "LOC",
            "ROC",
        ],
        "EMG": [
            "EMG",
            "EMG1",
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
        """Load the harmonized CSV into a dict keyed by nsrrid (as string)."""
        meta = cls.load_harmonized_csv(root)
        return {str(int(k)): v for k, v in meta.items()}

    def _row_key_of(self, subject_id: str) -> Optional[str]:
        return str(int(self._nsrrid_of(subject_id)))

    def _nsrrid_of(self, subject_id: str) -> str:
        """Extract the NSRR id from a subject_id by stripping the longest
        matching session stem (``chat-baseline-nonrandomized-`` first)."""
        for prefix in self.SESSION_STEM_PRIORITY:
            if subject_id.startswith(prefix):
                return subject_id[len(prefix) :]
        return super()._nsrrid_of(subject_id)

    def _list_subjects(self):
        root = Path(self.root)
        specs = self._pair_edfs_with_xml(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
        )
        # Enrich with metadata from the harmonized CSV.
        # subject_id is e.g. "chat-baseline-300002" -> nsrrid "300002"
        for spec in specs:
            row = self._subject_row(spec.subject_id)
            if row is not None:
                spec.external_meta.update(row)
        return specs
