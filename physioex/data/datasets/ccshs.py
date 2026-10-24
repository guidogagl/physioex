"""CCSHS (Cleveland Children's Sleep and Health Study, NSRR ``ccshs-trec``).

Layout on disk::

    <root>/
        polysomnography/
            edfs/ccshs-trec-1800001.edf
            annotations-events-nsrr/ccshs-trec-1800001-nsrr.xml
        datasets/ccshs-trec-dataset-0.8.0.csv

One recording per subject; subject ID = EDF stem (``ccshs-trec-<nsrrid>``).
EEG C3/C4 are recorded against the contralateral mastoid (A2/A1) and are
re-referenced by the differential-pair resolution (``("C3", "A2")``).
EDF is mixed-rate (EEG/EOG 128 Hz, chin EMG 256 Hz).

Eleven recordings are excluded by default (``CCSHSDataset.EXCLUDED_SUBJECTS``): nine whose
per-stage minutes in the NSRR XML disagree with the stage minutes of the NSRR summary
dataset CSV, and two that have no row in that CSV. Pass ``exclude_subjects=[]`` to keep them.

Stage vocabulary in the NSRR XML (``EventType`` ``Stages|Stages``): ``Wake|0``,
``Stage 1..4 sleep|1..4``, ``REM sleep|5``, ``Movement|6``. Stage 3 and 4 are
merged to N3. Movement/unscored -> -1.  Any *other* stage concept raises, so
a vocabulary change cannot silently become "unscored".
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from physioex.data.base import SubjectSpec, get_data_root
from physioex.data.datasets._nsrr import _NSRRBaseDataset
from physioex.data.cleaning.montage import CleanMontage

logger = logging.getLogger("physioex.data")

CCSHS_STAGE_MAP: Dict[str, int] = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,  # S3 + S4 -> N3
    "REM sleep|5": 4,
    "Movement|6": -1,
    "Unscored|9": -1,
}


def parse_ccshs_xml(xml_path, epoch_length_sec: float = 30.0) -> Dict[str, Any]:
    """Parse a CCSHS NSRR XML.

    Returns dict with ``labels`` (int16, indexed by event ONSET, length
    ``floor(recording_duration / epoch)``), ``duration_sec`` (from the
    "Recording Start Time" event, else the end of the last stage event) and
    ``clock_time`` (string or None).
    """
    root = ET.parse(str(xml_path)).getroot()
    events = []
    rec_dur: Optional[float] = None
    clock: Optional[str] = None
    for ev in root.iter("ScoredEvent"):
        etype = (ev.findtext("EventType") or "").strip()
        concept = (ev.findtext("EventConcept") or "").strip()
        if concept == "Recording Start Time":
            rec_dur = float(ev.findtext("Duration"))
            clock = ev.findtext("ClockTime")
            continue
        if not etype.startswith("Stages"):
            continue
        if concept not in CCSHS_STAGE_MAP:
            raise ValueError(f"{xml_path}: unknown stage concept {concept!r}")
        events.append(
            (float(ev.findtext("Start")), float(ev.findtext("Duration")), concept)
        )
    if not events:
        return {
            "labels": np.array([], dtype=np.int16),
            "duration_sec": rec_dur,
            "clock_time": clock,
        }
    end = max(s + d for s, d, _ in events)
    total = rec_dur if rec_dur is not None else end
    n = int(total // epoch_length_sec)
    labels = np.full(n, -1, dtype=np.int16)
    for s, d, c in events:
        i0 = int(round(s / epoch_length_sec))
        i1 = min(int(round((s + d) / epoch_length_sec)), n)
        if i0 < n:
            labels[i0:i1] = CCSHS_STAGE_MAP[c]
    return {"labels": labels, "duration_sec": total, "clock_time": clock}


class CCSHSDataset(_NSRRBaseDataset):
    """Cleveland Children's Sleep and Health Study (~515 children/adolescents)."""

    DATASET_NAME = "ccshs"
    DATASET_SUBDIR = "ccshs"

    #: Recordings left out by default. Nine: stage minutes in the XML differ from the NSRR
    #: summary CSV (``ccshs-trec-dataset-*.csv``); two: no CSV row, so the hypnogram cannot be
    #: cross-checked. Subject ids are EDF stems.
    EXCLUDED_SUBJECTS = frozenset(
        f"ccshs-trec-{n}"
        for n in (
            1800064,
            1800224,
            1800306,
            1800384,
            1800432,
            1800458,
            1800620,
            1800640,
            1800873,  # CSV-discordant
            1800109,
            1800489,  # no CSV row
        )
    )

    #: Clean-channel recipe (physioex.data.cleaning). Raw electrodes: the derivations are built here.
    CLEAN_MONTAGE = CleanMontage(
        eeg=[("C4", "A1"), ("C3", "A2")],
        eog=[("LOC", "ROC")],
        chin=[("EMG1", "EMG2"), ("EMG1", "EMG3"), ("EMG2", "EMG3")],
        leg_l=[("LEFT LEG1", "LEFT LEG2")],
        leg_r=[("RIGHT LEG1", "RIGHT LEG2")],
        ecg=[("ECG1", "ECG2")],
        extra={"C3": [("C3", "A2")]},
        extra_eeg=["C3"],
        level="poor",
        mains=60.0,
    )

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [
            ("C3", "A2"),
            ("C4", "A1"),
            "C3",
            "C4",
        ],
        "EOG": [
            ("LOC", "ROC"),
            "LOC",
            "ROC",
        ],
        "EMG": [
            ("EMG1", "EMG2"),
            ("EMG1", "EMG3"),
            "EMG1",
        ],
    }

    def __init__(
        self,
        root: Optional[str] = None,
        exclude_subjects: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        """
        Args:
            root: dataset root (default ``<data_root>/ccshs``).
            exclude_subjects: subject ids (``ccshs-trec-<nsrrid>``) to leave out. ``None`` uses
                :attr:`EXCLUDED_SUBJECTS`; pass ``[]`` to keep every recording.
        """
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        self._exclude = frozenset(
            self.EXCLUDED_SUBJECTS if exclude_subjects is None else exclude_subjects
        )
        self._metadata = self._load_metadata(root)
        super().__init__(root=root, **kwargs)

    @staticmethod
    def _load_metadata(root: str) -> Dict[str, Dict[str, Any]]:
        # version-independent: ccshs-trec-dataset-<version>.csv (newest version wins)
        found = sorted((Path(root) / "datasets").glob("ccshs-trec-dataset-*.csv"))
        if not found:
            logger.warning(
                "CCSHS metadata CSV not found under %s", Path(root) / "datasets"
            )
            return {}
        csv_path = found[-1]
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)
        meta: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(int(row["nsrrid"]))
            meta[key] = {c: (None if pd.isna(v) else v) for c, v in row.items()}
        return meta

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        specs = self._pair_edfs_with_xml(
            root / "polysomnography" / "edfs",
            root / "polysomnography" / "annotations-events-nsrr",
        )
        if self._exclude:
            kept = [s for s in specs if s.subject_id not in self._exclude]
            if len(kept) != len(specs):
                logger.info(
                    "CCSHS: %d recordings excluded by default list",
                    len(specs) - len(kept),
                )
            specs = kept
        for spec in specs:
            nid = spec.subject_id.replace("ccshs-trec-", "")
            if nid in self._metadata:
                spec.external_meta.update(self._metadata[nid])
        return specs

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        return parse_ccshs_xml(spec.label_path, self.epoch_length_sec)["labels"]
