"""SleepEDF (PhysioNet Sleep Cassette) dataset.

Each subject-night has:
  - SC4XXXYZ-PSG.edf       -> polysomnography signals
  - SC4XXXYZ-Hypnogram.edf -> EDF with embedded sleep stage annotations

The first 7 characters of the filename (e.g. ``SC4001E``) form the pairing
key between PSG and hypnogram files.  The full PSG stem minus ``-PSG``
(e.g. ``SC4001E0``) is used as the unique subject_id (one entry per night).

Channel layout (100 Hz unless noted):
  EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental (1 Hz),
  Resp oro-nasal (1 Hz), Temp rectal (1 Hz), Event marker (1 Hz).

Annotation stage strings:
  Sleep stage W, Sleep stage 1/2/3/4, Sleep stage R,
  Sleep stage ?, Movement time.

Subject metadata (age and sex) is extracted from the EDF local patient
identification header field (format: ``X F X Female_33yr``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyedflib

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root

logger = logging.getLogger("physioex.data")


SLEEPEDF_STAGE_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # AASM: N4 -> N3
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}


class SleepEDFDataset(BasePhysioDataset):
    """PhysioNet Sleep-EDF Cassette dataset (~153 recordings, ~78 subjects)."""

    DATASET_NAME = "sleepedf"
    DATASET_SUBDIR = "physionet-sleep-data"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": ["EEG Fpz-Cz", "EEG Pz-Oz", "EEG"],
        "EOG": ["EOG horizontal", "EOG"],
        "EMG": ["EMG submental", "EMG"],
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
    # Metadata from EDF header
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_edf_patient_field(edf_path: Path) -> Dict[str, Any]:
        """Extract age and sex from the EDF local patient identification field.

        The SleepEDF format stores patient info in bytes 8-88 of the EDF
        header as space-separated subfields.  The relevant subfield has the
        form ``Female_33yr`` or ``Male_26yr``.

        Returns a dict with ``age`` (int or None) and ``sex`` ('M'/'F'/None).
        """
        meta: Dict[str, Any] = {}
        try:
            with open(edf_path, "rb") as f:
                raw_header = f.read(88)
            patient_field = raw_header[8:88].decode("ascii", errors="replace").strip()
            # Parse subfields separated by spaces
            for part in patient_field.split():
                # Match patterns like "Female_33yr" or "Male_26yr"
                m = re.match(r"(Female|Male)_(\d+)yr", part, re.IGNORECASE)
                if m:
                    sex_str = m.group(1).capitalize()
                    meta["sex"] = "F" if sex_str == "Female" else "M"
                    meta["age"] = int(m.group(2))
                    break
                # Also handle "F" or "M" standalone (EDF+ standard)
                if part in ("F", "M"):
                    meta["sex"] = part
        except Exception:
            pass
        return meta

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        if not root.exists():
            return []

        # Build a lookup: 7-char prefix -> hypnogram path
        hyp_by_prefix: Dict[str, Path] = {}
        for hyp in sorted(root.glob("*-Hypnogram.edf")):
            prefix7 = hyp.stem[:7]  # e.g. "SC4001E"
            hyp_by_prefix[prefix7] = hyp

        subjects: List[SubjectSpec] = []
        for psg in sorted(root.glob("*-PSG.edf")):
            prefix7 = psg.stem[:7]  # e.g. "SC4001E"
            hyp = hyp_by_prefix.get(prefix7)
            if hyp is None:
                continue
            # subject_id = PSG stem without "-PSG", e.g. "SC4001E0"
            subject_id = psg.stem.replace("-PSG", "")
            meta = self._parse_edf_patient_field(psg)
            subjects.append(
                SubjectSpec(
                    subject_id=subject_id,
                    edf_path=psg,
                    label_path=hyp,
                    external_meta=meta,
                )
            )
        return subjects

    def _subject_group_key(self, subject_id: str) -> str:
        """Return the real-subject key shared by all nights of the same person.

        Sleep-EDF naming: ``SC4XXYZE`` where ``XX`` is the subject pair,
        ``Y`` the night index (1 or 2), ``Z`` the subset, ``E`` the session.
        Recordings ``SC4XX1..`` and ``SC4XX2..`` belong to the same person.
        We group by the first 5 characters (``SC4XX``) so that both nights
        always land in the same split, preventing information leakage.
        """
        return subject_id[:5]

    def get_splits(self, fold: int = 0):
        """Subject-aware 70/15/15 split that keeps all nights of one person together."""
        import random as _random
        from collections import defaultdict

        rng = _random.Random(42 + int(fold))

        # Group recording-level subject_ids by real person
        groups: dict[str, list[str]] = defaultdict(list)
        for spec in self._subjects:
            key = self._subject_group_key(spec.subject_id)
            groups[key].append(spec.subject_id)

        # Shuffle at subject-group level
        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        n = len(group_keys)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)

        train_groups = group_keys[:n_train]
        valid_groups = group_keys[n_train : n_train + n_valid]
        test_groups = group_keys[n_train + n_valid :]

        # Expand back to individual recording subject_ids
        train = [sid for g in train_groups for sid in groups[g]]
        valid = [sid for g in valid_groups for sid in groups[g]]
        test = [sid for g in test_groups for sid in groups[g]]

        return train, valid, test

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse hypnogram EDF annotations into a per-epoch label array."""
        with pyedflib.EdfReader(str(spec.label_path)) as f:
            onsets, durations, stages = f.readAnnotations()

        if len(stages) == 0:
            return np.array([], dtype=np.int16)

        # Total recording span determines number of epochs
        total_sec = max(float(o) + float(d) for o, d in zip(onsets, durations))
        n_epochs = int(total_sec // self.epoch_length_sec)
        labels = np.full(n_epochs, -1, dtype=np.int16)

        for onset, duration, stage_str in zip(onsets, durations, stages):
            stage = SLEEPEDF_STAGE_MAP.get(str(stage_str).strip(), -1)
            i0 = int(round(float(onset) / self.epoch_length_sec))
            i1 = int(round((float(onset) + float(duration)) / self.epoch_length_sec))
            if i1 > n_epochs:
                i1 = n_epochs
            labels[i0:i1] = stage

        return labels
