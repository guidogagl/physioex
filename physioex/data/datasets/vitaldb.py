"""VitalDB intraoperative dataset (Seoul National University Hospital).

6,388 surgical cases recorded with VitalRecorder; 5,871 of them carry raw
two-channel frontal EEG from the BIS module (``BIS/EEG1_WAV``,
``BIS/EEG2_WAV``) at 128 Hz, 5,622 under general anaesthesia.

This is **not** a sleep dataset: there is no hypnogram, and sleep stages are not
defined under general anaesthesia.  What VitalDB provides instead is a
*case-level* clinical outcome (ICU admission, length of stay, AKI, ...).  The
dataset therefore broadcasts one case-level target across every epoch of that
case, which is the standard formulation for this task -- train at the window
level, aggregate to the case at evaluation time.

Expected layout on disk (produced by the downloader in the
``anesthesia-eeg-outcomes`` repository)::

    <root>/
        raw/<caseid>.vital        native VitalRecorder files
        meta/cases.csv            74 perioperative clinical parameters
        meta/trks.csv             caseid,tname,tid track index
        meta/labs.csv             perioperative laboratory results

Targets are supplied by the caller and are **not** defined here: the clinical
definitions (KDIGO staging, ICU admission, length-of-stay thresholds) belong to
the analysis project, not to a sleep-staging library.  Pass either a column name
from ``cases.csv`` or a callable over the case row::

    VitalDBDataset(root=..., target="my_precomputed_column")
    VitalDBDataset(root=..., target=lambda row: int(float(row["icu_days"] or 0) > 0))

With ``target=None`` every epoch is labelled ``-1`` (unlabelled), which is the
right mode for self-supervised or purely signal-level use.

Reference: Lee HC, Jung CW. *VitalDB, a high-fidelity multi-parameter vital
signs database in surgical patients.* Sci Data 9, 279 (2022).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root
from physioex.data.readers.edf import EDFHeader, ResolvedChannel
from physioex.data.readers.vital import probe_vital_header, read_vital_channel

logger = logging.getLogger("physioex.data")

# Track carried by every usable case; also the first EEG preference.
EEG1 = "BIS/EEG1_WAV"
EEG2 = "BIS/EEG2_WAV"

TargetSpec = Union[str, Callable[[Dict[str, Any]], int], None]


class VitalDBDataset(BasePhysioDataset):
    """VitalDB intraoperative EEG with a case-level target.

    Args:
        root: dataset root containing ``raw/`` and ``meta/``.  Defaults to
            ``$PHYSIOEX_DATA/vitaldb``.
        target: column name in ``cases.csv``, or ``callable(case_row) -> int``,
            or ``None`` for unlabelled use.  The returned value is broadcast to
            every epoch of the case; return ``-1`` for "undefined for this case"
            (the sentinel ignored by ``CrossEntropyLoss(ignore_index=-1)``).
        ane_type: keep only cases with this ``ane_type`` in ``cases.csv``.
            Defaults to ``"General"``; pass ``None`` to keep every case.
        require_tracks: cases must carry all of these tracks (checked against
            ``meta/trks.csv``, no file opening).  Defaults to the EEG channel.
        epoch_length_sec: defaults to 120 s -- the 2-minute window used by the
            intraoperative-EEG outcome literature, not the 30 s sleep epoch.

    Note:
        ``BasePhysioDataset._sanitize_labels`` enforces the AASM range
        ``{-1, 0..4}``.  Binary targets (0/1) pass through unchanged, but a
        target outside that range would be silently coerced to ``-1``; the
        constructor therefore rejects such targets up front.
    """

    DATASET_NAME = "vitaldb"
    DATASET_SUBDIR = "vitaldb"
    DEFAULT_EPOCH_LENGTH_SEC = 120.0

    # VitalDB exposes already-derived bipolar frontal EEG from the BIS sensor.
    # No EOG/EMG/ECG equivalents are exposed here: other waveforms in the file
    # (SNUADC/ECG_II, SNUADC/ART, ...) belong to different devices and are
    # requested by explicit track name when needed.
    CHANNEL_PREFERENCES = {
        "EEG": [EEG1, EEG2],
    }

    def __init__(
        self,
        root: Optional[str] = None,
        target: TargetSpec = None,
        ane_type: Optional[str] = "General",
        require_tracks: tuple = (EEG1,),
        **kwargs,
    ):
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)

        # All three must be set before super().__init__, which calls
        # _list_subjects() and then _read_subject_labels().
        self._target = target
        self._ane_type = ane_type
        self._require_tracks = tuple(require_tracks)
        self._cases: Dict[str, Dict[str, Any]] = {}
        # caseid -> resolved target, filled eagerly in _list_subjects
        self._targets: Dict[str, int] = {}

        # Sleep-specific runtime transforms are meaningless here: label 0 means
        # "negative outcome", not Wake.
        kwargs.setdefault("trim_excess_wake", False)
        kwargs.setdefault("channels", ["EEG"])

        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Clinical table
    # ------------------------------------------------------------------

    def _load_cases(self) -> Dict[str, Dict[str, Any]]:
        """Read ``meta/cases.csv`` into ``{caseid: row}``."""
        path = Path(self.root) / "meta" / "cases.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"[{self.DATASET_NAME}] missing clinical table {path}. "
                "Run scripts/download_vitaldb.py to populate meta/."
            )
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return {row["caseid"]: row for row in csv.DictReader(fh)}

    def _cases_with_tracks(self) -> Optional[set]:
        """Case ids carrying every required track, from ``meta/trks.csv``.

        Returns ``None`` when no tracks are required or the index is absent, in
        which case track availability is left to header probing.
        """
        if not self._require_tracks:
            return None
        path = Path(self.root) / "meta" / "trks.csv"
        if not path.exists():
            logger.warning(
                f"[{self.DATASET_NAME}] {path} not found; cannot pre-filter on "
                f"track availability"
            )
            return None
        have: Dict[str, set] = {}
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["tname"] in self._require_tracks:
                    have.setdefault(row["caseid"], set()).add(row["tname"])
        wanted = set(self._require_tracks)
        return {cid for cid, names in have.items() if wanted <= names}

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        raw_dir = Path(self.root) / "raw"
        if not raw_dir.exists():
            logger.warning(f"[{self.DATASET_NAME}] no raw/ directory under {self.root}")
            return []

        self._cases = self._load_cases()
        with_tracks = self._cases_with_tracks()

        subjects: List[SubjectSpec] = []
        for caseid, row in self._cases.items():
            if self._ane_type is not None and row.get("ane_type") != self._ane_type:
                continue
            if with_tracks is not None and caseid not in with_tracks:
                continue
            path = raw_dir / f"{caseid}.vital"
            if not path.exists():
                continue
            # Resolve the target here, not in _read_subject_labels: that runs
            # under _build_index's skip_corrupt guard, which would turn a bad
            # target into every case being silently dropped.
            self._targets[caseid] = self._resolve_target(row, caseid)
            subjects.append(
                SubjectSpec(
                    subject_id=caseid,
                    edf_path=path,
                    label_path=None,
                    # subjectid is the patient: splits must group on it, since a
                    # patient may contribute several cases.
                    external_meta=dict(row),
                )
            )
        subjects.sort(key=lambda s: int(s.subject_id))
        return subjects

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_splits(self, fold: int = 0):
        """Train/valid/test case ids, split **by patient**.

        Overrides the base 70/15/15 split, which shuffles ``subject_id`` — here
        the case, not the person.  225 of the 5,397 patients in the cohort
        contribute more than one operation, so a case-level split puts the same
        patient on both sides and inflates every score.  Patients are shuffled
        instead, and all of a patient's cases follow them into one split.
        """
        import random as _random

        by_patient: Dict[str, List[str]] = {}
        for spec in self._subjects:
            patient = (spec.external_meta or {}).get("subjectid") or spec.subject_id
            by_patient.setdefault(str(patient), []).append(spec.subject_id)

        patients = sorted(by_patient)
        _random.Random(42 + int(fold)).shuffle(patients)
        n = len(patients)
        n_train, n_valid = int(0.70 * n), int(0.15 * n)

        def cases_of(group):
            return [cid for p in group for cid in by_patient[p]]

        return (
            cases_of(patients[:n_train]),
            cases_of(patients[n_train : n_train + n_valid]),
            cases_of(patients[n_train + n_valid :]),
        )

    # ------------------------------------------------------------------
    # Non-EDF reading
    # ------------------------------------------------------------------

    def _read_edf_header(self, spec: SubjectSpec) -> EDFHeader:
        meta = spec.external_meta or {}
        return probe_vital_header(
            spec.edf_path,
            patient_meta={
                "age": meta.get("age"),
                "sex": meta.get("sex"),
                "patient_code": meta.get("subjectid"),
                "birthdate": None,
            },
        )

    def _read_subject_channel(self, spec: SubjectSpec, resolved: ResolvedChannel):
        return read_vital_channel(spec.edf_path, resolved)

    # ------------------------------------------------------------------
    # Case-level target broadcast over epochs
    # ------------------------------------------------------------------

    def _resolve_target(self, row: Dict[str, Any], caseid: str) -> int:
        """Evaluate the target for one case row; ``-1`` when undefined.

        Raises:
            ValueError: if the target is outside ``{-1, 0, 1}``.  Called from
            ``_list_subjects`` so the error surfaces at construction time
            instead of being swallowed by the ``skip_corrupt`` guard.
        """
        if self._target is None:
            return -1
        try:
            if callable(self._target):
                value = self._target(row)
            else:
                raw = row.get(self._target, "")
                if raw is None or str(raw).strip() == "":
                    return -1
                value = int(float(raw))
        except (TypeError, ValueError) as exc:
            logger.warning(
                f"[{self.DATASET_NAME}] case {caseid}: cannot resolve target "
                f"({exc}); labelling as -1"
            )
            return -1
        if value is None:
            return -1
        value = int(value)
        if value not in (-1, 0, 1):
            raise ValueError(
                f"[{self.DATASET_NAME}] case {caseid}: target={value} is "
                "outside {-1, 0, 1}; _sanitize_labels would coerce it to -1. "
                "Encode the outcome as a binary label."
            )
        return value

    def _target_value(self, spec: SubjectSpec) -> int:
        """Case-level target, resolved once during subject discovery."""
        return self._targets.get(spec.subject_id, -1)

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """One epoch label per physical epoch, all equal to the case target."""
        header = self._headers.get(spec.subject_id)
        if header is None:
            header = self._read_edf_header(spec)
        n_epochs = int(header.duration_sec // self.epoch_length_sec)
        if n_epochs <= 0:
            return np.array([], dtype=np.int16)
        return np.full(n_epochs, self._target_value(spec), dtype=np.int16)

    def _load_or_compute_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Always compute; never touch the on-disk label cache.

        The base implementation keys the label cache on ``(DATASET_NAME,
        subject_id)`` only.  Here the labels also depend on ``target``, so a
        cached array would be silently reused across different endpoints.
        Recomputing costs one ``np.full`` per case, so the cache buys nothing
        while the signal cache -- the expensive one -- keeps working normally.
        """
        labels = self._read_subject_labels(spec)
        labels = self._sanitize_labels(np.asarray(labels, dtype=np.int16))
        return labels.astype(np.int64, copy=False)
