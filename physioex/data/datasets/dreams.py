"""DREAMS micro-event databases (University of Mons / CHU de Charleroi): K-complexes and spindles.

Expected layout (the two ``.rar`` archives extracted as they are)::

    <root>/
        DatabaseKcomplexes/
            excerptN.edf                  # 30 min, 23 channels, 200 Hz
            Hypnogram_excerptN.txt        # "[Hypnogram]" + one stage per 5 s
            Visual_scoring1_excerptN.txt  # expert 1: "[vis1_Kcomplexes/CZ-A1]" + "onset duration" rows
            Visual_scoring2_excerptN.txt  # expert 2 (excerpts 1-5)
            Automatic_detection_excerptN.txt
        DatabaseSpindles/
            ... same names (8 excerpts; 1 and 3 have only C3-A1 at 100 / 50 Hz)

Each excerpt is a subject: ``kc_excerptN`` or ``sp_excerptN`` (``subset="kcomplexes"`` or
``"spindles"`` keeps one database). Expert events come as :class:`SleepEvent` with
``type`` ``"k_complex"`` or ``"spindle"`` and ``concept`` ``"vis1"``, ``"vis2"`` or
``"auto"``; the scoring channel is in ``extra["channel"]``.

Hypnogram (R&K, 5 s steps): 5 W, 4 REM, 3 S1, 2 S2, 1 S3, 0 S4, negative = movement /
unscored. A 30 s epoch takes the stage of at least 4 of its 6 steps (S3 and S4 count as N3),
otherwise it is unscored (-1).

Channels, inferred from the signals (no channel documentation is available offline):
EEG ``CZ-A1`` (the K-complex scoring channel), ``FP1``, ``FP2``, ``O1``, ``O2``; ``EOG1`` /
``EOG2`` are single eyes on the mastoid (correlated in NREM, less in REM), so the EOG is their
difference; ``EMG1`` is the chin (2.9 µV RMS 10-90 Hz in W, 0.6 µV asleep). ``EMG2`` and
``EMG3`` look like legs but their side is unknown, so they are not used. ``CZ2-A1`` is a second
Cz electrode and is left out of the cleaning (a near copy of ``CZ-A1``).

The EDFs violate the spec (a channel with physical min = max), so they are read with
:func:`physioex.data.readers.edf.read_edf_lenient`.

Licence CC BY-NC-ND 3.0: attribute University of Mons - TCTS Lab (S. Devuyst, T. Dutoit) and
ULB - CHU de Charleroi Sleep Laboratory (M. Kerkhofs).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root
from physioex.data.cleaning.montage import CleanMontage
from physioex.data.events import SleepEvent
from physioex.data.readers.annotations import parse_onset_duration_txt
from physioex.data.readers.edf import EDFHeader, ResolvedChannel, read_edf_lenient

#: R&K code of the DREAMS hypnograms -> physioex stage (0 W, 1 N1, 2 N2, 3 N3, 4 REM).
DREAMS_STAGE_MAP: Dict[int, int] = {5: 0, 4: 4, 3: 1, 2: 2, 1: 3, 0: 3}
STEP_SEC = 5.0
MIN_AGREE = 4  # of the 6 steps of a 30 s epoch

SUBSETS = {"kcomplexes": ("DatabaseKcomplexes", "kc", "k_complex"),
           "spindles": ("DatabaseSpindles", "sp", "spindle")}
SCORERS = {"Visual_scoring1": "vis1", "Visual_scoring2": "vis2", "Automatic_detection": "auto"}


@lru_cache(maxsize=4)
def _read(path: str, mtime: float) -> Dict:
    return read_edf_lenient(path)


def hypnogram_to_epochs(steps: np.ndarray, epoch_sec: float = 30.0) -> np.ndarray:
    """5 s R&K steps -> 30 s physioex labels (majority of at least ``MIN_AGREE`` steps)."""
    per = int(round(epoch_sec / STEP_SEC))
    n = len(steps) // per
    st = np.array([DREAMS_STAGE_MAP.get(int(s), -1) for s in steps[:n * per]], dtype=np.int16).reshape(n, per)
    out = np.full(n, -1, dtype=np.int16)
    for i, row in enumerate(st):
        v = row[row >= 0]
        if len(v):
            vals, cnt = np.unique(v, return_counts=True)
            if cnt.max() >= MIN_AGREE:
                out[i] = vals[cnt.argmax()]
    return out


class DREAMSDataset(BasePhysioDataset):
    """DREAMS K-complexes (10 excerpts) and spindles (8 excerpts) databases."""

    DATASET_NAME = "dreams"
    DATASET_SUBDIR = "dreams"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    CLEAN_MONTAGE = CleanMontage(
        eeg=["CZ-A1", "C3-A1"],
        eog=[("EOG1", "EOG2"), ("EOG1-A2", "EOG2-A2"), ("EOG1-A1", "EOG2-A1")],
        chin=["EMG1"],
        ecg=["ECG"],
        extra={
            "FP1": ["FP1-A2", "FP1-A1"],
            "FP2": ["FP2-A1"],
            "O1": ["O1-A2", "O1-A1"],
            "O2": ["O2-A1"],
        },
        extra_eeg=["FP1", "FP2", "O1", "O2"],
        level="rich",
        mains=50.0,
        notes="EOG1/EOG2 single eyes on the mastoid; EMG1 = chin (inferred from W vs sleep RMS); "
              "EMG2/EMG3 legs of unknown side, unused; CZ2-A1 unused.",
    )

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": ["CZ-A1", "C3-A1", "EEG"],
        "EOG": [("EOG1", "EOG2"), ("EOG1-A2", "EOG2-A2"), ("EOG1-A1", "EOG2-A1")],
        "EMG": ["EMG1"],
        "ECG": ["ECG"],
    }

    def __init__(self, root: Optional[str] = None, trim_excess_wake: bool = False, **kwargs):
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        # 30-minute excerpts: nothing to trim around the sleep period.
        super().__init__(root=root, trim_excess_wake=trim_excess_wake, **kwargs)

    # ------------------------------------------------------------------
    # Subjects, labels, events
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        out: List[SubjectSpec] = []
        for name, (folder, prefix, kind) in SUBSETS.items():
            d = root / folder
            if not d.is_dir():
                continue
            edfs = sorted(d.glob("excerpt*.edf"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
            for edf in edfs:
                n = re.sub(r"\D", "", edf.stem)
                hyp = d / f"Hypnogram_excerpt{n}.txt"
                ev = {SCORERS[s]: d / f"{s}_excerpt{n}.txt" for s in SCORERS if (d / f"{s}_excerpt{n}.txt").exists()}
                out.append(SubjectSpec(subject_id=f"{prefix}_excerpt{n}", edf_path=edf,
                                       label_path=hyp if hyp.exists() else None,
                                       external_meta={"subset": name, "event_type": kind,
                                                      "event_files": {k: str(v) for k, v in ev.items()}}))
        return out

    def _filter_subset(self, subjects: List[SubjectSpec], subset) -> List[SubjectSpec]:
        if subset in (None, "all"):
            return subjects
        if subset not in SUBSETS:
            raise ValueError(f"DREAMS subset must be one of {sorted(SUBSETS)} or 'all'; got {subset!r}")
        return [s for s in subjects if (s.external_meta or {}).get("subset") == subset]

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        if spec.label_path is None or not Path(spec.label_path).exists():
            dur = self._read_edf_header(spec).duration_sec
            return np.full(int(dur // self.epoch_length_sec), -1, dtype=np.int16)
        lines = [s.strip() for s in Path(spec.label_path).read_text(encoding="latin-1").splitlines()]
        steps = np.array([int(float(s)) for s in lines if s and not s.startswith("[")])
        return hypnogram_to_epochs(steps, self.epoch_length_sec)

    def _read_subject_events(self, spec: SubjectSpec) -> List[SleepEvent]:
        meta = spec.external_meta or {}
        kind = meta.get("event_type", "other")
        events: List[SleepEvent] = []
        for scorer, path in sorted(meta.get("event_files", {}).items()):
            header, rows = parse_onset_duration_txt(path)
            chan = header.split("/", 1)[1] if header and "/" in header else None
            events += [SleepEvent(type=kind, concept=scorer, onset_sec=float(o), duration_sec=float(d),
                                  extra={"channel": chan, "source": Path(path).name}) for o, d in rows]
        return events

    # ------------------------------------------------------------------
    # Non-compliant EDF: numpy reader
    # ------------------------------------------------------------------

    def _edf(self, spec: SubjectSpec) -> Dict:
        p = Path(spec.edf_path)
        return _read(str(p), p.stat().st_mtime)

    def _read_edf_header(self, spec: SubjectSpec) -> EDFHeader:
        e = self._edf(spec)
        return EDFHeader(
            available_channels=list(e["labels"]),
            channel_fs=dict(zip(e["labels"], e["fs"])),
            channel_units=dict(zip(e["labels"], e["units"])),
            patient_meta={"patient_code": None, "sex": None, "birthdate": None, "patient_name": None,
                          "patient_additional": None},
            duration_sec=float(e["duration_sec"]),
            source_mtime=Path(spec.edf_path).stat().st_mtime,
            start_sec=float(e["start_sec"]),
        )

    def _read_subject_channel(self, spec: SubjectSpec, resolved: ResolvedChannel) -> Tuple[np.ndarray, float]:
        e = self._edf(spec)
        idx = resolved._indices
        if not idx:
            raise ValueError(f"ResolvedChannel has no indices: {resolved!r}")
        if resolved.is_differential:
            a, b = e["signals"][idx[0]], e["signals"][idx[1]]
            m = min(len(a), len(b))
            return (a[:m] - b[:m]).astype(np.float32), resolved.fs_in
        return e["signals"][idx[0]].copy(), resolved.fs_in
