"""WSC (Wisconsin Sleep Cohort) dataset.

WSC has 5 longitudinal visits. Each visit is selected via the ``visit``
constructor parameter. Raw data lives in a flat ``polysomnography/``
directory with EDF signal files and annotation sidecars.

Annotation formats (tried in order):
  1. ``.allscore.txt`` -- timestamped stage events. Parser finds "START
     RECORDING" base time, then fills epochs between "STAGE - *" events.
  2. ``.stg.txt`` -- tab-separated epoch-level stage file with header
     ``Epoch\\tUser-Defined Stage\\tCAST-Defined Stage``.

Subjects without any annotation file are silently skipped.

Subject metadata is loaded from ``datasets/wsc-dataset-0.8.0.csv``, keyed
by ``(wsc_id, wsc_vst)`` and filtered to the selected visit.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root
from physioex.data.cleaning.montage import CleanMontage

logger = logging.getLogger("physioex.data")


# Stage mapping for .stg.txt files (column "User-Defined Stage").
# 0=W, 1=N1, 2=N2, 3=N3, 4=N3(merged), 5=REM, 6=unscored, 7=unscored
_STG_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: -1,
    7: -1,
}

# Stage mapping for .allscore.txt "STAGE - <name>" events.
_ALLSCORE_STAGE_MAP = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "NO STAGE": -1,
    "MVT": -1,
}


class WSCDataset(BasePhysioDataset):
    """Wisconsin Sleep Cohort dataset.

    Has 3 longitudinal visits. Each visit is selected via the ``visit``
    parameter.

    Args:
        visit: int (1, 2, or 3). Selects which visit's recordings to load.
        root: path to the WSC data directory
              (default: ``$PHYSIOEX_DATA/wsc``).
    """

    DATASET_NAME = "wsc"
    #: 2: allscore labels anchored at START RECORDING (paused stretches removed), labels cut at the EDF end.
    LABELS_REVISION = 2
    DATASET_SUBDIR = "wsc"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    # Channel preferences: WSC uses non-standard channel names.
    #: Clean-channel recipe (physioex.data.cleaning). EEG already referenced to M2; reference of
    #: E1/E2 not verified (the difference is exact only if it is common). ECG in part of the nights.
    CLEAN_MONTAGE = CleanMontage(
        eeg=["C3_M2", "C4_M1"],
        eog=[("E1", "E2")],
        chin=["chin", "cchin_l"],
        leg_l=["lleg_r"],
        ecg=["ECG"],
        extra={
            "F3": ["F3_M2"],
            "O1": ["O1_M2"],
            "C4": ["C4_M1"],
            "Cz": ["Cz_M2"],
            "Fz": ["Fz_M2"],
            "Pz": ["Pz_M2"],
        },
        extra_eeg=["F3", "O1", "C4", "Cz", "Fz", "Pz"],
        level="rich",
        mains=60.0,
        notes="E1/E2 reference to verify",
    )

    CHANNEL_PREFERENCES = {
        "EEG": [
            "C3_M2",
            "C4_M1",
            "C3_M1",
            "C4_M2",
            "F3_M2",
            "F4_M1",
            "O1_M2",  # variants in visit3/4
            "Fz_AVG",
            "C3_AVG",
            ("C3", "M2"),
            ("C4", "M1"),
            "EEG",
        ],
        "EOG": [
            ("E1", "E2"),
            "E1",
            "E2",
            "EOG",
        ],
        "EMG": [
            "chin",
            "cchin_l",
            "cchin_r",
            "rchin_l",
            "EMG",
        ],
    }

    def __init__(
        self,
        visit: int = 1,
        root: Optional[str] = None,
        **kwargs,
    ):
        if visit not in (1, 2, 3, 4, 5):
            raise ValueError(f"visit must be 1, 2, 3, 4, or 5; got {visit}")
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        self.visit = visit
        # Dynamic dataset name so cache dirs are separated per visit.
        self.DATASET_NAME = f"wsc_visit{visit}"
        self._metadata = self._load_metadata(root, visit)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(root: str, visit: int) -> Dict[str, Dict[str, Any]]:
        """Load WSC dataset CSV, filtered to the given visit.

        Returns a dict keyed by wsc_id (as string) with all columns as values.
        """
        csv_path = Path(root) / "datasets" / "wsc-dataset-0.8.0.csv"
        if not csv_path.exists():
            logger.warning("WSC metadata CSV not found: %s", csv_path)
            return {}
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)
        df_visit = df[df["wsc_vst"] == visit]
        meta: Dict[str, Dict[str, Any]] = {}
        for _, row in df_visit.iterrows():
            key = str(int(row["wsc_id"]))
            meta[key] = {
                col: (None if pd.isna(val) else val) for col, val in row.items()
            }
        return meta

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        poly_dir = Path(self.root) / "polysomnography"
        if not poly_dir.exists():
            return []

        subjects: List[SubjectSpec] = []
        visit_str = f"visit{self.visit}"

        for edf in sorted(poly_dir.glob(f"wsc-{visit_str}-*-nsrr.edf")):
            # Extract subject_id from filename.
            # Example: wsc-visit1-10119-nsrr.edf -> parts = ['wsc', 'visit1', '10119', 'nsrr']
            stem = edf.stem  # wsc-visit1-10119-nsrr
            parts = stem.split("-")
            if len(parts) < 4:
                continue
            subject_num = parts[2]

            # Build the base for annotation files: wsc-visit1-10119-nsrr
            base = f"wsc-{visit_str}-{subject_num}-nsrr"

            # Find annotation file (try .allscore.txt first, then .stg.txt)
            allscore = poly_dir / f"{base}.allscore.txt"
            stg = poly_dir / f"{base}.stg.txt"

            if allscore.exists():
                label_path = allscore
            elif stg.exists():
                label_path = stg
            else:
                continue  # skip subjects without annotations

            meta = dict(self._metadata.get(subject_num, {}))
            subjects.append(
                SubjectSpec(
                    subject_id=f"wsc-{visit_str}-{subject_num}",
                    edf_path=edf,
                    label_path=label_path,
                    external_meta=meta,
                )
            )

        return subjects

    # ------------------------------------------------------------------
    # Label reading
    # ------------------------------------------------------------------

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse WSC annotation file (.allscore.txt or .stg.txt)."""
        path = spec.label_path
        suffix = str(path)
        if suffix.endswith(".allscore.txt"):
            labels = self._parse_allscore(path)
        elif suffix.endswith(".stg.txt"):
            labels = self._parse_stg(path)
        else:
            return np.array([], dtype=np.int16)
        # In some nights the scoring runs past the end of the EDF; labels without signal are dropped.
        import pyedflib

        with pyedflib.EdfReader(str(spec.edf_path)) as f:
            n_signal = int(f.getFileDuration() // self.epoch_length_sec)
        return labels[:n_signal]

    # ------------------------------------------------------------------
    # .stg.txt parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stg(path: Path) -> np.ndarray:
        """Parse a WSC ``.stg.txt`` file.

        Format::

            Epoch\\tUser-Defined Stage\\tCAST-Defined Stage
            1\\t0\\t0
            2\\t2\\t2
            ...

        We use the "User-Defined Stage" (column 1) and map via ``_STG_MAP``.
        """
        with open(path) as f:
            lines = f.readlines()

        # Skip header if present
        if lines and lines[0].startswith("Epoch"):
            lines = lines[1:]

        stages = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            user_stage = int(parts[1])
            mapped = _STG_MAP.get(user_stage, -1)
            stages.append(mapped)

        return np.array(stages, dtype=np.int16)

    # ------------------------------------------------------------------
    # .allscore.txt parser
    # ------------------------------------------------------------------

    @staticmethod
    def _allscore_offsets(lines):
        """Seconds from "START RECORDING" for every line of an ``.allscore.txt``, ``None`` where unusable.

        Only lines from the START RECORDING line on are timed: some files list a
        ``STAGE - NO STAGE`` line with a clock time before the recording starts, which
        used to pull the time origin back by hours and shift every later label.
        Midnight is detected as a large backward jump of the clock, not by the hour of day.
        The time between a PAUSED line and the next RECORDING RESUMED line is removed,
        because the EDF has no samples for it.
        Returns ``None`` when there is no START RECORDING line.
        """

        def _clock(line):
            t = datetime.datetime.strptime(line.split("\t")[0].strip(), "%H:%M:%S.%f")
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

        start_i = next((i for i, ln in enumerate(lines) if "START RECORDING" in ln), None)
        if start_i is None:
            return None
        start_sec = _clock(lines[start_i])
        offsets = [None] * len(lines)
        day, prev = 0, start_sec
        skipped, pause_start = 0.0, None   # the EDF does not contain the paused stretches
        for i in range(start_i, len(lines)):
            try:
                t = _clock(lines[i])
            except (ValueError, TypeError):
                continue
            if t < prev - 6 * 3600:
                day += 1
            prev = t
            raw = day * 86400 + t - start_sec
            if "RECORDING RESUMED" in lines[i]:
                if pause_start is not None:
                    skipped += raw - pause_start
                    pause_start = None
            elif "PAUSED" in lines[i] and pause_start is None:
                pause_start = raw
            offsets[i] = raw - skipped
        return offsets

    @staticmethod
    def _parse_allscore(path: Path) -> np.ndarray:
        """Parse a WSC ``.allscore.txt`` file.

        Finds "START RECORDING" to establish base time, then parses
        "STAGE - <name>" events with timestamps. Fills epochs between events
        with the preceding stage value. Handles midnight crossing (hours <= 12
        get +1 day).

        Ported from ``physioex/preprocess/wsc.py:read_allscore_file()``.
        """
        with open(path, encoding="latin1") as f:
            lines = f.readlines()

        offsets = WSCDataset._allscore_offsets(lines)
        if offsets is None:
            return np.array([], dtype=np.int16)

        stages = []
        current = 0.0
        stage = 7  # unscored at the beginning (maps to -1)

        for line, off in zip(lines, offsets):
            if off is None or "STAGE - " not in line:
                continue

            new_stage_str = line.split("STAGE - ")[-1].strip()
            new_stage = _ALLSCORE_STAGE_MAP.get(new_stage_str)
            if new_stage is None:
                # Unrecognised stage label -- treat as unscored
                new_stage = -1

            # Fill epochs from the current stage onset to this one with the previous stage
            passed_epochs = int((off - current) / 30)
            # Map the initial "unscored" sentinel (7) to -1
            fill_val = -1 if stage == 7 else stage
            stages.extend([fill_val] * max(passed_epochs, 0))

            stage = new_stage
            current = max(off, current)

        # Tail: last stage event to the last timed line of the file
        last = max((o for o in offsets if o is not None), default=current)
        passed_epochs = int((last - current) / 30)
        fill_val = -1 if stage == 7 else stage
        stages.extend([fill_val] * max(passed_epochs, 0))

        return np.array(stages, dtype=np.int16)

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def _read_subject_events(self, spec):
        """Parse non-stage events from WSC .allscore.txt annotation files."""
        if spec.label_path is None:
            return []
        label = str(spec.label_path)
        events = self._read_subject_events_unfiltered(spec, label)
        if not events:
            return events
        # A few EDFs are cuts of a longer recording: events past the end of the signal are dropped.
        import pyedflib

        with pyedflib.EdfReader(str(spec.edf_path)) as f:
            end = float(f.getFileDuration())
        return [e for e in events if e.onset_sec <= end]

    def _read_subject_events_unfiltered(self, spec, label):
        if label.endswith(".allscore.txt"):
            return self._parse_allscore_events(spec.label_path)
        if label.endswith(".stg.txt"):
            # Gamma-system nights (2000-2009): events are in the ``.sco.txt`` next to the ``.stg.txt``
            sco = Path(label[: -len(".stg.txt")] + ".sco.txt")
            if sco.exists():
                import pyedflib

                # "Scan # x2" counts samples at twice the EDF rate (200/s for the 100 Hz
                # files, 256/s for the 128 Hz ones); checked against the clock column.
                with pyedflib.EdfReader(str(spec.edf_path)) as f:
                    scans_per_sec = 2.0 * float(f.getSampleFrequency(0))
                    start = f.getStartdatetime()
                start_sec = start.hour * 3600 + start.minute * 60 + start.second
                return self._parse_sco_events(sco, scans_per_sec, start_sec)
        return []

    @staticmethod
    def _sco_event(text: str):
        """Map the ``Marker Text`` of a ``.sco.txt`` row to ``(type, concept)``; ``None`` to skip it."""
        t = text.strip().lower()
        if not t:
            return None
        if "arousal" in t or t in ("respa", "spona"):
            if t.startswith("resp"):
                return "arousal", "Respiratory arousal"
            if t.startswith("spon"):
                return "arousal", "Spontaneous arousal"
            return "arousal", "Arousal"
        if "hypopnea" in t:
            return "respiratory", "Central hypopnea" if t.startswith("central") else "Hypopnea"
        if "apnea" in t or t == "oa":
            if t.startswith(("obs", "oa")):
                return "respiratory", "Obstructive apnea"
            if t.startswith("central"):
                return "respiratory", "Central apnea"
            if t.startswith("mixed"):
                return "respiratory", "Mixed apnea"
            return "respiratory", "Apnea"
        if t == "ca":
            return "respiratory", "Central apnea"
        if t == "ma":
            return "respiratory", "Mixed apnea"
        if t == "sao2":
            return "desaturation", "SaO2 desaturation"
        if t.startswith("plm"):
            return "limb_movement", "Periodic limb movement"
        if t in ("lma", "lm a"):
            return "limb_movement", "Limb movement with arousal"
        if t == "lm":
            return "limb_movement", "Limb movement"
        if t.startswith("bad ") or "artifact" in t:
            return "artifact", text.strip()
        return None

    @staticmethod
    def _parse_sco_events(
        path: Path, scans_per_sec: float = 200.0, start_clock_sec: Optional[float] = None
    ):
        """Parse the event rows of a WSC ``.sco.txt`` (Gamma system, 2000-2009).

        Columns: ``Epoch#, Scan # x2, Length (Scanx2), Marker Code, Marker Text,
        Channel #, Time, Value, Drop, Length``. Onset and duration are scans divided
        by ``scans_per_sec`` (the length is not measured for limb movements and arousals; a negative
        length means the value was written 100 times too large). Rows without a marker code are padding.

        A few files write the scan columns with two extra digits. When ``start_clock_sec``
        (clock time of the EDF start, in seconds) is given, the ``Time`` column is used as a
        check and, where it disagrees with the scan by more than 5 s, it gives the onset.
        """
        from physioex.data.events import SleepEvent

        rate = float(scans_per_sec)
        events = []
        with open(path, encoding="latin1") as f:
            rows = f.read().splitlines()[1:]
        for line in rows:
            parts = [x.strip() for x in line.split("\t")]
            if len(parts) < 7 or not parts[3] or not parts[1]:
                continue
            kind = WSCDataset._sco_event(parts[4])
            if kind is None:
                continue
            try:
                onset = float(parts[1]) / rate
                length = float(parts[2]) if parts[2] else 0.0
                # Some files store the length with two extra digits and a negative sign
                duration = length / rate if length >= 0 else -length / (rate * 100.0)
            except ValueError:
                continue
            if start_clock_sec is not None:
                try:
                    h, m, sec = (float(x) for x in parts[6].split(":")[:3] + ["0"] * (3 - len(parts[6].split(":"))))
                    clock = (h * 3600 + m * 60 + sec - start_clock_sec) % 86400
                except ValueError:
                    clock = None
                if clock is not None and abs(onset - clock) > 5.0:
                    scale = 100.0 if onset > 50 * max(clock, 1.0) else 1.0   # the "two extra digits" files
                    onset, duration = clock, duration / scale
            extra = {}
            if kind[0] == "desaturation":
                for key, idx in (("spO2_min", 7), ("spO2_drop", 8)):
                    try:
                        extra[key] = float(parts[idx])
                    except (ValueError, IndexError):
                        pass
            events.append(
                SleepEvent(
                    type=kind[0],
                    concept=kind[1],
                    onset_sec=onset,
                    duration_sec=duration,
                    extra=extra,
                )
            )
        return events

    @staticmethod
    def _parse_allscore_events(path: Path):
        """Parse event lines from a WSC ``.allscore.txt`` file.

        Extracts AROUSAL, RESPIRATORY EVENT, LM (limb movement),
        POSITION, DESATURATION, and SAO2 events and converts them
        to ``SleepEvent`` objects.
        """
        import re
        from physioex.data.events import SleepEvent

        with open(path, encoding="latin1") as f:
            lines = f.readlines()

        offsets = WSCDataset._allscore_offsets(lines)
        if offsets is None:
            return []

        # Patterns for event lines
        dur_pattern = re.compile(r"DUR:\s*([\d.]+)\s*SEC", re.IGNORECASE)
        desat_pattern = re.compile(r"DESAT\s+([\d.]+)\s*%", re.IGNORECASE)
        desat_from_to = re.compile(
            r"DESATURATION\s+FROM\s+([\d.]+)\s+TO\s+([\d.]+)", re.IGNORECASE
        )

        events = []
        for line, onset in zip(lines, offsets):
            if onset is None:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            time_str = parts[0]
            content = "\t".join(parts[1:]).strip()

            # Skip stage lines and non-event lines
            if "STAGE - " in content or "START RECORDING" in content:
                continue
            if "LIGHTS OFF" in content or "LIGHTS ON" in content:
                continue

            # Extract duration if present
            dur_match = dur_pattern.search(content)
            duration = float(dur_match.group(1)) if dur_match else 0.0

            extra = {}

            if content.startswith("AROUSAL"):
                category = "arousal"
                concept = content.replace("AROUSAL", "").strip(" -")
                if not concept:
                    concept = "Arousal"

            elif "RESPIRATORY EVENT" in content:
                category = "respiratory"
                # Extract concept (e.g. HYPOPNEA, OBSTRUCTIVE APNEA)
                parts_resp = content.split(" - ")
                concept = "Respiratory Event"
                for part in parts_resp:
                    part = part.strip()
                    if (
                        part
                        and part != "RESPIRATORY EVENT"
                        and not dur_pattern.match(part)
                    ):
                        if not desat_pattern.match(part):
                            concept = part.title()
                            break
                desat_m = desat_pattern.search(content)
                if desat_m:
                    extra["desaturation_pct"] = float(desat_m.group(1))

            elif content.startswith("LM"):
                category = "limb_movement"
                concept = content.replace("LM", "Limb Movement").strip(" -")
                if concept == "Limb Movement":
                    lm_parts = content.split(" - ")
                    if len(lm_parts) > 1:
                        concept = "Limb Movement - " + lm_parts[-1].strip()

            elif content.startswith("POSITION"):
                category = "position"
                pos_parts = content.split(" - ")
                concept = pos_parts[-1].strip() if len(pos_parts) > 1 else "Position"

            elif "DESATURATION" in content:
                category = "desaturation"
                m = desat_from_to.search(content)
                if m:
                    extra["spO2_from"] = float(m.group(1))
                    extra["spO2_to"] = float(m.group(2))
                concept = "Desaturation"

            elif content.startswith("SAO2"):
                category = "desaturation"
                concept = content.strip()

            else:
                continue  # skip unrecognised lines

            events.append(
                SleepEvent(
                    type=category,
                    concept=concept,
                    onset_sec=onset,
                    duration_sec=duration,
                    extra=extra if extra else {},
                )
            )

        return events
