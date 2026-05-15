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
    DATASET_SUBDIR = "wsc"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    # Channel preferences: WSC uses non-standard channel names.
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
            return self._parse_allscore(path)
        elif suffix.endswith(".stg.txt"):
            return self._parse_stg(path)
        else:
            return np.array([], dtype=np.int16)

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

        # Find START RECORDING time
        start_time = None
        for line in lines:
            if "START RECORDING" in line:
                time_str = line.split("\t")[0]
                start_time = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
                break

        if start_time is None:
            return np.array([], dtype=np.int16)

        stages = []
        current_time = start_time
        stage = 7  # unscored at the beginning (maps to -1)

        for line in lines:
            if "STAGE - " not in line:
                continue

            new_stage_str = line.split("STAGE - ")[-1].strip()
            new_stage = _ALLSCORE_STAGE_MAP.get(new_stage_str)
            if new_stage is None:
                # Unrecognised stage label -- treat as unscored
                new_stage = -1

            stage_time_str = line.split("\t")[0]
            stage_time = datetime.datetime.strptime(stage_time_str, "%H:%M:%S.%f")

            # Handle midnight crossing: hours <= 12 assumed next day
            if stage_time.hour <= 12:
                stage_time += datetime.timedelta(days=1)

            # Fill epochs from current_time to stage_time with the previous stage
            time_passed = stage_time - current_time
            passed_epochs = int(time_passed.total_seconds() / 30)
            # Map the initial "unscored" sentinel (7) to -1
            fill_val = -1 if stage == 7 else stage
            stages.extend([fill_val] * passed_epochs)

            stage = new_stage
            current_time = stage_time

        # Handle the tail: last STAGE event to end of file
        if lines:
            end_time_str = lines[-1].split("\t")[0]
            try:
                end_time = datetime.datetime.strptime(end_time_str, "%H:%M:%S.%f")
                end_time += datetime.timedelta(days=1)
                time_passed = end_time - current_time
                passed_epochs = int(time_passed.total_seconds() / 30)
                fill_val = -1 if stage == 7 else stage
                stages.extend([fill_val] * passed_epochs)
            except ValueError:
                pass  # malformed last line -- ignore

        return np.array(stages, dtype=np.int16)

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def _read_subject_events(self, spec):
        """Parse non-stage events from WSC .allscore.txt annotation files."""
        if spec.label_path is None or not str(spec.label_path).endswith(
            ".allscore.txt"
        ):
            return []
        return self._parse_allscore_events(spec.label_path)

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

        # Find START RECORDING time to compute absolute offsets
        start_time = None
        for line in lines:
            if "START RECORDING" in line:
                time_str = line.split("\t")[0]
                start_time = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
                break

        if start_time is None:
            return []

        def _to_seconds(time_str: str) -> float:
            """Convert a time string to seconds from recording start."""
            t = datetime.datetime.strptime(time_str.strip(), "%H:%M:%S.%f")
            # Handle midnight crossing: hours <= 12 assumed next day
            if t.hour <= 12:
                t += datetime.timedelta(days=1)
            diff = t - start_time
            return diff.total_seconds()

        # Patterns for event lines
        dur_pattern = re.compile(r"DUR:\s*([\d.]+)\s*SEC", re.IGNORECASE)
        desat_pattern = re.compile(r"DESAT\s+([\d.]+)\s*%", re.IGNORECASE)
        desat_from_to = re.compile(
            r"DESATURATION\s+FROM\s+([\d.]+)\s+TO\s+([\d.]+)", re.IGNORECASE
        )

        events = []
        for line in lines:
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

            try:
                onset = _to_seconds(time_str)
            except (ValueError, TypeError):
                continue

            if onset < 0:
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
