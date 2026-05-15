"""STAGES (Stanford Technology Analytics and Genomics in Sleep) dataset.

1,914 recordings across 13 clinical sites with 4 different channel naming
conventions. Bipolar EEG montage (C4-M1/C3-M2 equivalent across all sites),
EOG, chin EMG. Rich event annotations (respiratory, arousal, PLM).

Two subsetting dimensions:
  - site: one of 13 site codes or None for all
  - recording: "first", "second" (repeat _1 recordings), or "all"

Demographics from stages-harmonized-dataset-0.3.0.csv (10 columns)
and stages-dataset-0.3.0.csv (433 columns).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root
from physioex.data.readers.annotations import parse_stages_csv, STAGES_STAGE_MAP

logger = logging.getLogger("physioex.data")


SITES: List[str] = [
    "BOGN",
    "GSBB",
    "GSDV",
    "GSLH",
    "GSSA",
    "GSSW",
    "MSMI",
    "MSNF",
    "MSQW",
    "MSTH",
    "MSTR",
    "STLK",
    "STNF",
]

CHANNEL_PREFERENCES: Dict[str, List] = {
    "EEG": [
        # Grael/Compumedics style (GSBB, GSDV, GSLH, GSSA, GSSW, MSMI)
        "EEG_C4-A1",
        "EEG_C3-A2",
        "EEG_F4-A1",
        "EEG_F3-A2",
        "EEG_O2-A1",
        "EEG_O1-A2",
        # Hyphenated bipolar (STLK)
        "C4-M1",
        "C3-M2",
        "F4-M1",
        "F3-M2",
        "O2-M1",
        "O1-M2",
        # Concatenated (BOGN)
        "C4M1",
        "C3M2",
        "F4M1",
        "F3M2",
        "O2M1",
        "O1M2",
        # Bare + differential pair (MSNF, MSQW, MSTH, MSTR, STNF)
        ("C4", "M1"),
        ("C3", "M2"),
        ("F4", "M1"),
        ("F3", "M2"),
        ("O2", "M1"),
        ("O1", "M2"),
        # Extended 10-20 (Grael sites)
        "EEG_T4-A1",
        "EEG_T3-A2",
        "EEG_P4-A1",
        "EEG_P3-A2",
        "EEG_Fp2-A1",
        "EEG_Fp1-A2",
    ],
    "EOG": [
        "EOG_LOC-A2",
        "EOG_ROC-A2",  # Grael
        "E1M2",
        "E2M2",  # BOGN concatenated
        "E1",
        "E2",  # MSTR, STLK, STNF
        "EOG1",
        "EOG2",  # STNF variant
        "LOC",
        "ROC",  # MSQW
        "L-EOG",
        "R-EOG",  # MSNF
        "E1_(LEOG)",
        "E2_(REOG)",  # MSTH
    ],
    "EMG": [
        "EMG_Chin",  # Grael
        "CHIN",  # BOGN
        "Chin",  # STLK, STNF
        "EMG1",  # MSTH, MSTR
        "CHIN1",  # MSQW
        "EMG_#1",  # MSNF
        "Chin2",  # STNF variant
        "EMG_Aux1",  # Grael aux
    ],
    "ECG": [
        "ECG_II",
        "ECG_I",  # Grael
        "ECG_IIHF",  # Grael HF
        "EKG",  # BOGN, MSQW
        "ECG1",
        "EKG1",  # MSTR, MSTH
        "ECG",
        "ECG_2",  # STNF
        "EKG_#1",  # MSNF
    ],
    "LEG": [
        "Leg_1",
        "Leg_2",  # Grael
        "RLEG",
        "LLEG",  # BOGN, MSQW
        "LAT1-LAT2",
        "RAT1-RAT2",  # STLK
        "RAT",
        "LAT",  # STNF
        "R-Leg1",
        "L-Leg1",  # MSTR, MSNF
        "R-LEG_1",
        "L-LEG_1",  # MSTH
    ],
}


class STAGESDataset(BasePhysioDataset):
    """STAGES multi-site clinical sleep dataset.

    Args:
        site: One of the 13 site codes (e.g. ``"BOGN"``) or ``None`` for
            all sites.
        recording: ``"first"`` (default) for baseline recordings only,
            ``"second"`` for repeat recordings (``_1`` suffix), or
            ``"all"`` for both.
        root: Path to the STAGES data root directory.
    """

    DATASET_NAME = "stages"
    DATASET_SUBDIR = "stages"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0
    CHANNEL_PREFERENCES = CHANNEL_PREFERENCES

    def __init__(
        self,
        site: Optional[str] = None,
        recording: str = "first",
        root: Optional[str] = None,
        **kwargs,
    ):
        if site is not None and site not in SITES:
            raise ValueError(f"site must be one of {SITES} or None; got {site!r}")
        if recording not in ("first", "second", "all"):
            raise ValueError(
                f"recording must be 'first', 'second', or 'all'; " f"got {recording!r}"
            )

        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)

        self.site = site
        self.recording = recording

        # Dynamic dataset name for cache separation
        name = "stages"
        if site is not None:
            name = f"stages_{site}"
        if recording != "first":
            name = f"{name}_{recording}"
        self.DATASET_NAME = name

        # Load metadata before super().__init__ triggers _list_subjects
        self._harmonized_meta, self._full_meta = self._load_metadata(root)

        # Cache for parse_stages_csv results to avoid double-parsing
        # (labels + events are extracted from the same parse call).
        self._stages_parse_cache: Dict[str, Tuple[np.ndarray, list]] = {}

        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(
        root: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Load harmonized and full metadata CSVs.

        Returns:
            ``(harmonized_dict, full_dict)`` each keyed by ``subject_code``.
        """
        datasets_dir = Path(root) / "datasets"

        harmonized: Dict[str, Dict[str, Any]] = {}
        harmonized_path = datasets_dir / "stages-harmonized-dataset-0.3.0.csv"
        if harmonized_path.exists():
            import pandas as pd

            df = pd.read_csv(harmonized_path)
            for _, row in df.iterrows():
                key = str(row["subject_code"]).strip()
                harmonized[key] = {
                    col: (None if pd.isna(val) else val) for col, val in row.items()
                }
        else:
            logger.warning(
                "STAGES harmonized metadata CSV not found: %s",
                harmonized_path,
            )

        full: Dict[str, Dict[str, Any]] = {}
        full_path = datasets_dir / "stages-dataset-0.3.0.csv"
        if full_path.exists():
            import pandas as pd

            df = pd.read_csv(full_path, low_memory=False)
            for _, row in df.iterrows():
                key = str(row["subject_code"]).strip()
                full[key] = {
                    col: (None if pd.isna(val) else val) for col, val in row.items()
                }
        else:
            logger.warning("STAGES full metadata CSV not found: %s", full_path)

        return harmonized, full

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        psg_root = Path(self.root) / "original" / "STAGES PSGs"
        if not psg_root.exists():
            return []

        active_sites = [self.site] if self.site is not None else SITES

        subjects: List[SubjectSpec] = []
        for site_code in active_sites:
            site_dir = psg_root / site_code
            if not site_dir.exists():
                continue

            for edf_path in sorted(site_dir.glob("*.edf")):
                stem = edf_path.stem  # e.g. "GSBB00003" or "GSBB00003_1"
                csv_path = edf_path.with_suffix(".csv")

                # Skip EDFs without matching CSV
                if not csv_path.exists():
                    continue

                is_repeat = stem.endswith("_1")

                # Filter by recording parameter
                if self.recording == "first" and is_repeat:
                    continue
                if self.recording == "second" and not is_repeat:
                    continue

                # Build external metadata from harmonized + full CSVs.
                # The metadata key is the subject_code (same as stem).
                meta: Dict[str, Any] = {"site": site_code}
                if stem in self._harmonized_meta:
                    meta.update(self._harmonized_meta[stem])
                elif not is_repeat:
                    # For non-repeat, metadata might be keyed without suffix
                    pass
                if is_repeat:
                    # Repeat recordings: metadata is keyed by base ID
                    base_id = stem[:-2]  # strip "_1"
                    if base_id in self._harmonized_meta:
                        meta.update(self._harmonized_meta[base_id])
                    if base_id in self._full_meta:
                        meta.update(self._full_meta[base_id])
                else:
                    if stem in self._full_meta:
                        meta.update(self._full_meta[stem])

                subjects.append(
                    SubjectSpec(
                        subject_id=stem,
                        edf_path=edf_path,
                        label_path=csv_path,
                        external_meta=meta,
                    )
                )

        logger.info(
            "[%s] Discovered %d subjects (site=%s, recording=%s)",
            self.DATASET_NAME,
            len(subjects),
            self.site,
            self.recording,
        )
        return subjects

    # ------------------------------------------------------------------
    # Label + event reading
    # ------------------------------------------------------------------

    def _get_parse_result(self, spec: SubjectSpec) -> Tuple[np.ndarray, list]:
        """Return cached (labels, events) from parse_stages_csv."""
        if spec.subject_id not in self._stages_parse_cache:
            labels, events = parse_stages_csv(
                spec.label_path,
                epoch_length_sec=self.epoch_length_sec,
                stage_map=self.stage_map,
            )
            self._stages_parse_cache[spec.subject_id] = (labels, events)
        return self._stages_parse_cache[spec.subject_id]

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Return per-epoch AASM labels parsed from the STAGES CSV."""
        labels, _ = self._get_parse_result(spec)
        return labels

    def _read_subject_events(self, spec: SubjectSpec) -> list:
        """Return event list parsed from the STAGES CSV."""
        _, events = self._get_parse_result(spec)
        return events

    # ------------------------------------------------------------------
    # Splits: coherent across first/second recordings
    # ------------------------------------------------------------------

    def get_splits(self, fold: int = 0) -> Tuple[List[str], List[str], List[str]]:
        """Return train/valid/test subject_id lists.

        Splits are coherent across first/second recordings: the base
        subject ID (without ``_1`` suffix) determines which split a
        subject belongs to, so the same person always ends up in the
        same split regardless of whether this instance includes first,
        second, or all recordings.
        """
        # Collect ALL base subject IDs from the filesystem (not just
        # the ones in this instance) to ensure split stability.
        psg_root = Path(self.root) / "original" / "STAGES PSGs"
        active_sites = [self.site] if self.site is not None else SITES

        base_ids_set: set = set()
        for site_code in active_sites:
            site_dir = psg_root / site_code
            if not site_dir.exists():
                continue
            for edf_path in site_dir.glob("*.edf"):
                stem = edf_path.stem
                csv_path = edf_path.with_suffix(".csv")
                if not csv_path.exists():
                    continue
                # Strip _1 suffix to get base ID
                base_id = stem[:-2] if stem.endswith("_1") else stem
                base_ids_set.add(base_id)

        base_ids = sorted(base_ids_set)
        rng = random.Random(42 + int(fold))
        rng.shuffle(base_ids)

        n = len(base_ids)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)
        train_base = set(base_ids[:n_train])
        valid_base = set(base_ids[n_train : n_train + n_valid])
        test_base = set(base_ids[n_train + n_valid :])

        # Map back to actual subject_ids present in this instance
        actual_ids = {s.subject_id for s in self._subjects}

        def _map_to_actual(base_set: set) -> List[str]:
            result: List[str] = []
            for base_id in sorted(base_set):
                if base_id in actual_ids:
                    result.append(base_id)
                if f"{base_id}_1" in actual_ids:
                    result.append(f"{base_id}_1")
            return result

        return (
            _map_to_actual(train_base),
            _map_to_actual(valid_base),
            _map_to_actual(test_base),
        )
