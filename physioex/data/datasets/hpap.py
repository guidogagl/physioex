"""HomePAP sleep dataset, NSRR format.

Contains three subsets (lab-full, lab-split, home) selectable via the
``subset`` parameter.

Layout on disk::

    <root>/
        polysomnography/
            edfs/
                lab/
                    full/
                        homepap-lab-full-XXXXXXX.edf
                    split/
                        homepap-lab-split-XXXXXXX.edf
                home/
                    homepap-home-XXXXXXX.edf
            annotations-events-nsrr/
                lab/
                    full/
                        homepap-lab-full-XXXXXXX-nsrr.xml
                    split/
                        homepap-lab-split-XXXXXXX-nsrr.xml
                home/
                    homepap-home-XXXXXXX-nsrr.xml
        datasets/
            homepap-baseline-harmonized-dataset-0.2.0.csv

Subject ID = EDF stem (e.g. ``homepap-lab-full-1600267``).
Annotations are NSRR XML parsed by ``parse_nsrr_xml``.
Subject metadata is loaded from the NSRR harmonized CSV keyed by ``nsrrid``.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physioex.data.base import SubjectSpec, get_data_root
from physioex.data.datasets._nsrr import _NSRRBaseDataset
from physioex.data.readers.edf import EDFHeader, ResolvedChannel

logger = logging.getLogger("physioex.data")


class HPAPDataset(_NSRRBaseDataset):
    """HomePAP sleep dataset with subset selection."""

    DATASET_NAME = "hpap"
    DATASET_SUBDIR = "homepap"

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [
            ("C4", "M1"),
            ("C3", "M2"),
            "C4-M1",
            "C3-M2",  # pre-referenced (lab-split, some lab-full)
            ("C4", "A1"),
            ("C3", "A2"),
            "C4-A1",
            "C3-A2",
            "C4",
            "C3",  # bare electrodes (no ref, e.g. 1600047)
            "EEG",
            "EEG1",
            "EEG2",
        ],
        "EOG": [
            ("E1", "E2"),  # dominant pair (212 subjects)
            ("E-1", "E-2"),  # hyphenated variant (lab-split/lab-full)
            "E1-E2",  # pre-referenced variant
            ("E1", "M2"),
            ("E2", "M1"),  # alternative derivations
            "E1-M2",
            "E2-M1",  # pre-referenced variants
            ("LOC", "ROC"),
            "LOC",
            "ROC",
            ("L-EOG", "R-EOG"),
            "L-EOG",
            "R-EOG",
            "EOG",
        ],
        "EMG": [
            ("Lchin", "Cchin"),  # standard chin EMG derivation (125+ subjects)
            ("LChin", "CChin"),
            ("LCHIN", "CCHIN"),
            ("LChin", "RChin"),  # LChin+RChin without CChin
            ("Lchin", "Rchin"),
            "Lchin-Cchin",  # pre-referenced variant
            "Chin1-Chin2",
            ("Chin1", "Chin2"),
            ("EMG1", "EMG2"),  # numbered EMG channels (some lab-full)
            "EMG1",
            "Chin",
            "CHIN",
            "Chin EMG",
            "EMG Chin",
            "EMG",
        ],
        "ECG": [
            ("ECG1", "ECG3"),  # standard ECG derivation (145 subjects)
            "ECG3-ECG1",  # pre-referenced variant (29 subjects)
            "ECG",
            "ECG1",
            "ECG2",
            ("EKG1", "EKG3"),
            "EKG",
            "EKG1",
        ],
    }

    # Subdirectory mapping per subset: (edf_subdir, xml_subdir)
    SUBSET_DIRS = {
        "lab-full": ("lab/full", "lab/full"),
        "lab-split": ("lab/split", "lab/split"),
        "home": ("home", "home"),
    }
    DEFAULT_SUBSET = "all"

    def __init__(
        self,
        root: Optional[str] = None,
        subset: str = "all",
        **kwargs,
    ):
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        # Store the HPAP subset choice *before* calling super().__init__,
        # which calls _list_subjects.  We use _hpap_subset to avoid
        # colliding with the base class's generic ``subset`` parameter.
        self._hpap_subset = subset
        self._metadata = self._load_metadata(root)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metadata(root: str) -> Dict[str, Dict[str, Any]]:
        """Load harmonized CSV into a dict keyed by nsrrid (as string)."""
        csv_path = (
            Path(root) / "datasets" / "homepap-baseline-harmonized-dataset-0.2.0.csv"
        )
        if not csv_path.exists():
            logger.warning("HPAP metadata CSV not found: %s", csv_path)
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
        edf_root = root / "polysomnography" / "edfs"
        xml_root = root / "polysomnography" / "annotations-events-nsrr"

        # Validate subset *before* checking existence so invalid values
        # always raise, even when the root directory is empty / missing.
        subset = self._hpap_subset or self.DEFAULT_SUBSET
        if subset == "all":
            subsets_to_enum = list(self.SUBSET_DIRS.keys())
        elif subset in self.SUBSET_DIRS:
            subsets_to_enum = [subset]
        else:
            raise ValueError(
                f"Unknown subset {subset!r}. "
                f"Available: {sorted(self.SUBSET_DIRS)} or 'all'"
            )

        if not edf_root.exists():
            return []

        specs: List[SubjectSpec] = []
        for sub in subsets_to_enum:
            edf_sub, xml_sub = self.SUBSET_DIRS[sub]
            specs.extend(
                self._pair_edfs_with_xml(
                    edf_root / edf_sub,
                    xml_root / xml_sub,
                )
            )

        # Enrich with metadata from harmonized CSV.
        # subject_id is e.g. "homepap-lab-full-1600001" -> nsrrid "1600001"
        for spec in specs:
            num_id = spec.subject_id.rsplit("-", 1)[-1]
            if num_id in self._metadata:
                spec.external_meta.update(self._metadata[num_id])

        return specs

    # ------------------------------------------------------------------
    # Subject-aware splitting
    # ------------------------------------------------------------------

    def get_splits(self, fold: int = 0):
        """Subject-aware 70/15/15 split grouping all subsets of the same person.

        The same nsrrid (numeric ID after the last hyphen in the subject_id)
        can appear in lab-full, lab-split, and home subsets.  All recordings
        of one person are guaranteed to land in the same split.
        """
        import random as _random
        from collections import defaultdict

        rng = _random.Random(42 + int(fold))

        # Group recording-level subject_ids by nsrrid
        groups: dict[str, list[str]] = defaultdict(list)
        for spec in self._subjects:
            nsrrid = spec.subject_id.rsplit("-", 1)[-1]
            groups[nsrrid].append(spec.subject_id)

        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        n = len(group_keys)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)

        train = [sid for g in group_keys[:n_train] for sid in groups[g]]
        valid = [
            sid for g in group_keys[n_train : n_train + n_valid] for sid in groups[g]
        ]
        test = [sid for g in group_keys[n_train + n_valid :] for sid in groups[g]]

        return train, valid, test

    # ------------------------------------------------------------------
    # MNE-based fallback for non-compliant EDFs (home subset)
    # ------------------------------------------------------------------
    # The home-subset EDFs have non-ASCII characters in the EDF Physical
    # Dimension field (degree symbol for position channels), which makes
    # pyedflib reject them.  We fall back to MNE-Python, which is more
    # tolerant.

    def _read_edf_header(self, spec: SubjectSpec) -> EDFHeader:
        """Try pyedflib first; fall back to MNE for non-compliant EDFs."""
        try:
            return super()._read_edf_header(spec)
        except OSError:
            return self._read_edf_header_mne(spec)

    @staticmethod
    def _read_edf_header_mne(spec: SubjectSpec) -> EDFHeader:
        """Read EDF header using MNE-Python (tolerant of minor spec violations)."""
        import mne

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(str(spec.edf_path), preload=False, verbose=False)

        labels = list(raw.ch_names)
        fs_map = {ch: float(raw.info["sfreq"]) for ch in labels}
        # MNE may report per-channel fs via annotations or channel info;
        # for EDF all channels in a data record share the record duration
        # but can differ in number of samples.  Use info dict when possible.
        try:
            for i, ch in enumerate(labels):
                ch_info = raw.info["chs"][i]
                # MNE stores calibrated units, but we can still get the
                # physical dimension string from the raw EDF header
                pass  # sfreq is already global; per-channel handled below
        except Exception:
            pass

        # Per-channel sample rates via _raw_extras (EDF-specific)
        try:
            extras = raw._raw_extras[0]
            if "n_samps" in extras:
                rec_dur = extras.get("record_length", [1.0])
                if hasattr(rec_dur, "__len__"):
                    rec_dur = rec_dur[0] if len(rec_dur) else 1.0
                rec_dur = float(rec_dur)
                n_samps = extras["n_samps"]
                for i, ch in enumerate(labels):
                    if i < len(n_samps):
                        fs_map[ch] = float(n_samps[i]) / rec_dur
        except Exception:
            pass

        units = {ch: "" for ch in labels}
        duration = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
        mtime = spec.edf_path.stat().st_mtime

        # Patient metadata
        patient: Dict = {
            "patient_code": None,
            "sex": None,
            "birthdate": None,
            "patient_name": None,
            "patient_additional": None,
        }
        try:
            subj = raw.info.get("subject_info", {}) or {}
            if subj.get("his_id"):
                patient["patient_code"] = str(subj["his_id"])
            sex_code = subj.get("sex", 0)
            patient["sex"] = {1: "M", 2: "F"}.get(sex_code)
            bd = subj.get("birthday")
            if bd is not None:
                patient["birthdate"] = (
                    bd.isoformat() if hasattr(bd, "isoformat") else str(bd)
                )
        except Exception:
            pass

        return EDFHeader(
            available_channels=labels,
            channel_fs=fs_map,
            channel_units=units,
            patient_meta=patient,
            duration_sec=duration,
            source_mtime=mtime,
        )

    def _read_subject_channel(
        self, spec: SubjectSpec, resolved: ResolvedChannel
    ) -> Tuple[np.ndarray, float]:
        """Try pyedflib first; fall back to MNE for non-compliant EDFs."""
        try:
            return super()._read_subject_channel(spec, resolved)
        except OSError:
            return self._read_subject_channel_mne(spec, resolved)

    @staticmethod
    def _read_subject_channel_mne(
        spec: SubjectSpec, resolved: ResolvedChannel
    ) -> Tuple[np.ndarray, float]:
        """Read a channel using MNE-Python as fallback."""
        import mne

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(str(spec.edf_path), preload=False, verbose=False)

        labels_upper = {ch.upper(): ch for ch in raw.ch_names}
        fs = resolved.fs_in

        if resolved.is_differential:
            phys_a, phys_b = resolved.physical
            ch_a = labels_upper.get(phys_a.upper(), phys_a)
            ch_b = labels_upper.get(phys_b.upper(), phys_b)
            raw_a = raw.copy().pick([ch_a]).load_data(verbose=False)
            raw_b = raw.copy().pick([ch_b]).load_data(verbose=False)
            sa = raw_a.get_data(verbose=False)[0].astype(np.float32)
            sb = raw_b.get_data(verbose=False)[0].astype(np.float32)
            m = min(sa.shape[0], sb.shape[0])
            return (sa[:m] - sb[:m]), fs
        else:
            phys = resolved.physical
            ch_name = labels_upper.get(phys.upper(), phys)
            picked = raw.copy().pick([ch_name]).load_data(verbose=False)
            sig = picked.get_data(verbose=False)[0].astype(np.float32)
            return sig, fs
