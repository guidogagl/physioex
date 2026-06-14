"""Alzheimer's disease dataset (UZ Leuven).

69 subjects: 37 AD + 32 HC (healthy controls), prefix-encoded (AD* vs HC*).
Each subject has one night recording with average-reference EEG montage,
differential EOG, chin EMG, and ECG -- all at 200 Hz, 30s epochs.

Annotations are TSV files with 7-8 comment header lines (starting with ``#``),
followed by tab-separated ``onset\\tend\\tstage`` rows.
Stage vocabulary: Wake, S1, S2, S3, REM.

The ``subset`` parameter allows selecting only AD or HC subjects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root


# AASM 5-class mapping from TSV stage strings.
_STAGE_MAP: Dict[str, int] = {
    "Wake": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "REM": 4,
}


class AlzheimersDataset(BasePhysioDataset):
    """Alzheimer's disease sleep dataset (UZ Leuven).

    Args:
        subset: ``"AD"`` for Alzheimer's patients only, ``"HC"`` for healthy
            controls only, or ``None`` for all subjects.
        root: path to the ``AlzheimerData`` directory.
    """

    DATASET_NAME = "alzheimers"
    DATASET_SUBDIR = "AlzheimerData"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    # Channel preference lists.
    # Raw EDF uses average reference; re-reference to pseudo-mastoid
    # via T9/T10 (≈ M1/M2) to match AASM-standard C4-M1 montage
    # used by pre-trained sleep staging models.
    # Two hardware variants: 33 subjects have EMG1/EMG2, 36 have EMG Chin.
    CHANNEL_PREFERENCES = {
        "EEG": [
            ("EEG C4-REF", "EEG T9-REF"),   # C4-T9 ≈ C4-M1
            ("EEG C3-REF", "EEG T10-REF"),  # C3-T10 ≈ C3-M2
            ("EEG F4-REF", "EEG T9-REF"),
            ("EEG F3-REF", "EEG T10-REF"),
            ("EEG O2-REF", "EEG T9-REF"),
            ("EEG O1-REF", "EEG T10-REF"),
        ],
        "EOG": [
            ("EEG EOG1-REF", "EEG EOG2-REF"),
        ],
        "EMG": [
            "EMG Chin",  # 36 subjects (newer hardware)
            ("EMG1", "EMG2"),  # 33 subjects (older hardware)
        ],
        "ECG": [
            "ECG V1",
            "ECG",
        ],
    }

    def __init__(
        self,
        subset: Optional[str] = None,
        root: Optional[str] = None,
        **kwargs,
    ):
        if subset is not None and subset not in ("AD", "HC"):
            raise ValueError(f"subset must be 'AD', 'HC', or None; got {subset!r}")
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        # Store before super().__init__ which calls _list_subjects.
        self._subset_filter = subset
        self._label_offset: Dict[str, int] = {}
        if subset:
            self.DATASET_NAME = f"alzheimers_{subset.lower()}"
        super().__init__(root=root, **kwargs)

        # The label array now has the correct onset-based alignment,
        # but epochs 0..first_annotation are all -1 (pre-sleep).
        # Trim the dataset range to only the annotated region so
        # training doesn't waste time on unannotated daytime epochs.
        for spec in self._subjects:
            labels = self._get_labels(spec)
            scored_mask = labels >= 0
            if scored_mask.any():
                first = int(np.argmax(scored_mask))
                last = len(labels) - 1 - int(np.argmax(scored_mask[::-1]))
                # Keep 30 min of context before/after annotations
                keep_epochs = int(30 * 60 / self.epoch_length_sec)  # 60 epochs
                start = max(0, first - keep_epochs)
                end = min(len(labels), last + 1 + keep_epochs)
                self._n_epochs[spec.subject_id] = end
                self._label_offset[spec.subject_id] = start
            # else: keep base class defaults

        # Recompute flat index with trimmed ranges
        self._subject_ranges = []
        running = 0
        for spec in self._subjects:
            n = self._n_epochs[spec.subject_id]
            start = self._label_offset.get(spec.subject_id, 0)
            usable = n - start
            count = (usable - self.sequence_length + 1) if self.sequence_length > 0 else 1
            count = max(0, count)
            self._subject_ranges.append((spec.subject_id, running, running + count))
            running += count
        self._length = running

    # ------------------------------------------------------------------
    # Index mapping: shift flat indices by label offset
    # ------------------------------------------------------------------

    def _get_sequence_item(self, flat_idx: int):
        spec, local_offset = self._find_subject_for_flat_idx(flat_idx)
        # Shift by label_offset so index 0 maps to first annotated region
        epoch_start = local_offset + self._label_offset.get(spec.subject_id, 0)
        epoch_end = epoch_start + self.sequence_length
        return self._build_item(spec, epoch_start, epoch_end)

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        data_dir = Path(self.root) / "Data"
        if not data_dir.exists():
            return []

        subjects: List[SubjectSpec] = []
        for subj_dir in sorted(data_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj_id = subj_dir.name

            # Filter by subset prefix.
            if self._subset_filter == "AD" and not subj_id.startswith("AD"):
                continue
            if self._subset_filter == "HC" and not subj_id.startswith("HC"):
                continue

            edf = subj_dir / f"{subj_id}_r1.edf"
            tsv = subj_dir / f"{subj_id}_r1_a1.tsv"
            if not edf.exists() or not tsv.exists():
                continue

            subjects.append(
                SubjectSpec(
                    subject_id=subj_id,
                    edf_path=edf,
                    label_path=tsv,
                    external_meta={
                        "group": "AD" if subj_id.startswith("AD") else "HC",
                    },
                )
            )
        return subjects

    # ------------------------------------------------------------------
    # Label reading
    # ------------------------------------------------------------------

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse Alzheimer's TSV annotation file.

        Format: comment lines starting with ``#`` (7-8 lines), an optional
        blank line, then tab-separated ``onset\\tend\\tstage`` rows.  Onset
        and end are in seconds; each row spans one 30s epoch.

        Annotations do NOT start at the beginning of the EDF recording —
        they typically begin 8-13h into the 20-24h signal (sleep period).
        We use the ``onset`` column to place each label at the correct
        signal epoch index, padding unannotated epochs with ``-1``.
        """
        rows: List[tuple] = []
        with open(spec.label_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("onset"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                onset_sec = float(parts[0])
                stage_str = parts[2].strip()
                epoch_idx = int(onset_sec / self.epoch_length_sec)
                rows.append((epoch_idx, _STAGE_MAP.get(stage_str, -1)))

        if not rows:
            return np.array([], dtype=np.int16)

        # Build label array spanning from epoch 0 to the last annotated epoch.
        last_epoch = max(idx for idx, _ in rows)
        labels = np.full(last_epoch + 1, -1, dtype=np.int16)
        for epoch_idx, stage in rows:
            labels[epoch_idx] = stage

        return labels
