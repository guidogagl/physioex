"""MASS (Montreal Archive of Sleep Studies) dataset.

5 cohorts (SS01-SS05), selected via the ``cohort`` constructor parameter.
Raw data lives under ``MASS/Original/SS{cohort:02d}/`` with EDF signal files
and separate annotation sidecars.

Annotation formats (tried in order per cohort):
  1. EDF+ annotations (``_Annotations.edf`` for SS01/SS03, ``_Base.edf`` for
     SS02) -- standard embedded annotations readable via pyedflib.
  2. SAF text (``_saf.txt``) -- binary EDF annotation format in text with
     ``\\x15`` and ``\\x14`` delimiters.  Available for all cohorts and the
     ONLY source for SS04/SS05.

Epoch lengths differ per cohort (R&K scored cohorts use 20s page size):
  - SS01, SS03: 30-second epochs (AASM)
  - SS02, SS04, SS05: 20-second epochs (R&K)

For 20-second cohorts, Huy Phan's convention is applied: each 20s epoch
is expanded to a 30s signal window by padding ±5s of context from adjacent
signal, keeping the downstream model input always at 30s / 3000 samples
(at 100 Hz). The first and last epochs are dropped (no context available).
See: Phan et al., IEEE TNSRE 2019 (SeqSleepNet paper, PMC6481557).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root


# R&K / AASM stage mapping -- same convention as SleepEDF.
MASS_STAGE_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # AASM: N4 -> N3
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
}


class MASSDataset(BasePhysioDataset):
    """MASS (Montreal Archive of Sleep Studies) dataset.

    Has 5 cohorts (SS01-SS05). Each cohort is selected via the ``cohort``
    parameter.

    Args:
        cohort: int (1-5). Selects which cohort's recordings to load.
        root: path to the MASS data directory
              (default: ``$PHYSIOEX_DATA/MASS/Original``).
    """

    DATASET_NAME = "mass"
    DATASET_SUBDIR = "MASS/Original"

    # Epoch length varies by cohort (verified from on-disk annotation files).
    COHORT_EPOCH_SEC: Dict[int, float] = {
        1: 30.0,
        2: 20.0,
        3: 30.0,
        4: 20.0,
        5: 20.0,
    }

    # Channel preference lists for generic modality requests.
    # MASS uses two derivation types: -CLE (Contralateral Linked Ear, majority)
    # and -LER (Linked Ear Reference, ~6 subjects in SS01). Both must be listed.
    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [
            "EEG C3-CLE",
            "EEG C4-CLE",
            "EEG Cz-CLE",
            "EEG F3-CLE",
            "EEG F4-CLE",
            "EEG Fz-CLE",
            "EEG O1-CLE",
            "EEG O2-CLE",
            "EEG Pz-CLE",
            # LER (Linked Ear Reference) fallback for subjects without CLE
            "EEG C3-LER",
            "EEG C4-LER",
            "EEG Cz-LER",
            "EEG F3-LER",
            "EEG F4-LER",
            "EEG Fz-LER",
            "EEG O1-LER",
            "EEG O2-LER",
            "EEG Pz-LER",
        ],
        "EOG": [
            "EOG Left Horiz",
            "EOG Right Horiz",
            "EOG Upper Vertic",
            "EOG Lower Vertic",
        ],
        "EMG": [
            "EMG Chin1",
            "EMG Chin2",
            "EMG Chin3",
            "EMG Chin",
        ],
        "ECG": ["ECG I", "ECG", "ECG II", "ECG III"],
    }

    # EDF annotation file patterns to search per cohort (tried in order).
    # SS04/SS05 have only compressed .edf.gz annotations on disk, so we
    # fall back to _saf.txt exclusively.
    ANNOTATION_PATTERNS: Dict[int, List[str]] = {
        1: ["_Annotations.edf"],
        2: ["_Base.edf"],
        3: ["_Annotations.edf"],
        4: [],  # only _saf.txt
        5: [],  # only _saf.txt (EDF annotations not reliably available)
    }

    # Seconds of context to pad on each side of a 20s epoch to create
    # a 30s window (Phan convention).
    PAD_SEC: float = 5.0
    # Output epoch length (always 30s regardless of native annotation epoch)
    OUTPUT_EPOCH_SEC: float = 30.0

    def __init__(
        self,
        cohort: int = 1,
        root: Optional[str] = None,
        **kwargs,
    ):
        if cohort not in (1, 2, 3, 4, 5):
            raise ValueError(f"cohort must be 1-5; got {cohort}")
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        self.cohort = cohort
        self._is_20s = cohort in (2, 4, 5)
        # Dynamic dataset name so cache dirs are separated per cohort.
        self.DATASET_NAME = f"mass_ss{cohort:02d}"
        # For label parsing, keep native epoch_length_sec (20s or 30s).
        # Signal epoching is handled by our override of _epoch_and_run_pipeline.
        native_epoch = self.COHORT_EPOCH_SEC.get(cohort, 30.0)
        kwargs.setdefault("epoch_length_sec", native_epoch)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        cohort_dir = Path(self.root) / f"SS{self.cohort:02d}"
        if not cohort_dir.exists():
            return []

        ann_dir = cohort_dir / "annotations"
        subjects: List[SubjectSpec] = []

        for psg in sorted(cohort_dir.glob("*PSG.edf")):
            # Subject ID from filename: "01-01-0001 PSG.edf" -> "01-01-0001"
            subject_id = psg.stem.replace(" PSG", "")

            # Find annotation file
            label_path: Optional[Path] = None

            # Try EDF annotation patterns first
            for pattern in self.ANNOTATION_PATTERNS.get(self.cohort, []):
                candidate = ann_dir / f"{subject_id}{pattern}"
                if candidate.exists():
                    label_path = candidate
                    break

            # Fallback to _saf.txt
            if label_path is None:
                saf = ann_dir / f"{subject_id}_saf.txt"
                if saf.exists():
                    label_path = saf

            if label_path is None:
                continue

            subjects.append(
                SubjectSpec(
                    subject_id=subject_id,
                    edf_path=psg,
                    label_path=label_path,
                )
            )

        return subjects

    # ------------------------------------------------------------------
    # Label reading
    # ------------------------------------------------------------------

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse MASS annotation file, then drop first/last epoch for 20s
        cohorts (matching the ±5s signal padding that drops those epochs)."""
        path = spec.label_path
        suffix = str(path)
        if suffix.endswith(".edf"):
            labels = self._parse_annotation_edf(path)
        elif suffix.endswith("_saf.txt"):
            labels = self._parse_saf(path)
        else:
            return np.array([], dtype=np.int16)

        # For 20s cohorts, drop first and last epoch labels to align with
        # the signal windows produced by _epoch_and_run_pipeline (which
        # skips those epochs due to lack of ±5s padding context).
        if self._is_20s and labels.shape[0] >= 3:
            labels = labels[1:-1]

        return labels

    # ------------------------------------------------------------------
    # EDF+ annotation parser
    # ------------------------------------------------------------------

    def _parse_annotation_edf(self, path: Path) -> np.ndarray:
        """Parse EDF+ embedded annotations (same format as SleepEDF/HMC).

        Filters by ``"Sleep stage"`` prefix to extract only sleep stages,
        ignoring other events (arousals, PLMS, apneas, etc.).
        """
        import pyedflib

        with pyedflib.EdfReader(str(path)) as f:
            onsets, durations, labels = f.readAnnotations()

        # Filter only sleep stage annotations
        stages_data = [
            (float(o), float(d), str(l))
            for o, d, l in zip(onsets, durations, labels)
            if "Sleep stage" in str(l)
        ]

        if not stages_data:
            return np.array([], dtype=np.int16)

        total = max(o + d for o, d, _ in stages_data)
        n_epochs = int(total // self.epoch_length_sec)
        labels_arr = np.full(n_epochs, -1, dtype=np.int16)

        for onset, duration, stage_str in stages_data:
            stage = MASS_STAGE_MAP.get(stage_str.strip(), -1)
            i0 = int(round(onset / self.epoch_length_sec))
            i1 = int(round((onset + duration) / self.epoch_length_sec))
            if i1 > n_epochs:
                i1 = n_epochs
            labels_arr[i0:i1] = stage

        return labels_arr

    # ------------------------------------------------------------------
    # SAF text parser
    # ------------------------------------------------------------------

    def _parse_saf(self, path: Path) -> np.ndarray:
        r"""Parse Compumedics SAF text file.

        Format: each line is ``{onset}\x15{duration}\x14{stage_name}\x14\x00``
        where ``\x15`` (NAK) and ``\x14`` (DC4) are EDF annotation delimiters.
        """
        with open(path, encoding="latin1") as f:
            raw = f.read()

        stages_data = []
        for line in raw.split("\n"):
            line = line.strip("\x00").strip()
            if not line or "Sleep stage" not in line:
                continue
            # Split by \x14 (DC4) to get fields
            parts = line.split("\x14")
            if len(parts) < 2:
                continue
            # First part: "onset\x15duration"
            time_part = parts[0]
            stage_name = parts[1].strip()

            if "\x15" not in time_part:
                continue
            onset_str, dur_str = time_part.split("\x15", 1)

            try:
                onset = float(onset_str)
                duration = float(dur_str)
            except ValueError:
                continue

            stage = MASS_STAGE_MAP.get(stage_name, -1)
            stages_data.append((onset, duration, stage))

        if not stages_data:
            return np.array([], dtype=np.int16)

        total = max(o + d for o, d, _ in stages_data)
        n_epochs = int(total // self.epoch_length_sec)
        labels_arr = np.full(n_epochs, -1, dtype=np.int16)

        for onset, duration, stage in stages_data:
            i0 = int(round(onset / self.epoch_length_sec))
            i1 = int(round((onset + duration) / self.epoch_length_sec))
            if i1 > n_epochs:
                i1 = n_epochs
            labels_arr[i0:i1] = stage

        return labels_arr

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def _read_subject_events(self, spec):
        """Extract non-stage events from MASS annotation files.

        For EDF annotation files, reads all annotations and filters out
        sleep stage entries. Remaining annotations (arousals, PLMs,
        apneas, etc.) are converted to SleepEvent objects.
        For SAF text files, similarly filters non-stage entries.
        """
        from physioex.data.events import SleepEvent

        path = spec.label_path
        if path is None:
            return []

        suffix = str(path)
        if suffix.endswith(".edf"):
            return self._parse_annotation_edf_events(path)
        elif suffix.endswith("_saf.txt"):
            return self._parse_saf_events(path)
        return []

    def _parse_annotation_edf_events(self, path: Path):
        """Parse non-stage events from an EDF+ annotation file."""
        from physioex.data.events import SleepEvent

        _CATEGORY_MAP = {
            "microarousal": "arousal",
            "arousal": "arousal",
            "plms": "limb_movement",
            "plm": "limb_movement",
            "leg movement": "limb_movement",
            "limb movement": "limb_movement",
            "obstructiveapnea": "respiratory",
            "obstructive apnea": "respiratory",
            "centralapnea": "respiratory",
            "central apnea": "respiratory",
            "mixedapnea": "respiratory",
            "mixed apnea": "respiratory",
            "hypopnea": "respiratory",
            "apnea": "respiratory",
            "desaturation": "desaturation",
        }

        import pyedflib

        with pyedflib.EdfReader(str(path)) as f:
            onsets, durations, labels = f.readAnnotations()

        events = []
        for onset, duration, label in zip(onsets, durations, labels):
            label_str = str(label).strip()
            # Skip sleep stage annotations
            if label_str.startswith("Sleep stage"):
                continue
            # Skip empty annotations
            if not label_str:
                continue

            # Determine category from annotation text
            label_lower = label_str.lower().replace(" ", "")
            category = "other"
            for key, cat in _CATEGORY_MAP.items():
                if key.replace(" ", "") in label_lower:
                    category = cat
                    break

            events.append(
                SleepEvent(
                    type=category,
                    concept=label_str,
                    onset_sec=float(onset),
                    duration_sec=float(duration),
                )
            )

        return events

    def _parse_saf_events(self, path: Path):
        r"""Parse non-stage events from a SAF text file.

        Format: each line is ``{onset}\x15{duration}\x14{annotation}\x14\x00``
        """
        from physioex.data.events import SleepEvent

        _CATEGORY_MAP = {
            "microarousal": "arousal",
            "arousal": "arousal",
            "plms": "limb_movement",
            "plm": "limb_movement",
            "leg movement": "limb_movement",
            "limb movement": "limb_movement",
            "obstructiveapnea": "respiratory",
            "obstructive apnea": "respiratory",
            "centralapnea": "respiratory",
            "central apnea": "respiratory",
            "mixedapnea": "respiratory",
            "mixed apnea": "respiratory",
            "hypopnea": "respiratory",
            "apnea": "respiratory",
            "desaturation": "desaturation",
        }

        with open(path, encoding="latin1") as f:
            raw = f.read()

        events = []
        for line in raw.split("\n"):
            line = line.strip("\x00").strip()
            if not line:
                continue
            # Skip sleep stage annotations
            if "Sleep stage" in line:
                continue

            parts = line.split("\x14")
            if len(parts) < 2:
                continue
            time_part = parts[0]
            annotation = parts[1].strip()

            if not annotation or "\x15" not in time_part:
                continue

            onset_str, dur_str = time_part.split("\x15", 1)
            try:
                onset = float(onset_str)
                duration = float(dur_str)
            except ValueError:
                continue

            # Determine category
            ann_lower = annotation.lower().replace(" ", "")
            category = "other"
            for key, cat in _CATEGORY_MAP.items():
                if key.replace(" ", "") in ann_lower:
                    category = cat
                    break

            events.append(
                SleepEvent(
                    type=category,
                    concept=annotation,
                    onset_sec=onset,
                    duration_sec=duration,
                )
            )

        return events

    # ------------------------------------------------------------------
    # 20s → 30s epoch expansion (Phan / SeqSleepNet convention)
    # ------------------------------------------------------------------

    def _epoch_and_run_pipeline(self, raw, fs_in, compiled, resolved):
        """Override base-class epoching for 20-second cohorts.

        For 30s cohorts (SS01, SS03): delegates to the standard base-class
        implementation unchanged.

        For 20s cohorts (SS02, SS04, SS05): applies Phan's ±5s padding.
        Each 20s annotation window is expanded to a 30s signal window by
        extracting ``[onset - 5s, onset + 25s]`` from the continuous
        (already filtered + resampled) signal. The first and last 20s
        epochs are dropped (no context). The corresponding label entries
        are also dropped in ``_read_subject_labels``.

        Reference: Phan et al., IEEE TNSRE 2019 (SeqSleepNet),
        ``preprare_raw_data.m``.
        """
        if not self._is_20s:
            return super()._epoch_and_run_pipeline(
                raw,
                fs_in,
                compiled,
                resolved,
            )

        # --- Run non-domain-changing steps on full 1D signal
        x = np.asarray(raw, dtype=np.float32)
        fs = fs_in
        domain_step = None

        for step in compiled.steps:
            if step.fs_out == 0:
                domain_step = step
                break
            x = step.apply(x)
            fs = step.fs_out

        # --- Extract ±5s padded 30s windows around each 20s epoch
        sps_20 = int(round(20.0 * fs))
        sps_30 = int(round(self.OUTPUT_EPOCH_SEC * fs))
        pad_sps = int(round(self.PAD_SEC * fs))

        total_samples = x.shape[-1]
        n_20_epochs = total_samples // sps_20

        windows = []
        for i in range(1, n_20_epochs - 1):
            start = i * sps_20 - pad_sps
            end = start + sps_30
            if start < 0 or end > total_samples:
                continue
            windows.append(x[..., start:end])

        if not windows:
            return np.zeros((0, sps_30), dtype=np.float32)

        epoched = np.stack(windows, axis=0)

        # --- Apply domain-changing step (spectrogram) if present
        if domain_step is not None:
            epoched = domain_step.apply(epoched)

        return np.ascontiguousarray(epoched)
