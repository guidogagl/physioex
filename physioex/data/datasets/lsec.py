"""LeuvenSleepEpilepsyChildren (LSEC): pediatric overnight video-EEG with manual sleep staging.

Recordings of children aged 4-18 years admitted for 24 h video-EEG, in three groups: children without epilepsy
(``Control``), with well-controlled epilepsy (``WCE``) and with drug-resistant epilepsy (``DRE``). Manual staging by one
scorer (AASM). Reference: Proost et al., *Automated sleep staging on reduced channels in children with epilepsy*,
Front. Neurol. 15:1390465 (2024), doi:10.3389/fneur.2024.1390465.

**The data are not distributed with physioex and are not public.** This module only reads a local copy that the user
obtained from the authors. Expected layout (one folder per group, folder names as received)::

    <root>/
        DRE/                          DRE_<n>.EDF   DRE_<n>_annotations.csv
        WCE/                          WCE_<n>.EDF   WCE_<n>_annotations.csv
        Patienten zonder epilepsie/   Control_<n>.EDF   Control_<n>_annotations.csv

Only ``<name>_<n>.EDF`` files are subjects; other EDF files of the folders (derived or reduced-montage exports such as
``*_SD.edf`` and ``*_CONTINUOUS.edf``) and acquisition-software sidecar files are ignored. A recording without its
annotation CSV is skipped.

Signals: 256 Hz, full 10-20 montage with ``A1``/``A2``, ``LOC``/``ROC``, ``EMG Chin`` (``EMG Li``/``EMG Re`` are leg
channels and are never used as chin) and ``ECG``. ``C3``/``C4`` are stored as separate channels, so the usual mastoid
derivations are formed by the differential-pair resolution: ``("C3", "A2")`` by default, ``("C4", "A1")`` with
``eeg_derivation="C4-A1"``. The EDF also holds technical channels (``CPU``, ``Mem``, ``Rec Cntr`` ...) that are never
requested.

Labels: UTF-16 text, one line per 30 s epoch, ``<epoch index><TAB><stage>`` with stage in ``WAKE|N1|N2|N3|REM``. The
index counts epochs from the start of the EDF and covers only the night (it does not start at 0). An unknown stage
raises. Epochs without a line are ``-1``.

Label alignment. The raw index is not always consistent with the signal: in some recordings the index runs past the end
of the EDF, and the best-matching shift between hypnogram and signal features is not zero. ``label_offset`` selects:

* ``"index"`` (default): the index is taken as is;
* ``"audited"``: each recording is shifted by ``offset_epochs`` read from ``label_offsets_path`` (CSV with the columns
  ``subject_id,offset_epochs``; a signal epoch is ``index + offset_epochs``). The offsets are derived from the data
  and are not part of the package; without the file ``"audited"`` raises.

In both modes labels outside the signal (before epoch 0 or past the end of the EDF) are dropped, never evaluated
against padding. The cache stores the raw labels only, so both modes share it.
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec, get_data_root

logger = logging.getLogger("physioex.data")

LSEC_STAGE_MAP: Dict[str, int] = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

#: folder name (as received) -> group label
LSEC_GROUPS: Dict[str, str] = {
    "DRE": "DRE",
    "WCE": "WCE",
    "Patienten zonder epilepsie": "Control",
}

_SUBJECT_EDF = re.compile(r"^(?P<stem>[A-Za-z]+_\d+)\.edf$", re.IGNORECASE)
# the received files contain one misspelt name (``annontations``): accept both spellings
_ANNOTATION_SUFFIXES = ("_annotations.csv", "_annontations.csv")

_EEG_PAIRS = {"C3-A2": ("C3", "A2"), "C4-A1": ("C4", "A1")}


def parse_lsec_labels(csv_path) -> np.ndarray:
    """Parse an annotation CSV into raw per-epoch labels (int16, ``-1`` where no line).

    The array has ``max(index) + 1`` entries: ``labels[i]`` is the stage of the epoch with index ``i``.
    """
    raw = Path(csv_path).read_bytes()
    try:
        text = raw.decode("utf-16")
    except UnicodeError:
        text = raw.decode("utf-8-sig")
    idx: List[int] = []
    stg: List[int] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) != 2:
            raise ValueError(f"{csv_path}: expected '<index><TAB><stage>', got {line!r}")
        if parts[1] not in LSEC_STAGE_MAP:
            raise ValueError(f"{csv_path}: unknown stage {parts[1]!r}")
        idx.append(int(parts[0]))
        stg.append(LSEC_STAGE_MAP[parts[1]])
    if not idx:
        return np.array([], dtype=np.int16)
    if min(idx) < 0:
        raise ValueError(f"{csv_path}: negative epoch index")
    labels = np.full(max(idx) + 1, -1, dtype=np.int16)
    labels[np.asarray(idx, dtype=np.int64)] = np.asarray(stg, dtype=np.int16)
    return labels


def shift_and_cut(labels: np.ndarray, offset: int, n_signal_epochs: int) -> np.ndarray:
    """Move ``labels[i]`` to position ``i + offset`` and cut to ``n_signal_epochs`` (``-1`` elsewhere)."""
    out = np.full(max(int(n_signal_epochs), 0), -1, dtype=np.int16)
    n = len(labels)
    if n == 0 or len(out) == 0:
        return out
    src0 = max(0, -offset)
    src1 = min(n, len(out) - offset)
    if src1 > src0:
        out[src0 + offset : src1 + offset] = labels[src0:src1]
    return out


def _read_offsets(path) -> Dict[str, int]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "subject_id" not in rows[0] or "offset_epochs" not in rows[0]:
        raise ValueError(f"{path}: needs the columns subject_id,offset_epochs")
    return {r["subject_id"]: int(float(r["offset_epochs"])) for r in rows}


def _edf_header_fields(edf_path) -> Dict[str, Any]:
    """Patient group of the raw EDF header: ``patient_code``, ``sex``, ``birthdate`` (ISO) and ``age_years``.

    Read from the first 256 bytes (the patient field of these files is not EDF+ compliant, so the generic header
    probe does not expose it). Anything missing is ``None``; nothing here is written to disk.
    """
    out: Dict[str, Any] = {"patient_code": None, "sex": None, "birthdate": None, "age_years": None}
    try:
        with open(edf_path, "rb") as f:
            h = f.read(256)
        patient = h[8:88].decode("ascii", "replace").split()
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", " ".join(patient))
        hashes = [p for p in patient if p.startswith("##")]
        out["patient_code"] = hashes[0] if hashes else None
        out["sex"] = next((p for p in patient if p in ("F", "M")), None)
        d, mo, y = (int(x) for x in h[168:176].decode("ascii").split("."))
        y += 2000 if y < 85 else 1900
        if m:
            from datetime import date

            birth = date(int(m[1]), int(m[2]), int(m[3]))
            out["birthdate"] = birth.isoformat()
            out["age_years"] = round((date(y, mo, d) - birth).days / 365.25, 2)
    except Exception:
        pass
    return out


class LeuvenSleepEpilepsyChildrenDataset(BasePhysioDataset):
    """Overnight video-EEG of children with and without epilepsy (see the module docstring)."""

    DATASET_NAME = "lsec"
    DATASET_SUBDIR = "LeuvenSleepEpilepsyChildren"
    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    CHANNEL_PREFERENCES: Dict[str, List] = {
        "EEG": [("C3", "A2"), ("C4", "A1"), "C3", "C4"],
        "EOG": [("LOC", "ROC"), "LOC", "ROC"],
        "EMG": ["EMG Chin"],
        "ECG": ["ECG"],
    }

    def __init__(
        self,
        root: Optional[str] = None,
        eeg_derivation: str = "C3-A2",
        label_offset: str = "index",
        label_offsets_path: Optional[str] = None,
        exclude_subjects: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        """
        Args:
            root: dataset root (default ``<data_root>/LeuvenSleepEpilepsyChildren``).
            eeg_derivation: ``"C3-A2"`` (default) or ``"C4-A1"``: the derivation a request for ``"EEG"`` resolves to.
            label_offset: ``"index"`` or ``"audited"`` (see the module docstring).
            label_offsets_path: CSV ``subject_id,offset_epochs``; required for ``label_offset="audited"``.
            exclude_subjects: subject ids (``<Group>_<n>``) to leave out. Nothing is excluded by default.
            subset: (``BasePhysioDataset`` argument) one group: ``"Control"``, ``"WCE"`` or ``"DRE"``.
        """
        if eeg_derivation not in _EEG_PAIRS:
            raise ValueError(f"eeg_derivation must be one of {sorted(_EEG_PAIRS)}, got {eeg_derivation!r}")
        if label_offset not in ("index", "audited"):
            raise ValueError(f"label_offset must be 'index' or 'audited', got {label_offset!r}")
        self.eeg_derivation = eeg_derivation
        self.label_offset = label_offset
        self._offsets: Dict[str, int] = {}
        if label_offset == "audited":
            if label_offsets_path is None:
                raise ValueError("label_offset='audited' needs label_offsets_path (CSV subject_id,offset_epochs)")
            self._offsets = _read_offsets(label_offsets_path)
        self._exclude = frozenset(exclude_subjects or ())
        # the requested derivation goes first; the instance attribute shadows the class default
        prefs = {k: list(v) for k, v in self.CHANNEL_PREFERENCES.items()}
        chosen = _EEG_PAIRS[eeg_derivation]
        prefs["EEG"] = [chosen] + [p for p in prefs["EEG"] if p != chosen]
        self.CHANNEL_PREFERENCES = prefs
        if root is None:
            root = str(get_data_root() / self.DATASET_SUBDIR)
        super().__init__(root=root, **kwargs)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _list_subjects(self) -> List[SubjectSpec]:
        root = Path(self.root)
        if not root.exists():
            return []
        specs: List[SubjectSpec] = []
        for folder, group in LSEC_GROUPS.items():
            gdir = root / folder
            if not gdir.is_dir():
                continue
            for edf in sorted(gdir.iterdir()):
                m = _SUBJECT_EDF.match(edf.name)
                if m is None or edf.suffix.lower() != ".edf":
                    continue
                stem = m.group("stem")
                csv_path = next(
                    (gdir / f"{stem}{suf}" for suf in _ANNOTATION_SUFFIXES if (gdir / f"{stem}{suf}").exists()), None
                )
                if csv_path is None or stem in self._exclude:
                    continue
                meta = {"group": group, **_edf_header_fields(edf)}
                specs.append(SubjectSpec(subject_id=stem, edf_path=edf, label_path=csv_path, external_meta=meta))
        if self.label_offset == "audited":
            # fail here: a label-read error in ``_build_index`` would only drop the subject with a warning
            missing = sorted(s.subject_id for s in specs if s.subject_id not in self._offsets)
            if missing:
                raise KeyError(f"no offset in label_offsets_path for {len(missing)} recording(s): {missing[:10]}")
        return specs

    def _filter_subset(self, subjects: List[SubjectSpec], subset: str) -> List[SubjectSpec]:
        groups = set(LSEC_GROUPS.values())
        if subset not in groups:
            raise ValueError(f"subset must be one of {sorted(groups)}, got {subset!r}")
        return [s for s in subjects if s.external_meta.get("group") == subset]

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Raw labels by epoch index (no shift, no cut): this is what the label cache stores."""
        return parse_lsec_labels(spec.label_path)

    def _get_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Cached raw labels -> shift (``audited`` mode) -> cut to the EDF -> ``_trim_excess_wake``."""
        sid = spec.subject_id
        if sid in self._labels_cache:
            return self._labels_cache[sid]
        raw = self._load_or_compute_labels(spec)
        offset = 0
        if self.label_offset == "audited":
            if sid not in self._offsets:
                raise KeyError(f"{sid}: no offset in label_offsets_path (label_offset='audited')")
            offset = self._offsets[sid]
        header = self._headers.get(sid)
        if header is not None:
            n_sig = int(header.duration_sec // self.epoch_length_sec)
            labels = shift_and_cut(raw, offset, n_sig)
            n_in, n_out = int((raw >= 0).sum()), int((labels >= 0).sum())
            if n_out != n_in:
                logger.info("[lsec] %s: %d of %d scored epochs fall outside the signal and are dropped", sid, n_in - n_out, n_in)
        else:
            labels = np.asarray(raw, dtype=np.int16)
        if self.trim_excess_wake:
            labels = self._trim_excess_wake(labels)
        self._labels_cache[sid] = labels
        return labels
