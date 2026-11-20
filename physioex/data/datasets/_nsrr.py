"""Shared helpers for NSRR-format datasets (EDF + NSRR XML annotations).

All three NSRR-family datasets (MESA, MrOS, HomePAP) share the same annotation
format: ``<stem>-nsrr.xml`` files parsed by ``parse_nsrr_xml``.  This private
base class factors out subject enumeration (EDF/XML pairing) and label reading
so each concrete dataset only needs to define channel preferences, default root,
and ``_list_subjects`` (which calls ``_pair_edfs_with_xml`` with the right paths).
"""
from __future__ import annotations

import logging
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec
from physioex.data.readers.annotations import parse_nsrr_xml

logger = logging.getLogger("physioex.data")


class _NSRRBaseDataset(BasePhysioDataset):
    """Common behaviour: label parsing + subject enumeration for NSRR layouts.

    Subclasses configure the metadata and session-id mapping via the class
    attributes below (all have conservative defaults that leave the feature
    inert):

    - ``HARMONIZED_CSV_GLOB``: glob (relative to ``<root>/datasets``) for the
      NSRR harmonized dataset CSV. ``None`` disables ``subject_metadata()``.
    - ``CSV_ID_COLUMN``: column whose value identifies a subject row.
    - ``DATASET_PREFIX``: the session-id prefix that identifies this dataset
      (e.g. ``"ccshs-trec-``), used by ``get_splits(split="wav2sleep-census")``
      to pick out the ids that belong to this dataset.
    """

    DEFAULT_EPOCH_LENGTH_SEC = 30.0

    #: Prefix of the session ids that identifies this dataset (census splits).
    DATASET_PREFIX: Optional[str] = None

    #: Glob (relative to ``<root>/datasets``) for the NSRR harmonized CSV.
    HARMONIZED_CSV_GLOB: Optional[str] = None
    #: CSV column whose value identifies a subject row (e.g. ``"nsrrid"``).
    CSV_ID_COLUMN: Optional[str] = None

    # ------------------------------------------------------------------
    # Session ids (as distributed by NSRR == the EDF stem)
    # ------------------------------------------------------------------

    def session_id(self, subject_spec_or_subject_id: Any) -> str:
        """Return the NSRR session id (the EDF stem as distributed by NSRR).

        For the NSRR datasets the subject_id IS the EDF stem, so this is the
        identity mapping (e.g. ``"shhs1-200121"``, ``"mesa-sleep-0036"``,
        ``"mros-visit1-aa0001"``, ``"cfs-visit5-800002"``).  Datasets whose
        subject_id differs from the EDF stem (WSC drops the trailing
        ``-nsrr``) override this method.
        """
        if isinstance(subject_spec_or_subject_id, str):
            return subject_spec_or_subject_id
        return subject_spec_or_subject_id.subject_id

    def _nsrrid_of(self, subject_id: str) -> str:
        """Extract the NSRR id from a subject_id (used for the CSV lookup).

        Default: the part after the dataset's fixed session stem; datasets
        with variable stems (CHAT) or non-numeric ids override this.
        """
        return subject_id.rsplit("-", 1)[-1]

    # ------------------------------------------------------------------
    # Subject metadata (harmonized CSV + EDF header)
    # ------------------------------------------------------------------

    @classmethod
    def load_harmonized_csv(cls, root: str) -> Dict[str, Dict[str, Any]]:
        """Load the NSRR harmonized dataset CSV into a dict keyed by row id.

        The CSV is found by ``HARMONIZED_CSV_GLOB`` under ``<root>/datasets``
        (newest version wins) and keyed by ``CSV_ID_COLUMN``.  Returns ``{}``
        (with a warning) when the file is absent, so datasets that do not
        define a glob keep working unchanged.
        """
        glob = cls.HARMONIZED_CSV_GLOB
        id_col = cls.CSV_ID_COLUMN
        if not glob or not id_col:
            return {}
        found = sorted((Path(root) / "datasets").glob(glob))
        if not found:
            logger.warning(
                "%s metadata CSV not found under %s (looking for %s)",
                cls.__name__,
                Path(root) / "datasets",
                glob,
            )
            return {}
        csv_path = found[-1]
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)
        if id_col not in df.columns:
            logger.warning(
                "%s: column %r missing from %s; subject metadata will be None",
                cls.__name__,
                id_col,
                csv_path,
            )
            return {}
        meta: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(row[id_col])
            meta[key] = {c: (None if pd.isna(v) else v) for c, v in row.items()}
        return meta

    def _row_key_of(self, subject_id: str) -> Optional[str]:
        """Key of the harmonized CSV row matching ``subject_id`` (default: the NSRR id).

        Datasets whose id column carries more than the NSRR id (e.g. a visit
        number) override this.
        """
        return self._nsrrid_of(subject_id)

    def _subject_row(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Return the harmonized-CSV row for ``subject_id`` (or ``None``).

        Tries the dataset-specific row key first, then the full subject_id
        (covers datasets whose ``_metadata`` is keyed by the complete id).
        """
        meta = getattr(self, "_metadata", None)
        if not meta:
            return None
        key = self._row_key_of(subject_id)
        row = meta.get(key, meta.get(subject_id))
        if row is None:
            logger.debug(
                "[%s] subject_metadata: no row in the harmonized CSV for key %r",
                self.DATASET_NAME,
                key,
            )
        return row

    @staticmethod
    def _harmonized_age_sex(
        row: Optional[Dict[str, Any]]
    ) -> "tuple[Optional[float], Optional[str]]":
        """Extract ``(age_years, sex)`` from a harmonized CSV row.

        Uses the NSRR standard columns ``nsrr_age`` / ``nsrr_sex``; a missing
        column or value yields ``None``.
        """
        if not row:
            return None, None
        age = row.get("nsrr_age")
        try:
            age = float(age) if age is not None and not pd_isna(age) else None
        except (TypeError, ValueError):
            age = None
        sex = row.get("nsrr_sex")
        if sex is None or (isinstance(sex, float) and pd_isna(sex)):
            sex = None
        elif isinstance(sex, str):
            s = sex.strip().upper()
            if s[:1] == "M":
                sex = "M"
            elif s[:1] == "F":
                sex = "F"
            else:
                sex = None
        else:
            sex = None
        return age, sex

    def subject_metadata(self, subject_id: str) -> Dict[str, Any]:
        """Return ``{"age": float|None, "sex": "M"|"F"|None, "recording_start": time|None}``.

        Age and sex come from the NSRR harmonized dataset CSV
        (``nsrr_age`` / ``nsrr_sex``); ``recording_start`` is the clock time of
        the EDF header startdatetime. A missing file or value yields ``None``
        (with a debug log); this never raises.
        """
        age, sex = self._harmonized_age_sex(self._subject_row(subject_id))
        return {
            "age": age,
            "sex": sex,
            "recording_start": self._recording_start(subject_id),
        }

    def _recording_start(self, subject_id: str) -> Optional[time]:
        """Clock time (``datetime.time``) of the EDF header startdatetime."""
        header = getattr(self, "_headers", {}).get(subject_id)
        if header is not None and getattr(header, "start_sec", None):
            sec = int(float(header.start_sec) % 86400)
            return time(sec // 3600, (sec % 3600) // 60, sec % 60)
        # Header not probed (e.g. metadata queried before init) -- probe it.
        try:
            for spec in getattr(self, "_subjects", []):
                if spec.subject_id != subject_id:
                    continue
                import pyedflib

                with pyedflib.EdfReader(str(spec.edf_path)) as f:
                    dt = f.getStartdatetime()
                if dt is None:
                    return None
                return time(dt.hour, dt.minute, dt.second)
        except Exception as exc:
            logger.debug(
                "[%s] recording_start: could not read EDF header of %r: %s",
                self.DATASET_NAME,
                subject_id,
                exc,
            )
        return None

    # ------------------------------------------------------------------
    # EDF header: record the startdatetime for subject_metadata()
    # ------------------------------------------------------------------

    def _read_edf_header(self, spec: SubjectSpec):
        """Probe the EDF header and remember the startdatetime as seconds-of-day
        (so ``subject_metadata()`` can report ``recording_start`` without
        re-reading the file)."""
        from physioex.data.readers.edf import probe_edf_header

        header = probe_edf_header(spec.edf_path)
        try:
            import pyedflib

            with pyedflib.EdfReader(str(spec.edf_path)) as f:
                dt = f.getStartdatetime()
            if dt is not None:
                header.start_sec = float(
                    dt.hour * 3600 + dt.minute * 60 + dt.second
                )
        except Exception as exc:
            logger.debug(
                "[%s] %s: could not read the EDF startdatetime: %s",
                self.DATASET_NAME,
                spec.subject_id,
                exc,
            )
        return header

    def _pair_edfs_with_xml(self, edf_dir: Path, xml_dir: Path) -> List[SubjectSpec]:
        """Given parallel EDF + XML directories, yield matched SubjectSpec entries.

        Tries ``{stem}-nsrr.xml`` first, then ``{stem}.xml`` as a fallback.
        """
        if not edf_dir.exists():
            return []
        specs: List[SubjectSpec] = []
        for edf in sorted(edf_dir.glob("*.edf")):
            stem = edf.stem
            # Try both naming conventions
            xml = xml_dir / f"{stem}-nsrr.xml"
            if not xml.exists():
                xml = xml_dir / f"{stem}.xml"
            if not xml.exists():
                continue
            specs.append(
                SubjectSpec(
                    subject_id=stem,
                    edf_path=edf,
                    label_path=xml,
                )
            )
        return specs

    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Parse NSRR XML annotations into per-epoch labels."""
        return parse_nsrr_xml(
            spec.label_path,
            epoch_length_sec=self.epoch_length_sec,
        )

    def _read_subject_events(self, spec):
        """Parse non-stage events from the NSRR XML annotation file."""
        from physioex.data.readers.annotations import parse_nsrr_xml_events

        if spec.label_path is None:
            return []
        return parse_nsrr_xml_events(spec.label_path)

    # ------------------------------------------------------------------
    # Named external splits (wav2sleep)
    # ------------------------------------------------------------------

    #: Key under ``_splits/wav2sleep/`` holding this dataset's split lists.
    #: Defaults to the registry name; SHHS overrides it (its DATASET_NAME is
    #: ``shhs_visit<N>`` but the splits live in the shared ``shhs`` directory).
    DATASET_SPLIT_KEY: Optional[str] = None

    def get_splits(self, fold: int = 0, split: Optional[str] = None):
        """Return ``(train, valid, test)`` subject_id lists.

        ``split`` names a published external split:

        - ``"wav2sleep"``: the train/val/test session lists of Carter &
          Tarassenko (2024), shipped in
          ``_splits/wav2sleep/<dataset>/{train,val,test}.txt``.  Only subjects
          present in this dataset instance are returned; ids missing locally
          are counted and logged once at info level.  Datasets without a
          train/val list (e.g. MrOS, test-only) get empty train/valid.
        - ``"wav2sleep-census"``: the census-balanced lists of Jones et al.
          (2024); ids from several datasets, only those belonging to this
          one are returned (no train list -> train is empty).
        - ``None``: the dataset's own splitting behaviour (default random
          70/15/15, or a fixed benchmark split).
        """
        if split is None:
            return super().get_splits(fold=fold)
        if not split.startswith("wav2sleep"):
            raise ValueError(
                f"Unknown split {split!r}; expected 'wav2sleep' or 'wav2sleep-census'"
            )
        from physioex.data.datasets._splits import load_wav2sleep_split

        census = split.endswith("-census")
        split_key = self.DATASET_SPLIT_KEY or self.DATASET_NAME
        present = {s.subject_id for s in self._subjects}
        out = []
        for part in ("train", "val", "test"):
            ids = load_wav2sleep_split(split_key, part, census=census)
            if census:
                ids = [i for i in ids if i.startswith(self.DATASET_PREFIX)]
            kept = [i for i in ids if i in present]
            missing = len(ids) - len(kept)
            if missing:
                logger.info(
                    "[%s] wav2sleep %s: %d of %d %s ids not present locally; dropped",
                    self.DATASET_NAME,
                    "census" if census else f"fold {fold}",
                    missing,
                    len(ids),
                    part,
                )
            out.append(kept)
        return out[0], out[1], out[2]


def pd_isna(val) -> bool:
    """True for pandas NaN/NaT (and plain None); no pandas import at module level."""
    if val is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(val))
    except (ImportError, TypeError, ValueError):
        return False
