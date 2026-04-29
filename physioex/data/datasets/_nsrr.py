"""Shared helpers for NSRR-format datasets (EDF + NSRR XML annotations).

All three NSRR-family datasets (MESA, MrOS, HomePAP) share the same annotation
format: ``<stem>-nsrr.xml`` files parsed by ``parse_nsrr_xml``.  This private
base class factors out subject enumeration (EDF/XML pairing) and label reading
so each concrete dataset only needs to define channel preferences, default root,
and ``_list_subjects`` (which calls ``_pair_edfs_with_xml`` with the right paths).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from physioex.data.base import BasePhysioDataset, SubjectSpec
from physioex.data.readers.annotations import parse_nsrr_xml


class _NSRRBaseDataset(BasePhysioDataset):
    """Common behaviour: label parsing + subject enumeration for NSRR layouts."""

    DEFAULT_EPOCH_LENGTH_SEC = 30.0

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
