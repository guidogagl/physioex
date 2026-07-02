"""Annotation parsers for sleep staging label formats.

Supports:
  - NSRR XML (used by MESA, SHHS, MrOS, HPAP)
  - TSV (Alzheimer's, Parkinson's, some others)
  - Embedded EDF annotations (HMC, Napping -- handled via pyedflib.readAnnotations)
  - STAGES CSV (13-site clinical sleep dataset with 3 encoding variants)

Each parser returns a 1D numpy.ndarray of int16 per-epoch labels, where:
  - 0-4 are AASM 5-class mappings (W, N1, N2, N3, REM)
  - -1 denotes unscored / invalid epochs
"""
from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np


# NSRR standard stage names -> 5-class AASM
# Based on physioex/preprocess/utils/sleepdata.py:read_sleepdata_annotation (stages_to list)
NSRR_STAGE_MAP: Dict[str, int] = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,  # N4 -> N3
    "REM sleep|5": 4,
    "Unscored|9": -1,
}


def parse_nsrr_xml(
    xml_path: Union[str, Path],
    epoch_length_sec: float = 30.0,
    stage_map: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Parse an NSRR-format XML annotation file.

    The XML structure is a sequence of ScoredEvent entries with EventType
    "Stages|Stages". Each event has Start (sec), Duration (sec), EventConcept
    (stage name matching NSRR_STAGE_MAP keys).

    Returns: int16 array of per-epoch labels, length = total_duration // epoch_length.
    Unscored or out-of-range get -1.
    """
    if stage_map is None:
        stage_map = NSRR_STAGE_MAP
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    events = []
    for event in root.iter("ScoredEvent"):
        etype = event.findtext("EventType", default="")
        if not etype.startswith("Stages"):
            continue
        concept = event.findtext("EventConcept", default="")
        try:
            start = float(event.findtext("Start", default="0"))
            duration = float(event.findtext("Duration", default="0"))
        except ValueError:
            continue
        events.append((start, duration, concept))

    if not events:
        return np.array([], dtype=np.int16)

    total = max(s + d for s, d, _ in events)
    n_epochs = int(total // epoch_length_sec)
    labels = np.full(n_epochs, -1, dtype=np.int16)
    for start, duration, concept in events:
        stage = stage_map.get(concept, -1)
        i0 = int(round(start / epoch_length_sec))
        i1 = int(round((start + duration) / epoch_length_sec))
        if i1 > n_epochs:
            i1 = n_epochs
        labels[i0:i1] = stage
    return labels


def parse_tsv_annotations(
    tsv_path: Union[str, Path],
    epoch_length_sec: float = 30.0,
    stage_map: Optional[Dict[str, int]] = None,
    start_col: str = "start",
    end_col: str = "end",
    stage_col: str = "stage",
) -> np.ndarray:
    """Parse a tab-separated annotation file with columns [start, end, stage].

    Start and end are in seconds. Stage is a string to be mapped via stage_map.
    """
    if stage_map is None:
        # Common AASM mapping
        stage_map = {
            "W": 0,
            "Wake": 0,
            "N1": 1,
            "S1": 1,
            "N2": 2,
            "S2": 2,
            "N3": 3,
            "S3": 3,
            "N4": 3,
            "S4": 3,
            "R": 4,
            "REM": 4,
        }
    import csv

    events = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                s = float(row[start_col])
                e = float(row[end_col])
            except (KeyError, ValueError):
                continue
            stage_str = row.get(stage_col, "").strip()
            stage = stage_map.get(stage_str, -1)
            events.append((s, e, stage))

    if not events:
        return np.array([], dtype=np.int16)

    total = max(e for _, e, _ in events)
    n_epochs = int(total // epoch_length_sec)
    labels = np.full(n_epochs, -1, dtype=np.int16)
    for s, e, stage in events:
        i0 = int(round(s / epoch_length_sec))
        i1 = int(round(e / epoch_length_sec))
        if i1 > n_epochs:
            i1 = n_epochs
        labels[i0:i1] = stage
    return labels


def parse_nsrr_xml_events(xml_path: Union[str, Path]) -> List:
    """Parse non-stage events from an NSRR XML file.

    Returns list of ``SleepEvent`` objects for respiratory, arousal,
    limb movement, desaturation, and other events.
    """
    from physioex.data.events import SleepEvent

    CATEGORY_MAP = {
        "Respiratory|Respiratory": "respiratory",
        "Arousals|Arousals": "arousal",
        "Limb Movement|Limb Movement": "limb_movement",
    }

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    events: List = []
    for event in root.iter("ScoredEvent"):
        etype = event.findtext("EventType", default="")
        if etype.startswith("Stages") or not etype:
            continue
        concept = event.findtext("EventConcept", default="")
        try:
            start = float(event.findtext("Start", default="0"))
            duration = float(event.findtext("Duration", default="0"))
        except ValueError:
            continue
        category = CATEGORY_MAP.get(etype, "other")
        clean_concept = concept.split("|")[0].strip() if "|" in concept else concept
        if "desaturation" in clean_concept.lower():
            category = "desaturation"
        extra: Dict[str, object] = {}
        for tag in ("SpO2Nadir", "SpO2Baseline", "SignalLocation"):
            val = event.findtext(tag)
            if val:
                key = tag[0].lower() + tag[1:]  # camelCase
                try:
                    extra[key] = float(val)
                except ValueError:
                    extra[key] = val
        events.append(
            SleepEvent(
                type=category,
                concept=clean_concept,
                onset_sec=start,
                duration_sec=duration,
                extra=extra,
            )
        )
    return events


# ---------------------------------------------------------------------------
# STAGES CSV annotation parser
# ---------------------------------------------------------------------------

logger = logging.getLogger("physioex.data")

# STAGES sleep stage vocabulary -> AASM 5-class mapping.
# "Awake" is intentionally excluded: it is a short-duration technician
# annotation (typically 3-21s), NOT a stage label.
STAGES_STAGE_MAP: Dict[str, int] = {
    "Wake": 0,
    "Stage1": 1,
    "Stage2": 2,
    "Stage3": 3,
    "REM": 4,
    "UnknownStage": -1,
    "No Stage": -1,
    "MT": -1,
    "mt": -1,
}

# Non-stage event classification for STAGES CSVs.
_STAGES_EVENT_CATEGORIES: Dict[str, str] = {
    # Respiratory
    "Hypopnea": "respiratory",
    "ObstructiveApnea": "respiratory",
    "CentralApnea": "respiratory",
    "MixedApnea": "respiratory",
    "RERA": "respiratory",
    "FlowLimitation": "respiratory",
    # Desaturation
    "Desaturation": "desaturation",
    "Desaturation w/ Respiratory": "desaturation",
    "SpO2 artifact": "desaturation",
    # Arousal
    "Arousal": "arousal",
    "Arousal w/ Respiratory": "arousal",
    "Spontaneous Arousal": "arousal",
    # Limb movement
    "Left Leg": "limb_movement",
    "Right Leg": "limb_movement",
    "Both Leg": "limb_movement",
    "PLM": "limb_movement",
    "Periodic Leg Movement": "limb_movement",
    # Position
    "Supine": "position",
    "Right": "position",
    "Left": "position",
    "Prone": "position",
    "Left Side": "position",
    "Right Side": "position",
}

# Maximum plausible duration for a single stage annotation (12 hours).
# Anything above this threshold is clamped to epoch_length_sec.  This
# handles the anomalous 2592000s (30 days) duration in 3 STNF files.
_MAX_PLAUSIBLE_DURATION_SEC = 43200.0


def _parse_time_to_seconds(time_str: str) -> float:
    """Parse HH:MM:SS into seconds-from-midnight."""
    parts = time_str.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0
    return h * 3600 + m * 60 + s


def parse_stages_csv(
    csv_path: Union[str, Path],
    epoch_length_sec: float = 30.0,
    stage_map: Optional[Dict[str, int]] = None,
    edf_start_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List]:
    """Parse a STAGES CSV annotation file.

    The CSV has columns ``Start Time, Duration (seconds), Event``.  Three
    encoding variants exist across the 13 STAGES sites:

    1. **dur=0** (BOGN): each stage line has duration 0, timestamps 30s apart.
    2. **dur=30** (MSTR, STNF, STLK, MSTH): each stage line has duration 30.
    3. **blocks** (GSDV, GSBB, GSLH, GSSA, GSSW, MSMI, MSNF, MSQW): stage
       lines have variable durations (multiples of 30s) representing N
       consecutive epochs.

    Args:
        edf_start_sec: EDF recording start as seconds-of-day. When provided,
            labels are aligned to the EDF start rather than to the first CSV
            timestamp, fixing misalignment when the CSV begins after the EDF.

    Returns:
        ``(labels, events)`` where *labels* is an ``int16`` numpy array of
        per-epoch AASM labels (``-1`` for unscored) and *events* is a list
        of ``SleepEvent`` objects for non-stage annotations.
    """
    from physioex.data.events import SleepEvent

    if stage_map is None:
        stage_map = STAGES_STAGE_MAP

    # ------------------------------------------------------------------
    # 1. Read all rows
    # ------------------------------------------------------------------
    rows: List[Tuple[str, float, str]] = []  # (time_str, duration, event)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_str = (row.get("Start Time") or "").strip()
            dur_str = (row.get("Duration (seconds)") or "0").strip()
            event_str = (row.get("Event") or "").strip()
            if not time_str or not event_str:
                continue
            try:
                dur = float(dur_str)
            except ValueError:
                dur = 0.0
            rows.append((time_str, dur, event_str))

    if not rows:
        return np.array([], dtype=np.int16), []

    # ------------------------------------------------------------------
    # 2. Parse timestamps with midnight-crossing detection
    # ------------------------------------------------------------------
    raw_seconds = [_parse_time_to_seconds(t) for t, _, _ in rows]

    # Detect midnight crossings: when time jumps backward by > 12 hours,
    # add 24h (86400s) to all subsequent timestamps.
    offsets = [0.0] * len(raw_seconds)
    cumulative_offset = 0.0
    for i in range(1, len(raw_seconds)):
        if (
            raw_seconds[i] + cumulative_offset
            < raw_seconds[i - 1] + offsets[i - 1] - 43200
        ):
            cumulative_offset += 86400.0
        offsets[i] = cumulative_offset
    abs_seconds = [raw_seconds[i] + offsets[i] for i in range(len(raw_seconds))]

    # Recording start: prefer EDF start time (seconds-of-day) when
    # available; otherwise fall back to the earliest CSV timestamp.
    if edf_start_sec is not None:
        # Apply the same midnight-crossing logic to the EDF start time:
        # express it in the same absolute-second space as abs_seconds.
        edf_abs = edf_start_sec
        # If EDF start is in the evening and abs_seconds[0] has crossed
        # midnight, adjust (edf is before midnight, CSV continued after).
        if abs_seconds[0] - edf_abs > 43200:
            pass  # edf_abs is already correct (before midnight)
        elif edf_abs - abs_seconds[0] > 43200:
            edf_abs += 86400.0
        recording_start = edf_abs
    else:
        recording_start = abs_seconds[0]
    rel_seconds = [s - recording_start for s in abs_seconds]

    # ------------------------------------------------------------------
    # 3. Separate stage lines from event lines
    # ------------------------------------------------------------------
    stage_rows: List[Tuple[float, float, str]] = []  # (rel_sec, duration, stage_name)
    event_rows: List[Tuple[float, float, str]] = []  # (rel_sec, duration, event_name)

    for i, (_, dur, event_str) in enumerate(rows):
        if event_str in stage_map:
            stage_rows.append((rel_seconds[i], dur, event_str))
        else:
            event_rows.append((rel_seconds[i], dur, event_str))

    # ------------------------------------------------------------------
    # 4. Build labels array from stage lines
    # ------------------------------------------------------------------
    # Use timestamp-based positioning: each stage entry is placed at
    # epoch_idx = round(rel_sec / epoch_length_sec).  This correctly
    # handles (a) pre-scoring gaps (pre-lights-off period), (b) duplicate
    # timestamps (e.g. BOGN dur=0 with stage + MT at the same time),
    # and (c) inter-stage gaps.  For duplicates, actual sleep stages
    # (label >= 0) take precedence over unscored markers (label == -1).
    if stage_rows:
        # Determine array size from the last stage entry
        last_rel_sec = stage_rows[-1][0]
        last_dur = stage_rows[-1][1]
        if last_dur > _MAX_PLAUSIBLE_DURATION_SEC:
            last_dur = epoch_length_sec
        if last_dur <= 0:
            last_dur = epoch_length_sec
        total_sec = last_rel_sec + last_dur
        n_total = max(1, int(round(total_sec / epoch_length_sec)))

        labels_arr = np.full(n_total, -1, dtype=np.int16)

        for rel_sec, dur, stage_name in stage_rows:
            label = stage_map.get(stage_name, -1)

            # Clamp anomalous durations (e.g. 2592000s in 3 STNF files)
            if dur > _MAX_PLAUSIBLE_DURATION_SEC:
                dur = epoch_length_sec

            epoch_idx = int(round(rel_sec / epoch_length_sec))
            if epoch_idx < 0:
                continue

            if dur <= 0:
                # dur=0 encoding: place 1 epoch at its timestamp position
                if epoch_idx < n_total:
                    # Actual stages (>= 0) take precedence over MT/unscored (-1)
                    if label >= 0 or labels_arr[epoch_idx] == -1:
                        labels_arr[epoch_idx] = label
            else:
                n_epochs = max(1, int(dur / epoch_length_sec))
                end_idx = min(epoch_idx + n_epochs, n_total)
                for j in range(epoch_idx, end_idx):
                    if label >= 0 or labels_arr[j] == -1:
                        labels_arr[j] = label

        labels = labels_arr
    else:
        labels = np.array([], dtype=np.int16)

    # ------------------------------------------------------------------
    # 5. Build event list from non-stage lines
    # ------------------------------------------------------------------
    events: List[SleepEvent] = []
    for rel_sec, dur, event_name in event_rows:
        category = _STAGES_EVENT_CATEGORIES.get(event_name)
        if category is None:
            # Skip non-categorized events (calibration, technician notes, etc.)
            # unless they look like they might be meaningful
            category = "other"
        events.append(
            SleepEvent(
                type=category,
                concept=event_name,
                onset_sec=rel_sec,
                duration_sec=dur,
            )
        )

    return labels, events
