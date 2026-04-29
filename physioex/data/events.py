"""Per-epoch event metadata for sleep staging datasets.

Provides a ``SleepEvent`` dataclass for representing temporal events
(respiratory, arousal, limb movement, etc.) and utilities for mapping
them to per-epoch event lists suitable for use in training/analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SleepEvent:
    type: str  # "respiratory" | "arousal" | "limb_movement" | "position" | "desaturation" | "other"
    concept: str  # e.g. "Hypopnea", "ASDA arousal", "PLM (Left)", "Supine"
    onset_sec: float  # absolute seconds from recording start
    duration_sec: float
    extra: Dict[str, Any] = field(default_factory=dict)


def map_events_to_epochs(
    events: List[SleepEvent],
    n_epochs: int,
    epoch_length_sec: float = 30.0,
) -> List[List[Dict[str, Any]]]:
    """Map a flat list of events to per-epoch event lists.

    An event overlaps epoch *N* if ``[onset, onset+dur)`` intersects
    ``[N*epoch_len, (N+1)*epoch_len)``.

    Returns:
        List of length *n_epochs*, each element a list of event dicts.
    """
    result: List[List[Dict[str, Any]]] = [[] for _ in range(n_epochs)]
    for ev in events:
        ev_start = ev.onset_sec
        ev_end = ev_start + max(
            ev.duration_sec, 0.001
        )  # treat 0-duration as instantaneous
        first_epoch = max(0, int(ev_start // epoch_length_sec))
        last_epoch = min(n_epochs - 1, int((ev_end - 0.001) // epoch_length_sec))
        d = event_to_dict(ev)
        for ep in range(first_epoch, last_epoch + 1):
            result[ep].append(d)
    return result


def event_to_dict(ev: SleepEvent) -> Dict[str, Any]:
    """Convert a ``SleepEvent`` to a plain dict."""
    d: Dict[str, Any] = {
        "type": ev.type,
        "concept": ev.concept,
        "onset_sec": ev.onset_sec,
        "duration_sec": ev.duration_sec,
    }
    if ev.extra:
        d["extra"] = ev.extra
    return d


def events_to_dicts(events: List[SleepEvent]) -> List[Dict[str, Any]]:
    """Convert a list of ``SleepEvent`` objects to a list of dicts."""
    return [event_to_dict(ev) for ev in events]


def dicts_to_events(dicts: List[Dict[str, Any]]) -> List[SleepEvent]:
    """Reconstruct ``SleepEvent`` objects from a list of dicts."""
    return [
        SleepEvent(
            type=d["type"],
            concept=d["concept"],
            onset_sec=d["onset_sec"],
            duration_sec=d["duration_sec"],
            extra=d.get("extra", {}),
        )
        for d in dicts
    ]
