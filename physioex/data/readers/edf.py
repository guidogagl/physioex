"""EDF header probing and channel resolution.

Built on pyedflib. Reuses channel-matching logic from
physioex/preprocess/utils/sleepdata.py (case-insensitive with tuple support)
and extends it with a "claimed set" tracker so repeated modality requests
(e.g. ["EEG", "EEG"]) resolve to distinct physical channels.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyedflib


# Default modality preference lists -- sourced from existing library constants.
# Each entry is either a single channel name (string) or a tuple (A, B) for
# differential pair (signal[A] - signal[B]).
DEFAULT_EEG_PREFERENCES = [
    ("C4", "M1"),
    ("C3", "M2"),
    ("C4", "A1"),
    ("C3", "A2"),
    "C4-M1",
    "C3-M2",
    "C4-A1",
    "C3-A2",
    "EEG(sec)",
    "EEG",
    "EEG1",
]
DEFAULT_EOG_PREFERENCES = [
    ("EOG(L)", "EOG(R)"),
    ("E1", "M2"),
    ("E2", "M1"),
    ("LOC", "ROC"),
    "EOG",
]
DEFAULT_EMG_PREFERENCES = [
    ("LCHIN", "CCHIN"),
    ("CHIN1", "CHIN2"),
    "EMG",
    "EMG Chin",
]
DEFAULT_ECG_PREFERENCES = ["ECG", "ECG1", "ECG2", "EKG"]


DEFAULT_PREFERENCES: Dict[str, List[Union[str, Tuple[str, str]]]] = {
    "EEG": DEFAULT_EEG_PREFERENCES,
    "EOG": DEFAULT_EOG_PREFERENCES,
    "EMG": DEFAULT_EMG_PREFERENCES,
    "ECG": DEFAULT_ECG_PREFERENCES,
}


KNOWN_MODALITIES = set(DEFAULT_PREFERENCES.keys())


class ChannelNotAvailableError(ValueError):
    """Raised when a channel request cannot be satisfied by an EDF."""


@dataclass
class EDFHeader:
    available_channels: List[str]  # physical names
    channel_fs: Dict[str, float]  # sample rate per channel
    channel_units: Dict[str, str]  # physical dimension per channel
    patient_meta: Dict[str, Optional[str]]  # age, sex, patient_code, birthdate
    duration_sec: float
    source_mtime: float  # EDF file mtime for cache invalidation
    start_sec: float = 0.0  # recording start as seconds-of-day (HH*3600+MM*60+SS)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "EDFHeader":
        return cls(
            available_channels=list(d["available_channels"]),
            channel_fs={str(k): float(v) for k, v in d["channel_fs"].items()},
            channel_units={str(k): str(v) for k, v in d["channel_units"].items()},
            patient_meta=dict(d["patient_meta"]),
            duration_sec=float(d["duration_sec"]),
            source_mtime=float(d["source_mtime"]),
            start_sec=float(d.get("start_sec", 0.0)),
        )


@dataclass
class ResolvedChannel:
    request: Union[str, Tuple[str, str]]  # user input (modality name or specific)
    physical: Union[str, Tuple[str, str]]  # resolved physical channel
    modality: Optional[str]  # classified modality if known
    fs_in: float  # native sample rate from EDF
    is_differential: bool
    # Original EDF channel indices (for efficient reading)
    _indices: Tuple[int, ...] = field(default=())


# ---------- Header probe ----------


def probe_edf_header(edf_path: Union[str, Path]) -> EDFHeader:
    """Read the EDF structural header only (no signal data)."""
    edf_path = Path(edf_path)
    mtime = edf_path.stat().st_mtime

    with pyedflib.EdfReader(str(edf_path)) as f:
        labels = list(f.getSignalLabels())
        n = len(labels)
        fs = {labels[i]: float(f.getSampleFrequency(i)) for i in range(n)}
        units = {labels[i]: str(f.getPhysicalDimension(i)).strip() for i in range(n)}
        duration = float(f.getFileDuration())
        try:
            dt = f.getStartdatetime()
            start_sec = float(dt.hour * 3600 + dt.minute * 60 + dt.second)
        except Exception:
            start_sec = 0.0

        # Patient metadata -- pyedflib exposes several fields; be defensive.
        def _safe(getter):
            try:
                v = getter()
                if v is None:
                    return None
                v = v.strip() if isinstance(v, str) else v
                return v if v else None
            except Exception:
                return None

        patient = {
            "patient_code": _safe(f.getPatientCode),
            "sex": _safe(f.getSex) if hasattr(f, "getSex") else _safe(f.getGender),
            "birthdate": _safe(f.getBirthdate),
            "patient_name": _safe(f.getPatientName),
            "patient_additional": _safe(f.getPatientAdditional),
        }
        # birthdate is a datetime-like or string -- normalize to ISO string if possible
        bd = patient["birthdate"]
        if bd is not None and not isinstance(bd, str):
            try:
                patient["birthdate"] = (
                    bd.isoformat() if hasattr(bd, "isoformat") else str(bd)
                )
            except Exception:
                patient["birthdate"] = str(bd)

    return EDFHeader(
        available_channels=labels,
        channel_fs=fs,
        channel_units=units,
        patient_meta=patient,
        duration_sec=duration,
        source_mtime=mtime,
        start_sec=start_sec,
    )


# ---------- Channel resolution ----------


def _iter_matches(candidate, available_upper: Dict[str, int]):
    """Given a preference entry (string or tuple) and a mapping
    UPPERCASE_NAME -> original_index, return the resolved form as a tuple
    (physical_or_tuple, index_or_tuple), or None if not satisfied."""
    if isinstance(candidate, tuple):
        if len(candidate) != 2:
            return None
        a, b = candidate
        au = a.upper() if isinstance(a, str) else None
        bu = b.upper() if isinstance(b, str) else None
        if au in available_upper and bu in available_upper:
            ia = available_upper[au]
            ib = available_upper[bu]
            return ((a, b), (ia, ib))
        return None
    else:
        cu = candidate.upper() if isinstance(candidate, str) else None
        if cu in available_upper:
            i = available_upper[cu]
            return (candidate, (i,))
        return None


def _build_upper_index(available: List[str]) -> Dict[str, int]:
    """Map upper(label) -> first index in available list."""
    out: Dict[str, int] = {}
    for i, lab in enumerate(available):
        key = lab.upper().strip()
        if key not in out:
            out[key] = i
    return out


def _make_claimed_key(physical) -> Union[Tuple[str, ...], str]:
    """Produce a hashable key representing a physical channel (single or differential pair)."""
    if isinstance(physical, tuple):
        return tuple(p.upper() if isinstance(p, str) else p for p in physical)
    if isinstance(physical, str):
        return physical.upper()
    return physical


def resolve_channels(
    requests: List[Union[str, Dict]],
    available: List[str],
    fs_map: Dict[str, float],
    preferences: Dict[str, List] = None,
    allow_missing: bool = False,
) -> List[Optional[ResolvedChannel]]:
    """Resolve user channel requests against the EDF's available channels.

    Args:
        requests: list of requests. Each may be:
            - string matching a known modality ("EEG", "EOG", "EMG", "ECG")
              -> take next unclaimed preference match
            - string naming a specific channel ("C4-M2") -> case-insensitive exact match
            - dict with keys {"modality": ..., "preference": ...} or {"name": ...}
        available: list of physical channel labels from the EDF header
        fs_map: mapping physical_label -> sample rate (for fs_in propagation)
        preferences: dict[modality -> list[candidate]]. Defaults to DEFAULT_PREFERENCES.
        allow_missing: if True, unresolvable requests return ``None`` in the
            output list instead of raising. The subject is not dropped —
            the caller (``BasePhysioDataset._build_item``) will zero-fill
            the missing channel.

    Returns:
        Ordered list of ResolvedChannel (or ``None`` for missing channels
        when ``allow_missing=True``) matching ``requests``.

    Raises:
        ChannelNotAvailableError if any request cannot be satisfied AND
        ``allow_missing`` is ``False``.
    """
    if preferences is None:
        preferences = DEFAULT_PREFERENCES

    upper_index = _build_upper_index(available)
    claimed: set = set()
    resolved: List[ResolvedChannel] = []

    for req in requests:
        # Normalize request
        modality = None
        custom_pref = None
        specific = None
        original_req = req

        if isinstance(req, dict):
            if "name" in req:
                specific = req["name"]
            elif "modality" in req:
                modality = req["modality"]
                custom_pref = req.get("preference")
            else:
                raise ValueError(
                    f"Request dict must have 'name' or 'modality': {req!r}"
                )
        elif isinstance(req, str):
            if req in KNOWN_MODALITIES:
                modality = req
            else:
                specific = req
        else:
            raise TypeError(f"Request must be str or dict, got {type(req)}: {req!r}")

        if modality is not None:
            # Walk preference list, skip already-claimed physical channels
            pref_list = (
                custom_pref
                if custom_pref is not None
                else preferences.get(modality, [])
            )
            hit = None
            for cand in pref_list:
                m = _iter_matches(cand, upper_index)
                if m is None:
                    continue
                physical, indices = m
                key = _make_claimed_key(physical)
                if key in claimed:
                    continue
                hit = (physical, indices)
                break
            if hit is None:
                if allow_missing:
                    resolved.append(None)
                    continue
                raise ChannelNotAvailableError(
                    f"No available channel for modality {modality!r} "
                    f"(preferences={pref_list!r}, available={available!r}, claimed={claimed!r})"
                )
            physical, indices = hit
            claimed.add(_make_claimed_key(physical))
            is_diff = isinstance(physical, tuple)
            fs_in = float(fs_map[available[indices[0]]])
            resolved.append(
                ResolvedChannel(
                    request=original_req,
                    physical=physical,
                    modality=modality,
                    fs_in=fs_in,
                    is_differential=is_diff,
                    _indices=indices,
                )
            )
        else:
            # Specific -- may be a tuple (differential) or a string (single channel)
            if isinstance(specific, tuple):
                m = _iter_matches(specific, upper_index)
            else:
                # Case-insensitive exact match
                m = (
                    _iter_matches(specific, upper_index)
                    if isinstance(specific, str)
                    else None
                )
            if m is None:
                if allow_missing:
                    resolved.append(None)
                    continue
                raise ChannelNotAvailableError(
                    f"Specific channel {specific!r} not in available={available!r}"
                )
            physical, indices = m
            key = _make_claimed_key(physical)
            if key in claimed:
                if allow_missing:
                    resolved.append(None)
                    continue
                raise ChannelNotAvailableError(
                    f"Specific channel {physical!r} already claimed by earlier request"
                )
            claimed.add(key)
            is_diff = isinstance(physical, tuple)
            # Classify modality heuristically based on which preference list it appears in
            inferred_modality = _classify_modality(physical, preferences)
            fs_in = float(fs_map[available[indices[0]]])
            resolved.append(
                ResolvedChannel(
                    request=original_req,
                    physical=physical,
                    modality=inferred_modality,
                    fs_in=fs_in,
                    is_differential=is_diff,
                    _indices=indices,
                )
            )

    return resolved


def _classify_modality(physical, preferences: Dict[str, List]) -> Optional[str]:
    """Heuristic: scan preference lists to classify a specific physical channel."""
    target = _make_claimed_key(physical)
    for modality, pref_list in preferences.items():
        for cand in pref_list:
            if _make_claimed_key(cand) == target:
                return modality
    return None


# ---------- Single channel read ----------


def _unit_scale_to_uv(unit: str) -> float:
    """Return the multiplier to convert a physical dimension to microvolts.

    Recognized units (case-insensitive): mV -> 1000, V -> 1e6, uV/µV -> 1.
    Unknown or empty units return 1.0 (no scaling).
    """
    u = unit.strip().lower()
    if u in ("mv",):
        return 1000.0
    if u in ("v",):
        return 1e6
    # uV, µV, empty, or unrecognized -> no scaling
    return 1.0


def read_channel(
    edf_reader: pyedflib.EdfReader, resolved: ResolvedChannel
) -> Tuple[np.ndarray, float]:
    """Read a single resolved channel from an opened EdfReader.

    Returns (signal, fs). For differential pairs, returns signal[A] - signal[B].
    Signals are automatically converted to microvolts (µV) based on the EDF
    physical dimension field (e.g. mV -> ×1000).

    The caller is responsible for opening/closing the EdfReader.
    """
    indices = resolved._indices
    fs = resolved.fs_in

    if not indices:
        raise ValueError(f"ResolvedChannel has no indices: {resolved!r}")

    if resolved.is_differential:
        if len(indices) != 2:
            raise ValueError(f"Differential channel requires 2 indices, got {indices}")
        sa = edf_reader.readSignal(indices[0]).astype(np.float32)
        sb = edf_reader.readSignal(indices[1]).astype(np.float32)
        # Scale both channels to µV before subtraction
        scale_a = _unit_scale_to_uv(edf_reader.getPhysicalDimension(indices[0]))
        scale_b = _unit_scale_to_uv(edf_reader.getPhysicalDimension(indices[1]))
        if scale_a != 1.0:
            sa *= scale_a
        if scale_b != 1.0:
            sb *= scale_b
        # Trim to common length (paranoia; EDF usually has same length per channel)
        m = min(sa.shape[0], sb.shape[0])
        return (sa[:m] - sb[:m]), fs
    else:
        sig = edf_reader.readSignal(indices[0]).astype(np.float32)
        scale = _unit_scale_to_uv(edf_reader.getPhysicalDimension(indices[0]))
        if scale != 1.0:
            sig *= scale
        return sig, fs


def read_channels_from_edf(
    edf_path: Union[str, Path], resolved_list: List[ResolvedChannel]
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Convenience: open EDF once, read all resolved channels.

    Returns dict keyed by encoded physical channel name (as stored in the cache dir name).
    """
    try:
        from physioex.data.cache import _encode_physical
    except ImportError:
        # TODO: cache module not yet available; inline a local duplicate
        def _encode_physical(physical):
            def _sanitize(s):
                return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

            if isinstance(physical, (tuple, list)):
                return "__".join(_sanitize(p) for p in physical)
            return _sanitize(physical)

    out: Dict[str, Tuple[np.ndarray, float]] = {}
    with pyedflib.EdfReader(str(edf_path)) as f:
        for rc in resolved_list:
            sig, fs = read_channel(f, rc)
            out[_encode_physical(rc.physical)] = (sig, fs)
    return out
