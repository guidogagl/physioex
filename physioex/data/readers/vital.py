"""Reader for VitalRecorder ``.vital`` files (VitalDB).

Mirrors the role of :mod:`physioex.data.readers.edf` for a non-EDF container:
:func:`probe_vital_header` returns the same :class:`EDFHeader` structure the
dataset layer expects, and :func:`read_vital_channel` returns ``(signal, fs)``
just like :func:`~physioex.data.readers.edf.read_channel`.

A ``.vital`` file holds many tracks from several devices, named
``<device>/<track>`` (e.g. ``BIS/EEG1_WAV``, ``Solar8000/HR``).  Only
**waveform** tracks -- those with a non-zero sample rate -- are exposed as
channels; numeric tracks (``BIS/BIS``, ``BIS/SR``, ``BIS/SQI``, ...) are
irregularly sampled derived indices and belong to the analysis layer, not to
the epoched signal pipeline.

``vitaldb`` is an optional dependency (extra ``datasets``) and is imported
lazily inside the functions, so importing physioex never requires it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from physioex.data.readers.edf import EDFHeader, ResolvedChannel

__all__ = [
    "VitalTrackNotFoundError",
    "probe_vital_header",
    "read_vital_channel",
    "list_vital_tracks",
]


class VitalTrackNotFoundError(KeyError):
    """Raised when a requested track is absent from a ``.vital`` file."""


def _import_vitaldb():
    try:
        from vitaldb import VitalFile
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading .vital files requires the 'vitaldb' package. "
            "Install it with: pip install 'physioex[datasets]'  (or: pip install vitaldb)"
        ) from exc
    return VitalFile


# Physical-dimension strings seen in VitalDB, mapped to a factor converting to µV.
_UNIT_SCALE_TO_UV: Dict[str, float] = {
    "uv": 1.0,
    "µv": 1.0,
    "mv": 1e3,
    "v": 1e6,
}


def _unit_scale_to_uv(unit: Optional[str]) -> float:
    """Factor converting *unit* to microvolts; 1.0 when unknown or unitless."""
    if not unit:
        return 1.0
    return _UNIT_SCALE_TO_UV.get(unit.strip().lower(), 1.0)


def list_vital_tracks(path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Return ``{track_name: {srate, unit, type}}`` for every track in the file.

    Unlike :func:`probe_vital_header` this keeps numeric tracks too, so callers
    that need ``BIS/SR`` or ``BIS/SQI`` can discover them.
    """
    VitalFile = _import_vitaldb()
    vf = VitalFile(str(path), header_only=True)
    return {
        name: {
            "srate": float(getattr(trk, "srate", 0.0) or 0.0),
            "unit": getattr(trk, "unit", "") or "",
            "type": int(getattr(trk, "type", 0) or 0),
        }
        for name, trk in vf.trks.items()
    }


def probe_vital_header(
    path: Union[str, Path], patient_meta: Optional[Dict[str, Any]] = None
) -> EDFHeader:
    """Read the structural header of a ``.vital`` file (no signal data).

    Only waveform tracks (``srate > 0``) are listed as available channels.

    Args:
        path: path to the ``.vital`` file.
        patient_meta: optional external metadata (age, sex, ...).  ``.vital``
            files carry no patient identification, so demographics come from the
            dataset's clinical table.
    """
    VitalFile = _import_vitaldb()
    path = Path(path)
    vf = VitalFile(str(path), header_only=True)

    channels, fs_map, units = [], {}, {}
    for name, trk in vf.trks.items():
        srate = float(getattr(trk, "srate", 0.0) or 0.0)
        if srate <= 0:
            continue  # numeric / string track: not an epochable channel
        channels.append(name)
        fs_map[name] = srate
        units[name] = getattr(trk, "unit", "") or ""

    duration = float(vf.dtend - vf.dtstart) if vf.dtend and vf.dtstart else 0.0
    # dtstart is an absolute POSIX timestamp; express the start as seconds of day
    # to match the EDF convention used elsewhere in physioex.
    start_sec = float(vf.dtstart % 86400) if vf.dtstart else 0.0

    return EDFHeader(
        available_channels=sorted(channels),
        channel_fs=fs_map,
        channel_units=units,
        patient_meta=dict(patient_meta or {}),
        duration_sec=duration,
        source_mtime=path.stat().st_mtime,
        start_sec=start_sec,
    )


def read_vital_channel(
    path: Union[str, Path], resolved: ResolvedChannel
) -> Tuple[np.ndarray, float]:
    """Read one waveform track, resampled onto its native grid.

    Returns ``(signal_uv, fs)``.  Gaps -- ``.vital`` recordings are not
    contiguous, monitors disconnect -- come back from ``to_numpy`` as NaN and
    are replaced with zeros, the same convention ``_safe_slice`` uses for
    missing signal elsewhere in physioex.

    Differential requests are not supported: VitalDB exposes already-derived
    bipolar EEG channels.
    """
    if resolved.is_differential:
        raise ValueError(
            f"Differential channels are not supported for .vital files: {resolved!r}"
        )

    VitalFile = _import_vitaldb()
    name = resolved.physical
    if not isinstance(name, str):
        raise ValueError(f"Expected a track name, got {name!r}")

    vf = VitalFile(str(path), track_names=[name])
    trk = vf.trks.get(name)
    if trk is None:
        raise VitalTrackNotFoundError(f"{name!r} not found in {path}")

    fs = float(getattr(trk, "srate", 0.0) or 0.0)
    if fs <= 0:
        raise ValueError(f"Track {name!r} in {path} is not a waveform (srate={fs})")

    arr = vf.to_numpy([name], 1.0 / fs)
    sig = np.asarray(arr, dtype=np.float32).reshape(-1)
    np.nan_to_num(sig, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    scale = _unit_scale_to_uv(getattr(trk, "unit", ""))
    if scale != 1.0:
        sig *= scale

    return sig, fs
