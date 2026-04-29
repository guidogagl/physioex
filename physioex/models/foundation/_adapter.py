"""Adapter between physioex tensor format and EEGBenchmarks batch contract.

Handles:
1. Tensor → batch conversion with proper metadata
2. Channel name mapping (raw EDF names → standard 10-20 names)
3. Recording-level statistics (mean, std, q95) for models that need them
4. CheckpointSpec construction for backbone instantiation
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

_EEGBENCH_SRC = "/home/dev/EEGBenchmarks/src"
try:
    import importlib

    if (
        importlib.util.find_spec("eegbenchmarks") is None
        and _EEGBENCH_SRC not in sys.path
    ):
        sys.path.insert(0, _EEGBENCH_SRC)
except Exception:
    pass


def _strip_to_standard(name: str) -> str:
    """Best-effort: 'EEG C3-M2' → 'C3', 'C3_M2' → 'C3', 'EEG1' → 'EEG1'.

    Matches EEGBenchmarks' physioex_base._strip_to_standard().
    """
    name = name.strip()
    for prefix in ("EEG ", "EOG ", "EMG ", "ECG "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    m = re.split(r"[-_]", name)
    return m[0] if m else name


def map_channel_names(
    raw_names: List[str],
    channel_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Map raw physioex channel names to standard 10-20 names.

    If channel_map is provided, uses it. Otherwise falls back to
    _strip_to_standard() (removes "EEG " prefix and reference suffix).

    Args:
        raw_names: channel names from physioex (e.g., ["EEG F4-M1", "EEG C4-M1"])
        channel_map: explicit mapping (e.g., {"EEG F4-M1": "F4"})

    Returns:
        Standard names (e.g., ["F4", "C4"])
    """
    if channel_map:
        return [channel_map.get(ch, _strip_to_standard(ch)) for ch in raw_names]
    return [_strip_to_standard(ch) for ch in raw_names]


def compute_recording_stats(x: Tensor) -> Dict[str, Any]:
    """Compute per-recording normalization statistics from a signal tensor.

    Args:
        x: (B, C, T) or (n_epochs, C, T) signal tensor in µV.

    Returns:
        dict with:
            recording_mean: (C,) mean per channel
            recording_std: (C,) std per channel
            recording_q95: (C,) 95th percentile of |signal| per channel
            amplitude_range: scalar, max - min across all channels (for BENDR SCALE)
    """
    # Flatten batch for recording-level stats
    if x.ndim == 3:
        x_flat = x.reshape(-1, x.shape[1], x.shape[2])  # (N, C, T)
    else:
        x_flat = x.unsqueeze(0)

    x_np = x_flat.detach().cpu().float().numpy()

    # Per-channel stats across all epochs and time
    x_2d = x_np.reshape(x_np.shape[0] * x_np.shape[2], x_np.shape[1])  # (N*T, C)

    recording_mean = x_2d.mean(axis=0)  # (C,)
    recording_std = x_2d.std(axis=0)  # (C,)
    recording_q95 = np.percentile(np.abs(x_2d), 95, axis=0)  # (C,)

    # Amplitude range for BENDR SCALE channel
    amp_range = float(x_np.max() - x_np.min())

    return {
        "recording_mean": recording_mean.tolist(),
        "recording_std": recording_std.tolist(),
        "recording_q95": recording_q95.tolist(),
        "amplitude_range": amp_range,
    }


def tensor_to_benchmark_batch(
    x: Tensor,
    channel_names: Optional[List[str]] = None,
    sampling_rate: Optional[float] = None,
    channel_map: Optional[Dict[str, str]] = None,
    include_recording_stats: bool = True,
) -> Dict[str, Any]:
    """Convert a (B, C, T) tensor to the EEGBenchmarks batch format.

    Args:
        x: (B, C, T) multi-channel signal tensor in µV.
        channel_names: raw channel names from physioex EDF.
        sampling_rate: signal sampling rate in Hz.
        channel_map: raw → standard 10-20 name mapping.
        include_recording_stats: if True, compute and include
            recording_mean/std/q95/amplitude_range in metadata.

    Returns:
        dict matching the BenchmarkBatch contract, with standard channel
        names and recording statistics.
    """
    B, C, T = x.shape
    raw_names = channel_names or [f"EEG{j}" for j in range(C)]
    std_names = map_channel_names(raw_names, channel_map)

    # Recording stats (computed once for the batch)
    stats = compute_recording_stats(x) if include_recording_stats else {}

    meta = []
    for i in range(B):
        m: Dict[str, Any] = {
            "channels": list(std_names),
            "channels_raw": list(raw_names),
            "unit": "uV",
        }
        if sampling_rate is not None:
            m["sampling_rate"] = sampling_rate
        # Add recording-level stats
        if stats:
            m["recording_mean"] = stats["recording_mean"]
            m["recording_std"] = stats["recording_std"]
            m["recording_q95"] = stats["recording_q95"]
            m["amplitude_range"] = stats["amplitude_range"]
        meta.append(m)

    return {
        "signals": {"eeg": x, "full_signal": x},
        "label": torch.zeros(B, dtype=torch.long, device=x.device),
        "meta": meta,
    }


def make_checkpoint_spec(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    expected_sampling_rate: Optional[float] = None,
    expected_channels: Sequence[str] = ("eeg",),
    embedding_dim: Optional[int] = None,
    source_type: str = "local",
    source_reference: str = "",
) -> "CheckpointSpec":
    """Build a minimal CheckpointSpec for backbone instantiation."""
    from eegbenchmarks.benchmarking_helpers.contracts import CheckpointSpec

    return CheckpointSpec(
        identifier=f"{model_name}_physioex",
        model_family=model_name,
        variant="physioex_wrapper",
        source_type=source_type,
        source_reference=source_reference,
        checkpoint_path=checkpoint_path,
        input_kind="raw_timeseries",
        expected_channels=tuple(expected_channels),
        expected_sampling_rate=expected_sampling_rate,
        expected_epoch_seconds=30.0,
        pretraining_datasets=(),
        finetuned_datasets=(),
        embedding_dim=embedding_dim,
        wrapper_name=model_name,
        status="ready",
        runtime_overrides={},
    )
