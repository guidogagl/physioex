"""Shared channel mapping / padding utilities for foundation model wrappers."""
from __future__ import annotations

import torch


def pad_or_repeat_channels(x: torch.Tensor, target_n: int) -> torch.Tensor:
    """Pad (zero-fill) or repeat channels to reach *target_n*.

    Input:  (B, C, T)
    Output: (B, target_n, T)

    If C < target_n the first C slots are filled and the rest are zeros.
    If C > target_n only the first target_n channels are kept.
    If C == target_n the tensor is returned as-is.
    """
    B, C, T = x.shape
    if C == target_n:
        return x
    if C > target_n:
        return x[:, :target_n, :]
    out = torch.zeros(B, target_n, T, dtype=x.dtype, device=x.device)
    out[:, :C, :] = x
    return out


# ── BENDR 20-channel layout ─────────────────────────────────────────

BENDR_TARGET_CHANNELS: tuple[str, ...] = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
    "SCALE",
)
BENDR_SCALE_INDEX = len(BENDR_TARGET_CHANNELS) - 1

_CHANNEL_ALIASES = {
    "EEG FPZ-CZ": "FZ",
    "FPZ-CZ": "FZ",
    "FPZ": "FZ",
    "FPZCZ": "FZ",
    "PZ-OZ": "PZ",
    "PZOZ": "PZ",
    "CZ": "CZ",
    "C3-A2": "C3",
    "C4-A1": "C4",
    "O1-A2": "O1",
    "O2-A1": "O2",
    "F3-A2": "F3",
    "F4-A1": "F4",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}
_IGNORED_PREFIXES = ("EOG", "EMG", "RESP", "EVENT", "TEMP", "EKG", "ECG")
_DEFAULT_FALLBACK = "FZ"


def _normalize_channel_name(name: str | None) -> str:
    if name is None:
        return ""
    candidate = str(name).strip().upper().replace("_", "-")
    candidate = " ".join(candidate.split())
    if candidate.startswith("EEG "):
        candidate = candidate[4:]
    if any(candidate.startswith(p) for p in _IGNORED_PREFIXES):
        return ""
    candidate = candidate.replace(" ", "")
    return _CHANNEL_ALIASES.get(candidate, candidate)


def map_to_bendr_layout(
    x: torch.Tensor,
    channel_names: list[str] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Map arbitrary EEG channels to BENDR's 20-channel layout.

    Args:
        x: (B, C_in, T)
        channel_names: optional list of C_in channel names.

    Returns:
        mapped: (B, 20, T) with zeros in unmapped slots and active
            channels scaled to [-1, 1].
        active_indices: list of BENDR slot indices that received data.
    """
    if x.ndim == 2:
        x = x.unsqueeze(1)
    B, C_in, T = x.shape

    names = list(channel_names or [])
    if len(names) < C_in:
        names.extend([""] * (C_in - len(names)))
    names = names[:C_in]

    normalized = [_normalize_channel_name(n) for n in names]
    if C_in == 1 and not normalized[0]:
        normalized[0] = _DEFAULT_FALLBACK

    mapped = torch.zeros(
        B, len(BENDR_TARGET_CHANNELS), T, dtype=x.dtype, device=x.device
    )
    target_idx = {n: i for i, n in enumerate(BENDR_TARGET_CHANNELS)}
    used: set[int] = set()
    active: list[int] = []

    for src_i, norm_name in enumerate(normalized):
        if not norm_name:
            continue
        dest = target_idx.get(norm_name)
        if dest is None or dest == BENDR_SCALE_INDEX or dest in used:
            continue
        mapped[:, dest] = x[:, src_i]
        used.add(dest)
        active.append(dest)

    # Scale active channels to [-1, 1]
    if active:
        act = mapped[:, active]
        flat = act.reshape(B, -1)
        mn = flat.min(dim=1).values.view(-1, 1, 1)
        mx = flat.max(dim=1).values.view(-1, 1, 1)
        denom = (mx - mn).clamp_min(1e-6)
        mapped = mapped.clone()
        mapped[:, active] = ((act - mn) / denom) * 2.0 - 1.0

    return mapped, active
