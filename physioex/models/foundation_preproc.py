"""Shared preprocessing operations for foundation model wrappers.

All operations are pure PyTorch — differentiable and GPU-compatible.
No numpy conversions, no torch.no_grad(). Compatible with CSD
(SpectralGradients needs gradient flow through the encoder).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Normalization ─────────────────────────────────────────────────


def mean_center(x: Tensor) -> Tensor:
    """Remove per-channel DC offset. (B, C, T) -> (B, C, T)."""
    return x - x.mean(dim=-1, keepdim=True)


def clip_uv(x: Tensor, max_uv: float = 100.0) -> Tensor:
    """Clamp amplitude to ±max_uv. (B, C, T) -> (B, C, T)."""
    return x.clamp(-max_uv, max_uv)


def scale_div(x: Tensor, divisor: float = 100.0) -> Tensor:
    """Divide by a constant (e.g., 100 for µV → ~[-1,1]). (B, C, T) -> (B, C, T)."""
    return x / divisor


def clip_and_scale(x: Tensor, max_uv: float = 100.0) -> Tensor:
    """Clip to ±max_uv then divide by max_uv. (B, C, T) -> (B, C, T)."""
    return clip_uv(x, max_uv) / max_uv


def zscore(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Per-sample z-score normalization. (B, C, T) -> (B, C, T)."""
    mu = x.mean(dim=dim, keepdim=True)
    sigma = x.std(dim=dim, keepdim=True).clamp(min=eps)
    return (x - mu) / sigma


def zscore_clip(x: Tensor, sigma_clip: float = 15.0, eps: float = 1e-6) -> Tensor:
    """Z-score with ±sigma_clip clipping. (B, C, T) -> (B, C, T)."""
    z = zscore(x, dim=-1, eps=eps)
    return z.clamp(-sigma_clip, sigma_clip)


def zscore_per_recording(
    x: Tensor, sigma_clip: float = 15.0, eps: float = 1e-6
) -> Tensor:
    """Per-recording z-score with optional ±σ clip.

    Statistics computed over (C, T) → per-sample mean/std.
    Used by REVE: mean over all channels together, not per-channel.

    Args:
        x: (B, C, T) input
        sigma_clip: Clip to ±sigma_clip after z-score. None = no clip.
        eps: Minimum std value.

    Returns:
        (B, C, T) normalized signal
    """
    if x.ndim != 3:
        raise ValueError(
            f"zscore_per_recording expects (B, C, T); got shape {tuple(x.shape)}"
        )
    # Reduce over (channels, time) → per-sample mean/std
    mu = x.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    sigma = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(eps)
    z = (x - mu) / sigma
    if sigma_clip is not None:
        z = z.clamp(-sigma_clip, sigma_clip)
    return z


def q95_normalize(x: Tensor, q: float = 0.95, eps: float = 1e-8) -> Tensor:
    """Per-channel 95th-percentile normalization. (B, C, T) -> (B, C, T)."""
    B, C, T = x.shape
    x_flat = x.abs().reshape(B, C, -1)
    q95 = torch.quantile(x_flat, q, dim=-1, keepdim=True)  # (B, C, 1)
    return x / (q95 + eps)


def q95_normalize_with_stats(
    x: Tensor, q95_per_channel: Tensor, eps: float = 1e-8
) -> Tensor:
    """Normalize using precomputed per-channel q95 from recording stats.

    Args:
        x: (B, C, T)
        q95_per_channel: (C,) precomputed q95 values
    """
    q95 = (
        q95_per_channel.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
    )  # (1, C, 1)
    return x / (q95 + eps)


def zscore_with_stats(
    x: Tensor, mean_per_channel: Tensor, std_per_channel: Tensor, eps: float = 1e-6
) -> Tensor:
    """Z-score using precomputed per-channel stats from recording.

    Args:
        x: (B, C, T)
        mean_per_channel: (C,) precomputed mean
        std_per_channel: (C,) precomputed std
    """
    mu = (
        mean_per_channel.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
    )  # (1, C, 1)
    sigma = std_per_channel.to(x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
    return (x - mu) / sigma.clamp(min=eps)


def minmax_scale(x: Tensor, dim: int = -1) -> Tensor:
    """Per-channel min-max scaling to [-1, 1]. (B, C, T) -> (B, C, T)."""
    mn = x.min(dim=dim, keepdim=True).values
    mx = x.max(dim=dim, keepdim=True).values
    denom = (mx - mn).clamp(min=1e-6)
    return 2.0 * (x - mn) / denom - 1.0


# ── Channel operations ────────────────────────────────────────────


def pad_channels(x: Tensor, target_n: int) -> Tensor:
    """Zero-pad channel dimension to target_n. (B, C, T) -> (B, target_n, T)."""
    B, C, T = x.shape
    if C >= target_n:
        return x[:, :target_n, :]
    out = torch.zeros(B, target_n, T, dtype=x.dtype, device=x.device)
    out[:, :C, :] = x
    return out


def select_channels(x: Tensor, indices: List[int]) -> Tensor:
    """Select specific channels by index. (B, C, T) -> (B, len(indices), T)."""
    return x[:, indices, :]


def strip_zero_channels(x: Tensor) -> Tuple[Tensor, List[int]]:
    """Remove channels that are all-zero. Returns (filtered_x, kept_indices)."""
    B, C, T = x.shape
    nonzero = x.abs().sum(dim=(0, 2)) > 0  # (C,)
    kept = nonzero.nonzero(as_tuple=True)[0].tolist()
    if len(kept) == C:
        return x, kept
    return x[:, kept, :], kept


def map_channels_to_layout(
    x: Tensor,
    input_names: List[str],
    target_layout: List[str],
    aliases: Dict[str, str] = None,
) -> Tuple[Tensor, List[int]]:
    """Map input channels to a target layout by name matching.

    Args:
        x: (B, C_in, T) input signal
        input_names: list of C_in standard channel names (from dataset channel_map)
        target_layout: list of target channel names the model expects
        aliases: additional name aliases (e.g., {"T3": "T7"})

    Returns:
        mapped: (B, len(target_layout), T) with zero-fill for missing channels
        active_indices: list of target indices that received real data
    """
    B, C_in, T = x.shape
    n_target = len(target_layout)
    mapped = torch.zeros(B, n_target, T, dtype=x.dtype, device=x.device)
    active = []

    # Build lookup: normalized name → input index
    aliases = aliases or {}
    input_lookup = {}
    for i, name in enumerate(input_names):
        upper = name.upper().strip()
        input_lookup[upper] = i
        # Also register aliased form
        if upper in aliases:
            input_lookup[aliases[upper].upper()] = i

    for tgt_idx, tgt_name in enumerate(target_layout):
        upper = tgt_name.upper().strip()
        src_idx = input_lookup.get(upper)
        if src_idx is None and upper in aliases:
            src_idx = input_lookup.get(aliases[upper].upper())
        if src_idx is not None and src_idx < C_in:
            mapped[:, tgt_idx, :] = x[:, src_idx, :]
            active.append(tgt_idx)

    return mapped, active


def create_padding_mask(
    n_channels: int, n_valid: int, batch_size: int, device: torch.device
) -> Tensor:
    """Create boolean padding mask. True = padded/invalid, False = real.

    Returns: (B, n_channels) bool tensor.
    """
    mask = torch.ones(batch_size, n_channels, dtype=torch.bool, device=device)
    mask[:, :n_valid] = False
    return mask


# ── Recording stats ──────────────────────────────────────────────


def compute_channel_stats(signal: Tensor) -> Dict[str, float]:
    """Compute statistics for a single channel signal.

    Args:
        signal: (n_epochs, T) or (T,) tensor

    Returns:
        dict with mean, std, q95, min, max
    """
    flat = signal.reshape(-1).float()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "q95": float(torch.quantile(flat.abs(), 0.95)),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }
