"""BENDR — BENchmark D Representations (Kostas et al., 2021).

Pure encoder: (B, L, C, T) -> (B, L, D)
Fixed 20-channel layout (19 EEG + 1 SCALE) in standard 10-20 montage.

Embeddings: encoder features -> (B, 2048).

Preprocessing (pure PyTorch, differentiable):
  1. Mean-center per channel
  2. Map to 20-channel 10-20 layout via alias matching
  3. Per-sequence min-max scaling to [-1,1] on active channels
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from physioex.models.foundation_base import FoundationEncoder
from physioex.models.foundation_preproc import mean_center, minmax_scale

_BENDR_TARGET_CHANNELS = (
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
_BENDR_SCALE_INDEX = len(_BENDR_TARGET_CHANNELS) - 1
_DEFAULT_FALLBACK_CHANNEL = "FZ"

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


def _normalize_channel_name(name: str) -> str:
    if not name:
        return ""
    candidate = name.strip().upper().replace("_", "-")
    candidate = " ".join(candidate.split())
    if candidate.startswith("EEG "):
        candidate = candidate[4:]
    if any(candidate.startswith(p) for p in _IGNORED_PREFIXES):
        return ""
    candidate = candidate.replace(" ", "")
    return _CHANNEL_ALIASES.get(candidate, candidate)


def _map_to_bendr_layout(
    x: torch.Tensor,
    channel_names: List[str],
) -> Tuple[torch.Tensor, List[int]]:
    """Map input channels to BENDR's 20-ch 10-20 layout.

    Returns: (mapped_x, active_indices) where active_indices lists
    target indices that received real data (excluding SCALE).
    """
    B, C_in, T = x.shape
    n_target = len(_BENDR_TARGET_CHANNELS)
    mapped = torch.zeros(B, n_target, T, dtype=x.dtype, device=x.device)
    target_index = {name: idx for idx, name in enumerate(_BENDR_TARGET_CHANNELS)}
    used_targets: set = set()
    active: List[int] = []

    normalized = [_normalize_channel_name(ch) for ch in channel_names]
    if C_in == 1 and not normalized[0]:
        normalized[0] = _DEFAULT_FALLBACK_CHANNEL

    for src_idx, norm_name in enumerate(normalized[:C_in]):
        if not norm_name:
            continue
        dest_idx = target_index.get(norm_name)
        if dest_idx is None or dest_idx == _BENDR_SCALE_INDEX:
            continue
        if dest_idx in used_targets:
            continue
        mapped[:, dest_idx, :] = x[:, src_idx, :]
        used_targets.add(dest_idx)
        active.append(dest_idx)

    if not active:
        raise ValueError(
            f"BENDR: 0 channels matched the 10-20 layout. "
            f"Input channel names: {channel_names!r}, "
            f"normalized: {normalized!r}"
        )

    return mapped, active


def _per_seq_minmax(x: torch.Tensor, active_indices: List[int]) -> torch.Tensor:
    """Min-max scale to [-1,1] over active channels only."""
    if not active_indices:
        return x
    active = x[:, active_indices]
    flat = active.reshape(active.shape[0], -1)
    min_val = flat.min(dim=1).values.view(-1, 1, 1)
    max_val = flat.max(dim=1).values.view(-1, 1, 1)
    denom = (max_val - min_val).clamp_min(1e-6)
    x = x.clone()
    x[:, active_indices] = ((active - min_val) / denom) * 2.0 - 1.0
    return x


def _extract_state_dict(payload) -> dict:
    if isinstance(payload, dict):
        if "encoder_state_dict" in payload and "contextualizer_state_dict" in payload:
            flat = {}
            for k, v in payload["encoder_state_dict"].items():
                flat[f"encoder.{k}"] = v
            for k, v in payload["contextualizer_state_dict"].items():
                flat[f"contextualizer.{k}"] = v
            return flat
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    cleaned = {}
    for key, value in payload.items():
        updated = str(key)
        for prefix in ("module.", "model."):
            if updated.startswith(prefix):
                updated = updated[len(prefix) :]
        cleaned[updated] = value
    return cleaned


class BENDREncoder(FoundationEncoder):
    """BENDR maps input channels to a fixed 20-channel 10-20 layout."""

    MODEL_NAME = "bendr"
    PIPELINE_PRESET = "bendr"
    CHANNEL_STRATEGY = "layout"

    def __init__(
        self,
        in_chan: int,
        checkpoint_path: str | None = None,
        channel_names=None,
        channel_map=None,
        **kwargs,
    ):
        from physioex.models.foundation_checkpoints import ensure_checkpoint

        self._channel_names = channel_names or []
        self._channel_map = channel_map or {}
        resolved = ensure_checkpoint("bendr", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )
        super().__init__(
            in_chan=in_chan,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        from braindecode.models import BENDR

        n_chans = len(_BENDR_TARGET_CHANNELS)
        target_len = 7680  # 30s × 256Hz
        model = BENDR(
            n_chans=n_chans,
            n_chans_pretrained=n_chans,
            n_times=target_len,
            input_window_seconds=30.0,
            sfreq=256.0,
            n_outputs=1,
            final_layer=False,
            encoder_only=True,
        )
        return model

    def _get_embedding_dim(self) -> int:
        return 2048

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            # Try from_pretrained via HuggingFace
            from braindecode.models import BENDR

            n_chans = len(_BENDR_TARGET_CHANNELS)
            try:
                model = BENDR.from_pretrained(
                    "braindecode/braindecode-bendr",
                    n_chans=n_chans,
                    n_chans_pretrained=n_chans,
                    n_times=7680,
                    input_window_seconds=30.0,
                    sfreq=256.0,
                    n_outputs=1,
                    final_layer=False,
                    encoder_only=True,
                    strict=False,
                )
                self.encoder.load_state_dict(model.state_dict(), strict=False)
            except Exception:
                pass  # Use random init if HF download fails
            return
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _extract_state_dict(payload)
        self.encoder.load_state_dict(state, strict=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 20, T) with DC removal + layout mapping + minmax."""
        x = mean_center(x)

        # Map to BENDR 20-channel layout
        x, active = _map_to_bendr_layout(x, self._resolved_names)

        # Per-sequence min-max scaling on active channels
        x = _per_seq_minmax(x, active)

        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 20, T) -> (B, 2048) embeddings."""
        output = self.encoder(x, return_features=True)
        features = output["features"]
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        return features
