"""BIOT — Biosignal IO Transformer (Yang et al., NeurIPS 2023).

Pure encoder: (B, L, C, T) -> (B, L, D)
Fixed 16-channel double-banana montage. Derives bipolar pairs from
available unipolar electrodes. Per-channel q95 normalization.

Embeddings: encoder internal mean -> (B, 256).

Preprocessing (pure PyTorch, differentiable):
  1. Mean-center per channel
  2. Map to 16-channel double-banana layout (derive bipolar pairs)
  3. Per-channel q95 normalization
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from physioex.models.foundation_base import FoundationEncoder
from physioex.models.foundation_preproc import mean_center, q95_normalize

_N_CHANNELS = 16

_BIOT_SLOT_ORDER = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]

_BIOT_DERIVATION = [
    ("FP1", "F7"),
    ("F7", "T7"),
    ("T7", "P7"),
    ("P7", "O1"),
    ("FP2", "F8"),
    ("F8", "T8"),
    ("T8", "P8"),
    ("P8", "O2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

_10_20_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _strip_encoder_prefix(state: dict) -> dict:
    if any(key.startswith("encoder.") for key in state.keys()):
        return {
            key[len("encoder.") :]: val
            for key, val in state.items()
            if key.startswith("encoder.")
        }
    return state


def _map_to_biot_layout(
    x: torch.Tensor,
    channel_names: List[str],
) -> Tuple[torch.Tensor, List[int]]:
    """Map unipolar channels to BIOT's 16-ch double-banana layout.

    For each target slot, tries in order:
      1. Direct match (e.g., input already has "FP1-F7")
      2. Bipolar derivation (subtract electrode B from A)
      3. Approximate (use only electrode A)
      4. Zero-pad

    Returns: (mapped_x, active_indices)
    """
    B, C_in, T = x.shape

    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(channel_names):
        upper = name.upper().strip()
        name_to_idx[upper] = i
        canonical = _10_20_ALIASES.get(upper)
        if canonical and canonical not in name_to_idx:
            name_to_idx[canonical] = i

    mapped = torch.zeros(B, _N_CHANNELS, T, dtype=x.dtype, device=x.device)
    active: List[int] = []

    for slot_idx, slot_name in enumerate(_BIOT_SLOT_ORDER):
        elec_a, elec_b = _BIOT_DERIVATION[slot_idx]

        # Direct match
        if slot_name in name_to_idx:
            mapped[:, slot_idx, :] = x[:, name_to_idx[slot_name], :]
            active.append(slot_idx)
            continue

        idx_a = name_to_idx.get(elec_a)
        idx_b = name_to_idx.get(elec_b)

        if idx_a is not None and idx_b is not None:
            # Bipolar derivation
            mapped[:, slot_idx, :] = x[:, idx_a, :] - x[:, idx_b, :]
            active.append(slot_idx)
        elif idx_a is not None:
            # Approximate: use electrode A only
            mapped[:, slot_idx, :] = x[:, idx_a, :]
            active.append(slot_idx)
        elif idx_b is not None:
            # Approximate: use electrode B only
            mapped[:, slot_idx, :] = x[:, idx_b, :]
            active.append(slot_idx)
        # else: zero-padded

    return mapped, active


class BIOTEncoder(FoundationEncoder):
    """BIOT with 16-channel double-banana layout + q95 normalization."""

    MODEL_NAME = "biot"
    PIPELINE_PRESET = "biot"
    CHANNEL_STRATEGY = "pad_fixed"

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
        resolved = ensure_checkpoint("biot", checkpoint_path)
        self._ckpt_path = resolved
        # Resolve standard names from channel_map
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
        from braindecode.models import BIOT

        model = BIOT(
            n_chans=_N_CHANNELS,
            n_times=6000,  # 30s × 200Hz
            sfreq=200,
            n_outputs=5,
            return_feature=True,
        )
        return model

    def _get_embedding_dim(self) -> int:
        return 256

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            return
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _strip_encoder_prefix(state)
        self.encoder.encoder.load_state_dict(state, strict=True)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 16, T) with DC removal + bipolar mapping + q95."""
        x = mean_center(x)

        # Map to 16-channel double-banana layout
        x, active = _map_to_biot_layout(x, self._resolved_names)

        # q95 normalization only on active (non-zero-padded) channels
        if active:
            active_t = torch.tensor(active, device=x.device, dtype=torch.long)
            x_active = q95_normalize(x[:, active_t, :])
            x = x.clone()
            x[:, active_t, :] = x_active

        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 16, T) -> (B, 256) embeddings."""
        _, emb = self.encoder(x)  # BIOT returns (logits, embeddings)
        return emb
