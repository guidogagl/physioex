"""REVE — Representations from EEG via Vision-language model (Chau et al., 2024).

Pure encoder: (B, L, C, T) -> (B, L, D)
Channel-name-aware via position bank. Per-recording z-score + ±15σ clip.

Embeddings: mean over patches and channels -> (B, 512).

Preprocessing (pure PyTorch, differentiable):
  1. Mean-center per channel
  2. Per-sample z-score + ±15σ clip
  3. Strip all-zero channels
  4. Resolve channel names for position bank
"""

from typing import Dict, List

import torch
import torch.nn as nn

from physioex.models.foundation_base import FoundationEncoder
from physioex.models.foundation_preproc import (
    mean_center,
    zscore_per_recording,
    strip_zero_channels,
)

def _import_automodel():
    try:
        from transformers import AutoModel
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "The REVE encoder requires the 'transformers' package. "
            "Install it with: pip install 'physioex[foundation]'"
        ) from e
    return AutoModel


_SIGMA_CLIP = 15.0

_BIPOLAR_TO_STANDARD = {
    "C3-A2": "C3",
    "C4-A1": "C4",
    "O1-A2": "O1",
    "O2-A1": "O2",
    "F3-A2": "F3",
    "F4-A1": "F4",
    "FZ-CZ": "FZ",
    "CZ-PZ": "PZ",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


class REVEEncoder(FoundationEncoder):
    """REVE with position bank for channel-aware embeddings."""

    MODEL_NAME = "reve"
    PIPELINE_PRESET = "reve"
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
        resolved = ensure_checkpoint("reve", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )
        # Resolve to standard 10-20 names for position bank
        self._standard_names = (
            [_BIPOLAR_TO_STANDARD.get(n, n) for n in self._resolved_names]
            if self._resolved_names
            else ["Fpz"]
        )
        super().__init__(
            in_chan=in_chan,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        AutoModel = _import_automodel()

        model = AutoModel.from_pretrained(
            self._ckpt_path or "brain-bzh/reve-base",
            trust_remote_code=True,
        )
        return model

    def _get_embedding_dim(self) -> int:
        return 512

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        # Already loaded in _build_encoder via from_pretrained
        pass

    def _build_pos_bank(self):
        """Build position bank (lazy, first call only)."""
        if not hasattr(self, "_pos_bank"):
            from physioex.models.foundation_checkpoints import ensure_reve_positions

            AutoModel = _import_automodel()
            pos_path = ensure_reve_positions()
            self._pos_bank = AutoModel.from_pretrained(
                pos_path,
                trust_remote_code=True,
            )
            device = next(self.encoder.parameters()).device
            self._pos_bank.eval()
            self._pos_bank.to(device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C', T) with DC removal + per-recording zscore + clip."""
        x = mean_center(x)
        x = zscore_per_recording(x, sigma_clip=_SIGMA_CLIP)
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 512) embeddings."""
        # Strip all-zero channels
        x_clean, kept = strip_zero_channels(x)
        ch_names = (
            [self._standard_names[i] for i in kept]
            if len(kept) < len(self._standard_names)
            else self._standard_names
        )

        # Build position bank on first use
        self._build_pos_bank()

        # Get position embeddings for channel names
        positions = (
            self._pos_bank(ch_names).unsqueeze(0).expand(x_clean.shape[0], -1, -1)
        )

        # Forward
        out = self.encoder(x_clean, positions)
        # out: (B, C, num_patches, 512) -> pool to (B, 512)
        return out.mean(dim=2).mean(dim=1)
