"""TF-C — Time-Frequency Consistency (Zhang et al., NeurIPS 2022).

Self-contained wrapper: uses vendored TFC encoder, no EEGBenchmarks.
Single-channel model: selects best channel, center-crops to 178 samples.
Embedding = concat(h_time, h_freq) -> (B, 356).

Preprocessing (pure PyTorch, differentiable):
  1. Select best EEG channel by preference order
  2. Center-crop to 178 samples
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.models.foundation._base import FoundationModelWrapper
from physioex.models.foundation.vendored.tfc_encoder import (
    TFC,
    TFCConfig,
    load_pretrained_tfc,
    to_freq_domain,
)


_TFC_CHANNEL_PREFERENCE = ["FPZ", "FZ", "CZ", "C3", "C4", "FP1", "F3", "F4", "O1"]


class TFCSleepNet(FoundationModelWrapper):
    """TF-C: single-channel time + frequency transformer."""

    MODEL_NAME = "tfc"
    PIPELINE_PRESET = "tfc"
    CHANNEL_STRATEGY = "first"

    def __init__(
        self,
        n_classes,
        in_chan,
        sequence_length=1,
        checkpoint_path=None,
        channel_names=None,
        channel_map=None,
        **kwargs,
    ):
        from physioex.models.foundation._checkpoints import ensure_checkpoint

        self._channel_names = channel_names or []
        self._channel_map = channel_map or {}
        resolved = ensure_checkpoint("tfc", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )
        self._cfg = TFCConfig()
        # Find best channel index
        self._best_ch = self._find_best_channel()
        super().__init__(
            n_classes=n_classes,
            in_chan=in_chan,
            sequence_length=sequence_length,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _find_best_channel(self) -> int:
        if not self._resolved_names:
            return 0
        anchors = []
        for ch in self._resolved_names:
            anchor = ch.upper().replace(" ", "").split("-")[0].split("_")[0]
            anchor = re.sub(r"[AM]\d+$", "", anchor)
            anchors.append(anchor)
        for pref in _TFC_CHANNEL_PREFERENCE:
            for i, a in enumerate(anchors):
                if a == pref:
                    return i
        return 0

    def _build_encoder(self, **kwargs) -> nn.Module:
        return TFC(self._cfg)

    def _get_embedding_dim(self) -> int:
        return 356  # 178 + 178

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            return
        model, _ = load_pretrained_tfc(checkpoint_path)
        self.encoder.load_state_dict(model.state_dict(), strict=True)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 1, 178)."""
        # Select best channel
        x = x[:, self._best_ch : self._best_ch + 1, :]

        # Center-crop to ts_length_aligned (178)
        T_align = self._cfg.ts_length_aligned
        start = max(0, (x.shape[-1] - T_align) // 2)
        x = x[..., start : start + T_align]
        if x.shape[-1] < T_align:
            pad = T_align - x.shape[-1]
            x = F.pad(x, (0, pad))
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 178) -> (B, 356)."""
        x_freq = to_freq_domain(x)
        h_time, _, h_freq, _ = self.encoder(x, x_freq)
        return torch.cat([h_time, h_freq], dim=-1)
