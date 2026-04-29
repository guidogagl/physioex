"""Signal-JEPA — Joint Embedding Predictive Architecture (Guetschel et al., 2024).

Self-contained wrapper: imports from braindecode only, no EEGBenchmarks.
Fixed 19-channel standard 10-20 montage with spatial positional encoder.

Embeddings: mean over tokens -> (B, 64).

Preprocessing (pure PyTorch, differentiable):
  1. Map to 19-channel layout via alias matching
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from physioex.models.foundation._base import FoundationModelWrapper
from physioex.models.foundation._preproc import map_channels_to_layout

logger = logging.getLogger("physioex.foundation")

_SJEPA_TARGET_CHANNELS = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
)

_CHANNEL_ALIASES = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "FPZ-CZ": "FZ",
    "PZ-OZ": "PZ",
    "EEG FPZ-CZ": "FZ",
    "EEG PZ-OZ": "PZ",
    "C3-A2": "C3",
    "C4-A1": "C4",
    "O1-A2": "O1",
    "O2-A1": "O2",
    "F3-A2": "F3",
    "F4-A1": "F4",
}

_MIN_MATCH_RATIO = 0.5


class SJEPASleepNet(FoundationModelWrapper):
    """Signal-JEPA with fixed 19-channel 10-20 layout."""

    MODEL_NAME = "sjepa"
    PIPELINE_PRESET = "sjepa"
    CHANNEL_STRATEGY = "layout"

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
        resolved = ensure_checkpoint("sjepa", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )
        super().__init__(
            n_classes=n_classes,
            in_chan=in_chan,
            sequence_length=sequence_length,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        from braindecode.models import SignalJEPA

        model = SignalJEPA.from_pretrained("braindecode/SignalJEPA-pretrained")
        self._fix_spatial_pos_embedding(model)
        return model

    def _fix_spatial_pos_embedding(self, model) -> None:
        """Fix braindecode bug: _SpatialPosEmbedding reads loc[3:6] instead of loc[0:3]."""
        chs_info = getattr(model, "chs_info", None)
        if chs_info is None:
            return

        for name, module in model.named_modules():
            if type(module).__name__ not in (
                "_SpatialPosEmbedding",
                "_ChannelEmbedding",
            ):
                continue
            if not torch.isnan(module.weight.data).any().item():
                continue

            correct_locs = []
            for ch in chs_info:
                loc = ch.get("loc", None)
                if loc is not None and hasattr(loc, "__len__") and len(loc) >= 3:
                    correct_locs.append([float(loc[0]), float(loc[1]), float(loc[2])])
                else:
                    correct_locs.append(None)

            valid = [l for l in correct_locs if l is not None]
            if not valid:
                continue

            coords_by_dim = list(zip(*valid))
            new_ranges = [(min(c), max(c)) for c in coords_by_dim]
            mins, maxs = zip(*new_ranges)
            new_max_abs = max(abs(min(mins)), abs(max(maxs)))
            if new_max_abs < 1e-10:
                continue

            module.coordinate_ranges = new_ranges
            module.max_abs_coordinate = new_max_abs
            module.channel_locations = correct_locs
            module.reset_parameters()
            logger.warning(
                "SJEPA: patched %s spatial pos embedding (braindecode loc[3:6] bug)",
                name,
            )

    def _get_embedding_dim(self) -> int:
        return 64

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        # Already loaded in _build_encoder via from_pretrained
        pass

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 19, T) mapped to SJEPA layout."""
        target_layout = list(_SJEPA_TARGET_CHANNELS)
        mapped, active = map_channels_to_layout(
            x,
            self._resolved_names,
            target_layout,
            _CHANNEL_ALIASES,
        )

        # Check match ratio
        n_matched = len(active)
        match_ratio = n_matched / len(target_layout)
        if match_ratio < _MIN_MATCH_RATIO:
            raise ValueError(
                f"SJEPA: channel match ratio {match_ratio:.2f} < "
                f"{_MIN_MATCH_RATIO}. Input channels: {self._resolved_names!r}"
            )

        return mapped

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 19, T) -> (B, 64) embeddings."""
        out = self.encoder(x, return_features=True)
        features = out["features"]  # (B, L, 64)
        return features.mean(dim=1)  # (B, 64)
