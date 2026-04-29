"""SleepFM — Sleep Foundation Model (Thapa et al., ICML 2024 / Nature Medicine 2025).

Self-contained wrapper: uses vendored SetTransformer, no EEGBenchmarks.
BAS (brain-activity-signals) branch: pads to 10 channel slots.

Embeddings: SetTransformer temporal pooling -> (B, 128).

Preprocessing (pure PyTorch, differentiable):
  1. Per-sample z-score normalization
  2. Strip all-zero channels
  3. Pad to 10 BAS channel slots
  4. Create padding mask
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from physioex.models.foundation._base import FoundationModelWrapper
from physioex.models.foundation._preproc import (
    zscore,
    strip_zero_channels,
    pad_channels,
    create_padding_mask,
)
from physioex.models.foundation.vendored.sleepfm_encoder import SetTransformer


_BAS_CHANNELS = 10
_DEFAULT_CONFIG = {
    "in_channels": 1,
    "patch_size": 640,  # 5 s × 128 Hz
    "embed_dim": 128,
    "num_heads": 8,
    "num_layers": 6,
    "pooling_head": 8,
    "dropout": 0.0,
    "sampling_freq": 128,
    "BAS_CHANNELS": 10,
    "max_seq_length": 128,
}


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


class SleepFMSleepNet(FoundationModelWrapper):
    """SleepFM pads input to 10 channel slots and masks unused ones."""

    MODEL_NAME = "sleepfm"
    PIPELINE_PRESET = "sleepfm"
    CHANNEL_STRATEGY = "pad_fixed"

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
        resolved = ensure_checkpoint("sleepfm", checkpoint_path)
        self._ckpt_path = resolved

        # Load companion config if present
        cfg_path = Path(resolved).parent / "config.json"
        self._cfg = dict(_DEFAULT_CONFIG)
        if cfg_path.exists():
            on_disk = json.loads(cfg_path.read_text())
            for key in (
                "patch_size",
                "embed_dim",
                "num_heads",
                "num_layers",
                "pooling_head",
                "sampling_freq",
                "BAS_CHANNELS",
            ):
                if key in on_disk:
                    self._cfg[key] = on_disk[key]

        self._bas_channels = int(self._cfg["BAS_CHANNELS"])

        super().__init__(
            n_classes=n_classes,
            in_chan=in_chan,
            sequence_length=sequence_length,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        return SetTransformer(
            in_channels=int(self._cfg["in_channels"]),
            patch_size=int(self._cfg["patch_size"]),
            embed_dim=int(self._cfg["embed_dim"]),
            num_heads=int(self._cfg["num_heads"]),
            num_layers=int(self._cfg["num_layers"]),
            pooling_head=int(self._cfg["pooling_head"]),
            dropout=0.0,  # eval: no dropout
            max_seq_length=int(self._cfg["max_seq_length"]),
        )

    def _get_embedding_dim(self) -> int:
        return int(self._cfg["embed_dim"])

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            return
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = (
            payload["state_dict"]
            if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )
        state = _strip_module_prefix(state)
        self.encoder.load_state_dict(state, strict=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 10, T) with z-score + pad to BAS channels."""
        # z-score normalization
        x = zscore(x)

        # Strip all-zero channels
        x, kept = strip_zero_channels(x)
        n_valid = x.shape[1]

        # Pad to BAS_CHANNELS
        x = pad_channels(x, self._bas_channels)

        # Store valid count for mask creation in _encode
        self._n_valid = min(n_valid, self._bas_channels)

        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 10, T) -> (B, 128) embeddings."""
        B = x.shape[0]
        # Padding mask: True = padded/invalid
        mask = create_padding_mask(
            self._bas_channels,
            self._n_valid,
            B,
            x.device,
        )
        pooled, _ = self.encoder(x, mask)
        return pooled
