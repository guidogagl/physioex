"""CBraMod — Criss-Cross Brain Foundation Model (Jiang et al., ICLR 2025).

Self-contained wrapper: imports from braindecode only, no EEGBenchmarks.
Embeddings: mean over patches and channels -> (B, 200).

Preprocessing (pure PyTorch, differentiable):
  1. Mean-center per channel
  2. Clip ±100 µV, divide by 100 → ~[-1,1]
  3. Strip all-zero channels
"""
from __future__ import annotations

import torch
import torch.nn as nn

from physioex.models.foundation._base import FoundationModelWrapper
from physioex.models.foundation._preproc import (
    mean_center,
    clip_and_scale,
    strip_zero_channels,
)


class CBraModSleepNet(FoundationModelWrapper):
    """CBraMod accepts any number of EEG channels, pools over them."""

    MODEL_NAME = "cbramod"
    PIPELINE_PRESET = "cbramod"
    CHANNEL_STRATEGY = "all"

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

        self._channel_names = channel_names
        self._channel_map = channel_map
        resolved = ensure_checkpoint("cbramod", checkpoint_path)
        self._ckpt_path = resolved
        super().__init__(
            n_classes=n_classes,
            in_chan=in_chan,
            sequence_length=sequence_length,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        from braindecode.models import CBraMod

        if self._ckpt_path is None:
            model = CBraMod.from_pretrained(
                "braindecode/cbramod-pretrained",
                return_encoder_output=True,
            )
        else:
            model = CBraMod(
                n_chans=1,
                n_times=6000,
                sfreq=200,
                return_encoder_output=True,
            )
        return model

    def _get_embedding_dim(self) -> int:
        return 200

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(state, strict=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C', T) with mean-center + clip/scale."""
        x = mean_center(x)
        x = clip_and_scale(x, max_uv=100.0)
        # Strip all-zero channels (e.g., missing channels after dataset selection)
        x, _ = strip_zero_channels(x)
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 200) embeddings."""
        out = self.encoder(x)  # (B, C, n_patch, 200)
        return out.mean(dim=(1, 2))  # (B, 200)
