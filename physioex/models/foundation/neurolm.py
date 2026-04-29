"""NeuroLM — Neural Language Model (Liu et al., 2024).

Self-contained wrapper: uses vendored NeuralTransformer, no EEGBenchmarks.
Patch-based tokenization at 200 Hz. Channels mapped to VQ vocabulary indices.

Embeddings: masked mean over valid tokens -> (B, 768).

Preprocessing (pure PyTorch, differentiable):
  1. x / 100 scaling (paper §3.2)
  2. Strip all-zero channels
  3. Tokenize: patchify signal + map channels to VQ vocabulary indices
"""
from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from physioex.models.foundation._base import FoundationModelWrapper
from physioex.models.foundation._preproc import scale_div, strip_zero_channels
from physioex.models.foundation.vendored.neurolm_encoder import (
    NeuralTransformer,
    NTConfig,
)

logger = logging.getLogger("physioex.foundation")

_SCALE_DIVISOR = 100.0

# NeuroLM's VQ channel vocabulary (standard 10-20 + extensions)
_STANDARD_1020 = [
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFZ",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POZ",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "O9",
    "CB1",
    "CB2",
    "IZ",
    "O10",
    "T3",
    "T5",
    "T4",
    "T6",
    "M1",
    "M2",
    "A1",
    "A2",
    "CFC1",
    "CFC2",
    "CFC3",
    "CFC4",
    "CFC5",
    "CFC6",
    "CFC7",
    "CFC8",
    "CCP1",
    "CCP2",
    "CCP3",
    "CCP4",
    "CCP5",
    "CCP6",
    "CCP7",
    "CCP8",
    "T1",
    "T2",
    "FTT9h",
    "TTP7h",
    "TPP9h",
    "FTT10h",
    "TPP8h",
    "TPP10h",
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
    "pad",
    "I1",
    "I2",
]

_PAD_IDX = _STANDARD_1020.index("pad")

_BIPOLAR_TO_STANDARD = {
    "C3-A2": "C3",
    "C4-A1": "C4",
    "O1-A2": "O1",
    "O2-A1": "O2",
    "F3-A2": "F3",
    "F4-A1": "F4",
    "FZ-CZ": "FZ",
    "CZ-PZ": "PZ",
}


def _channel_idx(name: str) -> int:
    upper = name.upper()
    if upper in _STANDARD_1020:
        return _STANDARD_1020.index(upper)
    std = _BIPOLAR_TO_STANDARD.get(name, name).upper()
    if std in _STANDARD_1020:
        return _STANDARD_1020.index(std)
    if name in _STANDARD_1020:
        return _STANDARD_1020.index(name)
    raise ValueError(
        f"NeuroLM: channel label {name!r} not in VQ vocabulary. "
        f"Known aliases: {sorted(_BIPOLAR_TO_STANDARD)!r}."
    )


def _extract_encoder_state(state: dict) -> tuple:
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    prefixes = ("_orig_mod.VQ.encoder.", "VQ.encoder.")
    extracted = {}
    used_prefix = None
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                used_prefix = prefix
                break
    if not extracted:
        raise KeyError("Could not find NeuroLM VQ encoder weights in checkpoint.")
    return extracted, used_prefix


class NeuroLMSleepNet(FoundationModelWrapper):
    """NeuroLM with patch-based tokenization and VQ channel vocabulary."""

    MODEL_NAME = "neurolm"
    PIPELINE_PRESET = "neurolm"
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

        self._channel_names = channel_names or []
        self._channel_map = channel_map or {}
        resolved = ensure_checkpoint("neurolm", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )

        # Pre-load checkpoint to get encoder config
        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        self._encoder_conf = NTConfig(**ckpt["encoder_args"])
        self._encoder_state, _ = _extract_encoder_state(ckpt)

        super().__init__(
            n_classes=n_classes,
            in_chan=in_chan,
            sequence_length=sequence_length,
            checkpoint_path=resolved,
            **kwargs,
        )

    def _build_encoder(self, **kwargs) -> nn.Module:
        return NeuralTransformer(self._encoder_conf)

    def _get_embedding_dim(self) -> int:
        return 768

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        self.encoder.load_state_dict(self._encoder_state, strict=True)
        # Free reference to state dict
        del self._encoder_state

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C', T) with /100 scaling + zero-strip."""
        x = scale_div(x, _SCALE_DIVISOR)

        # Strip all-zero channels
        if self._resolved_names:
            x_clean, kept = strip_zero_channels(x)
            self._kept_names = (
                [self._resolved_names[i] for i in kept]
                if len(kept) < len(self._resolved_names)
                else self._resolved_names
            )
        else:
            x_clean = x
            self._kept_names = self._resolved_names or ["FPZ"]

        self._clean_x = x_clean
        return x_clean

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C', T) -> (B, 768) embeddings via tokenization + transformer."""
        batch_size, n_chans, T = x.shape
        block_size = int(self._encoder_conf.block_size)
        patch_size = int(self._encoder_conf.patch_size)
        n_time_patches = T // patch_size

        ch_indices = [_channel_idx(c) for c in self._kept_names[:n_chans]]

        tokens = torch.zeros(
            batch_size, block_size, patch_size, device=x.device, dtype=torch.float32
        )
        input_chans = torch.full(
            (batch_size, block_size), _PAD_IDX, device=x.device, dtype=torch.long
        )
        input_times = torch.zeros(
            batch_size, block_size, device=x.device, dtype=torch.long
        )
        input_mask = torch.zeros(
            batch_size, block_size, device=x.device, dtype=torch.bool
        )

        time_idx = torch.arange(n_time_patches, device=x.device, dtype=torch.long)
        pos = 0
        for ch_i, ch_idx in enumerate(ch_indices):
            if pos + n_time_patches > block_size:
                break
            ch_signal = x[:, ch_i, :]
            patches = ch_signal.reshape(batch_size, n_time_patches, patch_size)
            tokens[:, pos : pos + n_time_patches] = patches
            input_chans[:, pos : pos + n_time_patches] = ch_idx
            input_times[:, pos : pos + n_time_patches] = time_idx
            input_mask[:, pos : pos + n_time_patches] = True
            pos += n_time_patches

        attn_mask = input_mask.unsqueeze(1).repeat(1, block_size, 1).unsqueeze(1)

        features = self.encoder.forward_features(
            tokens,
            input_chans=input_chans,
            input_times=input_times,
            mask=attn_mask,
            return_all_tokens=True,
        )

        # Masked mean over valid tokens
        valid = input_mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled
