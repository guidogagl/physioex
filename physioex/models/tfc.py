"""TF-C — Time-Frequency Consistency (Zhang et al., NeurIPS 2022).

Single-channel model: selects best channel, center-crops to 178 samples.
Embedding = concat(h_time, h_freq) -> (B, 356).

Preprocessing (pure PyTorch, differentiable):
  1. Select best EEG channel by preference order
  2. Center-crop to 178 samples

Vendored from: https://github.com/mims-harvard/TFC-pretraining
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from physioex.models.foundation_base import FoundationEncoder

_TFC_CHANNEL_PREFERENCE = ["FPZ", "FZ", "CZ", "C3", "C4", "FP1", "F3", "F4", "O1"]


@dataclass(frozen=True)
class TFCConfig:
    ts_length_aligned: int = 178
    input_channels: int = 1
    nhead: int = 2
    num_layers: int = 2
    dim_feedforward_mult: int = 2
    proj_hidden: int = 256
    proj_out: int = 128


class TFC(nn.Module):
    """TFC encoder without classifier head."""

    def __init__(self, cfg: TFCConfig = TFCConfig()):
        super().__init__()
        d = cfg.ts_length_aligned
        enc_t = TransformerEncoderLayer(
            d, dim_feedforward=cfg.dim_feedforward_mult * d, nhead=cfg.nhead
        )
        self.transformer_encoder_t = TransformerEncoder(enc_t, cfg.num_layers)
        self.projector_t = nn.Sequential(
            nn.Linear(d, cfg.proj_hidden),
            nn.BatchNorm1d(cfg.proj_hidden),
            nn.ReLU(),
            nn.Linear(cfg.proj_hidden, cfg.proj_out),
        )
        enc_f = TransformerEncoderLayer(
            d, dim_feedforward=cfg.dim_feedforward_mult * d, nhead=cfg.nhead
        )
        self.transformer_encoder_f = TransformerEncoder(enc_f, cfg.num_layers)
        self.projector_f = nn.Sequential(
            nn.Linear(d, cfg.proj_hidden),
            nn.BatchNorm1d(cfg.proj_hidden),
            nn.ReLU(),
            nn.Linear(cfg.proj_hidden, cfg.proj_out),
        )

    def forward(self, x_time: torch.Tensor, x_freq: torch.Tensor):
        h_time = self.transformer_encoder_t(x_time)
        h_time = h_time.reshape(h_time.shape[0], -1)
        z_time = self.projector_t(h_time)
        h_freq = self.transformer_encoder_f(x_freq)
        h_freq = h_freq.reshape(h_freq.shape[0], -1)
        z_freq = self.projector_f(h_freq)
        return h_time, z_time, h_freq, z_freq


def to_freq_domain(x: torch.Tensor) -> torch.Tensor:
    """Upstream freq-domain transform: abs(fft(x))."""
    return fft.fft(x).abs()


def load_pretrained_tfc(checkpoint_path: str) -> tuple[TFC, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = TFC(TFCConfig())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(missing)[:10],
        "unexpected_keys": list(unexpected)[:10],
    }


class TFCEncoder(FoundationEncoder):
    """TF-C: single-channel time + frequency transformer."""

    MODEL_NAME = "tfc"
    PIPELINE_PRESET = "tfc"
    CHANNEL_STRATEGY = "first"

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
        resolved = ensure_checkpoint("tfc", checkpoint_path)
        self._ckpt_path = resolved
        self._resolved_names = (
            [self._channel_map.get(ch, ch) for ch in self._channel_names]
            if self._channel_names
            else []
        )
        self._cfg = TFCConfig()
        self._best_ch = self._find_best_channel()
        super().__init__(
            in_chan=in_chan,
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
        return 356

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            return
        model, _ = load_pretrained_tfc(checkpoint_path)
        self.encoder.load_state_dict(model.state_dict(), strict=True)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, 1, 178)."""
        x = x[:, self._best_ch : self._best_ch + 1, :]
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
