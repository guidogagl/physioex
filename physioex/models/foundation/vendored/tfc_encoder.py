"""Vendored TF-C (Time-Frequency Consistency) encoder.

Source: https://github.com/mims-harvard/TFC-pretraining
The pretrained weights are from the SleepEEG → Epilepsy run.

We only keep the frozen time-branch and frequency-branch encoders plus the
projection heads; the downstream classifier is discarded.

Licence: see the TFC-pretraining repository for licence terms.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.fft as fft
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


@dataclass(frozen=True)
class TFCConfig:
    ts_length_aligned: int = 178
    input_channels: int = 1
    # Transformer encoder params from the released checkpoint:
    nhead: int = 2
    num_layers: int = 2
    dim_feedforward_mult: int = 2  # dim_feedforward = 2 * ts_length_aligned
    # Projection head params:
    proj_hidden: int = 256
    proj_out: int = 128


class TFC(nn.Module):
    """Direct port of TFC-pretraining/code/TFC/model.py::TFC (without the classifier head).

    forward(x_time, x_freq) takes two tensors of shape (B, 1, TSlength_aligned).
    Returns (h_time, z_time, h_freq, z_freq) exactly as the upstream model does.
    For linear probing we use torch.cat([h_time, h_freq], dim=-1) of shape
    (B, 2*TSlength_aligned) = (B, 356) by default.
    """

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
        h_time = h_time.reshape(h_time.shape[0], -1)  # (B, 178)
        z_time = self.projector_t(h_time)  # (B, 128)
        h_freq = self.transformer_encoder_f(x_freq)
        h_freq = h_freq.reshape(h_freq.shape[0], -1)  # (B, 178)
        z_freq = self.projector_f(h_freq)  # (B, 128)
        return h_time, z_time, h_freq, z_freq


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


def to_freq_domain(x: torch.Tensor) -> torch.Tensor:
    """Upstream freq-domain transform: abs(fft(x))."""
    return fft.fft(x).abs()
