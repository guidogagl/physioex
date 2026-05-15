"""SleepFM — Sleep Foundation Model (Thapa et al., ICML 2024 / Nature Medicine 2025).

BAS (brain-activity-signals) branch: pads to 10 channel slots.
Embeddings: SetTransformer temporal pooling -> (B, 128).

Preprocessing (pure PyTorch, differentiable):
  1. Per-sample z-score normalization
  2. Strip all-zero channels
  3. Pad to 10 BAS channel slots
  4. Create padding mask

Vendored from: https://github.com/zou-group/sleepfm-clinical
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

from physioex.models.foundation_base import FoundationEncoder
from physioex.models.foundation_preproc import (
    zscore,
    strip_zero_channels,
    pad_channels,
    create_padding_mask,
)

_BAS_CHANNELS = 10
_DEFAULT_CONFIG = {
    "in_channels": 1,
    "patch_size": 640,
    "embed_dim": 128,
    "num_heads": 8,
    "num_layers": 6,
    "pooling_head": 8,
    "dropout": 0.0,
    "sampling_freq": 128,
    "BAS_CHANNELS": 10,
    "max_seq_length": 128,
}


class Tokenizer(nn.Module):
    def __init__(self, input_size: int = 640, output_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tokenizer = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, input_size // 2]),
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, input_size // 4]),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, input_size // 8]),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, input_size // 16]),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, input_size // 32]),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, input_size // 64]),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.view(B, C, -1, self.input_size)
        x = x.permute(0, 1, 2, 3).contiguous().view(-1, 1, self.input_size)
        x = self.tokenizer(x)
        x = x.view(B, C, -1, self.output_size)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            if key_padding_mask.size(1) == 1:
                return x.mean(dim=1)
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)
            transformer_output = self.transformer_layer(
                x, src_key_padding_mask=key_padding_mask
            )
            attention_mask = (~key_padding_mask).float().unsqueeze(-1)
            pooled_output = (transformer_output * attention_mask).sum(
                dim=1
            ) / attention_mask.sum(dim=1).clamp(min=1)
        else:
            transformer_output = self.transformer_layer(x)
            pooled_output = transformer_output.mean(dim=1)
        return pooled_output


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SetTransformer(nn.Module):
    """SleepFM's per-modality encoder."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        pooling_head: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 128,
    ):
        super().__init__()
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)
        self.spatial_pooling = AttentionPooling(
            embed_dim, num_heads=pooling_head, dropout=dropout
        )
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.temporal_pooling = AttentionPooling(
            embed_dim, num_heads=pooling_head, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.patch_embedding(x)
        B, C, S, E = x.shape
        x = rearrange(x, "b c s e -> (b s) c e")
        mask = mask.unsqueeze(1).expand(-1, S, -1)
        mask = rearrange(mask, "b t c -> (b t) c")
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        x = self.spatial_pooling(x, mask)
        x = x.view(B, S, E)
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        embedding = x.clone()
        x = self.temporal_pooling(x)
        return x, embedding


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


class SleepFMEncoder(FoundationEncoder):
    """SleepFM pads input to 10 channel slots and masks unused ones."""

    MODEL_NAME = "sleepfm"
    PIPELINE_PRESET = "sleepfm"
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
        resolved = ensure_checkpoint("sleepfm", checkpoint_path)
        self._ckpt_path = resolved

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
            in_chan=in_chan,
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
            dropout=0.0,
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
        x = zscore(x)
        x, kept = strip_zero_channels(x)
        n_valid = x.shape[1]
        x = pad_channels(x, self._bas_channels)
        self._n_valid = min(n_valid, self._bas_channels)
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 10, T) -> (B, 128) embeddings."""
        B = x.shape[0]
        mask = create_padding_mask(
            self._bas_channels,
            self._n_valid,
            B,
            x.device,
        )
        pooled, _ = self.encoder(x, mask)
        return pooled
