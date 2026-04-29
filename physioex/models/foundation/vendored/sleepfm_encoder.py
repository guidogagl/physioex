"""Vendored SleepFM SetTransformer encoder.

Source: https://github.com/zou-group/sleepfm-clinical
(sleepfm/models/models.py). Only the SetTransformer and its dependencies
(Tokenizer, AttentionPooling, PositionalEncoding) are kept here — the
fine-tuning/diagnosis heads are intentionally dropped.

Licence: see the sleepfm-clinical repository for licence terms.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from einops import rearrange


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
    """SleepFM's per-modality encoder.

    forward(x, mask) expects:
        x:    (B, C, T) where T is a multiple of patch_size; C is the padded
              channel slot count for this modality (10 for BAS = brain activity
              signals in the released model_base).
        mask: (B, C) boolean padding mask — True marks padding (i.e. invalid/
              zero-filled channels); False marks real signal.
    Returns (pooled, embedding):
        pooled:    (B, embed_dim)
        embedding: (B, S, embed_dim) where S = T // patch_size (per-patch tokens).
    """

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
