"""CoRe-Sleep: Coordinated Representation multimodal fusion for sleep staging.

Reimplementation of Kontras et al. 2024 (IEEE TNSRE) for PhysioEx.
Supports bimodal (EEG + EOG) and unimodal (EEG-only) operation.

Reference:
    Kontras, K., Chatzichristos, C., Phan, H., Suykens, J., & De Vos, M.
    "CoRe-Sleep: A Multimodal Fusion Framework for Time Series Robust to
    Imperfect Modalities", IEEE TNSRE, 2024.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding added to token sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        return self.dropout(x + self.pos_embed[:, : x.size(1)])


class _InnerEncoder(nn.Module):
    """Intra-epoch transformer encoder.

    Processes the STFT time-frequency sequence within a single 30-second
    epoch.  A learnable [CLS] token aggregates the sequence into a single
    embedding vector per epoch.

    Input:  (B, T, D)  where T = STFT time steps (~29)
    Output: (B, D)     the [CLS] token embedding
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.3,
        max_len: int = 64,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = _LearnedPositionalEncoding(d_model, max_len + 1, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, D)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x[:, 0]  # (B, D) — CLS token


class _CrossModalAttention(nn.Module):
    """Bidirectional cross-attention between two modality streams.

    Each modality attends to the other, producing cross-modal features.
    The combined output is the sum of the two cross-attended streams.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.cross_attn_eeg = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_eog = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_eeg = nn.LayerNorm(d_model)
        self.norm_eog = nn.LayerNorm(d_model)

    def forward(self, eeg, eog):
        # eeg, eog: (B, L, D)
        eeg_cross, _ = self.cross_attn_eeg(
            query=self.norm_eeg(eeg), key=eog, value=eog
        )
        eog_cross, _ = self.cross_attn_eog(
            query=self.norm_eog(eog), key=eeg, value=eeg
        )
        return eeg + eeg_cross + eog + eog_cross  # (B, L, D)


class _OuterEncoder(nn.Module):
    """Inter-epoch transformer encoder.

    Processes the sequence of epoch embeddings to capture temporal
    context across the full recording window (e.g. 21 epochs = ~10 min).

    Input:  (B, L, D)
    Output: (B, L, D)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.3,
        max_len: int = 256,
    ):
        super().__init__()
        self.pos_enc = _LearnedPositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.pos_enc(x)
        return self.encoder(x)


class CoReSleep(nn.Module):
    """CoRe-Sleep: multimodal sleep staging with cross-modal attention.

    Accepts spectrogram input ``(B, L, C, T, F)`` where C=2 (EEG + EOG)
    or C=1 (EEG only).  When C=1, cross-modal attention is bypassed.

    Args:
        n_classes: Number of sleep stages (default 5: W, N1, N2, N3, REM).
        in_chan: Number of input channels. 1 = EEG only, 2 = EEG + EOG.
        F: Number of STFT frequency bins (default 129 for nfft=256).
        d_model: Transformer model dimension (default 128).
        n_heads: Number of attention heads (default 8).
        n_inner_layers: Inner (intra-epoch) transformer layers (default 4).
        n_outer_layers: Outer (inter-epoch) transformer layers (default 4).
        d_ff: Feed-forward dimension (default 1024).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 2,
        F: int = 129,
        d_model: int = 128,
        n_heads: int = 8,
        n_inner_layers: int = 4,
        n_outer_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.d_model = d_model

        # Input projection: frequency bins → d_model
        self.input_proj = nn.Linear(F, d_model)

        # Shared inner encoder (intra-epoch, processes STFT time steps)
        self.inner_encoder = _InnerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_inner_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Cross-modal attention (only used when in_chan >= 2)
        if in_chan >= 2:
            self.cross_attention = _CrossModalAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout
            )

        # Outer encoder (inter-epoch, captures temporal context)
        self.outer_encoder = _OuterEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_outer_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Classification head
        self.clf = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes),
        )

    def _encode_modality(self, x):
        """Encode a single modality through input projection + inner encoder.

        Args:
            x: (B*L, T, F) spectrogram for one modality.

        Returns:
            (B*L, d_model) per-epoch embedding.
        """
        x = self.input_proj(x)  # (B*L, T, d_model)
        return self.inner_encoder(x)  # (B*L, d_model)

    def encode(self, x):
        """Encode input spectrograms to contextualized per-epoch embeddings.

        Args:
            x: (B, L, C, T, F) spectrogram input. C=1 or C=2.

        Returns:
            (B, L, d_model) contextualized epoch embeddings.
        """
        B, L, C, T, F = x.size()

        if C >= 2:
            # Bimodal: separate EEG (ch0) and EOG (ch1)
            eeg = x[:, :, 0].reshape(B * L, T, F)  # (B*L, T, F)
            eog = x[:, :, 1].reshape(B * L, T, F)

            eeg_emb = self._encode_modality(eeg).reshape(B, L, -1)  # (B, L, D)
            eog_emb = self._encode_modality(eog).reshape(B, L, -1)

            # Cross-modal fusion
            fused = self.cross_attention(eeg_emb, eog_emb)  # (B, L, D)
        else:
            # Unimodal: EEG only
            eeg = x[:, :, 0].reshape(B * L, T, F)
            fused = self._encode_modality(eeg).reshape(B, L, -1)  # (B, L, D)

        # Inter-epoch context
        return self.outer_encoder(fused)  # (B, L, D)

    def forward(self, x):
        """Forward pass for sleep stage classification.

        Args:
            x: (B, L, C, T, F) spectrogram input.

        Returns:
            (B, L, n_classes) per-epoch logits.
        """
        x = self.encode(x)  # (B, L, d_model)

        B, L, D = x.size()
        x = x.reshape(B * L, D)
        x = self.clf(x)
        return x.reshape(B, L, -1)


if __name__ == "__main__":
    from torchinfo import summary

    # Bimodal test
    model = CoReSleep(n_classes=5, in_chan=2)
    print("CoReSleep (bimodal EEG+EOG):")
    summary(model, (2, 21, 2, 29, 129))

    # Unimodal test
    model_uni = CoReSleep(n_classes=5, in_chan=1)
    print("\nCoReSleep (unimodal EEG):")
    summary(model_uni, (2, 21, 1, 29, 129))
