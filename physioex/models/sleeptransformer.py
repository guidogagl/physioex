"""SleepTransformer (Phan et al. 2022) for automatic sleep staging.

Reference: "SleepTransformer: Automatic Sleep Staging with Interpretability
and Uncertainty Quantification", IEEE TBIOM 2022 (arXiv:2105.11043).

Architecture (from the paper):
  Input: STFT spectrogram (T=29 time frames, F=129 freq bins) per epoch
         Truncated to F=128 for transformer compatibility

  Epoch Transformer:
    - Sinusoidal positional encoding
    - 4 TransformerEncoder layers (d=128, 8 heads, ff=1024, dropout=0.1)
    - Attention pooling → 128-dim epoch embedding

  Sequence Transformer:
    - Input: L epoch embeddings (128-dim)
    - Sinusoidal positional encoding
    - 4 TransformerEncoder layers (d=128, 8 heads, ff=1024, dropout=0.1)

  Classification Head (per epoch):
    - FC(128 → 1024) + ReLU + Dropout(0.1)
    - FC(1024 → 1024) + ReLU + Dropout(0.1)
    - FC(1024 → n_classes)

PhysioEx integration:
  Input:  (B, L, C, T, F) — spectrograms from "seqsleepnet" pipeline preset
  Output: (B, L, n_classes) — per-epoch logits (sequence-to-sequence)
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    """Attention-based pooling over the time dimension.

    Computes a weighted combination of the input sequence using a
    learnable attention vector, collapsing (B, T, D) → (B, D).
    """

    def __init__(self, d_model: int, attention_size: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )

    def forward(self, x):
        # x: (B, T, D)
        weights = self.attention(x)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)  # (B, D)


class EpochTransformer(nn.Module):
    """Epoch-level transformer encoder.

    Processes a single epoch's spectrogram (T time frames × F freq bins)
    through a transformer and attention-pools to a single embedding vector.

    Input:  (B, C, T, F) — C channels, T=29 time frames, F=129 freq bins
    Output: (B, d_model) — epoch embedding
    """

    def __init__(
        self,
        d_model: int = 128,
        in_chan: int = 1,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attention_size: int = 128,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.freq_dim = d_model              # per-channel truncation target
        self.d_model = d_model * in_chan      # transformer hidden dimension

        self.pe = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attention_pool = AttentionPooling(self.d_model, attention_size)

    def forward(self, x):
        # x: (B, C, T, F) — spectrogram
        B, C, T, F = x.shape

        # Truncate frequency bins per channel, then concatenate
        x = x[..., : self.freq_dim]  # (B, C, T, freq_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * self.freq_dim)  # (B, T, d_model)

        x = self.pe(x)
        x = self.encoder(x)  # (B, T, d_model)
        x = self.attention_pool(x)  # (B, d_model)

        return x


class SequenceTransformer(nn.Module):
    """Sequence-level transformer encoder.

    Processes a sequence of epoch embeddings through a transformer.

    Input:  (B, L, d_model) — L epoch embeddings
    Output: (B, L, d_model) — contextualized epoch embeddings
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pe = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (B, L, d_model)
        x = self.pe(x)
        x = self.encoder(x)
        return x


class SleepTransformer(nn.Module):
    """SleepTransformer (Phan et al. 2022).

    Sequence-to-sequence sleep staging model based entirely on transformers.
    Processes spectrogram input through an epoch-level transformer and a
    sequence-level transformer, then classifies each epoch with a 2-layer
    FC head.

    Input:  (B, L, C, T, F) — spectrograms (e.g. from "seqsleepnet" preset)
    Output: (B, L, n_classes) — per-epoch logits

    Args:
        n_classes: Number of sleep stages (default 5).
        in_chan: Number of input channels (default 1).
        d_model: Transformer hidden dimension (default 128).
        n_heads: Number of attention heads (default 8).
        n_epoch_layers: Epoch transformer layers (default 4).
        n_seq_layers: Sequence transformer layers (default 4).
        d_ff: Feedforward dimension (default 1024).
        d_clf: Classification head hidden size (default 1024).
        dropout: Dropout rate (default 0.1).
        attention_size: Attention pooling hidden size (default 128).
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        n_epoch_layers: int = 4,
        n_seq_layers: int = 4,
        d_ff: int = 1024,
        d_clf: int = 1024,
        dropout: float = 0.1,
        attention_size: int = 128,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.in_chan = in_chan

        # Epoch transformer: truncates each channel to d_model, concatenates → d_model*in_chan
        hidden_d = d_model * in_chan

        self.epoch_encoder = EpochTransformer(
            d_model=d_model,
            in_chan=in_chan,
            n_heads=n_heads,
            n_layers=n_epoch_layers,
            d_ff=d_ff,
            dropout=dropout,
            attention_size=attention_size,
        )

        self.sequence_encoder = SequenceTransformer(
            d_model=hidden_d,
            n_heads=n_heads,
            n_layers=n_seq_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Paper: two FC layers of 1024 with ReLU, then linear to n_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_d, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, n_classes),
        )

    def encode(self, x):
        """Encode input spectrograms to per-epoch embeddings.

        Args:
            x: (B, L, C, T, F) spectrogram input.

        Returns:
            (B, L, d_model) contextualized epoch embeddings.
        """
        B, L, C, T, F = x.shape

        # Encode each epoch independently
        x = x.reshape(B * L, C, T, F)
        x = self.epoch_encoder(x)  # (B*L, d_model)
        x = x.reshape(B, L, -1)  # (B, L, d_model)

        # Contextualize with sequence transformer
        x = self.sequence_encoder(x)  # (B, L, d_model)

        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, L, C, T, F) spectrogram input.

        Returns:
            (B, L, n_classes) per-epoch logits.
        """
        embeddings = self.encode(x)  # (B, L, d_model)

        # Classify each epoch
        B, L, D = embeddings.shape
        logits = self.classifier(embeddings.reshape(B * L, D))  # (B*L, n_classes)
        logits = logits.reshape(B, L, -1)  # (B, L, n_classes)

        return logits


if __name__ == "__main__":
    from torchinfo import summary

    model = SleepTransformer(n_classes=5, in_chan=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SleepTransformer: {n_params:,} parameters")

    # Input: spectrograms (B=4, L=21, C=1, T=29, F=129)
    x = torch.randn(4, 21, 1, 29, 129)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (4, 21, 5)

    # Verify gradient flow
    y.sum().backward()
    print("Gradient flow: OK")

    summary(model, input_size=(2, 21, 1, 29, 129))
