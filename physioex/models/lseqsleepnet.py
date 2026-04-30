"""L-SeqSleepNet (Phan et al. 2023) for whole-cycle sleep staging.

Reference: "L-SeqSleepNet: Whole-cycle Long Sequence Modelling for
Automatic Sleep Staging", IEEE TNSRE 2023 (arXiv:2301.03441).

Architecture:
  1. Epoch Encoder (same as SeqSleepNet):
     - Learnable filterbank (F=129 → M=32)
     - BiLSTM (He/2=64, total 128)
     - Gated attention (A=64) → 128-dim epoch embedding

  2. Long Sequence Model (fold-process-unfold):
     - Fold: L=200 → B=10 subsequences of K=20
     - Intra-subsequence BiLSTM (Hss/2=64) + FC + LayerNorm + Residual
     - Inter-subsequence BiLSTM (Hms/2=64) + FC + LayerNorm + Residual
     - Unfold: reconstruct L=200 sequence

  3. Classification Head:
     - FC(128 → 512) + ReLU + Dropout
     - FC(512 → 512) + ReLU + Dropout
     - FC(512 → 5)

PhysioEx integration:
  Input:  (B, L, C, T, F) — spectrograms from "seqsleepnet" pipeline
  Output: (B, L, n_classes) — per-epoch logits (sequence-to-sequence)
  Default L=200 (~100 min, one full sleep cycle)
"""

import torch
import torch.nn as nn

from physioex.models.seqsleepnet import AttentionLayer, LearnableFilterbank


class EpochEncoder(nn.Module):
    """Epoch-level encoder identical to SeqSleepNet.

    Input:  (B, C, T, F) spectrogram per epoch
    Output: (B, He) epoch embedding
    """

    def __init__(
        self,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        hidden_size: int = 64,
        attention_size: int = 64,
    ):
        super().__init__()
        self.filterbank = LearnableFilterbank(
            in_chan, F, D, nfft, sf, lowfreq, highfreq
        )

        self.blstm = nn.LSTM(
            D * in_chan,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AttentionLayer(2 * hidden_size, attention_size)

    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.shape
        x = self.filterbank(x)  # (B, C, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, C*D)
        x, _ = self.blstm(x)  # (B, T, 2*hidden)
        x = self.attention(x)  # (B, 2*hidden)
        return x


class FoldProcessUnfold(nn.Module):
    """Hierarchical fold-process-unfold for long sequence modelling.

    Folds a sequence of L embeddings into B subsequences of K,
    applies intra-subsequence and inter-subsequence BiLSTMs with
    residual connections and layer normalization, then unfolds.

    Input:  (batch, L, D)
    Output: (batch, L, D)
    """

    def __init__(
        self,
        d_model: int = 128,
        B: int = 10,
        K: int = 20,
        hidden_ss: int = 64,
        hidden_ms: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.B = B
        self.K = K
        self.d_model = d_model

        # Intra-subsequence BiLSTM
        self.blstm_ss = nn.LSTM(
            d_model,
            hidden_ss,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_ss = nn.Linear(2 * hidden_ss, d_model)
        self.ln_ss = nn.LayerNorm(d_model)
        self.drop_ss = nn.Dropout(dropout)

        # Inter-subsequence BiLSTM
        self.blstm_ms = nn.LSTM(
            d_model,
            hidden_ms,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_ms = nn.Linear(2 * hidden_ms, d_model)
        self.ln_ms = nn.LayerNorm(d_model)
        self.drop_ms = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, L, D)
        batch, L, D = x.shape

        # Pad if L is not exactly B*K
        target_len = self.B * self.K
        if L < target_len:
            pad = torch.zeros(batch, target_len - L, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif L > target_len:
            x = x[:, :target_len]

        # ① Fold: (batch, B*K, D) → (batch, B, K, D)
        x = x.reshape(batch, self.B, self.K, D)

        # ② Intra-subsequence BiLSTM: process each of B subsequences
        # Reshape to (batch*B, K, D)
        x_intra = x.reshape(batch * self.B, self.K, D)
        residual = x_intra
        out_ss, _ = self.blstm_ss(x_intra)  # (batch*B, K, 2*hidden_ss)
        out_ss = self.fc_ss(out_ss)  # (batch*B, K, D)
        out_ss = self.ln_ss(out_ss + residual)  # residual + layernorm
        out_ss = self.drop_ss(out_ss)

        # Reshape back: (batch, B, K, D)
        out_ss = out_ss.reshape(batch, self.B, self.K, D)

        # ③ Inter-subsequence BiLSTM: for each position k, process across B
        # Permute to (batch, K, B, D)
        x_inter = out_ss.permute(0, 2, 1, 3)
        # Reshape to (batch*K, B, D)
        x_inter = x_inter.reshape(batch * self.K, self.B, D)
        residual = x_inter
        out_ms, _ = self.blstm_ms(x_inter)  # (batch*K, B, 2*hidden_ms)
        out_ms = self.fc_ms(out_ms)  # (batch*K, B, D)
        out_ms = self.ln_ms(out_ms + residual)  # residual + layernorm
        out_ms = self.drop_ms(out_ms)

        # Reshape: (batch, K, B, D) → (batch, B, K, D)
        out_ms = out_ms.reshape(batch, self.K, self.B, D).permute(0, 2, 1, 3)

        # ④ Unfold: (batch, B, K, D) → (batch, B*K, D)
        output = out_ms.reshape(batch, self.B * self.K, D)

        # Trim back to original length
        return output[:, :L]


class LSeqSleepNet(nn.Module):
    """L-SeqSleepNet (Phan et al. 2023).

    Whole-cycle long sequence model for automatic sleep staging. Processes
    L=200 epochs (~100 min) using a hierarchical fold-process-unfold
    strategy with BiLSTMs.

    Input:  (batch, L, C, T, F) — spectrograms from "seqsleepnet" preset
    Output: (batch, L, n_classes) — per-epoch logits

    Args:
        n_classes: Number of sleep stages (default 5).
        in_chan: Number of input channels (default 1).
        F: Frequency bins in spectrogram (default 129).
        D: Learnable filterbank output dim (default 32).
        nfft: FFT size for filterbank init (default 256).
        sf: Sampling frequency (default 100).
        lowfreq: Low frequency cutoff (default 0).
        highfreq: High frequency cutoff (default 50).
        epoch_hidden: Epoch BiLSTM hidden size per direction (default 64).
        epoch_attention: Epoch attention size (default 64).
        B: Number of subsequences in fold (default 10).
        K: Length of each subsequence (default 20).
        seq_hidden_ss: Intra-subsequence BiLSTM hidden per direction (default 64).
        seq_hidden_ms: Inter-subsequence BiLSTM hidden per direction (default 64).
        d_clf: Classification head hidden size (default 512).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_chan: int = 1,
        F: int = 129,
        D: int = 32,
        nfft: int = 256,
        sf: int = 100,
        lowfreq: int = 0,
        highfreq: int = 50,
        epoch_hidden: int = 64,
        epoch_attention: int = 64,
        B: int = 10,
        K: int = 20,
        seq_hidden_ss: int = 64,
        seq_hidden_ms: int = 64,
        d_clf: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes

        d_model = 2 * epoch_hidden  # 128

        self.epoch_encoder = EpochEncoder(
            in_chan=in_chan,
            F=F,
            D=D,
            nfft=nfft,
            sf=sf,
            lowfreq=lowfreq,
            highfreq=highfreq,
            hidden_size=epoch_hidden,
            attention_size=epoch_attention,
        )

        self.sequence_model = FoldProcessUnfold(
            d_model=d_model,
            B=B,
            K=K,
            hidden_ss=seq_hidden_ss,
            hidden_ms=seq_hidden_ms,
            dropout=dropout,
        )

        # Paper: two FC layers of 512 with ReLU, then linear to n_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, d_clf),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_clf, n_classes),
        )

    def encode(self, x):
        """Encode spectrograms to contextualized epoch embeddings.

        Args:
            x: (batch, L, C, T, F)

        Returns:
            (batch, L, d_model) epoch embeddings after sequence modelling
        """
        batch, L, C, T, F = x.shape

        # Encode each epoch independently
        x = x.reshape(batch * L, C, T, F)
        x = self.epoch_encoder(x)  # (batch*L, d_model)
        x = x.reshape(batch, L, -1)  # (batch, L, d_model)

        # Long sequence modelling
        x = self.sequence_model(x)  # (batch, L, d_model)

        return x

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, L, C, T, F)

        Returns:
            (batch, L, n_classes) per-epoch logits
        """
        embeddings = self.encode(x)  # (batch, L, d_model)

        batch, L, D = embeddings.shape
        logits = self.classifier(embeddings.reshape(batch * L, D))
        return logits.reshape(batch, L, -1)


if __name__ == "__main__":
    from torchinfo import summary

    # Default config: L=200, B=10, K=20
    model = LSeqSleepNet(n_classes=5, in_chan=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"L-SeqSleepNet: {n_params:,} parameters")

    # Test with L=200 (full sleep cycle)
    x = torch.randn(2, 200, 1, 29, 129)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (2, 200, 5)

    # Test with shorter sequence (L=50, padded internally)
    x_short = torch.randn(2, 50, 1, 29, 129)
    y_short = model(x_short)
    print(f"Short input: {x_short.shape} -> Output: {y_short.shape}")
    assert y_short.shape == (2, 50, 5)

    y.sum().backward()
    print("Gradient flow: OK")
