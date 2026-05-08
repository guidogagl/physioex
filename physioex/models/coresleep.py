"""CoRe-Sleep: Coordinated Representation multimodal fusion for sleep staging.

Faithful reimplementation of Kontras et al. 2024 (IEEE TNSRE) for PhysioEx,
based on the original repository: https://github.com/kkontras/CoRe-Sleep

Architecture:
    - Per-modality inner (intra-epoch) transformer encoders with [CLS] aggregation
    - Cross-attention at both inner and outer levels
    - Per-modality outer (inter-epoch) transformer encoders
    - Three classification heads: combined, EEG-only, EOG-only
    - CLIP-style alignment projections for contrastive loss

``forward()`` returns a dict with keys: ``"combined"``, ``"eeg"``, ``"eog"``,
``"align_eeg"``, ``"align_eog"`` (bimodal) or just ``"combined"`` (unimodal).
``encode()`` returns ``(B, L, D)`` for embedding extraction (always from the
fused representation).

Reference:
    Kontras, K., Chatzichristos, C., Phan, H., Suykens, J., & De Vos, M.
    "CoRe-Sleep: A Multimodal Fusion Framework for Time Series Robust to
    Imperfect Modalities", IEEE TNSRE, 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class _LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding added to token sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pos_embed[:, : x.size(1)])


class _InnerEncoder(nn.Module):
    """Intra-epoch transformer encoder with [CLS] aggregation.

    Processes the STFT time-frequency sequence within a single 30-second
    epoch.  A learnable [CLS] token aggregates the sequence.

    Supports optional cross-attention: when ``cross_kv`` is provided,
    a cross-attention layer is applied after self-attention, allowing
    the CLS token of one modality to attend to features of the other.

    Input:  (B, T, D)
    Output: (B, D) — the [CLS] token embedding
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
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-attention (CLS of this modality attends to other modality)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

    def forward(self, x, cross_kv=None):
        """
        Args:
            x: (B, T, D) input tokens for this modality.
            cross_kv: (B, T, D) tokens from the other modality (optional).

        Returns:
            (B, D) — CLS embedding after self-attention and optional cross-attention.
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, D)
        x = self.pos_enc(x)
        x = self.encoder(x)

        if cross_kv is not None:
            # Cross-attention: CLS queries the other modality's features
            cls_token = x[:, :1]  # (B, 1, D)
            cls_cross, _ = self.cross_attn(
                query=self.cross_norm(cls_token),
                key=cross_kv,
                value=cross_kv,
            )
            x = x.clone()
            x[:, 0] = x[:, 0] + cls_cross.squeeze(1)  # residual

        return x[:, 0]  # (B, D)


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
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-attention at outer level
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

    def forward(self, x, cross_kv=None):
        """
        Args:
            x: (B, L, D) epoch embeddings for this modality.
            cross_kv: (B, L, D) epoch embeddings from the other modality (optional).

        Returns:
            (B, L, D) contextualized epoch embeddings.
        """
        x = self.pos_enc(x)
        x = self.encoder(x)

        if cross_kv is not None:
            cross_out, _ = self.cross_attn(
                query=self.cross_norm(x),
                key=cross_kv,
                value=cross_kv,
            )
            x = x + cross_out  # residual

        return x


def _make_head(d_model: int, d_ff: int, n_classes: int, dropout: float) -> nn.Sequential:
    """3-layer classification MLP matching the original implementation."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, n_classes),
    )


class CoReSleep(nn.Module):
    """CoRe-Sleep: multimodal sleep staging with cross-modal attention.

    Accepts spectrogram input ``(B, L, C, T, F)`` where C=2 (EEG + EOG)
    or C=1 (EEG only).

    ``forward()`` returns a **dict** with classification logits and alignment
    features (for the multi-task loss used during training):
        - ``"combined"``: ``(B, L, n_classes)`` — fused prediction
        - ``"eeg"``: ``(B, L, n_classes)`` — EEG-only prediction (bimodal only)
        - ``"eog"``: ``(B, L, n_classes)`` — EOG-only prediction (bimodal only)
        - ``"align_eeg"``: ``(B, L, D)`` — L2-normalized EEG features (bimodal only)
        - ``"align_eog"``: ``(B, L, D)`` — L2-normalized EOG features (bimodal only)

    ``encode()`` returns ``(B, L, d_model)`` for embedding extraction.

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

        # Per-modality input projections
        self.proj_eeg = nn.Linear(F, d_model)
        if in_chan >= 2:
            self.proj_eog = nn.Linear(F, d_model)

        # Per-modality inner encoders (separate weights)
        inner_kwargs = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_inner_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.inner_eeg = _InnerEncoder(**inner_kwargs)
        if in_chan >= 2:
            self.inner_eog = _InnerEncoder(**inner_kwargs)

        # Per-modality outer encoders (separate weights)
        outer_kwargs = dict(
            d_model=d_model, n_heads=n_heads, n_layers=n_outer_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.outer_eeg = _OuterEncoder(**outer_kwargs)
        if in_chan >= 2:
            self.outer_eog = _OuterEncoder(**outer_kwargs)

        # Classification heads
        self.clf_combined = _make_head(d_model, d_ff, n_classes, dropout)
        if in_chan >= 2:
            self.clf_eeg = _make_head(d_model, d_ff, n_classes, dropout)
            self.clf_eog = _make_head(d_model, d_ff, n_classes, dropout)

            # CLIP alignment projections
            self.align_proj_eeg = nn.Linear(d_model, d_model)
            self.align_proj_eog = nn.Linear(d_model, d_model)

    def _encode_inner(self, x_eeg, x_eog=None):
        """Run inner encoders with cross-attention between modalities.

        Args:
            x_eeg: (B*L, T, F) EEG spectrogram.
            x_eog: (B*L, T, F) EOG spectrogram or None.

        Returns:
            eeg_emb: (B*L, D) EEG epoch embeddings.
            eog_emb: (B*L, D) EOG epoch embeddings or None.
        """
        eeg_proj = self.proj_eeg(x_eeg)  # (B*L, T, D)

        if x_eog is not None:
            eog_proj = self.proj_eog(x_eog)  # (B*L, T, D)
            # Cross-attention at inner level: each CLS attends to other modality
            eeg_emb = self.inner_eeg(eeg_proj, cross_kv=eog_proj)
            eog_emb = self.inner_eog(eog_proj, cross_kv=eeg_proj)
            return eeg_emb, eog_emb
        else:
            eeg_emb = self.inner_eeg(eeg_proj)
            return eeg_emb, None

    def _encode_outer(self, eeg_seq, eog_seq=None):
        """Run outer encoders with cross-attention between modalities.

        Args:
            eeg_seq: (B, L, D) EEG epoch sequence.
            eog_seq: (B, L, D) EOG epoch sequence or None.

        Returns:
            eeg_ctx: (B, L, D) contextualized EEG.
            eog_ctx: (B, L, D) contextualized EOG or None.
            fused: (B, L, D) additive fusion of both.
        """
        if eog_seq is not None:
            eeg_ctx = self.outer_eeg(eeg_seq, cross_kv=eog_seq)
            eog_ctx = self.outer_eog(eog_seq, cross_kv=eeg_seq)
            fused = eeg_ctx + eog_ctx  # additive fusion
            return eeg_ctx, eog_ctx, fused
        else:
            eeg_ctx = self.outer_eeg(eeg_seq)
            return eeg_ctx, None, eeg_ctx

    def encode(self, x):
        """Encode input spectrograms to contextualized per-epoch embeddings.

        Returns the fused (combined) representation for embedding extraction.

        Args:
            x: (B, L, C, T, F) spectrogram input. C=1 or C=2.

        Returns:
            (B, L, d_model) contextualized epoch embeddings.
        """
        B, L, C, T, F = x.size()

        eeg = x[:, :, 0].reshape(B * L, T, F)
        eog = x[:, :, 1].reshape(B * L, T, F) if C >= 2 else None

        eeg_emb, eog_emb = self._encode_inner(eeg, eog)

        eeg_seq = eeg_emb.reshape(B, L, -1)
        eog_seq = eog_emb.reshape(B, L, -1) if eog_emb is not None else None

        _, _, fused = self._encode_outer(eeg_seq, eog_seq)
        return fused  # (B, L, D)

    def forward(self, x):
        """Forward pass for sleep stage classification.

        Args:
            x: (B, L, C, T, F) spectrogram input.

        Returns:
            dict with keys:
                - "combined": (B, L, n_classes) fused logits
                - "eeg": (B, L, n_classes) EEG-only logits (bimodal only)
                - "eog": (B, L, n_classes) EOG-only logits (bimodal only)
                - "align_eeg": (B, L, D) normalized EEG features (bimodal only)
                - "align_eog": (B, L, D) normalized EOG features (bimodal only)
        """
        B, L, C, T, F = x.size()

        eeg = x[:, :, 0].reshape(B * L, T, F)
        eog = x[:, :, 1].reshape(B * L, T, F) if C >= 2 else None

        eeg_emb, eog_emb = self._encode_inner(eeg, eog)

        eeg_seq = eeg_emb.reshape(B, L, -1)
        eog_seq = eog_emb.reshape(B, L, -1) if eog_emb is not None else None

        eeg_ctx, eog_ctx, fused = self._encode_outer(eeg_seq, eog_seq)

        # Classification: combined head always, per-modality heads for bimodal
        result = {
            "combined": self.clf_combined(fused.reshape(B * L, -1)).reshape(B, L, -1),
        }

        if self.in_chan >= 2 and eog_ctx is not None:
            result["eeg"] = self.clf_eeg(eeg_ctx.reshape(B * L, -1)).reshape(B, L, -1)
            result["eog"] = self.clf_eog(eog_ctx.reshape(B * L, -1)).reshape(B, L, -1)
            # CLIP alignment features (L2-normalized)
            result["align_eeg"] = Fn.normalize(self.align_proj_eeg(eeg_ctx), dim=-1)
            result["align_eog"] = Fn.normalize(self.align_proj_eog(eog_ctx), dim=-1)

        return result


if __name__ == "__main__":
    from torchinfo import summary

    # Bimodal test
    model = CoReSleep(n_classes=5, in_chan=2)
    print("CoReSleep (bimodal EEG+EOG):")
    x = torch.randn(2, 21, 2, 29, 129)
    out = model(x)
    print(f"  Output keys: {list(out.keys())}")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    print(f"  Encode: {model.encode(x).shape}")

    # Unimodal test
    model_uni = CoReSleep(n_classes=5, in_chan=1)
    print("\nCoReSleep (unimodal EEG):")
    x_uni = torch.randn(2, 21, 1, 29, 129)
    out_uni = model_uni(x_uni)
    print(f"  Output keys: {list(out_uni.keys())}")
    print(f"  combined: {out_uni['combined'].shape}")
    print(f"  Encode: {model_uni.encode(x_uni).shape}")
