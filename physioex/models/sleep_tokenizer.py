"""SleepTokenizer: epoch-level prototype tokenizer for polysomnography.

Maps each 30-second multi-channel PSG epoch to an L2-normalized embedding
on the unit hypersphere. Supports any number of input channels (1 to 38+)
of heterogeneous types (EEG, EOG, EMG, ECG, respiratory, SpO2, etc.).

Architecture:
    1. Per-channel U-Net encoder-decoder (shared weights) for multi-scale
       temporal feature extraction from raw time series.
    2. Modality type embedding (14 signal types) for channel identity.
    3. Modality-level dropout for robustness to missing channels.
    4. Set Transformer (SAB + PMA) for cross-channel mixing and
       aggregation to a fixed-size embedding regardless of channel count.
    5. Prototypical classifier (distance-based, EMA centroids) producing
       a metric embedding space where k-means is mathematically justified.
    6. Post-training tokenization via k-means with user-chosen K.

``forward()`` returns a dict with logits, per-channel logits, and embedding.
``encode()`` returns L2-normalized embeddings for downstream use.
``fit_prototypes(K)`` + ``tokenize()`` for discrete tokenization at any K.

References:
    - U-Sleep: Perslev et al. 2021, npj Digital Medicine
    - Set Transformer: Lee et al. 2019, ICML
    - Prototypical Networks: Snell et al. 2017, NeurIPS
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import modality types from shared data module
from physioex.data.modality import (
    ModalityType,
    MODALITY_TYPES as _DATA_MODALITY_TYPES,
    N_MODALITY_TYPES,
    infer_channel_modality,
)


# ═══════════════════════════════════════════════════════════════════════
# Modality type registry — 15 types, validated on 807 channels (98.4%)
# ═══════════════════════════════════════════════════════════════════════

# Keep local copy for backward compatibility (same as shared module)
MODALITY_TYPES = dict(_DATA_MODALITY_TYPES)


def infer_modality(channel_name: str, hint: Optional[str] = None) -> int:
    """Classify a channel name into a modality type index.

    Args:
        channel_name: Physical channel label (e.g. ``"EEG C4-M1"``).
        hint: Optional modality from ``channel_info["modality"]``
              (one of ``"EEG"``/``"EOG"``/``"EMG"``/``"ECG"`` or ``None``).

    Returns:
        Integer index into :data:`MODALITY_TYPES`.

    Note:
        This function now wraps the shared implementation from
        physioex.data.modality for consistency across the codebase.
    """
    result = infer_channel_modality(channel_name, hint)
    return int(result)  # Convert ModalityType enum to int


def build_modality_ids(batch: dict) -> torch.Tensor:
    """Build ``(B, C)`` modality-type index tensor from a collated batch.

    With the new MODALITY_INDEX channel naming (e.g., "EEG_0", "EOG_1"),
    this function simply parses the modality from the channel name.

    Args:
        batch: Dict with ``channel_order`` (list of C channel names).

    Returns:
        Tensor of shape ``(B, C)`` with modality type indices.
    """
    channel_order = batch["channel_order"]
    B = batch["labels"].shape[0]
    C = len(channel_order)
    ids = torch.zeros(B, C, dtype=torch.long)

    # Parse modality from channel name: "EEG_0" -> "EEG" -> 0
    for c, name in enumerate(channel_order):
        modality_str = name.split("_")[0]  # Extract modality part
        modality_idx = MODALITY_TYPES.get(modality_str, MODALITY_TYPES["OTHER"])
        ids[:, c] = modality_idx

    return ids


# ═══════════════════════════════════════════════════════════════════════
# U-Net per-channel encoder-decoder
# ═══════════════════════════════════════════════════════════════════════


class _EncoderBlock(nn.Module):
    """Conv + BN + ReLU → MaxPool(2).  Returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        skip = x
        if x.shape[-1] % 2 == 1:
            x = F.pad(x, (0, 1))
        x = self.pool(x)
        return x, skip


class _DecoderBlock(nn.Module):
    """Upsample → Conv(k=2) + ELU + BN → Cat(skip) → Conv(k) + ELU + BN."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, k: int = 9):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_ch, out_ch, kernel_size=2, padding=0),
            nn.ELU(),
            nn.BatchNorm1d(out_ch),
        )
        self.post_skip = nn.Sequential(
            nn.Conv1d(out_ch + skip_ch, out_ch, k, padding=k // 2),
            nn.ELU(),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        # Crop to match skip
        min_t = min(x.shape[-1], skip.shape[-1])
        x = x[:, :, :min_t]
        skip = skip[:, :, :min_t]
        x = torch.cat([x, skip], dim=1)
        x = self.post_skip(x)
        return x


class ChannelUNet(nn.Module):
    """Per-channel U-Net encoder + partial decoder.

    Processes ``(N, 1, T)`` and produces ``(N, d_out)`` via AvgPool.

    Args:
        depth: Number of encoder levels (default 5).
        filters: Channel counts per level. Length must be ``depth + 2``
                 (input + depth encoder levels + bottom).
        k: Kernel size for all convolutions (default 9).
        decode_levels: How many decoder levels to use (default 3, stops at L2).
    """

    def __init__(
        self,
        depth: int = 5,
        filters: Optional[List[int]] = None,
        k: int = 9,
        decode_levels: int = 3,
    ):
        super().__init__()
        if filters is None:
            filters = [1, 64, 128, 256, 256, 256]
        assert len(filters) == depth + 1, (
            f"filters must have {depth + 1} entries, got {len(filters)}"
        )
        self.depth = depth
        self.filters = filters
        self.decode_levels = decode_levels

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                _EncoderBlock(filters[i], filters[i + 1], k)
            )

        # Bottom
        self.bottom = nn.Sequential(
            nn.Conv1d(filters[depth], filters[depth], k, padding=k // 2),
            nn.BatchNorm1d(filters[depth]),
            nn.ReLU(),
        )

        # Partial decoder
        self.decoders = nn.ModuleList()
        for i in range(decode_levels):
            enc_level = depth - i  # 5, 4, 3
            target_level = enc_level - 1  # 4, 3, 2
            in_ch = filters[enc_level]
            out_ch = filters[target_level]
            skip_ch = filters[enc_level]  # skip from encoder at enc_level
            self.decoders.append(_DecoderBlock(in_ch, out_ch, skip_ch, k))

        # Output dim = filters at the decoder stop level
        self.d_out = filters[depth - decode_levels]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, T)`` → ``(N, d_out)``."""
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottom(x)

        for i, dec in enumerate(self.decoders):
            skip = skips[self.depth - 1 - i]
            x = dec(x, skip)

        x = self.avg_pool(x).squeeze(-1)  # (N, d_out)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Set Transformer components
# ═══════════════════════════════════════════════════════════════════════


class SAB(nn.Module):
    """Self-Attention Block from Set Transformer."""

    def __init__(
        self, d_model: int = 128, n_heads: int = 4, ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention from Set Transformer.

    *k* learned seed vectors attend to all *C* channel tokens, producing
    a fixed ``(N, k, d_model)`` output regardless of *C*.
    """

    def __init__(
        self, d_model: int = 128, n_heads: int = 4, n_seeds: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seeds = self.seeds.expand(x.size(0), -1, -1)
        out = self.norm(seeds + self.attn(seeds, x, x)[0])
        return out  # (N, n_seeds, d_model)


# ═══════════════════════════════════════════════════════════════════════
# Prototypical classifier
# ═══════════════════════════════════════════════════════════════════════


class PrototypicalClassifier(nn.Module):
    """Distance-based classification with EMA class centroids.

    No learnable parameters — centroids are ``register_buffer`` updated
    externally via :meth:`update_centroids` (called from the training
    script, not from ``forward``).
    """

    def __init__(
        self, d_model: int, n_classes: int, tau: float = 0.1,
        ema_alpha: float = 0.99,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = ema_alpha
        # Initialize centroids randomly on the unit hypersphere so that
        # gradients flow from the very first batch (zero centroids →
        # zero gradients through cosine similarity).
        init_centroids = F.normalize(torch.randn(n_classes, d_model), dim=1)
        self.register_buffer("centroids", init_centroids)
        self.register_buffer(
            "initialized", torch.zeros(n_classes, dtype=torch.bool)
        )

    @torch.no_grad()
    def update_centroids(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """EMA update of class centroids.

        Args:
            embeddings: ``(N, d_model)`` L2-normalized.
            labels: ``(N,)`` integer class labels.  Values ``< 0`` are ignored.
        """
        for k in range(self.centroids.size(0)):
            mask = labels == k
            if not mask.any():
                continue
            class_mean = F.normalize(embeddings[mask].mean(dim=0), dim=0)
            if self.initialized[k]:
                self.centroids[k] = (
                    self.alpha * self.centroids[k]
                    + (1 - self.alpha) * class_mean
                )
                self.centroids[k] = F.normalize(self.centroids[k], dim=0)
            else:
                self.centroids[k] = class_mean
                self.initialized[k] = True

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarity / tau → logits.

        Args:
            embeddings: ``(N, d_model)`` L2-normalized.

        Returns:
            ``(N, n_classes)`` logits.
        """
        centroids_norm = F.normalize(self.centroids, dim=1)
        return (embeddings @ centroids_norm.T) / self.tau


# ═══════════════════════════════════════════════════════════════════════
# Modality-level dropout
# ═══════════════════════════════════════════════════════════════════════


def modality_dropout(
    x: torch.Tensor,
    modality_ids: torch.Tensor,
    p_batch: float = 0.3,
    p_modality: float = 0.3,
) -> torch.Tensor:
    """Drop entire modality types for a fraction of the batch.

    70% of the batch sees all channels (no bias). 30% gets modality-level
    dropout to learn robustness. At least 1 channel always survives.

    Args:
        x: ``(N, C, D)`` channel features.
        modality_ids: ``(N, C)`` modality type indices.
        p_batch: Fraction of batch that gets dropout.
        p_modality: Probability of dropping each modality TYPE.

    Returns:
        ``(N, C, D)`` with zeroed channels for dropped modalities.
    """
    N, C, D = x.shape
    device = x.device

    batch_mask = torch.rand(N, device=device) < p_batch
    drop = torch.zeros(N, C, dtype=torch.bool, device=device)

    for mod in modality_ids.unique():
        mod_drop = (torch.rand(N, device=device) < p_modality) & batch_mask
        mod_channels = modality_ids == mod
        drop |= mod_drop.unsqueeze(1) & mod_channels

    # Safety: at least 1 channel per sample
    all_dropped = drop.all(dim=1)
    if all_dropped.any():
        for idx in all_dropped.nonzero(as_tuple=True)[0]:
            keep = torch.randint(C, (1,), device=device)
            drop[idx, keep] = False

    x = x.clone()
    x[drop] = 0
    return x


# ═══════════════════════════════════════════════════════════════════════
# SleepTokenizer — main model
# ═══════════════════════════════════════════════════════════════════════


class SleepTokenizer(nn.Module):
    """Epoch-level prototype tokenizer for polysomnography.

    Args:
        n_classes: Number of sleep stages (default 5).
        d_model: Embedding dimension (default 128).
        T: Samples per epoch (default 3000 = 30 s @ 100 Hz).
        sf: Sampling frequency in Hz (default 100).
        unet_depth: U-Net encoder depth (default 5).
        unet_filters: Filter counts per encoder level.
        unet_k: Convolution kernel size (default 9).
        unet_decode_levels: Partial decoder depth (default 3).
        n_sab_layers: Number of SAB layers in channel mixer (default 2).
        n_heads: Attention heads (default 4).
        ff_dim: Feed-forward dimension in SAB (default 256).
        n_seeds: PMA seed vectors (default 4).
        n_modality_types: Modality embedding vocabulary (default 14).
        tau: Prototypical classifier temperature (default 0.1).
        ema_alpha: EMA decay for centroid update (default 0.99).
        dropout: Dropout rate (default 0.1).
        p_batch_dropout: Fraction of batch for modality dropout (default 0.3).
        p_modality_dropout: Per-modality drop probability (default 0.3).
    """

    def __init__(
        self,
        n_classes: int = 5,
        d_model: int = 128,
        T: int = 3000,
        sf: int = 100,
        unet_depth: int = 5,
        unet_filters: Optional[List[int]] = None,
        unet_k: int = 9,
        unet_decode_levels: int = 3,
        n_sab_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_seeds: int = 4,
        n_modality_types: int = N_MODALITY_TYPES,
        tau: float = 0.1,
        ema_alpha: float = 0.99,
        dropout: float = 0.1,
        p_batch_dropout: float = 0.3,
        p_modality_dropout: float = 0.3,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.p_batch_dropout = p_batch_dropout
        self.p_modality_dropout = p_modality_dropout

        # 1. Per-channel U-Net
        if unet_filters is None:
            unet_filters = [1, 64, 128, 256, 256, 256]
        self.channel_unet = ChannelUNet(
            depth=unet_depth,
            filters=unet_filters,
            k=unet_k,
            decode_levels=unet_decode_levels,
        )
        unet_out_dim = self.channel_unet.d_out

        # Projection if U-Net output dim != d_model
        if unet_out_dim != d_model:
            self.proj = nn.Sequential(
                nn.LayerNorm(unet_out_dim),
                nn.Linear(unet_out_dim, d_model),
                nn.GELU(),
            )
        else:
            self.proj = nn.Identity()

        # 2. Modality type embedding
        self.modality_embed = nn.Embedding(n_modality_types, d_model)

        # 3. Channel mixer: SAB layers
        self.sab_layers = nn.ModuleList(
            [SAB(d_model, n_heads, ff_dim, dropout) for _ in range(n_sab_layers)]
        )

        # 4. PMA pooling
        self.pma = PMA(d_model, n_heads, n_seeds, dropout)
        self.pma_proj = nn.Sequential(
            nn.Linear(n_seeds * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # 5. Per-channel auxiliary classifier (Linear — for gradient to U-Net)
        self.chan_clf = nn.Linear(d_model, n_classes)

        # 6. Prototypical classifier (distance-based, no learnable params)
        self.clf = PrototypicalClassifier(d_model, n_classes, tau, ema_alpha)

        # Post-hoc prototypes (set by fit_prototypes)
        self._proto_centroids: Optional[torch.Tensor] = None

    def _encode_channels(
        self, x: torch.Tensor, modality_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode per-channel and aggregate to epoch embedding.

        Args:
            x: ``(N, C, T)`` raw multi-channel epoch.
            modality_ids: ``(N, C)`` modality type indices.

        Returns:
            ``(N, d_model)`` L2-normalized embedding.
        """
        N, C, T = x.shape

        # Per-channel U-Net (shared weights, batched)
        x_flat = x.reshape(N * C, 1, T)  # (N*C, 1, T)
        ch_emb = self.channel_unet(x_flat)  # (N*C, unet_out_dim)
        ch_emb = self.proj(ch_emb)  # (N*C, d_model)
        ch_emb = ch_emb.reshape(N, C, self.d_model)  # (N, C, d_model)

        # Modality type embedding
        ch_emb = ch_emb + self.modality_embed(modality_ids)

        # Modality-level dropout (training only)
        if self.training:
            mix_emb = modality_dropout(
                ch_emb, modality_ids,
                self.p_batch_dropout, self.p_modality_dropout,
            )
        else:    
            mix_emb = ch_emb  # (N, C, d_model)

        # Cross-channel mixing (SAB)
        for sab in self.sab_layers:
            mix_emb = sab(mix_emb)

        # PMA aggregation → fixed-size embedding
        pooled = self.pma(mix_emb)  # (N, n_seeds, d_model)
        embedding = self.pma_proj(pooled.flatten(1))  # (N, d_model)

        # L2 normalize onto unit hypersphere
        embedding = F.normalize(embedding, dim=-1)

        return embedding, ch_emb

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x: ``(B, L, C, T)`` or ``(B, C, T)`` raw time series.
            modality_ids: ``(B, C)`` modality type per channel.

        Returns:
            Dict with keys ``logits``, ``channel_logits``, ``embedding``,
            ``data_mask``.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        B, L, C, T = x.shape

        # Detect zero-filled (absent) channels — before any processing
        data_mask = (x.abs().sum(dim=(1, 3)) == 0)  # (B, C) True = absent

        # Flatten sequence dimension
        x_flat = x.reshape(B * L, C, T)
        mod_flat = modality_ids.unsqueeze(1).expand(B, L, C).reshape(B * L, C)

        # Encode
        embedding, ch_emb = self._encode_channels(x_flat, mod_flat)

        # Prototypical classification
        logits = self.clf(embedding)  # (B*L, n_classes)

        # Per-channel auxiliary classification
        channel_logits = self.chan_clf(ch_emb)  # (B*L, C, n_classes)

        return {
            "logits": logits.reshape(B, L, -1),
            "channel_logits": channel_logits.reshape(B, L, C, -1),
            "embedding": embedding.reshape(B, L, -1),
            "data_mask": data_mask,
        }

    @torch.no_grad()
    def encode(
        self, x: torch.Tensor, modality_ids: torch.Tensor
    ) -> torch.Tensor:
        """L2-normalized continuous embeddings.

        Args:
            x: ``(B, L, C, T)`` or ``(B, C, T)``.
            modality_ids: ``(B, C)``.

        Returns:
            ``(B, L, d_model)`` on the unit hypersphere.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        B, L, C, T = x.shape
        x_flat = x.reshape(B * L, C, T)
        mod_flat = modality_ids.unsqueeze(1).expand(B, L, C).reshape(B * L, C)
        embedding, _ = self._encode_channels(x_flat, mod_flat)
        return embedding.reshape(B, L, -1)

    def fit_prototypes(
        self,
        embeddings: torch.Tensor,
        K: int,
        max_iter: int = 100,
        n_init: int = 5,
    ) -> torch.Tensor:
        """Fit K prototypes via k-means on pre-computed embeddings.

        Args:
            embeddings: ``(N, d_model)`` L2-normalized embeddings
                        (e.g. from running ``encode`` over the dataset).
            K: Number of prototypes.
            max_iter: k-means iterations.
            n_init: k-means restarts.

        Returns:
            ``(K, d_model)`` prototype centroids (also stored internally).
        """
        from sklearn.cluster import KMeans

        emb_np = embeddings.detach().cpu().numpy()
        km = KMeans(n_clusters=K, max_iter=max_iter, n_init=n_init)
        km.fit(emb_np)
        centroids = torch.from_numpy(km.cluster_centers_).float()
        centroids = F.normalize(centroids, dim=1)
        self._proto_centroids = centroids
        return centroids

    @torch.no_grad()
    def tokenize(
        self, x: torch.Tensor, modality_ids: torch.Tensor
    ) -> torch.Tensor:
        """Map epochs to nearest prototype index.

        Requires :meth:`fit_prototypes` to have been called first.

        Args:
            x: ``(B, L, C, T)`` or ``(B, C, T)``.
            modality_ids: ``(B, C)``.

        Returns:
            ``(B, L)`` long tensor with indices in ``[0, K)``.
        """
        if self._proto_centroids is None:
            raise RuntimeError(
                "Call fit_prototypes() before tokenize()"
            )
        emb = self.encode(x, modality_ids)  # (B, L, d_model)
        B, L, D = emb.shape
        centroids = self._proto_centroids.to(emb.device)  # (K, D)
        # Cosine similarity → nearest
        sim = emb.reshape(-1, D) @ centroids.T  # (B*L, K)
        indices = sim.argmax(dim=-1)  # (B*L,)
        return indices.reshape(B, L)

    def get_prototypes(self) -> Optional[torch.Tensor]:
        """Return current prototype centroids or ``None``."""
        return self._proto_centroids


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    model = SleepTokenizer(n_classes=5, d_model=128)
    total = sum(p.numel() for p in model.parameters())
    print(f"SleepTokenizer: {total:,} params")

    for C in [1, 3, 8, 20, 38]:
        x = torch.randn(2, 1, C, 3000)
        mod = torch.randint(0, N_MODALITY_TYPES, (2, C))
        out = model(x, mod)
        print(
            f"  C={C:>2d}: logits={out['logits'].shape}, "
            f"emb={out['embedding'].shape}, "
            f"chan_logits={out['channel_logits'].shape}"
        )

    # Encode + tokenize test
    x = torch.randn(4, 1, 3, 3000)
    mod = torch.zeros(4, 3, dtype=torch.long)
    model.eval()
    emb = model.encode(x, mod)
    print(f"\n  encode: {emb.shape}, norm={emb.norm(dim=-1).mean():.4f}")

    # Fit prototypes
    all_emb = model.encode(torch.randn(100, 1, 3, 3000), mod[:1].expand(100, -1))
    centroids = model.fit_prototypes(all_emb.squeeze(1), K=10)
    tokens = model.tokenize(x, mod)
    print(f"  tokenize(K=10): {tokens.shape}, range=[{tokens.min()}, {tokens.max()}]")
    print("\nAll tests passed.")
