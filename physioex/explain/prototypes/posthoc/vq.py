"""Post-hoc codebook learning via Vector Quantization (K-Means).

Inspired by Rymarczyk et al., "ProtoQuant", 2025 (arXiv:2602.06592) and
Ge et al., "Vector Quantized Latent Concepts", 2025 (arXiv:2602.02726).

Given pre-extracted epoch embeddings Z from a frozen encoder h(x), learns
a discrete codebook C = {c_1, ..., c_M} via K-Means clustering.  At
inference time, each epoch embedding is quantized (replaced by its nearest
codebook entry) before being passed to the frozen sequence encoder and
classifier.

The codebook is class-agnostic (unsupervised): prototypes emerge from
the natural clustering structure of the embedding space.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def learn_codebook_kmeans(
    Z: np.ndarray,
    n_prototypes: int = 50,
    random_state: int = 42,
    batch_size: int = 4096,
    max_iter: int = 300,
) -> np.ndarray:
    """Learn a codebook via K-Means clustering on epoch embeddings.

    Args:
        Z: (N, d_model) epoch embeddings (float32).
        n_prototypes: Number of codebook entries (clusters).
        random_state: Seed for K-Means.
        batch_size: Mini-batch size for MiniBatchKMeans.
        max_iter: Maximum iterations.

    Returns:
        codebook: (n_prototypes, d_model) cluster centroids.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=3,
    )
    kmeans.fit(Z)
    return kmeans.cluster_centers_.astype(np.float32)


def quantize_embeddings(
    Z: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each embedding to its nearest codebook entry.

    Args:
        Z: (N, d_model) epoch embeddings.
        codebook: (M, d_model) codebook entries.

    Returns:
        Tuple of:
            Z_quantized: (N, d_model) quantized embeddings
            assignments: (N,) index of assigned codebook entry per embedding
    """
    # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z.c
    Z_sq = (Z ** 2).sum(axis=1, keepdims=True)      # (N, 1)
    C_sq = (codebook ** 2).sum(axis=1, keepdims=True).T  # (1, M)
    dist = Z_sq + C_sq - 2 * (Z @ codebook.T)       # (N, M)

    assignments = dist.argmin(axis=1)                # (N,)
    Z_quantized = codebook[assignments]              # (N, d_model)

    return Z_quantized, assignments
