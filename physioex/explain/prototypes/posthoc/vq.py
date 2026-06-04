"""Post-hoc codebook learning via Vector Quantization.

Inspired by Rymarczyk et al., "ProtoQuant", 2025 (arXiv:2602.06592) and
Ge et al., "Vector Quantized Latent Concepts", 2025 (arXiv:2602.02726).

Given pre-extracted epoch embeddings Z from a frozen encoder h(x), learns
a discrete codebook via two stages:

  Stage 1 (init): K-Means clustering on embeddings → initial codebook
  Stage 2 (train): Supervised refinement with classification loss +
      straight-through estimator. The codebook is optimized so that
      quantized embeddings fed to the frozen downstream model (sequence
      encoder + classifier) preserve classification accuracy.

At inference, each epoch embedding is quantized (replaced by its nearest
codebook entry) before being passed to the frozen sequence encoder and
classifier.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import torch
import torch.nn as nn


def learn_codebook_kmeans(
    Z: np.ndarray,
    n_prototypes: int = 50,
    random_state: int = 42,
    batch_size: int = 4096,
    max_iter: int = 300,
) -> np.ndarray:
    """Learn initial codebook via K-Means clustering.

    This is Stage 1 (initialization). Use :func:`train_codebook` for
    supervised refinement (Stage 2).

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
    """Quantize each embedding to its nearest codebook entry (numpy).

    Args:
        Z: (N, d_model) epoch embeddings.
        codebook: (M, d_model) codebook entries.

    Returns:
        Tuple of:
            Z_quantized: (N, d_model) quantized embeddings
            assignments: (N,) index of assigned codebook entry
    """
    Z_sq = (Z ** 2).sum(axis=1, keepdims=True)
    C_sq = (codebook ** 2).sum(axis=1, keepdims=True).T
    dist = Z_sq + C_sq - 2 * (Z @ codebook.T)
    assignments = dist.argmin(axis=1)
    Z_quantized = codebook[assignments]
    return Z_quantized, assignments


class VQBottleneck(nn.Module):
    """Differentiable vector quantization with straight-through estimator.

    Forward: z → z_q = codebook[argmin_k ||z - c_k||²]
    Backward: gradients pass through as if z_q = z (STE),
              but codebook receives gradients from ||sg[z] - c_q||²

    Args:
        codebook_init: (M, d_model) initial codebook (e.g. from K-Means).
        commitment_weight: Weight β for commitment loss ||z - sg[c_q]||².
    """

    def __init__(self, codebook_init: np.ndarray, commitment_weight: float = 0.25):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.codebook = nn.Parameter(
            torch.from_numpy(codebook_init.copy()).float()
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize with STE.

        Args:
            z: (N, d_model) embeddings.

        Returns:
            z_q: (N, d_model) quantized embeddings (STE gradients to z)
            vq_loss: scalar, codebook_loss + β * commitment_loss
        """
        # Distances (L2)
        dist = (
            (z ** 2).sum(dim=1, keepdim=True)
            + (self.codebook ** 2).sum(dim=1, keepdim=True).T
            - 2 * z @ self.codebook.T
        )
        idx = dist.argmin(dim=1)
        z_q = self.codebook[idx]

        # Losses
        codebook_loss = ((z.detach() - z_q) ** 2).mean()
        commitment_loss = ((z - z_q.detach()) ** 2).mean()
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss


def train_codebook(
    Z_train: np.ndarray,
    Y_train: np.ndarray,
    downstream_fn: Callable[[torch.Tensor], torch.Tensor],
    codebook_init: np.ndarray,
    n_epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    commitment_weight: float = 0.25,
    device: str = "cpu",
    sequence_length: int = 21,
) -> np.ndarray:
    """Supervised codebook refinement (Stage 2).

    Optimizes the codebook so that quantized embeddings, when passed
    through the frozen downstream model, preserve classification accuracy.

    The training loop:
      1. Sample batch of (z, y) from training embeddings
      2. Quantize: z_q = VQBottleneck(z) with STE
      3. Reshape to sequences of length L, forward through downstream_fn
      4. Loss = CE(logits, y) + β * vq_loss
      5. Update only the codebook parameters

    Args:
        Z_train: (N, d_model) training epoch embeddings.
        Y_train: (N,) training labels.
        downstream_fn: Callable that takes (B, L, d_model) quantized
            embeddings and returns (B, L, n_classes) logits.
            Typically: sequence_encoder → classifier (both frozen).
        codebook_init: (M, d_model) initial codebook from K-Means.
        n_epochs: Training epochs.
        batch_size: Must be divisible by sequence_length.
        lr: Learning rate for codebook.
        commitment_weight: β for commitment loss.
        device: Torch device string.
        sequence_length: L, sequence length to reshape embeddings into.

    Returns:
        codebook: (M, d_model) optimized codebook as numpy array.
    """
    dev = torch.device(device)

    # Filter unscored
    valid = Y_train >= 0
    Z_train = Z_train[valid]
    Y_train = Y_train[valid]

    # Ensure batch_size is divisible by L
    batch_size = (batch_size // sequence_length) * sequence_length

    vq = VQBottleneck(codebook_init, commitment_weight).to(dev)
    optimizer = torch.optim.Adam(vq.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Z_t = torch.from_numpy(Z_train).float()
    Y_t = torch.from_numpy(Y_train).long()

    n_samples = len(Z_t)

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        Z_t = Z_t[perm]
        Y_t = Y_t[perm]

        total_loss = 0.0
        total_ce = 0.0
        total_vq = 0.0
        n_batches = 0

        for i in range(0, n_samples - batch_size + 1, batch_size):
            z = Z_t[i : i + batch_size].to(dev)
            y = Y_t[i : i + batch_size].to(dev)

            # Quantize with STE
            z_q, vq_loss = vq(z)

            # Reshape to sequences and forward through frozen downstream
            B = batch_size // sequence_length
            z_seq = z_q.reshape(B, sequence_length, -1)
            y_seq = y.reshape(B, sequence_length)

            with torch.no_grad():
                logits = downstream_fn(z_seq)  # (B, L, n_classes)

            # Classification loss (needs gradients through z_q → codebook)
            # Re-run with gradients enabled for the quantized input
            logits = downstream_fn(z_seq)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            y_flat = y_seq.reshape(-1)

            ce_loss = loss_fn(logits_flat, y_flat)
            loss = ce_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_vq += vq_loss.item()
            n_batches += 1

        if n_batches > 0:
            avg_loss = total_loss / n_batches
            avg_ce = total_ce / n_batches
            avg_vq = total_vq / n_batches
            print(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"loss={avg_loss:.4f} (CE={avg_ce:.4f}, VQ={avg_vq:.4f})"
            )

    return vq.codebook.detach().cpu().numpy()
