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
    Z_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
    n_epochs: int = 20,
    patience: int = 5,
    batch_size: int = 2048,
    lr: float = 1e-3,
    commitment_weight: float = 0.25,
    device: str = "cpu",
    sequence_length: int = 21,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Supervised codebook refinement (Stage 2).

    Optimizes the codebook so that quantized embeddings, when passed
    through the frozen downstream model, preserve classification accuracy.

    Monitors validation loss for early stopping and returns the best
    codebook (lowest val loss).

    Args:
        Z_train: (N, d_model) training epoch embeddings.
        Y_train: (N,) training labels.
        downstream_fn: Callable that takes (B, L, d_model) quantized
            embeddings and returns (B, L, n_classes) logits.
            Typically: sequence_encoder -> classifier (both frozen).
        codebook_init: (M, d_model) initial codebook from K-Means.
        Z_val: (N_val, d_model) validation embeddings. If None, no
            early stopping — runs all n_epochs.
        Y_val: (N_val,) validation labels.
        n_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without val improvement).
        batch_size: Must be divisible by sequence_length.
        lr: Learning rate for codebook.
        commitment_weight: beta for commitment loss.
        device: Torch device string.
        sequence_length: L, sequence length to reshape embeddings into.
        save_path: If set, save best codebook to this path on every
            val improvement. Allows recovery if the job crashes.

    Returns:
        codebook: (M, d_model) best codebook as numpy array.
    """
    dev = torch.device(device)
    L = sequence_length

    # Filter unscored
    valid_mask = Y_train >= 0
    Z_train, Y_train = Z_train[valid_mask], Y_train[valid_mask]

    has_val = Z_val is not None and Y_val is not None
    if has_val:
        val_mask = Y_val >= 0
        Z_val, Y_val = Z_val[val_mask], Y_val[val_mask]

    # Ensure batch_size is divisible by L
    batch_size = (batch_size // L) * L

    vq = VQBottleneck(codebook_init, commitment_weight).to(dev)
    optimizer = torch.optim.Adam(vq.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Z_t = torch.from_numpy(Z_train).float()
    Y_t = torch.from_numpy(Y_train).long()
    n_samples = len(Z_t)

    if has_val:
        Z_v = torch.from_numpy(Z_val).float()
        Y_v = torch.from_numpy(Y_val).long()
        n_val = len(Z_v)
        val_batch = (n_val // L) * L  # trim to multiple of L

    best_val_loss = float("inf")
    best_codebook = codebook_init.copy()
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # --- Train ---
        vq.train()
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

            z_q, vq_loss = vq(z)

            B = batch_size // L
            logits = downstream_fn(z_q.reshape(B, L, -1))
            logits_flat = logits.reshape(-1, logits.shape[-1])
            y_flat = y.reshape(B, L).reshape(-1)

            ce_loss = loss_fn(logits_flat, y_flat)
            loss = ce_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_vq += vq_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce = total_ce / max(n_batches, 1)
        avg_vq = total_vq / max(n_batches, 1)

        # --- Validation ---
        val_str = ""
        if has_val:
            vq.eval()
            val_ce_total = 0.0
            val_n = 0
            with torch.no_grad():
                for i in range(0, val_batch - batch_size + 1, batch_size):
                    z = Z_v[i : i + batch_size].to(dev)
                    y = Y_v[i : i + batch_size].to(dev)

                    z_q, _ = vq(z)
                    B = batch_size // L
                    logits = downstream_fn(z_q.reshape(B, L, -1))
                    logits_flat = logits.reshape(-1, logits.shape[-1])
                    y_flat = y.reshape(B, L).reshape(-1)

                    val_ce_total += loss_fn(logits_flat, y_flat).item()
                    val_n += 1

            val_loss = val_ce_total / max(val_n, 1)
            val_str = f", val_CE={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_codebook = vq.codebook.detach().cpu().numpy().copy()
                epochs_no_improve = 0
                if save_path is not None:
                    np.save(save_path, best_codebook)
            else:
                epochs_no_improve += 1

        print(
            f"  Epoch {epoch+1}/{n_epochs}: "
            f"loss={avg_loss:.4f} (CE={avg_ce:.4f}, VQ={avg_vq:.4f}){val_str}"
        )

        if has_val and epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if not has_val:
        best_codebook = vq.codebook.detach().cpu().numpy()

    return best_codebook
