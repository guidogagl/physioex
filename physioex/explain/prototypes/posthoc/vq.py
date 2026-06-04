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


class _WindowIndex:
    """Maps a flat window index to (subject_idx, start_epoch).

    Avoids materializing all windows in memory. Each subject with N epochs
    contributes max(0, N - L + 1) windows.
    """

    def __init__(self, subjects: list[tuple[np.ndarray, np.ndarray]], L: int):
        self.subjects = subjects
        self.L = L
        self.offsets = []  # (cumulative_start, subject_idx)
        total = 0
        for i, (Z, _) in enumerate(subjects):
            n_win = max(0, Z.shape[0] - L + 1)
            if n_win > 0:
                self.offsets.append((total, i))
                total += n_win
        self.n_windows = total

    def get_batch(self, indices: np.ndarray, device: torch.device):
        """Fetch a batch of windows by flat indices.

        Returns (B, L, d_model) and (B, L) tensors.
        """
        L = self.L
        z_list, y_list = [], []
        for idx in indices:
            # Binary search for subject
            lo, hi = 0, len(self.offsets) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.offsets[mid][0] <= idx:
                    lo = mid
                else:
                    hi = mid - 1
            cum_start, subj_i = self.offsets[lo]
            win_start = idx - cum_start
            Z_s, Y_s = self.subjects[subj_i]
            z_list.append(Z_s[win_start : win_start + L])
            y_list.append(Y_s[win_start : win_start + L])

        z = torch.from_numpy(np.stack(z_list)).float().to(device)
        y = torch.from_numpy(np.stack(y_list)).long().to(device)
        return z, y


def train_codebook(
    train_subjects: list[tuple[np.ndarray, np.ndarray]],
    downstream_fn: Callable[[torch.Tensor], torch.Tensor],
    codebook_init: np.ndarray,
    val_subjects: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    n_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    commitment_weight: float = 0.25,
    device: str = "cpu",
    sequence_length: int = 21,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Supervised codebook refinement (Stage 2).

    Optimizes the codebook so that quantized embeddings, when passed
    through the frozen downstream model, preserve classification accuracy.

    Uses stride-1 sliding windows on real per-subject recordings so the
    sequence encoder sees coherent temporal context. Windows are fetched
    on-the-fly (no pre-materialization) to avoid OOM on large datasets.

    Args:
        train_subjects: List of (Z, Y) tuples per subject.
            Z: (n_epochs, d_model), Y: (n_epochs,).
        downstream_fn: Takes (B, L, d_model) -> (B, L, n_classes) logits.
        codebook_init: (M, d_model) initial codebook from K-Means.
        val_subjects: Optional list of (Z, Y) per subject for validation.
        n_epochs: Maximum training epochs.
        patience: Early stopping patience.
        batch_size: Number of L-length windows per optimization step.
        lr: Learning rate for codebook.
        commitment_weight: beta for commitment loss.
        device: Torch device string.
        sequence_length: L, window length for sliding windows.
        save_path: Save best codebook here on every val improvement.

    Returns:
        codebook: (M, d_model) best codebook as numpy array.
    """
    dev = torch.device(device)
    L = sequence_length

    train_idx = _WindowIndex(train_subjects, L)
    print(f"  Train: {train_idx.n_windows} windows from {len(train_subjects)} subjects")

    has_val = val_subjects is not None and len(val_subjects) > 0
    val_idx = None
    if has_val:
        val_idx = _WindowIndex(val_subjects, L)
        print(f"  Valid: {val_idx.n_windows} windows from {len(val_subjects)} subjects")

    vq = VQBottleneck(codebook_init, commitment_weight).to(dev)
    optimizer = torch.optim.Adam(vq.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_loss = float("inf")
    best_codebook = codebook_init.copy()
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # --- Train ---
        vq.train()
        perm = np.random.permutation(train_idx.n_windows)

        total_loss = 0.0
        total_ce = 0.0
        total_vq = 0.0
        n_batches = 0

        for i in range(0, train_idx.n_windows - batch_size + 1, batch_size):
            batch_idx = perm[i : i + batch_size]
            z, y = train_idx.get_batch(batch_idx, dev)  # (B, L, D), (B, L)

            # Quantize each epoch independently
            B, Lw, D = z.shape
            z_q_flat, vq_loss = vq(z.reshape(B * Lw, D))
            z_q = z_q_flat.reshape(B, Lw, D)

            logits = downstream_fn(z_q)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            y_flat = y.reshape(-1)

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
            val_correct = 0
            val_total = 0
            val_n = 0
            with torch.no_grad():
                for i in range(0, val_idx.n_windows - batch_size + 1, batch_size):
                    z, y = val_idx.get_batch(
                        np.arange(i, i + batch_size), dev
                    )
                    B, Lw, D = z.shape
                    z_q_flat, _ = vq(z.reshape(B * Lw, D))
                    logits = downstream_fn(z_q_flat.reshape(B, Lw, D))
                    logits_flat = logits.reshape(-1, logits.shape[-1])
                    y_flat = y.reshape(-1)

                    scored = y_flat >= 0
                    if scored.any():
                        val_ce_total += loss_fn(logits_flat, y_flat).item()
                        val_correct += (logits_flat[scored].argmax(dim=1) == y_flat[scored]).sum().item()
                        val_total += scored.sum().item()
                    val_n += 1

            val_loss = val_ce_total / max(val_n, 1)
            val_acc = val_correct / max(val_total, 1)
            val_str = f", val_CE={val_loss:.4f}, val_acc={val_acc:.4f}"

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
