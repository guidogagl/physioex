"""Utility functions for CSD example: linear probe training and loading.

Provides a LinearProbeWithLN module (LayerNorm + Linear) compatible with
CSD's probe_weights format, plus functions to train on embeddings and
save/load probe weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class LinearProbeWithLN(nn.Module):
    """Linear classifier with LayerNorm, compatible with CSD probe_weights.

    Architecture: embedding -> LayerNorm -> Linear -> logits

    This matches the probe structure expected by CSD, which requires:
    - "ln": LayerNorm weights (normalized embeddings)
    - "W": Linear weights (n_classes, D)

    Args:
        embedding_dim: Dimension of input embeddings (D).
        n_classes: Number of output classes (default 5 for sleep staging).
    """

    def __init__(self, embedding_dim: int, n_classes: int = 5):
        super().__init__()
        self.ln = nn.LayerNorm(embedding_dim)
        self.W = nn.Linear(embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> LayerNorm -> Linear -> logits.

        Args:
            x: (B, D) embeddings.

        Returns:
            (B, n_classes) logits.
        """
        x_normed = self.ln(x)
        return self.W(x_normed)

    def to_probe_weights(self) -> Dict[str, nn.Module]:
        """Convert to CSD-compatible probe_weights dict.

        Returns:
            {"ln": LayerNorm module, "W": Linear weight tensor}
        """
        return {"ln": self.ln, "W": self.W.weight}


def load_embeddings_from_cache(
    model_name: str,
    dataset_name: str,
    subject_ids: List[str],
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-computed embeddings from cache for specified subjects.

    Uses the PhysioEx embedding cache system to load .npy files
    instead of recomputing embeddings.

    Args:
        model_name: Model name for cache directory (e.g., "cbramod").
        dataset_name: Dataset name for cache (e.g., "mass_ss03").
        subject_ids: List of subject IDs to load.
        cache_dir: Override cache root directory.

    Returns:
        embeddings: (N, D) concatenated embeddings from all subjects.
        labels: (N,) corresponding sleep stage labels.
    """
    from physioex.models.embed import _cache_root, _load_npy_as_float32

    emb_dir = _cache_root(cache_dir) / model_name / dataset_name
    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_dir}. "
            f"Extract them first with extract_embeddings() for {model_name}/{dataset_name}"
        )

    all_embeddings = []
    all_labels = []

    for subject_id in subject_ids:
        subj_dir = emb_dir / subject_id
        emb_path = subj_dir / "embeddings.npy"
        lbl_path = subj_dir / "labels.npy"

        if not emb_path.exists() or not lbl_path.exists():
            print(f"  [SKIP] {subject_id}: no cached embeddings")
            continue

        try:
            emb = _load_npy_as_float32(emb_path)
            lbl = np.load(str(lbl_path)).astype(np.int64)

            # Filter out unscored epochs
            valid_mask = lbl >= 0
            emb = emb[valid_mask]
            lbl = lbl[valid_mask]

            all_embeddings.append(emb)
            all_labels.append(lbl)

            print(f"  {subject_id}: {len(emb)} epochs loaded")

        except Exception as e:
            print(f"  [SKIP] {subject_id}: {e}")
            continue

    if not all_embeddings:
        raise ValueError(f"No embeddings loaded for any of {len(subject_ids)} subjects")

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def train_and_save_probe(
    model_name: str,
    dataset_name: str,
    train_subjects: List[str],
    valid_subjects: List[str],
    output_path: str | Path,
    embedding_dim: int,
    device: str = "cpu",
    max_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    n_classes: int = 5,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Train linear probe on cached embeddings and save weights.

    Loads pre-computed embeddings from the PhysioEx cache, performs
    standard scaling, trains LinearProbeWithLN, evaluates on validation
    set, and saves the trained probe weights.

    Args:
        model_name: Model name for cache (e.g., "cbramod").
        dataset_name: Dataset name for cache (e.g., "mass_ss03").
        train_subjects: Subject IDs for training.
        valid_subjects: Subject IDs for validation.
        output_path: Path to save probe.pt and metrics.json.
        embedding_dim: Embedding dimension D.
        device: Device string.
        max_epochs: Training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        batch_size: Training batch size.
        n_classes: Number of classes.
        cache_dir: Override cache root directory.

    Returns:
        Results dict with metrics and probe info.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings from cache...")
    train_emb, train_lbl = load_embeddings_from_cache(
        model_name, dataset_name, train_subjects, cache_dir
    )
    valid_emb, valid_lbl = load_embeddings_from_cache(
        model_name, dataset_name, valid_subjects, cache_dir
    )

    print(f"Train: {train_emb.shape[0]} epochs")
    print(f"Valid: {valid_emb.shape[0]} epochs")

    # Standard scaling
    mean = train_emb.mean(axis=0)
    std = train_emb.std(axis=0) + 1e-8
    train_emb_scaled = (train_emb - mean) / std
    valid_emb_scaled = (valid_emb - mean) / std

    # Convert to tensors
    X_train = torch.from_numpy(train_emb_scaled).float()
    y_train = torch.from_numpy(train_lbl).long()
    X_valid = torch.from_numpy(valid_emb_scaled).float()
    y_valid = torch.from_numpy(valid_lbl).long()

    D = X_train.shape[1]

    # Create probe
    probe = LinearProbeWithLN(embedding_dim=D, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    # Training loop
    best_acc = 0.0
    patience_counter = 0
    patience = 10

    print("\nTraining...")
    for epoch in range(max_epochs):
        probe.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(probe(xb), yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * xb.shape[0]

        # Evaluate
        probe.eval()
        with torch.no_grad():
            valid_logits = probe(X_valid.to(device))
            valid_preds = valid_logits.argmax(dim=-1).cpu()
            acc = (valid_preds == y_valid).float().mean().item()

        epoch_loss = epoch_loss / len(X_train)
        print(f"  Epoch {epoch+1}/{max_epochs}: loss={epoch_loss:.4f}, val_acc={acc:.4f}")

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            # Save best probe
            torch.save(probe.state_dict(), output_path / "probe.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best probe for final eval
    probe.load_state_dict(torch.load(output_path / "probe.pt", weights_only=True))
    probe.eval()

    # Final metrics
    with torch.no_grad():
        X_valid_dev = X_valid.to(device)
        y_valid_dev = y_valid.to(device)
        valid_logits = probe(X_valid_dev)
        valid_preds = valid_logits.argmax(dim=-1)

        from physioex.train.metrics import (
            accuracy_score,
            f1_score,
            cohen_kappa_score,
        )

        final_acc = accuracy_score(valid_logits, y_valid_dev, ignore_index=None)
        final_f1 = f1_score(valid_logits, y_valid_dev, ignore_index=None)
        final_kappa = cohen_kappa_score(valid_logits, y_valid_dev, ignore_index=None)

    print(f"\nFinal: ACC={final_acc:.4f}, MF1={final_f1:.4f}, kappa={final_kappa:.4f}")

    # Save results
    results = {
        "embedding_dim": D,
        "n_classes": n_classes,
        "n_train_subjects": len(train_subjects),
        "n_valid_subjects": len(valid_subjects),
        "n_train_epochs": len(train_emb),
        "n_valid_epochs": len(valid_emb),
        "train_subjects": train_subjects,
        "valid_subjects": valid_subjects,
        "metrics": {
            "accuracy": round(float(final_acc), 4),
            "macro_f1": round(float(final_f1), 4),
            "kappa": round(float(final_kappa), 4),
        },
        "config": {
            "max_epochs": max_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved probe to {output_path / 'probe.pt'}")
    print(f"Saved metrics to {output_path / 'metrics.json'}")

    return results


def load_probe(
    probe_path: str | Path,
    embedding_dim: int,
    n_classes: int = 5,
    device: str = "cpu",
) -> Dict[str, nn.Module]:
    """Load trained probe weights for CSD.

    Args:
        probe_path: Path to probe.pt file.
        embedding_dim: Embedding dimension D.
        n_classes: Number of classes.
        device: Device to load modules on.

    Returns:
        probe_weights dict: {"ln": LayerNorm, "W": Linear weight}
    """
    probe_path = Path(probe_path)

    probe = LinearProbeWithLN(embedding_dim=embedding_dim, n_classes=n_classes)
    state = torch.load(probe_path, map_location=device, weights_only=True)
    probe.load_state_dict(state)
    probe = probe.to(device).eval()

    return probe.to_probe_weights()


def split_subjects(
    subject_ids: List[str],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Random split subjects into train and valid sets.

    Args:
        subject_ids: List of all subject IDs.
        train_ratio: Fraction for training (default 0.7).
        seed: Random seed for reproducibility.

    Returns:
        train_subjects, valid_subjects: Lists of subject IDs.
    """
    rng = np.random.RandomState(seed)
    n_train = int(len(subject_ids) * train_ratio)
    indices = rng.permutation(len(subject_ids))

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]

    return (
        [subject_ids[i] for i in train_indices],
        [subject_ids[i] for i in valid_indices],
    )
