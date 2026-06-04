"""Shared utilities for post-hoc prototype learning.

Functions for loading pre-extracted epoch embeddings, nearest-prototype
classification, and evaluation metrics.  Works with any model that
produces per-epoch embeddings in R^d_model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def load_epoch_embeddings(
    emb_dir: str | Path,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-extracted epoch embeddings and labels for a split.

    Expects files ``{split}_embeddings.npy`` and ``{split}_labels.npy``
    in *emb_dir*.

    Args:
        emb_dir: Directory containing the .npy files.
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        Tuple of (Z, Y) where Z is (N, d_model) float32 and Y is (N,) int64.
    """
    emb_dir = Path(emb_dir)
    Z = np.load(emb_dir / f"{split}_embeddings.npy").astype(np.float32)
    Y = np.load(emb_dir / f"{split}_labels.npy").astype(np.int64)
    return Z, Y


def nearest_prototype_classify(
    Z: np.ndarray,
    prototypes: np.ndarray,
    proto_labels: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Classify embeddings via nearest prototype.

    For each embedding z_i, finds the prototype with highest similarity
    (cosine) or lowest distance (euclidean) and assigns its class label.

    Args:
        Z: (N, d_model) embeddings to classify.
        prototypes: (M, d_model) prototype vectors.
        proto_labels: (M,) class label for each prototype.
        metric: ``"cosine"`` or ``"euclidean"``.

    Returns:
        (N,) predicted class labels.
    """
    if metric == "cosine":
        # Normalize both
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        P_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        # Similarity matrix (N, M)
        sim = Z_norm @ P_norm.T
        best_idx = sim.argmax(axis=1)
    elif metric == "euclidean":
        # Distance matrix (N, M) via broadcasting
        # ||z - p||^2 = ||z||^2 + ||p||^2 - 2 z.p
        Z_sq = (Z ** 2).sum(axis=1, keepdims=True)  # (N, 1)
        P_sq = (prototypes ** 2).sum(axis=1, keepdims=True).T  # (1, M)
        dist = Z_sq + P_sq - 2 * (Z @ prototypes.T)
        best_idx = dist.argmin(axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    return proto_labels[best_idx]


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    n_classes: int = 5,
) -> dict:
    """Compute standard sleep staging metrics.

    Args:
        y_true: (N,) ground truth labels.
        y_pred: (N,) predicted labels.
        class_names: Names for each class (default: W, N1, N2, N3, REM).
        n_classes: Number of classes.

    Returns:
        Dict with accuracy, f1_macro, kappa, per_class_f1, confusion_matrix.
    """
    if class_names is None:
        class_names = ["W", "N1", "N2", "N3", "REM"][:n_classes]

    # Filter out unscored (-1)
    mask = y_true >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    N = len(y_true)
    accuracy = float((y_true == y_pred).sum()) / N if N > 0 else 0.0

    # Per-class F1
    per_class_f1 = {}
    for c, name in enumerate(class_names):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[name] = round(f1, 4)

    f1_macro = float(np.mean(list(per_class_f1.values())))

    # Cohen's kappa
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    p_o = cm.trace() / cm.sum() if cm.sum() > 0 else 0.0
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = float((row_sums * col_sums).sum()) / (cm.sum() ** 2) if cm.sum() > 0 else 0.0
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "kappa": round(kappa, 4),
        "per_class_f1": per_class_f1,
        "n_epochs": N,
        "confusion_matrix": cm.tolist(),
    }
