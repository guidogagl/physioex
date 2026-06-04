"""Post-hoc prototype discovery via Non-negative Matrix Factorization.

Inspired by Tan, Zhou & Chen, "Post-hoc Part-Prototype Networks", ICML 2024
(arXiv:2406.03421).

Given pre-extracted epoch embeddings Z and labels Y from a frozen encoder h(x),
discovers k prototypes per class by applying NMF on class-conditional embeddings.

Each class c is described by k basis vectors (prototypes) such that the
embeddings of class c can be approximately reconstructed as non-negative
linear combinations of these prototypes.

Since NMF requires non-negative inputs and neural embeddings may contain
negative values, we apply a shift: Z_pos = Z - Z.min(axis=0).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF


def discover_prototypes_nmf(
    Z: np.ndarray,
    Y: np.ndarray,
    k_per_class: int = 10,
    n_classes: int = 5,
    random_state: int = 42,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Discover prototypes via per-class NMF decomposition.

    For each class c, collects all embeddings F_c = Z[Y == c] and runs
    NMF: F_c ≈ E_c @ P_c, where P_c has k rows (the prototypes).

    Args:
        Z: (N, d_model) epoch embeddings (float32).
        Y: (N,) integer class labels.
        k_per_class: Number of prototypes to discover per class.
        n_classes: Total number of classes.
        random_state: Seed for NMF.
        max_iter: Maximum NMF iterations.

    Returns:
        Tuple of:
            prototypes: (k_per_class * n_classes, d_model)
            proto_labels: (k_per_class * n_classes,) class label per prototype
    """
    # Shift to non-negative (per-feature minimum)
    z_min = Z.min(axis=0, keepdims=True)
    Z_pos = Z - z_min  # all >= 0

    prototypes_list = []
    labels_list = []

    for c in range(n_classes):
        mask = Y == c
        F_c = Z_pos[mask]

        if len(F_c) < k_per_class:
            # Not enough samples — use all as prototypes
            k = len(F_c)
        else:
            k = k_per_class

        if len(F_c) == 0:
            continue

        nmf = NMF(
            n_components=k,
            init="nndsvda",
            random_state=random_state,
            max_iter=max_iter,
        )
        nmf.fit(F_c)
        P_c = nmf.components_  # (k, d_model) in shifted space

        # Shift back to original space
        P_c = P_c + z_min

        prototypes_list.append(P_c)
        labels_list.append(np.full(k, c, dtype=np.int64))

    prototypes = np.concatenate(prototypes_list, axis=0)
    proto_labels = np.concatenate(labels_list, axis=0)

    return prototypes, proto_labels
