"""Blind source separation across the channels of a night."""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np


def sobi(
    X: np.ndarray,
    lags: Sequence[int],
    max_sweeps: int = 100,
    tol: float = 1e-8,
    rank_tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """SOBI (Belouchrani et al. 1997): whitening, then joint diagonalisation of
    the symmetrised lagged covariances by Jacobi rotations.

    ``X`` is (channels, samples). Returns ``(U, A)``: unmixing and mixing
    matrices, with sources ``S = U @ X`` and ``X ≈ A @ S`` (after removing the
    channel means). Rank-deficient inputs (e.g. average reference, or a channel
    that is a combination of others) keep only the whitening directions with
    eigenvalue above ``rank_tol`` times the largest, so U is (rank, channels).
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(1, keepdims=True)
    d, E = np.linalg.eigh(np.cov(X))
    keep = d > rank_tol * d.max()
    d, E = d[keep], E[:, keep]
    W = (E / np.sqrt(d)).T
    Z = W @ X
    n, T = Z.shape
    M = []
    for t in lags:
        C = Z[:, t:] @ Z[:, :-t].T / (T - t)
        M.append((C + C.T) / 2)
    M = np.concatenate(M, 1)
    V = np.eye(n)
    for _ in range(max_sweeps):
        done = True
        for p in range(n - 1):
            for q in range(p + 1, n):
                g = np.array([M[p, p::n] - M[q, q::n], M[p, q::n] + M[q, p::n]])
                gg = g @ g.T
                ton, toff = gg[0, 0] - gg[1, 1], gg[0, 1] + gg[1, 0]
                th = 0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                c, s = np.cos(th), np.sin(th)
                if abs(s) > tol:
                    done = False
                    G = np.array([[c, -s], [s, c]])
                    idx = [p, q]
                    M[idx, :] = G.T @ M[idx, :]
                    for k in range(len(lags)):
                        cols = [p + k * n, q + k * n]
                        M[:, cols] = M[:, cols] @ G
                    V[:, idx] = V[:, idx] @ G
        if done:
            break
    U = V.T @ W
    return U, np.linalg.pinv(U)


class SplitResult(NamedTuple):
    eog_ocular: Dict[str, np.ndarray]  # EOG channels rebuilt from the ocular components
    eeg_cerebral: Dict[
        str, np.ndarray
    ]  # EEG channels rebuilt from the other components
    eog_cerebral: Dict[str, np.ndarray]  # what the split removed from the EOG channels
    ocular: np.ndarray  # bool per component
    mixing: np.ndarray  # A, (channels, components), rows in the order of keys
    keys: List[str]


def split_ocular(
    eog: Dict[str, np.ndarray],
    eeg: Dict[str, np.ndarray],
    fit: Optional[np.ndarray] = None,
    lags: Sequence[int] = tuple(range(1, 101, 5)),
    subsample: int = 2,
) -> SplitResult:
    """Two-way SOBI split of the night into ocular and non-ocular parts.

    SOBI is estimated on the samples in ``fit`` (all if None), taking one every
    ``subsample``; the sources are then computed on the whole night. A
    component is ocular when its largest absolute loading is on an EOG channel.
    No share threshold: with frontal EEG channels the ocular sources load there
    too, so a threshold would depend on the montage.

    The two parts sum back to the mean-removed input on every channel.
    """
    keys = list(eog) + list(eeg)
    X = np.stack(
        [np.asarray(eog[k] if k in eog else eeg[k], dtype=np.float64) for k in keys]
    )
    X = X - X.mean(1, keepdims=True)
    idx = (
        np.arange(X.shape[1])
        if fit is None
        else np.where(np.asarray(fit, dtype=bool))[0]
    )
    U, A = sobi(X[:, idx[::subsample]], lags=lags)
    S = U @ X
    ocu = np.abs(A).argmax(0) < len(eog)
    Xo, Xc = A[:, ocu] @ S[ocu], A[:, ~ocu] @ S[~ocu]
    ne = len(eog)
    return SplitResult(
        eog_ocular={k: Xo[i] for i, k in enumerate(eog)},
        eeg_cerebral={k: Xc[ne + i] for i, k in enumerate(eeg)},
        eog_cerebral={k: Xc[i] for i, k in enumerate(eog)},
        ocular=ocu,
        mixing=A,
        keys=keys,
    )
