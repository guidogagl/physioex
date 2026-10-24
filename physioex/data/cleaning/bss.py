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


def amuse(X: np.ndarray, lag: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BSS-CCA in its AMUSE form (De Clercq et al. 2006): whitening, then the
    eigenvectors of the symmetrised lag-``lag`` covariance. The eigenvalues are
    the lag autocorrelations of the sources, so broadband muscle activity ends
    up in the components with the lowest values.

    Returns ``(U, A, autocorr)``, sources sorted by decreasing autocorrelation.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(1, keepdims=True)
    d, E = np.linalg.eigh(np.cov(X))
    keep = d > 1e-10 * d.max()
    W = (E[:, keep] / np.sqrt(d[keep])).T
    Z = W @ X
    C = Z[:, lag:] @ Z[:, :-lag].T / (Z.shape[1] - lag)
    r, V = np.linalg.eigh((C + C.T) / 2)
    order = np.argsort(r)[::-1]
    U = V[:, order].T @ W
    return U, np.linalg.pinv(U), r[order]


def _is_front(k: str) -> bool:
    u = k.upper()
    return u.startswith(("FP", "F", "EOG", "E1", "E2"))


def _is_back(k: str) -> bool:
    u = k.upper()
    return u.startswith(("O", "P", "CZOZ"))


def classify_ocular(
    S: np.ndarray,
    A: np.ndarray,
    keys: Sequence[str],
    ocular_keys: Sequence[str],
    fs: float,
    active: np.ndarray,
    quiet: np.ndarray,
    lf_hz: float = 5.0,
    lf_share: float = 0.6,
    activity_ratio: float = 1.5,
    back_over_front: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Ocular components by three criteria, all required:

    (a) topography: the largest contribution ``|A[i, k]| * std(S_k)`` is on an
        ocular or frontal channel (EOG, Fp*, F*), and occipito-parietal
        channels carry less than ``back_over_front`` of it;
    (b) spectrum: at least ``lf_share`` of the 0.3-35 Hz power below ``lf_hz``
        (slow eye movements, saccades and blinks);
    (c) state: variance in ``active`` samples (REM + wake) at least
        ``activity_ratio`` times the variance in ``quiet`` samples (N2 + N3),
        which keeps frontal slow waves out.

    Returns the boolean mask and the per-component criteria for diagnostics.
    """
    keys = list(keys)
    contrib = np.abs(A) * S.std(1)[None, :]
    front = np.array([k in ocular_keys or _is_front(k) for k in keys])
    back = np.array([_is_back(k) for k in keys])
    top = contrib.argmax(0)
    fmax = contrib[front].max(0) if front.any() else np.zeros(S.shape[0])
    bmax = contrib[back].max(0) if back.any() else np.zeros(S.shape[0])
    a = front[top] & (bmax < back_over_front * np.maximum(fmax, 1e-12))
    F = np.fft.rfft(S, axis=1)
    f = np.fft.rfftfreq(S.shape[1], 1.0 / fs)
    P = np.abs(F) ** 2
    tot = P[:, (f >= 0.3) & (f < 35)].sum(1) + 1e-12
    lf = P[:, (f >= 0.3) & (f < lf_hz)].sum(1) / tot
    b = lf >= lf_share
    if active.any() and quiet.any():
        ratio = S[:, active].var(1) / (S[:, quiet].var(1) + 1e-12)
    else:
        ratio = np.full(S.shape[0], np.nan)
    c = ratio >= activity_ratio
    return a & b & c, {"topography": a, "lf_share": lf, "activity_ratio": ratio}
