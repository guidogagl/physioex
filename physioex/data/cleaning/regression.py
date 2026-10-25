"""Band-wise linear regression between channels of the same night."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Partition of 0.3-35 Hz used for the band-wise regression. Edges are contiguous,
# so the bands of fft_band sum back to the 0.3-35 Hz content of the input.
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.3, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 35.0),
)


def fft_band(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    """Exact band split in the FFT domain: keep the bins with ``lo <= f < hi``.

    Contiguous bands sum back to the signal restricted to their union, which
    overlapping FIR bands do not.
    """
    x = np.asarray(x, dtype=np.float64)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return np.fft.irfft(X * ((f >= lo) & (f < hi)), len(x))


def epoch_mask(
    hypnogram: Sequence[int],
    stages: Iterable[int],
    n: int,
    fs: float,
    epoch_s: float = 30.0,
) -> np.ndarray:
    """Boolean sample mask of length ``n``, True inside the epochs whose label is in ``stages``."""
    step = int(round(epoch_s * fs))
    m = np.zeros(n, dtype=bool)
    for e in np.where(np.isin(np.asarray(hypnogram), list(stages)))[0]:
        m[e * step : (e + 1) * step] = True
    return m


def band_regression(
    target: Dict[str, np.ndarray],
    regressors: Dict[str, np.ndarray],
    fit: np.ndarray,
    fs: float,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    check: Optional[np.ndarray] = None,
    max_gain: float = np.inf,
    undone: Optional[List[Tuple[str, float]]] = None,
    rcond: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, float], np.ndarray]]:
    """Remove from each target channel its band-wise linear prediction from the regressors.

    In every band, each target is regressed on all the regressors with least
    squares fitted only on the samples where ``fit`` is True; the residual is
    then computed on the whole night. The output is the sum of the residuals
    over ``bands``, so content outside the bands is dropped.

    Typical use: target = EOG, regressors = EEG, fit = N2+N3 epochs, where the
    eyes are still and what the EOG shares with the EEG is brain.

    A subtraction lowers the power of the band on the fit samples, but it can
    raise it elsewhere when a regressor is near flat in the band (a dead
    electrode gives huge coefficients). With ``check`` (a sample mask) a band
    whose median squared amplitude over ``check`` grows above ``max_gain`` times
    that of the input is left
    as it was, its coefficients set to 0 and ``(name, lo)`` appended to
    ``undone``.

    Two nearly collinear regressors (two electrodes that differ by little) give
    huge coefficients of opposite sign: they cancel on the fit samples and blow up
    the small difference elsewhere. With ``rcond > 0`` the regressors are
    standardised on the fit samples and the directions whose singular value is
    below ``rcond`` times the largest are dropped from the solution.

    Returns ``(cleaned, coef)`` with ``coef[(name, lo)]`` the coefficients of
    the band starting at ``lo``, one per regressor in insertion order.
    """
    fit = np.asarray(fit, dtype=bool)
    if not fit.any():
        raise ValueError("band_regression: the fit mask selects no sample")
    out = {k: np.zeros(len(v), dtype=np.float64) for k, v in target.items()}
    coef: Dict[Tuple[str, float], np.ndarray] = {}
    for lo, hi in bands:
        E = np.stack([fft_band(v, lo, hi, fs) for v in regressors.values()], 1)
        for k, v in target.items():
            y = fft_band(v, lo, hi, fs)
            if rcond > 0:
                sd = E[fit].std(0)
                sd[sd == 0] = 1.0
                b = np.linalg.lstsq(E[fit] / sd, y[fit], rcond=rcond)[0] / sd
            else:
                b, *_ = np.linalg.lstsq(E[fit], y[fit], rcond=None)
            r = y - E @ b
            # median of the squares: a few huge artefacts must not hide a band that grows everywhere else
            if check is not None and check.any() and np.median(r[check] ** 2) > max_gain * np.median(y[check] ** 2):
                r, b = y, np.zeros_like(b)
                if undone is not None:
                    undone.append((k, lo))
            out[k] += r
            coef[(k, lo)] = b
    return out, coef
