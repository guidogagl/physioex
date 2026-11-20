import logging
from typing import Optional

import numpy as np
from physioex.data.pipeline import PreprocessingStep, CompiledStep

logger = logging.getLogger("physioex.data.steps")


class ZScoreNormalize(PreprocessingStep):
    """Per-subject per-channel z-score normalization.

    Available as an optional pipeline element; not part of default presets.
    Global dataset-level scaling is a future phase.
    """

    def __init__(self, mode: str = "per_subject", eps: float = 1e-8):
        if mode not in ("per_subject",):
            raise ValueError(f"Only mode='per_subject' supported; got {mode!r}")
        self.mode = mode
        self.eps = float(eps)

    def spec(self) -> str:
        return self._build_spec(eps=self.eps, mode=self.mode)

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        eps = self.eps

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float32)
            mean = x.mean()
            std = x.std() + eps
            return ((x - mean) / std).astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)


class Clip(PreprocessingStep):
    """Symmetric outlier clipping around the signal mean.

    ``y = clip(x, mean - n_std*std, mean + n_std*std)`` with the mean and
    standard deviation (``ddof=0``) computed over the whole signal. This
    is the ``np.clip(x, x.mean()-3*x.std(), x.mean()+3*x.std())`` step of
    the SleepPPG-Net reference preprocessing. The sampling rate is
    unchanged.

    Parameters
    ----------
    n_std : float
        Number of standard deviations for the clip bounds (default 3.0).
    """

    def __init__(self, n_std: float = 3.0):
        self.n_std = float(n_std)

    def spec(self) -> str:
        return self._build_spec(n_std=self.n_std)

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        n_std = self.n_std

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            mean = x.mean()
            std = x.std(ddof=0)
            lo = mean - n_std * std
            hi = mean + n_std * std
            return np.clip(x, lo, hi).astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)


class ZScore(PreprocessingStep):
    """Whole-signal z-score with a tunable ``ddof`` and a std floor.

    ``y = (x - mean) / max(std, eps)`` where the mean and standard
    deviation (with the given ``ddof``) are computed over the whole
    signal. Used by the wav2sleep reference preprocessing, which
    z-scores each 10 h recording with ``ddof=1, eps=1e-6`` **over the
    full grid including the zero padding** (the padding is therefore part
    of the statistics -- keep :class:`~physioex.data.steps.interp.FixedLength`
    before this step in that pipeline).

    If ``std == 0`` and ``eps == 0`` the step returns ``x - mean``
    (a constant signal is degenerate for a ratio).

    Parameters
    ----------
    ddof : int
        Delta degrees of freedom for the standard deviation (default 0).
    eps : float
        Floor for the standard deviation (default 0.0).
    """

    def __init__(self, ddof: int = 0, eps: float = 0.0):
        if ddof < 0:
            raise ValueError(f"ddof must be >= 0; got {ddof}")
        self.ddof = int(ddof)
        self.eps = float(eps)

    def spec(self) -> str:
        return self._build_spec(ddof=self.ddof, eps=self.eps)

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        ddof = self.ddof
        eps = self.eps

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            mean = x.mean()
            # ddof is only valid for a non-empty signal; a degenerate
            # (all-NaN) signal is left unchanged.
            if x.size == 0 or not np.isfinite(x).any():
                return x.astype(np.float32)
            std = x.std(ddof=ddof)
            if std == 0 and eps == 0:
                return (x - mean).astype(np.float32)
            return ((x - mean) / max(std, eps)).astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)
