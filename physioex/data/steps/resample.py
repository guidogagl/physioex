from typing import Optional

import numpy as np
from scipy.signal import resample
from physioex.data.pipeline import PreprocessingStep, CompiledStep


class Resample(PreprocessingStep):
    """Resample to target sampling rate."""

    def __init__(self, target_fs: float):
        self.target_fs = float(target_fs)

    def spec(self) -> str:
        return self._build_spec(target_fs=self.target_fs)

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        target = self.target_fs
        if fs_in == target:
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32), fs_out=target
            )

        def apply(signal: np.ndarray) -> np.ndarray:
            n_out = int(round(signal.shape[-1] * target / fs_in))
            y = resample(np.asarray(signal, dtype=np.float64), n_out, axis=-1)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=target)


class FFTResample(PreprocessingStep):
    """FFT-based resample (``scipy.signal.resample``) to a target rate.

    Distinct from :class:`Resample`: the output length is computed with
    ``int(len(x) * target_fs / fs)`` (truncating, no rounding), which is
    what the SleepPPG-Net reference code uses.

    Parameters
    ----------
    target_fs : float
        Target sampling rate in Hz.
    """

    def __init__(self, target_fs: float):
        if target_fs <= 0:
            raise ValueError(f"target_fs must be > 0; got {target_fs}")
        self.target_fs = float(target_fs)

    def spec(self) -> str:
        return self._build_spec(target_fs=self.target_fs)

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        target = self.target_fs
        if fs_in == target:
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32), fs_out=target
            )

        def apply(signal: np.ndarray) -> np.ndarray:
            n_out = int(signal.shape[-1] * target / fs_in)
            y = resample(np.asarray(signal, dtype=np.float64), n_out)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=target)
