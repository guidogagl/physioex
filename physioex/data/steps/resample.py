import numpy as np
from scipy.signal import resample
from physioex.data.pipeline import PreprocessingStep, CompiledStep


class Resample(PreprocessingStep):
    """Resample to target sampling rate."""

    def __init__(self, target_fs: float):
        self.target_fs = float(target_fs)

    def spec(self) -> str:
        return self._build_spec(target_fs=self.target_fs)

    def compile(self, fs_in: float) -> CompiledStep:
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
