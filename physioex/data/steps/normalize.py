import numpy as np
from physioex.data.pipeline import PreprocessingStep, CompiledStep


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

    def compile(self, fs_in: float) -> CompiledStep:
        eps = self.eps

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float32)
            mean = x.mean()
            std = x.std() + eps
            return ((x - mean) / std).astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)
