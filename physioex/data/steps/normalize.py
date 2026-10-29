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


class NanToZero(PreprocessingStep):
    """Replace NaN (and +-inf) with finite values (``np.nan_to_num``)."""

    def spec(self) -> str:
        return self._build_spec()

    def compile(self, fs_in: float) -> CompiledStep:
        return CompiledStep(
            apply=lambda x: np.nan_to_num(np.asarray(x, dtype=np.float64)), fs_out=fs_in
        )


class RobustScale(PreprocessingStep):
    """Robust per-night scale: ``(x - median) / (IQR / 1.349 + eps)``, clipped to +-clip.

    A detached electrode or a saturated stretch does not shrink the rest of the
    night (the sd was off by 5-10x there). The statistics skip samples that are
    exactly zero (zero-filled gaps and the stretch outside the night window of
    the clean channels) and, for long nights, use a regular subsample of 2 M
    points. Same spec as ``microenc.phys_steps.RobustScale`` of protosleepnet-v3,
    so the caches written there are reused.
    """

    IQR_TO_SD = 1.349  # IQR of a Gaussian = 1.349 sd

    def __init__(self, clip: float = 20.0, eps: float = 1e-6):
        self.clip = float(clip)
        self.eps = float(eps)

    def spec(self) -> str:
        return self._build_spec(clip=self.clip, eps=self.eps, k=self.IQR_TO_SD)

    def compile(self, fs_in: float) -> CompiledStep:
        clip, eps, k = self.clip, self.eps, self.IQR_TO_SD

        def apply(x):
            x = np.asarray(x, dtype=np.float64)
            v = x.reshape(-1)
            v = v[:: max(1, len(v) // 2_000_000)]
            v = v[v != 0]
            if len(v) < 100:
                v = x.reshape(-1)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            s = (q3 - q1) / k + eps
            return np.clip((x - med) / s, -clip, clip).astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)


def clean_pipelines() -> dict:
    """Pipelines for the clean channels (``physioex.data.cleaning``).

    The cleaner already band-limits and removes the mains, so only the robust
    per-night scale is applied. Keys = names of the clean channels.
    """
    from physioex.data.pipeline import PreprocessingPipeline

    p = PreprocessingPipeline([NanToZero(), RobustScale()])
    return {k: p for k in ("EEG_clean", "EOG_clean", "EMG_clean", "EMG_ENV_clean", "LEG_L_clean", "LEG_R_clean")}
