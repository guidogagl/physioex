"""High-pass filter step for EMG preprocessing.

Matches Huy Phan's EMG preprocessing (FIR high-pass at 10 Hz) and the
AASM standard for chin EMG (10-100 Hz band).
"""
import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt
from physioex.data.pipeline import PreprocessingStep, CompiledStep

logger = logging.getLogger("physioex.data.steps")


class HighPassFilter(PreprocessingStep):
    """IIR Butterworth high-pass filter (zero-phase via filtfilt).

    Used for chin EMG in sleep staging: removes slow-wave crosstalk and
    movement artifacts below ``cutoff`` Hz while preserving tonic muscle
    activity above it.

    If the input sampling rate is too low for the cutoff (cutoff >= Nyquist),
    the filter is **skipped** with a warning instead of raising an error.
    This handles datasets like SleepEDF where EMG is sampled at 1 Hz.

    Parameters
    ----------
    cutoff : float
        -3 dB cutoff frequency in Hz (default 10.0, per AASM / Phan).
    order : int
        Filter order (default 5).
    """

    def __init__(self, cutoff: float = 10.0, order: int = 5):
        self.cutoff = float(cutoff)
        self.order = int(order)

    def spec(self) -> str:
        return self._build_spec(cutoff=self.cutoff, order=self.order)

    def compile(self, fs_in: float) -> CompiledStep:
        nyq = 0.5 * fs_in
        if not 0 < self.cutoff < nyq:
            logger.warning(
                f"HighPassFilter(cutoff={self.cutoff}) skipped: cutoff >= "
                f"Nyquist ({nyq} Hz at fs={fs_in}). Signal passed through."
            )
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32),
                fs_out=fs_in,
            )
        sos = butter(self.order, self.cutoff / nyq, btype="high", output="sos")

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            y = sosfiltfilt(sos, x, axis=-1)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)
