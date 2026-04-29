import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt
from physioex.data.pipeline import PreprocessingStep, CompiledStep

logger = logging.getLogger("physioex.data.steps")


class BandpassFilter(PreprocessingStep):
    """IIR Butterworth bandpass filter (zero-phase via sosfiltfilt).

    Uses second-order sections (SOS) representation for numerical stability.
    The ba-form (transfer function) is numerically unstable for long signals
    at high sampling rates with low normalized cutoffs (e.g., 0.3 Hz at
    500 Hz → normalized = 0.0012), causing NaN output. SOS avoids this.

    If the input sampling rate is too low for the filter band (high >= Nyquist),
    the filter is **skipped** with a warning.
    """

    def __init__(
        self,
        low: float = 0.3,
        high: float = 40.0,
        order: int = 5,
        ftype: str = "butter",
    ):
        if ftype != "butter":
            raise NotImplementedError(
                f"Only ftype='butter' is supported; got {ftype!r}"
            )
        self.low = float(low)
        self.high = float(high)
        self.order = int(order)
        self.ftype = ftype

    def spec(self) -> str:
        return self._build_spec(
            ftype=self.ftype, high=self.high, low=self.low, order=self.order
        )

    def compile(self, fs_in: float) -> CompiledStep:
        nyq = 0.5 * fs_in
        if not (0 < self.low < nyq and self.low < self.high < nyq):
            logger.warning(
                f"BandpassFilter(low={self.low}, high={self.high}) skipped: "
                f"band out of range for fs={fs_in} (Nyquist={nyq}). "
                f"Signal passed through."
            )
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32),
                fs_out=fs_in,
            )
        # Use SOS (second-order sections) for numerical stability.
        # ba-form filtfilt produces NaN on long signals (>1M samples) at
        # high fs with low normalized cutoffs (e.g., 0.3/250 = 0.0012).
        sos = butter(
            self.order, [self.low / nyq, self.high / nyq], btype="band", output="sos"
        )

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            y = sosfiltfilt(sos, x, axis=-1)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)
