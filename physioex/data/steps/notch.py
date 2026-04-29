import numpy as np
from scipy.signal import iirnotch, sosfilt, sosfiltfilt
from physioex.data.pipeline import PreprocessingStep, CompiledStep


class NotchFilter(PreprocessingStep):
    """IIR notch filter (powerline removal).

    Uses second-order sections (SOS) for numerical stability on long signals.
    """

    def __init__(self, freq: float = 50.0, quality: float = 30.0):
        self.freq = float(freq)
        self.quality = float(quality)

    def spec(self) -> str:
        return self._build_spec(freq=self.freq, quality=self.quality)

    def compile(self, fs_in: float) -> CompiledStep:
        nyq = 0.5 * fs_in
        if not 0 < self.freq < nyq:
            raise ValueError(f"Notch frequency {self.freq} out of range (0, {nyq})")
        # iirnotch returns ba-form; convert to SOS for stability
        b, a = iirnotch(self.freq, self.quality, fs_in)
        # For a 2nd-order notch, ba is already short (3 coefficients),
        # but we wrap it in SOS format for consistency with other filters.
        # np.array([[b0, b1, b2, 1, a1, a2]]) is the SOS form.
        sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            y = sosfiltfilt(sos, x, axis=-1)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)
