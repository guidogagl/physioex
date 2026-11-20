"""IIR filter steps (Chebyshev lowpass)."""
import logging
from typing import Optional

import numpy as np
from scipy.signal import cheby2, sosfiltfilt
from physioex.data.pipeline import PreprocessingStep, CompiledStep

logger = logging.getLogger("physioex.data.steps")


class ChebyshevLowpass(PreprocessingStep):
    """Type-II Chebyshev IIR lowpass (zero-phase via ``sosfiltfilt``).

    Type-II (``cheby2``) attenuation is specified in the stop band rather
    than ripple in the pass band: gain is flat (within numerical error) up
    to ``cutoff_hz`` and at most ``-stop_db`` dB below
    ``cutoff_hz * (1 + transition)``. This is the filter used by the
    SleepPPG-Net / SPPPPG-Net reference code (8th order, 8 Hz, -40 dB).

    Uses second-order sections (SOS) for numerical stability on long
    signals at high sampling rates (see ``BandpassFilter``).

    If ``cutoff_hz`` is not below Nyquist the step is **skipped** with a
    warning (the signal is passed through unchanged).

    Parameters
    ----------
    cutoff_hz : float
        Lowpass cutoff in Hz (default 8.0).
    order : int
        Filter order (default 8).
    stop_db : float
        Minimum attenuation (dB) in the stop band (default 40.0).
    """

    def __init__(self, cutoff_hz: float = 8.0, order: int = 8, stop_db: float = 40.0):
        if cutoff_hz <= 0:
            raise ValueError(f"cutoff_hz must be > 0; got {cutoff_hz}")
        if order < 1:
            raise ValueError(f"order must be >= 1; got {order}")
        self.cutoff_hz = float(cutoff_hz)
        self.order = int(order)
        self.stop_db = float(stop_db)

    def spec(self) -> str:
        return self._build_spec(
            cutoff_hz=self.cutoff_hz, order=self.order, stop_db=self.stop_db
        )

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        nyq = 0.5 * fs_in
        if not 0 < self.cutoff_hz < nyq:
            logger.warning(
                f"ChebyshevLowpass(cutoff_hz={self.cutoff_hz}) skipped: "
                f"cutoff >= Nyquist ({nyq} Hz at fs={fs_in}). "
                f"Signal passed through."
            )
            return CompiledStep(
                apply=lambda x: np.asarray(x, dtype=np.float32),
                fs_out=fs_in,
            )
        sos = cheby2(
            self.order,
            self.stop_db,
            self.cutoff_hz / nyq,
            btype="low",
            output="sos",
        )

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            y = sosfiltfilt(sos, x, axis=-1)
            return y.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_in)
