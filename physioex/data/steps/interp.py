"""Grid-interpolation resampling and fixed-length truncation/padding.

The ``InterpResample`` step replicates the wav2sleep reference
preprocessing (Carter & Tarassenko 2024): the signal is linearly
interpolated onto a fixed samples-per-hour grid, NaN gaps are filled
from the surrounding valid samples, and samples outside the valid
recording span are zero.
"""
import logging
from typing import Optional

import numpy as np
from physioex.data.pipeline import PreprocessingStep, CompiledStep

logger = logging.getLogger("physioex.data.steps")


class InterpResample(PreprocessingStep):
    """Linearly interpolate a signal onto a fixed time grid.

    Output sample ``k`` (``k = 0 .. n_out - 1``) is the input value at
    time ``t_k = (k + 1) * epoch_sec / samples_per_epoch`` (seconds),
    where input sample ``i`` sits at time ``i / fs``. The last output
    sample therefore lands exactly at the end of the covered window.

    NaN input samples are dropped before interpolating; interior gaps
    are filled by linear interpolation between the surrounding valid
    samples (equivalent to pandas
    ``Series.interpolate(method="index", limit_area="inside")``).
    Output times before the first or beyond the last valid input sample
    map to ``0.0`` (no extrapolation).

    Output length is ``n_out = floor(len(x) / fs * samples_per_epoch /
    epoch_sec)`` and ``fs_out = samples_per_epoch / epoch_sec``. The
    output is float32.

    Parameters
    ----------
    samples_per_epoch : int
        Number of output samples per ``epoch_sec`` window (e.g. 1024 for
        a 30 s epoch at the wav2sleep grid).
    epoch_sec : float
        Window length in seconds (default 30.0).
    """

    def __init__(self, samples_per_epoch: int, epoch_sec: float = 30.0):
        if samples_per_epoch < 1:
            raise ValueError(f"samples_per_epoch must be >= 1; got {samples_per_epoch}")
        if epoch_sec <= 0:
            raise ValueError(f"epoch_sec must be > 0; got {epoch_sec}")
        self.samples_per_epoch = int(samples_per_epoch)
        self.epoch_sec = float(epoch_sec)

    def spec(self) -> str:
        return self._build_spec(
            epoch_sec=self.epoch_sec, samples_per_epoch=self.samples_per_epoch
        )

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        spe = self.samples_per_epoch
        epoch_sec = self.epoch_sec
        fs_out = spe / epoch_sec

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            n = x.shape[-1]
            n_out = int(np.floor(n / fs_in * fs_out))
            if n_out <= 0:
                return np.zeros(0, dtype=np.float32)
            t_out = (np.arange(n_out) + 1) / fs_out
            # Drop NaN samples; interior gaps are then filled by linear
            # interpolation between the surrounding valid samples.
            valid = np.isfinite(x)
            if not valid.any():
                return np.zeros(n_out, dtype=np.float32)
            tv = np.arange(n, dtype=np.float64) / fs_in
            xv = x[valid]
            tv = tv[valid]
            # Times outside the valid span map to 0.0 (no extrapolation).
            mask = (t_out >= tv[0]) & (t_out <= tv[-1])
            out = np.zeros(n_out, dtype=np.float64)
            if mask.any():
                out[mask] = np.interp(t_out[mask], tv, xv)
            return out.astype(np.float32)

        return CompiledStep(apply=apply, fs_out=fs_out)


class FixedLength(PreprocessingStep):
    """Pad the END with ``fill`` or truncate the END to a fixed length.

    ``len(y) == round(seconds * fs_in)``. The output length depends on
    the input sampling rate only, so the sampling rate is unchanged.

    Parameters
    ----------
    seconds : float
        Target signal length in seconds (e.g. 36000 = 10 h).
    fill : float
        Value used when padding (default 0.0).
    """

    def __init__(self, seconds: float, fill: float = 0.0):
        if seconds <= 0:
            raise ValueError(f"seconds must be > 0; got {seconds}")
        self.seconds = float(seconds)
        self.fill = float(fill)

    def spec(self) -> str:
        return self._build_spec(fill=self.fill, seconds=self.seconds)

    def compile(self, fs_in: float, header: Optional[dict] = None) -> CompiledStep:
        n_target = int(round(self.seconds * fs_in))
        fill = self.fill

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float32)
            n = x.shape[-1]
            if n == n_target:
                return x
            if n < n_target:
                pad = np.full(n_target - n, fill, dtype=np.float32)
                return np.concatenate([x, pad])
            return x[..., :n_target]

        return CompiledStep(apply=apply, fs_out=fs_in)
