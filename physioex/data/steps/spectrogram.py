import numpy as np
from scipy.signal import get_window, spectrogram
from physioex.data.pipeline import PreprocessingStep, CompiledStep


class XSleepNetSpectrogram(PreprocessingStep):
    """STFT spectrogram with power-dB scaling, matching Huy Phan's
    SeqSleepNet / XSleepNet / SleepTransformer preprocessing.

    Pipeline::

        signal -> STFT (Hamming, 2 s window, 50% overlap, 256-pt FFT)
                -> |X|^2       (power spectrum)
                -> 10*log10    (dB, power-scale)
                -> clip_db     (optional noise-floor removal, e.g. -25 dB)

    Output shape: ``(..., T, F)`` where T = time bins, F = nfft//2 + 1.
    ``fs_out`` is set to 0 because the output is in the time-frequency
    domain (not a resampled time series).

    Parameters
    ----------
    nperseg : int
        STFT window length in samples (default 200 = 2 s at 100 Hz).
    noverlap : int
        STFT overlap in samples (default 100 = 1 s, 50%).
    nfft : int
        FFT size (default 256).
    window : str
        Window function name (default ``"hamming"``).
    log_scale : bool
        If True (default), apply power-dB scaling: ``10 * log10(|X|^2 + eps)``.
    clip_db : float or None
        If not None, clip the dB spectrogram at this floor value.
        Recommended ``-25`` to suppress low-power noise introduced by
        filter stop-band leakage and zero-padding artifacts.
    """

    def __init__(
        self,
        nperseg: int = 200,
        noverlap: int = 100,
        nfft: int = 256,
        window: str = "hamming",
        log_scale: bool = True,
        clip_db: float = None,
    ):
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.nfft = int(nfft)
        self.window = window
        self.log_scale = bool(log_scale)
        self.clip_db = float(clip_db) if clip_db is not None else None

    def spec(self) -> str:
        return self._build_spec(
            clip_db=self.clip_db if self.clip_db is not None else "None",
            log_scale=self.log_scale,
            nfft=self.nfft,
            noverlap=self.noverlap,
            nperseg=self.nperseg,
            window=self.window,
        )

    def compile(self, fs_in: float) -> CompiledStep:
        win = get_window(self.window, self.nperseg)
        nperseg = self.nperseg
        noverlap = self.noverlap
        nfft = self.nfft
        log_scale = self.log_scale
        clip_db = self.clip_db
        eps = np.finfo(np.float32).eps

        def _stft_one(epoch_1d: np.ndarray) -> np.ndarray:
            """STFT of a single 1-D signal segment."""
            _, _, Sxx = spectrogram(
                epoch_1d,
                fs=fs_in,
                window=win,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
            )
            # Power spectrum: |X|^2
            out = Sxx  # scipy.signal.spectrogram already returns |X|^2 (PSD)
            if log_scale:
                # 10 * log10(power + eps)  ≡  20 * log10(|X| + eps)
                out = 10.0 * np.log10(out + eps)
                if clip_db is not None:
                    np.clip(out, a_min=clip_db, a_max=None, out=out)
            # Transpose to (T, F):  scipy returns (F, T)
            return np.transpose(out, (1, 0))

        def apply(signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=np.float64)
            if x.ndim == 1:
                return _stft_one(x).astype(np.float32)
            elif x.ndim == 2:
                return np.stack(
                    [_stft_one(epoch) for epoch in x],
                    axis=0,
                ).astype(np.float32)
            else:
                raise ValueError(f"Unsupported signal ndim={x.ndim}")

        return CompiledStep(apply=apply, fs_out=0)
