"""Night-level channel cleaning: brain-only EEG, eye-only EOG, muscle-only EMG.

Unlike the per-channel :class:`~physioex.data.pipeline.PreprocessingStep`, these
functions see all the channels of a night at once, plus the hypnogram that
decides where each regression is fitted. They are pure numpy and know nothing
about datasets: the per-dataset recipe lives with the dataset classes.
"""

from physioex.data.cleaning.bss import SplitResult, sobi, split_ocular
from physioex.data.cleaning.montage import (
    CleanMontage,
    Fallback,
    ResolvedDerivation,
    ResolvedMontage,
    read_derivation,
    resolve_montage,
)
from physioex.data.cleaning.regression import (
    DEFAULT_BANDS,
    band_regression,
    epoch_mask,
    fft_band,
)

__all__ = [
    "CleanMontage",
    "DEFAULT_BANDS",
    "Fallback",
    "ResolvedDerivation",
    "ResolvedMontage",
    "read_derivation",
    "resolve_montage",
    "SplitResult",
    "band_regression",
    "epoch_mask",
    "fft_band",
    "sobi",
    "split_ocular",
]
