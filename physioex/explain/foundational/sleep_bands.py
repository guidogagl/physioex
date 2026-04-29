"""Sleep-specific EEG frequency band definitions.

Band boundaries follow AASM guidelines and standard sleep EEG literature.
Used by CSD and MultiChannelSpectralGradients for domain-aware frequency
ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrequencyBand:
    """A named frequency band with physiological significance."""

    name: str
    low: float  # Hz, inclusive
    high: float  # Hz, exclusive
    description: str


# AASM + sleep literature standard bands
SLEEP_BANDS: List[FrequencyBand] = [
    FrequencyBand(
        "slow_oscillation",
        0.3,
        0.5,
        "Slow oscillations, K-complex core, SO-spindle coupling",
    ),
    FrequencyBand("delta", 0.5, 4.0, "N3 marker, slow-wave activity"),
    FrequencyBand("theta", 4.0, 8.0, "N1/REM, hippocampal theta, drowsiness"),
    FrequencyBand("alpha", 8.0, 11.0, "Wake eyes-closed, alpha intrusion in N1"),
    FrequencyBand("sigma_low", 11.0, 13.0, "Slow spindles (frontal), N2 marker"),
    FrequencyBand(
        "sigma_high", 13.0, 16.0, "Fast spindles (central/parietal), N2 marker"
    ),
    FrequencyBand("beta_low", 16.0, 20.0, "Wake, light sleep microarousals"),
    FrequencyBand("beta_high", 20.0, 30.0, "Active wake, cortical desynchronization"),
    FrequencyBand("gamma", 30.0, 45.0, "Cognitive processing, EMG contamination"),
    FrequencyBand("mains", 48.0, 52.0, "50 Hz power line artifact (EU)"),
    FrequencyBand("high_freq", 52.0, 100.0, "High-frequency: EMG, movement artifacts"),
]

SLEEP_BAND_NAMES: List[str] = [b.name for b in SLEEP_BANDS]


def bands_to_bin_ranges(
    bands: List[FrequencyBand], fs: float, signal_length: int
) -> List[Tuple[str, int, int]]:
    """Convert frequency bands to DFT bin ranges.

    Returns:
        List of (band_name, bin_start, bin_end) tuples.
        Bins are for torch.fft.rfft output (0 to signal_length//2).
    """
    freq_res = fs / signal_length
    n_freqs = signal_length // 2 + 1
    result = []
    for band in bands:
        bin_start = max(0, round(band.low / freq_res))
        bin_end = min(n_freqs, round(band.high / freq_res))
        if bin_start < bin_end:
            result.append((band.name, bin_start, bin_end))
    return result


def band_center_frequencies(
    bands: List[FrequencyBand],
) -> Tensor:
    """Return center frequency of each band."""
    return torch.tensor([(b.low + b.high) / 2 for b in bands])


def band_names(bands: List[FrequencyBand] = None) -> List[str]:
    """Return band names."""
    if bands is None:
        bands = SLEEP_BANDS
    return [b.name for b in bands]
