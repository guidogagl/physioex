"""Unit tests for physioex.explain.foundational.sleep_bands."""
import torch

from physioex.explain.foundational.sleep_bands import (
    FrequencyBand,
    SLEEP_BANDS,
    SLEEP_BAND_NAMES,
    bands_to_bin_ranges,
    band_center_frequencies,
    band_names,
)


def test_frequency_band_is_frozen():
    b = FrequencyBand("delta", 0.5, 4.0, "desc")
    assert b.name == "delta" and b.low == 0.5 and b.high == 4.0
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        b.low = 1.0  # frozen dataclass


def test_sleep_bands_catalogue_consistency():
    assert len(SLEEP_BANDS) == 11
    assert SLEEP_BAND_NAMES == [b.name for b in SLEEP_BANDS]
    # bands are ordered and non-overlapping (low < high, ascending)
    for b in SLEEP_BANDS:
        assert b.low < b.high
    lows = [b.low for b in SLEEP_BANDS]
    assert lows == sorted(lows)


def test_bands_to_bin_ranges():
    fs, n = 100.0, 3000  # freq_res = 100/3000 ≈ 0.0333 Hz
    ranges = bands_to_bin_ranges(SLEEP_BANDS, fs=fs, signal_length=n)
    assert ranges  # non-empty
    freq_res = fs / n
    n_freqs = n // 2 + 1
    for name, start, end in ranges:
        assert 0 <= start < end <= n_freqs
    # delta 0.5-4.0 Hz -> bins round(0.5/fr)..round(4.0/fr)
    delta = [r for r in ranges if r[0] == "delta"][0]
    assert delta[1] == round(0.5 / freq_res)
    assert delta[2] == round(4.0 / freq_res)


def test_bands_to_bin_ranges_drops_empty_bands():
    # With a tiny signal, high-frequency narrow bands collapse (start==end) and
    # must be excluded.
    ranges = bands_to_bin_ranges(SLEEP_BANDS, fs=100.0, signal_length=64)
    for _name, start, end in ranges:
        assert start < end


def test_band_center_frequencies():
    centers = band_center_frequencies(SLEEP_BANDS)
    assert centers.shape == (len(SLEEP_BANDS),)
    expected0 = (SLEEP_BANDS[0].low + SLEEP_BANDS[0].high) / 2
    assert torch.isclose(centers[0], torch.tensor(expected0))


def test_band_names_default_and_explicit():
    assert band_names() == SLEEP_BAND_NAMES
    subset = SLEEP_BANDS[:3]
    assert band_names(subset) == [b.name for b in subset]
