import numpy as np


def generate_frequency_bands(nyquist, num_bands, start_freq=0.1, mode="log"):
    assert mode in {"linear", "log"}, "Mode must be either 'linear' or 'log'"
    if mode == "linear":
        bands = [
            [i * (nyquist - 1) / num_bands, (i + 1) * (nyquist - 1) / num_bands]
            for i in range(num_bands)
        ]

        bands[0][0] = start_freq

    elif mode == "log":
        log_freqs = np.logspace(
            np.log10(start_freq), np.log10(nyquist - 1), num_bands + 1
        )
        bands = [[log_freqs[i], log_freqs[i + 1]] for i in range(num_bands)]
    return bands
