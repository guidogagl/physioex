"""Label-free night states for the cleaning: which epochs to fit each regression on.

The cleaning never reads the hypnogram (the clean channels feed label-free
models; stages are for validation only). The fits need epochs where a source is
quiet or active, and these come from the signals themselves, per 30 s epoch:

* eye index: log power 0.3-4 Hz of the EOG minus that of the EEG. Eye
  movements raise the EOG far above the EEG; slow brain activity raises both.
  ``quiet`` = epochs at or below the ``quiet_q`` quantile (eyes still: what is
  left in the EOG is brain), ``active`` = at or above ``active_q`` (eyes moving);
* delta index: log power 0.5-2 Hz of the EEG; ``deep`` = at or above
  ``deep_q`` (slow-wave epochs, where the brain-EOG coupling differs);
* sigma index: share of 11-16 Hz in the 0.3-35 Hz EEG power; ``sigma`` = at or
  above ``sigma_q`` (spindle-rich epochs);
* ``artefact``: EEG or EOG variance above ``artefact`` times the night median;
  excluded from every mask.

Masks are per sample at ``fs``; samples after the last full epoch are in none.
"""

from typing import Dict

import numpy as np

from physioex.data.cleaning.regression import fft_band


def _epoch_power(x: np.ndarray, lo: float, hi: float, fs: float, n_ep: int, ep: int):
    b = fft_band(x, lo, hi, fs)[: n_ep * ep].reshape(n_ep, ep)
    return (b**2).mean(1)


def state_masks(
    eeg: np.ndarray,
    eog: np.ndarray,
    fs: float,
    epoch_s: float = 30.0,
    quiet_q: float = 0.4,
    active_q: float = 0.8,
    deep_q: float = 0.75,
    sigma_q: float = 0.7,
    artefact: float = 20.0,
) -> Dict[str, np.ndarray]:
    ep = int(round(epoch_s * fs))
    n = min(len(eeg), len(eog))
    n_ep = n // ep
    out = {k: np.zeros(n, bool) for k in ("ok", "quiet", "active", "deep", "sigma")}
    if n_ep < 2:
        return out
    tiny = 1e-12
    v_eeg = _epoch_power(eeg, 0.3, 35.0, fs, n_ep, ep)
    v_eog = _epoch_power(eog, 0.3, 35.0, fs, n_ep, ep)
    ok = (v_eeg < artefact * np.median(v_eeg)) & (v_eog < artefact * np.median(v_eog))
    ok &= (v_eeg > tiny) & (v_eog > tiny)
    eye = np.log(_epoch_power(eog, 0.3, 4.0, fs, n_ep, ep) + tiny) - np.log(
        _epoch_power(eeg, 0.3, 4.0, fs, n_ep, ep) + tiny
    )
    delta = np.log(_epoch_power(eeg, 0.5, 2.0, fs, n_ep, ep) + tiny)
    sigma = _epoch_power(eeg, 11.0, 16.0, fs, n_ep, ep) / (v_eeg + tiny)
    ep_masks = {"ok": ok}
    if ok.any():
        q = lambda v, p: np.quantile(v[ok], p)  # noqa: E731
        ep_masks["quiet"] = ok & (eye <= q(eye, quiet_q))
        ep_masks["active"] = ok & (eye >= q(eye, active_q))
        ep_masks["deep"] = ok & (delta >= q(delta, deep_q))
        ep_masks["sigma"] = ok & (sigma >= q(sigma, sigma_q))
    for k, m in ep_masks.items():
        out[k][: n_ep * ep] = np.repeat(m, ep)
    return out
