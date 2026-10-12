import numpy as np
import torch

from physioex.data.sampling import make_site_sampler, site_balanced_weights


class _DS:
    def __init__(self, subjects):  # subjects: list of window counts
        self._subject_ranges, run = [], 0
        for i, n in enumerate(subjects):
            self._subject_ranges.append((f"s{i}", run, run + n))
            run += n
        self.n = run


class _Multi:
    def __init__(self, dss):
        self._datasets = dss
        self._offsets, run = [], 0
        for i, d in enumerate(dss):
            self._offsets.append((i, run, run + d.n))
            run += d.n
        self.n = run


def _setup():
    # big site (3 subj x 1000 w), small site split over two datasets (visits), uneven subjects
    multi = _Multi([_DS([1000, 1000, 1000]), _DS([10, 30]), _DS([5])])
    return multi, ["big", "small", "small"], np.arange(multi.n)


def test_equal_site_mass():
    multi, sites, idx = _setup()
    w, rep = site_balanced_weights(multi, idx, sites)
    assert abs(w.sum() - 1) < 1e-9
    assert abs(rep["big"]["mass"] - 0.5) < 1e-9 and abs(rep["small"]["mass"] - 0.5) < 1e-9
    assert rep["small"]["subjects"] == 3 and rep["big"]["subjects"] == 3


def test_equal_subject_mass_within_site():
    multi, sites, idx = _setup()
    w, _ = site_balanced_weights(multi, idx, sites)
    big = [w[0:1000].sum(), w[1000:2000].sum(), w[2000:3000].sum()]
    small = [w[3000:3010].sum(), w[3010:3040].sum(), w[3040:3045].sum()]
    assert np.allclose(big, 0.5 / 3) and np.allclose(small, 0.5 / 3)


def test_subset_of_train_indices_and_sampler_frequencies():
    multi, sites, _ = _setup()
    idx = np.concatenate([np.arange(0, 2000), np.arange(3000, 3045)])  # one big subject held out
    sampler, rep = make_site_sampler(multi, idx, sites, num_samples=200_000, seed=1)
    assert rep["big"]["subjects"] == 2 and abs(rep["big"]["mass"] - 0.5) < 1e-9
    draws = np.fromiter(iter(sampler), dtype=np.int64)
    in_big = (idx[draws] < 3000).mean()
    assert abs(in_big - 0.5) < 0.01
