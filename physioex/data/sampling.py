"""Site-balanced sampling for ``MultiDataset`` training.

Every *site* gets the same total probability mass, and inside a site every
*subject* gets the same mass, regardless of how many subjects or epochs the
site/subject has::

    p(window w of subject u in site s)  ∝  1 / (N_s * W_u)

``N_s`` = number of training subjects of site ``s``; ``W_u`` = number of
training windows of subject ``u``.  The weights are normalised to sum to 1.
A *site* may span several wrapped datasets (e.g. the visits of one cohort):
pass the same site name for each of them in ``site_of_dataset``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def _window_owner(multi, train_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map flat MultiDataset indices to (dataset_idx, subject_ordinal)."""
    ds_idx = np.empty(len(train_indices), dtype=np.int64)
    subj = np.empty(len(train_indices), dtype=np.int64)
    for d, start, end in multi._offsets:
        m = (train_indices >= start) & (train_indices < end)
        if not m.any():
            continue
        local = train_indices[m] - start
        ranges = multi._datasets[d]._subject_ranges  # [(sid, s, e)]
        starts = np.asarray([r[1] for r in ranges], dtype=np.int64)
        pos = np.searchsorted(starts, local, side="right") - 1
        ds_idx[m] = d
        subj[m] = pos
    return ds_idx, subj


def site_balanced_weights(
    multi, train_indices: Sequence[int], site_of_dataset: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Return per-window weights (sum 1) and a per-site report."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(site_of_dataset) != len(multi._datasets):
        raise ValueError("site_of_dataset must have one entry per wrapped dataset")
    ds_idx, subj = _window_owner(multi, train_indices)
    sites = np.asarray([site_of_dataset[d] for d in ds_idx])

    # windows per (dataset, subject) unit
    unit = ds_idx * (subj.max() + 1) + subj
    uniq_units, inv, w_u = np.unique(unit, return_inverse=True, return_counts=True)
    unit_site = np.empty(len(uniq_units), dtype=object)
    unit_site[inv] = sites
    n_s: Dict[str, int] = defaultdict(int)
    for s in unit_site:
        n_s[s] += 1

    n_site_per_window = np.asarray([n_s[s] for s in sites], dtype=np.float64)
    weights = 1.0 / (n_site_per_window * w_u[inv].astype(np.float64))
    weights /= weights.sum()

    report = {}
    for s in sorted(n_s):
        m = sites == s
        report[s] = {
            "subjects": int(n_s[s]),
            "windows": int(m.sum()),
            "mass": float(weights[m].sum()),
        }
    return weights, report


def make_site_sampler(
    multi,
    train_indices: Sequence[int],
    site_of_dataset: Sequence[str],
    num_samples: int,
    seed: int = 0,
) -> Tuple[WeightedRandomSampler, Dict[str, dict]]:
    """WeightedRandomSampler over *positions in train_indices* (use with Subset)."""
    w, report = site_balanced_weights(multi, train_indices, site_of_dataset)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=g,
    )
    return sampler, report
