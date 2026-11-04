"""Merged training data over several sleep datasets, with the ProtoSleepNet splits.

The pretraining study (04.11.2026) trains the clean-channel encoders on combinations of
datasets ("items") and tests every model on the test split of all of them, plus clinical
cohorts and MASS SS01 that never enter training. This module holds:

* :data:`UNITS` – one entry per physioex dataset instance (cache unit): its
  :data:`~physioex.models.foundation_datasets.DATASET_CONFIGS` key, the site key of the
  ProtoSleepNet splits JSON (``outputs/psn-multi/splits_29sites.json``), the pool item it
  belongs to (``None`` for test-only units) and the subgroup reported at test time;
* :data:`POOL` – the ten combinable items;
* :func:`build_train` – MultiDataset of the units of the chosen items, flat train indices
  (train split only), validation windows balanced per item, ``site_of_dataset`` for
  :func:`physioex.data.sampling.make_site_sampler` (equal mass per item);
* :func:`build_test` – MultiDataset of every unit and the list of test nights: test split of
  the pool units, all nights of the test-only units;
* :class:`ResampledView` – resamples one modality to a fixed rate when the item is loaded
  (EMG: 256/125/100 Hz depending on the dataset), so the cache is never recomputed.

Only nights whose channel is already in the cache are used: physioex computes a missing
cache entry on first access (cleaner run plus night cache on disk), which a training job
must never trigger. :func:`cached_subjects` checks the files without reading them.
"""
from __future__ import annotations

import importlib
import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

#: name, DATASET_CONFIGS key, ctor kwargs, splits site key, pool item (None = test only), subgroup
UNITS: List[dict] = [
    {"name": "shhs_visit1", "config": "shhs_v1", "site": "shhs", "item": "shhs"},
    {"name": "shhs_visit2", "config": "shhs_v2", "site": "shhs", "item": "shhs"},
    {"name": "mesa", "config": "mesa", "site": "mesa", "item": "mesa"},
    {"name": "mros", "config": "mros", "site": "mros", "item": "mros"},
    {"name": "wsc_visit1", "config": "wsc", "kwargs": {"visit": 1}, "site": "wsc", "item": "wsc"},
    {"name": "wsc_visit2", "config": "wsc_v2", "kwargs": {"visit": 2}, "site": "wsc", "item": "wsc"},
    {"name": "wsc_visit3", "config": "wsc_v3", "kwargs": {"visit": 3}, "site": "wsc", "item": "wsc"},
    {"name": "wsc_visit4", "config": "wsc_v4", "kwargs": {"visit": 4}, "site": "wsc", "item": "wsc"},
    # wsc_visit5 (2 nights in the splits) has no clean cache: left out
    {"name": "hpap_lab_full", "config": "hpap_lab_full", "site": "hpap", "item": "hpap"},
    {"name": "hpap_lab_split", "config": "hpap_lab_split", "site": "hpap", "item": "hpap"},
    {"name": "ccshs", "config": "ccshs", "site": "ccshs", "item": "ccshs"},
    {"name": "dcsm", "config": "dcsm", "site": "dcsm", "item": "dcsm"},
    {"name": "hmc", "config": "hmc", "site": "hmc", "item": "hmc"},
    {"name": "sleepedf", "config": "sleepedf", "site": "sleepedf", "item": "sleepedf"},
    {"name": "mass_ss02", "config": "mass_ss02", "site": "mass_ss02", "item": "mass"},
    {"name": "mass_ss03", "config": "mass_ss03", "site": "mass_ss03", "item": "mass"},
    {"name": "mass_ss04", "config": "mass_ss04", "site": "mass_ss04", "item": "mass"},
    {"name": "mass_ss05", "config": "mass_ss05", "site": "mass_ss05", "item": "mass"},
    # test only, every night, reported per subgroup
    {"name": "mass_ss01", "config": "mass_ss01", "site": "mass_ss01", "item": None, "subgroup": "mass_ss01"},
    {"name": "alzheimers_ad", "config": "alzheimers_ad", "site": "alzheimers_ad", "item": None, "subgroup": "alzheimers_ad"},
    {"name": "alzheimers_hc", "config": "alzheimers_hc", "site": "alzheimers_hc", "item": None, "subgroup": "alzheimers_hc"},
    {"name": "parkinsons_night_pd", "config": "parkinsons_night_pd", "site": "parkinsons_night_pd", "item": None,
     "subgroup": "parkinsons_night_pd"},
    {"name": "parkinsons_night_hoa", "config": "parkinsons_night_hoa", "site": "parkinsons_night_hoa", "item": None,
     "subgroup": "parkinsons_night_hoa"},
    {"name": "parkinsons_nap_pd", "config": "parkinsons_nap_pd", "site": "parkinsons_nap_pd", "item": None,
     "subgroup": "parkinsons_nap_pd"},
    {"name": "parkinsons_nap_hoa", "config": "parkinsons_nap_hoa", "site": "parkinsons_nap_hoa", "item": None,
     "subgroup": "parkinsons_nap_hoa"},
    {"name": "lsec_control", "config": "lsec", "kwargs": {"subset": "Control"}, "site": "lsec", "item": None,
     "subgroup": "lsec_control"},
    {"name": "lsec_wce", "config": "lsec", "kwargs": {"subset": "WCE"}, "site": "lsec", "item": None,
     "subgroup": "lsec_wce"},
    {"name": "lsec_dre", "config": "lsec", "kwargs": {"subset": "DRE"}, "site": "lsec", "item": None,
     "subgroup": "lsec_dre"},
]
for _u in UNITS:  # the subgroup of a pool unit is its splits site (mass_ss02 ... ss05 kept apart in the report)
    _u.setdefault("kwargs", {})
    _u.setdefault("subgroup", _u["site"])

POOL: Tuple[str, ...] = ("shhs", "mros", "mesa", "wsc", "ccshs", "hpap", "dcsm", "hmc", "sleepedf", "mass")
BY_NAME = {u["name"]: u for u in UNITS}


def units_of(items: Sequence[str]) -> List[str]:
    """Unit names of the pool items, in :data:`UNITS` order."""
    bad = set(items) - set(POOL)
    if bad:
        raise ValueError(f"not pool items: {sorted(bad)} (pool: {POOL})")
    return [u["name"] for u in UNITS if u["item"] in set(items)]


def test_only_units() -> List[str]:
    return [u["name"] for u in UNITS if u["item"] is None]


def lsec_audit_kwargs(config: str) -> dict:
    """LSEC labels shifted by the alignment audit and the ambiguous recordings left out
    (``label_offsets.csv`` + ``exclude_audited.txt`` in ``$PHYSIOEX_LSEC_AUDIT``, from the
    ProtoSleepNet audit of 19.10.2026: 49 recordings are off by up to 39 epochs). Without the
    variable the index labels are used, with a warning: the LSEC kappas are then not valid."""
    if config != "lsec":
        return {}
    d = os.environ.get("PHYSIOEX_LSEC_AUDIT")
    if not d:
        warnings.warn("PHYSIOEX_LSEC_AUDIT not set: LSEC with index labels (not audited)")
        return {}
    excl = Path(d, "exclude_audited.txt").read_text().replace("\n", ",").split(",")
    return {"label_offset": "audited", "label_offsets_path": str(Path(d, "label_offsets.csv")),
            "exclude_subjects": [x.strip() for x in excl if x.strip()]}


def make_dataset(name: str, channels: Sequence[str], pipelines, sequence_length: int, cache_dir: Optional[str],
                 memmap_cache_size: int = 200, load_events: bool = False, **extra):
    """One physioex dataset of :data:`UNITS` (class and defaults from ``DATASET_CONFIGS``).

    ``load_events`` is False by default: items then carry empty ``events`` and no raw
    annotation file is opened. With many workers on GPFS that parse (one per subject and
    worker) saturated the client and broke the 04.11 smoke runs.
    """
    from physioex.models.foundation_datasets import DATASET_CONFIGS

    u = BY_NAME[name]
    dc = DATASET_CONFIGS[u["config"]]
    cls = getattr(importlib.import_module(dc.module_path), dc.class_name)
    kw = {**dc.extra_kwargs, **u["kwargs"], **lsec_audit_kwargs(u["config"]), **extra}
    ds = cls(channels=list(channels), pipelines=pipelines, sequence_length=sequence_length, cache_dir=cache_dir,
             memmap_cache_size=memmap_cache_size, **kw)
    ds.load_events = bool(load_events)
    return ds


def cached_subjects(ds) -> set:
    """Subjects of ``ds`` whose requested channels are all in the cache (files checked, not read)."""
    if not getattr(ds, "cache_enabled", False):
        return set(ds.get_subjects())
    ok = set()
    for sid in ds.get_subjects():
        rcs = ds._resolved.get(sid) or []
        good = bool(rcs)
        for rc in rcs:
            if rc is None or getattr(rc, "physical", None) is None:
                good = False
                break
            p = ds.cache.signal_path(ds.DATASET_NAME, sid, rc.physical, ds._pipeline_for_channel(rc).hash())
            if not p.exists():
                good = False
                break
        if good:
            ok.add(sid)
    return ok


def _split_ids(splits: dict, site: str, unit: str, key: str) -> List[str]:
    return [sid for u, sid in splits.get(site, {}).get(key, []) if u == unit]


@dataclass
class MergedData:
    multi: object
    unit_names: List[str]
    #: one entry per unit: the item it belongs to (``None`` for test-only units)
    item_of_dataset: List[Optional[str]]
    train_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    valid_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    #: (ds_idx, subject_id, item or None, subgroup)
    test: List[tuple] = field(default_factory=list)
    report: dict = field(default_factory=dict)


def _flat(multi, d: int, ids: List[str]) -> np.ndarray:
    off = {i: s for i, s, _ in multi._offsets}[d]
    if not ids:
        return np.zeros(0, np.int64)
    return np.asarray(multi._datasets[d]._subject_ids_to_flat_indices(ids), dtype=np.int64) + off


def _build(names, channels, pipelines, sequence_length, cache_dir, memmap_cache_size, ds_kwargs):
    from physioex.data.multi import MultiDataset

    dss = [make_dataset(n, channels, pipelines, sequence_length, cache_dir, memmap_cache_size, **ds_kwargs)
           for n in names]
    return MultiDataset(dss, memmap_cache_size=memmap_cache_size), dss


def build_train(items: Sequence[str], splits_path: str, channels: Sequence[str], pipelines, sequence_length: int,
                cache_dir: Optional[str], valid_per_item: int = 5000, seed: int = 0, memmap_cache_size: int = 0,
                allow_empty: bool = False, **ds_kwargs) -> MergedData:
    """Merged training data of the pool ``items``: train split for training, validation windows
    of the val split (``valid_per_item`` per item, drawn without replacement, fixed seed).

    ``memmap_cache_size`` is the total of open memmaps per process (main or DataLoader worker),
    shared among the units by their number of nights. Default 0: no LRU, the night's file is
    mapped for the item and released after it. Every open memmap holds a GPFS file open, and the
    client of a Sofia node stalls as a whole (every process, even ``ls``) once ~800 cache files
    are open on the node (measured 04.11.2026, acc019: one process with no cap hung after ~830
    open memmaps; capped at 64 it ran 60 batches without a stall). The count adds up over
    processes and DataLoader workers (train and valid loaders are both persistent). A reopen
    costs a few ms on a healthy client, and with random sampling over thousands of nights the
    LRU hardly hits anyway.

    ``allow_empty``: an item without cached nights for the channel (Sleep-EDF has no clean EMG)
    is reported in ``report["empty_items"]`` instead of raising; it then gets no sampling mass."""
    splits = json.loads(Path(splits_path).read_text())
    names = units_of(items)
    multi, dss = _build(names, channels, pipelines, sequence_length, cache_dir, memmap_cache_size, ds_kwargs)
    rng = np.random.default_rng(seed)
    train, valid_by_item, rep = [], defaultdict(list), {}
    for d, (n, ds) in enumerate(zip(names, dss)):
        u = BY_NAME[n]
        have = cached_subjects(ds)
        tr = [s for s in _split_ids(splits, u["site"], n, "train") if s in have]
        va = [s for s in _split_ids(splits, u["site"], n, "val") if s in have]
        n_tr_all = len(_split_ids(splits, u["site"], n, "train"))
        train.append(_flat(multi, d, tr))
        valid_by_item[u["item"]].append(_flat(multi, d, va))
        rep[n] = {"item": u["item"], "train_nights": len(tr), "train_nights_in_splits": n_tr_all,
                  "val_nights": len(va), "train_windows": int(len(train[-1])), "train_ids": tr, "val_ids": va}
    valid = []
    for it in items:
        v = np.concatenate(valid_by_item[it]) if valid_by_item[it] else np.zeros(0, np.int64)
        if len(v) > valid_per_item:
            v = rng.choice(v, valid_per_item, replace=False)
        valid.append(np.sort(v))
    md = MergedData(multi, names, [BY_NAME[n]["item"] for n in names],
                    train_idx=np.sort(np.concatenate(train)), valid_idx=np.sort(np.concatenate(valid)), report=rep)
    empty = [it for it in items if not any(rep[n]["train_nights"] for n in names if BY_NAME[n]["item"] == it)]
    md.report["empty_items"] = empty
    if empty and not allow_empty:
        raise ValueError(f"no cached training nights for items {empty}")
    if len(empty) == len(items):
        raise ValueError(f"no cached training nights for any of {list(items)}")
    return md


def build_test(splits_path: str, channels: Sequence[str], pipelines, sequence_length: int, cache_dir: Optional[str],
               units: Optional[Sequence[str]] = None, memmap_cache_size: int = 50, **ds_kwargs) -> MergedData:
    """Every test night: test split of the pool units, all cached nights of the test-only units."""
    splits = json.loads(Path(splits_path).read_text())
    names = list(units) if units is not None else [u["name"] for u in UNITS]
    multi, dss = _build(names, channels, pipelines, sequence_length, cache_dir, memmap_cache_size, ds_kwargs)
    test, rep = [], {}
    for d, (n, ds) in enumerate(zip(names, dss)):
        u = BY_NAME[n]
        have = cached_subjects(ds)
        ids = sorted(have) if u["item"] is None else [s for s in _split_ids(splits, u["site"], n, "test") if s in have]
        test.extend((d, s, u["item"], u["subgroup"]) for s in ids)
        rep[n] = {"item": u["item"], "subgroup": u["subgroup"], "test_nights": len(ids),
                  "not_cached": len(set(ds.get_subjects()) - have)}
    return MergedData(multi, names, [BY_NAME[n]["item"] for n in names], test=test, report=rep)


def _resample_poly(x: np.ndarray, fs_in: float, fs_out: float, n_out: int) -> np.ndarray:
    from scipy.signal import resample_poly

    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    y = resample_poly(x, fr.numerator, fr.denominator, axis=-1)
    if y.shape[-1] >= n_out:
        return y[..., :n_out]
    return np.pad(y, [(0, 0)] * (y.ndim - 1) + [(0, n_out - y.shape[-1])])


class ResampledView(Dataset):
    """Items of ``base`` with the channels of ``modality`` (e.g. ``"EMG"``) resampled to ``fs``.

    The channel arrives as ``(L, fs_in * epoch_s)`` at the cached rate of its dataset; the L epochs
    are joined, resampled once (polyphase, ``scipy.signal.resample_poly``) and cut back into L
    epochs of ``fs * epoch_s`` samples, so every dataset yields the same shape. Rates within 1e-3
    of ``fs`` pass through unchanged.
    """

    def __init__(self, base, modality: str, fs: float, epoch_s: float = 30.0):
        self.base, self.modality, self.fs, self.epoch_s = base, modality, float(fs), float(epoch_s)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        info = item.get("channel_info", {})
        for k, x in list(item["signals"].items()):
            if not k.startswith(self.modality + "_"):
                continue
            arr = x.numpy() if torch.is_tensor(x) else np.asarray(x)
            L = arr.shape[0]
            fs_in = float((info.get(k) or {}).get("fs_out") or arr.shape[-1] / self.epoch_s)
            n_ep = int(round(self.fs * self.epoch_s))
            if abs(fs_in - self.fs) < 1e-3 and arr.shape[-1] == n_ep:
                continue
            y = _resample_poly(arr.reshape(-1).astype(np.float64), fs_in, self.fs, L * n_ep).reshape(L, n_ep)
            item["signals"][k] = torch.from_numpy(y.astype(np.float32))
            if k in info:
                info[k] = {**info[k], "fs_out": self.fs}
        return item

    def __getattr__(self, name):  # MultiDataset helpers (_offsets, _datasets, ...) pass through
        if name in ("base", "modality", "fs", "epoch_s"):
            raise AttributeError(name)
        return getattr(self.base, name)
