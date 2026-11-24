"""Cache-only reader of the clean channels, for ChannelNet (physioex/models/seqsleepnet_clean.py).

Reads the cache layout directly, so it works where the raw EDF files are absent (vaglini mini-cache) and on the full
cache on Sofia:

    <root>/<dataset>/signals/<subject>/<CHAN>_clean_<hash>/<pipeline>/signal.npy (+ signal.meta.json)
    <root>/<dataset>/labels/<subject>.npy

Per night:
- signal and labels both start at the first epoch and may differ in length at the end (labels beyond the
  recording, or unscored recording tail): both are truncated to the shorter, as physioex does with label -1;
- the cleaner zeroes everything outside the recording window, which RobustScale turns into a constant: flat epochs
  (std < 1e-3 over the epoch or over any 1-s segment) at the start and end of the night are trimmed away, inner
  flat or partly flat epochs and unscored epochs stay in the sequence context but are masked in the loss (label -1);
- EMG is resampled to 100 Hz once for the whole night (continuous signal) and carries the band correction
  offset_db = 10 log10(B / 40), B = min(100, fs/2) - 10, applied to the band power by the model.

Splits are per subject: either given explicitly ({dataset: {train, valid, test}}, e.g. physioex fold 0 written by
``BasePhysioDataset.get_splits``), or from a hash of "<dataset>/<subject>" (70/15/15). Either way the EEG, EOG and
EMG networks see the same subjects in the same split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

EPOCH = 3000          # samples per 30-s epoch at 100 Hz
FLAT_STD = 1e-3


def split_of(dataset: str, subject: str) -> str:
    u = int(hashlib.md5(f"{dataset}/{subject}".encode()).hexdigest(), 16) % 100
    return "train" if u < 70 else ("valid" if u < 85 else "test")


def flat_epochs(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """(fully flat, partly flat) masks per epoch: std over the epoch, and over each 1-s segment."""
    full = x.std(1) < FLAT_STD
    seg = int(round(fs))
    n = (x.shape[1] // seg) * seg
    part = (x[:, :n].reshape(len(x), -1, seg).std(2) < FLAT_STD).any(1)
    return full, part & ~full


def to_100hz(x: np.ndarray, fs: float) -> np.ndarray:
    """Whole night (n, T) at fs -> (n, 3000) at 100 Hz, resampled as one continuous signal."""
    if fs == 100.0:
        return x.astype(np.float32)
    fr = Fraction(100.0 / fs).limit_denominator(4096)
    y = resample_poly(x.astype(np.float32).reshape(-1), fr.numerator, fr.denominator)
    n = len(x)
    y = np.pad(y, (0, max(0, n * EPOCH - len(y))))[: n * EPOCH]
    return y.reshape(n, EPOCH).astype(np.float32)


class LazyArray:
    """Read-only (n, E) array kept on local disk and memory-mapped on access, with a small per-process LRU of open
    maps (``local_dir`` option of the readers): RAM stays free and only a few files are open at a time (a few
    hundred open memory maps on GPFS stall the client; SHHS has ~12k signal files)."""

    def __init__(self, path: str, shape: Tuple[int, ...], dtype: str):
        self.path, self.shape, self.dtype = str(path), tuple(shape), np.dtype(dtype)

    def _arr(self) -> np.ndarray:
        return _open_map(self.path)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self._arr()[idx]

    def reshape(self, *shape):
        return self._arr().reshape(*shape)

    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self._arr())
        return a if dtype is None else a.astype(dtype)


@lru_cache(maxsize=64)
def _open_map(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _spill(x: np.ndarray, path: Path) -> LazyArray:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(x))
    return LazyArray(path, x.shape, str(x.dtype))


def _load_spilled(args):
    """Worker of the parallel readers: load one night, write its signals to local disk, return it with LazyArrays."""
    kind, root, ds, subj, mods, pipe, local = args
    try:
        if kind == "single":
            night, why = load_night(Path(root), ds, subj, mods, pipe=pipe)
        else:
            night, why = load_multimodal_night(Path(root), ds, subj, mods, pipe=pipe)
    except Exception as e:                       # unreadable night: counted, never fatal
        return ds, subj, None, f"error {type(e).__name__}"
    if night is None:
        return ds, subj, None, why
    base = Path(local) / ds / subj
    if kind == "single":
        night.x = _spill(night.x, base / f"{mods}.npy")
    else:
        night.x = {m: _spill(v, base / f"{m}.npy") for m, v in night.x.items()}
    return ds, subj, night, why


def _read_all(kind, root, datasets, mods, split, local_dir, workers, pipe=None):
    """[(split name, night)] and skip counts. With local_dir the nights are read in parallel and spilled to disk."""
    jobs, where_of, skipped = [], {}, {}
    for ds in datasets:
        where = None if split is None else {s: k for k, ids in split[ds].items() for s in ids}
        for subj in sorted(p.name for p in (Path(root) / ds / "signals").iterdir()):
            if where is not None and subj not in where:
                skipped["not in split"] = skipped.get("not in split", 0) + 1
                continue
            where_of[(ds, subj)] = where[subj] if where is not None else split_of(ds, subj)
            jobs.append((kind, str(root), ds, subj, mods, pipe, None if local_dir is None else str(local_dir)))
    if local_dir is None:
        results = []
        for j in jobs:
            if kind == "single":
                night, why = load_night(Path(root), j[2], j[3], mods, pipe=pipe)
            else:
                night, why = load_multimodal_night(Path(root), j[2], j[3], mods, pipe=pipe)
            results.append((j[2], j[3], night, why))
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(max(1, workers)) as pool:
            results = pool.map(_load_spilled, jobs, chunksize=4)
    out = []
    for ds, subj, night, why in results:
        if night is None:
            skipped[why] = skipped.get(why, 0) + 1
        else:
            out.append((where_of[(ds, subj)], night))
    return out, skipped


@dataclass
class Night:
    dataset: str
    subject: str
    x: np.ndarray            # (n, 3000) float16/float32, 100 Hz
    y: np.ndarray            # (n,) int64, -1 = masked
    offset_db: float
    fs: float


def load_night(root: Path, dataset: str, subject: str, modality: str,
               pipe: Optional[str] = None) -> Tuple[Optional[Night], str]:
    # <hash>/ holds the epoched pipeline output; night/ (full cache) is the continuous cleaned night: not used here
    sig = [p for p in (root / dataset / "signals" / subject).glob(f"{modality}_clean_*/*/signal.npy")
           if p.parent.name != "night"]
    if len(sig) > 1 and pipe is not None:          # several epoching pipelines: keep the requested one
        sig = [p for p in sig if p.parent.name == pipe]
    lab = root / dataset / "labels" / f"{subject}.npy"
    if len(sig) != 1 or not lab.exists():
        return None, "no signal or labels"
    fs = float(json.loads((sig[0].parent / "signal.meta.json").read_text())["fs_out"])
    x = np.load(sig[0], mmap_mode="r")
    y = np.load(lab).astype(np.int64)
    n = min(len(x), len(y))
    x, y = np.asarray(x[:n]), y[:n]
    full, part = flat_epochs(x.astype(np.float32), fs)
    keep = np.where(~full)[0]
    if len(keep) == 0:
        return None, "all flat"
    a, b = keep[0], keep[-1] + 1                      # trim the zeroed lights-off/on margins
    x, y, full, part = x[a:b], y[a:b].copy(), full[a:b], part[a:b]
    y[(y < 0) | (y > 4) | full | part] = -1
    if (y >= 0).sum() == 0:
        return None, "no scored epoch"
    offset = 0.0
    if modality == "EMG":
        x = to_100hz(x, fs).astype(np.float16)
        offset = float(10 * np.log10((min(100.0, fs / 2) - 10.0) / 40.0))
    elif fs != 100.0:
        return None, f"fs {fs}"
    return Night(dataset, subject, x, y, offset, fs), "ok"


class CleanCacheNights:
    """All nights of some datasets for one modality, grouped by split. ``local_dir``: read the nights in parallel
    (``workers`` processes) and keep their signals on that local disk, memory-mapped on access (large datasets)."""

    def __init__(self, root: str, datasets: Sequence[str], modality: str, log=print,
                 split: Optional[Dict[str, Dict[str, List[str]]]] = None, local_dir: Optional[str] = None,
                 workers: int = 8, pipe: Optional[str] = None):
        self.root, self.modality = Path(root), modality
        self.nights: Dict[str, List[Night]] = {"train": [], "valid": [], "test": []}
        read, self.skipped = _read_all("single", root, datasets, modality, split, local_dir, workers, pipe)
        for k, night in read:
            self.nights[k].append(night)
        log(f"[clean_cache] {modality}: " + ", ".join(f"{k} {len(v)} nights" for k, v in self.nights.items())
            + f"; skipped {self.skipped}")


class SequenceDataset(Dataset):
    """Windows of L consecutive epochs. stride < L for training; eval windows tile each night (last one aligned to
    the end), and ``windows`` keeps (night, start) so predictions can be put back per epoch."""

    def __init__(self, nights: List[Night], seq_len: int = 21, stride: Optional[int] = None):
        self.nights, self.L = nights, seq_len
        stride = stride or seq_len
        self.windows: List[Tuple[int, int]] = []
        for k, nt in enumerate(nights):
            n = len(nt.y)
            if n < seq_len:
                continue
            starts = list(range(0, n - seq_len + 1, stride))
            if starts[-1] != n - seq_len:
                starts.append(n - seq_len)
            self.windows += [(k, s) for s in starts if (nt.y[s:s + seq_len] >= 0).any()]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        k, s = self.windows[i]
        nt = self.nights[k]
        x = torch.from_numpy(np.array(nt.x[s:s + self.L], dtype=np.float32))
        y = torch.from_numpy(nt.y[s:s + self.L])
        return x, y, torch.tensor(nt.offset_db, dtype=torch.float32)


class RandomOffsetSequences(Dataset):
    """Training windows re-cut at every pass (TinySleepNet-style sequence augmentation).

    ``set_epoch(k)`` draws one offset o ~ U[0, L) per night and cuts non-overlapping windows starting at o; the epochs
    before o and after the last full window are covered by one extra window aligned to the night start / end, so
    every epoch is seen at least once per pass (twice at most, only at the night edges). Each window is read from the
    contiguous night signal shifted by a random number of samples in [-max_shift, max_shift], clamped to the night;
    labels are unchanged. Workers must be re-created after ``set_epoch`` (build a new DataLoader every pass)."""

    def __init__(self, nights: List[Night], seq_len: int = 20, max_shift: int = 300, seed: int = 0):
        self.nights, self.L, self.max_shift, self.seed = nights, seq_len, max_shift, seed
        self.windows: List[Tuple[int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, k: int):
        rng = np.random.default_rng(self.seed + k)
        L, self.windows = self.L, []
        for i, nt in enumerate(self.nights):
            n = len(nt.y)
            if n < L:
                continue
            o = int(rng.integers(0, L))
            starts = list(range(o, n - L + 1, L))
            if o > 0:
                starts.insert(0, 0)
            if not starts or starts[-1] + L < n:
                starts.append(n - L)
            self.windows += [(i, s) for s in starts if (nt.y[s:s + L] >= 0).any()]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, j):
        i, s = self.windows[j]
        nt = self.nights[i]
        E = nt.x.shape[1]
        a = s * E
        if self.max_shift > 0:
            a += int(torch.randint(-self.max_shift, self.max_shift + 1, ()))   # torch RNG: distinct per worker
            a = min(max(a, 0), (len(nt.y) - self.L) * E)
        flat = nt.x.reshape(-1)
        x = np.array(flat[a:a + self.L * E], dtype=np.float32).reshape(self.L, E)
        return torch.from_numpy(x), torch.from_numpy(nt.y[s:s + self.L]), torch.tensor(nt.offset_db,
                                                                                       dtype=torch.float32)


def class_counts(nights: List[Night]) -> np.ndarray:
    c = np.zeros(5, np.int64)
    for nt in nights:
        v = nt.y[nt.y >= 0]
        c += np.bincount(v, minlength=5)
    return c


# ---------------------------------------------------------------------------------------------------------------------
# Three modalities of the same night, aligned (multimodal ChannelNet, physioex/models/channelnet_mm.py)
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class MultiNight:
    dataset: str
    subject: str
    x: Dict[str, np.ndarray]     # {m: (n, 3000)} at 100 Hz
    y: np.ndarray                # (n,) int64, -1 = masked
    offset_db: Dict[str, float]


def load_multimodal_night(root: Path, dataset: str, subject: str,
                          modalities: Sequence[str] = ("EEG", "EOG", "EMG"),
                          pipe: Optional[str] = None) -> Tuple[Optional[MultiNight], str]:
    """One night with all modalities: truncated to the shortest of signals and labels (aligned from the first epoch),
    margins trimmed where any modality is fully flat, labels masked where any modality is flat or partly flat.
    pipe: epoching-pipeline folder to use when a channel has several (e.g. "702cb90947921189", the one of MASS)."""
    lab = root / dataset / "labels" / f"{subject}.npy"
    if not lab.exists():
        return None, "no labels"
    y = np.load(lab).astype(np.int64)
    xs, fss = {}, {}
    for m in modalities:
        sig = [p for p in (root / dataset / "signals" / subject).glob(f"{m}_clean_*/*/signal.npy")
               if p.parent.name != "night"]
        if len(sig) > 1 and pipe is not None:      # several epoching pipelines: keep the requested one
            sig = [p for p in sig if p.parent.name == pipe]
        if len(sig) != 1:
            return None, f"no {m}"
        fss[m] = float(json.loads((sig[0].parent / "signal.meta.json").read_text())["fs_out"])
        xs[m] = np.load(sig[0], mmap_mode="r")
        if m != "EMG" and fss[m] != 100.0:
            return None, f"{m} fs {fss[m]}"
    n = min([len(y)] + [len(v) for v in xs.values()])
    if n == 0:
        return None, "empty"
    y = y[:n].copy()
    full = np.zeros(n, bool)
    part = np.zeros(n, bool)
    for m in modalities:
        xs[m] = np.asarray(xs[m][:n])
        f, p = flat_epochs(xs[m].astype(np.float32), fss[m])
        full |= f
        part |= p
    keep = np.where(~full)[0]
    if len(keep) == 0:
        return None, "all flat"
    a, b = keep[0], keep[-1] + 1
    y, full, part = y[a:b], full[a:b], part[a:b]
    y[(y < 0) | (y > 4) | full | part] = -1
    if (y >= 0).sum() == 0:
        return None, "no scored epoch"
    offs = {}
    for m in modalities:
        x = xs[m][a:b]
        if m == "EMG":
            xs[m] = to_100hz(x, fss[m]).astype(np.float16)
            offs[m] = float(10 * np.log10((min(100.0, fss[m] / 2) - 10.0) / 40.0))
        else:
            xs[m] = x
            offs[m] = 0.0
    return MultiNight(dataset, subject, xs, y, offs), "ok"


class CleanCacheMultiNights:
    """All nights with every requested modality, grouped by split (explicit split file or subject hash).
    ``local_dir`` / ``workers`` / ``pipe``: as ``CleanCacheNights`` and ``load_multimodal_night``."""

    def __init__(self, root: str, datasets: Sequence[str], modalities: Sequence[str] = ("EEG", "EOG", "EMG"),
                 log=print, split: Optional[Dict[str, Dict[str, List[str]]]] = None,
                 local_dir: Optional[str] = None, workers: int = 8, pipe: Optional[str] = None):
        self.root, self.modalities = Path(root), list(modalities)
        self.nights: Dict[str, List[MultiNight]] = {"train": [], "valid": [], "test": []}
        read, self.skipped = _read_all("multi", root, datasets, self.modalities, split, local_dir, workers, pipe)
        for k, night in read:
            self.nights[k].append(night)
        log(f"[clean_cache] {'+'.join(self.modalities)}: "
            + ", ".join(f"{k} {len(v)} nights" for k, v in self.nights.items()) + f"; skipped {self.skipped}")


class MultiSequenceDataset(Dataset):
    """Windows of L consecutive epochs over all modalities: ({m: (L, 3000)}, (L,), {m: offset_db})."""

    def __init__(self, nights: List[MultiNight], seq_len: int = 20, stride: Optional[int] = None):
        self.nights, self.L = nights, seq_len
        stride = stride or seq_len
        self.windows: List[Tuple[int, int]] = []
        for k, nt in enumerate(nights):
            n = len(nt.y)
            if n < seq_len:
                continue
            starts = list(range(0, n - seq_len + 1, stride))
            if starts[-1] != n - seq_len:
                starts.append(n - seq_len)
            self.windows += [(k, s) for s in starts if (nt.y[s:s + seq_len] >= 0).any()]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        k, s = self.windows[i]
        nt = self.nights[k]
        x = {m: torch.from_numpy(np.array(v[s:s + self.L], dtype=np.float32)) for m, v in nt.x.items()}
        off = {m: torch.tensor(v, dtype=torch.float32) for m, v in nt.offset_db.items()}
        return x, torch.from_numpy(nt.y[s:s + self.L]), off


def multi_class_counts(nights: List[MultiNight]) -> np.ndarray:
    c = np.zeros(5, np.int64)
    for nt in nights:
        c += np.bincount(nt.y[nt.y >= 0], minlength=5)
    return c


class MultiRandomOffsetSequences(RandomOffsetSequences):
    """``RandomOffsetSequences`` over ``MultiNight``: same windows, one time shift shared by all modalities
    (they stay aligned): ({m: (L, 3000)}, (L,), {m: offset_db})."""

    def __getitem__(self, j):
        i, s = self.windows[j]
        nt = self.nights[i]
        E = next(iter(nt.x.values())).shape[1]
        a = s * E
        if self.max_shift > 0:
            a += int(torch.randint(-self.max_shift, self.max_shift + 1, ()))
            a = min(max(a, 0), (len(nt.y) - self.L) * E)
        x = {m: torch.from_numpy(np.array(v.reshape(-1)[a:a + self.L * E], dtype=np.float32).reshape(self.L, E))
             for m, v in nt.x.items()}
        off = {m: torch.tensor(v, dtype=torch.float32) for m, v in nt.offset_db.items()}
        return x, torch.from_numpy(nt.y[s:s + self.L]), off
