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


@dataclass
class Night:
    dataset: str
    subject: str
    x: np.ndarray            # (n, 3000) float16/float32, 100 Hz
    y: np.ndarray            # (n,) int64, -1 = masked
    offset_db: float
    fs: float


def load_night(root: Path, dataset: str, subject: str, modality: str) -> Tuple[Optional[Night], str]:
    # <hash>/ holds the epoched pipeline output; night/ (full cache) is the continuous cleaned night: not used here
    sig = [p for p in (root / dataset / "signals" / subject).glob(f"{modality}_clean_*/*/signal.npy")
           if p.parent.name != "night"]
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
    """All nights of some datasets for one modality, grouped by split."""

    def __init__(self, root: str, datasets: Sequence[str], modality: str, log=print,
                 split: Optional[Dict[str, Dict[str, List[str]]]] = None):
        self.root, self.modality = Path(root), modality
        self.nights: Dict[str, List[Night]] = {"train": [], "valid": [], "test": []}
        self.skipped: Dict[str, int] = {}
        for ds in datasets:
            where = None if split is None else {s: k for k, ids in split[ds].items() for s in ids}
            for subj in sorted(p.name for p in (self.root / ds / "signals").iterdir()):
                if where is not None and subj not in where:
                    self.skipped["not in split"] = self.skipped.get("not in split", 0) + 1
                    continue
                night, why = load_night(self.root, ds, subj, modality)
                if night is None:
                    self.skipped[why] = self.skipped.get(why, 0) + 1
                    continue
                self.nights[where[subj] if where is not None else split_of(ds, subj)].append(night)
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


def class_counts(nights: List[Night]) -> np.ndarray:
    c = np.zeros(5, np.int64)
    for nt in nights:
        v = nt.y[nt.y >= 0]
        c += np.bincount(v, minlength=5)
    return c
