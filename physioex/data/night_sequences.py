"""Long training sequences from whole nights (xmai-lab/sleep-mechinterp#32).

A sequence is an index array into one night's epochs, drawn by one of:
    whole       the real night
    perm        cycle permutation: the night is cut at sleep-cycle boundaries read from the hypnogram (end of a REM
                period of >= rem_min epochs followed by non-REM, end of a W bout of >= w_min epochs); the first segment
                (sleep onset) stays in place and the others are shuffled. A permutation is accepted only if every new
                junction (last label of a segment -> first label of the next) has probability >= p_min in the training
                hypnograms; otherwise redraw (max_tries), then the real night. Fewer than 3 segments: the real night.
    permrand    as perm with the same number of cuts at random positions and no junction rule
    crop        contiguous crop of length log-uniform in [min_len, N]
    l20         contiguous window of exactly 20 epochs (the original training input)

``SequenceSampler`` yields padded batches {x: {m: (B, Lmax, E) float32}, y (B, Lmax) with -1 padding, off {m: (B,)},
lengths (B,)}; long sequences are grouped by similar length (pool of 4 batches sorted by length).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

N_STAGES = 5
W, REM = 0, 4


def transition_matrix(ys: Sequence[np.ndarray]) -> np.ndarray:
    """Row-normalised 5x5 stage-transition matrix over consecutive scored epochs."""
    c = np.zeros((N_STAGES, N_STAGES))
    for y in ys:
        a, b = y[:-1], y[1:]
        ok = (a >= 0) & (b >= 0)
        np.add.at(c, (a[ok], b[ok]), 1)
    return c / np.maximum(c.sum(1, keepdims=True), 1)


def _runs(y: np.ndarray, label: int):
    """(start, end) of the maximal runs of `label` (end exclusive)."""
    m = np.concatenate([[0], (y == label).astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def cycle_cuts(y: np.ndarray, rem_min: int = 10, w_min: int = 5, margin: int = 5, min_gap: int = 10) -> List[int]:
    """Cut positions (segment starts) at sleep-cycle boundaries."""
    n = len(y)
    cand = [e for s, e in _runs(y, REM) if e - s >= rem_min and e < n and y[e] != REM]
    cand += [e for s, e in _runs(y, W) if e - s >= w_min and e < n]
    cuts = []
    for c in sorted(set(int(c) for c in cand)):
        if margin <= c <= n - margin and (not cuts or c - cuts[-1] >= min_gap):
            cuts.append(c)
    return cuts


def segments(n: int, cuts: Sequence[int]):
    b = [0, *cuts, n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def _edge(y, s, e, last):
    """Last (or first) scored label of y[s:e], -1 if none."""
    v = y[s:e][y[s:e] >= 0]
    return int(v[-1] if last else v[0]) if len(v) else -1


def junctions_ok(y, segs, trans, p_min):
    for (s1, e1), (s2, e2) in zip(segs[:-1], segs[1:]):
        a, b = _edge(y, s1, e1, True), _edge(y, s2, e2, False)
        if a >= 0 and b >= 0 and trans[a, b] < p_min:
            return False
    return True


def permute_night(y: np.ndarray, trans: np.ndarray, rng: np.random.Generator, mode: str = "cycle",
                  p_min: float = 0.01, max_tries: int = 20, **cut_kw) -> np.ndarray:
    """Index array of a permuted night (identity when no valid permutation exists)."""
    n = len(y)
    cuts = cycle_cuts(y, **cut_kw)
    ident = np.arange(n)
    if len(cuts) < 2:                                       # fewer than 3 segments
        return ident
    if mode == "random":
        k = len(cuts)
        cuts = sorted(rng.choice(np.arange(10, n - 10), size=k, replace=False).tolist())
    segs = segments(n, cuts)
    for _ in range(max_tries):
        rest = [segs[i] for i in rng.permutation(np.arange(1, len(segs)))]
        new = [segs[0], *rest]
        if mode == "random" or junctions_ok(y, new, trans, p_min):
            return np.concatenate([np.arange(s, e) for s, e in new])
    return ident


def crop(n: int, rng: np.random.Generator, min_len: int = 20) -> np.ndarray:
    L = int(round(np.exp(rng.uniform(np.log(min_len), np.log(n))))) if n > min_len else n
    s = int(rng.integers(0, n - L + 1))
    return np.arange(s, s + L)


MIXES = {                                                   # (mode, probability) per configuration family
    "mix": {"whole": 1 / 3, "perm": 1 / 3, "crop": 1 / 3},
    "whole": {"whole": 1.0},
    "crop": {"crop": 1.0},
    "perm": {"perm": 1.0},
    "noperm": {"whole": 0.5, "crop": 0.5},
    "permrand": {"whole": 1 / 3, "permrand": 1 / 3, "crop": 1 / 3},
    "l20": {"l20": 1.0},
}


class SequenceSampler:
    """Infinite iterator of padded batches drawn from in-memory nights (night.x {m: (N, E)}, night.y, offset_db)."""

    def __init__(self, nights, mix: Dict[str, float], trans: np.ndarray, seed: int = 0, batch: int = 4,
                 batch_l20: int = 32, pool: int = 4):
        self.nights = nights
        self.modes, p = zip(*mix.items())
        self.p = np.array(p) / sum(p)
        self.trans = trans
        self.rng = np.random.default_rng(seed)
        self.l20 = self.modes == ("l20",)
        self.batch = batch_l20 if self.l20 else batch
        self.pool = pool
        self.mods = list(nights[0].x)
        if self.l20:                                         # windows drawn in proportion to night length
            w = np.array([max(len(nt.y) - 19, 0) for nt in nights], float)
            self.w = w / w.sum()

    def draw(self):
        rng = self.rng
        if self.l20:
            k = int(rng.choice(len(self.nights), p=self.w))
            s = int(rng.integers(0, len(self.nights[k].y) - 19))
            return k, np.arange(s, s + 20)
        k = int(rng.integers(len(self.nights)))
        y = self.nights[k].y
        mode = self.modes[int(rng.choice(len(self.modes), p=self.p))]
        if mode == "whole":
            idx = np.arange(len(y))
        elif mode == "perm":
            idx = permute_night(y, self.trans, rng, "cycle")
        elif mode == "permrand":
            idx = permute_night(y, self.trans, rng, "random")
        else:
            idx = crop(len(y), rng)
        return k, idx

    def collate(self, items):
        L = max(len(i) for _, i in items)
        B = len(items)
        x = {m: torch.zeros(B, L, self.nights[0].x[m].shape[-1]) for m in self.mods}
        y = torch.full((B, L), -1, dtype=torch.long)
        off = {m: torch.zeros(B) for m in self.mods}
        for b, (k, idx) in enumerate(items):
            nt = self.nights[k]
            n = len(idx)
            for m in self.mods:
                x[m][b, :n] = torch.from_numpy(np.asarray(nt.x[m][idx], dtype=np.float32))
                off[m][b] = float(nt.offset_db[m])
            y[b, :n] = torch.from_numpy(nt.y[idx].astype(np.int64))
        return {"x": x, "y": y, "off": off, "lengths": torch.tensor([len(i) for _, i in items])}

    def __iter__(self):
        while True:
            if self.l20:
                yield self.collate([self.draw() for _ in range(self.batch)])
                continue
            items = sorted((self.draw() for _ in range(self.batch * self.pool)), key=lambda t: len(t[1]))
            groups = [items[i:i + self.batch] for i in range(0, len(items), self.batch)]
            for g in (groups[i] for i in self.rng.permutation(len(groups))):
                yield self.collate(g)
