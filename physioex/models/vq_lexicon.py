"""Event lexicon of the quantized P latent of an MSAE-VQ model (explainability layer).

A :class:`Lexicon` turns the quantized P tokens of an epoch (``(n, T, d_p)``, 0.5 s tokens) into
named words and events. It is fitted without labels on a reference cohort (DCSM): each basis is a
FastICA of the tokens (``pq1``) or of 3 consecutive tokens (``pq3``), and a word is a threshold,
set at a DCSM quantile, on

- ``signed``: one coordinate (lower or upper tail),
- ``folded``: the absolute value of one coordinate,
- ``energy``: the sum of squares of a few coordinates, optionally averaged over +-1 token.

The words are only *named* with expert labels (DREAMS): the name of an event (``k_complex``,
``spindle``) is the union of the words chosen for it. Fitting, selection and validation live in
``physioex.train.bin.vq_event_lexicon2`` / ``vq_event_finalize``; this module is the frozen,
model-side reader saved with the run (``lexicon.json`` + ``lexicon.npz``).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

TOK_S = 0.5


def ctx3(f: np.ndarray) -> np.ndarray:
    """``(n, T, d) -> (n*T, 3d)``: previous, current, next token (edges repeated)."""
    n, T, d = f.shape
    v = f.reshape(-1, d).astype(np.float32)
    return np.concatenate([np.r_[v[:1], v[:-1]], v, np.r_[v[1:], v[-1:]]], 1)


class Lexicon:
    def __init__(self, bases: Dict[str, dict], words: List[dict], meta: dict | None = None):
        """``bases[name] = {"rep": "pq1"|"pq3", "mean": (D_in,), "unmix": (k, D_in), "mix": (D_in, k)}``;
        ``words[i] = {"name", "event", "basis", "kind", "dirs", "threshold", "side", "smooth"}``."""
        self.bases, self.words, self.meta = bases, words, meta or {}

    # ------------------------------------------------------------------ reading
    def coords(self, pq: np.ndarray, basis: str) -> np.ndarray:
        b = self.bases[basis]
        X = pq.reshape(-1, pq.shape[-1]).astype(np.float32) if b["rep"] == "pq1" else ctx3(pq)
        return (X - b["mean"]) @ b["unmix"].T

    def word_values(self, Z: np.ndarray, w: dict, T: int) -> np.ndarray:
        if w["kind"] == "signed":
            return Z[:, w["dirs"][0]]
        if w["kind"] == "folded":
            return np.abs(Z[:, w["dirs"][0]])
        e = (Z[:, w["dirs"]] ** 2).sum(1)
        if not w.get("smooth"):
            return e
        e2 = e.reshape(-1, T)
        pad = np.concatenate([e2[:, :1], e2, e2[:, -1:]], 1)
        return ((pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3).reshape(-1)

    def fire(self, pq: np.ndarray, continuous: bool = True) -> Dict[str, np.ndarray]:
        """Word masks over the ``n*T`` tokens of ``pq`` (``(n, T, d_p)``). ``continuous``: the epochs
        are consecutive (smoothing and 3-token context run across epoch borders)."""
        n, T, _ = pq.shape
        Zc = {b: self.coords(pq, b) for b in {w["basis"] for w in self.words}}
        out = {}
        for w in self.words:
            v = self.word_values(Zc[w["basis"]], w, n * T if continuous else T)
            out[w["name"]] = v >= w["threshold"] if w["side"] > 0 else v <= w["threshold"]
        return out

    def events(self, pq: np.ndarray, continuous: bool = True) -> Dict[str, List[dict]]:
        """Events per name: runs of the union of the event's words (gaps of one token merged),
        with the words that fired inside. Times in seconds from the start of ``pq``."""
        masks = self.fire(pq, continuous)
        out = {}
        for ev in sorted({w["event"] for w in self.words}):
            names = [w["name"] for w in self.words if w["event"] == ev]
            m = np.any([masks[k] for k in names], 0)
            idx = np.flatnonzero(m)
            runs = []
            if len(idx):
                starts = np.r_[idx[0], idx[1:][np.diff(idx) > 2]]
                ends = np.r_[idx[:-1][np.diff(idx) > 2], idx[-1]]
                for s, e in zip(starts, ends):
                    runs.append({"start_s": float(s * TOK_S), "stop_s": float((e + 1) * TOK_S),
                                 "words": [k for k in names if masks[k][s:e + 1].any()]})
            out[ev] = runs
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        arrays = {f"{b}__{k}": v[k] for b, v in self.bases.items() for k in ("mean", "unmix", "mix")}
        np.savez_compressed(os.path.join(path, "lexicon.npz"), **arrays)
        json.dump({"bases": {b: {"rep": v["rep"]} for b, v in self.bases.items()}, "words": self.words, "meta": self.meta},
                  open(os.path.join(path, "lexicon.json"), "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        js = json.load(open(os.path.join(path, "lexicon.json")))
        z = np.load(os.path.join(path, "lexicon.npz"))
        bases = {b: {"rep": v["rep"], **{k: z[f"{b}__{k}"] for k in ("mean", "unmix", "mix")}} for b, v in js["bases"].items()}
        return cls(bases, js["words"], js.get("meta"))
