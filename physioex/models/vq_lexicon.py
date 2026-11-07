"""Event lexicon of the quantized P latent of an MSAE-VQ model (explainability layer).

A :class:`Lexicon` turns the quantized P tokens of an epoch (``(n, T, d_p)``, 0.5 s tokens) into
named words and events. It is fitted without labels on a reference cohort (DCSM): each basis is a
FastICA of the tokens (``pq1``) or of 3 consecutive tokens (``pq3``), and a word is a threshold,
set at a DCSM quantile, on

- ``signed``: one coordinate (lower or upper tail),
- ``folded``: the absolute value of one coordinate,
- ``energy``: the sum of squares of a few coordinates, optionally averaged over +-1 token
  (``smooth``) or over a centred window of ``win`` tokens; with ``rel`` the value is the log ratio
  of that energy to the median of the ``rel`` tokens before the window (an abrupt change, as an
  arousal).

A ``code`` word reads the discrete P codes instead of ``pq``: its basis (``rep: "code"``) holds the
log-lift of every code of every RVQ level (``loglift``, ``(L, n_codes)``), the score of a token is the
sum of the log-lifts of its codes (naive Bayes over the levels), optionally averaged over ``win``
tokens. The codes alone determine the word.

A ``code_prefix`` word (08.11.2026) is a compact code family: its basis (``rep: "code_prefix"``) holds the
prefixes of the word (``pref_keys0``: first RVQ code; ``pref_keys1``: first two codes as
``c0 * n_codes + c1``, sorted) and their log-lifts (``pref_w0``, ``pref_w1``); the score of a token is the
largest log-lift among its prefixes in the word (0 outside it), optionally averaged over ``win`` tokens.

An event can carry post-processing rules in ``meta["event_rules"][event]``: ``gap`` (largest gap,
in tokens, merged inside one event; default 1) and ``min_tok`` (shortest event kept; default 1).

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
_ARRAYS = ("mean", "unmix", "mix", "loglift", "pref_keys0", "pref_w0", "pref_keys1", "pref_w1")


def ctx3(f: np.ndarray) -> np.ndarray:
    """``(n, T, d) -> (n*T, 3d)``: previous, current, next token (edges repeated)."""
    n, T, d = f.shape
    v = f.reshape(-1, d).astype(np.float32)
    return np.concatenate([np.r_[v[:1], v[:-1]], v, np.r_[v[1:], v[-1:]]], 1)


def window_mean(e: np.ndarray, win: int, T: int) -> np.ndarray:
    """Centred mean over ``win`` tokens inside rows of ``T`` tokens (edges repeated)."""
    h = win // 2
    e2 = e.reshape(-1, T)
    pad = np.concatenate([np.repeat(e2[:, :1], h, 1), e2, np.repeat(e2[:, -1:], win - 1 - h, 1)], 1)
    c = np.cumsum(np.concatenate([np.zeros((len(e2), 1)), pad], 1), 1)
    return ((c[:, win:] - c[:, :-win]) / win).reshape(-1)


def rel_change(e: np.ndarray, base: int, win: int, T: int) -> np.ndarray:
    """``log(e[t]) - log(median(e[t - win//2 - base : t - win//2]))`` inside rows of ``T`` tokens;
    the first tokens of a row use the tokens available before them (at least one)."""
    from numpy.lib.stride_tricks import sliding_window_view

    h = win // 2
    L = np.log(np.maximum(e.reshape(-1, T), 1e-6))
    P = np.concatenate([np.full((len(L), base), np.nan), L], 1)
    W = sliding_window_view(P, base, axis=1)  # W[:, k] = L[:, k - base:k]
    hi = np.maximum(1, np.arange(T) - h)
    return (L - np.nanmedian(W[:, hi], -1)).reshape(-1)


def token_runs(m: np.ndarray, gap: int = 1, min_tok: int = 1) -> list:
    """Runs of positive tokens, gaps of up to ``gap`` tokens merged, runs shorter than ``min_tok``
    dropped. Returns ``[(first, last)]`` token indices (inclusive)."""
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    br = np.diff(idx) > gap + 1
    starts = np.r_[idx[0], idx[1:][br]]
    ends = np.r_[idx[:-1][br], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_tok]


class Lexicon:
    def __init__(self, bases: Dict[str, dict], words: List[dict], meta: dict | None = None):
        """``bases[name] = {"rep": "pq1"|"pq3", "mean": (D_in,), "unmix": (k, D_in), "mix": (D_in, k)}``
        plus, for a multimodal lexicon, ``"mod"`` (the modality whose P tokens the basis reads; default
        ``"eeg"``); ``words[i] = {"name", "event", "basis", "kind", "dirs", "threshold", "side", "smooth"}``
        (extra keys, e.g. the gates of the word, are kept as they are). A code basis is
        ``{"rep": "code", "mod", "loglift": (L, n_codes)}`` and its words have ``kind: "code"``."""
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
        win = int(w.get("win", 3 if w.get("smooth") else 1))
        if win > 1:
            e = window_mean(e, win, T)
        if w.get("rel"):
            e = rel_change(e, int(w["rel"]), win, T)
        return e

    def mod_of(self, basis: str) -> str:
        return self.bases[basis].get("mod", "eeg")

    def code_values(self, codes: np.ndarray, w: dict, T: int) -> np.ndarray:
        """Naive-Bayes score of the code word ``w`` on ``codes`` (``(n, T, L)`` P codes)."""
        ll = self.bases[w["basis"]]["loglift"]
        c = codes.reshape(-1, codes.shape[-1])
        v = sum(ll[l][c[:, l].astype(np.int64)] for l in range(ll.shape[0])).astype(np.float64)
        win = int(w.get("win", 1))
        return window_mean(v, win, T) if win > 1 else v

    def prefix_values(self, codes: np.ndarray, w: dict, T: int) -> np.ndarray:
        """Score of the compact code word ``w`` on ``codes`` (``(n, T, L)`` P codes)."""
        b = self.bases[w["basis"]]
        c = codes.reshape(-1, codes.shape[-1]).astype(np.int64)
        nc = int(w["n_codes"])
        v = np.zeros(len(c), np.float32)
        for lvl in (0, 1):
            keys = b.get(f"pref_keys{lvl}")
            if keys is None or not len(keys) or c.shape[1] <= lvl:
                continue
            k = c[:, 0] if lvl == 0 else c[:, 0] * nc + c[:, 1]
            pos = np.searchsorted(keys, k).clip(0, len(keys) - 1)
            v = np.maximum(v, np.where(keys[pos] == k, b[f"pref_w{lvl}"][pos], np.float32(0)))
        win = int(w.get("win", 1))
        return window_mean(v.astype(np.float64), win, T) if win > 1 else v

    def fire(self, pq, continuous: bool = True, codes=None) -> Dict[str, np.ndarray]:
        """Word masks over the ``n*T`` tokens of ``pq`` (``(n, T, d_p)``, or ``{mod: (n, T, d_p)}`` for a
        multimodal lexicon: the words of a missing modality are left out). ``codes``: the P codes
        (``(n, T, L)`` or ``{mod: ...}``) read by the ``code`` words; without them those words are left
        out. ``continuous``: the epochs are consecutive (smoothing and 3-token context run across epoch
        borders)."""
        pqs = pq if isinstance(pq, dict) else None
        cds = codes if isinstance(codes, dict) or codes is None else None

        def have(w):
            m = self.mod_of(w["basis"])
            if w.get("kind") in ("code", "code_prefix"):
                return codes is not None and (cds is None or m in cds)
            return pq is not None and (pqs is None or m in pqs)

        words = [w for w in self.words if have(w)]
        src = {b: (pqs[self.mod_of(b)] if pqs is not None else pq)
               for b in {w["basis"] for w in words if w.get("kind") not in ("code", "code_prefix")}}
        Zc = {b: self.coords(x, b) for b, x in src.items()}
        out = {}
        for w in words:
            if w.get("kind") in ("code", "code_prefix"):
                c = cds[self.mod_of(w["basis"])] if cds is not None else codes
                fn = self.code_values if w["kind"] == "code" else self.prefix_values
                v = fn(c, w, c.shape[0] * c.shape[1] if continuous else c.shape[1])
            else:
                n, T, _ = src[w["basis"]].shape
                v = self.word_values(Zc[w["basis"]], w, n * T if continuous else T)
            out[w["name"]] = v >= w["threshold"] if w["side"] > 0 else v <= w["threshold"]
        return out

    def events(self, pq: np.ndarray, continuous: bool = True, codes=None) -> Dict[str, List[dict]]:
        """Events per name: runs of the union of the event's words (gaps merged and short runs
        dropped as in ``meta["event_rules"]``, default: gaps of one token), with the words that fired inside. Times in seconds from the start of ``pq``."""
        masks = self.fire(pq, continuous, codes)
        out = {}
        for ev in sorted({w["event"] for w in self.words if w["name"] in masks}):
            names = [w["name"] for w in self.words if w["event"] == ev and w["name"] in masks]
            m = np.any([masks[k] for k in names], 0)
            rule = self.meta.get("event_rules", {}).get(ev, {})
            runs = []
            for s, e in token_runs(m, rule.get("gap", 1), rule.get("min_tok", 1)):
                runs.append({"start_s": float(s * TOK_S), "stop_s": float((e + 1) * TOK_S),
                             "words": [k for k in names if masks[k][s:e + 1].any()]})
            out[ev] = runs
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        arrays = {f"{b}__{k}": v[k] for b, v in self.bases.items() for k in _ARRAYS if k in v}
        np.savez_compressed(os.path.join(path, "lexicon.npz"), **arrays)
        js_b = {b: {"rep": v["rep"], **({"mod": v["mod"]} if "mod" in v else {})} for b, v in self.bases.items()}
        json.dump({"bases": js_b, "words": self.words, "meta": self.meta},
                  open(os.path.join(path, "lexicon.json"), "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        js = json.load(open(os.path.join(path, "lexicon.json")))
        z = np.load(os.path.join(path, "lexicon.npz"))
        bases = {b: {**v, **{k: z[f"{b}__{k}"] for k in _ARRAYS if f"{b}__{k}" in z.files}} for b, v in js["bases"].items()}
        return cls(bases, js["words"], js.get("meta"))
