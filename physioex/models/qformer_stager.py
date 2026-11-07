"""Q-Former stager on frozen MSAE / MSAE-VQ latents (downstream check; labels only here).

The encoder is frozen: :func:`extract_latents` stores ``N`` ``(128,)``, ``E`` ``(128,)`` and
``P`` ``(T_p, d_p)`` for every scored epoch of the main sleep period (MSAE-VQ: the quantized
latents of the eval prefix ``vq_levels``). :class:`QFormerStager` reads one epoch:

* tokens: ``[W_n N + type_N, W_e E + type_E, W_p P_j + type_P + pos_j]`` -> ``(2 + T_p, d)``;
* ``k`` learned queries, ``layers`` blocks of self-attention among the queries, cross-attention
  from the queries to the tokens and an FFN (pre-norm, as in the BLIP-2 Q-Former);
* attention pooling of the ``k`` outputs with one learned query, then a linear layer -> 5 stages.

:class:`LinearStager` (``[N; E]`` -> 5) is the reference at the same data. The checkpoint is
chosen on the mean per-night kappa of the validation nights; the test nights are read once.
:func:`attention_by_stage` gives, per stage, the share of the last cross-attention on N, E and
on every P token (which part of the epoch the stager reads).

``python -m physioex.models.qformer_stager extract --run RUN [--vq N8E8P6] --out DIR``
``python -m physioex.models.qformer_stager train --data DIR --arch qformer|linear --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPLITS = ("train", "valid", "test")
STAGES = ("W", "N1", "N2", "N3", "REM")


# ----------------------------------------------------------------------------
# Latents of every scored epoch
# ----------------------------------------------------------------------------


@torch.no_grad()
def extract_latents(model, ds, subject_ids: Sequence[str], batch: int = 128) -> Dict[str, np.ndarray]:
    """Frozen latents of every scored epoch of the main sleep period whose window fits in the
    night (the first and last ``L // 2`` epochs of a night have no full context and are left out).
    Returns ``N``, ``E`` ``(M, dim)`` and ``P`` ``(M, T_p, d_p)`` in float16, ``y`` ``(M,)``,
    ``night`` (position in ``subject_ids``) and ``epoch`` (index in the night)."""
    device = next(model.parameters()).device
    model.eval()
    have = {s.subject_id: s for s in ds._subjects}
    half = ds.sequence_length // 2
    acc = {k: [] for k in ("N", "E", "P", "y", "night", "epoch")}
    for i, sid in enumerate(subject_ids):
        if sid not in have:
            continue
        n = ds._n_epochs[sid]
        lo, hi = ds._main_period.get(sid, (0, n)) if hasattr(ds, "_main_period") else (0, n)
        lo, hi = max(lo, half), min(hi, n - half)
        if hi <= lo:
            continue
        item = ds._build_item(have[sid], 0, n)
        x = next(iter(item["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        y = item["labels"].numpy().reshape(-1)
        ts = np.arange(lo, hi)
        ts = ts[y[ts] >= 0]
        for j in range(0, len(ts), batch):
            tb = ts[j:j + batch]
            w = np.stack([x[t - half:t + half + 1] for t in tb])
            z = model.encode(torch.from_numpy(w)[:, :, None].to(device))
            for k in ("N", "E", "P"):
                acc[k].append(z[k].float().cpu().numpy().astype(np.float16))
        acc["y"].append(y[ts].astype(np.int8))
        acc["night"].append(np.full(len(ts), i, np.int16))
        acc["epoch"].append(ts.astype(np.int32))
    return {k: np.concatenate(v) for k, v in acc.items()}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ln_q, self.ln_c, self.ln_kv, self.ln_f = (nn.LayerNorm(d) for _ in range(4))
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        h = self.ln_q(q)
        q = q + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        kv = self.ln_kv(kv)
        a, w = self.cross_attn(self.ln_c(q), kv, kv, need_weights=True, average_attn_weights=True)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.ln_f(q)))
        return q, w  # w: (B, k, 2 + T_p)


class QFormerStager(nn.Module):
    """``k`` learned queries read the N, E and P tokens of one epoch; attention pooling -> stage."""

    def __init__(self, dim: int = 128, d_p: int = 16, n_tok: int = 60, d: int = 128, k: int = 8,
                 layers: int = 2, heads: int = 4, n_classes: int = 5, dropout: float = 0.1, **_):
        super().__init__()
        self.proj_n, self.proj_e, self.proj_p = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(d_p, d)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList([_Block(d, heads, dropout) for _ in range(layers)])
        self.pool_query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.last_attn = None

    def tokens(self, n, e, p):
        return torch.cat([(self.proj_n(n) + self.type_emb[0])[:, None], (self.proj_e(e) + self.type_emb[1])[:, None],
                          self.proj_p(p) + self.type_emb[2] + self.pos], 1)

    def embed(self, n, e, p):
        """Pooled vector of one epoch, ``(B, d)``; sets ``last_attn``."""
        kv = self.tokens(n, e, p)
        q = self.queries.expand(len(kv), -1, -1)
        for blk in self.blocks:
            q, w = blk(q, kv)
        self.last_attn = w.detach().mean(1)  # last cross-attention, mean over queries: (B, 2 + T_p)
        return self.pool(self.pool_query.expand(len(q), -1, -1), q, q, need_weights=False)[0][:, 0]

    def forward(self, n, e, p):
        return self.head(self.ln(self.embed(n, e, p)))


class LinearStager(nn.Module):
    """Reference: one linear layer on ``[N; E]`` (P unused), as the logistic probe."""

    def __init__(self, dim: int = 128, n_classes: int = 5, **_):
        super().__init__()
        self.head = nn.Linear(2 * dim, n_classes)
        self.last_attn = None

    def forward(self, n, e, p):
        return self.head(torch.cat([n, e], -1))


class _LinearEmbed(nn.Module):
    """Per-epoch embedding of ``[N; E]`` (P unused) for :class:`SeqStager`."""

    def __init__(self, dim: int = 128, d: int = 128, **_):
        super().__init__()
        self.proj = nn.Linear(2 * dim, d)
        self.last_attn = None

    def embed(self, n, e, p):
        return self.proj(torch.cat([n, e], -1))


class SeqStager(nn.Module):
    """Sequence of ``seq_len`` consecutive epochs, as ProtoSleepNet: a per-epoch embedding
    (``epoch="qformer"``: the Q-Former above without its head; ``"linear"``: ``[N; E]``),
    a temporal transformer over the sequence, a stage for every epoch. Input ``(B, L, ...)``,
    output ``(B, L, n_classes)``."""

    def __init__(self, epoch: str = "qformer", seq_len: int = 21, seq_layers: int = 2, d: int = 128,
                 heads: int = 4, n_classes: int = 5, dropout: float = 0.1, **kw):
        super().__init__()
        self.seq_len = seq_len
        enc = QFormerStager if epoch == "qformer" else _LinearEmbed
        self.enc = enc(d=d, heads=heads, n_classes=n_classes, dropout=dropout, **kw)
        if epoch == "qformer":  # head of the per-epoch model unused
            del self.enc.head, self.enc.ln
        self.pos_t = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, seq_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.last_attn = None

    def forward(self, n, e, p):
        B, L = n.shape[:2]
        h = self.enc.embed(n.flatten(0, 1), e.flatten(0, 1), p.flatten(0, 1)).view(B, L, -1)
        self.last_attn = self.enc.last_attn  # (B * L, 2 + T_p) or None
        return self.head(self.ln(self.temporal(h + self.pos_t[:L])))


class _JointQFormer(nn.Module):
    """Early fusion at token level: one Q-Former whose ``k`` queries read the N, E and P tokens of
    every modality at once. Every token: per-modality projection + type (N/E/P) + per-modality
    position (P) + modality embedding. Same blocks and pooling as :class:`QFormerStager`."""

    def __init__(self, spec, d: int = 128, k: int = 24, layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spec = spec
        self.proj_n = nn.ModuleList([nn.Linear(dim, d) for _, dim, _, _ in spec])
        self.proj_e = nn.ModuleList([nn.Linear(dim, d) for _, dim, _, _ in spec])
        self.proj_p = nn.ModuleList([nn.Linear(d_p, d) for _, _, d_p, _ in spec])
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(n_tok, d) * 0.02) for *_, n_tok in spec])
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.mod_emb = nn.Parameter(torch.randn(len(spec), d) * 0.02)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList([_Block(d, heads, dropout) for _ in range(layers)])
        self.pool_query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.last_attn = None

    def embed(self, n, e, p):
        toks, a, t = [], 0, 0
        for i, (_, dim, d_p, n_tok) in enumerate(self.spec):
            m = self.mod_emb[i]
            toks += [(self.proj_n[i](n[:, a:a + dim]) + self.type_emb[0] + m)[:, None],
                     (self.proj_e[i](e[:, a:a + dim]) + self.type_emb[1] + m)[:, None],
                     self.proj_p[i](p[:, t:t + n_tok, :d_p]) + self.type_emb[2] + self.pos[i] + m]
            a, t = a + dim, t + n_tok
        kv = torch.cat(toks, 1)
        q = self.queries.expand(len(kv), -1, -1)
        for blk in self.blocks:
            q, w = blk(q, kv)
        self.last_attn = w.detach().mean(1)  # (B, sum(2 + T_m))
        return self.pool(self.pool_query.expand(len(q), -1, -1), q, q, need_weights=False)[0][:, 0]


class MultiSeqStager(nn.Module):
    """:class:`SeqStager` over several modalities (EEG, EOG, EMG). Every modality has its own
    per-epoch embedding (``epoch="qformer"`` or ``"linear"``) plus a learned modality embedding;
    the modality embeddings of an epoch are concatenated and projected to ``d``, then the
    temporal transformer and the head as in :class:`SeqStager`.

    Inputs are the packed tensors of :func:`pack_modalities`: ``N``, ``E`` ``(B, L, sum dim_m)``
    and ``P`` ``(B, L, sum T_m, max d_p)``; ``spec`` gives ``(name, dim, d_p, n_tok)`` per modality."""

    def __init__(self, spec, epoch: str = "qformer", seq_len: int = 21, seq_layers: int = 2, d: int = 128,
                 heads: int = 4, n_classes: int = 5, dropout: float = 0.1, k: int = 8, layers: int = 2,
                 fusion: str = "late", mod_dropout: float = 0.0, acc_momentum: float = 0.9, **_):
        super().__init__()
        self.spec, self.seq_len, self.fusion = [tuple(s) for s in spec], seq_len, fusion
        self.mod_dropout, self.acc_momentum = mod_dropout, acc_momentum
        if mod_dropout:  # accuracy-weighted modality dropout as ProtoSleepNet (ZeroEmbeddingDropout)
            assert fusion == "late", "modality dropout: late fusion only"
            # per-modality heads on the detached embeddings: they estimate each modality's accuracy
            # (the drop weights) without changing the representation
            self.aux = nn.ModuleList([nn.Linear(d, n_classes) for _ in spec])
            self.register_buffer("mod_acc", torch.full((len(spec),), 1.0 / n_classes))
        self.aux_logits = None
        if fusion == "early":  # one Q-Former on the tokens of all modalities, k queries per modality
            assert epoch == "qformer", "early fusion: seqqformer only"
            self.joint = _JointQFormer(self.spec, d=d, k=k * len(self.spec), layers=layers, heads=heads,
                                       dropout=dropout)
        encs = []
        for _, dim, d_p, n_tok in (self.spec if fusion == "late" else ()):
            if epoch == "qformer":
                enc = QFormerStager(dim=dim, d_p=d_p, n_tok=n_tok, d=d, k=k, layers=layers, heads=heads, dropout=dropout)
                del enc.head, enc.ln
            else:
                enc = _LinearEmbed(dim=dim, d=d)
            encs.append(enc)
        self.encs = nn.ModuleList(encs)
        if fusion == "late":
            self.mod_emb = nn.Parameter(torch.randn(len(self.spec), d) * 0.02)
            self.fuse = nn.Linear(len(self.spec) * d, d)
        self.pos_t = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, seq_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.last_attn = None

    def forward(self, n, e, p):
        B, L = n.shape[:2]
        n, e, p = n.flatten(0, 1), e.flatten(0, 1), p.flatten(0, 1)
        if self.fusion == "early":
            h = self.joint.embed(n, e, p).view(B, L, -1)
            return self.head(self.ln(self.temporal(h + self.pos_t[:L])))
        hs, a, t = [], 0, 0
        for i, (enc, (_, dim, d_p, n_tok)) in enumerate(zip(self.encs, self.spec)):
            h = enc.embed(n[:, a:a + dim], e[:, a:a + dim], p[:, t:t + n_tok, :d_p])
            hs.append(h + self.mod_emb[i])
            a, t = a + dim, t + n_tok
        hs = torch.stack(hs, 1)  # (B * L, M, d)
        # last cross-attention of every modality's Q-Former, concatenated: (B * L, sum(2 + T_m))
        att = [enc.last_attn for enc in self.encs]
        self.last_attn = torch.cat(att, -1) if all(x is not None for x in att) else None
        if self.mod_dropout and self.training:
            self.aux_logits = torch.stack([head(hs[:, i].detach()) for i, head in enumerate(self.aux)], 1)
            hs = self.drop_modalities(hs, n.dtype)
        h = self.fuse(hs.flatten(1)).view(B, L, -1)
        return self.head(self.ln(self.temporal(h + self.pos_t[:L])))


    def zero_embeddings(self, dtype=torch.float32):
        """``(M, d)``: every modality's embedding of the all-zero input (= the training mean after
        :func:`standardize`), plus its modality embedding; the replacement of a dropped modality."""
        out = []
        for i, (enc, (_, dim, d_p, n_tok)) in enumerate(zip(self.encs, self.spec)):
            dev = self.mod_emb.device
            z = enc.embed(torch.zeros(1, dim, device=dev, dtype=dtype), torch.zeros(1, dim, device=dev, dtype=dtype),
                          torch.zeros(1, n_tok, d_p, device=dev, dtype=dtype))
            out.append(z[0] + self.mod_emb[i])
        return torch.stack(out)

    def drop_modalities(self, hs, dtype=torch.float32):
        """ProtoSleepNet's ZeroEmbeddingDropout: with probability ``mod_dropout`` an epoch loses
        ``n ~ U{1, ..., M-1}`` modalities, drawn without replacement with ``p_m = acc_m / sum acc``
        (the modalities the stager relies on most are dropped most often); their embeddings are
        replaced by the zero-input embeddings."""
        N, M, _ = hs.shape
        dev = hs.device
        prob = self.mod_acc.clamp_min(0)
        prob = prob / prob.sum() if prob.sum() > 0 else torch.full((M,), 1.0 / M, device=dev)
        apply = torch.rand(N, device=dev) < self.mod_dropout
        n_drop = torch.randint(1, M, (N,), device=dev)
        # weighted sampling without replacement, vectorized: Gumbel top-k on log p
        keys = prob.log()[None] - torch.log(-torch.log(torch.rand(N, M, device=dev).clamp_min(1e-12)))
        rank = keys.argsort(1, descending=True).argsort(1)  # rank of every modality per epoch
        mask = (rank < n_drop[:, None]) & apply[:, None]  # (N, M)
        return torch.where(mask[..., None], self.zero_embeddings(dtype)[None], hs)

    def aux_step(self, y: torch.Tensor) -> torch.Tensor:
        """Loss of the per-modality heads (detached inputs) and EMA update of ``mod_acc``."""
        if self.aux_logits is None:
            return torch.zeros((), device=y.device)
        lg, yy = self.aux_logits, y.reshape(-1)
        ok = yy >= 0
        loss = F.cross_entropy(lg[ok].flatten(0, 1), yy[ok].repeat_interleave(lg.shape[1]))
        with torch.no_grad():
            acc = (lg[ok].argmax(-1) == yy[ok, None]).float().mean(0)
            self.mod_acc.mul_(self.acc_momentum).add_((1 - self.acc_momentum) * acc)
        self.aux_logits = None
        return loss


def pack_modalities(splits_by_mod: Dict[str, Dict[str, Dict[str, np.ndarray]]], drop: Sequence[str] = (),
                    stats: Dict[str, Dict[str, tuple]] = None):
    """Standardize every modality on its own training statistics (:func:`standardize`), align
    them on ``(night, epoch)`` of the first modality (the reference: EEG) and pack them for
    :class:`MultiSeqStager`. An epoch missing in a modality (e.g. a night without EMG) gets zeros,
    i.e. the training mean. ``drop``: levels removed from the input, ``"eog:N"``, ``"emg:P"``...
    (set to zero in every split). Returns ``(data, spec, missing)``; ``missing[split][mod]`` counts
    the zero-filled epochs. ``stats``: ``{mod: latent_stats}`` stored with a trained stager, for
    splits without ``"train"`` (evaluation of new nights)."""
    mods = list(splits_by_mod)
    std = {m: standardize(splits_by_mod[m], (stats or {}).get(m)) for m in mods}
    ref = std[mods[0]]
    first = {m: next(iter(std[m].values())) for m in mods}
    dpmax = max(first[m]["P"].shape[-1] for m in mods)
    spec = [(m, first[m]["N"].shape[-1], first[m]["P"].shape[-1], first[m]["P"].shape[1]) for m in mods]
    gone = {tuple(x.split(":")) for x in drop}
    unknown = gone - {(m, lv) for m in mods for lv in ("N", "E", "P")}
    assert not unknown, f"unknown levels to drop: {sorted(unknown)}"
    data, missing = {}, {}
    for s in ref:
        key = ref[s]["night"] * 10 ** 6 + ref[s]["epoch"]
        M = len(key)
        cols = {"N": [], "E": [], "P": []}
        missing[s] = {}
        for m, dim, d_p, n_tok in spec:
            d = std[m][s]
            row = dict(zip((d["night"] * 10 ** 6 + d["epoch"]).tolist(), range(len(d["night"]))))
            idx = torch.tensor([row.get(k, -1) for k in key.tolist()])
            ok = idx >= 0
            missing[s][m] = int((~ok).sum())
            for lv in ("N", "E", "P"):
                src = d[lv]
                out = torch.zeros((M,) + tuple(src.shape[1:]))
                if (m, lv) not in gone:
                    out[ok] = src[idx[ok]]
                if lv == "P" and d_p < dpmax:
                    out = F.pad(out, (0, dpmax - d_p))
                cols[lv].append(out)
        data[s] = {"N": torch.cat(cols["N"], -1), "E": torch.cat(cols["E"], -1), "P": torch.cat(cols["P"], 1),
                   "y": ref[s]["y"], "night": ref[s]["night"], "epoch": ref[s]["epoch"]}
    return data, spec, missing


def _seq(epoch):
    return lambda **kw: SeqStager(epoch=epoch, **kw)


ARCHS = {"qformer": QFormerStager, "linear": LinearStager, "seqqformer": _seq("qformer"), "seqlinear": _seq("linear")}


# ----------------------------------------------------------------------------
# Data, training, metrics
# ----------------------------------------------------------------------------


def latent_stats(tr: Dict[str, np.ndarray]) -> Dict[str, tuple]:
    """``{level: (mean, std)}`` of the training latents, per dimension (P: per channel over all
    tokens); the statistics :func:`standardize` uses."""
    stats = {}
    for k in ("N", "E", "P"):
        a = tr[k].astype(np.float32).reshape(-1, tr[k].shape[-1])
        stats[k] = (a.mean(0), a.std(0) + 1e-6)
    return stats


def standardize(splits: Dict[str, Dict[str, np.ndarray]], stats: Dict[str, tuple] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per-dimension z-score with the training statistics (P: per channel over all tokens);
    ``stats`` (:func:`latent_stats`) replaces those of ``splits["train"]``."""
    stats = stats or latent_stats(splits["train"])
    out = {}
    for s, d in splits.items():
        out[s] = {k: torch.from_numpy((d[k].astype(np.float32) - stats[k][0]) / stats[k][1]) for k in ("N", "E", "P")}
        out[s]["y"] = torch.from_numpy(d["y"].astype(np.int64))
        out[s]["night"] = d["night"].astype(np.int64)
        if "epoch" in d:
            out[s]["epoch"] = d["epoch"].astype(np.int64)
    return out


# ----------------------------------------------------------------------------
# Latents of whole training splits: on disk, read per batch (07.11.2026)
# ----------------------------------------------------------------------------


class LatentWriter:
    """Appends the latents of one split, night by night, to ``<dir>/<split>_<key>.bin`` (raw, C
    order; shapes in ``<split>.json``) and, with ``stats``, accumulates the statistics of
    :func:`latent_stats` in float64. The split never sits in memory: the train split of the whole
    pool is ~13k nights, ~100 GB of EEG ``P`` in float16."""

    def __init__(self, d: str, split: str, stats: bool = False):
        os.makedirs(d, exist_ok=True)
        self.d, self.split, self.n = d, split, 0
        self.fh, self.shape, self.dtype = {}, {}, {}
        self.sums = {} if stats else None

    def add(self, **arrs):
        n = {len(v) for v in arrs.values()}
        assert len(n) == 1, f"rows differ: {n}"
        for k, v in arrs.items():
            v = np.ascontiguousarray(v)
            if k not in self.fh:
                self.fh[k] = open(os.path.join(self.d, f"{self.split}_{k}.bin"), "wb")
                self.shape[k], self.dtype[k] = tuple(v.shape[1:]), v.dtype.str
            assert tuple(v.shape[1:]) == self.shape[k] and v.dtype.str == self.dtype[k], k
            self.fh[k].write(v.tobytes())
            if self.sums is not None and k in ("N", "E", "P"):
                f = v.astype(np.float64).reshape(-1, v.shape[-1])
                acc = self.sums.setdefault(k, [0, 0.0, 0.0])
                acc[0] += len(f)
                acc[1] = acc[1] + f.sum(0)
                acc[2] = acc[2] + (f * f).sum(0)
        self.n += n.pop()

    def close(self):
        """Closes the files, writes ``<split>.json`` and returns the statistics (``None`` without)."""
        for f in self.fh.values():
            f.close()
        json.dump({"n": self.n, "shape": {k: list(v) for k, v in self.shape.items()}, "dtype": self.dtype},
                  open(os.path.join(self.d, f"{self.split}.json"), "w"))
        if self.sums is None:
            return None
        out = {}
        for k, (n, s1, s2) in self.sums.items():
            m = s1 / n
            out[k] = (m.astype(np.float32), (np.sqrt(np.maximum(s2 / n - m * m, 0.0)) + 1e-6).astype(np.float32))
        return out


def save_stats(path: str, st: Dict[str, tuple]):
    np.savez(path, **{f"{k}_{i}": v[i] for k, v in st.items() for i in (0, 1)})


def load_stats(path: str) -> Dict[str, tuple]:
    z = np.load(path)
    return {k: (z[f"{k}_0"], z[f"{k}_1"]) for k in "NEP"}


def has_split(d: str, split: str) -> bool:
    return os.path.exists(os.path.join(d, f"{split}.json")) or os.path.exists(os.path.join(d, f"latents_{split}.npz"))


def open_latents(d: str, split: str) -> Dict[str, np.ndarray]:
    """Latents of :class:`LatentWriter`, memory-mapped (``y``, ``night``, ``epoch`` in memory), or
    the ``latents_<split>.npz`` of :func:`extract_latents` (loaded)."""
    js = os.path.join(d, f"{split}.json")
    if not os.path.exists(js):
        return dict(np.load(os.path.join(d, f"latents_{split}.npz")))
    meta = json.load(open(js))
    out = {}
    for k, shp in meta["shape"].items():
        a = np.memmap(os.path.join(d, f"{split}_{k}.bin"), dtype=np.dtype(meta["dtype"][k]), mode="r",
                      shape=(meta["n"], *shp))
        out[k] = a if k in ("N", "E", "P") else np.array(a)
    return out


class LazyLevel:
    """One level (``N``, ``E`` or ``P``) of the stager input, read and standardized per batch, as
    :func:`standardize` + :func:`pack_modalities` build it in memory: per modality ``(arr, rows,
    mean, std, gone)``, sample ``i`` is row ``rows[i]`` of ``arr`` (``rows`` ``None`` = identity,
    ``-1`` = missing epoch -> zeros, i.e. the training mean; ``gone`` = level dropped -> zeros); the
    modalities are concatenated on the last axis (``N``, ``E``) or on the token axis (``P``, channels
    zero-padded to ``dpmax``). Indexing takes a slice or an integer array / tensor of any shape."""

    def __init__(self, parts, level: str, dpmax: int = 0):
        self.parts, self.level, self.dpmax = parts, level, dpmax
        arr, rows = parts[0][0], parts[0][1]
        self.n = len(arr) if rows is None else len(rows)
        self.shape = (self.n,) + self._row_shape()

    def _row_shape(self):
        shapes = [tuple(p[0].shape[1:]) for p in self.parts]
        if self.level == "P":
            dp = max(self.dpmax, max(s[-1] for s in shapes))
            return (sum(s[0] for s in shapes), dp)
        return shapes[0][:-1] + (sum(s[-1] for s in shapes),)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(self.n)[idx]
        elif isinstance(idx, torch.Tensor):
            idx = idx.numpy()
        idx = np.asarray(idx, dtype=np.int64)
        outs = []
        for arr, rows, mean, std, gone in self.parts:
            x = np.zeros(idx.shape + tuple(arr.shape[1:]), np.float32)
            if not gone:
                r = idx if rows is None else rows[idx]
                ok = r >= 0
                if ok.all():
                    x[...] = (arr[r].astype(np.float32) - mean) / std
                elif ok.any():
                    x[ok] = (arr[r[ok]].astype(np.float32) - mean) / std
            if self.level == "P" and x.shape[-1] < self.dpmax:
                x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.dpmax - x.shape[-1])])
            outs.append(x)
        x = outs[0] if len(outs) == 1 else np.concatenate(outs, -2 if self.level == "P" else -1)
        return torch.from_numpy(x)


def lazy_modalities(dirs: Dict[str, str], drop: Sequence[str] = (), stats: Dict[str, Dict[str, tuple]] = None):
    """:func:`pack_modalities` on the latents of :class:`LatentWriter` (or ``.npz``), without building
    the packed arrays: same alignment on ``(night, epoch)`` of the first modality, same
    standardization (``stats`` per modality; default ``<dir>/stats.npz``, else computed on the
    train split), same zero-filling and padding, read per batch by :class:`LazyLevel`. Returns
    ``(data, spec, missing)`` as :func:`pack_modalities`."""
    mods = list(dirs)
    raw = {m: {s: open_latents(dirs[m], s) for s in SPLITS if has_split(dirs[m], s)} for m in mods}
    st = {}
    for m in mods:
        f = os.path.join(dirs[m], "stats.npz")
        st[m] = (stats or {}).get(m) or (load_stats(f) if os.path.exists(f) else latent_stats(raw[m]["train"]))
    first = {m: next(iter(raw[m].values())) for m in mods}
    dpmax = max(first[m]["P"].shape[-1] for m in mods)
    spec = [(m, first[m]["N"].shape[-1], first[m]["P"].shape[-1], first[m]["P"].shape[1]) for m in mods]
    gone = {tuple(x.split(":")) for x in drop}
    unknown = gone - {(m, lv) for m in mods for lv in ("N", "E", "P")}
    assert not unknown, f"unknown levels to drop: {sorted(unknown)}"
    data, missing = {}, {}
    for s, ref in raw[mods[0]].items():
        key = ref["night"].astype(np.int64) * 10 ** 6 + ref["epoch"].astype(np.int64)
        rows, missing[s] = {}, {}
        for i, m in enumerate(mods):
            if i == 0:
                rows[m] = None
                missing[s][m] = 0
                continue
            d = raw[m][s]
            km = d["night"].astype(np.int64) * 10 ** 6 + d["epoch"].astype(np.int64)
            order = np.argsort(km, kind="stable")
            pos = np.minimum(np.searchsorted(km[order], key), len(km) - 1)
            hit = km[order][pos] == key
            rows[m] = np.where(hit, order[pos], -1).astype(np.int64)
            missing[s][m] = int((~hit).sum())
        data[s] = {lv: LazyLevel([(raw[m][s][lv], rows[m], st[m][lv][0], st[m][lv][1], (m, lv) in gone) for m in mods],
                                 lv, dpmax if len(mods) > 1 else 0)
                   for lv in ("N", "E", "P")}
        data[s]["y"] = torch.from_numpy(ref["y"].astype(np.int64))
        data[s]["night"] = ref["night"].astype(np.int64)
        data[s]["epoch"] = ref["epoch"].astype(np.int64)
    return data, spec, missing


def night_groups(d) -> List[np.ndarray]:
    """Sample indices of every night, in epoch order (the extracted epochs of a night; unscored
    epochs inside the main period are skipped, so a sequence is consecutive *scored* epochs)."""
    ep = d["epoch"] if "epoch" in d else np.arange(len(d["night"]))
    order = np.lexsort((ep, d["night"]))
    return np.split(order, np.flatnonzero(np.diff(d["night"][order])) + 1)


def train_windows(groups, L: int, rng: np.random.Generator) -> np.ndarray:
    """Non-overlapping sequences of ``L`` epochs with a random offset per night: ``(S, L)``."""
    out = []
    for g in groups:
        for s in range(int(rng.integers(0, L)), len(g) - L + 1, L):
            out.append(g[s:s + L])
    return np.stack(out)


def all_windows(groups, L: int) -> np.ndarray:
    """Every sequence of ``L`` epochs with stride 1, ``(S, L)``; every night needs >= ``L`` epochs."""
    assert all(len(g) >= L for g in groups), "a night is shorter than the sequence"
    return np.concatenate([np.lib.stride_tricks.sliding_window_view(g, L) for g in groups])


@torch.no_grad()
def predict(model, d, batch: int = 2048, device="cpu"):
    L = getattr(model, "seq_len", 1)
    if L > 1:
        return _predict_seq(model, d, L, batch, device)
    model.eval()
    logits, attn = [], []
    for i in range(0, len(d["y"]), batch):
        sl = slice(i, i + batch)
        logits.append(model(d["N"][sl].to(device), d["E"][sl].to(device), d["P"][sl].to(device)).float().cpu())
        if model.last_attn is not None:
            attn.append(model.last_attn.float().cpu())
    return torch.cat(logits), (torch.cat(attn) if attn else None)


@torch.no_grad()
def _predict_seq(model, d, L, batch, device):
    """Voting as ProtoSleepNet: every window of ``L`` epochs (stride 1), class probabilities
    averaged over the windows that contain an epoch. Returns log of the mean probabilities.
    Each night is read once and its windows are built on the device (with memory-mapped
    latents, :class:`LazyLevel`, the windows would otherwise read every epoch ``L`` times)."""
    model.eval()
    M = len(d["y"])
    prob, cnt, attn = torch.zeros(M, len(STAGES)), torch.zeros(M), None
    nb = max(1, batch // L)
    for g in night_groups(d):
        assert len(g) >= L, "a night is shorter than the sequence"
        g = torch.from_numpy(g)
        X = [d[k][g].to(device) for k in ("N", "E", "P")]
        W = torch.arange(len(g)).unfold(0, L, 1)
        for i in range(0, len(W), nb):
            w = W[i:i + nb]
            wd = w.to(device)
            out = model(*(x[wd] for x in X)).float().softmax(-1).cpu()
            gi = g[w].flatten()
            prob.index_add_(0, gi, out.flatten(0, 1))
            cnt.index_add_(0, gi, torch.ones(gi.numel()))
            if model.last_attn is not None:
                a = model.last_attn.float().cpu()
                attn = torch.zeros(M, a.shape[-1]) if attn is None else attn
                attn.index_add_(0, gi, a.flatten(0, 1) if a.dim() == 3 else a)
    return (prob / cnt[:, None]).clamp_min(1e-8).log(), (attn / cnt[:, None] if attn is not None else None)


def metrics(pred: np.ndarray, y: np.ndarray, night: np.ndarray) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

    per = [cohen_kappa_score(y[night == g], pred[night == g]) for g in np.unique(night)]
    return {"kappa_night_mean": float(np.mean(per)), "kappa_night_sd": float(np.std(per)),
            "kappa_pooled": float(cohen_kappa_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "kappa_per_night": [float(v) for v in per], "nights": [int(g) for g in np.unique(night)],
            "confusion": confusion_matrix(y, pred, labels=range(len(STAGES))).tolist()}


def attention_by_stage(attn: torch.Tensor, y: np.ndarray) -> dict:
    """Mean share of the last cross-attention on N, E, all of P and every P token, per true stage."""
    out = {}
    for c, name in enumerate(STAGES):
        a = attn[torch.from_numpy(y == c)]
        if len(a):
            m = a.mean(0).numpy()
            out[name] = {"N": float(m[0]), "E": float(m[1]), "P": float(m[2:].sum()), "P_tokens": m[2:].round(5).tolist()}
    return out


def train(model, data, epochs: int = 30, lr: float = 3e-4, batch: int = 512, weight_decay: float = 0.05,
          patience: int = 6, device="cpu", seed: int = 0, sched: str = "onecycle", evals_per_epoch: int = 1,
          log=print, prefetch_threads: int = 6):
    """AdamW; best state on the mean per-night kappa of ``data["valid"]``, evaluated
    ``evals_per_epoch`` times per epoch. ``sched``: ``"onecycle"`` (peak at 5%) or ``"cosine"``
    (2% linear warm-up, then cosine to 0 over all ``epochs``). ``patience`` counts evaluations
    without improvement (0 = no early stopping)."""
    import math

    torch.manual_seed(seed)
    model.to(device)
    tr = data["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # steps per epoch (seq models: sequences of L epochs, ~len/L per epoch with random offsets)
    L0 = getattr(model, "seq_len", 1)
    per_epoch = (len(tr["y"]) + batch - 1) // batch if L0 == 1 else \
        (sum(len(g) // L0 for g in night_groups(tr)) + max(1, batch // L0) - 1) // max(1, batch // L0)
    steps = epochs * per_epoch
    if sched == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    else:
        warm = max(1, int(0.02 * steps))
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))
    best, best_state, wait, hist = -1.0, None, 0, []
    g = torch.Generator().manual_seed(seed)
    va = data["valid"]
    L = getattr(model, "seq_len", 1)
    groups, rng = (night_groups(tr), np.random.default_rng(seed)) if L > 1 else (None, None)
    for ep in range(epochs):
        model.train()
        # seq models: ``batch`` counts epochs, i.e. batch // L sequences per step
        units = torch.from_numpy(train_windows(groups, L, rng)) if L > 1 else torch.arange(len(tr["y"]))
        perm, tot, seen = units[torch.randperm(len(units), generator=g)], 0.0, 0
        nb = max(1, batch // L)
        # batches read ahead by threads (same batches, same order): with latents on disk the
        # single-threaded read and standardization kept the GPU at ~60% (07.11.2026)
        starts = list(range(0, len(perm), nb))
        # evaluation points on the real number of steps: with a random offset per night the windows are
        # fewer than ``per_epoch`` and the end-of-epoch evaluation never ran (07.11.2026)
        eval_at = {round(len(starts) * (q + 1) / evals_per_epoch) for q in range(evals_per_epoch)}
        fetch = lambda i: [tr[k][perm[i:i + nb]] for k in ("N", "E", "P", "y")]  # noqa: E731
        pool = ThreadPoolExecutor(prefetch_threads)
        ahead = deque(pool.submit(fetch, i) for i in starts[:2 * prefetch_threads])
        for b, i in enumerate(starts, 1):
            idx = perm[i:i + nb]
            n_, e_, p_, y_ = ahead.popleft().result()
            if b - 1 + 2 * prefetch_threads < len(starts):
                ahead.append(pool.submit(fetch, starts[b - 1 + 2 * prefetch_threads]))
            logits = model(n_.to(device), e_.to(device), p_.to(device))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_.reshape(-1).to(device))
            if hasattr(model, "aux_step"):  # modality dropout: per-modality heads, accuracy estimates
                loss = loss + model.aux_step(tr["y"][idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot, seen = tot + loss.item() * idx.numel(), seen + idx.numel()
            if b not in eval_at:
                continue
            k = metrics(predict(model, va, batch=max(2048, 512 * L), device=device)[0].argmax(1).numpy(), va["y"].numpy(),
                        va["night"])["kappa_night_mean"]
            model.train()
            at = round(ep + b / len(starts), 3)
            hist.append({"epoch": at, "train_loss": tot / seen, "valid_kappa": k, "lr": opt.param_groups[0]["lr"]})
            # flush: with stdout on a file the lines stayed buffered until the end of the run (07.11.2026)
            log(f"[qformer_stager] epoch {at}/{epochs} loss {tot / seen:.4f} valid kappa {k:.4f}", flush=True)
            if k > best:
                best, wait = k, 0
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            else:
                wait += 1
        pool.shutdown(wait=True)
        if patience and wait >= patience:
            break
    model.load_state_dict(best_state)
    return hist


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _extract(a):
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run

    run, ds, model = _load_run(a.run, cache_dir=a.cache_dir, root=a.root)
    if a.vq:
        model.vq_levels = parse_vq_key(a.vq)
    elif hasattr(model, "latent_bits"):
        model.latent_bits = a.bits
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    meta = {"run": os.path.abspath(a.run), "vq": a.vq, "bits": None if a.vq else a.bits, "splits": {}}
    for s in SPLITS:
        d = extract_latents(model, ds, run["splits"][s], batch=a.batch)
        np.savez(os.path.join(a.out, f"latents_{s}.npz"), **d)
        meta["splits"][s] = {"nights": len(run["splits"][s]), "epochs": int(len(d["y"])),
                             "stage_counts": np.bincount(d["y"], minlength=len(STAGES)).tolist()}
        print(f"[qformer_stager] {s}: {len(d['y'])} epochs", flush=True)
    json.dump(meta, open(os.path.join(a.out, "latents.json"), "w"), indent=1)


def _load(d):
    return {s: dict(np.load(os.path.join(d, f"latents_{s}.npz"))) for s in SPLITS}


def _train(a):
    extra = {"seq_len": a.seq_len, "seq_layers": a.seq_layers} if a.arch.startswith("seq") else {}
    info = {}
    dirs = dict(x.split("=", 1) for x in a.mods) if a.mods else {"x": a.data}
    on_disk = all(os.path.exists(os.path.join(d, "train.json")) for d in dirs.values())  # LatentWriter
    if a.mods:  # multimodal: one latents directory per modality, the first is the reference
        assert a.arch.startswith("seq"), "multimodal stager: seqqformer or seqlinear"
        if on_disk:
            data, spec, missing = lazy_modalities(dirs, a.drop)
        else:
            data, spec, missing = pack_modalities({m: _load(d) for m, d in dirs.items()}, a.drop)
        info = {"spec": spec, "missing": missing, "drop": a.drop}
        print(f"[qformer_stager] modalities {spec}, zero-filled epochs {missing}", flush=True)
        torch.manual_seed(a.seed)  # the seed fixes the initial weights too
        model = MultiSeqStager(spec, epoch=a.arch[3:], d=a.d, k=a.k, layers=a.layers, heads=a.heads,
                               dropout=a.dropout, fusion=a.fusion, mod_dropout=a.mod_dropout, **extra)
    else:
        if on_disk:
            data, spec, _ = lazy_modalities(dirs)
            _, dim, d_p, n_tok = spec[0]
        else:
            raw = _load(a.data)
            data = standardize(raw)
            tr = raw["train"]
            dim, d_p, n_tok = tr["N"].shape[-1], tr["P"].shape[-1], tr["P"].shape[1]
        torch.manual_seed(a.seed)  # the seed fixes the initial weights too
        model = ARCHS[a.arch](dim=dim, d_p=d_p, n_tok=n_tok, d=a.d, k=a.k,
                              layers=a.layers, heads=a.heads, dropout=a.dropout, **extra)
    print(f"[qformer_stager] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, "
          f"{len(data['train']['y'])} train epochs, {len(np.unique(data['valid']['night']))} valid nights", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hist = train(model, data, epochs=a.epochs, lr=a.lr, batch=a.batch, patience=a.patience, device=device, seed=a.seed,
                 sched=a.sched, evals_per_epoch=a.evals_per_epoch)
    res = {"args": vars(a), "params": sum(p.numel() for p in model.parameters()), "history": hist, **info}
    for s in ("valid", "test"):  # test read once, with the checkpoint chosen on valid
        if s not in data:  # latents on disk: no copy of valid as test
            continue
        logits, attn = predict(model, data[s], device=device)
        y = data[s]["y"].numpy()
        res[s] = metrics(logits.argmax(1).numpy(), y, data[s]["night"])
        if attn is not None and not a.mods:  # multimodal attention: per modality, see stager_explain
            res[s]["attention_by_stage"] = attention_by_stage(attn, y)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "stager.json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(a.out, "stager.pt"))
    print(json.dumps({s: {k: round(res[s][k], 4) for k in ("kappa_night_mean", "kappa_pooled", "macro_f1")}
                      for s in ("valid", "test") if s in res}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run", required=True)
    e.add_argument("--vq", default=None, help="MSAE-VQ prefix, e.g. N8E8P6")
    e.add_argument("--bits", type=int, default=0, help="MSAE: bit depth of E and P (0 = continuous)")
    e.add_argument("--cache-dir", default=None)
    e.add_argument("--root", default=None)
    e.add_argument("--batch", type=int, default=128)
    e.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=None, help="latents directory (one modality; .npz or LatentWriter files)")
    t.add_argument("--mods", nargs="*", default=None,
                   help="multimodal: name=latents_dir per modality, the first is the reference (eeg=... eog=... emg=...)")
    t.add_argument("--drop", nargs="*", default=(), help="multimodal ablation: levels set to zero, e.g. eog:N emg:P")
    t.add_argument("--mod-dropout", type=float, default=0.0,
                   help="multimodal, late fusion: accuracy-weighted modality dropout as ProtoSleepNet (p_apply, PSN 0.5)")
    t.add_argument("--fusion", choices=("late", "early"), default="late",
                   help="multimodal: late = one Q-Former per modality, fused after pooling; early = one Q-Former "
                        "with k x modalities queries on the tokens of all modalities")
    t.add_argument("--arch", choices=sorted(ARCHS), default="qformer")
    t.add_argument("--d", type=int, default=128)
    t.add_argument("--k", type=int, default=8, help="learned queries")
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--seq-len", type=int, default=21, help="seq archs: epochs per sequence (ProtoSleepTransformer: 21)")
    t.add_argument("--seq-layers", type=int, default=2, help="seq archs: temporal transformer layers")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=6, help="evaluations without improvement (0 = off)")
    t.add_argument("--sched", choices=("onecycle", "cosine"), default="onecycle")
    t.add_argument("--evals-per-epoch", type=int, default=1)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train" and not (a.data or a.mods):
        ap.error("train: --data or --mods")
    (_extract if a.cmd == "extract" else _train)(a)


if __name__ == "__main__":
    main()
