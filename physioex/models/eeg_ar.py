"""EEGAR: next-token model on the codes of the 2 s patch codec (plan of 26.11.2026, xmai-lab/sleep-lens#17, S3 #20).

The codes of a night (:mod:`physioex.train.bin.patch_codec` ``dump``: ``(P, L)`` per patch grid) are read
as one flat sequence, level by level inside each patch: ``c_1^1 .. c_1^L, c_2^1 .. c_2^L, ...``. Every
(level, code) pair is its own token, so the vocabulary is ``N_SPECIAL + L * K``; the output of a position
is a softmax over the ``K`` codes of the level that comes next (that level is known). The model is a
decoder-only transformer (pre-LN, rotary positions, causal attention) with tied input and output tables.

**Surprise.** For each code the model gives ``nll = -log p(c | past)`` and the entropy ``H`` of the
predicted distribution. The surprise of a patch is ``S = sum_l (nll_l - H_l)`` over all levels: the
excess over what the model itself expected, so the noisy fine levels add little (plan rev. 1, point 1).
:func:`token_scores` averages the 4 patch grids shifted by 0.5 s onto the 0.5 s token grid.

**FIM** (fill in the middle, plan point 5 and S6). A fraction of the training windows is rearranged as
``[PRE] prefix [SUF] suffix [MID] middle``, cut at patch boundaries. Every token keeps the position it has
in the original window, and a sentinel takes the position just before the token it announces, so in
both layouts a position predicts the token at the next position, and the model knows how long the hole
is. :meth:`EEGAR.infill` samples the middle of a span with the prefix and suffix around it.

**PatchAR** (S3 rev. 2, codec 3 = transform codec rev. 4b: 8 block gains of 400 codes and 200 coefficients of
31 codes per patch). With 208 codes per patch the flat sequence (31,200 tokens for 5 min) is out of reach, so
:class:`PatchAR` runs over patches: a patch enters as the sum of the embeddings of its codes (one table per
level), and the output of a place gives one softmax per level for the patch at the next place, independent
given the past (factorized). Same sequence layouts at the patch level (plain and FIM, sentinels, positions),
same surprise ``S = sum_l (nll_l - H_l)`` over the 208 codes of a patch. Both models expose ``make_batch``,
``score_windows``, ``code_nll`` and ``infill``, which the scoring and editing functions below use.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, PRE, SUF, MID = range(5)
N_SPECIAL = 5
IGNORE = -100


# ---------------------------------------------------------------------------- sequences
def code_tokens(codes: np.ndarray, n_codes: int) -> np.ndarray:
    """``(P, L)`` codes -> ``(P * L,)`` token ids, level by level inside each patch."""
    P, L = codes.shape
    return (N_SPECIAL + np.arange(L)[None] * n_codes + codes.astype(np.int64)).reshape(-1)


def make_example(codes: np.ndarray, n_codes: int, span: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    """One training or scoring sequence from a window of ``W`` patches, ``(W, L)`` codes.

    ``span = (a, b)``: FIM layout with the middle on patches ``a .. b - 1`` (``1 <= a < b < W``);
    ``None``: plain layout ``[BOS] codes``. Returns arrays of length ``W * L + 3`` (padded):
    ``inp`` token ids, ``pos`` positions, ``tgt`` the code expected at the next position (index within
    its level) or ``IGNORE``, ``lev`` the level of that code (0 when ignored), ``where`` the index of that
    code in the flat window (-1 when ignored)."""
    W, L = codes.shape
    tok = code_tokens(codes, n_codes)
    c = codes.reshape(-1).astype(np.int64)
    T = W * L
    idx = np.arange(T)
    one = lambda v: np.array([v], np.int64)  # noqa: E731
    if span is None:
        seq = np.concatenate([one(BOS), tok])
        pos = np.concatenate([one(0), idx + 1])
        src = np.concatenate([one(-1), idx])  # flat index of the code at each place, -1 for a sentinel
    else:
        a, b = span
        assert 1 <= a < b < W, (a, b, W)
        pre, mid, suf = idx[: a * L], idx[a * L: b * L], idx[b * L:]
        seq = np.concatenate([one(PRE), tok[pre], one(SUF), tok[suf], one(MID), tok[mid]])
        pos = np.concatenate([one(0), pre + 1, one(b * L), suf + 1, one(a * L), mid + 1])
        src = np.concatenate([one(-1), pre, one(-1), suf, one(-1), mid])
    nxt = np.concatenate([src[1:], one(-1)])  # a sentinel is never a target: its place is known
    ok = nxt >= 0
    S, n = T + 3, len(seq)
    out = {"inp": np.full(S, PAD, np.int64), "pos": np.zeros(S, np.int64), "tgt": np.full(S, IGNORE, np.int64),
           "lev": np.zeros(S, np.int64), "where": np.full(S, -1, np.int64)}
    out["inp"][:n], out["pos"][:n] = seq, pos
    out["tgt"][:n] = np.where(ok, c[np.maximum(nxt, 0)], IGNORE)
    out["lev"][:n] = np.where(ok, nxt % L, 0)
    out["where"][:n] = nxt
    return out


def collate(examples) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([e[k] for e in examples])) for k in examples[0]}


# ---------------------------------------------------------------------------- model
def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4, dropout: float = 0.0):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, mlp * d, bias=False)
        self.fc2 = nn.Linear(mlp * d, d, bias=False)
        self.drop = dropout

    def forward(self, x, cos, sin):
        B, S, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, self.dh).unbind(2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(B, S, d))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class EEGAR(nn.Module):
    """Decoder-only transformer over level-specific code tokens (module docstring)."""

    def __init__(self, n_codes: int = 1024, levels: int = 8, d_model: int = 512, layers: int = 12, heads: int = 8,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.n_codes, self.levels, self.d = n_codes, levels, d_model
        self.emb = nn.Embedding(N_SPECIAL + levels * n_codes, d_model)
        self.lev_emb = nn.Embedding(levels, d_model)  # level of the code to predict
        self.blocks = nn.ModuleList([_Block(d_model, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        dh = d_model // heads
        self.register_buffer("inv_freq", rope_base ** (-torch.arange(0, dh, 2).float() / dh), persistent=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.lev_emb.weight, std=0.02)
        for blk in self.blocks:
            for lin in (blk.qkv, blk.fc1):
                nn.init.normal_(lin.weight, std=0.02)
            for lin in (blk.proj, blk.fc2):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * layers))

    def hidden(self, inp: torch.Tensor, pos: torch.Tensor, lev: torch.Tensor) -> torch.Tensor:
        x = self.emb(inp) + self.lev_emb(lev)
        ang = pos.float()[..., None] * self.inv_freq  # (B, S, dh / 2)
        cos, sin = ang.cos()[:, :, None].to(x.dtype), ang.sin()[:, :, None].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)

    def level_logits(self, h: torch.Tensor, lev: torch.Tensor):
        """``h (M, d)``, ``lev (M,)`` -> iterator of ``(mask, logits (m, K))`` per level present."""
        for l in torch.unique(lev).tolist():
            m = lev == l
            w = self.emb.weight[N_SPECIAL + l * self.n_codes: N_SPECIAL + (l + 1) * self.n_codes]
            yield m, h[m] @ w.T.to(h.dtype)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mean cross-entropy over the code targets, and its sum and count per level."""
        h = self.hidden(batch["inp"], batch["pos"], batch["lev"])
        keep = batch["tgt"] != IGNORE
        h, tgt, lev = h[keep], batch["tgt"][keep], batch["lev"][keep]
        tot = h.new_zeros((), dtype=torch.float32)
        per_sum = torch.zeros(self.levels, device=h.device)
        per_n = torch.zeros(self.levels, device=h.device)
        for m, logits in self.level_logits(h, lev):
            ce = F.cross_entropy(logits.float(), tgt[m], reduction="none")
            tot = tot + ce.sum()
            l = int(lev[m][0])
            per_sum[l] += ce.detach().sum()
            per_n[l] += m.sum()
        return tot / keep.sum().clamp_min(1), {"sum": per_sum, "n": per_n}

    @torch.no_grad()
    def nll_entropy(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-target ``nll`` and entropy (nats), ``(B, S)`` each (0 where ignored)."""
        h = self.hidden(batch["inp"], batch["pos"], batch["lev"])
        keep = batch["tgt"] != IGNORE
        nll = torch.zeros(keep.shape, device=h.device)
        ent = torch.zeros(keep.shape, device=h.device)
        hk, tgt, lev = h[keep], batch["tgt"][keep], batch["lev"][keep]
        n_k, e_k = torch.zeros(len(hk), device=h.device), torch.zeros(len(hk), device=h.device)
        for m, logits in self.level_logits(hk, lev):
            lp = torch.log_softmax(logits.float(), -1)
            n_k[m] = -lp.gather(1, tgt[m][:, None])[:, 0]
            e_k[m] = -(lp.exp() * lp).sum(-1)
        nll[keep], ent[keep] = n_k, e_k
        return nll, ent

    def make_batch(self, items) -> Dict[str, torch.Tensor]:
        """``items``: ``[(codes (W, L), span or None)]`` -> batch of :func:`make_example` sequences."""
        return collate([make_example(np.asarray(c), self.n_codes, sp) for c, sp in items])

    @torch.no_grad()
    def score_windows(self, windows, device=None) -> Tuple[np.ndarray, np.ndarray]:
        """Plain layout: ``nll`` and entropy ``(B, W, L)`` of every code of windows of equal length."""
        device = device or self.emb.weight.device
        W, L = np.asarray(windows[0]).shape
        b = {k: v.to(device) for k, v in self.make_batch([(c, None) for c in windows]).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            n, e = self.nll_entropy(b)
        n, e = n[:, : W * L].float().cpu().numpy(), e[:, : W * L].float().cpu().numpy()  # plain: target j = code j
        return n.reshape(-1, W, L), e.reshape(-1, W, L)

    @torch.no_grad()
    def code_nll(self, batch: Dict[str, torch.Tensor]):
        """``nll``, level and part (0 plain, 1 FIM middle, 2 rest of a FIM sequence) of every target code."""
        nll, _ = self.nll_entropy(batch)
        keep = batch["tgt"] != IGNORE
        part = _layout_part(batch["inp"][:, 0] == BOS, batch["inp"] == MID)
        return nll[keep], batch["lev"][keep], part[keep]

    @torch.no_grad()
    def infill(self, codes: torch.Tensor, a: int, b: int, n_samples: int = 8, temperature: float = 1.0,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample the codes of patches ``a .. b - 1`` of a window ``codes (W, L)`` given the rest (FIM).
        Returns ``(n_samples, b - a, L)``. One full forward per sampled code (no cache): spans are short."""
        W, L = codes.shape
        dev = self.emb.weight.device
        ex = make_example(codes.cpu().numpy(), self.n_codes, (a, b))
        n_mid = (b - a) * L
        start = W * L + 3 - n_mid  # index of the first middle token in the FIM layout
        inp = torch.from_numpy(ex["inp"]).to(dev).repeat(n_samples, 1)
        pos = torch.from_numpy(ex["pos"]).to(dev).repeat(n_samples, 1)
        lev = torch.from_numpy(ex["lev"]).to(dev).repeat(n_samples, 1)
        out = torch.empty(n_samples, n_mid, dtype=torch.long, device=dev)
        for t in range(n_mid):
            j = start - 1 + t  # position whose target is middle token t
            h = self.hidden(inp[:, : j + 1], pos[:, : j + 1], lev[:, : j + 1])[:, j]
            l = (a * L + t) % L
            w = self.emb.weight[N_SPECIAL + l * self.n_codes: N_SPECIAL + (l + 1) * self.n_codes]
            p = torch.softmax((h @ w.T.to(h.dtype)).float() / temperature, -1)
            c = torch.multinomial(p, 1, generator=generator)[:, 0]
            out[:, t] = c
            if t + 1 < n_mid:
                inp[:, j + 1] = N_SPECIAL + l * self.n_codes + c
        return out.view(n_samples, b - a, L)


def _layout_part(plain: torch.Tensor, is_mid: torch.Tensor) -> torch.Tensor:
    """``plain (B,)``, ``is_mid (B, S)`` (the MID sentinel) -> ``(B, S)``: 0 in a plain sequence, 1 from the
    MID sentinel on (its targets are the middle), 2 before it."""
    S = is_mid.shape[1]
    at = torch.where(is_mid.any(1), is_mid.float().argmax(1), torch.full_like(plain, S, dtype=torch.long))
    part = torch.where(torch.arange(S, device=is_mid.device)[None] >= at[:, None], 1, 2)
    return torch.where(plain[:, None], 0, part)


# ---------------------------------------------------------------------------- patch-level model (S3 rev. 2)
P_PAD, P_BOS, P_PRE, P_SUF, P_MID, P_PATCH = range(6)


def patch_example(W: int, span: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    """Index arrays of one :class:`PatchAR` sequence over a window of ``W`` patches (the codes travel apart).

    ``span = (a, b)``: FIM layout ``[PRE] 0 .. a-1 [SUF] b .. W-1 [MID] a .. b-1`` (``1 <= a < b < W``);
    ``None``: ``[BOS] 0 .. W-1``. Length ``W + 3`` (padded with ``P_PAD``). ``kind``: ``P_PATCH`` or the
    sentinel at each place; ``src``: the window patch read at each place (0 at a sentinel, unused); ``pos``:
    rotary position (a patch keeps its place in the window, a sentinel takes the place just before the patch
    it announces, as in :func:`make_example`); ``nxt``: the window patch expected at the next place, or -1."""
    idx = np.arange(W)
    if span is None:
        kind = np.r_[P_BOS, np.full(W, P_PATCH)]
        src, pos = np.r_[-1, idx], np.r_[0, idx + 1]
    else:
        a, b = span
        assert 1 <= a < b < W, (a, b, W)
        kind = np.r_[P_PRE, np.full(a, P_PATCH), P_SUF, np.full(W - b, P_PATCH), P_MID, np.full(b - a, P_PATCH)]
        src = np.r_[-1, idx[:a], -1, idx[b:], -1, idx[a:b]]
        pos = np.r_[0, idx[:a] + 1, b, idx[b:] + 1, a, idx[a:b] + 1]
    nxt = np.r_[src[1:], -1]  # a sentinel is never a target: its place is known
    n, S = len(kind), W + 3
    out = {"kind": np.full(S, P_PAD, np.int64), "src": np.zeros(S, np.int64), "pos": np.zeros(S, np.int64),
           "nxt": np.full(S, -1, np.int64)}
    out["kind"][:n], out["src"][:n], out["pos"][:n], out["nxt"][:n] = kind, np.maximum(src, 0), pos, nxt
    return out


class PatchAR(nn.Module):
    """Decoder-only transformer over patches, for codes with one vocabulary per level (module docstring)."""

    def __init__(self, sizes, d_model: int = 512, layers: int = 12, heads: int = 8, dropout: float = 0.0,
                 rope_base: float = 10000.0):
        super().__init__()
        self.sizes = [int(v) for v in sizes]
        self.levels, self.n_codes, self.d = len(self.sizes), max(self.sizes), d_model
        self.register_buffer("offsets", torch.as_tensor(np.r_[0, np.cumsum(self.sizes)[:-1]], dtype=torch.long),
                             persistent=False)
        runs = []  # (first level, levels, vocabulary) of each run of equal vocabularies
        for l, v in enumerate(self.sizes):
            if runs and runs[-1][2] == v:
                runs[-1][1] += 1
            else:
                runs.append([l, 1, v])
        self.runs = [tuple(r) for r in runs]
        self.emb = nn.Embedding(sum(self.sizes), d_model)  # one row per (level, code)
        self.special = nn.Embedding(P_PATCH, d_model)  # PAD, BOS, PRE, SUF, MID
        self.blocks = nn.ModuleList([_Block(d_model, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, sum(self.sizes))
        dh = d_model // heads
        self.register_buffer("inv_freq", rope_base ** (-torch.arange(0, dh, 2).float() / dh), persistent=False)
        nn.init.normal_(self.emb.weight, std=0.02 / math.sqrt(self.levels))  # a patch is a sum of L rows
        nn.init.normal_(self.special.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        for blk in self.blocks:
            for lin in (blk.qkv, blk.fc1):
                nn.init.normal_(lin.weight, std=0.02)
            for lin in (blk.proj, blk.fc2):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * layers))

    def hidden(self, codes: torch.Tensor, kind: torch.Tensor, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``codes (B, W, L)``, ``kind``/``src``/``pos (B, S)`` -> ``(B, S, d)``."""
        B, S = kind.shape
        c = codes.gather(1, src[..., None].expand(B, S, self.levels)).long() + self.offsets
        x = F.embedding_bag(c.reshape(B * S, self.levels), self.emb.weight, mode="sum").view(B, S, self.d)
        x = torch.where((kind == P_PATCH)[..., None], x, self.special(kind.clamp_max(P_PATCH - 1)))
        ang = pos.float()[..., None] * self.inv_freq
        cos, sin = ang.cos()[:, :, None].to(x.dtype), ang.sin()[:, :, None].to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.norm(x)

    def log_probs(self, h: torch.Tensor):
        """``h (..., d)`` -> one ``(..., levels of the run, vocabulary)`` log-softmax per run of ``runs``."""
        z = self.head(h).float()
        out, k = [], 0
        for _, n, v in self.runs:
            out.append(torch.log_softmax(z[..., k:k + n * v].reshape(*z.shape[:-1], n, v), -1))
            k += n * v
        return out

    def _nll(self, lps, tgt: torch.Tensor) -> torch.Tensor:
        """Log-probabilities per run and targets ``(M, L)`` -> ``nll (M, L)``."""
        return torch.cat([-lp.gather(-1, tgt[:, l0:l0 + n, None].long())[..., 0]
                          for (l0, n, _), lp in zip(self.runs, lps)], 1)

    def _targets(self, batch: Dict[str, torch.Tensor]):
        B, S = batch["kind"].shape
        keep = batch["nxt"] >= 0
        tgt = batch["codes"].gather(1, batch["nxt"].clamp_min(0)[..., None].expand(B, S, self.levels))
        return keep, tgt[keep]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mean cross-entropy over the target codes, and its sum per level and the number of target patches."""
        h = self.hidden(batch["codes"], batch["kind"], batch["src"], batch["pos"])
        keep, tgt = self._targets(batch)
        nll = self._nll(self.log_probs(h[keep]), tgt)
        return nll.mean(), {"sum": nll.detach().sum(0), "n": keep.sum()}

    @torch.no_grad()
    def nll_entropy(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-code ``nll`` and entropy (nats), ``(B, S, L)`` each (0 at places without a target)."""
        h = self.hidden(batch["codes"], batch["kind"], batch["src"], batch["pos"])
        keep, tgt = self._targets(batch)
        B, S = keep.shape
        nll = torch.zeros(B, S, self.levels, device=h.device)
        ent = torch.zeros(B, S, self.levels, device=h.device)
        lps = self.log_probs(h[keep])
        nll[keep] = self._nll(lps, tgt)
        ent[keep] = torch.cat([-(lp.exp() * lp).sum(-1) for lp in lps], 1)
        return nll, ent

    def make_batch(self, items) -> Dict[str, torch.Tensor]:
        """``items``: ``[(codes (W, L), span or None)]``, one ``W`` -> ``codes (B, W, L)`` and the index arrays."""
        out = collate([patch_example(len(c), sp) for c, sp in items])
        out["codes"] = torch.from_numpy(np.stack([np.asarray(c) for c, _ in items]).astype(np.int16))
        return out

    @torch.no_grad()
    def score_windows(self, windows, device=None) -> Tuple[np.ndarray, np.ndarray]:
        """Plain layout: ``nll`` and entropy ``(B, W, L)`` of every code of windows of equal length."""
        device = device or self.emb.weight.device
        W = len(windows[0])
        b = {k: v.to(device) for k, v in self.make_batch([(c, None) for c in windows]).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            n, e = self.nll_entropy(b)
        return n[:, :W].float().cpu().numpy(), e[:, :W].float().cpu().numpy()  # plain: place k targets patch k

    @torch.no_grad()
    def patch_states(self, windows, device=None) -> np.ndarray:
        """Plain layout: last-layer state ``(B, W, d)`` of the place that reads each patch (place ``k + 1`` reads
        patch ``k``), for windows of equal length: the AR's feature of a patch and its past (S5 type, codec 3)."""
        device = device or self.emb.weight.device
        W = len(windows[0])
        b = {k: v.to(device) for k, v in self.make_batch([(c, None) for c in windows]).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h = self.hidden(b["codes"], b["kind"], b["src"], b["pos"])
        return h[:, 1:W + 1].float().cpu().numpy()

    @torch.no_grad()
    def code_nll(self, batch: Dict[str, torch.Tensor]):
        """``nll``, level and part (0 plain, 1 FIM middle, 2 rest of a FIM sequence) of every target code."""
        nll, _ = self.nll_entropy(batch)
        keep = batch["nxt"] >= 0
        part = _layout_part(batch["kind"][:, 0] == P_BOS, batch["kind"] == P_MID)[keep]
        lev = torch.arange(self.levels, device=nll.device).expand(len(part), self.levels)
        return nll[keep].reshape(-1), lev.reshape(-1), part[:, None].expand(-1, self.levels).reshape(-1)

    @torch.no_grad()
    def infill(self, codes: torch.Tensor, a: int, b: int, n_samples: int = 8, temperature: float = 1.0,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample patches ``a .. b - 1`` of a window ``codes (W, L)`` given the rest (FIM), every code of a patch
        from its own softmax. Returns ``(n_samples, b - a, L)``. One forward per sampled patch (no cache)."""
        W, L = codes.shape
        dev = self.emb.weight.device
        ex = {k: v.to(dev).repeat(n_samples, 1) for k, v in collate([patch_example(W, (a, b))]).items()}
        cw = codes.to(dev).long()[None].repeat(n_samples, 1, 1)
        start = W + 3 - (b - a)  # place of the first middle patch in the FIM layout
        for t in range(b - a):
            j = start - 1 + t  # place whose target is middle patch a + t (the MID sentinel for t = 0)
            h = self.hidden(cw, ex["kind"][:, : j + 1], ex["src"][:, : j + 1], ex["pos"][:, : j + 1])[:, j]
            new = []
            for lp in self.log_probs(h):  # (n_samples, levels of the run, vocabulary)
                p = torch.softmax(lp / temperature, -1)
                new.append(torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).view(p.shape[:-1]))
            cw[:, a + t] = torch.cat(new, 1)
        return cw[:, a:b]


def build_from_args(a: dict):
    if a.get("model") == "patch_ar":
        return PatchAR(sizes=a["sizes"], d_model=a["d_model"], layers=a["layers"], heads=a["heads"],
                       dropout=a.get("dropout", 0.0))
    return EEGAR(n_codes=a["n_codes"], levels=a["levels"], d_model=a["d_model"], layers=a["layers"],
                 heads=a["heads"], dropout=a.get("dropout", 0.0))


def _device(model) -> torch.device:
    return next(model.parameters()).device


# ---------------------------------------------------------------------------- scoring
def _slide(fn, codes: np.ndarray, window: int, hop: int, batch: int):
    """Per-patch outputs of a whole night from sliding windows of ``min(window, P)`` patches: ``fn(windows)``
    gives arrays ``(B, W, ...)``; each patch is taken once, from the first window where it lies in the last
    ``hop`` patches (all patches of the first window)."""
    P = len(codes)
    W = min(window, P)
    starts = list(range(0, P - W + 1, hop))
    if starts[-1] + W < P:
        starts.append(P - W)
    outs, done = None, 0
    for i in range(0, len(starts), batch):
        ss = starts[i: i + batch]
        res = fn([codes[s: s + W] for s in ss])
        if outs is None:
            outs = [np.zeros((P,) + r.shape[2:], np.float32) for r in res]
        for k, s in enumerate(ss):
            lo = max(done, s)
            if lo < s + W:
                for o, r in zip(outs, res):
                    o[lo: s + W] = r[k][lo - s:]
                done = s + W
    return outs


@torch.no_grad()
def hidden_codes(model, codes: np.ndarray, window: int = 150, hop: int = 75, batch: int = 8, device=None) -> np.ndarray:
    """``(P, d)`` state of the patch model on every patch of a night (:meth:`PatchAR.patch_states`), with the
    windows of :func:`score_codes`."""
    device = device or _device(model)
    return _slide(lambda ws: (model.patch_states(ws, device),), codes, window, hop, batch)[0]


@torch.no_grad()
def score_codes(model, codes: np.ndarray, window: int = 150, hop: int = 75, batch: int = 8,
                device=None) -> Tuple[np.ndarray, np.ndarray]:
    """``nll`` and entropy ``(P, L)`` of every code of a night, plain layout, sliding windows of ``window``
    patches: each patch is scored once, in the first window where it lies in the last ``hop`` patches
    (all patches of the first window), so after the first window it has at least ``window - hop``
    patches of context."""
    device = device or _device(model)
    nll, ent = _slide(lambda ws: model.score_windows(ws, device), codes, window, hop, batch)
    return nll, ent


def token_scores(per_grid: Dict[int, np.ndarray], n_samples: int, patch: int = 200, tok: int = 50) -> np.ndarray:
    """Mean over the patch grids of a per-patch score, on the 0.5 s token grid.

    ``per_grid[o]``: ``(P_o,)`` scores of the patches of the grid shifted by ``o`` samples (patch ``p``
    covers samples ``o + patch * p`` to ``o + patch * (p + 1)``). Returns ``(n_samples // tok,)``; a token
    covered by no patch is NaN."""
    n_tok = n_samples // tok
    acc, cnt = np.zeros(n_tok), np.zeros(n_tok)
    per = patch // tok
    for o, s in per_grid.items():
        idx = o // tok + per * np.arange(len(s))
        for k in range(per):
            j = idx + k
            ok = j < n_tok
            np.add.at(acc, j[ok], s[ok])
            np.add.at(cnt, j[ok], 1)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------------------------------- edits (S6, #23)
@torch.no_grad()
def window_surprise(model, items, batch: int = 16, device=None):
    """Surprise ``S = sum_l (nll - H)`` per patch of short stretches, plain layout, one window each.

    ``items``: list of ``(codes (W, L), lo, hi)``; returns one ``(hi - lo,)`` array per item, the patches
    ``lo .. hi - 1`` of its window scored with the window's earlier patches as context. Windows of the
    same length are batched."""
    device = device or _device(model)
    out = [None] * len(items)
    by_len: Dict[int, list] = {}
    for i, (c, lo, hi) in enumerate(items):
        by_len.setdefault(len(c), []).append(i)
    for _, idx in by_len.items():
        for j in range(0, len(idx), batch):
            ii = idx[j: j + batch]
            n, e = model.score_windows([np.asarray(items[i][0]) for i in ii], device)
            s = (n - e).sum(-1)
            for k, i in enumerate(ii):
                _, lo, hi = items[i]
                out[i] = s[k][lo:hi]
    return out


def window_around(P: int, lo: int, hi: int, window: int, centre: bool = True) -> int:
    """Start of a window of ``min(window, P)`` patches that contains ``lo .. hi - 1``: centred on the
    stretch (``centre``) or ending at ``hi`` (plain scoring needs only the past), clipped to the night."""
    W = min(window, P)
    s = lo - (W - (hi - lo)) // 2 if centre else hi - W
    return int(min(max(s, 0), P - W))


@torch.no_grad()
def close_event(model, codes: np.ndarray, a: int, thr: float, max_len: int = 10, follow: int = 3,
                window: int = 150, temperature: float = 1.0, generator: Optional[torch.Generator] = None):
    """Closure rule of the plan (rev. 1, point 5). From the start patch ``a`` the end ``b`` grows up to
    ``a + max_len``; for each ``b`` the stretch ``a .. b - 1`` is refilled by FIM (one sample) and the
    ``follow`` patches after it are scored in the plain layout with the refilled stretch as their past.
    The end is the first ``b`` whose ``follow`` patches have a mean surprise ``<= thr``. Returns
    ``(b, fill (b - a, L))``, or ``(None, None)`` when no ``b`` works: a state transition, not an event."""
    P, L = codes.shape
    dev = _device(model)
    for b in range(a + 1, min(a + max_len, P - follow) + 1):
        s = window_around(P, a, b, window)
        W = min(window, P)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            fill = model.infill(torch.from_numpy(np.ascontiguousarray(codes[s: s + W])), a - s, b - s, 1,
                                temperature, generator)[0].cpu().numpy()
        new = codes.copy()
        new[a:b] = fill
        s2 = window_around(P, b, b + follow, window, centre=False)
        S = window_surprise(model, [(new[s2: s2 + W], b - s2, b + follow - s2)])[0]
        if float(S.mean()) <= thr:
            return b, fill
    return None, None
