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


def build_from_args(a: dict) -> EEGAR:
    return EEGAR(n_codes=a["n_codes"], levels=a["levels"], d_model=a["d_model"], layers=a["layers"],
                 heads=a["heads"], dropout=a.get("dropout", 0.0))


# ---------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_codes(model: EEGAR, codes: np.ndarray, window: int = 150, hop: int = 75, batch: int = 8,
                device=None) -> Tuple[np.ndarray, np.ndarray]:
    """``nll`` and entropy ``(P, L)`` of every code of a night, plain layout, sliding windows of ``window``
    patches: each patch is scored once, in the first window where it lies in the last ``hop`` patches
    (all patches of the first window), so after the first window it has at least ``window - hop``
    patches of context."""
    device = device or model.emb.weight.device
    P, L = codes.shape
    W = min(window, P)
    starts = list(range(0, P - W + 1, hop))
    if starts[-1] + W < P:
        starts.append(P - W)
    nll, ent = np.zeros((P, L), np.float32), np.zeros((P, L), np.float32)
    done = 0
    for i in range(0, len(starts), batch):
        ss = starts[i: i + batch]
        b = {k: v.to(device) for k, v in collate([make_example(codes[s: s + W], model.n_codes) for s in ss]).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            n, e = model.nll_entropy(b)
        n, e = n[:, : W * L].float().cpu().numpy(), e[:, : W * L].float().cpu().numpy()  # plain: target j = code j
        for k, s in enumerate(ss):
            lo = max(done, s)
            if lo < s + W:
                nll[lo: s + W] = n[k].reshape(W, L)[lo - s:]
                ent[lo: s + W] = e[k].reshape(W, L)[lo - s:]
                done = s + W
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
