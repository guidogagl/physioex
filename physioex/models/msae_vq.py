"""MSAE-VQ: the MSAE hierarchy N -> E -> P with a residual vector quantizer on every level.

The continuous :class:`~physioex.models.msae.MSAE` stays as it is (encoder, FiLM conditioning
N -> E and [N; E] -> P, decoder, texture, spectral heads); each latent goes through a
:class:`ResidualVQ` and the conditioning always uses the *quantized* latents:

* **N** ``(128,)``: computed from the 40 neighbours as in MSAE, then RVQ_N (``K_N`` codes, up to 4
  levels). It is transmitted, so its bits count; the code is a discrete context state.
* **E** ``(128,)``: FiLM on ``N_q``, ``tanh``, RVQ_E (``K`` codes, up to 8 levels).
* **P** ``(T_p, d_p)``: FiLM on ``[N_q; E_q]``, ``tanh``, RVQ_P per token (codebooks shared
  across the tokens, up to 8 levels).

Training uses quantizer dropout (SoundStream, Zeghidour et al. 2021): with probability
``q_dropout`` each level keeps only its first ``k ~ U{1..L}`` quantizers, so one model serves
every prefix of levels. In eval the prefix is ``vq_levels`` (``{"N": a, "E": b, "P": c}``).
Codebooks: EMA updates, k-means initialisation (:func:`fit_vq_codebooks`), dead-code restart
(Jukebox, Dhariwal et al. 2020), commitment 0.25 (:class:`~physioex.models.msvq.EMAVQ`).

Discrete tokens: :meth:`MSAEVQ.codes` gives the code indices, :meth:`MSAEVQ.embed` maps
(prefixes of) codes back to latents, :meth:`MSAEVQ.bits_per_epoch` counts
``L_N log2 K_N + L_E log2 K + T_p L_P log2 K``.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from physioex.models.msae import MSAE, MSAETrainer
from physioex.models.msvq import EMAVQ, code_stats

SCALES = ("N", "E", "P")


def _kmeans(z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    if len(z) < k:  # too few vectors: duplicates with a little noise
        z = z[np.random.default_rng(seed).integers(0, len(z), k)] + 1e-3 * np.random.default_rng(seed).standard_normal((k, z.shape[1]))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=max(4096, 2 * k), max_iter=50, n_init=1)
    return km.fit(z.astype(np.float32)).cluster_centers_.astype(np.float32)


class ResidualVQ(nn.Module):
    """``n_levels`` :class:`EMAVQ` in series, each on the residual of the previous ones.

    ``forward(z, k)`` quantizes ``(M, dim)`` with the first ``k`` levels and returns
    ``(z_q, codes (M, k), commitment loss)``; ``z_q`` carries the straight-through gradient.
    """

    def __init__(self, n_levels: int, n_codes: int, dim: int, commitment_weight: float = 0.25,
                 restart_after: int = 30):
        super().__init__()
        self.n_levels, self.n_codes, self.dim, self.restart_base = n_levels, n_codes, dim, restart_after
        self.levels = nn.ModuleList([EMAVQ(n_codes, dim, commitment_weight, restart_after=restart_after)
                                     for _ in range(n_levels)])

    def forward(self, z: torch.Tensor, k: Optional[int] = None):
        with torch.autocast(z.device.type, enabled=False):  # distances and argmin in float32
            return self._forward(z.float(), self.n_levels if k is None else k)

    def _forward(self, z: torch.Tensor, k: int):
        r, q_sum, loss, codes = z, torch.zeros_like(z), z.new_zeros(()), []
        for vq in self.levels[:k]:
            # a code is dead after ~4 expected uses without a hit (K / vectors per step grows with K)
            vq.restart_after = max(self.restart_base, int(4 * self.n_codes / max(1, len(z))))
            q, l_ = vq(r)
            q = q.detach()  # value of the code (EMAVQ returns r + (q - r).detach())
            loss = loss + l_
            codes.append(vq.last_idx)
            q_sum = q_sum + q
            r = r - q
        return z + (q_sum - z).detach(), torch.stack(codes, 1), loss

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """``(M, k)`` indices -> ``(M, dim)`` sum of the first ``k`` codewords."""
        return sum(self.levels[l].codebook[codes[:, l]] for l in range(codes.shape[1]))

    @torch.no_grad()
    def kmeans_init(self, z: torch.Tensor, seed: int = 0) -> None:
        """k-means on ``z``, then on each residual, level by level."""
        r = z.detach().float()
        for l, vq in enumerate(self.levels):
            cb = torch.from_numpy(_kmeans(r.cpu().numpy(), self.n_codes, seed + l)).to(r.device)
            vq.codebook.copy_(cb)
            vq.ema_sum.copy_(cb)
            vq.ema_count.fill_(1.0)
            vq.initialised.fill_(True)
            d = r.pow(2).sum(1, keepdim=True) + cb.pow(2).sum(1)[None] - 2 * r @ cb.T
            r = r - cb[d.argmin(1)]


class MSAEVQ(MSAE):
    """:class:`MSAE` with a :class:`ResidualVQ` on N, E and P (see module docstring)."""

    def __init__(self, *args, codes: int = 1024, codes_n: int = 256, levels: Sequence[int] = (4, 8, 8),
                 q_dropout: float = 0.5, restart_after: int = 30, **kw):
        kw["latent_bits"] = 0  # tanh only: the RVQ is the channel
        super().__init__(*args, **kw)
        self.n_codes = {"N": codes_n, "E": codes, "P": codes}
        self.max_levels = dict(zip(SCALES, levels))
        self.vq_levels = dict(self.max_levels)  # prefix used in eval
        self.q_dropout = q_dropout
        self.rvq = nn.ModuleDict({
            "N": ResidualVQ(levels[0], codes_n, self.dim, restart_after=restart_after),
            "E": ResidualVQ(levels[1], codes, self.dim, restart_after=restart_after),
            "P": ResidualVQ(levels[2], codes, self.d_p, restart_after=restart_after)})
        self.bypass_vq = False  # continuous latents (for fit_vq_codebooks)
        self.last_codes: Dict[str, torch.Tensor] = {}
        self.last_vq_loss = torch.zeros(())

    def _k(self, s: str) -> int:
        """Levels used now: a random prefix in training (quantizer dropout), ``vq_levels`` in eval."""
        if self.training and torch.rand(()) < self.q_dropout:
            return int(torch.randint(1, self.max_levels[s] + 1, ()))
        return self.vq_levels[s] if not self.training else self.max_levels[s]

    def _q(self, s: str, z: torch.Tensor):
        if self.bypass_vq:
            return z.float(), None, z.new_zeros(())
        shape = z.shape
        zq, codes, loss = self.rvq[s](z.reshape(-1, shape[-1]), self._k(s))
        return zq.reshape(shape), codes.reshape(*shape[:-1], -1), loss

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``(B, L, 1, 3000) -> {"N", "E", "P"}`` quantized latents (MSAE shapes) and ``"codes"``:
        ``{"N": (B, k_N), "E": (B, k_E), "P": (B, T_p, k_P)}``."""
        x = x[:, :, 0].float()
        n, cn, ln = self._q("N", self.encode_context(x))
        h = self.features(x[:, self.c])
        g, b = self.e_film(n).chunk(2, -1)
        he = h * (1 + g[:, None]) + b[:, None]
        e = torch.tanh(self.e_out(self.e_pool(self.e_query.expand(len(h), -1, -1), he, he, need_weights=False)[0][:, 0]))
        e, ce, le = self._q("E", e)
        g, b = self.p_film(torch.cat([n, e], -1)).chunk(2, -1)
        p = torch.tanh(self.p_out(h * (1 + g[:, None]) + b[:, None]))
        p, cp, lp = self._q("P", p)
        self.last_codes = {"N": cn, "E": ce, "P": cp}
        self.last_vq_loss = ln + le + lp
        return {"N": n, "E": e, "P": p, "codes": self.last_codes}

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Code indices with the eval prefix ``vq_levels``."""
        was = self.training
        self.eval()
        c = self.encode(x)["codes"]
        self.train(was)
        return c

    def embed(self, codes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Codes (any prefix of levels) -> quantized latents ``{"N", "E", "P"}``."""
        out = {s: self.rvq[s].lookup(codes[s]) for s in ("N", "E")}
        cp = codes["P"]
        out["P"] = self.rvq["P"].lookup(cp.reshape(-1, cp.shape[-1])).reshape(*cp.shape[:-1], self.d_p)
        return out

    def bits_per_epoch(self, levels: Optional[Dict[str, int]] = None) -> int:
        lv = levels or self.vq_levels
        b = {s: math.log2(self.n_codes[s]) for s in SCALES}
        return int(round(lv["N"] * b["N"] + lv["E"] * b["E"] + self.n_tok * lv["P"] * b["P"]))


def vq_key(levels: Dict[str, int]) -> str:
    """``{"N": 4, "E": 8, "P": 6}`` -> ``"vq_N4E8P6"`` (key in recon_eval / probe / noninferiority)."""
    return "vq_" + "".join(f"{s}{levels[s]}" for s in SCALES)


def parse_vq_key(key: str) -> Dict[str, int]:
    import re

    m = re.fullmatch(r"(?:vq_)?N(\d+)E(\d+)P(\d+)", key)
    if m is None:
        raise ValueError(f"not a vq key: {key}")
    return dict(zip(SCALES, map(int, m.groups())))


def default_vq_grid(max_levels: Dict[str, int]) -> list:
    """One axis at a time from the full model: N, E, P prefixes with the other two at maximum."""
    grid = []
    for s in SCALES:
        for k in sorted({1, 2, *range(2, max_levels[s] + 1, 1 if s == "P" else 2), max_levels[s]}):
            lv = dict(max_levels, **{s: k})
            if lv not in grid:
                grid.append(lv)
    return grid


@torch.no_grad()
def fit_vq_codebooks(model: MSAEVQ, loader, min_per_code: int = 10, min_windows: int = 2000,
                     max_batches: int = 1000, max_vectors: int = 200000, seed: int = 0) -> Dict[str, int]:
    """k-means initialisation of the three RVQs on the continuous latents of ``loader`` (call after
    the MSAE weights are loaded), with at least ``min_per_code`` vectors per code and
    ``min_windows`` windows per scale (P vectors of one window are correlated). N is fitted first; E and P are then encoded with ``N_q`` (and
    ``E_q``) so the codebooks see the latents the trained model will see. Returns vectors used."""
    from physioex.data.collate import stack_channels

    device = next(model.parameters()).device
    model.eval()
    need = min_per_code * max(model.n_codes.values())
    used = {}
    for s in SCALES:
        acc, n = [], 0
        # scales already fitted are quantized (all levels), the current one is continuous
        for i, batch in enumerate(loader):
            x = stack_channels(batch).to(device)
            z = _encode_partial(model, x, s)
            acc.append(z.reshape(-1, z.shape[-1]).cpu())
            n += len(x)
            if (n * (1 if s != "P" else model.n_tok) >= need and n >= min_windows) or i + 1 >= max_batches:
                break
        z = torch.cat(acc)
        if len(z) > max_vectors:
            z = z[torch.randperm(len(z), generator=torch.Generator().manual_seed(seed))[:max_vectors]]
        model.rvq[s].kmeans_init(z.to(device), seed)
        used[s] = len(z)
    return used


def _encode_partial(model: MSAEVQ, x: torch.Tensor, scale: str) -> torch.Tensor:
    """Continuous latent of ``scale`` with the earlier scales quantized (all levels)."""
    model.eval()
    xs = x[:, :, 0].float()
    n = model.encode_context(xs)
    if scale == "N":
        return n
    n = model.rvq["N"](n)[0]
    h = model.features(xs[:, model.c])
    g, b = model.e_film(n).chunk(2, -1)
    he = h * (1 + g[:, None]) + b[:, None]
    e = torch.tanh(model.e_out(model.e_pool(model.e_query.expand(len(h), -1, -1), he, he, need_weights=False)[0][:, 0]))
    if scale == "E":
        return e
    e = model.rvq["E"](e)[0]
    g, b = model.p_film(torch.cat([n, e], -1)).chunk(2, -1)
    return torch.tanh(model.p_out(h * (1 + g[:, None]) + b[:, None]))


class MSAEVQTrainer(MSAETrainer):
    """:class:`MSAETrainer` losses + the commitment of the three RVQs; logs code usage of level 1."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        loss, r2w, extra = MSAETrainer._step(model, batch, loss_fn, device)
        loss = loss + model.last_vq_loss
        extra["commit"] = float(model.last_vq_loss)
        for s, c in model.last_codes.items():
            if c is not None:
                extra[f"perplexity_{s}1"] = code_stats(c[..., 0], model.n_codes[s])[0]
        return loss, r2w, extra
