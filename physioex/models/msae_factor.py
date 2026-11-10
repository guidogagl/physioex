"""Factorized PSG encoder of phase 2 bis (``protosleepnet-v3/docs/PIANO_FASE_2B.md``, 09.11.2026).

Three factors, each with its own objective, and a multiplicative decoder that needs every one:

* **A, event atoms (per token).** A dictionary of ``n_atoms`` shift-invariant waveforms of
  ``atom_sec`` seconds, placed on a grid of ``sub`` positions per token. The encoder predicts a
  non-negative amplitude per atom and position (ReLU, ``top_k`` per token, L1 penalty), and the
  atom waveform of the epoch is the transposed convolution of the amplitudes with the unit-norm
  kernels. An atom fires where there is a shape, not where there is energy.
* **S, state (per epoch).** Attention pooling of the token features, ``tanh``. Trained by a
  temporal InfoNCE: the epoch and its next neighbour are positives, the other windows of the
  batch (other nights and times) negatives; and by the stationary background it decodes: a
  noise texture with one gain per band and epoch, matched to the median band power of the
  epoch's 1 s segments (robust to the events).
* **N, subject (per night).** The neighbours' mean token features through a small transformer
  (as in :class:`physioex.models.msae.MSAE`), ``tanh``. It decodes a per-band gain (a filter)
  applied to ``background(S) + atoms(A)``; a site head without reversal keeps the site there.
* **Adversary.** Site heads on the pooled atoms and on S through gradient reversal: the event
  and state factors must not read the site.
* **Weak rule supervision (optional, declared).** ``l_rule > 0``: a linear head on the atom
  amplitudes per token predicts the token masks of four AASM-style rules computed on the GPU
  (:func:`rule_masks`: K-complex, slow-wave activity, spindle, alpha), BCE with class weights.

Loss of the reconstruction: the ``"band"`` recipe of the codec (phase-exact MSE up to
``low_hz``, log band power per 1 s segment above ``pow_hz``, random grid shift). The ``encode``
output keeps the names of the codec (``N``, ``E`` = S, ``P`` = A amplitudes per token) so the
dump, the probes and the swap diagnostics (``concept_scan``, ``concept_leak``) run unchanged.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.models.msae import (
    LOW_HZ, POW_BANDS, STRIDES, MultiScaleStem, _down, band_logpow, grid_shift, lowpass_mse, stem_kernels,
)
from physioex.models.msvq import _autocast, _r2, _transformer
from physioex.train.trainer import Trainer

RULES = ("kc", "swa", "spindle", "alpha")


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _Reverse.apply(x, lam)


def _band_masks(n: int, fs: int, bands: Sequence[Sequence[float]], device) -> torch.Tensor:
    """``(len(bands), n // 2 + 1)`` rfft masks of the bands (Hz)."""
    f = torch.fft.rfftfreq(n, d=1 / fs, device=device)
    return torch.stack([(f >= a) & (f < b) for a, b in bands]).float()


def band_filter(x: torch.Tensor, gains: torch.Tensor, fs: int, bands: Sequence[Sequence[float]]) -> torch.Tensor:
    """``x (B, n)`` filtered by a piecewise-constant gain per band ``gains (B, len(bands))``;
    frequencies outside every band keep gain 1."""
    masks = _band_masks(x.shape[-1], fs, bands, x.device)  # (nb, F)
    g = 1 + (gains[:, :, None] - 1) * masks[None]  # (B, nb, F)
    g = g.prod(1)  # bands are disjoint: one factor per bin
    return torch.fft.irfft(torch.fft.rfft(x.float(), dim=-1) * g, n=x.shape[-1], dim=-1)


class MSAEFactor(nn.Module):
    def __init__(self, sequence_length: int = 41, width: int = 32, dim: int = 128, n_atoms: int = 64,
                 atom_sec: float = 2.0, sub: int = 2, top_k: int = 2, d_s: int = 64, d_n: int = 64,
                 token_sec: float = 0.5, fs: int = 100, pow_bands: Sequence[Sequence[float]] = None,
                 low_hz: float = LOW_HZ, pow_hz: float = 4.5, band_weight: float = 3.0, low_weight: float = 1.0,
                 l_sparse: float = 0.05, l_ctr: float = 0.5, l_adv: float = 0.2, l_bg: float = 1.0,
                 l_rule: float = 0.0, n_sites: int = 1, temperature: float = 0.1, use_n: bool = True):
        super().__init__()
        assert fs in (100, 200), fs
        self.L, self.c, self.dim, self.fs = sequence_length, sequence_length // 2, dim, fs
        self.epoch = 30 * fs
        self.pow_bands = tuple(tuple(b) for b in (pow_bands or POW_BANDS))
        self.low_hz, self.pow_hz, self.band_weight, self.low_weight = low_hz, pow_hz, band_weight, low_weight
        self.l_sparse, self.l_ctr, self.l_adv, self.l_bg, self.l_rule = l_sparse, l_ctr, l_adv, l_bg, l_rule
        self.temperature, self.use_n, self.n_sites = temperature, use_n, n_sites
        self.ctx_chunk = 1024  # epochs per forward pass of the neighbours' features (memory only)
        self.strides = (2,) * (fs // 100 - 1) + STRIDES[token_sec]
        self.n_tok = self.epoch // math.prod(self.strides)
        self.n_atoms, self.sub, self.top_k = n_atoms, sub, top_k
        self.d_p = n_atoms * sub  # width of "P" (amplitudes per token) for the dump tools
        self.d_s, self.d_n = d_s, d_n
        self.tex_bands = [i for i, (a, _) in enumerate(self.pow_bands) if a > pow_hz]
        # encoder (as the codec): multi-scale stem + SEANet + token transformer
        ch = [width * 2 ** min(i, 3) for i in range(len(self.strides) + 1)]
        self.stem = MultiScaleStem(width, stem_kernels(fs))
        self.down = nn.Sequential(*[_down(ch[i], ch[i + 1], s) for i, s in enumerate(self.strides)],
                                  nn.ELU(), nn.Conv1d(ch[-1], dim, 3, padding=1))
        self.pos_tok = nn.Parameter(torch.zeros(1, self.n_tok, dim))
        self.enc_tok = _transformer(dim, 2)
        # A: amplitudes per token, atom and sub-position
        self.a_out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_atoms * sub))
        k_len = int(atom_sec * fs)
        self.kernels = nn.Parameter(torch.randn(n_atoms, 1, k_len) * 0.1)
        self.hop = self.epoch // (self.n_tok * sub)
        # S: attention pooling, tanh; projection for the contrastive term
        self.s_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.s_pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.s_out = nn.Linear(dim, d_s)
        self.s_proj = nn.Sequential(nn.Linear(d_s, d_s), nn.GELU(), nn.Linear(d_s, d_s))
        self.bg_head = nn.Linear(d_s, len(self.tex_bands))  # log10 gain of the texture per band
        nn.init.zeros_(self.bg_head.weight); nn.init.constant_(self.bg_head.bias, -1.0)
        # N: neighbours' mean features -> transformer (t masked) -> tanh; per-band gain of the decoder
        self.n_in = nn.Linear(dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_epoch = nn.Parameter(torch.zeros(1, sequence_length, dim))
        self.enc_n = _transformer(dim, 2)
        self.n_out = nn.Linear(dim, d_n)
        self.gain_bands = ((0.0, self.pow_bands[0][0]),) + self.pow_bands
        self.gain_head = nn.Linear(d_n, len(self.gain_bands))  # log10 gain per band
        nn.init.zeros_(self.gain_head.weight); nn.init.zeros_(self.gain_head.bias)
        # heads: site from N (plain), site from A and S (reversed), rules from A
        self.site_n = nn.Linear(d_n, n_sites)
        self.site_a = nn.Sequential(nn.Linear(n_atoms, 64), nn.GELU(), nn.Linear(64, n_sites))
        self.site_s = nn.Sequential(nn.Linear(d_s, 64), nn.GELU(), nn.Linear(64, n_sites))
        self.rule_head = nn.Linear(n_atoms * sub, len(RULES))
        for p in (self.pos_tok, self.s_query, self.mask_token, self.pos_epoch):
            nn.init.normal_(p, std=0.02)
        self.vq_levels, self.n_codes = {}, {}  # read by the dump tools written for the codec
        self.site_of_dataset: Optional[Sequence[int]] = None

    # --- encoder -----------------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(self.stem(x[:, None])).transpose(1, 2)
        return self.enc_tok(h + self.pos_tok)

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[:2]
        if not self.use_n:
            return x.new_zeros(B, self.d_n)
        idx = [i for i in range(L) if i != self.c]
        with torch.no_grad():  # chunked: a whole-night extraction sends thousands of epochs at once
            xs = x[:, idx].reshape(-1, self.epoch)
            f = torch.cat([self.features(xs[i:i + self.ctx_chunk]).mean(1) for i in range(0, len(xs), self.ctx_chunk)])
        z = self.n_in(f).reshape(B, L - 1, -1)
        z = torch.cat([z[:, :self.c], self.mask_token.expand(B, 1, -1), z[:, self.c:]], 1)
        return torch.tanh(self.n_out(self.enc_n(z + self.pos_epoch)[:, self.c]))

    def state(self, h: torch.Tensor) -> torch.Tensor:
        s = self.s_pool(self.s_query.expand(len(h), -1, -1), h, h, need_weights=False)[0][:, 0]
        return torch.tanh(self.s_out(s))

    def atoms(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, T, dim) -> (B, T, n_atoms * sub)`` non-negative amplitudes, ``top_k`` per token."""
        a = F.relu(self.a_out(h))
        if self.top_k and self.top_k < a.shape[-1]:
            top = a.topk(self.top_k, dim=-1).indices
            a = torch.zeros_like(a).scatter(-1, top, a.gather(-1, top))
        return a

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``(B, L, 1, epoch) -> {"N": (B, d_n), "E": S (B, d_s), "P": A (B, T, n_atoms * sub), "codes": {}}``."""
        x = x[:, :, 0].float()
        n = self.encode_context(x)
        h = self.features(x[:, self.c])
        return {"N": n, "E": self.state(h), "P": self.atoms(h), "codes": {}}

    def _q(self, s: str, z: torch.Tensor):  # continuous factors: the codec's requantization is the identity
        return z, None, z.new_zeros(())

    # --- decoder -----------------------------------------------------------
    def atom_wave(self, a: torch.Tensor) -> torch.Tensor:
        """``(B, T, n_atoms * sub) -> (B, epoch)`` sum of the atoms placed on the sub-token grid."""
        B, T, _ = a.shape
        amp = a.reshape(B, T, self.n_atoms, self.sub).permute(0, 2, 1, 3).reshape(B, self.n_atoms, T * self.sub)
        k = self.kernels / (self.kernels.norm(dim=-1, keepdim=True) + 1e-6)
        pad = k.shape[-1] // 2
        w = F.conv_transpose1d(amp.float(), k.float(), stride=self.hop, padding=0)[:, 0]
        return w[:, pad:pad + self.epoch]

    def texture(self, s: torch.Tensor) -> torch.Tensor:
        """``(B, d_s) -> (B, epoch)`` noise texture: unit-RMS band-passed white noise x per-epoch gain."""
        lg = self.bg_head(s).float()
        g = 10 ** (lg + (lg.clamp(-6, 2) - lg).detach())
        spec = torch.fft.rfft(torch.randn(len(s), self.epoch, device=s.device), dim=-1)
        masks = _band_masks(self.epoch, self.fs, [self.pow_bands[i] for i in self.tex_bands], s.device)
        out = 0
        for j, m in enumerate(masks):
            nb = torch.fft.irfft(spec * m, n=self.epoch, dim=-1)
            out = out + g[:, j:j + 1] * nb / (nb.pow(2).mean(-1, keepdim=True).sqrt() + 1e-8)
        return out

    def gain(self, n: torch.Tensor) -> torch.Tensor:
        return 10 ** self.gain_head(n).float().clamp(-2, 2)

    def decode(self, n, e=None, p=None, texture: bool = False, parts: bool = False):
        """``x = f(N) * (background(S) + atoms(A))``. Missing S or A count as silent. With
        ``texture=True`` returns ``(atoms part, background part)`` both filtered by N, as the codec."""
        B = len(n)
        wa = self.atom_wave(p) if p is not None else n.new_zeros(B, self.epoch)
        wb = self.texture(e) if e is not None else n.new_zeros(B, self.epoch)
        g = self.gain(n)
        det, bg = band_filter(wa, g, self.fs, self.gain_bands), band_filter(wb, g, self.fs, self.gain_bands)
        if texture or parts:
            return det, bg
        return det + bg

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Everything the trainer needs: the factors, the two decoded parts, the next epoch's S
        projection, the heads."""
        xf = x[:, :, 0].float()
        n = self.encode_context(xf)
        h = self.features(xf[:, self.c])
        s, a = self.state(h), self.atoms(h)
        h2 = self.features(xf[:, self.c + 1])
        s2 = self.state(h2)
        det, bg = self.decode(n, s, a, parts=True)
        a_pool = a.reshape(len(a), self.n_tok, self.n_atoms, self.sub).amax(-1).mean(1)  # (B, n_atoms)
        out = {"N": n, "E": s, "P": a, "wave_det": det, "wave_bg": bg, "wave": det + bg,
               "z1": F.normalize(self.s_proj(s), dim=-1), "z2": F.normalize(self.s_proj(s2), dim=-1),
               "site_n": self.site_n(n), "site_a": self.site_a(reverse(a_pool, 1.0)),
               "site_s": self.site_s(reverse(s, 1.0)), "rules": self.rule_head(a)}
        return out

    def bits_per_epoch(self) -> int:
        return 8 * (self.d_n + self.d_s + self.n_tok * self.d_p)


# ---------------------------------------------------------------------------- rule masks (GPU)
def _bandpass(x: torch.Tensor, lo: float, hi: float, fs: int) -> torch.Tensor:
    f = torch.fft.rfftfreq(x.shape[-1], d=1 / fs, device=x.device)
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * ((f >= lo) & (f < hi)), n=x.shape[-1], dim=-1)


def _tok_reduce(x: torch.Tensor, n_tok: int, fn: str) -> torch.Tensor:
    B, n = x.shape
    v = x.reshape(B, n_tok, n // n_tok)
    if fn == "rms":
        return v.pow(2).mean(-1).sqrt()
    if fn == "p2p":
        return v.amax(-1) - v.amin(-1)
    return v.mean(-1)


def _ctx_max(v: torch.Tensor, w: int) -> torch.Tensor:
    """Max over ``±w`` tokens."""
    return F.max_pool1d(v[:, None], 2 * w + 1, stride=1, padding=w)[:, 0]


@torch.no_grad()
def rule_masks(x: torch.Tensor, fs: int, n_tok: int, swa_p2p: float = 3.0) -> torch.Tensor:
    """``(B, epoch) -> (B, n_tok, 4)`` float token masks of the weak rules, mirrors of
    :mod:`physioex.explain.concept_refs` thresholds on the token grid. The cached clean EEG is
    ``RobustScale``-normalized per night (``x_uV = x * scale + median``, scale ~ 15-30 uV on EEG), so
    the one absolute threshold is given in those units: ``swa_p2p`` 3.0 ~ 75 uV at a night scale of 25 uV.
    ``kc`` 0.3-4 Hz peak-to-peak over ±1 token >= 15 x the epoch's median |x|;
    ``swa`` 0.5-2 Hz peak-to-peak over ±1 token >= ``swa_p2p``;
    ``spindle`` 12-15 Hz RMS >= trimmed mean + 1.5 sd of the epoch and relative sigma power >= 0.2;
    ``alpha`` 8-13 Hz relative power >= 0.3, above theta (4-7) and beta (13-30)."""
    x = x.float()
    x_kc, x_sw = _bandpass(x, 0.3, 4, fs), _bandpass(x, 0.5, 2, fs)
    mad = x_kc.abs().median(-1).values[:, None] + 1e-9
    kc = _ctx_max(_tok_reduce(x_kc, n_tok, "p2p"), 1) >= 15 * mad
    swa = _ctx_max(_tok_reduce(x_sw, n_tok, "p2p"), 1) >= swa_p2p
    sig = _bandpass(x, 12, 15, fs)
    rms = _tok_reduce(sig, n_tok, "rms")
    lo, hi = rms.quantile(0.1, dim=-1, keepdim=True), rms.quantile(0.9, dim=-1, keepdim=True)
    keep = (rms >= lo) & (rms <= hi)
    cnt = keep.sum(-1, keepdim=True).clamp_min(1)
    mu = (rms * keep).sum(-1, keepdim=True) / cnt
    sd = (((rms - mu) ** 2) * keep).sum(-1, keepdim=True).div(cnt).sqrt()
    bands = {k: _tok_reduce(_bandpass(x, a, b, fs), n_tok, "rms") ** 2 for k, (a, b) in
             {"d": (0.5, 4), "t": (4, 7), "a": (8, 13), "s": (12, 15), "b": (13, 30), "w": (1, 30)}.items()}
    rel_s = F.avg_pool1d(bands["s"][:, None], 5, stride=1, padding=2)[:, 0] / (
        F.avg_pool1d(bands["w"][:, None], 5, stride=1, padding=2)[:, 0] + 1e-12)
    spindle = (rms >= mu + 1.5 * sd) & (rel_s >= 0.2)
    tot = bands["d"] + bands["t"] + bands["a"] + bands["b"] + 1e-12
    ra = bands["a"] / tot
    alpha = (ra >= 0.3) & (bands["a"] > bands["t"]) & (bands["a"] > bands["b"])
    return torch.stack([kc, swa, spindle, alpha], -1).float()


# ---------------------------------------------------------------------------- trainer
def sites_of(batch, model) -> Optional[torch.Tensor]:
    """Site index per window from ``batch["subject"][i]["dataset_idx"]`` and ``model.site_of_dataset``."""
    subs = batch.get("subject")
    sod = getattr(model, "site_of_dataset", None)
    if not subs or sod is None or not isinstance(subs[0], dict) or "dataset_idx" not in subs[0]:
        return None
    return torch.tensor([sod[int(s["dataset_idx"])] for s in subs], dtype=torch.long)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    logits = z1 @ z2.T / tau
    y = torch.arange(len(z1), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


class MSAEFactorTrainer(Trainer):
    """``acc`` = R^2 of the log band power of the decoded epoch (as the codec's ``band`` recipe)."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        from physioex.data.collate import stack_channels

        x = stack_channels(batch).to(device)
        xc = x[:, x.shape[1] // 2, 0].float()
        site = sites_of(batch, model)
        site = site.to(device) if site is not None else None
        with _autocast(device):
            out = model(x)
        return MSAEFactorTrainer._losses(model, out, xc, site)

    @staticmethod
    def _losses(model, out, xc, site):
        w, det, bg = out["wave"].float(), out["wave_det"].float(), out["wave_bg"].float()
        xg, gw, gb = grid_shift(model.training, xc, w, bg, seg=model.fs)
        bt = band_logpow(xg, model.fs, model.pow_bands)
        hi = model.tex_bands
        bw_ = band_logpow(gw, model.fs, model.pow_bands)
        l_pow = (bw_[..., hi] - bt[..., hi]).abs().mean()
        low = lowpass_mse(w, xc, model.low_hz, model.fs).mean()
        # background: its band power (constant in the epoch) matches the median segment of the epoch
        bb = band_logpow(gb, model.fs, model.pow_bands)[..., hi].mean(1)
        l_bg = (bb - bt[..., hi].median(1).values).abs().mean()
        a = out["P"].float()
        l_sparse = a.mean()
        l_ctr = info_nce(out["z1"].float(), out["z2"].float(), model.temperature)
        loss = model.band_weight * l_pow + model.low_weight * low + model.l_bg * l_bg \
            + model.l_sparse * l_sparse + model.l_ctr * l_ctr
        extra = {"pow": l_pow.item(), "mse_low": low.item(), "bg": l_bg.item(), "sparse": l_sparse.item(),
                 "ctr": l_ctr.item(), "r2_pow": _r2(bw_, bt).item(), "r2_wave": _r2(w, xc).item(),
                 "r2_low": (1 - low / (lowpass_mse(torch.zeros_like(xc), xc, model.low_hz, model.fs).mean() + 1e-9)).item(),
                 "active": (a > 0).float().mean().item(), "det_share": (det.pow(2).mean() / (w.pow(2).mean() + 1e-9)).item()}
        if site is not None and model.n_sites > 1:
            l_n = F.cross_entropy(out["site_n"].float(), site)
            l_a = F.cross_entropy(out["site_a"].float(), site)
            l_s = F.cross_entropy(out["site_s"].float(), site)
            loss = loss + 0.1 * l_n + model.l_adv * (l_a + l_s)
            extra.update(site_n=(out["site_n"].argmax(-1) == site).float().mean().item(),
                         site_a=(out["site_a"].argmax(-1) == site).float().mean().item(),
                         site_s=(out["site_s"].argmax(-1) == site).float().mean().item())
        if model.l_rule > 0:
            y = rule_masks(xc, model.fs, model.n_tok)
            pos = y.mean((0, 1)).clamp(1e-3, 1 - 1e-3)
            l_rule = F.binary_cross_entropy_with_logits(out["rules"].float(), y, pos_weight=(1 - pos) / pos)
            loss = loss + model.l_rule * l_rule
            extra["rule"] = l_rule.item()
            with torch.no_grad():
                for j, r in enumerate(RULES):
                    extra[f"rule_{r}_pos"] = y[..., j].mean().item()
        return loss, _r2(bw_, bt), extra


def fit_factor_norms(model: MSAEFactor, loader, max_batches: int = 20) -> None:
    """Nothing to fit (no spectral heads); kept for symmetry with the codec scripts."""
    return None
