"""Common encode/decode/intervene surface of the psg_gen models (SPEC section 2, ``model.py``).

``PSGGenBase`` wraps the shared label-free inference (:func:`inference.decompose`) and exposes
the codec-like interface of :mod:`physioex.models.msae_factor` (``N``/``E``/``P``, ``forward``,
``loss``, ``bits_per_epoch``), so ``pretrain_combo.encode_night`` and ``concept_scan`` work
unchanged. The events and the background are explicit: ``decode`` rebuilds the signal from them,
``intervene`` edits them (remove a class or a family, swap the background, scale), never the
input. P1, P2 and P3 override :meth:`PSGGenBase.decompose_batch` and :meth:`PSGGenBase.loss`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from physioex.models.msae_factor import rule_masks
from physioex.models.msvq import _r2
from physioex.models.psg_gen.background import gain_g, synth_background
from physioex.models.psg_gen.dicts import BG_BANDS, CLASSES, Dictionary, FS
from physioex.models.psg_gen.inference import Events, counts, decompose, reconstruct

Tensor = torch.Tensor
RULES = ("kc", "swa", "spindle", "alpha")


def band_logpow_epoch(x: Tensor, fs: int = FS, bands=BG_BANDS) -> Tensor:
    """(B, T) -> (B, n_bands) log10 band power (power per Hz) of the whole epoch."""
    p = torch.fft.rfft(x.float(), dim=-1).abs() ** 2 / x.shape[-1]
    f = torch.fft.rfftfreq(x.shape[-1], 1 / fs, device=x.device)
    return torch.stack([p[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands], -1).clamp_min(1e-10).log10()


class PSGGenBase(nn.Module):
    """Common encode/decode/intervene surface. Sub-classes override ``decompose_batch`` and
    ``loss``. No learnable parameters here: the dictionary and the thresholds are buffers."""

    def __init__(self, dictionary: Dictionary, budget: dict[int, int], thresholds: Tensor,
                 fs: int = FS, sequence_length: int = 41, token_sec: float = 0.5, n_sites: int = 1):
        super().__init__()
        self.fs, self.L, self.n_sites = fs, sequence_length, n_sites
        self.budget = {int(k): int(v) for k, v in budget.items()}
        self.register_buffer("atoms", dictionary.atoms.float().clone())
        self.register_buffer("quad", dictionary.quad.float().clone())
        self.register_buffer("thresholds", thresholds.float().clone())
        self._meta = dictionary
        self.n_tok = int(round(30.0 / token_sec))
        self.n_family = dictionary.n_family
        self.d_p = len(BG_BANDS) + self.n_family + len(CLASSES)
        self.d_e = len(BG_BANDS) + len(CLASSES) + 2
        self.d_n = 4
        self.vq_levels, self.n_codes, self.site_of_dataset = {}, {}, None

    # ------------------------------------------------------------------ dictionary
    @property
    def d(self) -> Dictionary:
        """The dictionary on the model device (atoms from the buffers, metadata from init)."""
        m = self._meta
        dev = self.atoms.device
        return Dictionary(atoms=self.atoms, quad=self.quad, cls=m.cls.to(dev), band=m.band.to(dev),
                          freq=m.freq.to(dev), dur=m.dur.to(dev), family=m.family.to(dev),
                          names=list(m.names))

    # ------------------------------------------------------------------ inference
    def decompose_batch(self, x: Tensor) -> dict:
        """(B, T) -> :func:`inference.decompose` output."""
        return decompose(x.float(), self.d, self.thresholds, self.budget, fs=self.fs)

    # ------------------------------------------------------------------ states
    def tokens(self, dec: dict, B: int) -> Tensor:
        """(B, n_tok, d_p): per 0.5 s token, [logP of the covering window; summed |amp| per
        family over the events whose support covers the token; counts per class of the events
        whose centre is in the token]."""
        dev = dec["logP"].device
        tok = 30.0 / self.n_tok
        tc = (torch.arange(self.n_tok, device=dev) + 0.5) * tok
        wi = (tc[:, None] - dec["centers"].to(dev)[None, :]).abs().argmin(-1)
        bg = dec["logP"][:, wi]                                       # (B, n_tok, 6)
        fam = torch.zeros(B, self.n_tok, self.n_family, device=dev)
        cnt = torch.zeros(B, self.n_tok, len(CLASSES), device=dev)
        ev: Events = dec["events"]
        if len(ev):
            d = self.d
            lo, hi = ev.onset - ev.dur / 2, ev.onset + ev.dur / 2
            cover = (tc[None, :] >= lo[:, None]) & (tc[None, :] <= hi[:, None])   # (n, n_tok)
            f = d.family[ev.atom]
            for i in range(len(ev)):  # n is O(budget x B): a loop is clear and cheap
                fam[ev.epoch[i], cover[i], f[i]] += ev.amp[i].abs()
            ti = (ev.onset / tok).long().clamp(0, self.n_tok - 1)
            cnt.index_put_((ev.epoch, ti, ev.cls), torch.ones(len(ev), device=dev), accumulate=True)
        return torch.cat([bg, fam, cnt], -1)

    def epoch_state(self, dec: dict, B: int) -> Tensor:
        """(B, d_e): [mean logP over windows (6); counts per class (3); mean change; mean eps]."""
        return torch.cat([dec["logP"].mean(1), dec["counts"].float(),
                          dec["change"].float().mean(1, keepdim=True),
                          dec["eps"].float().mean(1, keepdim=True)], -1)

    def night_state(self, x_ctx: Tensor) -> Tensor:
        """(B, L, T) -> (B, 4): [log10 g of the context; 20/50/80 % quantiles of the per-window
        (3 s) log10 RMS]."""
        out = []
        n = 3 * self.fs
        for v in x_ctx.float():
            g = gain_g(v.reshape(-1), fs=self.fs).median().clamp_min(1e-9).log10()
            m = v.numel() // n
            lr = v.reshape(-1)[:m * n].reshape(m, n).pow(2).mean(-1).sqrt().clamp_min(1e-9).log10()
            out.append(torch.cat([g[None].to(v.device), lr.quantile(torch.tensor([0.2, 0.5, 0.8], device=v.device))]))
        return torch.stack(out)

    # ------------------------------------------------------------------ codec surface
    def encode(self, x: Tensor) -> dict:
        """(B, L, C, T) -> {"N", "E", "P", "codes"} plus the decomposition of the centre epoch
        of channel 0 (``events``, ``logP``, ...)."""
        B, L = x.shape[:2]
        xc = x[:, L // 2, 0].float()
        dec = dict(self.decompose_batch(xc))
        dec.update(N=self.night_state(x[:, :, 0]), E=self.epoch_state(dec, B), P=self.tokens(dec, B),
                   codes={})
        return dec

    def decode(self, dec: dict, parts: bool = False, classes=None, seed: int | None = None,
               exclude_periodic: bool = False):
        """Events part (:func:`inference.reconstruct`) + background floor
        (:func:`background.synth_background`); ``parts=True`` returns (events, background)."""
        B, T = dec["logP"].shape[0], 30 * self.fs
        w_ev = reconstruct(dec["events"], self.d, B, T, fs=self.fs, classes=classes,
                           exclude_periodic=exclude_periodic)
        w_bg = synth_background(dec["logP"].float(), fs=self.fs, T=T, seed=0 if seed is None else seed)
        return (w_ev, w_bg) if parts else w_ev + w_bg

    def intervene(self, dec: dict, kind: str, **kw) -> dict:
        """A new decomposition with one edit, the input is never mutated. ``kind``:
        ``remove_class`` (cls), ``keep_only`` (cls), ``remove_family`` (family),
        ``swap_background`` (logP_from: another decomposition or a logP tensor),
        ``scale`` (factor, on the event amplitudes)."""
        out = dict(dec)
        ev: Events = dec["events"]
        if kind == "remove_class":
            ev = ev.select(ev.cls != int(kw["cls"]))
        elif kind == "keep_only":
            ev = ev.select(ev.cls == int(kw["cls"]))
        elif kind == "remove_family":
            ev = ev.select(self.d.family[ev.atom] != int(kw["family"]))
        elif kind == "scale":
            ev = ev.select(torch.ones(len(ev), dtype=torch.bool, device=ev.epoch.device))
            ev.amp = ev.amp * float(kw["factor"])
        elif kind == "swap_background":
            src = kw["logP_from"]
            out["logP"] = (src["logP"] if isinstance(src, dict) else src).clone()
            return out
        else:
            raise ValueError(f"unknown intervention {kind!r}")
        out["events"] = ev
        out["counts"] = counts(ev, dec["logP"].shape[0])
        return out

    def forward(self, x: Tensor) -> dict:
        dec = self.encode(x)
        w_ev, w_bg = self.decode(dec, parts=True)
        return {"N": dec["N"], "E": dec["E"], "P": dec["P"], "wave": w_ev + w_bg, "wave_det": w_ev,
                "wave_bg": w_bg, "dec": dec}

    def loss(self, out: dict, xc: Tensor, site: Tensor | None):
        """Base: zero loss; ``acc`` = R^2 of the epoch band log power of the decoded signal."""
        acc = _r2(band_logpow_epoch(out["wave"].float(), self.fs), band_logpow_epoch(xc, self.fs))
        zero = xc.new_zeros(())
        return zero, acc, {"r2_pow": float(acc), "base_share": 1.0}

    def bits_per_epoch(self) -> int:
        """Nominal size: per budgeted event atom index + onset + amplitude (8 bit) (+ phase
        8 bit for bursts), plus 8 bit per window and band of the background."""
        K, T = self.atoms.shape[0], 30 * self.fs
        per = math.ceil(math.log2(K)) + math.ceil(math.log2(T)) + 8
        ev = sum(v * (per + (8 if c == 0 else 0)) for c, v in self.budget.items())
        n_win = (T - 4 * self.fs) // self.fs + 1
        return int(ev + 8 * n_win * len(BG_BANDS))


@torch.no_grad()
def fidelity_gate(x: Tensor, x_hat: Tensor, fs: int = FS, n_tok: int = 60) -> dict:
    """Gate 0 (R12): per rule of :func:`msae_factor.rule_masks`, the token F1 between the masks
    of ``x`` and of ``x_hat`` with the thresholds frozen on ``x``; plus the residual energy
    share ||x - x_hat||^2 / ||x||^2."""
    m0 = rule_masks(x, fs, n_tok) > 0
    m1 = rule_masks(x_hat, fs, n_tok, ref=x) > 0
    out = {}
    for i, r in enumerate(RULES):
        a, b = m0[..., i], m1[..., i]
        tp = (a & b).sum().item()
        fp, fn = (~a & b).sum().item(), (a & ~b).sum().item()
        f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
        out[r] = {"f1": f1, "n_ref": int(a.sum()), "n_hat": int(b.sum())}
    out["resid_share"] = float((x - x_hat).pow(2).sum() / x.pow(2).sum().clamp_min(1e-12))
    return out
