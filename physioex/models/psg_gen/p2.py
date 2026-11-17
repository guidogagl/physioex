"""P2, structured neural (SPEC section 2, ``p2.py``).

Same event inference as P1 (selection, NMS, budgets, joint LS of :mod:`inference`), with the
parts that P1 fixes by hand made learnable:

* per-atom selection thresholds (a non-negative offset on the null thresholds), trained through soft gates
  ``sigmoid((score - thr) / T)`` that scale the amplitudes of the selected events;
* 16 free transient and 16 free slow-wave shapes in place of the analytic ones, with the class
  constraints imposed by a differentiable re-parametrisation (low-pass 5 Hz / 2 Hz, maximum
  duration 1 s / 2.5 s, zero mean, unit norm), so no projection step is needed after the
  optimiser.

The selection runs without gradient; the gradient reaches shapes and thresholds through the
differentiable re-synthesis of the selected events. Loss (R4, two residuals): time-domain MSE of
``x - events`` on the supports +-1 s, masked band log power of the residual against the
background ``logP`` with a dead zone, a price per class (L1 of the gated amplitudes: unit-norm
atoms, so equal energy costs the same), the night adversary (gradient reversal on the per-family
presences), a Gram coherence penalty on the free shapes (> 0.3 at lag 0) and a periodicity
penalty (gates of the transients flagged periodic). The weights are fixed at the first training
step so that every term has the same gradient norm; ``extra`` logs each term's share.
Ablation flags: ``fixed_dict`` (keep the analytic shapes), ``no_adv``, ``events_only`` (no
background term). The unrolled-ISTA candidate proposer of the design is not used: the shared
inference proposes the candidates (SPEC: "the unrolled codes only propose candidates").
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from physioex.models.msae_factor import reverse
from physioex.models.psg_gen.background import seg_band_logpow, synth_background
from physioex.models.psg_gen.dicts import (BG_BANDS, CLS_BURST, CLS_SLOWWAVE, CLS_TRANSIENT, FS,
                                           Dictionary)
from physioex.models.psg_gen.inference import (SW_SCORE_FLOOR, Events, decompose, null_thresholds,
                                               periodic_flag, support_mask)
from physioex.models.psg_gen.p1 import phase_surrogate
from physioex.models.psg_gen.model import PSGGenBase, band_logpow_epoch
from physioex.models.msvq import _r2

Tensor = torch.Tensor
TERMS = ("mse", "bg", "price", "adv", "gram", "per")
# prior importance applied after the equal-gradient-norm weights: with equal norms alone the
# price and the adversary undo the fit (measured 17.11.2026: support R^2 0.73 -> 0.44 in 200 steps)
PRIOR = {"mse": 1.0, "bg": 1.0, "price": 0.1, "adv": 0.2, "gram": 0.1, "per": 1.0}


def _free_dictionary(d: Dictionary, thr: Tensor, n_free: int) -> tuple[Dictionary, Tensor, Tensor]:
    """Bursts of ``d`` + ``n_free`` transient and ``n_free`` slow-wave slots initialised from the
    analytic atoms of those classes (cycled). Returns (dictionary, thresholds, source index)."""
    keep = torch.nonzero(d.cls == CLS_BURST, as_tuple=True)[0]
    rows, src = [keep], [keep]
    for c in (CLS_TRANSIENT, CLS_SLOWWAVE):
        ks = torch.nonzero(d.cls == c, as_tuple=True)[0]
        if ks.numel() == 0:
            continue
        sel = ks[torch.arange(n_free) % ks.numel()]
        rows.append(sel)
        src.append(sel)
    idx = torch.cat(rows)
    fam_b = d.family[keep]
    n_fb = int(fam_b.max()) + 1 if keep.numel() else 0
    fam_free = n_fb + torch.arange(idx.numel() - keep.numel())
    nd = Dictionary(atoms=d.atoms[idx].clone(), quad=d.quad[idx].clone(), cls=d.cls[idx].clone(),
                    band=d.band[idx].clone(), freq=d.freq[idx].clone(), dur=d.dur[idx].clone(),
                    family=torch.cat([fam_b, fam_free]),
                    names=[d.names[int(i)] if j < keep.numel() else f"free_{d.names[int(i)]}_{j - keep.numel()}"
                           for j, i in enumerate(idx.tolist())])
    return nd, thr[idx].clone(), torch.cat(src)


class PSGGenP2(PSGGenBase):
    def __init__(self, dictionary: Dictionary, budget: dict[int, int], thresholds: Tensor,
                 fs: int = FS, sequence_length: int = 41, token_sec: float = 0.5, n_sites: int = 1,
                 n_free: int = 16, gate_temp: float = 0.05, dead_zone: float = 0.1,
                 adv_weight: float = 0.2, n_night_slots: int = 256, fixed_dict: bool = False,
                 no_adv: bool = False, events_only: bool = False, train_passes: int = 1,
                 adv_ramp: int = 20000, null_every: int = 20):
        if not fixed_dict:
            dictionary, thresholds, _ = _free_dictionary(dictionary, thresholds, n_free)
        super().__init__(dictionary, budget, thresholds, fs=fs, sequence_length=sequence_length,
                         token_sec=token_sec, n_sites=n_sites)
        self.fixed_dict, self.no_adv, self.events_only = fixed_dict, no_adv, events_only
        self.gate_temp, self.dead_zone, self.adv_weight = gate_temp, dead_zone, adv_weight
        self.n_night_slots, self.train_passes, self.adv_ramp = n_night_slots, train_passes, adv_ramp
        self.register_buffer("n_steps", torch.tensor(0))
        self.null_every = null_every
        K, L = self.atoms.shape
        # learned thresholds only rise over the null (R13: the null q95 bounds the false alarms)
        self.thr_raw = nn.Parameter(torch.full((K,), -4.0))
        free = (self._meta.cls != CLS_BURST) & torch.tensor([not fixed_dict] * K)
        self.register_buffer("free_idx", torch.nonzero(free, as_tuple=True)[0])
        # x sqrt(L): O(1) entries, so an Adam step is a small relative change of the shape (the
        # re-parametrisation normalises, the scale is free)
        self.free_raw = nn.Parameter(self.atoms[self.free_idx].clone() * math.sqrt(L))
        # fixed parts of the re-parametrisation: frequency mask and duration window per slot
        f = torch.fft.rfftfreq(L, 1 / fs)
        cut = torch.where(self._meta.cls[self.free_idx] == CLS_SLOWWAVE, 2.0, 5.0)
        self.register_buffer("free_fmask", torch.sigmoid((cut[:, None] - f[None]) / 0.25))
        maxd = torch.where(self._meta.cls[self.free_idx] == CLS_SLOWWAVE, 2.5, 1.0)
        tt = (torch.arange(L) - L // 2).abs()[None] / fs
        self.register_buffer("free_twin", torch.sigmoid((maxd[:, None] / 2 - tt) / 0.05))
        self.adv = nn.Linear(self.n_family, n_night_slots)
        self.register_buffer("term_w", torch.ones(len(TERMS)))
        self.register_buffer("term_w_set", torch.tensor(False))

    # ------------------------------------------------------------------ dictionary
    def free_atoms(self) -> Tensor:
        a = torch.fft.irfft(torch.fft.rfft(self.free_raw, dim=-1) * self.free_fmask, n=self.free_raw.shape[-1])
        a = a * self.free_twin
        a = a - a.mean(-1, keepdim=True)
        return a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @property
    def d(self) -> Dictionary:
        base = super().d
        if self.free_idx.numel() == 0:
            return base
        atoms = base.atoms.clone()
        atoms[self.free_idx] = self.free_atoms()
        return Dictionary(atoms=atoms, quad=base.quad, cls=base.cls, band=base.band, freq=base.freq,
                          dur=base.dur, family=base.family, names=base.names)

    def thr_eff(self) -> Tensor:
        return self.thresholds + F.softplus(self.thr_raw)

    def decompose_batch(self, x: Tensor) -> dict:
        with torch.no_grad():
            d = self.d
            d = Dictionary(atoms=d.atoms.detach(), quad=d.quad, cls=d.cls, band=d.band, freq=d.freq,
                           dur=d.dur, family=d.family, names=d.names)
            if self.training and self.free_idx.numel() and int(self.n_steps) % self.null_every == 0:
                # the null of a free shape moves with the shape: re-estimated on phase-randomised
                # surrogates of the batch (as P1 after its M-step)
                fi = self.free_idx
                sub = Dictionary(atoms=d.atoms[fi], quad=d.quad[fi], cls=d.cls[fi], band=d.band[fi],
                                 freq=d.freq[fi], dur=d.dur[fi], family=d.family[fi],
                                 names=[d.names[i] for i in fi.tolist()])
                self.thresholds[fi] = null_thresholds(sub, phase_surrogate(x.float()), fs=self.fs)
            if self.training:
                # candidates at the null threshold: an event the learned threshold would drop stays
                # selected with a small gate and its support stays in the MSE, so raising a
                # threshold over a real event has a cost
                return decompose(x.float(), d, self.thresholds, self.budget, fs=self.fs,
                                 passes=self.train_passes)
            return decompose(x.float(), d, self.thr_eff().detach(), self.budget, fs=self.fs, passes=2)

    # ------------------------------------------------------------------ forward / loss
    def _resynth(self, ev: Events, A: Tensor, Q: Tensor, gate: Tensor, B: int, T: int) -> Tensor:
        """Differentiable :func:`inference.reconstruct` with gated amplitudes."""
        y = torch.zeros(B * T, device=A.device, dtype=A.dtype)
        if len(ev) == 0:
            return y.view(B, T)
        k = ev.atom
        w = ev.phase.cos()[:, None] * A[k] + ev.phase.sin()[:, None] * Q[k]
        Lk = A.shape[1]
        s0 = torch.round(ev.onset * self.fs).long() - Lk // 2
        cols = s0[:, None] + torch.arange(Lk, device=A.device)[None, :]
        valid = (cols >= 0) & (cols < T)
        idx = (ev.epoch[:, None] * T + cols)[valid]
        vals = ((ev.amp * gate)[:, None] * w)[valid]
        return y.index_add(0, idx, vals).view(B, T)

    def forward(self, x: Tensor) -> dict:
        dec = self.encode(x)
        B, T = dec["logP"].shape[0], 30 * self.fs
        ev: Events = dec["events"]
        d = self.d
        thr = self.thr_eff()
        # the acceptance threshold of the slow waves is capped when the p2p rule is on
        thr = torch.where(d.cls == CLS_SLOWWAVE, thr.clamp_max(SW_SCORE_FLOOR), thr)
        gate = torch.sigmoid((ev.score - thr[ev.atom]) / self.gate_temp) if len(ev) else ev.amp
        keep = ~ev.periodic if len(ev) else ev.periodic
        evk = ev.select(keep)
        w_det = self._resynth(evk, d.atoms, d.quad, gate[keep], B, T)
        with torch.no_grad():
            w_bg = synth_background(dec["logP"].float(), fs=self.fs, T=T, seed=0)
        dec["gate"] = gate
        return {"N": dec["N"], "E": dec["E"], "P": dec["P"], "wave": w_det + w_bg, "wave_det": w_det,
                "wave_bg": w_bg, "dec": dec}

    def _terms(self, out: dict, xc: Tensor) -> dict[str, Tensor]:
        dec, w_det = out["dec"], out["wave_det"].float()
        ev: Events = dec["events"]
        gate = dec["gate"]
        B, T = xc.shape
        zero = w_det.sum() * 0.0
        d = self.d
        with torch.no_grad():
            m = support_mask(ev.select(~ev.periodic), d, B, T, fs=self.fs, pad=1.0).float()
        resid = xc - w_det
        t = {"mse": (resid.pow(2) * m).sum() / (xc.pow(2) * m).sum().clamp_min(1e-9)}
        if self.events_only:
            t["bg"] = zero
        else:
            lp = seg_band_logpow(resid, self.fs)                         # (B, S, 6)
            dz = (lp.mean(1) - dec["logP"].detach().mean(1)).abs() - self.dead_zone
            t["bg"] = F.relu(dz).pow(2).mean()
        a = ev.amp.abs()
        t["price"] = (gate * a).sum() / a.sum().clamp_min(1e-9) if len(ev) else zero
        night = out.get("night")
        if self.no_adv or night is None or len(ev) == 0:
            t["adv"] = zero
        else:
            fam = d.family[ev.atom]
            pres = torch.zeros(B, self.n_family, device=xc.device).index_put(
                (ev.epoch, fam), gate, accumulate=True)
            pres = 1 - torch.exp(-pres)                                   # soft OR per family
            # DANN schedule (Ganin et al. 2016): the reversal grows from 0 to 1 over adv_ramp
            # steps, so the head learns before its reversed gradient moves the thresholds
            p = float(self.n_steps) / max(self.adv_ramp, 1)
            lam = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
            logits = self.adv(reverse(pres, lam))
            t["adv"] = F.cross_entropy(logits, night % self.n_night_slots)
        if self.free_idx.numel() > 1:
            fa = self.free_atoms()
            G = fa @ fa.T
            off = G - torch.diag(torch.diag(G))
            t["gram"] = F.relu(off.abs() - 0.3).pow(2).mean()
        else:
            t["gram"] = zero
        tr = (ev.cls == CLS_TRANSIENT) & periodic_flag(ev) if len(ev) else ev.periodic
        t["per"] = (gate * tr.float()).sum() / B if len(ev) else zero
        return t

    def _set_weights(self, t: dict[str, Tensor]) -> None:
        params = [p for n, p in self.named_parameters() if not n.startswith("adv.") and p.requires_grad]
        w = torch.ones(len(TERMS), device=self.term_w.device)
        norms = {}
        for i, k in enumerate(TERMS):
            if not t[k].requires_grad:
                continue
            g = torch.autograd.grad(t[k], params, retain_graph=True, allow_unused=True)
            n = math.sqrt(sum(float(x.pow(2).sum()) for x in g if x is not None))
            if n > 0:
                norms[k] = n
        if norms:
            ref = sum(norms.values()) / len(norms)
            for i, k in enumerate(TERMS):
                if k in norms:
                    w[i] = ref / norms[k]
        for i, k in enumerate(TERMS):
            w[i] *= self.adv_weight if k == "adv" else PRIOR[k]
        self.term_w.copy_(w)
        self.term_w_set.fill_(True)

    def loss(self, out: dict, xc: Tensor, site: Tensor | None):
        t = self._terms(out, xc.float())
        if self.training:
            self.n_steps += 1
        if self.training and not bool(self.term_w_set):
            self._set_weights(t)
        parts = {k: self.term_w[i] * t[k] for i, k in enumerate(TERMS)}
        loss = sum(parts.values())
        tot = sum(float(v.detach().abs()) for v in parts.values()) or 1.0
        acc = _r2(band_logpow_epoch(out["wave"].detach().float(), self.fs), band_logpow_epoch(xc, self.fs))
        extra = {f"{k}_share": float(v.detach().abs()) / tot for k, v in parts.items()}
        extra.update({k: float(v.detach()) for k, v in t.items()})
        extra["n_events"] = len(out["dec"]["events"])
        return loss, acc, extra
