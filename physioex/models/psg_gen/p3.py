"""P3, probabilistic (SPEC section 2, ``p3.py``).

* Events: the shared sequential inference (NMS = refractory term, ECG guard = anti-periodicity)
  with the selection gate in closed form instead of an empirical null: under H0 the whitened
  normalised correlation of an atom with m support samples has rho^2 ~ Beta(1/2, (m-1)/2)
  (Beta(1, (m-2)/2) for a cos/sin pair); Bonferroni over K atoms x T positions at level alpha
  gives the per-atom threshold via ``scipy.special.betaincinv``. m counts samples, not the
  effective degrees of freedom of a low-pass atom, so the gate is looser than the empirical null
  for the transients and slow waves (measured 17.11.2026: 0.65/0.57/0.51 vs 0.76/0.67/0.64).
* Presence: per-class Poisson counts per epoch with learnable rates (prior mean = the budget
  share, a marked point process with the NMS as refractory term); its NLL is the KL term.
* Background: per band a random-walk log level with a jump indicator shared by the bands
  (interacting-multiple-model Kalman filter, moment matched, plus RTS smoother). The observation
  is the q20 log band power of the masked residual per window; the prediction-error likelihood
  of the filter is the background NLL; ``change`` is the posterior jump probability.
* Amplitudes: the joint LS of the shared inference in the signal domain (the GLS variant of the
  design is not ported: with the whitened selection the two differ only in the weighting of the
  coloured noise; noted in the comparison doc).
* Night adversary on the per-family presences (gradient reversal; it only trains its head here
  since the presences come from a discrete selection).

The amortised proposal network of the design is optional and not part of this module.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import betaincinv

from physioex.models.msae_factor import reverse
from physioex.models.msvq import _r2
from physioex.models.psg_gen.background import synth_background
from physioex.models.psg_gen.dicts import BG_BANDS, CLASSES, FS, Dictionary
from physioex.models.psg_gen.inference import _support_boxes, decompose
from physioex.models.psg_gen.model import PSGGenBase, band_logpow_epoch

Tensor = torch.Tensor
TERMS = ("bg", "presence", "adv")


def beta_bonferroni_thresholds(d: Dictionary, T: int, alpha: float = 0.05, fs: int = FS) -> Tensor:
    """(K,) closed-form selection thresholds on the normalised correlation (see module doc)."""
    n_d, _ = _support_boxes(d, fs)
    m = n_d.double().numpy()
    pair = (d.quad.abs().sum(-1) > 0).numpy()
    p = 1.0 - alpha / (len(d) * T)
    t2 = np.where(pair, betaincinv(1.0, np.maximum(m - 2, 1) / 2, p), betaincinv(0.5, np.maximum(m - 1, 1) / 2, p))
    return torch.tensor(np.sqrt(t2), dtype=torch.float32)


def _norm_logpdf(v: Tensor, var: Tensor) -> Tensor:
    return -0.5 * (v.pow(2) / var + torch.log(2 * math.pi * var))


def jump_kalman(y: Tensor, log_q: Tensor, log_qj: Tensor, logit_pi: Tensor, log_r: Tensor):
    """IMM-lite filter + RTS smoother. y (B, W, nb) observed log10 levels; per-band ``log_q``,
    ``log_qj``, ``log_r`` (nb,), scalar ``logit_pi`` (the jump is shared by the bands). Returns (smoothed (B, W, nb), jump probability (B, W), negative log-likelihood per
    window and band (scalar mean))."""
    B, W, nb = y.shape
    q, qj, r = log_q.exp(), log_qj.exp(), log_r.exp()
    lp1, lp0 = F.logsigmoid(logit_pi), F.logsigmoid(-logit_pi)
    x, P = y[:, 0], r.expand(B, nb)
    xs_f, Ps_f, Pp_f, pj, ll = [x], [P], [P], [y.new_zeros(B)], []
    for w in range(1, W):
        Pp = P + q
        v = y[:, w] - x
        L0 = _norm_logpdf(v, Pp + r).sum(-1)
        L1 = _norm_logpdf(v, Pp + qj + r).sum(-1)
        a1, a0 = lp1 + L1, lp0 + L0
        lse = torch.logaddexp(a1, a0)
        p = torch.exp(a1 - lse)                                      # posterior jump probability
        ll.append(lse)
        Pp = Pp + p[:, None] * qj                                     # moment matching
        K = Pp / (Pp + r)
        x = x + K * v
        P = (1 - K) * Pp
        xs_f.append(x)
        Ps_f.append(P)
        Pp_f.append(Pp)
        pj.append(p)
    xs = [xs_f[-1]]
    for w in range(W - 2, -1, -1):                                   # RTS
        G = Ps_f[w] / Pp_f[w + 1]
        xs.append(xs_f[w] + G * (xs[-1] - xs_f[w]))
    xs = torch.stack(xs[::-1], 1)
    nll = -torch.stack(ll, 1).mean() / nb if ll else y.sum() * 0
    return xs, torch.stack(pj, 1), nll


class PSGGenP3(PSGGenBase):
    def __init__(self, dictionary: Dictionary, budget: dict[int, int], thresholds: Tensor | None = None,
                 fs: int = FS, sequence_length: int = 41, token_sec: float = 0.5, n_sites: int = 1,
                 alpha: float = 0.05, adv_weight: float = 0.2, n_night_slots: int = 256,
                 no_adv: bool = False):
        thr = beta_bonferroni_thresholds(dictionary, 30 * fs, alpha=alpha, fs=fs)
        super().__init__(dictionary, budget, thr, fs=fs, sequence_length=sequence_length,
                         token_sec=token_sec, n_sites=n_sites)
        if thresholds is not None:
            self.register_buffer("null_ref", thresholds.float().clone())  # for the comparison only
        nb = len(BG_BANDS)
        self.log_q = nn.Parameter(torch.full((nb,), math.log(0.01)))
        self.log_qj = nn.Parameter(torch.full((nb,), math.log(0.5)))
        self.logit_pi = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))  # one jump prior, shared
        self.log_r = nn.Parameter(torch.full((nb,), math.log(0.02)))
        tot = max(sum(self.budget.values()), 1)
        prior = torch.tensor([max(self.budget.get(c, 0), 1) / tot * 4.0 for c in range(len(CLASSES))])
        self.log_rate = nn.Parameter(prior.log())
        self.adv_weight, self.no_adv, self.n_night_slots = adv_weight, no_adv, n_night_slots
        self.adv = nn.Linear(self.n_family, n_night_slots)

    def encode(self, x: Tensor) -> dict:
        dec = super().encode(x)
        # the jump indicator of the state-space background replaces the closed-form change; the
        # observation is logP (q20 of the masked residual), the decoder keeps it
        _, pj, _ = jump_kalman(dec["logP"].float(), self.log_q, self.log_qj, self.logit_pi, self.log_r)
        dec["change"] = pj.detach()
        dec["E"] = self.epoch_state(dec, x.shape[0])
        return dec

    def decompose_batch(self, x: Tensor) -> dict:
        with torch.no_grad():
            return decompose(x.float(), self.d, self.thresholds, self.budget, fs=self.fs,
                             passes=1 if self.training else 2)

    def forward(self, x: Tensor) -> dict:
        dec = self.encode(x)
        T = 30 * self.fs
        xs, pj, nll = jump_kalman(dec["logP"].float(), self.log_q, self.log_qj, self.logit_pi, self.log_r)
        dec.update(logP_s=xs, nll_bg=nll)
        w_ev, w_bg = self.decode(dec, parts=True)
        return {"N": dec["N"], "E": dec["E"], "P": dec["P"], "wave": w_ev + w_bg, "wave_det": w_ev,
                "wave_bg": w_bg, "dec": dec}

    def loss(self, out: dict, xc: Tensor, site: Tensor | None):
        dec = out["dec"]
        B = xc.shape[0]
        cnt = dec["counts"].float()
        rate = self.log_rate.exp()[None].expand(B, -1)
        t = {"bg": dec["nll_bg"],
             "presence": F.poisson_nll_loss(self.log_rate[None].expand(B, -1), cnt, log_input=True, full=True)}
        night = out.get("night")
        ev = dec["events"]
        if self.no_adv or night is None or len(ev) == 0:
            t["adv"] = t["bg"] * 0.0
        else:
            pres = torch.zeros(B, self.n_family, device=xc.device)
            pres[ev.epoch, self.d.family[ev.atom]] = 1.0
            t["adv"] = F.cross_entropy(self.adv(reverse(pres, 1.0)), night % self.n_night_slots)
        w = {"bg": 1.0, "presence": 1.0, "adv": 0.0 if self.no_adv else self.adv_weight}
        parts = {k: w[k] * t[k] for k in TERMS}
        loss = sum(parts.values())
        tot = sum(float(v.detach().abs()) for v in parts.values()) or 1.0
        acc = _r2(band_logpow_epoch(out["wave"].detach().float(), self.fs), band_logpow_epoch(xc.float(), self.fs))
        extra = {f"{k}_share": float(v.detach().abs()) / tot for k, v in parts.items()}
        extra.update({k: float(v.detach()) for k, v in t.items()})
        extra.update({f"rate_{c}": float(rate[0, i]) for i, c in enumerate(CLASSES)})
        extra.update(n_events=len(ev), jump_mean=float(dec["change"].mean()))
        return loss, acc, extra
