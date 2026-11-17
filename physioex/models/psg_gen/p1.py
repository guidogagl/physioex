"""P1, classical first (SPEC section 2, ``p1.py``): no learnable parameters.

The background and the gain are closed form; the events come from the shared sequential
inference with a fixed analytic dictionary. :meth:`PSGGenP1.fit_transients` learns the
transient atoms (K-complex-like shapes) by alternating minimisation, as in convolutional
dictionary learning (alphacsc; Jas et al. 2017): E-step = :func:`inference.sparse_events` with
the current dictionary, M-step = least squares of each transient atom on the segments where it
was selected, then the class constraints (:func:`dicts.project_constraints`), restart of unused
atoms and merge of near duplicates. The null thresholds are then recomputed on phase-randomised
surrogates of the training epochs (same spectrum, no transient structure).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import torch

from physioex.models.psg_gen.dicts import CLS_BURST, CLS_SLOWWAVE, CLS_TRANSIENT, FS, Dictionary, coherence, project_constraints
from physioex.models.psg_gen.inference import decompose, null_thresholds, reconstruct, sparse_events
from physioex.models.psg_gen.model import PSGGenBase

Tensor = torch.Tensor


def phase_surrogate(x: Tensor, seed: int = 0) -> Tensor:
    """(B, T) -> (B, T) with the same amplitude spectrum per row and uniform random phases: the
    null of "no time-localised structure" for the thresholds."""
    g = torch.Generator(device=x.device.type).manual_seed(int(seed))
    X = torch.fft.rfft(x.float(), dim=-1)
    ph = torch.rand(X.shape, generator=g, device=x.device) * 2 * torch.pi
    ph[..., 0] = 0.0
    return torch.fft.irfft(X.abs() * torch.exp(1j * ph), n=x.shape[-1], dim=-1)


def _segments(x: Tensor, epoch: Tensor, onset_s: Tensor, L: int, fs: int) -> Tensor:
    """(n, L) windows of x (B, T) centred at the given onsets (seconds), zero outside."""
    T = x.shape[-1]
    s0 = torch.round(onset_s * fs).long() - L // 2
    cols = s0[:, None] + torch.arange(L, device=x.device)[None, :]
    valid = (cols >= 0) & (cols < T)
    out = torch.zeros(len(epoch), L, device=x.device, dtype=x.dtype)
    out[valid] = x[epoch[:, None].expand(-1, L)[valid], cols[valid]]
    return out


class PSGGenP1(PSGGenBase):
    """Classical model. Ablation flags: ``whiten`` (matched filter in the whitened domain),
    ``use_gain`` (divide the epoch by the night gain g before the inference),
    ``fixed_dict`` (skip :meth:`fit_transients`)."""

    def __init__(self, dictionary: Dictionary, budget: dict[int, int], thresholds: Tensor,
                 fs: int = FS, sequence_length: int = 41, token_sec: float = 0.5, n_sites: int = 1,
                 whiten: bool = True, use_gain: bool = True, fixed_dict: bool = False):
        super().__init__(dictionary, budget, thresholds, fs=fs, sequence_length=sequence_length,
                         token_sec=token_sec, n_sites=n_sites)
        self.whiten, self.use_gain, self.fixed_dict = whiten, use_gain, fixed_dict
        self.fit_log: list[dict] = []

    def decompose_batch(self, x: Tensor) -> dict:
        return decompose(x.float(), self.d, self.thresholds, self.budget, fs=self.fs, whiten=self.whiten)

    def encode(self, x: Tensor) -> dict:
        if not self.use_gain:
            return super().encode(x)
        # R7: the inference runs on the epoch divided by the night gain; amplitudes and the
        # background are put back in signal units, the gain is in N[:, 0]
        B, L = x.shape[:2]
        N = self.night_state(x[:, :, 0])
        g = (10.0 ** N[:, 0]).clamp_min(1e-6)
        xc = x[:, L // 2, 0].float() / g[:, None]
        dec = dict(self.decompose_batch(xc))
        ev = dec["events"]
        ev.amp = ev.amp * g[ev.epoch]
        lg = 2 * N[:, 0][:, None, None]
        dec.update(logP=dec["logP"] + lg, resid=dec["resid"] * g[:, None],
                   wave_events=dec["wave_events"] * g[:, None], gain=g)
        dec.update(N=N, E=self.epoch_state(dec, B), P=self.tokens(dec, B), codes={})
        return dec

    # ------------------------------------------------------------------ dictionary learning
    @torch.no_grad()
    def fit_transients(self, batches: Iterable[Tensor], n_iter: int = 2, merge_coh: float = 0.9,
                       min_use: int = 1, refresh_thresholds: bool = True) -> list[dict]:
        """Learn the transient atoms on ``batches`` of epochs (B, T) (or (B, L, C, T): the centre
        epoch of channel 0 is used). Returns and stores a per-iteration log."""
        if self.fixed_dict:
            return []
        xs = []
        for xb in batches:
            xb = xb.float().to(self.atoms.device)
            if xb.dim() == 4:
                xb = xb[:, xb.shape[1] // 2, 0]
            xs.append(xb)
        x = torch.cat(xs)
        B, T = x.shape
        fs, Lk = self.fs, self.atoms.shape[1]
        ks = torch.nonzero(self._meta.cls == CLS_TRANSIENT, as_tuple=True)[0].to(self.atoms.device)
        for it in range(int(n_iter)):
            d = self.d
            # E-step with the transients first: the shapes are learned where the transient class
            # fits; at inference the slow waves go first and compete by shape (inference.py)
            ev, resid = sparse_events(x, d, self.thresholds, self.budget, fs=fs,
                                      whiten=self.whiten, sw_p2p_min=3.0,
                                      order=(CLS_TRANSIENT, CLS_SLOWWAVE, CLS_BURST))
            atoms = self.atoms.clone()
            used, restarted, merged = {}, [], []
            for k in ks.tolist():
                sel = (ev.atom == k) & ~ev.periodic
                n_k = int(sel.sum())
                used[k] = n_k
                if n_k < min_use:
                    continue
                e = ev.select(sel)
                # own contribution added back to the residual: the segment is what the atom must
                # explain given all the other events
                seg = _segments(resid, e.epoch, e.onset, Lk, fs) + e.amp[:, None] * atoms[k][None]
                a = (e.amp[:, None] * seg).sum(0) / e.amp.pow(2).sum().clamp_min(1e-12)
                atoms[k] = a
            dk = project_constraints(replace(d, atoms=atoms.cpu(), quad=self.quad.cpu(),
                                             cls=d.cls.cpu(), band=d.band.cpu(), freq=d.freq.cpu(),
                                             dur=d.dur.cpu(), family=d.family.cpu()), fs=fs)
            atoms = dk.atoms.to(self.atoms.device)
            # merge near duplicates: the less used atom of a pair is restarted
            C = coherence(replace(dk, atoms=atoms.cpu()), same_class_only=True).abs()
            dead = [k for k in ks.tolist() if used[k] < min_use]
            for i, a in enumerate(ks.tolist()):
                for b in ks.tolist()[i + 1:]:
                    if float(C[a, b]) > merge_coh:
                        drop = a if used[a] < used[b] else b
                        if drop not in dead:
                            dead.append(drop)
                            merged.append((a, b))
            # restart from the highest-energy residual segments (low-passed by the projection)
            if dead:
                e_seg = resid.unfold(-1, Lk, Lk // 2).pow(2).sum(-1)       # (B, n_seg)
                top = e_seg.flatten().topk(len(dead)).indices
                for k, j in zip(dead, top.tolist()):
                    b, s = divmod(j, e_seg.shape[1])
                    atoms[k] = resid[b, s * (Lk // 2): s * (Lk // 2) + Lk]
                    restarted.append(k)
                dk = project_constraints(replace(dk, atoms=atoms.cpu()), fs=fs)
                atoms = dk.atoms.to(self.atoms.device)
            self.atoms.copy_(atoms)
            rec = reconstruct(ev, d, B, T, fs=fs)
            self.fit_log.append({"iter": it, "used": used, "restarted": restarted, "merged": merged,
                                 "resid_share": float(resid.pow(2).sum() / x.pow(2).sum()),
                                 "events_share": float(rec.pow(2).sum() / x.pow(2).sum())})
        if refresh_thresholds:
            self.thresholds.copy_(null_thresholds(self.d, phase_surrogate(x), fs=fs))
        return self.fit_log
