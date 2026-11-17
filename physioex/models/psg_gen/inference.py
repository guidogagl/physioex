"""Shared inference of :mod:`physioex.models.psg_gen` (R1, R5): the sequential, label-free
decomposition of an epoch batch into sparse events on a fixed dictionary plus a stationary
band background. Common to P1, P2 and P3; port of the designers' prototypes
(``_proto/p1lib.omp_conv``, ``_proto/p3infer.sparse_events``), re-ordered per class
(slow waves -> transients -> bursts) and on the centred-atom convention of
:mod:`physioex.models.psg_gen.dicts` (atom centre at ``L // 2``, onset = atom centre).

Everything is batched torch on the model device. Selection is by normalised correlation of
the (optionally whitened) residual with each atom, above a per-atom null threshold, with
non-maximum suppression (+- d/2, same class), a per-epoch budget per class, and a joint
least-squares refit of the amplitudes on the selected support (explaining-away).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from physioex.models.psg_gen.background import (
    background_estimate, change_indicator, floor_excess, whiten, whitening_filters,
)
from physioex.models.psg_gen.dicts import (
    BG_BANDS, CLS_BURST, CLS_SLOWWAVE, CLS_TRANSIENT, Dictionary, FS,
)

Tensor = torch.Tensor

# (field name -> dtype) of the Events rows
_EVENTS_DTYPES = dict(
    epoch=torch.long, atom=torch.long, cls=torch.long, onset=torch.float32,
    amp=torch.float32, phase=torch.float32, dur=torch.float32, score=torch.float32,
    periodic=torch.bool, channel=torch.long,
)
_FIELD_NAMES = tuple(_EVENTS_DTYPES)


@dataclass
class Events:
    """Flat event list, sorted by (epoch, onset): one row per selected atom placement."""
    epoch: Tensor    # (n,) long, index in the batch
    atom: Tensor     # (n,) long, index into the Dictionary
    cls: Tensor      # (n,) long
    onset: Tensor    # (n,) float, seconds from the epoch start, sample resolution, centre of the atom
    amp: Tensor      # (n,) float, signed amplitude in signal units (after joint LS)
    phase: Tensor    # (n,) float radians (bursts), 0 otherwise
    dur: Tensor      # (n,) float seconds
    score: Tensor    # (n,) float normalised correlation at selection
    periodic: Tensor # (n,) bool
    channel: Tensor  # (n,) long (0 for single channel)

    def to(self, device) -> "Events":
        return Events(*[getattr(self, f).to(device) for f in _FIELD_NAMES])

    def __len__(self) -> int:
        return self.epoch.shape[0]

    def select(self, mask: Tensor) -> "Events":
        """Rows where ``mask`` (n,) bool is true."""
        return Events(*[getattr(self, f)[mask] for f in _FIELD_NAMES])

    def by_class(self, cls: int) -> "Events":
        return self.select(self.cls == cls)

    @staticmethod
    def cat(*evs: "Events") -> "Events":
        """Concatenation, re-sorted by (epoch, onset)."""
        if not evs:
            raise ValueError("cat of no events: use Events.empty(device)")
        out = Events(*[torch.cat([getattr(e, f) for e in evs]) for f in _FIELD_NAMES])
        return _sorted_by_epoch_onset(out)

    @staticmethod
    def empty(device) -> "Events":
        return Events(*[torch.zeros(0, dtype=_EVENTS_DTYPES[f], device=device) for f in _FIELD_NAMES])

    def to_dict(self) -> dict[str, list]:
        """JSON-serialisable rows (per-night dump)."""
        return {f: getattr(self, f).tolist() for f in _FIELD_NAMES}


def _sorted_by_epoch_onset(ev: Events) -> Events:
    if len(ev) < 2:
        return ev
    _, p1 = torch.sort(ev.onset, stable=True)  # by onset
    _, p2 = torch.sort(ev.epoch[p1], stable=True)  # then by epoch (stable keeps onset order)
    order = p1[p2]
    return Events(*[getattr(ev, f)[order] for f in _FIELD_NAMES])


def _support_boxes(d: Dictionary, fs: int = FS) -> tuple[Tensor, Tensor]:
    """Per atom: the effective support (90 % energy duration, :attr:`Dictionary.dur` in
    seconds) in samples, and the (K, L) box with ones over that support, centred on the atom
    centre (``L // 2``)."""
    A = d.atoms
    L = A.shape[1]
    n_d = torch.round(d.dur * fs).long().clamp_(1, L)
    start = L // 2 - n_d // 2
    j = torch.arange(L, device=A.device)[None, :]
    boxes = ((j >= start[:, None]) & (j < (start + n_d)[:, None])).to(A.dtype)
    return n_d, boxes


def _edge_forbid(T: int, L: int, n_d: Tensor) -> Tensor:
    """(K, Tp) bool, True where the atom's support does not lie in [0, T): the conv index t is
    valid for atom k when t in [C + n_d//2, C + n_d//2 + T - n_d], C = (L - 1) - L // 2."""
    C = (L - 1) - L // 2
    lo = C + n_d // 2
    hi = lo + T - n_d
    t = torch.arange(T + L - 1, device=n_d.device)
    return (t[None, :] < lo[:, None]) | (t[None, :] > hi[:, None])


def _score_maps(r: Tensor, A: Tensor, Q: Tensor, boxes: Tensor, L: int, use_quad: bool):
    """Correlation maps of the residual with the (centred) atoms.

    r (B, T), A/Q (K, L) unit-norm atoms and their quadrature partners, boxes (K, L). Returns
    (c, c2, score) at conv index t; the atom centre is at t - ((L - 1) - L // 2). score is the
    normalised correlation (Cauchy-Schwarz cosine, cos/sin combined for pairs).
    """
    Tp = r.shape[1] + L - 1
    c = F.conv1d(r[:, None], A[:, None], padding=L - 1)
    if use_quad:
        c2 = F.conv1d(r[:, None], Q[:, None], padding=L - 1)
        mag = (c * c + c2 * c2).sqrt()
    else:
        c2 = None
        mag = c.abs()
    e = F.conv1d((r * r)[:, None], boxes[:, None], padding=L - 1).clamp_min(1e-6)
    return c, c2, mag / e.sqrt()


def reconstruct(ev: Events, d: Dictionary, B: int, T: int, fs: int = FS,
                classes: Sequence[int] | None = None, exclude_periodic: bool = False) -> Tensor:
    """Synthesis of the events part: sum of the placed atoms, (B, T). Bursts are placed with
    their phase (amp * (cos phase * atom + sin phase * quad)); the other classes have phase 0
    and quad 0, so the plain atom. Onsets are atom centres (sample resolution)."""
    device = ev.epoch.device
    y = torch.zeros(B, T, dtype=torch.float32, device=device)
    n = len(ev)
    if n == 0:
        return y
    keep = torch.ones(n, dtype=torch.bool, device=device)
    if classes is not None:
        keep &= torch.isin(ev.cls, torch.as_tensor(list(classes), dtype=torch.long, device=device))
    if exclude_periodic:
        keep &= ~ev.periodic
    A = d.atoms.to(device, dtype=torch.float32)
    Q = d.quad.to(device, dtype=torch.float32)
    k = ev.atom[keep]
    w = ev.phase[keep].cos()[:, None] * A[k] + ev.phase[keep].sin()[:, None] * Q[k]
    s0 = torch.round(ev.onset[keep] * fs).long() - A.shape[1] // 2
    cols = s0[:, None] + torch.arange(A.shape[1], device=device)[None, :]
    valid = (cols >= 0) & (cols < T)
    idx = (ev.epoch[keep][:, None] * T + cols)[valid]
    y.view(-1).scatter_add_(0, idx, (ev.amp[keep][:, None] * w)[valid])
    return y


def support_mask(ev: Events, d: Dictionary, B: int, T: int, fs: int = FS, pad: float = 0.0) -> Tensor:
    """(B, T) bool, True on the event supports (effective duration, plus ``pad`` s each side)."""
    device = ev.epoch.device
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    n = len(ev)
    if n == 0:
        return m
    h = torch.ceil((ev.dur / 2 + pad) * fs).long()
    tau = torch.round(ev.onset * fs).long()
    a0 = (tau - h).clamp_(min=0)
    a1 = (tau + h + 1).clamp_(max=T)
    span = (a1 - a0).max().item()
    offs = torch.arange(span, device=device)[None, :]
    cols = a0[:, None] + offs
    valid = offs < (a1 - a0)[:, None]
    idx = (ev.epoch[:, None] * T + cols)[valid]
    acc = torch.zeros(B * T, dtype=torch.float32, device=device)
    acc.scatter_add_(0, idx, torch.ones(idx.numel(), dtype=torch.float32, device=device))
    return acc.view(B, T) > 0


def null_thresholds(d: Dictionary, bg: Tensor, fs: int = FS, q: float = 0.95, n_max: int = 192) -> Tensor:
    """(K,) per-atom selection threshold fixed from a null (R13): the q-quantile over the null
    epochs of the best normalised correlation of the atom on event-free background ``bg`` (B, T),
    residual and atoms whitened with the background's robust PSD."""
    if bg.shape[0] > n_max:
        bg = bg[:n_max]
    B, T = bg.shape
    device = bg.device
    L = d.atoms.shape[1]
    A = d.atoms.to(device, dtype=torch.float32)
    Q = d.quad.to(device, dtype=torch.float32)
    n_d, boxes = _support_boxes(d, fs)
    n_d, boxes = n_d.to(device), boxes.to(device)
    W, _ = whitening_filters(bg, L, fs=fs)
    bw = whiten(bg, W)
    wexp = W.median(0).values[None].expand(A.shape[0], L)
    Aw = whiten(A, wexp)
    Aw = Aw / Aw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Qw = whiten(Q, wexp)
    Qw = Qw / Qw.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # pair stays balanced after whitening
    use_quad = bool((Q.abs() > 0).any(-1).any())
    forbid = _edge_forbid(T, L, n_d)
    Tp = T + L - 1
    best = torch.empty(len(A), B, device=device)
    chunk = max(1, int(2 ** 26 / max(len(A) * Tp, 1)))  # ~2^26 map elements per chunk
    for i in range(0, B, chunk):
        s = _score_maps(bw[i:i + chunk], Aw, Qw, boxes, L, use_quad)[2]
        best[:, i:i + chunk] = s.masked_fill(forbid[None], -1.0).amax(-1).t()
    return best.quantile(q, dim=1)


SW_SCORE_FLOOR = 0.35  # minimum shape correlation of a slow wave that meets the p2p rule


def _sparse_chunk(x: Tensor, d: Dictionary, thr: Tensor, budget: dict[int, int], fs: int,
                  whiten_w: Tensor | None, order: Sequence[int], amp_cap: float,
                  ridge: float, nms: bool, explain: bool, sw_p2p_min: float | None) -> tuple[Events, Tensor]:
    """Greedy sequential selection on one chunk of epochs (epochs are independent)."""
    B, T = x.shape
    device = x.device
    L = d.atoms.shape[1]
    C = (L - 1) - L // 2
    Tp = T + L - 1
    A = d.atoms.to(device, dtype=torch.float32)
    Q = d.quad.to(device, dtype=torch.float32)
    cls = d.cls.to(device)
    dur = d.dur.to(device)
    n_d, boxes = _support_boxes(d, fs)
    n_d, boxes = n_d.to(device), boxes.to(device)
    use_quad_all = (Q.abs() > 0).any(-1)
    # selection (scoring) domain: whitened residual and whitened atoms (matched filter in the
    # coloured noise); amplitudes are fitted in the original domain
    Aw, Qw = A, Q
    if whiten_w is not None:
        wexp = whiten_w.median(0).values[None].expand(A.shape[0], L)
        Aw = whiten(A, wexp)
        Aw = Aw / Aw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Qw = whiten(Q, wexp)
        Qw = Qw / Qw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    rms = x.pow(2).mean(-1).sqrt()
    amax = A.abs().amax(-1).clamp_min(1e-6)
    r = x.clone()
    max_cols = 2 * sum(max(int(v), 0) for v in budget.values())
    cols = torch.zeros(B, T, max_cols, device=device)
    col_k = torch.full((B, max_cols), -1, dtype=torch.long, device=device)
    amps = torch.zeros(B, max_cols, device=device)
    ncol = torch.zeros(B, dtype=torch.long, device=device)
    tg = torch.arange(Tp, device=device) - C  # atom-centre coordinate of the conv index
    ev_b, ev_col, kk_all, tt_all, sc_all = [], [], [], [], []
    kt = torch.nonzero(cls == CLS_TRANSIENT, as_tuple=True)[0]
    if kt.numel() == 0 or int(budget.get(CLS_TRANSIENT, 0)) <= 0 or CLS_TRANSIENT not in tuple(order):
        kt = None
    for ccls in order:
        bcls = int(budget.get(ccls, 0))
        ks = torch.nonzero(cls == ccls, as_tuple=True)[0]
        if bcls <= 0 or ks.numel() == 0:
            continue
        A_c, Q_c = A[ks], Q[ks]
        thr_c, n_d_c, boxes_c = thr[ks], n_d[ks], boxes[ks]
        if sw_p2p_min and ccls == CLS_SLOWWAVE:
            # AASM defines a slow wave by its amplitude: with the p2p rule active the rule is the
            # acceptance criterion and the null threshold is capped at a shape floor
            thr_c = thr_c.clamp_max(SW_SCORE_FLOOR)
        quad_rows = use_quad_all[ks]
        use_quad = bool(quad_rows.any())
        forbid = _edge_forbid(T, L, n_d_c)[None].expand(B, -1, -1).clone() if nms \
            else _edge_forbid(T, L, n_d_c)[None].expand(B, -1, -1)
        count = torch.zeros(B, dtype=torch.long, device=device)
        p2p_atom = (A_c.max(-1).values - A_c.min(-1).values) if (sw_p2p_min and ccls == CLS_SLOWWAVE) else None
        for _ in range(B * bcls):
            rw = whiten(r, whiten_w) if whiten_w is not None else r
            c, c2, score = _score_maps(rw, Aw[ks], Qw[ks], boxes_c, L, use_quad)
            s = score.masked_fill(forbid, -1.0)
            s = s.masked_fill(count[:, None, None] >= bcls, -1.0)
            if p2p_atom is not None:
                c_o = c if whiten_w is None else F.conv1d(r[:, None], A_c[:, None], padding=L - 1)
                s = s.masked_fill(c_o.abs() * p2p_atom[None, :, None] < sw_p2p_min, -1.0)
            if kt is not None and ccls == CLS_SLOWWAVE:
                # class competition by shape: a slow-wave candidate that a transient atom fits
                # better within +-0.25 s (a K-complex meeting the p2p rule) is left to the
                # transient pass
                st = _score_maps(rw, Aw[kt], Qw[kt], boxes[kt], L, False)[2].amax(1, keepdim=True)
                hw = int(0.25 * fs)
                st = F.max_pool1d(st, 2 * hw + 1, 1, hw)
                s = s.masked_fill(score < st, -1.0)
            best, idx = s.reshape(B, -1).max(-1)
            k_loc = idx // Tp
            t = idx % Tp
            ok = best >= thr_c[k_loc]
            if not ok.any():
                break
            bi = torch.nonzero(ok, as_tuple=True)[0]
            kk_l, tt = k_loc[bi], t[bi]
            tau = tt - C  # onset, atom centre, samples
            ev_b.append(bi)
            ev_col.append(ncol[bi])
            kk_all.append(ks[kk_l])
            tt_all.append(tt)
            sc_all.append(best[bi])
            count[bi] += 1
            if nms:
                # same-class candidates whose centre lies within +- 1/2 * max(d, d') of the
                # chosen centre are excluded (R5), as is the chosen (atom, onset) itself
                half = 0.5 * torch.maximum(n_d_c[kk_l, None].float(), n_d_c[None, :].float())
                win = (tg[None, None, :] >= (tau[:, None, None] - half[:, :, None])) \
                    & (tg[None, None, :] <= (tau[:, None, None] + half[:, :, None]))
                forbid[bi] = forbid[bi] | win
                forbid[bi, kk_l, tt] = True
            col = ncol[bi]
            pos = tau[:, None] + torch.arange(L, device=device)[None, :] - L // 2
            valid = (pos >= 0) & (pos < T)
            rows = bi[:, None].expand(-1, L)
            cols[rows[valid], pos[valid], col[:, None].expand(-1, L)[valid]] = A_c[kk_l][valid]
            col_k[bi, col] = ks[kk_l]
            has = quad_rows[kk_l]
            if has.any():
                v2 = valid & has[:, None]
                cols[rows[v2], pos[v2], (col + 1)[:, None].expand(-1, L)[v2]] = Q_c[kk_l][v2]
                col_k[bi[has], col[has] + 1] = ks[kk_l[has]]
            ncol[bi] += 1 + has.long()
            if explain:
                # joint LS refit on the support of every active epoch (explaining-away), ridged
                m = int(ncol[bi].max().item())
                Xb = cols[bi, :, :m]
                G = torch.bmm(Xb.transpose(1, 2), Xb) + ridge * torch.eye(m, device=device)[None]
                sol = torch.linalg.solve(G, torch.bmm(Xb.transpose(1, 2), x[bi][:, :, None]))[:, :, 0]
                cap = amp_cap * rms[bi][:, None] / amax[col_k[bi, :m]]
                sol = torch.clamp(sol, -cap, cap)
                amps[bi, :m] = sol
                r[bi] = x[bi] - torch.bmm(Xb, sol[:, :, None])[:, :, 0]
            else:
                # matching-pursuit ablation: project on the current residual, no joint refit
                cc = cols[bi, :, col]
                a1 = (cc * r[bi]).sum(-1) / (cc * cc).sum(-1).clamp_min(1e-9)
                r[bi] -= a1[:, None] * cc
                amps[bi, col] = a1
                if has.any():
                    c2_ = cols[bi[has], :, col[has] + 1]
                    a2 = (c2_ * r[bi[has]]).sum(-1) / (c2_ * c2_).sum(-1).clamp_min(1e-9)
                    r[bi[has]] -= a2[:, None] * c2_
                    amps[bi[has], col[has] + 1] = a2
    if not kk_all:
        return Events.empty(device), r
    ev_b = torch.cat(ev_b)
    ev_col = torch.cat(ev_col)
    kk = torch.cat(kk_all)
    tt = torch.cat(tt_all)
    sc = torch.cat(sc_all)
    burst = use_quad_all[kk]
    a1 = amps[ev_b, ev_col]
    a2 = amps[ev_b, ev_col + 1]
    amp = torch.where(burst, (a1 * a1 + a2 * a2).sqrt(), a1)
    phase = torch.where(burst, torch.atan2(a2, a1), torch.zeros_like(a1))
    ev = Events(
        epoch=ev_b, atom=kk, cls=cls[kk], onset=(tt - C).to(torch.float32) / fs,
        amp=amp, phase=phase, dur=dur[kk], score=sc,
        periodic=torch.zeros(kk.numel(), dtype=torch.bool, device=device),
        channel=torch.zeros(kk.numel(), dtype=torch.long, device=device))
    return _sorted_by_epoch_onset(ev), r


def sparse_events(x: Tensor, d: Dictionary, thresholds: Tensor, budget: dict[int, int],
                  fs: int = FS, whiten_w: Tensor | None = None,
                  order: Sequence[int] = (CLS_SLOWWAVE, CLS_TRANSIENT, CLS_BURST),
                  amp_cap: float = 5.0, ridge: float = 1e-3, nms: bool = True,
                  explain: bool = True, sw_p2p_min: float | None = None,
                  chunk: int = 48, whiten: bool = True, ecg_guard: bool = True,
                  ecg_budget: int = 40) -> tuple[Events, Tensor]:
    """One pass of the sequential selection (R1, R5) on an epoch batch x (B, T).

    Classes are processed in ``order`` (slow waves on the (whitened) residual, then transients,
    then bursts). Selection is by normalised correlation above the per-atom ``thresholds``
    (K,) with non-maximum suppression (+- d/2, same class) and a per-epoch budget per class
    (budget keys: CLS_*). Amplitudes by joint least squares on the selected support
    (``explain=True``; ``explain=False`` gives matching-pursuit projections, ablation), capped
    at ``amp_cap`` x epoch RMS. ``whiten_w`` (B, L) is the zero-phase whitening filter applied
    to the residual (and, by its median, to the atoms) for the selection; amplitudes stay in the
    original domain. With ``whiten_w=None`` the filter is estimated from ``x`` unless
    ``whiten=False`` (ablation). Returns (events, residual (B, T)).
    """
    B, T = x.shape
    device = x.device
    if whiten_w is None and whiten:
        # the null thresholds are calibrated on whitened scores (null_thresholds): scoring the
        # raw signal against them lets 1/f power pass as slow waves and transients
        whiten_w, _ = whitening_filters(x, d.atoms.shape[1], fs=fs)
    evs, rs = [], []
    for i in range(0, B, chunk):
        W = None if whiten_w is None else whiten_w[i:i + chunk]
        xi = x[i:i + chunk]
        per = Events.empty(device)
        if ecg_guard and int(budget.get(CLS_TRANSIENT, 0)) > 0:
            # ECG guard: a transient-only pass with a wide budget finds a periodic train; its
            # beats are kept as periodic events outside the transient budget and removed before
            # the budgeted pass, so a train cannot use up the budget of the real transients
            pe, _ = _sparse_chunk(xi, d, thresholds, {CLS_TRANSIENT: ecg_budget}, fs, W,
                                  (CLS_TRANSIENT,), amp_cap, ridge, nms, explain, None)
            flag = periodic_flag(pe)
            if flag.any():
                per = pe.select(flag)
                per.periodic = torch.ones(len(per), dtype=torch.bool, device=device)
                xi = xi - reconstruct(per, d, xi.shape[0], T, fs=fs)
        ev, r = _sparse_chunk(xi, d, thresholds, budget, fs, W, order,
                              amp_cap, ridge, nms, explain, sw_p2p_min)
        ev = _sorted_by_epoch_onset(Events.cat(per, ev)) if len(per) else ev
        ev.epoch = ev.epoch + i
        evs.append(ev)
        rs.append(r)
    ev = Events.cat(*evs) if evs else Events.empty(device)
    if ecg_guard and len(ev):
        ev.periodic = ev.periodic | periodic_flag(ev)  # budgeted transients on the train too
    return ev, torch.cat(rs)


def periodic_flag(ev: Events, cv_max: float = 0.12, min_count: int = 4,
                  period: tuple[float, float] = (0.33, 2.0)) -> Tensor:
    """(n,) bool: True on the transients of an epoch that form a quasi-periodic train (the ECG
    guard). Per epoch, all transient atoms together (a beat can be fitted by different atoms),
    sorted by onset: the median inter-onset interval must lie in ``period`` (30-180 bpm), and a
    transient is flagged when an interval to a neighbour is within ``cv_max`` of that median;
    the epoch needs >= ``min_count`` flagged transients, else none is flagged. Non-transient
    events are never flagged."""
    n = len(ev)
    out = torch.zeros(n, dtype=torch.bool, device=ev.epoch.device)
    tr = (ev.cls == CLS_TRANSIENT).nonzero(as_tuple=True)[0]
    if tr.numel() < min_count:
        return out
    for e in ev.epoch[tr].unique().tolist():
        idx = tr[ev.epoch[tr] == e]
        if idx.numel() < min_count:
            continue
        idx = idx[torch.argsort(ev.onset[idx])]
        ioi = ev.onset[idx][1:] - ev.onset[idx][:-1]
        med = ioi.median()
        if not (period[0] <= float(med) <= period[1]):
            continue
        good = (ioi - med).abs() <= cv_max * med                 # (m-1,) interval i -> i+1
        flag = torch.zeros(idx.numel(), dtype=torch.bool, device=out.device)
        flag[:-1] |= good
        flag[1:] |= good
        if int(flag.sum()) >= min_count:
            out[idx[flag]] = True
    return out


def counts(ev: Events, B: int, exclude_periodic: bool = True) -> Tensor:
    """(B, 3) number of events per class per epoch; periodic events (ECG) excluded."""
    c = torch.zeros(B, 3, dtype=torch.long, device=ev.epoch.device)
    n = len(ev)
    if n == 0:
        return c
    keep = torch.ones(n, dtype=torch.bool, device=ev.epoch.device)
    if exclude_periodic:
        keep &= ~ev.periodic
    idx = ev.epoch[keep] * 3 + ev.cls[keep]
    c.view(-1).index_add_(0, idx, torch.ones(idx.numel(), dtype=torch.long, device=idx.device))
    return c


def decompose(x: Tensor, d: Dictionary, thresholds: Tensor, budget: dict[int, int],
              fs: int = FS, passes: int = 2, bands=BG_BANDS, whiten: bool = True, **kw) -> dict:
    """The full sequential pipeline (R1) on an epoch batch: whitening from the robust PSD ->
    sparse_events -> background estimate on the masked residual -> re-whitening with the new
    background; ``passes`` times. ``kw`` is forwarded to :func:`sparse_events` (``sw_p2p_min``
    defaults to the AASM 20 % rule, 3.0 units). Returns
    ``{"events", "logP", "centers", "wave_events", "resid", "mask", "eps", "change", "counts"}``.
    """
    kw.setdefault("sw_p2p_min", 3.0)
    B, T = x.shape
    L = d.atoms.shape[1]
    W = whitening_filters(x, L, fs=fs)[0] if whiten else None  # whiten=False: ablation
    ev, resid = Events.empty(x.device), x
    for _ in range(max(int(passes), 1)):
        ev, resid = sparse_events(x, d, thresholds, budget, fs=fs, whiten_w=W, whiten=whiten, **kw)
        mask = support_mask(ev, d, B, T, fs=fs)
        bg = background_estimate(resid, mask, fs=fs, bands=bands)
        if whiten:
            W, _ = whitening_filters(resid, L, fs=fs)  # re-whiten with the (masked) background PSD
    logP = bg["logP"]
    ev.periodic = ev.periodic | periodic_flag(ev)  # keep the ECG-guard flags of sparse_events
    fe = floor_excess(resid, logP, fs=fs, bands=bands)
    return {
        "events": ev,
        "logP": logP,
        "centers": bg["centers"],
        "wave_events": reconstruct(ev, d, B, T, fs=fs),
        "resid": resid,
        "mask": mask,
        "eps": fe["eps"],
        "change": change_indicator(logP),
        "counts": counts(ev, B),
    }


def topk_per_token(x: Tensor, d: Dictionary, k: int = 2, token: float = 2.0, fs: int = FS) -> Events:
    """The matched-filter baseline of the original proposal (R13 ablation): top-k normalised
    correlations per ``token`` s and per class, projection amplitudes, no explaining-away, no
    NMS. No whitening, no refit."""
    B, T = x.shape
    device = x.device
    L = d.atoms.shape[1]
    C = (L - 1) - L // 2
    A = d.atoms.to(device, dtype=torch.float32)
    Q = d.quad.to(device, dtype=torch.float32)
    cls = d.cls.to(device)
    dur = d.dur.to(device)
    n_d, boxes = _support_boxes(d, fs)
    n_d, boxes = n_d.to(device), boxes.to(device)
    quad = (Q.abs() > 0).any(-1)
    use_quad = bool(quad.any())
    forbid = _edge_forbid(T, L, n_d)[None]
    seg = int(token * fs)
    n_tok = T // seg
    ev_b, ev_k, ev_t, ev_a, ev_ph, ev_s = [], [], [], [], [], []
    chunk = max(1, int(2 ** 26 / max(len(A) * (T + L - 1), 1)))
    for i in range(0, B, chunk):
        c, c2, score = _score_maps(x[i:i + chunk], A, Q, boxes, L, use_quad)
        s = score.masked_fill(forbid, -1.0)
        for b in range(c.shape[0]):
            for ccls in (CLS_BURST, CLS_TRANSIENT, CLS_SLOWWAVE):
                ks = torch.nonzero(cls == ccls, as_tuple=True)[0]
                if ks.numel() == 0:
                    continue
                sb = s[b, ks]
                for t0 in range(n_tok):
                    segm = sb[:, t0 * seg:(t0 + 1) * seg].clone()
                    for _ in range(k):
                        v = segm.max()
                        if v <= 0:
                            break
                        fidx = int(segm.flatten().argmax())
                        ki, ti = divmod(fidx, seg)
                        t = t0 * seg + ti
                        kk = ks[ki]
                        if bool(quad[kk]):
                            amp = (c[b, kk, t].pow(2) + c2[b, kk, t].pow(2)).sqrt()
                            ph = torch.atan2(c2[b, kk, t], c[b, kk, t])
                        else:
                            amp, ph = c[b, kk, t], c[b, kk, t] * 0
                        ev_b.append(i + b)
                        ev_k.append(kk)
                        ev_t.append(t)
                        ev_a.append(amp)
                        ev_ph.append(ph)
                        ev_s.append(v)
                        segm[ki, max(ti - 10, 0):ti + 10] = 0.0
    if not ev_k:
        return Events.empty(device)
    kk = torch.tensor(ev_k, dtype=torch.long, device=device)
    tt = torch.tensor(ev_t, dtype=torch.long, device=device)
    ev = Events(
        epoch=torch.tensor(ev_b, dtype=torch.long, device=device),
        atom=kk, cls=cls[kk], onset=(tt - C).to(torch.float32) / fs,
        amp=torch.tensor([float(a) for a in ev_a], dtype=torch.float32, device=device),
        phase=torch.tensor([float(p) for p in ev_ph], dtype=torch.float32, device=device),
        dur=dur[kk],
        score=torch.tensor([float(s_) for s_ in ev_s], dtype=torch.float32, device=device),
        periodic=torch.zeros(kk.numel(), dtype=torch.bool, device=device),
        channel=torch.zeros(kk.numel(), dtype=torch.long, device=device))
    return _sorted_by_epoch_onset(ev)
