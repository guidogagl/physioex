"""P3 inference: sequential decomposition (R1) with background-whitened matched filtering, greedy
explaining-away (GLS refit on the active set), NMS, closed-form false-alarm threshold, per-class
budgets, amplitude cap, periodicity cap (ECG guard). Two passes: background Q20 -> events ->
masked background -> events. CPU numpy; batched FFT over the whole dictionary.
"""
from __future__ import annotations

import numpy as np
from scipy.special import polygamma

import p3lib as L

FS = L.FS
DEFAULT_BUDGET = {"B": 8, "T": 6, "S": 20}
EDGE = FS  # 1 s at each epoch edge: no onset there (whitening edge transients); night-long overlap-add has no edges
RHO_MIN = {"B": 0.0, "T": 0.0, "S": 0.6}  # S: the wave must hold >= 36 % of its own support energy (raw domain)


class Prepared:
    """Dictionary spectra at nfft for one epoch length n."""

    def __init__(self, atoms, n):
        self.atoms, self.n = atoms, n
        self.nfft = 1 << (2 * n - 1).bit_length()
        K = len(atoms)
        self.m = np.array([len(a.wave) for a in atoms])
        self.cls = np.array([a.cls for a in atoms])
        self.dur = np.array([a.dur for a in atoms])
        self.peak = np.array([np.abs(a.wave).max() for a in atoms])
        self.W = np.stack([np.fft.rfft(a.wave, self.nfft) for a in atoms])
        self.Q = np.stack([np.fft.rfft(a.quad, self.nfft) if a.quad is not None else np.zeros(self.nfft // 2 + 1)
                           for a in atoms])
        self.is_b = np.array([a.quad is not None for a in atoms])
        self.box = np.stack([np.fft.rfft(np.ones(m), self.nfft) for m in self.m])
        self.f = np.fft.rfftfreq(self.nfft, 1 / FS)

    def whitener(self, P_bands, bands=L.BG_BANDS, f_min=0.3, f_max=35.0, interp=()):
        """Piecewise-constant background PSD from band powers -> gain G(f) = 1/sqrt(S(f)),
        log-smoothed over 0.5 Hz, zero below f_min. ``interp``: band indices whose level is replaced
        by log-log interpolation between the nearest non-interpolated bands (R3: the alpha band is
        excluded from the background at pass 0, before alpha runs are masked)."""
        f = self.f
        P_bands = np.array(P_bands, float).copy()
        if len(interp):
            keep = [i for i in range(len(bands)) if i not in interp]
            fc = np.log([np.sqrt(lo * hi) for lo, hi in bands])
            P_bands[list(interp)] = np.exp(np.interp(fc[list(interp)], fc[keep], np.log(P_bands[keep] + 1e-12)))
        # log-log linear interpolation of the band levels at the band centres, slope extrapolated
        # outside: a 6-parameter smooth PSD (the per-band state of the background)
        fc = np.log([np.sqrt(lo * hi) for lo, hi in bands])
        lp = np.log(P_bands + 1e-12)
        lf = np.log(np.maximum(f, 1e-3))
        logS = np.interp(lf, fc, lp)
        lo_slope = (lp[1] - lp[0]) / (fc[1] - fc[0])
        hi_slope = (lp[-1] - lp[-2]) / (fc[-1] - fc[-2])
        logS = np.where(lf < fc[0], lp[0] + lo_slope * (lf - fc[0]), logS)
        logS = np.where(lf > fc[-1], lp[-1] + hi_slope * (lf - fc[-1]), logS)
        G = np.exp(-0.5 * logS)
        # raised-cosine roll-offs: 0 below f_min, full at 0.5 Hz; full at 30 Hz, 0 above f_max
        lo_ramp = np.clip((f - f_min) / (0.5 - f_min), 0, 1)
        hi_ramp = np.clip((f_max - f) / (f_max - 30.0), 0, 1)
        G *= 0.5 * (1 - np.cos(np.pi * lo_ramp)) * 0.5 * (1 - np.cos(np.pi * hi_ramp))
        return G


def _corr(Rw, prep, G, norms):
    """Batched correlation of the (already whitened) residual spectrum Rw with the whitened atoms
    conj(Phi) * G, centred at the atom middle; bursts return the analytic magnitude and phase."""
    nfft, n = prep.nfft, prep.n
    c = np.fft.irfft(Rw[None] * np.conj(prep.W) * G[None], nfft, axis=1)
    out = np.empty((len(prep.atoms), n))
    ph = np.zeros_like(out)
    for k in range(len(prep.atoms)):
        out[k] = np.roll(c[k], prep.m[k] // 2)[:n]
    if prep.is_b.any():
        s = np.fft.irfft(Rw[None] * np.conj(prep.Q) * G[None], nfft, axis=1)
        for k in np.where(prep.is_b)[0]:
            sk = np.roll(s[k], prep.m[k] // 2)[:n]
            ph[k] = np.arctan2(sk, out[k])
            out[k] = np.hypot(out[k], sk)
    return out / norms[:, None], ph


def _local_energy(rw, prep, box=None, mw=None):
    """sqrt of the residual energy inside each atom's effective support (99 % energy of the
    whitened atom), so that rho = |c| / e stays a Cauchy-Schwarz cosine."""
    box = prep.box if box is None else box
    mw = prep.m if mw is None else mw
    R2 = np.fft.rfft(rw * rw, prep.nfft)
    e = np.fft.irfft(R2[None] * box, prep.nfft, axis=1)
    out = np.empty((len(prep.atoms), prep.n))
    for k in range(len(prep.atoms)):
        out[k] = np.roll(e[k], mw[k] // 2)[:prep.n]
    return np.sqrt(np.maximum(out, 1e-12))


def whitened_support(prep, G, frac=0.99):
    """Effective support (samples) of each whitened atom and the matching box spectra."""
    pw = np.fft.irfft(prep.W * G[None], prep.nfft, axis=1)
    pw = np.roll(pw, prep.nfft // 2 - 0, axis=1)  # atom centred near nfft//2 + m//2
    e = pw ** 2
    mw, box = np.empty(len(prep.atoms), int), np.empty_like(prep.box)
    for k in range(len(prep.atoms)):
        c = np.cumsum(e[k]) / e[k].sum()
        i0, i1 = np.searchsorted(c, (1 - frac) / 2), np.searchsorted(c, 1 - (1 - frac) / 2)
        mw[k] = max(i1 - i0, prep.m[k])
        box[k] = np.fft.rfft(np.ones(mw[k]), prep.nfft)
    return mw, box


def _place(atom, ti, phase, n):
    w = atom.wave if atom.quad is None else np.cos(phase) * atom.wave - np.sin(phase) * atom.quad
    m = len(w)
    v = np.zeros(n)
    i0 = ti - m // 2
    a0, a1 = max(i0, 0), min(i0 + m, n)
    v[a0:a1] = w[a0 - i0:a1 - i0]
    return v


def fa_threshold(prep, alpha=0.05, dof=None):
    """Closed-form quality gate (R13): under H0 the whitened normalised correlation rho of an atom
    with m samples satisfies rho^2 ~ Beta(1/2,(m-1)/2) (Beta(1,(m-2)/2) for the analytic pair), so
    P(rho > t) ~ exp(-m t^2 / 2); Bonferroni over K*n positions at level alpha gives
    t_k = sqrt(2 ln(K n / alpha) / m_k)."""
    z2 = 2 * np.log(len(prep.atoms) * prep.n / alpha)
    return np.sqrt(z2 / prep.m)


def sparse_events(x, prep, G, budget=None, alpha=0.05, amp_cap=5.0, explain=True, nms=True,
                  period_cap=4, period_cv=0.12, max_iter=80, select="gain", classes="BTS",
                  p2p_min=None, fixed=()):
    """One pass of greedy inference on x with whitening gain G.
    select: "gain" = max whitened energy explained (likelihood gain) among candidates passing the
    quality gate; "rho" = max normalised correlation."""
    budget = budget or DEFAULT_BUDGET
    n, nfft = prep.n, prep.nfft
    phi_w_norm = np.sqrt(np.sum(np.abs(np.fft.irfft(prep.W * G[None], nfft, axis=1)) ** 2, 1)) + 1e-12
    mw, box = whitened_support(prep, G)
    X = np.fft.rfft(x, nfft)
    xw = np.fft.irfft(X * G, nfft)[:n]
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    thr = fa_threshold(prep, alpha)
    p2p_min = p2p_min or {}
    p2p_atom = np.array([a.wave.max() - a.wave.min() for a in prep.atoms])
    used = {c: 0 for c in budget}
    active, cols, cols_w, onsets = [], [], [], {}
    forbid = np.zeros((len(prep.atoms), n), bool)
    forbid[:, :EDGE] = True
    forbid[:, n - EDGE:] = True
    for cc in budget:
        if cc not in classes:
            forbid[prep.cls == cc] = True
    amps = np.zeros(0)

    def _add(k, ti, phs):
        a = prep.atoms[k]
        active.append((k, ti, phs))
        col = _place(a, ti, phs, n)
        cols.append(col)
        cols_w.append(np.fft.irfft(np.fft.rfft(col, nfft) * G, nfft)[:n])
        if nms:
            same = np.where(prep.cls == a.cls)[0]
            for kk in same:
                half = (prep.m[k] + prep.m[kk]) // 2
                forbid[kk, max(ti - half, 0):ti + half] = True

    def _refit():
        A, Aw = np.stack(cols, 1), np.stack(cols_w, 1)
        if explain:
            am, *_ = np.linalg.lstsq(Aw, xw, rcond=None)
        else:
            am = np.array([cw @ xw / (cw @ cw) for cw in cols_w])
        cap = np.array([amp_cap * rms / prep.peak[kk] for kk, _, _ in active])
        return np.clip(am, -cap, cap), A, Aw

    gains = []
    for (k, ti, phs) in fixed:  # events kept from an earlier pass (refitted jointly)
        _add(k, ti, phs)
        gains.append(float("inf"))
        used[prep.atoms[k].cls] += 1
    rw = xw.copy()
    if cols:
        amps, A, Aw = _refit()
        rw = xw - Aw @ amps
    for _ in range(max_iter):
        Rw = np.fft.rfft(rw, nfft)
        c, ph = _corr(Rw, prep, G, phi_w_norm)
        e = _local_energy(rw, prep, box, mw)
        rho = np.abs(c) / e
        ok = (rho >= np.maximum(thr, np.array([RHO_MIN[c] for c in prep.cls]))[:, None]) & ~forbid
        for cc in budget:
            if used[cc] >= budget[cc]:
                ok[prep.cls == cc] = False
        for cc, pm in p2p_min.items():  # AASM amplitude gate (slow waves): p2p of the candidate >= pm
            sel = prep.cls == cc
            ok[sel] &= (np.abs(c[sel]) / phi_w_norm[sel, None] * p2p_atom[sel, None]) >= pm
        if not ok.any():
            break
        score = (c * c if select == "gain" else rho) * ok
        k, ti = np.unravel_index(np.argmax(score), score.shape)
        a = prep.atoms[k]
        if period_cap and a.cls == "T":
            o = onsets.get(k, [])
            if len(o) >= period_cap:
                forbid[k] = True
                continue
            if len(o) >= 2:
                ioi = np.diff(sorted(o + [ti]))
                if ioi.std() / ioi.mean() < period_cv:
                    forbid[k] = True
                    continue
        onsets.setdefault(k, []).append(ti)
        _add(k, ti, ph[k, ti])
        gains.append(float(c[k, ti] ** 2))
        used[a.cls] += 1
        amps, A, Aw = _refit()
        rw = xw - Aw @ amps
    events = []
    for j, (k, ti, phs) in enumerate(active):
        a = prep.atoms[k]
        events.append(dict(cls=a.cls, k=int(k), name=a.name, tau=ti / FS, amp=float(amps[j]), dur=a.dur,
                           f=a.f, phase=float(phs), gain=gains[j], peak=float(abs(amps[j]) * prep.peak[k])))
    recon = np.stack(cols, 1) @ amps if cols else np.zeros(n)
    return events, recon


def background(x, events, win=4.0, hop=1.0, pass0=False):
    """Per-window band powers (R10: 4 s window, 1 s hop). pass0: Q20 over windows with the Gamma
    correction (closed form); else masked mean over windows of the event residual."""
    if pass0:
        P, t = L.band_power_windows(x, win, hop)
        M = np.array([L.n_bins(b, win) for b in L.BG_BANDS])
        q = np.nanquantile(P, 0.2, axis=0) * np.array([L.q20_gamma_factor(m) for m in M])
        return P, t, q
    mask = L.event_mask(events, len(x), pad=0.5)
    P, t = L.band_power_windows(x, win, hop, mask=mask)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lvl = np.nanmean(P, 0)
    if np.isnan(lvl).any():  # all windows masked: fall back to Q20 on x
        P0, _, q = background(x, events, win, hop, pass0=True)
        lvl = np.where(np.isnan(lvl), q, lvl)
    return P, t, lvl


def change_points(P, M, z=3.0):
    """Sparse state-change indicator per window (R10): |delta log P| above z * sd of the Gamma log
    noise, sd^2 = 2 * trigamma(M) (closed form)."""
    lp = np.log(P + 1e-12)
    d = np.abs(np.diff(lp, axis=0))
    sd = np.sqrt(2 * polygamma(1, M))[None]
    return (d > z * sd).any(1)


def decompose(x, prep, passes=2, p2p_sw=3.0, sw_first=True, **kw):
    """R1 sequential decomposition. Pass S: slow waves on the raw signal (shape gate + AASM
    amplitude gate p2p >= p2p_sw in units of g), no whitening, so dense N3 slow waves are events
    before any background is estimated. Then: masked background on the residual -> whiten ->
    bursts and transients (slow waves kept and refitted jointly); repeated ``passes`` times with
    the background re-estimated on the masked residual. Returns events, recon, window table,
    epoch background levels, residual."""
    x = np.asarray(x, float)
    n = len(x)
    G1 = np.ones(prep.nfft // 2 + 1)
    G1[prep.f < 0.3] = 0
    fixed = []
    if sw_first and "S" in kw.get("classes", "BTS"):
        ev_s, _ = sparse_events(x, prep, G1, classes="S", p2p_min={"S": p2p_sw}, **{k: v for k, v in kw.items() if k != "classes"})
        fixed = [(e["k"], int(round(e["tau"] * FS)), e["phase"]) for e in ev_s]
    events, recon = [dict(cls="S", k=k, tau=ti / FS, dur=prep.dur[k]) for k, ti, _ in fixed], np.zeros(n)
    P, t, lvl = background(x, events) if events else background(x, [], pass0=True)
    for p in range(passes):
        G = prep.whitener(lvl, interp=(3,) if p == 0 else ())
        events, recon = sparse_events(x, prep, G, fixed=fixed, **kw)
        P, t, lvl = background(x - recon, events)
    return dict(events=events, recon=recon, P=P, t=t, level=lvl, resid=x - recon)


def topk_per_token(x, prep, k_per_tok=1, tok=2.0, G=None):
    """Baseline of the proposal: top-k per 2-s token on the matched filter (raw or whitened),
    projection amplitude, no refit, no NMS."""
    n, nfft = prep.n, prep.nfft
    if G is None:
        G = np.ones(nfft // 2 + 1)
    G2 = G * G
    norms = np.sqrt(np.sum(np.abs(np.fft.irfft(prep.W * G[None], nfft, axis=1)) ** 2, 1)) + 1e-12
    X = np.fft.rfft(x, nfft)
    c, ph = _corr(X * G, prep, G, norms)
    T = int(tok * FS)
    events, recon = [], np.zeros(n)
    for cc in np.unique(prep.cls):
        sub = np.where(prep.cls == cc)[0]
        for t0 in range(0, n, T):
            seg = np.abs(c[sub, t0:t0 + T]).copy()
            for _ in range(k_per_tok):
                k, ti = np.unravel_index(np.argmax(seg), seg.shape)
                if seg[k, ti] <= 0:
                    break
                kk = sub[k]
                col = _place(prep.atoms[kk], t0 + ti, ph[kk, t0 + ti], n)
                amp = c[kk, t0 + ti] / norms[kk]  # projection in the whitened metric
                recon += amp * col
                events.append(dict(cls=cc, k=int(kk), name=prep.atoms[kk].name, tau=(t0 + ti) / FS,
                                   amp=float(amp), dur=prep.atoms[kk].dur, f=prep.atoms[kk].f))
                seg[k, max(ti - 10, 0):ti + 10] = 0
    return events, recon
