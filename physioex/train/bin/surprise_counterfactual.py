"""Counterfactual edits with the next-token model on patch codes (plan of 26.11.2026, S6, xmai-lab/sleep-lens#23).

    python -m physioex.train.bin.surprise_counterfactual --codec CODEC_RUN --ar AR_RUN --codes CODES_DUMP \\
        --scan CONCEPT_DUMP --surprise AR_RUN/surprise --out OUT.json [--n-samples 8] [--stager]

Protocol V4 of :mod:`physioex.train.bin.concept_counterfactual`, with the 2 s patch codec as signal model and
the fill in the middle (FIM) of the next-token model (:mod:`physioex.models.eeg_ar`) as editor. The reference
events (DREAMS experts, the pool K-complex rule, MASS SS01 arousals) only say where the event is; the judges of
the edited signal are detectors: ``concept_refs`` K-complexes on the 30 s epoch, and the M1 arousal rule
(:func:`physioex.explain.manual_rules.detect_arousal`, EEG only, so onsets in N1-N3) from 40 s before to 20 s
after the event. The signal comes from the codes alone: x̂ is the decode of the original codes, x' the decode
of the edited ones (codec scale; the detectors read it in uV with the ``concept_scan`` night scale).

Rules fixed before the first result (27.11.2026):

* an event counts when its judge fires on the event tokens (+-1 token) in x̂ (as V4);
* span = the grid-0 patches that cover the event tokens +-1 token, at most ``MAX_SPAN`` patches;
* **remove**: the span refilled by FIM (``n_samples`` samples, temperature 1), decoded; efficacy = share of
  (event, sample) pairs where the judge no longer fires on the event tokens;
* **control (a), silent**: the same FIM edit on a span of equal length without events (same epoch for
  K-complexes, after the event for arousals); rate = share where the judge stops firing on the event;
* **control (b), random**: the span's codes replaced by those of a random span of the night, no FIM
  (reported without gate);
* **realism**: mean surprise S of the span and the ``FOLLOW`` patches after it (plain layout, one window that
  ends there) for the original, each FIM edit, control (b), and one event-free stretch of the same length and
  stage per event (no reference or rule token); pass = median of the FIM edits inside the interquartile range
  of the event-free stretches and below control (b) (paired Wilcoxon, one-sided p < 0.05);
* **insert** (K-complexes): the codes of a K-complex span of the same night pasted at a random patch of an
  epoch without one (no reference token, judge silent in x̂), at least one patch from the epoch borders;
  efficacy = the judge fires on the donor's event tokens moved there (rows where it fired before are left
  out); control: as many patches of the donor epoch without events pasted at the same place;
* **locality**: RMS of x' - x̂ inside the span, on the +-1 patch ring and beyond (0 by construction);
* **stager** (``--stager``): ΔM = log p of the scored stage under the combination's EEG stager (the path of
  the ``concept_scan`` dump: its encoder, its stager, voting over sequences) on x' minus on x̂, at the edited
  epoch, context = the decoded epochs around it (reflected at the night borders); first FIM sample only;
* **closure rule** (plan point 5, DREAMS): starts = grid-0 patches whose night z-scored surprise is > 2 and
  a local maximum, in N2/N3; end = :func:`physioex.models.eeg_ar.close_event` with the night's 75th
  percentile of patch surprise as threshold; no end = state transition. Reported: closures, lengths,
  overlap with any reference or rule mask against circular shifts, and the removal of the expert
  K-complexes a closed span covers;
* gate (plan S6) on DREAMS K-complexes: remove >= 0.9, control (a) <= 0.07, realism pass, insert >= 0.62.

T4 of plan rev. 2 of xmai-lab/sleep-lens#17 (#29; rules registered on the issue before the launch, 28.11.2026):
the codes of the learned tokenizer A (FSQ) and its decoder as signal model, no next-token model.

* ``--editor median`` (V4, deterministic, one sample): the span's codes become, on every FSQ dimension, the lower
  median of the levels of the free patches outside the span (all 4 tokens without reference or rule token and
  without a detection of the judge: K-complex, delta or spindle; arousal) of the same epoch; for arousals, of the
  judge window before the event. No free patch: the event is counted as ``no_free`` and left out. Control (a) is the
  same edit on the silent span (median of the free patches outside it); control (b) as above;
* without ``--ar``, the realism by surprise and the closure rule are not measurable (``None``);
* ``--check-codec`` (second judge, C = the zero-phase twin): C re-encodes the decoded signal on the 4 grids; token
  feature = mean over the 4 covering patches of C's FSQ values (``--check-feature``, default ``latent``, the best C
  feature of T2 on every set); score of class c (K-complex, arousal) = ``cos(f - mu, P_c) - cos(f - mu, P_bg)``,
  P = the T2 prototypes recomputed from C's codes on the pool test nights (:func:`pool_prototypes`), mu = the
  night mean of C(x̂) over its sleep tokens. C fires on a stretch when the score of one of its tokens exceeds the
  night's ``C_Q`` quantile of the scores in x̂ of the tokens of the set's stages without reference or rule token.
  Reported per set with the thresholds of S6: removal = share of the events where C fired in x̂ and no longer
  fires in x'; controls (a) and (b) the same way; insert = share where C fires on the moved tokens, among the
  insertions where it did not fire before;
* **realism without the AR** (in the gate in place of the realism of S6): the median over the events of the mean C
  score of the removed span must lie inside the interquartile range of the mean C score, in x̂, of the event-free
  stretches of the same length and stage (one per event); control (b) is reported;
* windows are decoded with the codec's receptive field + 1 patch of margin, so that they equal the decode of the
  whole night (S6 used 2 patches); the edit of a span reaches the signal up to the receptive field around it.

Writes ``OUT.json``: aggregate numbers only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

T, TPP, EPP = 60, 4, 15  # tokens per epoch, tokens per patch, patches per epoch
SPP = 200  # samples per patch
SETS = {  # name: (units (None = every unit but DREAMS and MASS SS01, as V4), reference, onset stages, judge)
    "dreams_kc": (("dreams",), "k_complex_vis1", (2, 3), "kc"),
    "pool_kc": (None, "k_complex_rule", (2, 3), "kc"),
    "mass_ss01_arousal": (("mass_ss01",), "arousal_ann", (1, 2, 3), "arousal"),
}
GATE = {"remove": 0.9, "ctrl_silent": 0.07, "insert": 0.62}
MAX_SPAN, FOLLOW = 10, 3
AR_BEFORE, AR_AFTER = 80, 40  # tokens of the arousal judge window around the event (40 s, 20 s)
MIN_ALIGN_R2 = 0.3
TOK = SPP // TPP  # samples per token
C_Q = 0.95  # T4: quantile of the night's event-free scores that the C judge must exceed
C_MARGIN = 3  # T4: patches decoded around a C window (C reads 1 patch around a patch, its whitener 32 samples)
C_CLASS = {"kc": "kc", "arousal": "arousal"}  # judge -> prototype class of the C judge


# ---------------------------------------------------------------------------- data
def load_night(a, unit, sid, L, scale_js):
    """Codes of grid 0 and the ``concept_scan`` masks of one night on the period's token grid."""
    from physioex.train.bin.concept_scan import load_refs

    key = f"{unit}/{sid}"
    pc, ps = os.path.join(a.codes, unit, f"{sid}.npz"), os.path.join(a.scan, unit, f"{sid}.npz")
    pr, pm = os.path.join(a.scan, "refs", unit, f"{sid}.npz"), os.path.join(a.scan, "m1", unit, f"{sid}.npz")
    sc = scale_js.get(key, {}).get("eeg", {}).get("scale")
    if sc is None or not all(os.path.exists(p) for p in (pc, ps, pr)):
        return None
    z, d = np.load(pc), np.load(ps)
    codes = np.ascontiguousarray(z["c0"][:, :L]).astype(np.int64)
    e0, P = int(z["e0"]), len(codes)
    n_ep = P // EPP
    ts, y = d["ts"].astype(np.int64), d["y"].astype(np.int64)
    j = ts - e0
    keep = np.flatnonzero((j >= 0) & (j < n_ep))
    tok = (j[keep][:, None] * T + np.arange(T)[None]).reshape(-1)
    src = (keep[:, None] * T + np.arange(T)[None]).reshape(-1)
    n_tok = n_ep * T
    stage = np.full(n_tok, -1, np.int64)
    stage[tok] = np.repeat(y[keep], T)
    masks = {}
    srcs = [("", load_refs(pr))] + ([("m1:", load_refs(pm))] if os.path.exists(pm) else [])  # both bit-packed
    for pre, mm in srcs:
        for k, v in mm.items():
            v = np.asarray(v)
            if k.startswith("_") or k.endswith("_epoch") or v.shape != (len(ts) * T,):
                continue
            m = np.zeros(n_tok, bool)
            m[tok] = v.astype(bool)[src]
            masks[pre + k] = m
    union = np.any(np.stack(list(masks.values())), 0) if masks else np.zeros(n_tok, bool)
    x_scan = d["x_eeg"][keep] if "x_eeg" in d.files else None
    return {"key": key, "codes": codes, "P": P, "n_ep": n_ep, "stage": stage, "dumped": stage >= 0,
            "masks": masks, "union": union, "scale": float(sc), "ep_scan": j[keep], "x_scan": x_scan}


def apply_edits(codes, a, b, edits):
    """Copy of patches ``a .. b - 1`` with the edits ``[(start patch, codes)]`` written where they overlap."""
    c = codes[a:b].copy()
    for p0, cc in edits:
        lo, hi = max(p0, a), min(p0 + len(cc), b)
        if lo < hi:
            c[lo - a:hi - a] = cc[lo - p0:hi - p0]
    return c


def decode_window(codec, codes, lo, hi, edits_list, dev, margin=None):
    """Decoded samples of patches ``lo .. hi - 1`` for each list of edits ``[(start patch, codes)]``; the default
    margin (the codec's receptive field + 1 patch) makes them equal to the decode of the whole night."""
    import torch

    margin = getattr(codec, "receptive_patches", 1) + 1 if margin is None else margin
    a, b = max(0, lo - margin), min(len(codes), hi + margin)
    stack = []
    for ed in edits_list:
        stack.append(apply_edits(codes, a, b, ed))
    with torch.no_grad():
        y = codec.decode_codes(torch.from_numpy(np.stack(stack)).to(dev)).float().cpu().numpy()
    return y[:, (lo - a) * SPP:(hi - a) * SPP]


def align_r2(codec, nd, dev, n=8):
    """Median R^2 of the decoded epochs against the ``concept_scan`` signal: same night, same epochs."""
    if nd["x_scan"] is None or not len(nd["ep_scan"]):
        return None
    idx = np.linspace(0, len(nd["ep_scan"]) - 1, min(n, len(nd["ep_scan"]))).astype(int)
    r2 = []
    for i in idx:
        jj = int(nd["ep_scan"][i])
        xh = decode_window(codec, nd["codes"], jj * EPP, (jj + 1) * EPP, [[]], dev)[0].astype(np.float64)
        xs = nd["x_scan"][i].astype(np.float64)
        r2.append(1 - ((xh - xs) ** 2).sum() / max(((xs - xs.mean()) ** 2).sum(), 1e-9))
    return float(np.median(r2))


# ---------------------------------------------------------------------------- judges
def judge_kc(u, fs):
    from physioex.train.bin.concept_counterfactual import detect

    return detect(u, fs)


def judge_arousal(u, fs, st_tok):
    from physioex.explain import concept_refs as CRF
    from physioex.explain.manual_rules import detect_arousal

    ar = detect_arousal(u.astype(np.float64), fs, None, st_tok)
    return {"arousal": CRF.samples_to_tokens(ar["arousal"] + ar["awakening"], len(u), fs)}


class Stager:
    """EEG stager of the combination behind the ``concept_scan`` dump, on decoded blocks of epochs."""

    def __init__(self, scan_dir, dev):
        from physioex.train.bin.pretrain_combo import load_s2, load_stagers

        meta = json.load(open(os.path.join(scan_dir, "meta.json")))
        run, self.model = load_s2(os.path.join(meta["combo_dir"], "eeg", "s2"), "eeg", dev)
        self.half = int(run["args"]["seq"]) // 2
        st, stats, smeta = load_stagers(os.path.join(meta["combo_dir"], meta["stager"]), ["eeg"], dev)
        self.stagers, self.dev = ({"eeg": st["eeg"]}, stats, smeta), dev
        self.seq = int(getattr(st["eeg"], "seq_len", 1))
        self.ctx = self.half + self.seq - 1
        self.errors = 0

    def logp(self, x):
        """``x (n, 3000)`` -> ``(n, 5)`` stage log-probabilities."""
        from physioex.train.bin.concept_scan import _reflect_night, _stage_logp
        from physioex.train.bin.pretrain_combo import encode_night

        n = len(x)
        xp, tp = _reflect_night(x.astype(np.float32), np.arange(n), self.half)
        z = encode_night(self.model, xp, tp, self.half)
        lat = {"eeg": {"N": z["N"], "E": z["E"], "P": z["P"], "y": np.zeros(n, np.int64),
                       "night": np.zeros(n, np.int64), "epoch": np.arange(n, dtype=np.int64)}}
        return _stage_logp(self.stagers, lat, ["eeg"], self.dev)["eeg"].astype(np.float32)

    def delta(self, codec, nd, jj, y_true, edits_list, dev):
        """ΔM of each edit list (index 0 = no edit) at epoch ``jj``, or None (night too short, or an error,
        counted in ``errors``: the stager is reported, not gated)."""
        try:
            return self._delta(codec, nd, jj, y_true, edits_list, dev)
        except Exception as e:  # noqa: BLE001
            self.errors += 1
            if self.errors == 1:
                print(f"[cf-ar] stager error (first): {e!r}", flush=True)
            return None

    def _delta(self, codec, nd, jj, y_true, edits_list, dev):
        lo_e, hi_e = max(0, jj - self.ctx), min(nd["n_ep"], jj + self.ctx + 1)
        if hi_e - lo_e < max(self.seq, 2 * self.half + 1):
            return None
        U = decode_window(codec, nd["codes"], lo_e * EPP, hi_e * EPP, edits_list, dev)
        lp = np.stack([self.logp(u.reshape(hi_e - lo_e, -1))[jj - lo_e] for u in U])
        return {"dM": (lp[1:, y_true] - lp[0, y_true]).tolist(),
                "flip": (lp[1:].argmax(1) != lp[0].argmax()).tolist()}


# ---------------------------------------------------------------------------- T4: V4 editor and the C judge
def median_codes(codes, levels, group):
    """Per-dimension lower median of the FSQ levels of ``codes (n, groups)`` -> ``(groups,)`` codes. The lower
    median of a dimension is one of its levels, so the result is a valid code."""
    radix = levels ** np.arange(group)
    d = (np.asarray(codes, np.int64)[..., None] // radix) % levels  # (n, groups, group) level indices
    return (np.sort(d, 0)[(len(d) - 1) // 2] * radix).sum(-1)


def pool_prototypes(scan, dump, fn, max_nights=0):
    """The T2 prototypes of one codec (pass 1 of :mod:`physioex.train.bin.semantic_gates`): the mean centred token
    feature over the sleep tokens of each class and of the background, on the pool test nights of the scan.
    ``fn``: codes ``(P, groups)`` (int64 tensor) -> ``(P, D)`` numpy features. Returns ``({class: unit vector},
    info)``."""
    import torch

    from physioex.train.bin.concept_scan import load_refs
    from physioex.train.bin.surprise_type import (CLASSES, OFFSETS, POOL_UNITS, SLEEP, centre, class_masks,
                                                  token_features, unit_cos)

    names = list(CLASSES) + ["background"]
    S, N, n_nights = {c: 0.0 for c in names}, {c: 0 for c in names}, 0
    for unit in POOL_UNITS:
        ud = os.path.join(scan, unit)
        k = 0
        for f in (sorted(f for f in os.listdir(ud) if f.endswith(".npz")) if os.path.isdir(ud) else []):
            if max_nights and k >= max_nights:
                break
            pr, pc = os.path.join(scan, "refs", unit, f), os.path.join(dump, unit, f)
            if not (os.path.exists(pr) and os.path.exists(pc)):
                continue
            d = np.load(os.path.join(ud, f))
            ts, stage = d["ts"], np.repeat(np.asarray(d["y"], np.int64), T)

            def keep(name, v):
                return not name.startswith("_") and not name.endswith("_epoch") and v.shape == stage.shape

            masks = {kk: v for kk, v in load_refs(pr).items() if keep(kk, v)}
            pm = os.path.join(scan, "m1", unit, f)
            if os.path.exists(pm):
                masks.update({f"m1:{kk}": v for kk, v in load_refs(pm).items() if keep(kk, v)})
            z = np.load(pc)
            per = {o: fn(torch.from_numpy(np.asarray(z[f"c{o}"], np.int64))) for o in OFFSETS if f"c{o}" in z.files}
            F = centre(token_features(per, ts, int(z["e0"])), stage).astype(np.float32)
            cm = class_masks(masks, len(stage))
            sleep = np.isin(stage, SLEEP) & np.isfinite(F).all(1)
            for c in names:
                N[c] += int((cm[c] & sleep).sum())
                S[c] = S[c] + F[cm[c] & sleep].sum(0, dtype=np.float64)
            k += 1
            n_nights += 1
    names = [c for c in names if N[c] > 0]
    P = unit_cos(np.stack([S[c] / N[c] for c in names]))
    return ({c: P[i].astype(np.float32) for i, c in enumerate(names)},
            {"n_pool_nights": n_nights, "names": names, "n_tokens": {c: N[c] for c in names},
             "cosine": np.round(P @ P.T, 3).tolist()})


class CJudge:
    """Second judge of T4 (#29): the spectral twin C on the decoded signal (rules in the module docstring)."""

    def __init__(self, run, dump, scan, dev, feature="latent", max_nights=0):
        from physioex.train.bin.patch_codec import load_codec
        from physioex.train.bin.semantic_gates import feature_fns

        _, self.codec = load_codec(run, dev)
        self.fn = feature_fns(self.codec)[feature]
        self.levels, self.dev = int(self.codec.vq_levels), dev
        self.P, self.info = pool_prototypes(scan, dump, self.feats, max_nights)
        self.F = self.mu = None

    def feats(self, c):
        return self.fn(c.to(self.dev)).float().cpu().numpy()

    def set_night(self, xhat, n_ep, stage):
        """Token features of C(x̂) over the night (``xhat``: ``(n,)`` tensor) and their mean over the sleep tokens."""
        from physioex.train.bin.patch_codec import OFFSETS, encode_signal
        from physioex.train.bin.surprise_type import SLEEP, token_features

        per = {o: self.feats(encode_signal(self.codec, xhat[o:], self.levels)) for o in OFFSETS}
        self.F = token_features(per, np.arange(n_ep), 0).astype(np.float32)
        ok = np.isin(stage, SLEEP) & np.isfinite(self.F).all(1)
        self.mu = self.F[ok].mean(0) if ok.any() else np.zeros(self.F.shape[1], np.float32)

    def score(self, F, c):
        """``cos(f - mu, P_c) - cos(f - mu, P_background)`` over the last axis."""
        f = F - self.mu
        return (f @ self.P[c] - f @ self.P["background"]) / np.maximum(np.linalg.norm(f, axis=-1), 1e-12)

    def window(self, U, lo, t0, t1):
        """Token features ``(V, t1 - t0, D)`` of the tokens ``t0 .. t1 - 1`` of the night from decoded signals
        ``U (V, n)`` that start at patch ``lo``: the grids and the mean over the covering patches of
        :func:`~physioex.train.bin.surprise_type.token_features`."""
        import torch

        from physioex.train.bin.patch_codec import OFFSETS

        x = torch.from_numpy(np.ascontiguousarray(U, np.float32)).to(self.dev)
        st = np.arange(t0, t1) * TOK - lo * SPP
        acc, cnt = None, np.zeros(t1 - t0)
        for o in OFFSETS:
            m = (x.shape[1] - o) // SPP
            if m <= 0:
                continue
            f = self.feats(self.codec.codes(x[:, o:o + m * SPP], self.levels))
            if acc is None:
                acc = np.zeros((len(U), t1 - t0, f.shape[-1]))
            p = (st - o) // SPP
            ok = (st >= o) & (p < m)
            acc[:, ok] += f[:, p[ok]]
            cnt[ok] += 1
        out = np.full(acc.shape, np.nan, np.float32)
        out[:, cnt > 0] = acc[:, cnt > 0] / cnt[cnt > 0, None]
        return out


# ---------------------------------------------------------------------------- events
def kc_events(nd, ref, stages):
    """Runs of the reference inside each dumped epoch of the stages: (event tokens, epoch)."""
    from physioex.explain import concept_refs as CRF

    out = []
    for jj in np.unique(np.flatnonzero(nd["dumped"]) // T):
        if nd["stage"][jj * T] not in stages:
            continue
        for ta, tb in CRF.runs(ref[jj * T:(jj + 1) * T]):
            out.append((jj * T + int(ta), jj * T + int(tb), int(jj)))
    return out


def arousal_events(nd, ref, stages):
    from physioex.explain import concept_refs as CRF

    out = []
    for ta, tb in CRF.runs(ref):
        lo, hi = ta - AR_BEFORE, tb + AR_AFTER
        if lo < 0 or hi > len(ref) or nd["stage"][ta] not in stages or nd["dumped"][lo:hi].mean() < 0.9:
            continue
        out.append((int(ta), int(tb), int(ta) // T))
    return out


def span_of(ta, tb, n_tok):
    """Event tokens +-1 token and the grid-0 patches that cover them."""
    ga, gb = max(0, ta - 1), min(n_tok, tb + 1)
    return ga, gb, ga // TPP, -(-gb // TPP)


# ---------------------------------------------------------------------------- one night
def night(a, nd, name, ref_key, stages, judge, model, codec, stager, dev, rng, gen, fs, surp, cj=None):
    import torch

    from physioex.explain import concept_refs as CRF
    from physioex.models.eeg_ar import close_event, window_around, window_surprise
    from physioex.train.bin.patch_codec import decode_signal

    codes, P, W, n = nd["codes"], nd["P"], a.window, a.n_samples
    ref, union, stage = nd["masks"][ref_key], nd["union"], nd["stage"]
    n_tok = nd["n_ep"] * T
    uv = nd["scale"]
    patch_free = np.zeros(P, bool)  # patches without any reference / rule token, all tokens scored
    pt = np.arange(nd["n_ep"] * EPP)
    patch_free[pt] = np.all([nd["dumped"][pt * TPP + k] & ~union[pt * TPP + k] for k in range(TPP)], 0)
    patch_stage = np.full(P, -1, np.int64)
    patch_stage[pt] = stage[pt * TPP]

    def free_span(n_p, st):
        ok = np.flatnonzero([patch_free[f:f + n_p + FOLLOW].all() and patch_stage[f] == st
                             for f in range(0, P - n_p - FOLLOW)])
        return int(rng.choice(ok)) if len(ok) else None

    def infill(lo, hi, k):
        s = window_around(P, lo, hi, W)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            f = model.infill(torch.from_numpy(codes[s:s + min(W, P)]), lo - s, hi - s, k, 1.0, gen)
        return f.cpu().numpy()

    def realism(items):
        """items: (start, n_patches, edits) -> mean S over the stretch and FOLLOW patches after it."""
        if model is None:  # no next-token model: not measurable
            return [None] * len(items)
        out = []
        for lo, n_p, ed in items:
            hi = lo + n_p + FOLLOW
            if hi > P:
                out.append(None)
                continue
            s = window_around(P, lo, hi, W, centre=False)
            out.append((apply_edits(codes, s, s + min(W, P), ed), lo - s, hi - s))
        ok = [i for i, it in enumerate(out) if it is not None]
        S = window_surprise(model, [out[i] for i in ok])
        res = [None] * len(items)
        for i, v in zip(ok, S):
            res[i] = float(v.mean())
        return res

    def judge_window(lo_p, hi_p, edits):
        U = decode_window(codec, codes, lo_p, hi_p, edits, dev) * uv
        if judge == "kc":
            return [judge_kc(u, fs) for u in U], U
        st_tok = stage[lo_p * TPP:hi_p * TPP]
        return [judge_arousal(u, fs, st_tok) for u in U], U

    def median_fill(ps, n_p):
        return np.repeat(median_codes(codes[ps], codec.fsq.levels, codec.fsq.group)[None], n_p, 0)

    def c_window(lo_p, hi_p, edits):
        """C scores (variants x tokens) of the patches lo_p - 1 .. hi_p, their first token, and the largest
        difference between the features of the first variant and those of the whole night (0 when it is x̂)."""
        lo_c, hi_c = max(0, lo_p - C_MARGIN), min(P, hi_p + C_MARGIN)
        t0, t1 = max(0, (lo_p - 1) * TPP), min(n_tok, (hi_p + 1) * TPP)
        Fw = cj.window(decode_window(codec, codes, lo_c, hi_c, edits, dev), lo_c, t0, t1)
        d = np.abs(Fw[0] - cj.F[t0:t1])
        return cj.score(Fw, ccls), t0, float(np.nanmax(d)) if np.isfinite(d).any() else None

    def c_fires(Sv, t0, ta_, tb_):
        v = Sv[ta_ - t0:tb_ - t0]
        return bool(tau is not None and np.isfinite(v).any() and np.nanmax(v) > tau)

    def c_mean(Sv, t0, ta_, tb_):
        v = Sv[max(0, ta_ - t0):tb_ - t0]
        return float(np.nanmean(v)) if np.isfinite(v).any() else None

    ccls, tau, s_hat = C_CLASS[judge], None, None
    if cj is not None:  # C on the decode of the whole night: centring mean and the night's threshold
        cj.set_night(decode_signal(codec, torch.from_numpy(codes).to(dev)), nd["n_ep"], stage)
        s_hat = cj.score(cj.F, ccls)
        ok = np.isin(stage, stages) & ~union & np.isfinite(s_hat)
        tau = float(np.quantile(s_hat[ok], C_Q)) if ok.sum() >= 100 else None

    key = "kc" if judge == "kc" else "arousal"
    evs = kc_events(nd, ref, stages) if judge == "kc" else arousal_events(nd, ref, stages)
    evs = [evs[i] for i in rng.permutation(len(evs))[: a.max_events]]
    rows, donors, base_by_ep = [], [], {}
    n_seen = {"events": len(evs), "too_long": 0, "not_detected": 0, "no_free": 0}  # too_long: also night borders
    for ta, tb, jj in evs:
        ga, gb, pa, pb = span_of(ta, tb, n_tok)
        n_p = pb - pa
        if n_p > MAX_SPAN or pa < 1 or pb > P - 1:  # FIM needs a prefix and a suffix
            n_seen["too_long"] += 1
            continue
        if judge == "kc":
            lo_p, hi_p = jj * EPP, (jj + 1) * EPP
        else:
            lo_p, hi_p = max(0, (ta - AR_BEFORE) // TPP), min(P, -(-(tb + AR_AFTER) // TPP))
        ev = slice(ga - lo_p * TPP, gb - lo_p * TPP)
        det0, U0 = judge_window(lo_p, hi_p, [[]])
        det0, U0 = det0[0], U0[0]
        if judge == "kc":
            base_by_ep[jj] = det0
        if not det0[key][ev].any():
            n_seen["not_detected"] += 1
            continue
        # free tokens: no reference or rule token, no detection of the judge (K-complexes: nor delta, spindle)
        quiet = ~union[lo_p * TPP:hi_p * TPP] & ~det0[key]
        if judge == "kc":
            quiet &= ~det0["delta"] & ~det0["spindle"]
        if a.editor == "median":  # V4: free patches of the epoch (arousals: of the window before the event)
            pool = range(lo_p, hi_p) if judge == "kc" else range(lo_p, pa - 1)
            free_p = [p for p in pool if not pa <= p < pb and quiet[(p - lo_p) * TPP:(p - lo_p + 1) * TPP].all()]
            if not free_p:
                n_seen["no_free"] += 1
                continue
            fills = [median_fill(free_p, n_p)]
        else:
            fills = infill(pa, pb, n)
        # control (a): silent span of equal length (same epoch / after the arousal)
        cand = [q for q in range(max(1, lo_p), min(hi_p, P - 1) - n_p + 1)
                if ((q + n_p < pa - 1 or q > pb) if judge == "kc" else q > pb)
                and quiet[(q - lo_p) * TPP:(q + n_p - lo_p) * TPP].all()]
        q = int(rng.choice(cand)) if cand else None
        if q is None:
            fills_a = []
        elif a.editor == "median":  # the same edit: median of the free patches outside the silent span
            fa = [p for p in free_p if not q <= p < q + n_p]
            fills_a = [median_fill(fa, n_p)] if fa else []
        else:
            fills_a = infill(q, q + n_p, n)
        # control (b): codes of a random span of the night, no FIM
        rr = [r for r in range(0, P - n_p) if r + n_p < pa - 1 or r > pb]
        r = int(rng.choice(rr))
        edits = [[(pa, f)] for f in fills] + [[(q, f)] for f in fills_a] + [[(pa, codes[r:r + n_p])]]
        det, U = judge_window(lo_p, hi_p, edits)
        kept = [bool(d[key][ev].any()) for d in det]
        row = {"night": nd["key"], "stage": int(stage[ta]), "n_patches": n_p,
               "remove_kept": kept[:n], "ctrl_silent_kept": kept[n:n + len(fills_a)], "ctrl_random_kept": kept[-1]}
        if cj is not None:  # second judge: C on the same decoded variants, x̂ first
            Sc, t0, err = c_window(lo_p, hi_p, [[]] + edits)
            row.update({"c_tau": tau, "c_err": err, "c_before": c_fires(Sc[0], t0, ga, gb),
                        "c_remove_kept": [c_fires(Sc[1 + i], t0, ga, gb) for i in range(n)],
                        "c_ctrl_silent_kept": [c_fires(Sc[1 + n + i], t0, ga, gb) for i in range(len(fills_a))],
                        "c_ctrl_random_kept": c_fires(Sc[-1], t0, ga, gb),
                        "c_S_orig": c_mean(Sc[0], t0, pa * TPP, pb * TPP),
                        "c_S_remove": _mean([c_mean(Sc[1 + i], t0, pa * TPP, pb * TPP) for i in range(n)]),
                        "c_S_ctrl_random": c_mean(Sc[-1], t0, pa * TPP, pb * TPP)})
        # locality and amplitude (codec scale / uV)
        dif = (U[:n] - U0[None]) / uv
        sp = np.zeros(U0.shape[0], bool)
        sp[(pa - lo_p) * SPP:(pb - lo_p) * SPP] = True
        ring = np.zeros_like(sp)
        ring[max(0, pa - 1 - lo_p) * SPP:min(hi_p - lo_p, pb + 1 - lo_p) * SPP] = True
        ring &= ~sp
        far = ~sp & ~ring
        rms = lambda m: float(np.sqrt((dif[:, m] ** 2).mean())) if m.any() else None  # noqa: E731
        row.update({"rms_in": rms(sp), "rms_ring": rms(ring), "rms_far": rms(far)})
        if judge == "kc":
            evs_s = slice(ev.start * 50, ev.stop * 50)
            ptp = lambda u: float(np.ptp(CRF.bandpass(u, 0.3, 4, fs)[evs_s]))  # noqa: E731
            row.update({"ptp_before": ptp(U0), "ptp_remove": float(np.mean([ptp(u) for u in U[:n]])),
                        "ptp_ctrl_random": ptp(U[-1])})
        # realism
        f0 = free_span(n_p, int(stage[ta]))
        items = [(pa, n_p, [])] + [(pa, n_p, [(pa, f)]) for f in fills] + [(pa, n_p, [(pa, codes[r:r + n_p])])]
        items += [(q, n_p, [(q, f)]) for f in fills_a[:1]] + ([(f0, n_p, [])] if f0 is not None else [])
        Rs = realism(items)
        row.update({"S_orig": Rs[0], "S_remove": float(np.mean([v for v in Rs[1:n + 1] if v is not None]))
                    if any(v is not None for v in Rs[1:n + 1]) else None, "S_ctrl_random": Rs[n + 1],
                    "S_ctrl_silent": Rs[n + 2] if len(fills_a) else None,
                    "S_free": Rs[-1] if f0 is not None else None})
        if cj is not None:  # realism without the AR: C score of an event-free stretch of the same length and stage
            row["c_S_free"] = c_mean(s_hat, 0, f0 * TPP, (f0 + n_p) * TPP) if f0 is not None else None
        if stager is not None:
            y_true = int(stage[jj * T])
            dm = stager.delta(codec, nd, jj, y_true, [[]] + edits[:1] + ([edits[n]] if len(fills_a) else [])
                              + [edits[-1]], dev) if y_true >= 0 else None
            if dm is not None:
                names = ["remove"] + (["ctrl_silent"] if len(fills_a) else []) + ["ctrl_random"]
                row.update({f"dM_{k}": v for k, v in zip(names, dm["dM"])})
                row.update({f"flip_{k}": v for k, v in zip(names, dm["flip"])})
        rows.append(row)
        if judge == "kc":
            donors.append((pa, pb, ta, tb, jj))

    adds = []
    if judge == "kc" and donors:  # insertion by code transplant
        eps = [e for e in np.unique(np.flatnonzero(nd["dumped"]) // T)
               if not ref[e * T:(e + 1) * T].any()]
        for jr in [eps[i] for i in rng.permutation(len(eps))][: len(donors)]:
            pa, pb, ta, tb, jd = donors[int(rng.integers(len(donors)))]
            n_p = pb - pa
            if EPP - n_p - 1 <= 1:
                continue
            s = jr * EPP + int(rng.integers(1, EPP - n_p - 1))
            m = slice(s * TPP + (ta - pa * TPP) - jr * T, s * TPP + (tb - pa * TPP) - jr * T)
            dj = base_by_ep[jd]
            quiet = ~(dj["kc"] | dj["delta"] | dj["spindle"]) & ~union[jd * T:(jd + 1) * T]
            sil = [p for p in range(EPP) if quiet[p * TPP:(p + 1) * TPP].all()]
            edits = [[]] + [[(s, codes[pa:pb])]] + ([[(s, codes[jd * EPP + np.array(sorted(rng.choice(
                sil, n_p, replace=False)))])]] if len(sil) >= n_p else [])
            det, _ = judge_window(jr * EPP, (jr + 1) * EPP, edits)
            if det[0]["kc"].any():  # V4: recipients without a K-complex in x̂
                continue
            row = {"night": nd["key"], "stage": int(stage[jr * T]), "n_patches": n_p,
                   "detected_before": bool(det[0]["kc"][m].any()), "add": bool(det[1]["kc"][m].any())}
            if len(edits) > 2:
                row["ctrl_add"] = bool(det[2]["kc"][m].any())
            if cj is not None:
                Sc, t0, _ = c_window(jr * EPP, (jr + 1) * EPP, edits)
                ma, mb = jr * T + m.start, jr * T + m.stop
                row.update({"c_before": c_fires(Sc[0], t0, ma, mb), "c_add": c_fires(Sc[1], t0, ma, mb)})
                if len(edits) > 2:
                    row["c_ctrl_add"] = c_fires(Sc[2], t0, ma, mb)
            Rs = realism([(s, n_p, e) for e in edits[1:]])
            row["S_add"] = Rs[0]
            row["S_ctrl_add"] = Rs[1] if len(Rs) > 1 else None
            if stager is not None and stage[jr * T] >= 0:
                dm = stager.delta(codec, nd, int(jr), int(stage[jr * T]), edits, dev)
                if dm is not None:
                    row.update({f"dM_{k}": v for k, v in zip(("add", "ctrl_add"), dm["dM"])})
            adds.append(row)

    closures = []
    if surp is not None and a.max_starts:  # closure rule from the surprise peaks (plan point 5)
        S0 = np.asarray(surp["S0"], np.float64)[:P]
        ok = np.isfinite(S0) & (patch_stage[: len(S0)] >= 0)
        if ok.sum() > 50:
            z = (S0 - S0[ok].mean()) / max(S0[ok].std(), 1e-9)
            thr = float(np.quantile(S0[ok], 0.75))
            peaks = [p for p in range(1, len(S0) - 1) if ok[p] and z[p] > 2 and z[p] >= z[p - 1]
                     and z[p] >= z[p + 1] and patch_stage[p] in (2, 3)]
            for p in [peaks[i] for i in rng.permutation(len(peaks))][: a.max_starts]:
                b, fill = close_event(model, codes, p, thr, MAX_SPAN, FOLLOW, W, 1.0, gen)
                row = {"night": nd["key"], "closed": b is not None}
                if b is not None:
                    tk = slice(p * TPP, b * TPP)
                    sh = int(rng.integers(T, max(T + 1, n_tok - T)))
                    row.update({"len": b - p, "explained": bool(union[tk].any()),
                                "explained_shifted": bool(np.roll(union, sh)[tk].any())})
                    cover = [(ta, tb) for ta, tb, jj in kc_events(nd, ref, (2, 3))
                             if p * TPP <= ta and tb <= b * TPP] if judge == "kc" else []
                    if cover:
                        jj = cover[0][0] // T
                        ga, gb, _, _ = span_of(*cover[0], n_tok)
                        ev = slice(ga - jj * T, gb - jj * T)
                        det, _ = judge_window(jj * EPP, (jj + 1) * EPP, [[], [(p, fill)]])
                        if det[0]["kc"][ev].any():
                            row["removes_expert_kc"] = not det[1]["kc"][ev].any()
                closures.append(row)
    return rows, adds, closures, n_seen


# ---------------------------------------------------------------------------- summary
def _med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _mean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def summarize(rows, adds, closures):
    from scipy.stats import wilcoxon

    out = {"n_events": len(rows)}
    if rows:
        rm = [1 - np.mean(r["remove_kept"]) for r in rows]
        cs = [1 - np.mean(r["ctrl_silent_kept"]) for r in rows if r["ctrl_silent_kept"]]
        out["remove"] = {"efficacy": float(np.mean(rm)), "all_samples": float(np.mean([v == 1 for v in rm])),
                         "ctrl_silent": float(np.mean(cs)) if cs else None, "n_ctrl_silent": len(cs),
                         "ctrl_random": float(np.mean([not r["ctrl_random_kept"] for r in rows]))}
        out["locality"] = {k: _med([r[k] for r in rows]) for k in ("rms_in", "rms_ring", "rms_far")}
        if "ptp_before" in rows[0]:
            out["ptp_drop_uv"] = {"remove": _med([r["ptp_before"] - r["ptp_remove"] for r in rows]),
                                  "ctrl_random": _med([r["ptp_before"] - r["ptp_ctrl_random"] for r in rows])}
        if any(r["S_remove"] is not None or r["S_free"] is not None for r in rows):
            free = [r["S_free"] for r in rows if r["S_free"] is not None]
            q = [float(np.quantile(free, .25)), float(np.quantile(free, .75))] if len(free) >= 4 else None
            pair = [(r["S_remove"], r["S_ctrl_random"]) for r in rows
                    if r["S_remove"] is not None and r["S_ctrl_random"] is not None]
            p = float(wilcoxon([x for x, _ in pair], [y for _, y in pair], alternative="less").pvalue) \
                if len(pair) > 5 else None
            med = {k: _med([r[k] for r in rows]) for k in ("S_orig", "S_remove", "S_ctrl_random", "S_ctrl_silent")}
            inside = q is not None and med["S_remove"] is not None and q[0] <= med["S_remove"] <= q[1]
            out["realism"] = {**med, "free_iqr": q, "n_free": len(free), "wilcoxon_remove_lt_random_p": p,
                              "remove_in_free_iqr": float(np.mean([q[0] <= r["S_remove"] <= q[1] for r in rows
                                                                   if r["S_remove"] is not None])) if q else None,
                              "pass": bool(inside and p is not None and p < 0.05
                                           and med["S_remove"] < med["S_ctrl_random"])}
        else:  # no next-token model
            out["realism"] = {"measurable": False, "pass": None}
        st = {k: _med([r.get(f"dM_{k}") for r in rows]) for k in ("remove", "ctrl_silent", "ctrl_random")}
        if any(v is not None for v in st.values()):
            fl = {k: float(np.mean([r[f"flip_{k}"] for r in rows if f"flip_{k}" in r]))
                  for k in ("remove", "ctrl_silent", "ctrl_random") if any(f"flip_{k}" in r for r in rows)}
            out["stager"] = {"dM_median": st, "flip_rate": fl,
                             "n": sum("dM_remove" in r for r in rows)}
    if adds:
        ok = [r for r in adds if not r["detected_before"]]
        out["insert"] = {"n": len(ok), "efficacy": float(np.mean([r["add"] for r in ok])) if ok else None,
                         "ctrl": float(np.mean([r["ctrl_add"] for r in ok if "ctrl_add" in r]))
                         if any("ctrl_add" in r for r in ok) else None,
                         "efficacy_n2n3": float(np.mean([r["add"] for r in ok if r["stage"] in (2, 3)]))
                         if any(r["stage"] in (2, 3) for r in ok) else None,
                         "S_add": _med([r["S_add"] for r in ok]), "S_ctrl_add": _med([r["S_ctrl_add"] for r in ok]),
                         "dM_add": _med([r.get("dM_add") for r in ok]),
                         "dM_ctrl_add": _med([r.get("dM_ctrl_add") for r in ok])}
    if closures:
        cl = [r for r in closures if r["closed"]]
        rem = [r["removes_expert_kc"] for r in cl if "removes_expert_kc" in r]
        out["closure"] = {"starts": len(closures), "closed": len(cl) / len(closures),
                          "len_median": _med([r["len"] for r in cl]),
                          "explained": float(np.mean([r["explained"] for r in cl])) if cl else None,
                          "explained_shifted": float(np.mean([r["explained_shifted"] for r in cl])) if cl else None,
                          "n_cover_expert_kc": len(rem), "removes_expert_kc": float(np.mean(rem)) if rem else None}
    return out


def summarize_c(rows, adds):
    """T4: the C judge on the same edits (rules in the module docstring)."""
    out = {"n_events": sum("c_before" in r for r in rows), "tau_median": _med([r.get("c_tau") for r in rows]),
           "window_err_max": max((r["c_err"] for r in rows if r.get("c_err") is not None), default=None)}
    det = [r for r in rows if r.get("c_before")]
    out["n_detected"] = len(det)
    if det:
        cs = [1 - np.mean(r["c_ctrl_silent_kept"]) for r in det if r["c_ctrl_silent_kept"]]
        out["remove"] = {"efficacy": float(np.mean([1 - np.mean(r["c_remove_kept"]) for r in det])),
                         "ctrl_silent": float(np.mean(cs)) if cs else None, "n_ctrl_silent": len(cs),
                         "ctrl_random": float(np.mean([not r["c_ctrl_random_kept"] for r in det]))}
    free = [r["c_S_free"] for r in rows if r.get("c_S_free") is not None]
    q = [float(np.quantile(free, .25)), float(np.quantile(free, .75))] if len(free) >= 4 else None
    med = {k: _med([r.get(f"c_S_{k}") for r in rows]) for k in ("orig", "remove", "ctrl_random")}
    rem = [r["c_S_remove"] for r in rows if r.get("c_S_remove") is not None]
    out["realism"] = {**{f"S_{k}": v for k, v in med.items()}, "free_iqr": q, "n_free": len(free),
                      "remove_in_free_iqr": float(np.mean([q[0] <= v <= q[1] for v in rem])) if q and rem else None,
                      "pass": bool(q is not None and med["remove"] is not None and q[0] <= med["remove"] <= q[1])}
    ok = [r for r in adds if "c_before" in r and not r["c_before"]]
    if ok:
        out["insert"] = {"n": len(ok), "n_fired_before": sum(bool(r.get("c_before")) for r in adds),
                         "efficacy": float(np.mean([r["c_add"] for r in ok])),
                         "ctrl": _mean([float(r["c_ctrl_add"]) for r in ok if "c_ctrl_add" in r])}
    return out


def main(argv=None):
    import torch

    from physioex.train.bin.eeg_ar import load_model
    from physioex.train.bin.patch_codec import load_codec

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codec", required=True, help="patch codec run")
    ap.add_argument("--ar", default=None, help="next-token model run (on the codec's codes); needed by --editor fim, "
                    "by the realism by surprise and by the closure rule")
    ap.add_argument("--editor", choices=("fim", "median"), default="fim",
                    help="fim: FIM of the next-token model; median: V4, deterministic (T4)")
    ap.add_argument("--check-codec", default=None, help="T4: codec run of the spectral twin C, the second judge")
    ap.add_argument("--check-codes", default=None, help="T4: codes dump of C (pool test nights: the prototypes)")
    ap.add_argument("--check-feature", default="latent", help="T4: C feature of the judge (semantic_gates)")
    ap.add_argument("--proto-max-nights", type=int, default=0,
                    help="T4: pool test nights per unit for the C prototypes (0 = all; smoke tests)")
    ap.add_argument("--codes", required=True, help="codes dump of the codec (eval nights)")
    ap.add_argument("--scan", required=True, help="concept_scan dump (ts, y, x_eeg; refs/, m1/, scale.json)")
    ap.add_argument("--surprise", default=None, help="surprise of the AR (closure rule); default <ar>/surprise")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-events", type=int, default=25, help="per night")
    ap.add_argument("--max-nights", type=int, default=20, help="per set")
    ap.add_argument("--max-starts", type=int, default=8, help="closure rule: surprise peaks per DREAMS night")
    ap.add_argument("--stager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    assert a.ar or a.editor == "median", "surprise_counterfactual: --editor fim needs --ar"
    assert not a.check_codec or a.check_codes, "surprise_counterfactual: --check-codec needs --check-codes"
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, codec = load_codec(a.codec, dev)
    model = None
    if a.ar:
        run, model = load_model(a.ar, dev)
        a.window, L = int(run["args"]["window"]), model.levels
    else:
        a.window, L = None, int(codec.vq_levels)
    if a.editor == "median":
        assert getattr(codec, "quantizer", None) == "fsq", "surprise_counterfactual: --editor median needs FSQ codes"
        a.n_samples = 1  # deterministic
    cj = None
    if a.check_codec:
        cj = CJudge(a.check_codec, a.check_codes, a.scan, dev, a.check_feature, a.proto_max_nights)
        print(f"[cf-ar] C judge: prototypes {json.dumps(cj.info)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    stager, stager_err = None, None
    if a.stager:
        try:
            stager = Stager(a.scan, dev)
        except Exception as e:  # noqa: BLE001
            stager_err = repr(e)
            print(f"[cf-ar] stager not loaded: {stager_err}", flush=True)
    meta = json.load(open(os.path.join(a.scan, "meta.json")))
    scale_js = json.load(open(os.path.join(a.scan, "scale.json")))
    surp_dir = a.surprise or (os.path.join(a.ar, "surprise") if a.ar else None)
    fs = float(codec.fs)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    res = {"rules": {"max_span": MAX_SPAN, "follow": FOLLOW, "n_samples": a.n_samples, "gate": GATE,
                     "arousal_window_tokens": [AR_BEFORE, AR_AFTER], "levels": L, "window": a.window,
                     "editor": a.editor, "decode_margin": getattr(codec, "receptive_patches", 1) + 1},
           "codec": os.path.abspath(a.codec), "ar": os.path.abspath(a.ar) if a.ar else None,
           "scan": os.path.abspath(a.scan), "sets": {}}
    if cj is not None:
        res["rules"]["c_judge"] = {"run": os.path.abspath(a.check_codec), "codes": os.path.abspath(a.check_codes),
                                   "feature": a.check_feature, "quantile": C_Q, "margin": C_MARGIN,
                                   "classes": C_CLASS, "prototypes": cj.info}
    if stager is not None:
        res["rules"]["stager"] = {"half": stager.half, "seq": stager.seq, "ctx_epochs": stager.ctx}
    if stager_err:
        res["rules"]["stager_error"] = stager_err
    with torch.inference_mode():
        for name in a.sets.split(","):
            units, ref_key, stages, judge = SETS[name]
            keys = sorted(k for k, v in meta["nights"].items()
                          if (v["unit"] in units if units else v["unit"] not in ("dreams", "mass_ss01")))
            keys = [keys[i] for i in sorted(rng.permutation(len(keys))[: a.max_nights])]
            rows, adds, closures, seen, r2s, skipped = [], [], [], {}, [], []
            for k in keys:
                unit, sid = k.split("/", 1)
                nd = load_night(a, unit, sid, L, scale_js)
                if nd is None or ref_key not in nd["masks"]:
                    skipped.append(k)
                    continue
                r2 = align_r2(codec, nd, dev)
                if r2 is not None:
                    r2s.append(r2)
                if r2 is None or r2 < MIN_ALIGN_R2:
                    skipped.append(k)
                    print(f"[cf-ar] {k}: alignment R^2 {r2} -> skipped", flush=True)
                    continue
                sp = os.path.join(surp_dir, unit, f"{sid}.npz") if surp_dir else None
                surp = np.load(sp) if sp and name == "dreams_kc" and os.path.exists(sp) else None
                r, ad, cl, ns = night(a, nd, name, ref_key, stages, judge, model, codec, stager, dev, rng, gen,
                                      fs, surp, cj)
                rows += r
                adds += ad
                closures += cl
                for kk, v in ns.items():
                    seen[kk] = seen.get(kk, 0) + v
                print(f"[cf-ar] {name} {k}: {len(r)} events, {len(ad)} inserts, {len(cl)} starts "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            res["sets"][name] = {"n_nights": len(keys) - len(skipped), "skipped": len(skipped), "counts": seen,
                                 "align_r2_median": _med(r2s), **summarize(rows, adds, closures)}
            if cj is not None:
                res["sets"][name]["c_judge"] = summarize_c(rows, adds)
            print(f"[cf-ar] {name}: {json.dumps(res['sets'][name])[:600]}", flush=True)
    g = res["sets"].get("dreams_kc", {})
    cg = g.get("c_judge") or {}
    real = (g.get("realism") or {}).get("pass") if model is not None else (cg.get("realism") or {}).get("pass")
    checks = {"remove": (g.get("remove") or {}).get("efficacy"),
              "ctrl_silent": (g.get("remove") or {}).get("ctrl_silent"),
              "realism": real, "insert": (g.get("insert") or {}).get("efficacy")}
    passed = {"remove": checks["remove"] is not None and checks["remove"] >= GATE["remove"],
              "ctrl_silent": checks["ctrl_silent"] is not None and checks["ctrl_silent"] <= GATE["ctrl_silent"],
              "realism": bool(checks["realism"]),
              "insert": checks["insert"] is not None and checks["insert"] >= GATE["insert"]}
    res["gate"] = {"values": checks, "pass_each": passed, "pass": all(passed.values()),
                   "realism_source": "AR surprise" if model is not None else
                   ("C judge, no AR (T4)" if cj is not None else "not measurable")}
    if cj is not None:  # the same thresholds with the C judge (reported)
        cv = {"remove": (cg.get("remove") or {}).get("efficacy"), "ctrl_silent": (cg.get("remove") or {}).get("ctrl_silent"),
              "insert": (cg.get("insert") or {}).get("efficacy")}
        cp = {"remove": cv["remove"] is not None and cv["remove"] >= GATE["remove"],
              "ctrl_silent": cv["ctrl_silent"] is not None and cv["ctrl_silent"] <= GATE["ctrl_silent"],
              "insert": cv["insert"] is not None and cv["insert"] >= GATE["insert"]}
        res["gate"]["c_judge"] = {"values": cv, "pass_each": cp, "pass": all(cp.values())}
    if stager is not None:
        res["rules"]["stager"]["errors"] = stager.errors
    res["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[cf-ar] gate {res['gate']} -> {a.out}", flush=True)
    print("[cf-ar] DONE", flush=True)


if __name__ == "__main__":
    main()
