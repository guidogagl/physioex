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
    srcs = [("", load_refs(pr))] + ([("m1:", dict(np.load(pm)))] if os.path.exists(pm) else [])
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


def decode_window(codec, codes, lo, hi, edits_list, dev, margin=2):
    """Decoded samples of patches ``lo .. hi - 1`` for each list of edits ``[(start patch, codes)]``."""
    import torch

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
def night(a, nd, name, ref_key, stages, judge, model, codec, stager, dev, rng, gen, fs, surp):
    import torch

    from physioex.explain import concept_refs as CRF
    from physioex.models.eeg_ar import close_event, window_around, window_surprise

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

    key = "kc" if judge == "kc" else "arousal"
    evs = kc_events(nd, ref, stages) if judge == "kc" else arousal_events(nd, ref, stages)
    evs = [evs[i] for i in rng.permutation(len(evs))[: a.max_events]]
    rows, donors, base_by_ep = [], [], {}
    n_seen = {"events": len(evs), "too_long": 0, "not_detected": 0}  # too_long: also spans at the night borders
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
        fills = infill(pa, pb, n)
        # control (a): silent span of equal length (same epoch / after the arousal)
        quiet = ~union[lo_p * TPP:hi_p * TPP] & ~det0[key]
        if judge == "kc":
            quiet &= ~det0["delta"] & ~det0["spindle"]
        cand = [q for q in range(max(1, lo_p), min(hi_p, P - 1) - n_p + 1)
                if ((q + n_p < pa - 1 or q > pb) if judge == "kc" else q > pb)
                and quiet[(q - lo_p) * TPP:(q + n_p - lo_p) * TPP].all()]
        q = int(rng.choice(cand)) if cand else None
        fills_a = infill(q, q + n_p, n) if q is not None else []
        # control (b): codes of a random span of the night, no FIM
        rr = [r for r in range(0, P - n_p) if r + n_p < pa - 1 or r > pb]
        r = int(rng.choice(rr))
        edits = [[(pa, f)] for f in fills] + [[(q, f)] for f in fills_a] + [[(pa, codes[r:r + n_p])]]
        det, U = judge_window(lo_p, hi_p, edits)
        kept = [bool(d[key][ev].any()) for d in det]
        row = {"night": nd["key"], "stage": int(stage[ta]), "n_patches": n_p,
               "remove_kept": kept[:n], "ctrl_silent_kept": kept[n:n + len(fills_a)], "ctrl_random_kept": kept[-1]}
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


def main(argv=None):
    import torch

    from physioex.train.bin.eeg_ar import load_model
    from physioex.train.bin.patch_codec import load_codec

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codec", required=True, help="patch codec run")
    ap.add_argument("--ar", required=True, help="next-token model run (on the codec's codes)")
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
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, codec = load_codec(a.codec, dev)
    run, model = load_model(a.ar, dev)
    a.window = int(run["args"]["window"])
    L = model.levels
    stager, stager_err = None, None
    if a.stager:
        try:
            stager = Stager(a.scan, dev)
        except Exception as e:  # noqa: BLE001
            stager_err = repr(e)
            print(f"[cf-ar] stager not loaded: {stager_err}", flush=True)
    meta = json.load(open(os.path.join(a.scan, "meta.json")))
    scale_js = json.load(open(os.path.join(a.scan, "scale.json")))
    surp_dir = a.surprise or os.path.join(a.ar, "surprise")
    fs = float(codec.fs)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    res = {"rules": {"max_span": MAX_SPAN, "follow": FOLLOW, "n_samples": a.n_samples, "gate": GATE,
                     "arousal_window_tokens": [AR_BEFORE, AR_AFTER], "levels": L, "window": a.window},
           "codec": os.path.abspath(a.codec), "ar": os.path.abspath(a.ar), "scan": os.path.abspath(a.scan),
           "sets": {}}
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
                sp = os.path.join(surp_dir, unit, f"{sid}.npz")
                surp = np.load(sp) if name == "dreams_kc" and os.path.exists(sp) else None
                r, ad, cl, ns = night(a, nd, name, ref_key, stages, judge, model, codec, stager, dev, rng, gen,
                                      fs, surp)
                rows += r
                adds += ad
                closures += cl
                for kk, v in ns.items():
                    seen[kk] = seen.get(kk, 0) + v
                print(f"[cf-ar] {name} {k}: {len(r)} events, {len(ad)} inserts, {len(cl)} starts "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            res["sets"][name] = {"n_nights": len(keys) - len(skipped), "skipped": len(skipped), "counts": seen,
                                 "align_r2_median": _med(r2s), **summarize(rows, adds, closures)}
            print(f"[cf-ar] {name}: {json.dumps(res['sets'][name])[:600]}", flush=True)
    g = res["sets"].get("dreams_kc", {})
    checks = {"remove": (g.get("remove") or {}).get("efficacy"),
              "ctrl_silent": (g.get("remove") or {}).get("ctrl_silent"),
              "realism": (g.get("realism") or {}).get("pass"), "insert": (g.get("insert") or {}).get("efficacy")}
    passed = {"remove": checks["remove"] is not None and checks["remove"] >= GATE["remove"],
              "ctrl_silent": checks["ctrl_silent"] is not None and checks["ctrl_silent"] <= GATE["ctrl_silent"],
              "realism": bool(checks["realism"]),
              "insert": checks["insert"] is not None and checks["insert"] >= GATE["insert"]}
    res["gate"] = {"values": checks, "pass_each": passed, "pass": all(passed.values())}
    if stager is not None:
        res["rules"]["stager"]["errors"] = stager.errors
    res["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[cf-ar] gate {res['gate']} -> {a.out}", flush=True)
    print("[cf-ar] DONE", flush=True)


if __name__ == "__main__":
    main()
