"""Battery of the composite concepts (plan of 09.11.2026, "revise and update" cycle, levels V1-V3).

Every iteration ``C<n>`` writes one JSON with the same schema (``battery/<iter>_<concept>.json``):

- ``V1`` components: the P word (logistic on the named P inputs, trained on the rule reference of the pool
  nights; leave-one-site-out scores on the pool, one pool fit scores DREAMS / MASS SS01) with its AP and the
  transport ``retC`` against the signal ceiling with the same context; the epoch predicates (ridge on
  ``Eq`` / ``Nq`` with their squares, target = fraction of the reference tokens in the epoch) with their
  Spearman on held-out sites and on DREAMS. The word threshold matches the pool rate of its reference; the
  predicate thresholds are the manual ones (``concept_rules``).
- ``V2`` detection: the composite rule from the lexicon components against the experts (DREAMS K-complexes,
  vis1 / vis2 / union / both) and, on the pool, against the rule references (circularity declared). Event
  purity, coverage and F1, 95 % bootstrap CI over the units, two nulls (predicates permuted among the epochs of
  the same night; word rolled inside the night), q95 over ``n_null`` draws. The same rule on the reference
  components (``oracle``) gives the transport of the composite: ``ret = F1 lexicon / F1 oracle``.
- ``V3`` localization: the rule without each component (predicate, isolation, everything but the word).

Expert against expert on the same excerpts is the human reference. Stages enter only as a diagnostic (where
the detections fall).

    python -m physioex.train.bin.concept_compose battery --concept kc --dump DUMP --out OUT --iter C1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.explain import concept_rules as CR
from physioex.train.bin import concept_rationale as RA

T = RA.T

CONCEPTS = {
    "kc": {
        "refs": ["delta", "alpha", "slow_wave", "k_complex_rule", "k_complex_vis1", "k_complex_vis2"],
        "word": {"ref": "delta", "inputs": ["P", "P_ctx±2"], "ceiling": ["spec_tok", "abs_amp", "rel_bg", "spec_ctx2"]},
        "preds": {"swa": {"ref": "delta", "inputs": ["Eq"]}, "alpha": {"ref": "alpha", "inputs": ["Eq", "Nq"]}},
        "wake_words": {"blink": {"ref": "blink", "inputs": ["P@eog"]}, "rem_eye": {"ref": "rem_eye", "inputs": ["P@eog"]},
                       "atonia": {"ref": "atonia", "inputs": ["P@emg"]}},
        "wake_refs": ["blink", "rem_eye", "atonia"],
        "rule": "kc = delta word event AND swa_frac < 0.20 AND alpha_frac <= 0.50 AND isolated +-4 tokens; "
                "n3 = delta word event AND swa_frac >= 0.20",
    },
    "rem_eye": {
        "refs": ["alpha", "rem_eye", "atonia"],
        "word": {"ref": "rem_eye", "inputs": ["P@eog"]},
        "tone": {"ref": "atonia", "inputs": ["P@emg"]},
        "tone_eq": {"ref": "atonia", "inputs": ["Eq@emg", "Eq@emg_rel"]},
        "preds": {"alpha": {"ref": "alpha", "inputs": ["Eq", "Nq"]}},
        "rule": "rem = rem_eye word event AND atonia_frac > 0.50 (night-relative chin tone) AND alpha_frac <= 0.50; "
                "wake = rem_eye word event AND NOT rem",
    },
    "alpha_ctx": {
        "refs": ["alpha", "alpha_rem", "alpha_nrem", "delta", "blink", "rem_eye", "atonia"],
        "word": {"ref": "alpha", "inputs": ["P", "P_ctx±2"]},
        "rule": "alpha word event labelled by the lexicon state of its epoch: W = AASM wake rule (alpha > 0.50, "
                "blinks, rapid eyes with tone); REM = not W, atonia > 0.50, swa < 0.20; NREM = the rest "
                "(alpha intrusion / alpha-delta)",
    },
    "cap_a1": {
        "refs": ["alpha", "delta", "blink", "rem_eye", "atonia", "cap_a1", "cap_a3"],
        "bands": {"lo": ["so", "delta"], "hi": ["alpha", "sigma", "beta"]},
        "rule": "CAP A1 = 0.5-4 Hz token power (read from P) >= (4/3)^2 x its 60 s moving median, 8-30 Hz not, "
                "runs 2-60 s, in lexicon NREM; A3 = 8-30 Hz rise",
    },
    "arousal": {
        "refs": ["alpha", "delta", "blink", "rem_eye", "atonia", "arousal_ann", "emg_rise"],
        "bands": {"fast": ["theta", "alpha", "beta"]},
        "emg": {"ref": "emg_rise", "inputs": ["P@emg"]},
        "rule": "arousal = theta+alpha+beta token power (read from P) >= (4/3)^2 x the median of the previous 10 s, "
                "3-60 s, preceded by >= 10 s of lexicon sleep; in lexicon REM with an EMG rise word overlapping",
    },
}


# ---------------------------------------------------------------------------- components
def _word(D, cfg, seed=0, max_fit=100_000):
    """Token scores of the P word: LOSO on the pool, one pool fit elsewhere; threshold at the pool rate."""
    r, h = RA.ref_mask(D, cfg["ref"])
    X0, _, bad = RA.build(D, cfg["inputs"], np.arange(len(r)))
    fin = np.isfinite(X0).all(1)  # optional modalities: NaN rows where the night lacks them
    pool = (D["set"] == "pool") & h & fin
    ip, te = np.flatnonzero(pool), np.flatnonzero(fin)
    gn = D["gname"]
    groups = sorted(set(gn[ip]))
    out = {}

    def scores(inputs, X=None):
        if X is None:
            X, _, b2 = RA.build(D, inputs, np.arange(len(r)))
            bad.extend(b2)
        full = np.full(len(r), np.nan)
        rng = np.random.default_rng(seed)
        full[te] = RA._fit_predict(X, [], r, ip, te, max_fit, rng)
        loso = RA.fold_probe(X[ip], [], r[ip], gn[ip], groups, max_fit=max_fit, seed=seed)
        return full, loso

    full, loso = scores(cfg["inputs"], X0)
    del X0
    cfull, closo = scores(cfg["ceiling"]) if cfg.get("ceiling") else (full * np.nan, loso * np.nan)
    ok = np.isfinite(loso)
    rate = float(r[ip].mean())
    thr = float(np.quantile(full[ip], 1 - rate))
    s = full.copy()
    s[ip] = loso
    word = (s >= thr) & np.isfinite(s)
    a_w = RA.ap(loso[ok], r[ip][ok])
    a_c = RA.ap(closo[ok], r[ip][ok]) if cfg.get("ceiling") else np.nan
    dre = (D["set"] == "dreams") & h
    out = {"ref": cfg["ref"], "inputs": cfg["inputs"], "ceiling_inputs": cfg.get("ceiling"), "skipped": bad,
           "pool_rate": rate, "threshold": thr, "ap_pool_loso": a_w, "ap_ceiling_loso": a_c,
           "retC": a_w / a_c if a_c else np.nan,
           "ap_dreams": RA.ap(full[dre], r[dre]) if dre.any() and r[dre].any() else None,
           "ap_ceiling_dreams": RA.ap(cfull[dre], r[dre]) if cfg.get("ceiling") and dre.any() and r[dre].any() else None,
           "available_frac": {k: float(fin[D["set"] == k].mean()) for k in sorted(set(D["set"]))},
           "fire_rate": {k: float(word[D["set"] == k].mean()) for k in sorted(set(D["set"]))}}
    return word, out, fin


def _pred(D, cfg, seed=0):
    """Epoch predicate: ridge on the epoch latents (with squares), target = reference fraction of the epoch."""
    from scipy.stats import spearmanr

    t, h = RA.epoch_target(D, cfg["ref"])
    X = np.hstack([np.hstack([RA.epoch_input(D, n), RA.epoch_input(D, n) ** 2]) for n in cfg["inputs"]])
    st = D["set"][::T]
    gn = D["gname"][::T]
    fin = np.isfinite(X).all(1)  # optional modalities: NaN epochs stay NaN
    pool = (st == "pool") & h & fin
    ip = np.flatnonzero(pool)
    rng = np.random.default_rng(seed)
    full = np.full(len(t), np.nan)
    full[fin] = RA._fit_predict(X, [], t, ip, np.flatnonzero(fin), 200_000, rng, ridge=True)
    loso = RA.fold_probe(X[ip], [], t[ip], gn[ip], sorted(set(gn[ip])), max_fit=200_000, seed=seed, ridge=True)
    v = full.copy()
    v[ip] = loso
    ok = np.isfinite(loso)
    dre = (st == "dreams") & h
    rho = lambda a, b: float(spearmanr(a, b)[0]) if len(a) > 2 else None
    return v, t, {"ref": cfg["ref"], "inputs": cfg["inputs"],
                  "spearman_pool_loso": rho(loso[ok], t[ip][ok]),
                  "spearman_dreams": rho(full[dre & fin], t[dre & fin]) if (dre & fin).any() else None,
                  "mean_pred_vs_true": {k: [float(np.nanmean(v[st == k])), float(np.nanmean(t[st == k]))]
                                        for k in sorted(set(st))}}


# ---------------------------------------------------------------------------- evaluation
def _agree(det, ref, night, sel, seed=0):
    a = CR.event_agreement(det, ref, night, sel)
    a["ci95_f1"] = CR.bootstrap_f1(_counts(det, ref, night, sel), seed=seed)
    return a


def _counts(det, ref, night, sel):
    ix = np.flatnonzero(sel)
    return CR.per_unit_counts(det[ix], ref[ix], night[ix])


def _nulls(D, word, swa, alpha, ref, sel, key, n_null, seed, wake=None):
    """Nulls on the whole nights that hold ``sel`` (tokens and epochs sliced together)."""
    rng = np.random.default_rng(seed + 7)
    ix = np.flatnonzero(np.isin(D["night"], np.unique(D["night"][sel])))
    ep = ix[::T] // T
    night, en = D["night"][ix], D["night"][ix][::T]
    word, ref, sel, swa, alpha = word[ix], ref[ix], sel[ix], swa[ep], alpha[ep]
    wake = None if wake is None else wake[ep]
    f_pred, f_word = [], []
    for _ in range(n_null):
        o = CR.kc_n3_rule(word, night, CR.permute_epochs(swa, en, rng), CR.permute_epochs(alpha, en, rng),
                          wake_ep=None if wake is None else CR.permute_epochs(wake, en, rng))
        f_pred.append(CR.event_agreement(o[key], ref, night, sel)["f1"])
        o = CR.kc_n3_rule(CR.shift_within_nights(word, night, rng), night, swa, alpha, wake_ep=wake)
        f_word.append(CR.event_agreement(o[key], ref, night, sel)["f1"])
    q = lambda f: float(np.nanquantile(f, 0.95)) if f else None
    return {"pred_perm_q95": q(f_pred), "pred_perm_mean": float(np.nanmean(f_pred)) if f_pred else None,
            "word_roll_q95": q(f_word), "word_roll_mean": float(np.nanmean(f_word)) if f_word else None,
            "n": n_null}


def _wake_clauses(src, st, y):
    """Diagnostic: share of epochs per stage (pool) where each wake clause fires, lexicon and oracle."""
    out = {}
    names = ["W", "N1", "N2", "N3", "R"]
    for k, (a, f, sw) in src.items():
        tone_low = np.nan_to_num(f["atonia"], nan=1.0) > CR.ATONIA_EPOCH
        cl = {"alpha": np.nan_to_num(a, nan=0.0) > CR.W_ALPHA,
              "blink": np.nan_to_num(f["blink"], nan=0.0) >= CR.EYE_PRESENT,
              "eye_tone": (np.nan_to_num(f["rem_eye"], nan=0.0) >= CR.EYE_PRESENT) & ~tone_low,
              "swa_ge_20": sw >= CR.N3_SWA}
        m = st == "pool"
        out[k] = {c: {names[s]: float(v[m & (y == s)].mean()) for s in range(5) if (m & (y == s)).any()}
                  for c, v in cl.items()}
    return out


def battery_kc(D, n_null=200, n_null_pool=20, seed=0, swa_source="eq", wake=False, n3_first=False, eye_eeg=False):
    """``swa_source``: ``eq`` = ridge on the epoch latents; ``words`` = fraction of the epoch's tokens where the
    slow-transient word fires (the predicate read on the word sequence, C1b). ``wake``: the alpha clause becomes
    the AASM wake rule on lexicon words (alpha, blink and rapid-eye words on ``P@eog``, atonia word on ``P@emg``;
    C1c), applied to KC and N3."""
    cfg = CONCEPTS["kc"]
    night = D["night"]
    word, v1w, _ = _word(D, cfg["word"], seed)
    swa, swa_t, v1s = _pred(D, cfg["preds"]["swa"], seed)
    if swa_source == "words":
        from scipy.stats import spearmanr

        swa = word.reshape(-1, T).mean(1)
        st = D["set"][::T]
        v1s = {"source": "words", "ref": "delta",
               "spearman_pool": float(spearmanr(swa[st == "pool"], swa_t[st == "pool"])[0]),
               "spearman_dreams": float(spearmanr(swa[st == "dreams"], swa_t[st == "dreams"])[0])}
    alpha, alpha_t, v1a = _pred(D, cfg["preds"]["alpha"], seed)
    dref, _ = RA.ref_mask(D, "delta")
    wl = wo = None
    v1x = {}
    if wake:
        fr, fo = {}, {}
        for name, wc in cfg["wake_words"].items():
            if eye_eeg and name in ("blink", "rem_eye"):  # C1e: the eye words also read the EEG, as their reference
                wc = {**wc, "inputs": wc["inputs"] + ["P"]}  # (deflection >= 2x larger on the EOG than on the EEG)
            w_, v1x[name], fin = _word(D, wc, seed)
            avail = fin.reshape(-1, T).all(1)
            fr[name] = np.where(avail, w_.reshape(-1, T).mean(1), np.nan)
            rr, hh = RA.ref_mask(D, wc["ref"])
            fo[name] = np.where(hh.reshape(-1, T).all(1), rr.reshape(-1, T).mean(1), np.nan)
        wl = CR.wake_rule(alpha, fr["blink"], fr["rem_eye"], fr["atonia"])
        wo = CR.wake_rule(alpha_t, fo["blink"], fo["rem_eye"], fo["atonia"])
        st = D["set"][::T]
        y = D["y"][::T]
        if n3_first:  # AASM precedence (C1d): an epoch with slow waves over >= 20 % is N3, never W
            wl = wl & ~(swa >= CR.N3_SWA)
            wo = wo & ~(swa_t >= CR.N3_SWA)
        v1x["wake_clauses"] = _wake_clauses({"lexicon": (alpha, fr, swa), "oracle": (alpha_t, fo, swa_t)}, st, y)
        v1x["wake_epochs"] = {k: {"lexicon": float(wl[st == k].mean()), "oracle": float(wo[st == k].mean()),
                                  "lexicon_in_W": float(wl[(st == k) & (y == 0)].mean()) if ((st == k) & (y == 0)).any() else None,
                                  "lexicon_in_sleep": float(wl[(st == k) & (y > 0)].mean()) if ((st == k) & (y > 0)).any() else None}
                              for k in sorted(set(st))}
    lex = CR.kc_n3_rule(word, night, swa, alpha, wake_ep=wl)
    ora = CR.kc_n3_rule(dref, night, swa_t, alpha_t, wake_ep=wo)
    abl = {"no_swa": CR.kc_n3_rule(word, night, swa, alpha, wake_ep=wl, use_swa=False),
           "no_alpha": CR.kc_n3_rule(word, night, swa, alpha, wake_ep=wl, use_alpha=False),
           "no_iso": CR.kc_n3_rule(word, night, swa, alpha, wake_ep=wl, use_iso=False),
           "word_only": CR.kc_n3_rule(word, night, swa, alpha, wake_ep=wl, use_swa=False, use_alpha=False, use_iso=False)}
    abl_o = {"no_swa": CR.kc_n3_rule(dref, night, swa_t, alpha_t, wake_ep=wo, use_swa=False),  # does the rule itself need it?
             "no_alpha": CR.kc_n3_rule(dref, night, swa_t, alpha_t, wake_ep=wo, use_alpha=False),
             "no_iso": CR.kc_n3_rule(dref, night, swa_t, alpha_t, wake_ep=wo, use_iso=False),
             "word_only": CR.kc_n3_rule(dref, night, swa_t, alpha_t, wake_ep=wo, use_swa=False, use_alpha=False, use_iso=False)}
    out = {"V1": {"word": v1w, "swa": v1s, "alpha": v1a, **v1x}, "V2": {}, "V3": {}}

    # DREAMS experts
    dre = D["set"] == "dreams"
    tg = {}
    for v in ("k_complex_vis1", "k_complex_vis2"):
        if v in D["refs"]:
            tg[v] = D["refs"][v]
    if len(tg) == 2:
        (r1, h1), (r2, h2) = tg["k_complex_vis1"], tg["k_complex_vis2"]
        tg["kc_union"] = (r1 | r2, h1 & h2)
        tg["kc_both"] = (r1 & r2, h1 & h2)
        both = dre & h1 & h2
        out["V2"]["dreams_expert_vs_expert"] = {"vis2_vs_vis1": _agree(r2, r1, night, both, seed),
                                                "n_excerpts": int(len(set(night[both])))}
    central = np.zeros(len(night), bool)
    for n in set(night[dre]):
        m = np.flatnonzero(night == n)
        ne = len(m) // T
        central[m[(ne // 2 - 10) * T:(ne // 2 + 10) * T]] = True
    dd = {}
    for name, (r, h) in tg.items():
        sel = dre & h
        if not r[sel].any():
            continue
        e = {"n_excerpts": int(len(set(night[sel]))),
             "lexicon": _agree(lex["kc"], r, night, sel, seed),
             "oracle": _agree(ora["kc"], r, night, sel, seed),
             "lexicon_central20": _agree(lex["kc"], r, night, sel & central, seed),
             "nulls": _nulls(D, word, swa, alpha, r, sel, "kc", n_null, seed, wl),
             "ablations": {k: CR.event_agreement(o["kc"], r, night, sel) for k, o in abl.items()},
             "ablations_oracle": {k: CR.event_agreement(o["kc"], r, night, sel) for k, o in abl_o.items()},
             "stage_of_detections": _stage_mix(D, lex["kc"], sel),
             "stage_of_reference": _stage_mix(D, r, sel)}
        fo = e["oracle"]["f1"]
        e["ret_vs_oracle"] = e["lexicon"]["f1"] / fo if fo else None
        dd[name] = e
    out["V2"]["dreams"] = dd

    # pool: rule references (declared circularity: the KC rule is stage-gated N2 by the hypnogram)
    pool = D["set"] == "pool"
    pp = {}
    krule, hk = D["refs"].get("k_complex_rule", (None, None))
    if krule is not None:
        sel = pool & hk
        pp["kc_vs_k_complex_rule"] = {
            "lexicon": _agree(lex["kc"], krule, night, sel, seed), "oracle": _agree(ora["kc"], krule, night, sel, seed),
            "nulls": _nulls(D, word, swa, alpha, krule, sel, "kc", n_null_pool, seed, wl),
            "ablations": {k: CR.event_agreement(o["kc"], krule, night, sel) for k, o in abl.items()},
            "ablations_oracle": {k: CR.event_agreement(o["kc"], krule, night, sel) for k, o in abl_o.items()},
            "per_site_f1": _per_site(D, lex["kc"], krule, sel),
            "stage_of_detections": _stage_mix(D, lex["kc"], sel),
            "circularity": "reference gated on the N2 hypnogram, amplitude relative to the epoch; the composite "
                           "uses no stage"}
    sw, hs = D["refs"].get("slow_wave", (None, None))
    if sw is not None:
        n3_ref = sw & np.repeat(swa_t >= CR.N3_SWA, T)
        sel = pool & hs
        pp["n3_vs_slow_wave_in_swa_epochs"] = {
            "lexicon": _agree(lex["n3"], n3_ref, night, sel, seed), "oracle": _agree(ora["n3"], n3_ref, night, sel, seed),
            "stage_of_detections": _stage_mix(D, lex["n3"], sel)}
        # K-complex against slow wave: of the pure rule events, how many the composite calls KC or N3
        if krule is not None:
            pp["kc_pure_vs_sw_pure"] = _pure_split(D, krule, sw, lex, hk & hs & pool)
    out["V2"]["pool"] = pp
    for k in ("dreams", "pool"):
        for name, e in out["V2"][k].items():
            if "ablations" in e and "lexicon" in e:
                f = e["lexicon"]["f1"]
                out["V3"][f"{k}:{name}"] = {a: (f - v["f1"]) for a, v in e["ablations"].items()}
            if "ablations_oracle" in e and "oracle" in e:
                f = e["oracle"]["f1"]
                out["V3"][f"{k}:{name}:oracle"] = {a: (f - v["f1"]) for a, v in e["ablations_oracle"].items()}
    return out


def _night_rel(D, key):
    """``key`` minus its per-night median (epoch rows): the chin tone relative to the night (C2b)."""
    if f"{key}_rel" in D or key not in D:
        return
    x = D[key]
    out = np.full_like(x, np.nan)
    for n in np.unique(D["night"]):
        m = D["night"] == n
        if np.isfinite(x[m]).all():
            out[m] = x[m] - np.median(x[m][::T], 0)
    D[f"{key}_rel"] = out


def _tone(D, source, seed=0):
    """Epoch atonia fraction from the lexicon: ``word`` = fraction of tokens of the P@emg atonia word (C2);
    ``eq`` = ridge on the EMG epoch latent, absolute and relative to the night (C2b)."""
    cfg = CONCEPTS["rem_eye"]
    if source == "eq":
        _night_rel(D, "Eq@emg")
        v, _, info = _pred(D, cfg["tone_eq"], seed)
        return v, info
    tw, info, fin_t = _word(D, cfg["tone"], seed)
    return np.where(fin_t.reshape(-1, T).all(1), tw.reshape(-1, T).mean(1), np.nan), info


def battery_rem(D, n_null=0, n_null_pool=20, seed=0, eye_eeg=False, tone_source="word"):
    """C2: rapid eye movements of REM against those of wake. Targets on the pool (DREAMS has no EOG): reference
    eye events in R epochs and in W epochs of the hypnogram (declared: the stage defines the target, the rule
    uses no stage). Measures: event agreement of the REM / wake calls, and the split accuracy on the word events
    that hit a reference eye event in R or W (the rem_eye_R_vs_W distinction that failed as a P word)."""
    cfg = CONCEPTS["rem_eye"]
    night = D["night"]
    wc = dict(cfg["word"])
    if eye_eeg:
        wc["inputs"] = wc["inputs"] + ["P"]
    word, v1w, fin = _word(D, wc, seed)
    aton, v1t = _tone(D, tone_source, seed)
    fin_t = np.repeat(np.isfinite(aton), T)
    alpha, alpha_t, v1a = _pred(D, cfg["preds"]["alpha"], seed)
    er, eh = RA.ref_mask(D, "rem_eye")
    ar, ah = RA.ref_mask(D, "atonia")
    aton_t = np.where(ah.reshape(-1, T).all(1), ar.reshape(-1, T).mean(1), np.nan)
    from scipy.stats import spearmanr

    ok = np.isfinite(aton) & np.isfinite(aton_t)
    v1t["spearman_epoch_frac"] = float(spearmanr(aton[ok], aton_t[ok])[0]) if ok.sum() > 2 else None
    lex = CR.rem_w_rule(word, night, aton, alpha)
    ora = CR.rem_w_rule(er, night, aton_t, alpha_t)
    abl = {"no_tone": CR.rem_w_rule(word, night, aton, alpha, use_tone=False),
           "no_alpha": CR.rem_w_rule(word, night, aton, alpha, use_alpha=False)}
    abl_o = {"no_tone": CR.rem_w_rule(er, night, aton_t, alpha_t, use_tone=False),
             "no_alpha": CR.rem_w_rule(er, night, aton_t, alpha_t, use_alpha=False)}
    y = np.repeat(D["y"][::T], T)
    sel = (D["set"] == "pool") & eh & fin & fin_t
    tgt = {"rem": er & (y == 4), "wake": er & (y == 0)}
    out = {"V1": {"word": v1w, "tone": v1t, "alpha": v1a}, "V2": {"pool": {}}, "V3": {}}
    for k, r in tgt.items():
        rng = np.random.default_rng(seed)
        en = night[::T]
        nul_p, nul_w = [], []
        for _ in range(n_null_pool):
            o = CR.rem_w_rule(word, night, CR.permute_epochs(aton, en, rng), CR.permute_epochs(alpha, en, rng))
            nul_p.append(CR.event_agreement(o[k], r, night, sel)["f1"])
            o = CR.rem_w_rule(CR.shift_within_nights(word, night, rng), night, aton, alpha)
            nul_w.append(CR.event_agreement(o[k], r, night, sel)["f1"])
        e = {"lexicon": _agree(lex[k], r, night, sel, seed), "oracle": _agree(ora[k], r, night, sel, seed),
             "word_only": CR.event_agreement(word, r, night, sel),
             "nulls": {"pred_perm_q95": float(np.quantile(nul_p, 0.95)) if nul_p else None,
                       "word_roll_q95": float(np.quantile(nul_w, 0.95)) if nul_w else None, "n": n_null_pool},
             "ablations": {a: CR.event_agreement(o[k], r, night, sel) for a, o in abl.items()},
             "ablations_oracle": {a: CR.event_agreement(o[k], r, night, sel) for a, o in abl_o.items()},
             "stage_of_detections": _stage_mix(D, lex[k], sel),
             "circularity": "target = reference eye events in the hypnogram stage; the rule uses no stage"}
        fo = e["oracle"]["f1"]
        e["ret_vs_oracle"] = e["lexicon"]["f1"] / fo if fo else None
        out["V2"]["pool"][k] = e
        out["V3"][f"pool:{k}"] = {a: e["lexicon"]["f1"] - v["f1"] for a, v in e["ablations"].items()}
        out["V3"][f"pool:{k}:oracle"] = {a: e["oracle"]["f1"] - v["f1"] for a, v in e["ablations_oracle"].items()}
    out["V2"]["pool"]["split_R_vs_W"] = {"lexicon": _split(D, lex, word, er, y, sel),
                                         "oracle": _split(D, ora, er, er, y, sel),
                                         "no_tone": _split(D, abl["no_tone"], word, er, y, sel),
                                         "no_alpha": _split(D, abl["no_alpha"], word, er, y, sel)}
    return out


def _lex_state(D, seed=0, eye_eeg=True, tone_source="word", blink_tone=False):
    """Per-epoch lexicon predicates shared by the rules: alpha (ridge on Eq+Nq), slow-wave fraction (delta word),
    eye and tone word fractions, AASM wake (C1e) and coarse state; the same from the references (oracle)."""
    cfg = CONCEPTS["kc"]
    dword, _, _ = _word(D, cfg["word"], seed)
    swa = dword.reshape(-1, T).mean(1)
    swa_t, _ = RA.epoch_target(D, "delta")
    alpha, alpha_t, v1a = _pred(D, cfg["preds"]["alpha"], seed)
    fr, fo, v1 = {}, {}, {"alpha": v1a}
    for name, wc in cfg["wake_words"].items():
        if eye_eeg and name in ("blink", "rem_eye"):
            wc = {**wc, "inputs": wc["inputs"] + ["P"]}
        w_, v1[name], fin = _word(D, wc, seed)
        fr[name] = np.where(fin.reshape(-1, T).all(1), w_.reshape(-1, T).mean(1), np.nan)
        rr, hh = RA.ref_mask(D, wc["ref"])
        fo[name] = np.where(hh.reshape(-1, T).all(1), rr.reshape(-1, T).mean(1), np.nan)
    if tone_source == "eq":
        fr["atonia"], v1["atonia_eq"] = _tone(D, "eq", seed)
    out = {}
    for k, (a, f, sw) in {"lexicon": (alpha, fr, swa), "oracle": (alpha_t, fo, swa_t)}.items():
        wk = CR.wake_rule(a, f["blink"], f["rem_eye"], f["atonia"], blink_needs_tone=blink_tone) & ~(sw >= CR.N3_SWA)
        out[k] = {"wake": wk, "state": CR.state_rule(wk, f["atonia"], sw), "aton": f["atonia"], "alpha": a, "swa": sw}
    return out, v1


def battery_alpha(D, n_null_pool=20, seed=0, tone_source="word", blink_tone=False):
    """C3: alpha in context. The alpha word's events labelled W / REM / NREM by the lexicon state of their epoch,
    against the alpha reference in W, in REM (``alpha_rem``) and in N2/N3 (``alpha_nrem``) of the hypnogram
    (declared: the stage defines the targets, the rule uses no stage). Pool only (the state needs EOG / EMG)."""
    cfg = CONCEPTS["alpha_ctx"]
    night = D["night"]
    word, v1w, fin = _word(D, cfg["word"], seed)
    S, v1 = _lex_state(D, seed, tone_source=tone_source, blink_tone=blink_tone)
    ar, ah = RA.ref_mask(D, "alpha")
    y = np.repeat(D["y"][::T], T)
    tgt = {"W": ar & (y == 0), "REM": RA.ref_mask(D, "alpha_rem")[0], "NREM": RA.ref_mask(D, "alpha_nrem")[0]}
    avail = np.repeat(np.isfinite(S["lexicon"]["aton"]), T)
    sel = (D["set"] == "pool") & ah & fin & avail
    lex = CR.in_state(word, night, S["lexicon"]["state"])
    ora = CR.in_state(ar, night, S["oracle"]["state"])
    st_ep = D["y"][::T]
    out = {"V1": {"word": v1w, **v1,
                  "state_vs_stage": {k: {CR.STATES[i]: {nm: float((S[k]["state"][(st_ep == s) & np.isfinite(S[k]["aton"])] == i).mean())
                                                        for s, nm in ((0, "W"), (4, "R"), (2, "N2"), (3, "N3"))}
                                         for i in range(3)} for k in ("lexicon", "oracle")}},
           "V2": {"pool": {}}, "V3": {}}
    en = night[::T]
    for k, r in tgt.items():
        rng = np.random.default_rng(seed)
        nul = [CR.event_agreement(CR.in_state(word, night, CR.permute_epochs(S["lexicon"]["state"], en, rng))[k],
                                  r, night, sel)["f1"] for _ in range(n_null_pool)]
        e = {"lexicon": _agree(lex[k], r, night, sel, seed), "oracle": _agree(ora[k], r, night, sel, seed),
             "word_only": CR.event_agreement(word, r, night, sel),
             "nulls": {"state_perm_q95": float(np.quantile(nul, 0.95)) if nul else None, "n": n_null_pool},
             "stage_of_detections": _stage_mix(D, lex[k], sel),
             "circularity": "target = alpha reference in the hypnogram stage; the rule uses no stage"}
        fo = e["oracle"]["f1"]
        e["ret_vs_oracle"] = e["lexicon"]["f1"] / fo if fo else None
        out["V2"]["pool"][k] = e
        out["V3"][f"pool:{k}"] = {"no_state": e["lexicon"]["f1"] - e["word_only"]["f1"]}
    return out


def _band_read(D, seed=0, max_fit=200_000, ctx=False):
    """Lexicon reading of the token log band powers: ridge from P (LOSO on the pool, one pool fit elsewhere) to
    each column of ``S``; returns the read and the true matrices and the per-band Spearman (pool, LOSO)."""
    from scipy.stats import spearmanr

    X, _, _ = RA.build(D, ["P", "P_ctx±2"] if ctx else ["P"], np.arange(len(D["y"])))  # C4b: the band windows span 3-4 tokens
    st, gn = D["set"], D["gname"]
    ip = np.flatnonzero(st == "pool")
    groups = sorted(set(gn[ip]))
    rng = np.random.default_rng(seed)
    S = D["S"][:, :len(RA.SPEC_WIN)]
    R = np.empty_like(S, dtype=np.float64)
    info = {}
    sub = rng.choice(len(ip), min(200_000, len(ip)), replace=False)
    for j, b in enumerate(RA.SPEC_WIN):
        t = S[:, j].astype(np.float64)
        R[:, j] = RA._fit_predict(X, [], t, ip, np.arange(len(t)), max_fit, rng, ridge=True)
        R[ip, j] = RA.fold_probe(X[ip], [], t[ip], gn[ip], groups, max_fit=max_fit, seed=seed, ridge=True)
        info[b] = float(spearmanr(R[ip[sub], j], t[ip[sub]])[0])
    return R, S, info


def _lse(M, cols):
    idx = [list(RA.SPEC_WIN).index(c) for c in cols]
    return np.logaddexp.reduce(M[:, idx], axis=1)


def battery_cap(D, n_null_pool=20, seed=0, tone_source="word", blink_tone=False, band_ctx=False):
    """C4: CAP phase A1 (and A3) from the lexicon's band reading with the background relation of the manual,
    against the reference rule (stage-gated N1-N3 by the hypnogram; declared circularity)."""
    cfg = CONCEPTS["cap_a1"]
    night = D["night"]
    R, S, v1b = _band_read(D, seed, ctx=band_ctx)
    St, v1 = _lex_state(D, seed, tone_source=tone_source, blink_tone=blink_tone)
    nrem_l = np.repeat(St["lexicon"]["state"] == 2, T)
    nrem_o = np.repeat(St["oracle"]["state"] == 2, T)
    lo_l, hi_l = _lse(R, cfg["bands"]["lo"]), _lse(R, cfg["bands"]["hi"])
    lo_t, hi_t = _lse(S, cfg["bands"]["lo"]), _lse(S, cfg["bands"]["hi"])
    lex = CR.cap_a_rule(lo_l, hi_l, night, nrem_l)
    ora = CR.cap_a_rule(lo_t, hi_t, night, nrem_o)
    abl = {"no_bg": CR.cap_a_rule(lo_l, hi_l, night, nrem_l, use_bg=False),
           "no_state": CR.cap_a_rule(lo_l, hi_l, night, None)}
    abl_o = {"no_bg": CR.cap_a_rule(lo_t, hi_t, night, nrem_o, use_bg=False),
             "no_state": CR.cap_a_rule(lo_t, hi_t, night, None)}
    avail = np.repeat(np.isfinite(St["lexicon"]["aton"]), T)
    out = {"V1": {"bands_spearman_pool_loso": v1b, **v1}, "V2": {"pool": {}}, "V3": {}}
    en = night[::T]
    for k in ("a1", "a3"):
        r, h = RA.ref_mask(D, f"cap_{k}")
        sel = (D["set"] == "pool") & h & avail
        rng = np.random.default_rng(seed)
        nul = []
        for _ in range(n_null_pool):
            st_p = CR.permute_epochs(St["lexicon"]["state"], en, rng)
            nul.append(CR.event_agreement(CR.cap_a_rule(lo_l, hi_l, night, np.repeat(st_p == 2, T))[k],
                                          r, night, sel)["f1"])
        e = {"lexicon": _agree(lex[k], r, night, sel, seed), "oracle": _agree(ora[k], r, night, sel, seed),
             "nulls": {"state_perm_q95": float(np.quantile(nul, 0.95)) if nul else None, "n": n_null_pool},
             "ablations": {a: CR.event_agreement(o[k], r, night, sel) for a, o in abl.items()},
             "ablations_oracle": {a: CR.event_agreement(o[k], r, night, sel) for a, o in abl_o.items()},
             "stage_of_detections": _stage_mix(D, lex[k], sel),
             "circularity": "reference gated on N1-N3 of the hypnogram; the rule uses the lexicon state"}
        fo = e["oracle"]["f1"]
        e["ret_vs_oracle"] = e["lexicon"]["f1"] / fo if fo else None
        out["V2"]["pool"][k] = e
        out["V3"][f"pool:{k}"] = {a: e["lexicon"]["f1"] - v["f1"] for a, v in e["ablations"].items()}
        out["V3"][f"pool:{k}:oracle"] = {a: e["oracle"]["f1"] - v["f1"] for a, v in e["ablations_oracle"].items()}
    return out


AR_RATIOS = (4 / 3, 1.5, 2.0, 2.5, 3.0, 4.0)  # RMS ratios tried in C6b (fixed before the run)


def battery_arousal(D, n_null_pool=20, seed=0, tone_source="eq", blink_tone=False, band_ctx=False, select_ratio=False):
    """C5 + C6: the fast-band reading (beta included, C5 = its Spearman in V1) and the AASM arousal rule on it,
    against the MASS SS01 experts and the NSRR / WSC technicians of the pool. The ``arousal`` reference is
    stage-free (annotations), so no circularity here."""
    cfg = CONCEPTS["arousal"]
    night = D["night"]
    R, S, v1b = _band_read(D, seed, ctx=band_ctx)
    St, v1 = _lex_state(D, seed, tone_source=tone_source, blink_tone=blink_tone)
    ew, v1e, fin_e = _word(D, cfg["emg"], seed)
    er, eh = RA.ref_mask(D, "emg_rise")
    fast_l, fast_t = _lse(R, cfg["bands"]["fast"]), _lse(S, cfg["bands"]["fast"])
    sl = {k: np.repeat(St[k]["state"] != 0, T) for k in ("lexicon", "oracle")}
    rm = {k: np.repeat(St[k]["state"] == 1, T) for k in ("lexicon", "oracle")}
    r, h = RA.ref_mask(D, "arousal_ann")
    ratio, grid = CR.CAP_RATIO, None
    if select_ratio:  # C6b: the threshold is chosen on the pool technicians with the true bands (oracle) only
        psel = (D["set"] == "pool") & h & np.repeat(np.isfinite(St["oracle"]["aton"]), T)
        grid = {f"{q:.3f}": CR.event_agreement(CR.arousal_rule(fast_t, night, sl["oracle"], rm["oracle"], er & eh,
                                                                ratio=q), r, night, psel)["f1"]
                for q in AR_RATIOS}
        ratio = float(max(grid, key=grid.get))
    lex = CR.arousal_rule(fast_l, night, sl["lexicon"], rm["lexicon"], ew, ratio=ratio)
    ora = CR.arousal_rule(fast_t, night, sl["oracle"], rm["oracle"], er & eh, ratio=ratio)
    abl = {"no_bg": CR.arousal_rule(fast_l, night, sl["lexicon"], rm["lexicon"], ew, use_bg=False),
           "no_prev": CR.arousal_rule(fast_l, night, sl["lexicon"], rm["lexicon"], ew, use_prev=False),
           "no_emg": CR.arousal_rule(fast_l, night, sl["lexicon"], rm["lexicon"], ew, use_emg=False),
           "no_state": CR.arousal_rule(fast_l, night, np.ones(len(night), bool), None, None, ratio=ratio)}
    abl_o = {"no_bg": CR.arousal_rule(fast_t, night, sl["oracle"], rm["oracle"], er & eh, use_bg=False),
             "no_prev": CR.arousal_rule(fast_t, night, sl["oracle"], rm["oracle"], er & eh, use_prev=False),
             "no_state": CR.arousal_rule(fast_t, night, np.ones(len(night), bool), None, None, ratio=ratio)}
    out = {"V1": {"bands_spearman_pool_loso": v1b, "emg_rise": v1e, **v1}, "V2": {}, "V3": {},
           "ratio": ratio, "ratio_grid_pool_oracle_f1": grid}
    avail = np.repeat(np.isfinite(St["lexicon"]["aton"]), T)
    en = night[::T]
    for setname in sorted(set(D["set"])):
        sel = (D["set"] == setname) & h & avail
        if not r[sel].any():
            continue
        rng = np.random.default_rng(seed)
        nul = []
        for _ in range(n_null_pool):
            st_p = CR.permute_epochs(St["lexicon"]["state"], en, rng)
            nul.append(CR.event_agreement(CR.arousal_rule(fast_l, night, np.repeat(st_p != 0, T),
                                                          np.repeat(st_p == 1, T), ew, ratio=ratio), r, night, sel)["f1"])
        e = {"n_units": int(len(set(night[sel]))),
             "lexicon": _agree(lex, r, night, sel, seed), "oracle": _agree(ora, r, night, sel, seed),
             "nulls": {"state_perm_q95": float(np.quantile(nul, 0.95)) if nul else None, "n": n_null_pool},
             "ablations": {a: CR.event_agreement(o, r, night, sel) for a, o in abl.items()},
             "ablations_oracle": {a: CR.event_agreement(o, r, night, sel) for a, o in abl_o.items()},
             "stage_of_detections": _stage_mix(D, lex, sel), "stage_of_reference": _stage_mix(D, r, sel)}
        fo = e["oracle"]["f1"]
        e["ret_vs_oracle"] = e["lexicon"]["f1"] / fo if fo else None
        out["V2"][setname] = e
        out["V3"][setname] = {a: e["lexicon"]["f1"] - v["f1"] for a, v in e["ablations"].items()}
        out["V3"][f"{setname}:oracle"] = {a: e["oracle"]["f1"] - v["f1"] for a, v in e["ablations_oracle"].items()}
    return out


def _split(D, o, word, ref, y, sel):
    """Of the word events in R or W epochs that hit a reference eye event: share called REM in R, called wake
    in W, and their mean (balanced accuracy of the R / W distinction)."""
    runs = CR.event_runs(word & sel, D["night"], gap=0)
    res = {}
    for st, k in ((4, "rem"), (0, "wake")):
        hit = [r for r in runs if y[(r[0] + r[1] - 1) // 2] == st and ref[r[0]:r[1]].any()]
        res[f"n_{k}"] = len(hit)
        res[f"acc_{k}"] = float(np.mean([o[k][a:b].any() for a, b in hit])) if hit else None
    if res["acc_rem"] is not None and res["acc_wake"] is not None:
        res["balanced"] = (res["acc_rem"] + res["acc_wake"]) / 2
    return res


def _stage_mix(D, det, sel):
    runs = CR.event_runs(det & sel, D["night"], gap=0)
    if not runs:
        return {}
    st = D["y"][[(a + b - 1) // 2 for a, b in runs]]
    names = ["W", "N1", "N2", "N3", "R"]
    return {names[s]: float((st == s).mean()) for s in range(5)}


def _per_site(D, det, ref, sel):
    out = {}
    for g in sorted(set(D["gname"][sel])):
        m = sel & (D["gname"] == g)
        out[g] = CR.event_agreement(det, ref, D["night"], m)["f1"]
    return out


def _pure_split(D, krule, sw, lex, sel):
    """Rule K-complexes with no slow wave within +-2 tokens and slow waves with no K-complex: fraction of each
    called KC / N3 / neither by the composite."""
    night = D["night"]
    near = lambda m: np.convolve(m.astype(float), np.ones(5), "same") > 0
    kp, sp = krule & ~near(sw) & sel, sw & ~near(krule) & sel
    out = {}
    for name, m in (("kc_pure", kp), ("sw_pure", sp)):
        runs = CR.event_runs(m, night, gap=0)
        if not runs:
            continue
        kc = np.array([lex["kc"][a:b].any() for a, b in runs])
        n3 = np.array([lex["n3"][a:b].any() for a, b in runs])
        out[name] = {"n": len(runs), "called_kc": float(kc.mean()), "called_n3": float(n3.mean()),
                     "neither": float((~kc & ~n3).mean())}
    return out


# ---------------------------------------------------------------------------- command
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def battery(a):
    t0 = time.time()
    cfg = CONCEPTS[a.concept]
    rem = a.concept in ("rem_eye", "alpha_ctx", "cap_a1", "arousal")
    refs = cfg["refs"] + (cfg["wake_refs"] if a.wake else [])
    D, keys, sets = RA.load(a.dump, "eeg", refs, frac=a.frac, n_ss01=a.n_ss01, seed=a.seed,
                            opt_mods=("eog", "emg") if (a.wake or rem) else ())
    print(f"[compose] loaded {len(keys)} nights, {len(D['y'])} tokens, {time.time() - t0:.0f} s", flush=True)
    if a.concept == "arousal":
        res = battery_arousal(D, n_null_pool=a.n_null_pool, seed=a.seed, tone_source=a.tone_source,
                              blink_tone=a.blink_tone, band_ctx=a.band_ctx, select_ratio=a.select_ratio)
    elif a.concept == "cap_a1":
        res = battery_cap(D, n_null_pool=a.n_null_pool, seed=a.seed, tone_source=a.tone_source,
                          blink_tone=a.blink_tone, band_ctx=a.band_ctx)
    elif a.concept == "alpha_ctx":
        res = battery_alpha(D, n_null_pool=a.n_null_pool, seed=a.seed, tone_source=a.tone_source,
                            blink_tone=a.blink_tone)
    elif rem:
        res = battery_rem(D, n_null_pool=a.n_null_pool, seed=a.seed, eye_eeg=a.eye_eeg, tone_source=a.tone_source)
    else:
        res = battery_kc(D, n_null=a.n_null, n_null_pool=a.n_null_pool, seed=a.seed, swa_source=a.swa_source,
                         wake=a.wake, n3_first=a.n3_first, eye_eeg=a.eye_eeg)
    out = {"iteration": a.iter, "concept": a.concept, "rule": cfg["rule"], "swa_source": a.swa_source, "wake": a.wake, "n3_first": a.n3_first, "eye_eeg": a.eye_eeg, "tone_source": a.tone_source, "blink_tone": a.blink_tone, "band_ctx": a.band_ctx,
           "thresholds": {"N3_SWA": CR.N3_SWA, "W_ALPHA": CR.W_ALPHA, "EYE_PRESENT": CR.EYE_PRESENT,
                          "ATONIA_EPOCH": CR.ATONIA_EPOCH, "KC_ISO_TOK": CR.KC_ISO_TOK,
                          "MERGE_GAP": CR.MERGE_GAP},
           "nights": {s: sum(v == s for v in sets.values()) for s in sorted(set(sets.values()))},
           "V0": None, "V4": None, **res, "minutes": (time.time() - t0) / 60}
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"{a.iter}_{a.concept}.json")
    json.dump(_jsonable(out), open(p, "w"), indent=1)
    print(f"[compose] wrote {p} ({out['minutes']:.1f} min)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("battery")
    b.add_argument("--concept", default="kc", choices=sorted(CONCEPTS))
    b.add_argument("--dump", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--iter", default="C1")
    b.add_argument("--frac", type=float, default=1.0)
    b.add_argument("--n-ss01", type=int, default=0)
    b.add_argument("--n-null", type=int, default=200)
    b.add_argument("--n-null-pool", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--swa-source", default="eq", choices=["eq", "words"])
    b.add_argument("--wake", action="store_true", help="C1c: AASM wake rule on lexicon words (needs EOG / EMG)")
    b.add_argument("--n3-first", action="store_true", help="C1d: AASM precedence, slow waves >= 20 %% is N3, never W")
    b.add_argument("--tone-source", default="word", choices=["word", "eq"], help="C2b: chin tone from the EMG epoch latent")
    b.add_argument("--blink-tone", action="store_true", help="C3b: blinks are wake only on a chin tone not low")
    b.add_argument("--select-ratio", action="store_true", help="C6b: arousal threshold chosen on the pool oracle")
    b.add_argument("--band-ctx", action="store_true", help="C4b: band powers read from P and P_ctx+-2")
    b.add_argument("--eye-eeg", action="store_true", help="C1e: blink / rem_eye words read P@eog and P (EEG)")
    a = ap.parse_args(argv)
    battery(a)


if __name__ == "__main__":
    main()
