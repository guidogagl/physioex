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
        "rule": "kc = delta word event AND swa_frac < 0.20 AND alpha_frac <= 0.50 AND isolated +-4 tokens; "
                "n3 = delta word event AND swa_frac >= 0.20",
    },
}


# ---------------------------------------------------------------------------- components
def _word(D, cfg, seed=0, max_fit=100_000):
    """Token scores of the P word: LOSO on the pool, one pool fit elsewhere; threshold at the pool rate."""
    r, h = RA.ref_mask(D, cfg["ref"])
    pool = (D["set"] == "pool") & h
    ip = np.flatnonzero(pool)
    gn = D["gname"]
    groups = sorted(set(gn[ip]))
    out = {}

    def scores(inputs):
        X, _, bad = RA.build(D, inputs, np.arange(len(r)))
        full = np.full(len(r), np.nan)
        rng = np.random.default_rng(seed)
        full[:] = RA._fit_predict(X, [], r, ip, np.arange(len(r)), max_fit, rng)
        loso = RA.fold_probe(X[ip], [], r[ip], gn[ip], groups, max_fit=max_fit, seed=seed)
        return full, loso, bad

    full, loso, bad = scores(cfg["inputs"])
    cfull, closo, cbad = scores(cfg["ceiling"])
    ok = np.isfinite(loso)
    rate = float(r[ip].mean())
    thr = float(np.quantile(full[ip], 1 - rate))
    s = full.copy()
    s[ip] = loso
    word = (s >= thr) & np.isfinite(s)
    a_w, a_c = RA.ap(loso[ok], r[ip][ok]), RA.ap(closo[ok], r[ip][ok])
    dre = (D["set"] == "dreams") & h
    out = {"ref": cfg["ref"], "inputs": cfg["inputs"], "ceiling_inputs": cfg["ceiling"], "skipped": bad + cbad,
           "pool_rate": rate, "threshold": thr, "ap_pool_loso": a_w, "ap_ceiling_loso": a_c,
           "retC": a_w / a_c if a_c else np.nan,
           "ap_dreams": RA.ap(full[dre], r[dre]) if dre.any() and r[dre].any() else None,
           "ap_ceiling_dreams": RA.ap(cfull[dre], r[dre]) if dre.any() and r[dre].any() else None,
           "fire_rate": {k: float(word[D["set"] == k].mean()) for k in sorted(set(D["set"]))}}
    return word, out


def _pred(D, cfg, seed=0):
    """Epoch predicate: ridge on the epoch latents (with squares), target = reference fraction of the epoch."""
    from scipy.stats import spearmanr

    t, h = RA.epoch_target(D, cfg["ref"])
    X = np.hstack([np.hstack([RA.epoch_input(D, n), RA.epoch_input(D, n) ** 2]) for n in cfg["inputs"]])
    st = D["set"][::T]
    gn = D["gname"][::T]
    pool = (st == "pool") & h
    ip = np.flatnonzero(pool)
    rng = np.random.default_rng(seed)
    full = RA._fit_predict(X, [], t, ip, np.arange(len(t)), 200_000, rng, ridge=True)
    loso = RA.fold_probe(X[ip], [], t[ip], gn[ip], sorted(set(gn[ip])), max_fit=200_000, seed=seed, ridge=True)
    v = full.copy()
    v[ip] = loso
    ok = np.isfinite(loso)
    dre = (st == "dreams") & h
    rho = lambda a, b: float(spearmanr(a, b)[0]) if len(a) > 2 else None
    return v, t, {"ref": cfg["ref"], "inputs": cfg["inputs"],
                  "spearman_pool_loso": rho(loso[ok], t[ip][ok]),
                  "spearman_dreams": rho(full[dre], t[dre]) if dre.any() else None,
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


def _nulls(D, word, swa, alpha, ref, sel, key, n_null, seed):
    """Nulls on the whole nights that hold ``sel`` (tokens and epochs sliced together)."""
    rng = np.random.default_rng(seed + 7)
    ix = np.flatnonzero(np.isin(D["night"], np.unique(D["night"][sel])))
    ep = ix[::T] // T
    night, en = D["night"][ix], D["night"][ix][::T]
    word, ref, sel, swa, alpha = word[ix], ref[ix], sel[ix], swa[ep], alpha[ep]
    f_pred, f_word = [], []
    for _ in range(n_null):
        o = CR.kc_n3_rule(word, night, CR.permute_epochs(swa, en, rng), CR.permute_epochs(alpha, en, rng))
        f_pred.append(CR.event_agreement(o[key], ref, night, sel)["f1"])
        o = CR.kc_n3_rule(CR.shift_within_nights(word, night, rng), night, swa, alpha)
        f_word.append(CR.event_agreement(o[key], ref, night, sel)["f1"])
    q = lambda f: float(np.nanquantile(f, 0.95)) if f else None
    return {"pred_perm_q95": q(f_pred), "pred_perm_mean": float(np.nanmean(f_pred)) if f_pred else None,
            "word_roll_q95": q(f_word), "word_roll_mean": float(np.nanmean(f_word)) if f_word else None,
            "n": n_null}


def battery_kc(D, n_null=200, n_null_pool=20, seed=0):
    cfg = CONCEPTS["kc"]
    night = D["night"]
    word, v1w = _word(D, cfg["word"], seed)
    swa, swa_t, v1s = _pred(D, cfg["preds"]["swa"], seed)
    alpha, alpha_t, v1a = _pred(D, cfg["preds"]["alpha"], seed)
    dref, _ = RA.ref_mask(D, "delta")
    lex = CR.kc_n3_rule(word, night, swa, alpha)
    ora = CR.kc_n3_rule(dref, night, swa_t, alpha_t)
    abl = {"no_swa": CR.kc_n3_rule(word, night, swa, alpha, use_swa=False),
           "no_alpha": CR.kc_n3_rule(word, night, swa, alpha, use_alpha=False),
           "no_iso": CR.kc_n3_rule(word, night, swa, alpha, use_iso=False),
           "word_only": CR.kc_n3_rule(word, night, swa, alpha, use_swa=False, use_alpha=False, use_iso=False)}
    abl_o = {"no_swa": CR.kc_n3_rule(dref, night, swa_t, alpha_t, use_swa=False),  # does the rule itself need it?
             "no_alpha": CR.kc_n3_rule(dref, night, swa_t, alpha_t, use_alpha=False),
             "no_iso": CR.kc_n3_rule(dref, night, swa_t, alpha_t, use_iso=False),
             "word_only": CR.kc_n3_rule(dref, night, swa_t, alpha_t, use_swa=False, use_alpha=False, use_iso=False)}
    out = {"V1": {"word": v1w, "swa": v1s, "alpha": v1a}, "V2": {}, "V3": {}}

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
             "nulls": _nulls(D, word, swa, alpha, r, sel, "kc", n_null, seed),
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
            "nulls": _nulls(D, word, swa, alpha, krule, sel, "kc", n_null_pool, seed),
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
    D, keys, sets = RA.load(a.dump, "eeg", cfg["refs"], frac=a.frac, n_ss01=a.n_ss01, seed=a.seed)
    print(f"[compose] loaded {len(keys)} nights, {len(D['y'])} tokens, {time.time() - t0:.0f} s", flush=True)
    res = battery_kc(D, n_null=a.n_null, n_null_pool=a.n_null_pool, seed=a.seed)
    out = {"iteration": a.iter, "concept": a.concept, "rule": cfg["rule"],
           "thresholds": {"N3_SWA": CR.N3_SWA, "W_ALPHA": CR.W_ALPHA, "KC_ISO_TOK": CR.KC_ISO_TOK,
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
    a = ap.parse_args(argv)
    battery(a)


if __name__ == "__main__":
    main()
