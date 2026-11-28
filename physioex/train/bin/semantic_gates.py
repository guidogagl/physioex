"""Semantic gates of the learned tokenizer, without the AR (plan rev. 2 of xmai-lab/sleep-lens#17, T2).

    python -m physioex.train.bin.semantic_gates --scan CONCEPT_DUMP \\
        --model A=RUN_A:DUMP_A --model B=RUN_B:DUMP_B --model C=RUN_C:DUMP_C [--model codec3=RUN_3:DUMP_3] \\
        --out OUT.json [--n-boot 1000] [--max-nights N]

Every ``--model NAME=RUN:DUMP`` is a codec run (``run.json``, ``model.pt``) with the dump of its codes on the nights
of the ``concept_scan`` dump (``patch_codec dump --nights-from``: 4 grids shifted by 0, 0.5, 1 and 1.5 s). The names
fix the roles: ``A`` the tokenizer (stage weight lambda), ``B`` the same without the stage (lambda 0), ``C`` the
spectral twin (zero-phase patches, lambda); any other name (codec 3) is reported only.

Rules registered before the first result (plan rev. 2, 28.11.2026):

* **patch features**: for an FSQ codec, ``sem`` = the quantised values of the semantic group g_s (the first
  ``sem_dims`` FSQ dimensions) and ``latent`` = all the FSQ values; for the transform codec (codec 3), ``klt`` = the
  dequantised KLT coefficients of the whitened patch, gains applied. Token feature = mean over the 4 patches that
  cover the 0.5 s token; each night's features are centred by their mean over the night's sleep tokens (N1-R), as
  in :mod:`physioex.train.bin.surprise_type`;
* **prototypes**: the mean centred feature over the sleep tokens of each class of
  :data:`~physioex.train.bin.surprise_type.CLASSES` and of the background, on the pool test nights of the dump
  (12 units; never DREAMS or MASS SS01);
* **token score** of class ``c``: ``cos(f, proto_c) - cos(f, proto_background)``. **AP** per 0.5 s token, pooled
  over the nights of a set, on the stratum tokens (DREAMS ``k_complex_vis1`` and ``spindle_vis1``: N2-N3; MASS SS01
  ``arousal_ann``: N1-R) where every model has a feature. 95 % CIs by a paired night bootstrap (``--n-boot``
  resamples; the AP of a resample is computed on the scores binned at :data:`N_BINS` quantiles of the set, the
  point estimates on the raw scores);
* **G-S1, stop gate**: ``AP(A sem) - max(AP(C sem), AP(C latent))``, the max taken inside every resample.
  DREAMS K-complexes: >= +0.05 with the 95 % CI above 0. DREAMS spindles and MASS SS01 arousals: >= -0.02
  (non-inferiority as written in the plan; whether the lower CI bound also clears -0.02 is reported as
  ``strict``). ``A latent`` is reported the same way;
* **G-S2** (reported): ``A - B`` on the same APs, per feature, with CIs;
* **G-S3** (reported): K-complex (``k_complex_rule`` or ``m1:kc_m1`` in N2) against slow wave (``slow_wave`` in N3)
  against vertex (``vertex`` or ``m1:vertex_m1`` in N1). Events = runs of each mask within its stage, event feature
  = mean token feature, type = the nearest of the 3 prototypes (cosine). Prototypes from one half of the pool test
  nights, events from the other (2 folds by night, alternating within each unit). Balanced accuracy of the pooled
  confusion, 95 % CI by night bootstrap. Also the share of the DREAMS expert K-complexes (onset in N2-N3) typed as
  K-complex rather than slow wave (2 prototypes, from all pool test nights);
* **G-S4** (reported): on the patches of grid 0 of the pool test nights, H(g_s) (plug-in and Miller-Madow, bits),
  the number of distinct g_s values and I(g_s; stage) / H(g_s), stage = the scored epoch holding the patch centre;
  the same I with the stages shuffled over the patches is the bias control.

Writes ``OUT.json``: aggregate numbers only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from physioex.train.bin.surprise_type import (CLASSES, EP, OFFSETS, PATCH, POOL_UNITS, SETS, SLEEP, T, centre,
                                              class_masks, night_boot, token_features, unit_cos)

#: G-S3 classes: masks (any of) and the stage of their tokens
GS3 = {"kc": (("k_complex_rule", "m1:kc_m1"), 2), "slow_wave": (("slow_wave",), 3), "vertex": (("vertex", "m1:vertex_m1"), 1)}
GATE_KC, GATE_NONINF = 0.05, -0.02
N_BINS = 4096


def feature_fns(codec) -> dict:
    """Patch features of a codec: name -> function of the codes ``(P, levels)`` (int64 tensor) -> ``(P, D)``."""
    q = getattr(codec, "quantizer", None)
    if q == "transform":
        return {"klt": lambda c: codec.coefficients(c[None])[0]}
    if q != "fsq":
        raise ValueError(f"semantic gates: FSQ or transform codecs only, not {q}")
    out = {}
    if getattr(codec, "sem_dims", 0):
        out["sem"] = lambda c: codec.fsq.lookup(c)[..., : codec.sem_dims]
    out["latent"] = lambda c: codec.fsq.lookup(c)
    return out


def sem_symbols(codec, codes: torch.Tensor) -> np.ndarray:
    """FSQ codes ``(P, groups)`` -> ``(P,)`` int64: the joint value of the ``sem_dims`` levels of g_s (mixed radix)."""
    fsq = codec.fsq
    radix = fsq.levels ** torch.arange(fsq.group, device=codes.device)
    d = (torch.div(codes.long()[..., None], radix, rounding_mode="floor") % fsq.levels).flatten(-2)[..., : codec.sem_dims]
    w = fsq.levels ** torch.arange(codec.sem_dims, device=codes.device, dtype=torch.int64)
    return (d * w).sum(-1).cpu().numpy()


def patch_stages(ts: np.ndarray, y: np.ndarray, e0: int, n_patches: int) -> np.ndarray:
    """Stage of the scored epoch that holds the centre of each patch of grid 0 (-1 elsewhere)."""
    ts = np.asarray(ts, np.int64) - e0
    lut = np.full(int(ts.max()) + 1 if len(ts) else 0, -1, np.int64)
    lut[ts] = np.asarray(y, np.int64)
    e = (np.arange(n_patches) * PATCH + PATCH // 2) // EP
    out = np.full(n_patches, -1, np.int64)
    ok = e < len(lut)
    out[ok] = lut[e[ok]]
    return out


def entropy_bits(counts: np.ndarray, miller_madow: bool = False) -> float:
    c = np.asarray(counts, np.float64)
    c = c[c > 0]
    n = c.sum()
    if not n:
        return float("nan")
    p = c / n
    h = float(-(p * np.log2(p)).sum())
    return h + (len(c) - 1) / (2 * n * math.log(2)) if miller_madow else h


def mutual_information(x: np.ndarray, s: np.ndarray, miller_madow: bool = False) -> float:
    """I(x; s) in bits from the joint counts (plug-in, or Miller-Madow on each entropy); ``s`` >= 0."""
    hx = entropy_bits(np.unique(x, return_counts=True)[1], miller_madow)
    hs = entropy_bits(np.unique(s, return_counts=True)[1], miller_madow)
    hj = entropy_bits(np.unique(x * (int(s.max()) + 1) + s, return_counts=True)[1], miller_madow)
    return hx + hs - hj


def binned_ap(pos: np.ndarray, neg: np.ndarray) -> float:
    """AP of scores in ascending bins (``pos``, ``neg``: counts per bin), the tokens of a bin tied: equal to
    ``average_precision_score`` on the bin indices."""
    p, q = np.asarray(pos, np.float64)[::-1], np.asarray(neg, np.float64)[::-1]
    P = p.sum()
    if not P or not q.sum():
        return float("nan")
    tp, fp = np.cumsum(p), np.cumsum(q)
    return float((p * tp / np.maximum(tp + fp, 1e-12)).sum() / P)


def night_hists(scores: list, refs: list, n_bins: int = N_BINS):
    """Per-night counts of positive and negative tokens in the quantile bins of the pooled scores."""
    edges = np.unique(np.quantile(np.concatenate(scores), np.linspace(0, 1, n_bins + 1)[1:-1]))
    nb = len(edges) + 1
    hp, hn = [], []
    for s, r in zip(scores, refs):
        b = np.searchsorted(edges, s, side="right")
        hp.append(np.bincount(b[r], minlength=nb))
        hn.append(np.bincount(b[~r], minlength=nb))
    return np.stack(hp).astype(np.float64), np.stack(hn).astype(np.float64)


def boot_aps(hists: dict, n_boot: int, seed: int) -> dict:
    """``key -> (Hp, Hn)`` per night -> ``key -> (n_boot,)`` APs, with the same night resamples for every key."""
    keys = list(hists)
    N = hists[keys[0]][0].shape[0]
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
        for k in keys:
            out[k][b] = binned_ap(w @ hists[k][0], w @ hists[k][1])
    return out


def ci95(v: np.ndarray):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def run_means(F: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Mean of the rows of ``F`` over every ``[start, end)`` of ``ev`` (rows assumed finite)."""
    if not len(ev):
        return np.zeros((0, F.shape[1]), np.float32)
    cs = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, 0, dtype=np.float64)])
    return ((cs[ev[:, 1]] - cs[ev[:, 0]]) / (ev[:, 1] - ev[:, 0])[:, None]).astype(np.float32)


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--scan", required=True, help="concept_scan dump (ts, y; refs/, m1/)")
    ap_.add_argument("--model", action="append", required=True, help="NAME=RUN:DUMP (A, B, C; others reported)")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--n-boot", type=int, default=1000)
    ap_.add_argument("--max-nights", type=int, default=0, help="per unit; smoke tests")
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args(argv)
    from physioex.train.bin import patch_codec
    from physioex.train.bin.concept_refs_eval import events
    from physioex.train.bin.concept_scan import load_refs
    from physioex.train.bin.surprise_eval import ap as exact_ap

    t0 = time.time()
    dev = torch.device("cpu")  # lookups only: no GPU needed
    models = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        run, dump = rest.rsplit(":", 1)
        _, codec = patch_codec.load_codec(run, dev)
        models[name] = {"codec": codec, "dump": dump, "run": run, "fns": feature_fns(codec)}
    assert "A" in models and "C" in models, "semantic gates: --model A=... and --model C=... are required"
    assert "sem" in models["A"]["fns"], "semantic gates: A needs sem_dims > 0"
    kinds = [(m, k) for m, d in models.items() for k in d["fns"]]
    label = {mk: f"{mk[0]}_{mk[1]}" for mk in kinds}

    @torch.no_grad()
    def load_night(unit, f):
        pr = os.path.join(a.scan, "refs", unit, f)
        paths = {m: os.path.join(d["dump"], unit, f) for m, d in models.items()}
        if not os.path.exists(pr) or not all(os.path.exists(p) for p in paths.values()):
            return None
        d = np.load(os.path.join(a.scan, unit, f))
        ts, y = d["ts"], np.asarray(d["y"], np.int64)
        stage = np.repeat(y, T)
        masks = {k: v for k, v in load_refs(pr).items()
                 if not k.startswith("_") and not k.endswith("_epoch") and v.shape == stage.shape}
        pm = os.path.join(a.scan, "m1", unit, f)
        if os.path.exists(pm):
            masks.update({f"m1:{k}": v for k, v in load_refs(pm).items()
                          if not k.startswith("_") and not k.endswith("_epoch") and v.shape == stage.shape})
        F, sym = {}, {}
        for m, md in models.items():
            z = np.load(paths[m])
            e0 = int(z["e0"])
            grids = {o: torch.from_numpy(np.asarray(z[f"c{o}"], np.int64)) for o in OFFSETS if f"c{o}" in z.files}
            for k, fn in md["fns"].items():
                F[(m, k)] = centre(token_features({o: fn(c).float().numpy() for o, c in grids.items()}, ts, e0),
                                   stage).astype(np.float32)
            if "sem" in md["fns"] and 0 in grids:
                st = patch_stages(ts, y, e0, len(grids[0]))
                keep = st >= 0
                sym[m] = (sem_symbols(md["codec"], grids[0])[keep], st[keep])
        fin = np.all(np.stack([np.isfinite(v).all(1) for v in F.values()]), 0)
        return {"key": f"{unit}/{f[:-4]}", "stage": stage, "masks": masks, "F": F, "fin": fin, "sym": sym}

    def nights_of(unit):
        ud = os.path.join(a.scan, unit)
        fs = sorted(f for f in os.listdir(ud) if f.endswith(".npz")) if os.path.isdir(ud) else []
        n = 0
        for f in fs:
            if a.max_nights and n >= a.max_nights:
                break
            nd = load_night(unit, f)
            if nd is not None:
                n += 1
                yield nd

    # pass 1: pool test nights -> prototypes, G-S3 events, G-S4 symbols
    names = list(CLASSES) + ["background"]
    S = {mk: {c: 0.0 for c in names} for mk in kinds}
    Nc = {c: 0 for c in names}
    S3 = {mk: [{c: 0.0 for c in GS3} for _ in range(2)] for mk in kinds}
    N3 = [{c: 0 for c in GS3} for _ in range(2)]
    ev3 = {mk: [] for mk in kinds}
    ev3_meta = []  # (night, fold, class) per event, the same order for every model
    syms = {m: ([], []) for m in models if "sem" in models[m]["fns"]}
    n_pool = 0
    for unit in POOL_UNITS:
        for j, nd in enumerate(nights_of(unit)):
            fold, st = j % 2, nd["stage"]
            cm = class_masks(nd["masks"], len(st))
            sleep = np.isin(st, SLEEP) & nd["fin"]
            g3, ev = {}, {}
            for ci, (c, (keys, s)) in enumerate(GS3.items()):
                m_ = np.zeros(len(st), bool)
                for k in keys:
                    if k in nd["masks"]:
                        m_ |= nd["masks"][k]
                g3[c] = m_ & (st == s) & nd["fin"]
                ev[c] = events(g3[c])
                N3[fold][c] += int(g3[c].sum())
                ev3_meta += [(n_pool, fold, ci)] * len(ev[c])
            for c in names:
                Nc[c] += int((cm[c] & sleep).sum())
            for mk in kinds:
                F = nd["F"][mk]
                for c in names:
                    S[mk][c] = S[mk][c] + F[cm[c] & sleep].sum(0, dtype=np.float64)
                for c in GS3:
                    S3[mk][fold][c] = S3[mk][fold][c] + F[g3[c]].sum(0, dtype=np.float64)
                    ev3[mk].append(run_means(F, ev[c]))
            for m, (x, s) in nd["sym"].items():
                syms[m][0].append(x)
                syms[m][1].append(s)
            n_pool += 1
            print(f"[semantic_gates] pool {nd['key']} ({time.time() - t0:.0f}s)", flush=True)
    names = [c for c in names if Nc[c] > 0]
    protos = {mk: unit_cos(np.stack([S[mk][c] / Nc[c] for c in names])) for mk in kinds}
    res = {"rules": {"gate_kc": GATE_KC, "gate_noninferiority": GATE_NONINF, "n_boot": a.n_boot, "n_bins": N_BINS,
                     "classes": {c: list(v) for c, v in CLASSES.items()},
                     "gs3_classes": {c: {"masks": list(k), "stage": s} for c, (k, s) in GS3.items()}},
           "models": {m: {"run": os.path.abspath(d["run"]), "dump": os.path.abspath(d["dump"]), "features": list(d["fns"])}
                      for m, d in models.items()},
           "prototypes": {"n_pool_nights": n_pool, "names": names, "n_tokens": {c: Nc[c] for c in names},
                          "cosine": {label[mk]: np.round(p @ p.T, 3).tolist() for mk, p in protos.items()}},
           "g_s1": {}, "g_s2": {}}

    # pass 2: DREAMS and MASS SS01 -> token APs (G-S1, G-S2) and the DREAMS K-complexes typed against slow waves
    p_ks = {mk: unit_cos(np.stack([(S3[mk][0][c] + S3[mk][1][c]) / max(N3[0][c] + N3[1][c], 1)
                                   for c in ("kc", "slow_wave")])) for mk in kinds}
    kc_typed = {mk: [] for mk in kinds}
    gate = {}
    for set_name, (unit, ref_key, cls, stages) in SETS.items():
        if cls not in names or "background" not in names:
            continue
        ci, ib = names.index(cls), names.index("background")
        per = []
        for nd in nights_of(unit):
            if ref_key not in nd["masks"]:
                continue
            stratum = np.isin(nd["stage"], stages) & nd["fin"]
            ref = nd["masks"][ref_key]
            sc = {}
            for mk in kinds:
                U = unit_cos(nd["F"][mk][stratum].astype(np.float64))
                sc[mk] = U @ protos[mk][ci] - U @ protos[mk][ib]
            per.append({"ref": ref[stratum], "score": sc})
            if set_name == "dreams_kc":
                ev = events(ref & nd["fin"])
                ev = ev[np.isin(nd["stage"][ev[:, 0]], stages)] if len(ev) else ev
                for mk in kinds:
                    t = np.argmax(unit_cos(run_means(nd["F"][mk], ev).astype(np.float64)) @ p_ks[mk].T, 1) if len(ev) else []
                    kc_typed[mk].append((int((np.asarray(t) == 0).sum()), len(ev)))
        if not per:
            continue
        refs = [p["ref"] for p in per]
        o = {"n_nights": len(per), "n_tokens": int(sum(len(r) for r in refs)), "stratum": list(stages),
             "base_rate": float(sum(r.sum() for r in refs) / max(sum(len(r) for r in refs), 1)), "ap": {}, "ap_binned": {}}
        hists = {}
        for mk in kinds:
            sc = [p["score"][mk] for p in per]
            o["ap"][label[mk]] = exact_ap(np.concatenate(sc), np.concatenate(refs))
            hists[mk] = night_hists(sc, refs)
            o["ap_binned"][label[mk]] = binned_ap(hists[mk][0].sum(0), hists[mk][1].sum(0))
        boot = boot_aps(hists, a.n_boot, a.seed) if len(per) > 1 and a.n_boot else None
        o["ci"] = {label[mk]: ci95(boot[mk]) for mk in kinds} if boot else {}
        c_kinds = [("C", k) for k in models["C"]["fns"]]
        twin_ap = max(o["ap"][label[mk]] for mk in c_kinds)
        o["twin_best"] = label[max(c_kinds, key=lambda mk: o["ap"][label[mk]])]
        d1 = {}
        for k in models["A"]["fns"]:
            e = {"delta": o["ap"][label[("A", k)]] - twin_ap}
            if boot:
                e["ci"] = ci95(boot[("A", k)] - np.max(np.stack([boot[mk] for mk in c_kinds]), 0))
            d1[k] = e
        o["a_minus_max_c"] = d1
        res["g_s1"][set_name] = o
        if "B" in models:
            d2 = {}
            for k in models["A"]["fns"]:
                if k in models["B"]["fns"]:
                    e = {"delta": o["ap"][label[("A", k)]] - o["ap"][label[("B", k)]]}
                    if boot:
                        e["ci"] = ci95(boot[("A", k)] - boot[("B", k)])
                    d2[k] = e
            res["g_s2"][set_name] = d2
        p = d1["sem"]
        lo = p.get("ci", [None])[0]
        if set_name == "dreams_kc":
            gate[set_name] = {"delta": p["delta"], "ci": p.get("ci"), "rule": f">= {GATE_KC}, CI > 0",
                              "pass": bool(p["delta"] >= GATE_KC and lo is not None and lo > 0)}
        else:
            gate[set_name] = {"delta": p["delta"], "ci": p.get("ci"), "rule": f">= {GATE_NONINF}",
                              "pass": bool(p["delta"] >= GATE_NONINF),
                              "strict": bool(lo is not None and lo >= GATE_NONINF)}
        print(f"[semantic_gates] {set_name}: {len(per)} nights, AP " + " ".join(f"{label[mk]} {o['ap'][label[mk]]:.3f}"
              for mk in kinds) + f" | A sem - max C {p['delta']:+.3f} {p.get('ci')}", flush=True)
    res["g_s1"]["gate"] = {"sets": gate, "pass": len(gate) == len(SETS) and all(v["pass"] for v in gate.values())}

    # G-S3: K-complex vs slow wave vs vertex, 2 folds by night
    meta = np.asarray(ev3_meta, np.int64).reshape(-1, 3)
    g3 = {"n_events": {c: int((meta[:, 2] == i).sum()) for i, c in enumerate(GS3)}, "models": {}}
    for mk in kinds:
        E = np.concatenate(ev3[mk]) if ev3[mk] else np.zeros((0, 1), np.float32)
        P3 = [unit_cos(np.stack([S3[mk][f][c] / max(N3[f][c], 1) for c in GS3])) for f in range(2)]
        pred = np.full(len(E), -1, np.int64)
        for f in range(2):
            sel = meta[:, 1] == f
            if sel.any():
                pred[sel] = np.argmax(unit_cos(E[sel].astype(np.float64)) @ P3[1 - f].T, 1)
        conf = np.zeros((3, 3), np.int64)
        np.add.at(conf, (meta[:, 2], pred), 1)
        per_night = []
        for nidx in np.unique(meta[:, 0]):
            s = meta[:, 0] == nidx
            per_night.append([(int(((meta[s, 2] == i) & (pred[s] == i)).sum()), int((meta[s, 2] == i).sum()))
                              for i in range(3)])

        def bal(groups):
            g = groups[0]
            r = [sum(x[i][0] for x in g) / sum(x[i][1] for x in g) for i in range(3) if sum(x[i][1] for x in g)]
            return float(np.mean(r)) if len(r) == 3 else float("nan")

        rec = conf.diagonal() / np.maximum(conf.sum(1), 1)
        o = {"balanced_accuracy": float(rec.mean()), "recall": {c: float(r) for c, r in zip(GS3, rec)},
             "confusion": conf.tolist(), "ci": night_boot([per_night], bal, a.n_boot, a.seed) if per_night else None}
        kt = kc_typed[mk]
        if kt and sum(n for _, n in kt):
            share = lambda groups: sum(h for h, _ in groups[0]) / max(sum(n for _, n in groups[0]), 1)  # noqa: E731
            o["dreams_kc_typed_kc"] = {"share": share([kt]), "ci": night_boot([kt], share, a.n_boot, a.seed),
                                       "n_events": int(sum(n for _, n in kt)), "n_nights": len(kt)}
        g3["models"][label[mk]] = o
    res["g_s3"] = g3

    # G-S4: entropy of g_s and information on the stage
    rng = np.random.default_rng(a.seed)
    g4 = {}
    for m, (xs, ss) in syms.items():
        if not xs:
            continue
        x, s = np.concatenate(xs), np.concatenate(ss)
        cnt = np.unique(x, return_counts=True)[1]
        h, i = entropy_bits(cnt), mutual_information(x, s)
        ish = mutual_information(x, rng.permutation(s))
        g4[m] = {"n_patches": int(len(x)), "distinct": int(len(cnt)),
                 "max_bits": float(models[m]["codec"].sem_dims * math.log2(models[m]["codec"].fsq.levels)),
                 "H_bits": h, "H_mm_bits": entropy_bits(cnt, True), "H_stage_bits": entropy_bits(np.bincount(s)),
                 "I_bits": i, "I_mm_bits": mutual_information(x, s, True), "I_shuffled_bits": ish,
                 "I_over_H": i / h if h else None, "I_minus_shuffled_over_H": (i - ish) / h if h else None}
    res["g_s4"] = g4
    res["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for mk in kinds:
        o = g3["models"][label[mk]]
        print(f"[semantic_gates] G-S3 {label[mk]}: balanced accuracy {o['balanced_accuracy']:.3f} {o['ci']}"
              + (f", DREAMS KC typed KC {o['dreams_kc_typed_kc']['share']:.3f}" if "dreams_kc_typed_kc" in o else ""),
              flush=True)
    for m, o in g4.items():
        print(f"[semantic_gates] G-S4 {m}: H {o['H_bits']:.2f} bits ({o['distinct']} values), I/H {o['I_over_H']:.3f}"
              f" (shuffled {o['I_shuffled_bits']:.3f} bits)", flush=True)
    print(f"[semantic_gates] G-S1 {res['g_s1']['gate']} -> {a.out}", flush=True)
    print("[semantic_gates] DONE", flush=True)


if __name__ == "__main__":
    main()
