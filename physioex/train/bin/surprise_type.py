"""Type of events from prototypes of the rule detectors (plan of 26.11.2026, S5, xmai-lab/sleep-lens#22).

    python -m physioex.train.bin.surprise_type --codec CODEC_RUN --codes CODES_DUMP --twin TWIN_DUMP \\
        --scan CONCEPT_DUMP --surprise AR_RUN/surprise --twin-surprise TWIN_AR_RUN/surprise --levels L --out OUT.json

The plan reads the type of an event from the codes of its span, with prototypes taken from the rule detectors
(M1 rules and ``concept_refs``) on the pool nights, then evaluated on DREAMS, with no fit on DREAMS.

Rules fixed before the first result (27.11.2026):

* **token feature** (``--feature latent``, codec 1): for the codes, the codec latent of the first ``L`` levels
  (``PatchCodec.embed``) of the 4
  patches (grids shifted by 0, 0.5, 1 and 1.5 s) that cover the 0.5 s token, averaged; for the spectral twin,
  the z-scored patch spectra rebuilt from its first ``L`` codebooks, averaged the same way. Each night's
  features are centred by their mean over the night's sleep tokens (N1-R) of the dumped epochs. With codec 3
  (``--feature ar``, registered 28.11 before any result: the transform codec has no learned latent, and its
  coefficients follow the phase of the signal) the patch feature is the state of the AR on that dump
  (:func:`~physioex.models.eeg_ar.hidden_codes`, the place that reads the patch), for the codes and for the
  zero-phase twin alike, then averaged and centred the same way;
* **prototypes**: mean centred feature over the sleep tokens of each class on the pool test nights of the
  ``concept_scan`` dump (12 units; never DREAMS or MASS SS01). Classes (:data:`CLASSES`): K-complex =
  ``k_complex_rule`` or ``m1:kc_m1``; spindle = ``spindle_rule``; arousal = ``m1:arousal_m1``; vertex = ``vertex``
  or ``m1:vertex_m1``; sawtooth = ``sawtooth`` or ``m1:sawtooth_m1``; alpha = ``alpha``; artifact =
  ``artifact_eeg``; background = sleep tokens in none of these masks. A token in two masks counts for both;
* **event feature** = mean of its token features; **type** = the prototype of highest cosine similarity;
* **expert events** (DREAMS ``k_complex_vis1`` and ``spindle_vis1``, onset in N2-N3; MASS SS01 ``arousal_ann``,
  onset in N1-R): share typed as their own class, against 3 random spans per event of the same length with
  onset in the same stratum, away from the set's events (chance); 95 % CI of the difference by night
  bootstrap (1000 resamples);
* **K-complex against spindle** on DREAMS: balanced accuracy of the choice between these two prototypes alone
  on the expert events (chance 0.5), codes, twin and their difference, with a 95 % CI by night bootstrap
  within each excerpt set (1000 resamples);
* **surprise events** (runs of ``z > 2`` within the stratum, as in :mod:`surprise_eval`; the codes' events
  with the codes' features, the twin's with the twin's): share typed as the expert's class among those that
  match an expert event (IoU >= 0.2), and the distribution of types over all events of the set.

Writes ``OUT.json``: aggregate numbers only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

T, TOK, EP = 60, 50, 3000  # tokens per epoch, samples per token, samples per epoch (100 Hz)
PATCH, OFFSETS = 200, (0, 50, 100, 150)
SLEEP, NREM23 = (1, 2, 3, 4), (2, 3)
POOL_UNITS = ("ccshs", "dcsm", "hmc", "hpap_lab_full", "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05", "mesa",
              "mros", "sleepedf", "wsc_visit1")
CLASSES = {"kc": ("k_complex_rule", "m1:kc_m1"), "spindle": ("spindle_rule",), "arousal": ("m1:arousal_m1",),
           "vertex": ("vertex", "m1:vertex_m1"), "sawtooth": ("sawtooth", "m1:sawtooth_m1"), "alpha": ("alpha",),
           "artifact": ("artifact_eeg",)}
NAMES = list(CLASSES) + ["background"]
SETS = {  # name: (unit, expert mask, class, stratum)
    "dreams_kc": ("dreams", "k_complex_vis1", "kc", NREM23),
    "dreams_spindle": ("dreams", "spindle_vis1", "spindle", NREM23),
    "mass_ss01_arousal": ("mass_ss01", "arousal_ann", "arousal", SLEEP),
}
Z_EVENT, N_CHANCE, N_BOOT = 2.0, 3, 1000


def token_features(per_patch: dict, ts: np.ndarray, e0: int) -> np.ndarray:
    """``(len(ts) * T, D)``: mean over the grids of the feature of the patch covering each token of the dumped
    epochs (NaN where no grid covers it). ``per_patch[o]``: ``(P_o, D)`` features of grid ``o`` (samples)."""
    start = ((np.asarray(ts, np.int64) - e0)[:, None] * EP + np.arange(T)[None] * TOK).reshape(-1)
    D = next(iter(per_patch.values())).shape[1]
    acc, cnt = np.zeros((len(start), D)), np.zeros(len(start))
    for o, f in per_patch.items():
        p = (start - o) // PATCH
        ok = (start >= o) & (p < len(f))
        acc[ok] += f[p[ok]]
        cnt[ok] += 1
    out = np.full_like(acc, np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0, None]
    return out


def centre(F: np.ndarray, stage: np.ndarray) -> np.ndarray:
    ok = np.isin(stage, SLEEP) & np.isfinite(F).all(1)
    return F - (F[ok].mean(0) if ok.any() else 0.0)


def class_masks(masks: dict, n: int) -> dict:
    out = {}
    for c, keys in CLASSES.items():
        m = np.zeros(n, bool)
        for k in keys:
            if k in masks:
                m |= masks[k]
        out[c] = m
    out["background"] = ~np.any(np.stack([out[c] for c in CLASSES]), 0)
    return out


def unit_cos(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def type_of(F: np.ndarray, ev: np.ndarray, protos: np.ndarray, names=None):
    """Index of the nearest prototype (cosine) for each ``[start, end)`` event; -1 where a token has no feature."""
    out = np.full(len(ev), -1, np.int64)
    for i, (a, b) in enumerate(ev):
        f = F[a:b]
        if len(f) and np.isfinite(f).all():
            out[i] = int(np.argmax(unit_cos(protos) @ unit_cos(f.mean(0))))
    return out


def chance_spans(ev: np.ndarray, stratum: np.ndarray, busy: np.ndarray, rng, n_per: int) -> np.ndarray:
    """``n_per`` random spans per event, same length, onset in the stratum, no token of ``busy``."""
    out, n = [], len(stratum)
    cand = np.flatnonzero(stratum)
    for a, b in ev:
        L = b - a
        got, tries = 0, 0
        while got < n_per and tries < 200 and len(cand):
            s = int(cand[rng.integers(len(cand))])
            tries += 1
            if s + L <= n and not busy[s:s + L].any():
                out.append((s, s + L))
                got += 1
    return np.array(out, np.int64).reshape(-1, 2)


def night_boot(groups, stat, n_boot: int, seed: int):
    """95 % CI of ``stat(list of per-night items)`` resampling nights within each group (list of lists)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(stat([[g[i] for i in rng.integers(0, len(g), len(g))] for g in groups]))
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [None, None]


def load_eval_night(a, unit, f, feats_fn, twin_fn, load_refs):
    sid = f[:-4]
    pc, pt = os.path.join(a.codes, unit, f), os.path.join(a.twin, unit, f)
    ps, pr = os.path.join(a.scan, unit, f), os.path.join(a.scan, "refs", unit, f)
    if not all(os.path.exists(p) for p in (pc, pt, ps, pr)):
        return None
    d, zc, zt = np.load(ps), np.load(pc), np.load(pt)
    ts, e0 = d["ts"], int(zc["e0"])
    if int(zt["e0"]) != e0:
        return None
    stage = np.repeat(np.asarray(d["y"], np.int64), T)
    masks = {k: v for k, v in load_refs(pr).items() if not k.endswith("_epoch") and v.shape == stage.shape}
    pm = os.path.join(a.scan, "m1", unit, f)
    if os.path.exists(pm):
        masks.update({f"m1:{k}": v for k, v in load_refs(pm).items()
                      if not k.endswith("_epoch") and v.shape == stage.shape})
    Fc = centre(token_features({o: feats_fn(zc[f"c{o}"]) for o in OFFSETS if f"c{o}" in zc.files}, ts, e0), stage)
    Ft = centre(token_features({o: twin_fn(zt[f"c{o}"]) for o in OFFSETS if f"c{o}" in zt.files}, ts, e0), stage)
    out = {"key": f"{unit}/{sid}", "stage": stage, "masks": masks, "F": {"codes": Fc, "twin": Ft}, "z": {}}
    for tag, folder in (("codes", a.surprise), ("twin", a.twin_surprise)):
        p = os.path.join(folder, unit, f) if folder else None
        if p and os.path.exists(p):
            from physioex.train.bin.surprise_eval import aligned_scores, zscore

            out["z"][tag] = zscore(aligned_scores(np.load(p), ts))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codec", required=True, help="patch codec run (run.json, model.pt)")
    ap.add_argument("--codes", required=True, help="codes dump of the codec")
    ap.add_argument("--twin", required=True, help="dump of the spectral twin (twin.pt and codes)")
    ap.add_argument("--scan", required=True, help="concept_scan dump (ts, y; refs/, m1/)")
    ap.add_argument("--surprise", default=None, help="surprise of the AR on the codes")
    ap.add_argument("--twin-surprise", default=None, help="surprise of the AR on the twin")
    ap.add_argument("--levels", type=int, required=True)
    ap.add_argument("--feature", choices=("latent", "ar"), default="latent",
                    help="latent: codec latent and twin codebooks (codec 1); ar: state of the AR on each dump (codec 3)")
    ap.add_argument("--ar", default=None, help="--feature ar: AR run on the codes")
    ap.add_argument("--twin-ar", default=None, help="--feature ar: AR run on the twin's codes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nights", type=int, default=0, help="per unit; smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from physioex.models.patch_codec import SpectralTwin
    from physioex.train.bin.concept_refs_eval import IOU, events
    from physioex.train.bin.concept_scan import load_refs
    from physioex.train.bin.patch_codec import load_codec
    from physioex.train.bin.surprise_eval import matched_refs

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = a.levels
    if a.feature == "ar":  # codec 3 (S3 rev. 2): the state of each AR on its own codes, codes and twin alike
        from physioex.models.eeg_ar import hidden_codes
        from physioex.train.bin.eeg_ar import load_model

        run_c, ar_c = load_model(a.ar, dev)
        run_t, ar_t = load_model(a.twin_ar, dev)

        def feats_fn(c):
            return hidden_codes(ar_c, np.ascontiguousarray(c[:, :ar_c.levels]), int(run_c["args"]["window"]))

        def twin_fn(c):
            return hidden_codes(ar_t, np.ascontiguousarray(c[:, :ar_t.levels]), int(run_t["args"]["window"]))
    else:
        _, codec = load_codec(a.codec, dev)
        tw = SpectralTwin.from_state(torch.load(os.path.join(a.twin, "twin.pt"), map_location="cpu"), dev)

        @torch.no_grad()
        def feats_fn(c):
            c = torch.from_numpy(np.asarray(c[:, :L], np.int64)).to(dev)
            return codec.embed(c[None])[0].float().cpu().numpy()

        @torch.no_grad()
        def twin_fn(c):
            c = torch.from_numpy(np.asarray(c[:, :L], np.int64)).to(dev)
            return sum(tw.codebooks[l][c[:, l]] for l in range(L)).float().cpu().numpy()

    def nights_of(unit):
        ud = os.path.join(a.scan, unit)
        fs = sorted(f for f in os.listdir(ud) if f.endswith(".npz")) if os.path.isdir(ud) else []
        n = 0
        for f in fs:
            if a.max_nights and n >= a.max_nights:
                break
            nd = load_eval_night(a, unit, f, feats_fn, twin_fn, load_refs)
            if nd is not None:
                n += 1
                yield nd

    # prototypes on the pool test nights
    sums = {s: {c: 0.0 for c in NAMES} for s in ("codes", "twin")}
    cnts = {c: 0 for c in NAMES}
    n_pool = 0
    for unit in POOL_UNITS:
        for nd in nights_of(unit):
            n_pool += 1
            cm = class_masks(nd["masks"], len(nd["stage"]))
            ok = np.isin(nd["stage"], SLEEP) & np.isfinite(nd["F"]["codes"]).all(1) & np.isfinite(nd["F"]["twin"]).all(1)
            for c in NAMES:
                m = cm[c] & ok
                cnts[c] += int(m.sum())
                for s in ("codes", "twin"):
                    sums[s][c] = sums[s][c] + nd["F"][s][m].sum(0)
    names = [c for c in NAMES if cnts[c] > 0]
    protos = {s: np.stack([sums[s][c] / cnts[c] for c in names]) for s in ("codes", "twin")}
    res = {"rules": {"z_event": Z_EVENT, "iou": IOU, "n_chance": N_CHANCE, "n_boot": N_BOOT, "levels": L,
                     "classes": {c: list(v) for c, v in CLASSES.items()}},
           "prototypes": {"n_pool_nights": n_pool, "n_tokens": {c: cnts[c] for c in names},
                          "cosine": {s: np.round(unit_cos(p) @ unit_cos(p).T, 3).tolist() for s, p in protos.items()},
                          "names": names},
           "sets": {}}
    ik, isp = names.index("kc") if "kc" in names else None, names.index("spindle") if "spindle" in names else None

    rng = np.random.default_rng(a.seed)
    two = {s: {"kc": [], "spindle": []} for s in ("codes", "twin")}  # per night: (hits, n) of the 2-class choice
    for name, (unit, ref_key, cls, stages) in SETS.items():
        ci = names.index(cls) if cls in names else None
        per_night, n_ev = {s: [] for s in ("codes", "twin")}, 0
        surp = {s: {"matched": 0, "own": 0, "dist": np.zeros(len(names), np.int64), "n": 0} for s in ("codes", "twin")}
        n_nights = 0
        for nd in nights_of(unit):
            if ref_key not in nd["masks"]:
                continue
            n_nights += 1
            stratum = np.isin(nd["stage"], stages)
            ref = nd["masks"][ref_key]
            ev = events(ref)
            ev = ev[stratum[ev[:, 0]]] if len(ev) else ev
            n_ev += len(ev)
            busy = np.zeros_like(ref)
            for s_, e_ in ev:
                busy[max(0, s_ - 2):e_ + 2] = True
            ch = chance_spans(ev, stratum, busy, rng, N_CHANCE)
            for s in ("codes", "twin"):
                F = nd["F"][s]
                ty, tc = type_of(F, ev, protos[s]), type_of(F, ch, protos[s])
                ok, okc = ty >= 0, tc >= 0
                per_night[s].append((int((ty[ok] == ci).sum()), int(ok.sum()), int((tc[okc] == ci).sum()), int(okc.sum()),
                                     np.bincount(ty[ok], minlength=len(names))))
                if unit == "dreams" and cls in ("kc", "spindle") and ik is not None and isp is not None:
                    t2 = type_of(F, ev, protos[s][[ik, isp]])
                    want = 0 if cls == "kc" else 1
                    two[s][cls].append((int((t2[t2 >= 0] == want).sum()), int((t2 >= 0).sum())))
                if s in nd["z"]:
                    z = nd["z"][s]
                    det = events((np.nan_to_num(z, nan=-np.inf) > Z_EVENT) & stratum)
                    if len(det):
                        td = type_of(F, det, protos[s])
                        m = matched_refs(ev, det, IOU) if len(ev) else np.zeros(len(det), bool)  # det events hitting a ref
                        surp[s]["n"] += int((td >= 0).sum())
                        surp[s]["dist"] += np.bincount(td[td >= 0], minlength=len(names))
                        surp[s]["matched"] += int((m & (td >= 0)).sum())
                        surp[s]["own"] += int((m & (td == ci)).sum())
        if not n_nights:
            continue
        o = {"n_nights": n_nights, "n_events": n_ev, "class": cls, "stratum": list(stages)}
        for s in ("codes", "twin"):
            pn = per_night[s]

            def lift(groups):
                g = groups[0]
                h, n, hc, nc = (sum(x[i] for x in g) for i in range(4))
                return h / n - hc / nc if n and nc else np.nan

            h, n, hc, nc = (sum(x[i] for x in pn) for i in range(4))
            dist = sum(x[4] for x in pn)
            o[s] = {"own_class_share": h / max(n, 1), "chance_share": hc / max(nc, 1), "lift": h / max(n, 1) - hc / max(nc, 1),
                    "ci_lift": night_boot([pn], lift, N_BOOT, a.seed),
                    "type_distribution": {c: round(float(v) / max(int(dist.sum()), 1), 4) for c, v in zip(names, dist)}}
            if surp[s]["n"]:
                o[s]["surprise_events"] = {
                    "n": surp[s]["n"], "matched": surp[s]["matched"],
                    "own_class_share_matched": surp[s]["own"] / max(surp[s]["matched"], 1),
                    "type_distribution": {c: round(float(v) / surp[s]["n"], 4) for c, v in zip(names, surp[s]["dist"])}}
        res["sets"][name] = o

    if all(two[s]["kc"] and two[s]["spindle"] for s in two):
        def bal(groups):
            r = [sum(h for h, _ in g) / max(sum(n for _, n in g), 1) for g in groups]
            return 0.5 * (r[0] + r[1])

        kv = {}
        for s in ("codes", "twin"):
            kv[s] = {"balanced_accuracy": bal([two[s]["kc"], two[s]["spindle"]]),
                     "ci": night_boot([two[s]["kc"], two[s]["spindle"]], bal, N_BOOT, a.seed)}
        pairs = [list(zip(two["codes"][c], two["twin"][c])) for c in ("kc", "spindle")]
        dl = lambda g: bal([[p[0] for p in x] for x in g]) - bal([[p[1] for p in x] for x in g])  # noqa: E731
        kv["delta"] = {"mean": kv["codes"]["balanced_accuracy"] - kv["twin"]["balanced_accuracy"],
                       "ci": night_boot(pairs, dl, N_BOOT, a.seed)}
        kv["n_nights"] = {c: len(two["codes"][c]) for c in ("kc", "spindle")}
        res["dreams_kc_vs_spindle"] = kv
    res["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("prototypes",)}, indent=None)[:600], flush=True)
    for name, o in res["sets"].items():
        print(f"[type] {name}: n={o['n_events']} own codes {o['codes']['own_class_share']:.3f} (chance "
              f"{o['codes']['chance_share']:.3f}) twin {o['twin']['own_class_share']:.3f} (chance "
              f"{o['twin']['chance_share']:.3f})", flush=True)
    if "dreams_kc_vs_spindle" in res:
        k = res["dreams_kc_vs_spindle"]
        print(f"[type] DREAMS KC vs spindle: codes {k['codes']['balanced_accuracy']:.3f} twin "
              f"{k['twin']['balanced_accuracy']:.3f} delta CI {k['delta']['ci']}", flush=True)


if __name__ == "__main__":
    main()
