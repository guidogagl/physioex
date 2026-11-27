"""Surprise-based event detection against experts and rules (plan of 26.11.2026, S5, xmai-lab/sleep-lens#22).

    python -m physioex.train.bin.surprise_eval --dump CONCEPT_DUMP --surprise AR_RUN/surprise \\
        --twin TWIN_RUN/surprise --out OUT.json [--n-boot 2000]

Inputs: the token surprise of two next-token models (:mod:`physioex.train.bin.eeg_ar` ``score``), one on the
codec's codes and one on their spectral twin, and the ``concept_scan`` dump of the evaluation nights
(``ts``, ``y`` of the scored epochs; ``refs/`` and ``m1/`` token masks, 60 tokens of 0.5 s per epoch). The
surprise of token ``t`` of dumped epoch ``ts_k`` is ``s[(ts_k - e0) * 60 + t]``; tokens without a score
are left out.

Rules fixed before the first result (27.11.2026):

* the surprise is z-scored within each night (mean and standard deviation of its scored tokens);
* **AP** per 0.5 s token (``average_precision_score``), pooled over the nights of a set, on the tokens of
  the reference's stratum (N2+N3 for K-complexes and spindles, as in ``concept_scan.TARGETS``; sleep N1-R
  for arousals) and, for context, on all tokens;
* **gate** (plan S5): ``AP(codes) - AP(twin) >= +0.05`` with the 95 % CI of a paired bootstrap over the
  nights above 0, on DREAMS K-complexes (``k_complex_vis1``) and spindles (``spindle_vis1``), stratum tokens;
* **events**: runs of tokens with ``z > 2`` (mu + 2 sigma of the night); F1 with IoU >= 0.2 and night
  bootstrap (:func:`physioex.train.bin.concept_refs_eval.bootstrap`), restricted to the stratum; the
  rule detectors (``refs/``) and the manual rules (``m1/``) are scored the same way at their own masks;
* **explained events**: share of the surprise events that overlap any reference or rule mask of the
  night, against the same events shifted circularly by a random offset within the night (chance);
* **recall by density**: expert events grouped by how many events of the same reference start within
  +-10 s (0, 1, >= 2), recall of the surprise events in each group.

Writes ``OUT.json``: aggregate numbers only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

T = 60  # tokens of 0.5 s per epoch
SETS = {  # name: (units, reference, alternative reference, rule masks for comparison, stratum stages)
    "dreams_kc": (("dreams",), "k_complex_vis1", "k_complex_vis2", ("refs:k_complex_rule", "m1:kc_m1"), (2, 3)),
    "dreams_spindle": (("dreams",), "spindle_vis1", "spindle_vis2", ("refs:spindle_rule",), (2, 3)),
    "mass_ss01_arousal": (("mass_ss01",), "arousal_ann", None, ("m1:arousal_m1",), (1, 2, 3, 4)),
    "pool_arousal": (("ccshs", "dcsm", "hmc", "hpap_lab_full", "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05",
                      "mesa", "mros", "sleepedf", "wsc_visit1"), "arousal_ann", None, ("m1:arousal_m1",), (1, 2, 3, 4)),
    "shhs_arousal": (("shhs_visit1",), "arousal_ann", None, ("m1:arousal_m1",), (1, 2, 3, 4)),
}
GATE_SETS, GATE_DELTA = ("dreams_kc", "dreams_spindle"), 0.05
Z_EVENT, DENSITY_S = 2.0, 10.0


def aligned_scores(surp: dict, ts: np.ndarray) -> np.ndarray:
    """Token scores of the dumped epochs ``ts`` (``(len(ts) * T,)``, NaN without a score)."""
    s, e0 = np.asarray(surp["s"], np.float64), int(surp["e0"])
    idx = (np.asarray(ts, np.int64)[:, None] - e0) * T + np.arange(T)[None]
    ok = (idx >= 0) & (idx < len(s))
    out = np.full(idx.shape, np.nan)
    out[ok] = s[idx[ok]]
    return out.reshape(-1)


def zscore(s: np.ndarray) -> np.ndarray:
    ok = np.isfinite(s)
    if ok.sum() < 2:
        return np.full_like(s, np.nan)
    return (s - s[ok].mean()) / max(s[ok].std(), 1e-9)


def ap(score: np.ndarray, ref: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if ref.sum() == 0 or ref.all():
        return float("nan")
    return float(average_precision_score(ref, score))


def paired_delta(nights, n_boot: int, seed: int = 0):
    """Pooled AP of the two scorers and of their difference, with a paired bootstrap over the nights.
    ``nights``: list of ``(score_a, score_b, ref)`` restricted to the evaluated tokens."""
    def pooled(idx):
        a = np.concatenate([nights[i][0] for i in idx])
        b = np.concatenate([nights[i][1] for i in idx])
        r = np.concatenate([nights[i][2] for i in idx])
        return ap(a, r), ap(b, r)

    a, b = pooled(range(len(nights)))
    out = {"ap_codes": a, "ap_twin": b, "delta": a - b}
    if len(nights) > 1 and n_boot:
        rng = np.random.default_rng(seed)
        bs = np.array([pooled(rng.integers(0, len(nights), len(nights))) for _ in range(n_boot)])
        d = bs[:, 0] - bs[:, 1]
        q = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]  # noqa: E731
        out.update({"ci_codes": q(bs[:, 0]), "ci_twin": q(bs[:, 1]), "ci_delta": q(d)})
    return out


def density_groups(ref_ev: np.ndarray, radius_tok: int) -> np.ndarray:
    """For each event, the number of other events of the same mask whose start lies within +-radius."""
    st = ref_ev[:, 0]
    return np.array([int((np.abs(st - a) <= radius_tok).sum()) - 1 for a in st], np.int64)


def matched_refs(det_ev: np.ndarray, ref_ev: np.ndarray, iou: float = 0.2) -> np.ndarray:
    """Boolean per reference event: some detected event reaches IoU >= iou with it (not one-to-one)."""
    out = np.zeros(len(ref_ev), bool)
    for j, (a, b) in enumerate(ref_ev):
        for c, d in det_ev:
            inter = min(b, d) - max(a, c)
            if inter > 0 and inter / (max(b, d) - min(a, c)) >= iou:
                out[j] = True
                break
    return out


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--dump", required=True, help="concept_scan dump (ts, y; refs/, m1/)")
    ap_.add_argument("--surprise", required=True, help="surprise folder of the model on the codec's codes")
    ap_.add_argument("--twin", required=True, help="surprise folder of the model on the spectral twin")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--n-boot", type=int, default=2000)
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args(argv)
    from physioex.train.bin.concept_refs_eval import IOU, bootstrap, events, night_counts, prf

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    res = {"rules": {"z_event": Z_EVENT, "iou": IOU, "gate_delta": GATE_DELTA, "density_s": DENSITY_S},
           "surprise": os.path.abspath(a.surprise), "twin": os.path.abspath(a.twin), "dump": os.path.abspath(a.dump),
           "sets": {}}
    for name, (units, ref_key, alt_key, rules, stages) in SETS.items():
        nights = []  # per night: dict of aligned arrays
        for unit in units:
            ud = os.path.join(a.dump, unit)
            if not os.path.isdir(ud):
                continue
            for f in sorted(os.listdir(ud)):
                sid = f[:-4]
                ps, pt = os.path.join(a.surprise, unit, f), os.path.join(a.twin, unit, f)
                pr = os.path.join(a.dump, "refs", unit, f)
                if not (f.endswith(".npz") and os.path.exists(ps) and os.path.exists(pt) and os.path.exists(pr)):
                    continue
                refs = np.load(pr)
                if ref_key not in refs.files:
                    continue
                d = np.load(os.path.join(ud, f))
                m1p = os.path.join(a.dump, "m1", unit, f)
                m1 = np.load(m1p) if os.path.exists(m1p) else None
                ts, y = d["ts"], np.repeat(np.asarray(d["y"]), T)
                sc, tw = aligned_scores(np.load(ps), ts), aligned_scores(np.load(pt), ts)
                ok = np.isfinite(sc) & np.isfinite(tw)
                masks = {k: refs[k].astype(bool) for k in refs.files
                         if not k.startswith("_") and not k.endswith("_epoch") and refs[k].shape == sc.shape}
                if m1 is not None:
                    masks.update({f"m1:{k}": m1[k].astype(bool) for k in m1.files
                                  if not k.startswith("_") and not k.endswith("_epoch") and m1[k].shape == sc.shape})
                rule = {}
                for r in rules:
                    src, key = r.split(":", 1)
                    mk = key if src == "refs" else f"m1:{key}"
                    if mk in masks:
                        rule[r] = masks[mk]
                nights.append({"sid": f"{unit}/{sid}", "z": zscore(np.where(ok, sc, np.nan)),
                               "zt": zscore(np.where(ok, tw, np.nan)), "ok": ok, "ref": masks[ref_key],
                               "alt": masks.get(alt_key) if alt_key else None, "rule": rule,
                               "stratum": np.isin(y, stages) & ok,
                               "union": np.any(np.stack(list(masks.values())), 0) if masks else np.zeros_like(ok)})
        if not nights:
            continue
        out = {"n_nights": len(nights), "reference": ref_key, "stratum": list(stages)}
        for scope in ("stratum", "all"):
            keep = [n["stratum"] if scope == "stratum" else n["ok"] for n in nights]
            trip = [(n["z"][k], n["zt"][k], n["ref"][k]) for n, k in zip(nights, keep)]
            o = paired_delta(trip, a.n_boot if scope == "stratum" else 0, a.seed)
            o["n_tokens"] = int(sum(k.sum() for k in keep))
            o["base_rate"] = float(sum(n["ref"][k].sum() for n, k in zip(nights, keep)) / max(o["n_tokens"], 1))
            o["ap_rules"] = {r: ap(np.concatenate([n["rule"][r][k] for n, k in zip(nights, keep)]).astype(float),
                                   np.concatenate([n["ref"][k] for n, k in zip(nights, keep)]))
                             for r in rules if all(r in n["rule"] for n in nights)}
            if alt_key and all(n["alt"] is not None for n in nights):
                o["ap_alt_reference"] = {
                    "codes": ap(np.concatenate([n["z"][k] for n, k in zip(nights, keep)]),
                                np.concatenate([n["alt"][k] for n, k in zip(nights, keep)])),
                    "twin": ap(np.concatenate([n["zt"][k] for n, k in zip(nights, keep)]),
                               np.concatenate([n["alt"][k] for n, k in zip(nights, keep)])),
                    "vis1_vs_vis2_f1": float(prf([night_counts(n["ref"], n["alt"], n["stratum"]) for n in nights])[2])}
            out[f"token_{scope}"] = o
        # events at mu + 2 sigma of the night, within the stratum
        ev = {}
        for tag, key in (("codes", "z"), ("twin", "zt")):
            det = [np.nan_to_num(n[key], nan=-np.inf) > Z_EVENT for n in nights]
            ev[tag] = bootstrap([night_counts(d, n["ref"], n["stratum"]) for d, n in zip(det, nights)], 1000, a.seed)
            # explained events: overlap with any reference or rule mask, against circular shifts
            hit, hit0, tot = 0, 0, 0
            for d, n in zip(det, nights):
                e = events(d & n["ok"])
                tot += len(e)
                u = n["union"]
                hit += sum(bool(u[s:t].any()) for s, t in e)
                sh = int(rng.integers(T, max(T + 1, len(u) - T)))
                us = np.roll(u, sh)
                hit0 += sum(bool(us[s:t].any()) for s, t in e)
            ev[tag]["explained"] = hit / max(tot, 1)
            ev[tag]["explained_shifted"] = hit0 / max(tot, 1)
            # recall of the reference events by local density
            groups = {"0": [0, 0], "1": [0, 0], ">=2": [0, 0]}
            for d, n in zip(det, nights):
                re = events(n["ref"] & n["stratum"])
                if not len(re):
                    continue
                g = density_groups(re, int(DENSITY_S * 2))
                mt = matched_refs(events(d & n["stratum"]), re, IOU)
                for gi, m in zip(g, mt):
                    k = "0" if gi == 0 else "1" if gi == 1 else ">=2"
                    groups[k][0] += int(m)
                    groups[k][1] += 1
            ev[tag]["recall_by_density"] = {k: {"recall": v[0] / v[1] if v[1] else None, "n": v[1]}
                                            for k, v in groups.items()}
        for r in rules:
            if all(r in n["rule"] for n in nights):
                ev[r] = bootstrap([night_counts(n["rule"][r], n["ref"], n["stratum"]) for n in nights], 1000, a.seed)
        out["events"] = ev
        res["sets"][name] = out
        print(f"[surprise_eval] {name}: {len(nights)} nights, AP codes {out['token_stratum']['ap_codes']:.3f} "
              f"twin {out['token_stratum']['ap_twin']:.3f} (base {out['token_stratum']['base_rate']:.3f}), "
              f"F1 codes {ev['codes']['f1']:.3f} twin {ev['twin']['f1']:.3f}", flush=True)
    gate = {}
    for g in GATE_SETS:
        o = res["sets"].get(g, {}).get("token_stratum")
        if o and "ci_delta" in o:
            gate[g] = {"delta": o["delta"], "ci_delta": o["ci_delta"],
                       "pass": bool(o["delta"] >= GATE_DELTA and o["ci_delta"][0] > 0)}
    res["gate"] = {"sets": gate, "pass": bool(gate) and all(v["pass"] for v in gate.values()) and len(gate) == len(GATE_SETS)}
    res["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[surprise_eval] gate {res['gate']} -> {a.out}", flush=True)
    print("[surprise_eval] DONE", flush=True)


if __name__ == "__main__":
    main()
