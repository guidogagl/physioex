"""V0 and V4 of the revise-and-update cycle (C0 / C8): does the codec keep the concepts, and can they be edited?

V0 (codec): the reference detectors (K-complex, delta / slow-wave fraction, spindles, alpha) on the original
epoch against the decoded one, with the prefixes N+E, N+E+P and N+E+P with the P tokens permuted inside the
epoch: event recall and precision of the decoded against the original (per token, overlap).

V4 (counterfactual), on epochs with a reference K-complex (DREAMS experts, pool rule):
- **P axis, remove**: the P tokens of the event (+-1 token) are replaced by the median P of the epoch's tokens
  without slow waves, requantized with the model's RVQ, decoded; efficacy = the detector fired on the event
  in the unedited decode and does not after. Controls: (a) the same replacement at as many random tokens
  without slow waves, (b) a random direction of the same norm at the event tokens.
- **P axis, add**: the P tokens of a K-complex of another epoch of the night pasted at a random place of an
  epoch without one; efficacy = detected after, not before. Control: tokens of the donor epoch without slow
  waves pasted at the same place.
- **predicate axis (E)**: the E code of the K-complex epoch is swapped with the one of an epoch of the same
  night with slow waves over >= 20 % (the N3 predicate), P kept; efficacy = the decoded slow-wave fraction
  crosses 0.20; control: E of another epoch below 0.20. Locality: the K-complex is still detected on its
  tokens.

Edit locations come from the references, not from the lexicon word (declared): V4 measures whether the codec
lets the concept be removed / added / moved to another context, which the explanation's counterfactual needs.

``python -m physioex.train.bin.concept_counterfactual --combo-dir DIR --dump DIR --out JSON``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.explain import concept_refs as CRF

T = 60
N3_SWA = 0.20


def reflect_index(n: int, half: int) -> np.ndarray:
    """Night indices padded by reflection (as ``vq_event_codes.reflect_index``, without its imports)."""
    i = np.abs(np.arange(-half, n + half))
    return np.where(i >= n, 2 * (n - 1) - i, i)


def _tok(iv, n_samples, fs):
    return CRF.samples_to_tokens(iv, n_samples, fs)


def detect(u: np.ndarray, fs: float) -> dict:
    """Token masks of one epoch (uV): K-complex, delta (AASM slow-wave activity), spindle, alpha."""
    n = len(u)
    alpha, _ = CRF.eeg_background(u, fs)
    return {"kc": _tok(CRF.detect_k_complex(u, fs), n, fs), "delta": _tok(CRF.detect_delta_waves(u, fs), n, fs),
            "spindle": _tok(CRF.detect_spindles(u, fs), n, fs), "alpha": np.asarray(alpha[:T], bool)}


def _hit(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the event runs of ``a`` overlapping ``b`` (nan without events)."""
    r = CRF.runs(a)
    return float(np.mean([b[s:e].any() for s, e in r])) if r else np.nan


class Codec:
    def __init__(self, combo_dir, dev, mod="eeg"):
        from physioex.train.bin.pretrain_combo import load_s2

        run, self.m = load_s2(os.path.join(combo_dir, mod, "s2"), mod, dev)
        self.half = run["args"]["seq"] // 2
        self.dev = dev

    def encode(self, xp, centres, batch=128):
        import torch

        out = {k: [] for k in "NEP"}
        with torch.no_grad():
            for j in range(0, len(centres), batch):
                tb = centres[j:j + batch]
                w = torch.from_numpy(np.stack([xp[t - self.half:t + self.half + 1] for t in tb]).astype(np.float32))
                z = self.m.encode(w[:, :, None].to(self.dev))
                for k in "NEP":
                    out[k].append(z[k].float())
        return {k: torch.cat(v) for k, v in out.items()}

    def requant(self, p):
        import torch

        with torch.no_grad():
            return self.m._q("P", p)[0]

    def decode(self, n, e, p, seed=0):
        from physioex.models.vq_explain import decode_full

        return decode_full(self.m, n, e, p, seed=seed).float().cpu().numpy()


def night(codec, z_npz, scale, fs, kc_ref, rng, max_ep=150, n_ctrl=1):
    """V0 and V4 rows of one night. ``kc_ref``: (n * T,) token mask of the reference K-complex."""
    import torch

    x = z_npz["x_eeg"].astype(np.float32)
    n = len(x)
    kc_ep = kc_ref.reshape(n, T)
    has = np.flatnonzero(kc_ep.any(1))
    rest = np.setdiff1d(np.arange(n), has)
    pick = np.sort(np.r_[rng.permutation(has)[:max_ep // 2], rng.permutation(rest)[:max_ep // 2]])
    xp = x[reflect_index(n, codec.half)]
    z = codec.encode(xp, pick + codec.half)
    uv = lambda a: a * scale  # robust-scaled -> uV (bandpassed detectors: the median does not matter)
    dec = {"NEP": codec.decode(z["N"], z["E"], z["P"]), "NE": codec.decode(z["N"], z["E"], torch.zeros_like(z["P"]))}
    perm = torch.stack([zp[torch.randperm(zp.shape[0], generator=torch.Generator().manual_seed(int(i)))]
                        for i, zp in enumerate(z["P"])])
    dec["NEPperm"] = codec.decode(z["N"], z["E"], perm)
    det_o = [detect(uv(x[e]), fs) for e in pick]
    det_d = {k: [detect(uv(v[i]), fs) for i in range(len(pick))] for k, v in dec.items()}
    v0 = []
    for i, e in enumerate(pick):
        row = {"epoch": int(e), "kc_ref": bool(kc_ep[e].any())}
        for c in ("kc", "delta", "spindle", "alpha"):
            o = det_o[i][c]
            row[f"{c}_frac_orig"] = float(o.mean())
            for k in dec:
                d = det_d[k][i][c]
                row[f"{c}_recall_{k}"] = _hit(o, d)
                row[f"{c}_prec_{k}"] = _hit(d, o)
                row[f"{c}_frac_{k}"] = float(d.mean())
        v0.append(row)

    # V4 on the K-complex epochs among the picked
    base = det_d["NEP"]
    v4 = []
    idx_kc = [i for i, e in enumerate(pick) if kc_ep[e].any()]
    idx_no = [i for i, e in enumerate(pick) if not kc_ep[e].any() and not base[i]["kc"].any()]
    sw_hi = [i for i, e in enumerate(pick) if base[i]["delta"].mean() >= N3_SWA]
    sw_lo = [i for i, e in enumerate(pick) if base[i]["delta"].mean() < N3_SWA]
    P = z["P"]
    for i in idx_kc:
        e = pick[i]
        runs = CRF.runs(kc_ep[e])
        a, b = runs[0]
        ev = np.zeros(T, bool)
        ev[max(0, a - 1):min(T, b + 1)] = True
        silent = ~base[i]["delta"] & ~ev
        if silent.sum() < ev.sum() + 2:
            continue
        before = bool(base[i]["kc"][ev].any())
        med = P[i][torch.from_numpy(silent)].median(0).values
        edits = {}
        p1 = P[i].clone()
        p1[torch.from_numpy(ev)] = med
        edits["remove"] = (p1, ev)
        cs = np.zeros(T, bool)
        cs[rng.choice(np.flatnonzero(silent), int(ev.sum()), replace=False)] = True
        p2 = P[i].clone()
        p2[torch.from_numpy(cs)] = med
        edits["ctrl_silent"] = (p2, ev)
        p3 = P[i].clone()
        d = (med - P[i][torch.from_numpy(ev)])
        r = torch.randn_like(d)
        r = r / r.norm(dim=1, keepdim=True) * d.norm(dim=1, keepdim=True)
        p3[torch.from_numpy(ev)] = P[i][torch.from_numpy(ev)] + r
        edits["ctrl_random"] = (p3, ev)
        row = {"epoch": int(e), "kind": "kc", "detected_before": before, "n_tok": int(ev.sum())}
        for k, (pe, m) in edits.items():
            pq = codec.requant(pe[None])
            w = codec.decode(z["N"][i:i + 1], z["E"][i:i + 1], pq)[0]
            dd = detect(uv(w), fs)
            row[f"{k}_kc_after"] = bool(dd["kc"][m].any())
            out_tok = ~(m | cs) if k == "ctrl_silent" else ~m
            diff = (w - dec["NEP"][i]).reshape(T, -1)
            row[f"{k}_rms_in"] = float(np.sqrt((diff[m] ** 2).mean()))
            row[f"{k}_rms_out"] = float(np.sqrt((diff[out_tok] ** 2).mean()))
            row[f"{k}_ptp_in"] = float(np.ptp(CRF.bandpass(uv(w), 0.3, 4, fs).reshape(T, -1)[m]))
        row["ptp_in_before"] = float(np.ptp(CRF.bandpass(uv(dec["NEP"][i]), 0.3, 4, fs).reshape(T, -1)[ev]))
        # predicate axis: E of an N3-like epoch (decoded slow waves >= 20 %), P kept
        for k, pool in (("E_n3", sw_hi), ("E_ctrl", [j for j in sw_lo if j != i])):
            if not pool:
                continue
            j = int(rng.choice(pool))
            w = codec.decode(z["N"][i:i + 1], z["E"][j:j + 1], P[i:i + 1])[0]
            dd = detect(uv(w), fs)
            row[f"{k}_swa_after"] = float(dd["delta"].mean())
            row[f"{k}_kc_kept"] = bool(dd["kc"][ev].any())
        row["swa_before"] = float(base[i]["delta"].mean())
        v4.append(row)
    # P axis, add: a K-complex of another epoch pasted into an epoch without one
    donors = [(i, CRF.runs(kc_ep[pick[i]])[0]) for i in idx_kc]
    for i in idx_no[:len(donors)]:
        j, (a, b) = donors[int(rng.integers(len(donors)))]
        L = b - a
        if L >= T - 2:
            continue
        s = int(rng.integers(1, T - L - 1))
        m = np.zeros(T, bool)
        m[s:s + L] = True
        silent_j = np.flatnonzero(~base[j]["delta"])
        row = {"epoch": int(pick[i]), "kind": "add", "n_tok": int(L), "detected_before": bool(base[i]["kc"][m].any())}
        for k in ("add", "ctrl_add"):
            pe = P[i].clone()
            if k == "add":
                pe[s:s + L] = P[j][a:b]
            else:
                if len(silent_j) < L:
                    continue
                pe[s:s + L] = P[j][torch.from_numpy(rng.choice(silent_j, L, replace=False))]
            w = codec.decode(z["N"][i:i + 1], z["E"][i:i + 1], codec.requant(pe[None]))[0]
            dd = detect(uv(w), fs)
            row[f"{k}_kc_after"] = bool(dd["kc"][m].any())
            row[f"{k}_ptp_in"] = float(np.ptp(CRF.bandpass(uv(w), 0.3, 4, fs).reshape(T, -1)[m]))
        v4.append(row)
    return v0, v4


def summarize(v0, v4):
    from scipy.stats import wilcoxon

    out = {"V0": {}, "V4": {}}
    for c in ("kc", "delta", "spindle", "alpha"):
        for k in ("NEP", "NE", "NEPperm"):
            r = np.array([row[f"{c}_recall_{k}"] for row in v0], float)
            p = np.array([row[f"{c}_prec_{k}"] for row in v0], float)
            out["V0"][f"{c}_{k}"] = {"recall": float(np.nanmean(r)), "precision": float(np.nanmean(p)),
                                     "n_epochs_with_event": int(np.isfinite(r).sum())}
        fo = np.array([row[f"{c}_frac_orig"] for row in v0])
        fd = np.array([row[f"{c}_frac_NEP"] for row in v0])
        out["V0"][f"{c}_frac_corr"] = float(np.corrcoef(fo, fd)[0, 1]) if fo.std() > 0 and fd.std() > 0 else None
    kc = [r for r in v4 if r["kind"] == "kc" and r["detected_before"]]
    add = [r for r in v4 if r["kind"] == "add" and not r["detected_before"]]
    rate = lambda rows, key, neg=True: (float(np.mean([not r[key] if neg else r[key] for r in rows if key in r]))
                                        if any(key in r for r in rows) else None)
    out["V4"]["remove"] = {"n": len(kc), "efficacy": rate(kc, "remove_kc_after"),
                           "ctrl_silent": rate(kc, "ctrl_silent_kc_after"), "ctrl_random": rate(kc, "ctrl_random_kc_after")}
    if len(kc) > 5:
        a = np.array([r["ptp_in_before"] - r["remove_ptp_in"] for r in kc])
        c = np.array([r["ptp_in_before"] - r["ctrl_random_ptp_in"] for r in kc])
        out["V4"]["remove"]["ptp_drop_median_uv"] = float(np.median(a))
        out["V4"]["remove"]["ptp_drop_ctrl_random_uv"] = float(np.median(c))
        out["V4"]["remove"]["wilcoxon_p_vs_random"] = float(wilcoxon(a, c).pvalue)
        out["V4"]["remove"]["locality_rms_out_over_in"] = float(np.median([r["remove_rms_out"] / (r["remove_rms_in"] + 1e-9) for r in kc]))
    out["V4"]["add"] = {"n": len(add), "efficacy": rate(add, "add_kc_after", neg=False),
                        "ctrl": rate(add, "ctrl_add_kc_after", neg=False)}
    pe = [r for r in v4 if r["kind"] == "kc" and r["swa_before"] < N3_SWA]
    if pe:
        n3 = [r for r in pe if "E_n3_swa_after" in r]
        ct = [r for r in pe if "E_ctrl_swa_after" in r]
        out["V4"]["predicate_E"] = {
            "n": len(n3), "efficacy": float(np.mean([r["E_n3_swa_after"] >= N3_SWA for r in n3])) if n3 else None,
            "ctrl": float(np.mean([r["E_ctrl_swa_after"] >= N3_SWA for r in ct])) if ct else None,
            "kc_kept_n3": float(np.mean([r["E_n3_kc_kept"] for r in n3 if r["detected_before"]])) if n3 else None,
            "kc_kept_ctrl": float(np.mean([r["E_ctrl_kc_kept"] for r in ct if r["detected_before"]])) if ct else None,
            "swa_after_median": float(np.median([r["E_n3_swa_after"] for r in n3])) if n3 else None}
    return out


def main(argv=None):
    import torch

    from physioex.train.bin.concept_scan import load_refs

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--combo-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pool", type=int, default=20)
    ap.add_argument("--max-ep", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = Codec(a.combo_dir, dev)
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = float(meta["fs"]["eeg"])
    rng = np.random.default_rng(a.seed)
    keys = sorted(meta["nights"])
    dre = [k for k in keys if meta["nights"][k]["unit"] == "dreams"]
    pool = [k for k in keys if meta["nights"][k]["unit"] not in ("dreams", "mass_ss01")]
    pool = [pool[i] for i in sorted(rng.choice(len(pool), min(a.n_pool, len(pool)), replace=False))]
    res = {}
    for setname, ks, ref in (("dreams", dre, "k_complex_vis1"), ("pool", pool, "k_complex_rule")):
        V0, V4 = [], []
        for k in ks:
            z = np.load(os.path.join(a.dump, f"{k}.npz"))
            rf = load_refs(os.path.join(a.dump, "refs", f"{k}.npz"))
            s = sc.get(k, {}).get("eeg", {}).get("scale")
            if s is None or ref not in rf or "x_eeg" not in z.files:
                continue
            v0, v4 = night(codec, z, float(s), fs, np.asarray(rf[ref], bool), rng, max_ep=a.max_ep)
            V0 += v0
            V4 += v4
            print(f"[cf] {setname} {k} v0={len(v0)} v4={len(v4)} {time.time() - t0:.0f}s", flush=True)
        res[setname] = {"n_nights": len(ks), **summarize(V0, V4)} if V0 else {"n_nights": 0}
    res["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    from physioex.train.bin.concept_compose import _jsonable

    json.dump(_jsonable(res), open(a.out, "w"), indent=1)
    print(f"[cf] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
