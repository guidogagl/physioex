"""Road C of the critical evaluation (09.11.2026): explanations of the stager's decision via the decoded signal,
without a semantic latent. For every sampled epoch the rule detectors on the original signal say which event
families are present (K-complex rule, spindle rule, slow waves, alpha); for every family present the tokens of
its events (+-1 token) are replaced in P by the median P of the silent tokens of the epoch (its neighbours when
too few), requantized with the model's RVQ, and the EEG stager reads the window again. Reported per epoch and
family: the drop of p(predicted stage), the flip rate, the same for two controls (the same replacement on random
silent tokens; a random direction of the same norm on the event tokens), and on the decoded epoch whether the
family's detector still fires (efficacy of the removal). The aggregate per predicted stage and family is the
explanation card that road C can offer: "p(N2) rests on the spindles by x and on the K-complexes by y".
The stager is replayed from the dump latents (``Nq`` / ``Eq`` / ``P``) and checked against ``logp_eeg`` of
the dump; the decoded checks use the epochs re-encoded by the codec.
``python -m physioex.train.bin.concept_explain_decoded --dump DIR --combo-dir DIR --out JSON [--n-nights 24 --max-ep 60]``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

T = 60
FAMILIES = {"kc": "k_complex_rule", "spindle": "spindle_rule", "swa": "delta", "alpha": "alpha"}
DET = {"kc": "kc", "spindle": "spindle", "swa": "delta", "alpha": "alpha"}
STAGES = ("W", "N1", "N2", "N3", "R")


def dilate(m, k=1):
    out = m.copy()
    for s in range(1, k + 1):
        out[s:] |= m[:-s]
        out[:-s] |= m[s:]
    return out


def window(n, e, w):
    lo, hi = max(0, e - w), min(n, e + w + 1)
    return lo, hi


def stage_probs(stagers, lat, lo, hi, dev):
    """p(stage) of the epochs ``lo:hi`` read as a short night."""
    from physioex.train.bin.concept_scan import _stage_logp

    sub = {k: v[lo:hi] for k, v in lat.items() if k in ("N", "E", "P", "y")}
    sub["night"] = np.zeros(hi - lo, np.int64)
    sub["epoch"] = np.arange(lo, hi)
    return np.exp(_stage_logp(stagers, {"eeg": sub}, ["eeg"], dev)["eeg"].astype(np.float32))


def epoch_edits(P_ctx, ev_ctx, silent_ctx, rng):
    from physioex.train.bin.concept_examples import edit_set

    return edit_set(P_ctx, ev_ctx, silent_ctx, rng)


def explain_night(key, z, q, refs, scale, fs, stagers, codec, dev, rng, max_ep=60, w=None, families=FAMILIES):
    """Rows of one night: (epoch, family, edit) with p before / after and the decoded check."""
    import torch

    from physioex.train.bin.concept_counterfactual import detect, reflect_index

    n = len(z["y"])
    lat = {"N": q["Nq_eeg"].astype(np.float32), "E": q["Eq_eeg"].astype(np.float32),
           "P": z["P_eeg"].astype(np.float32), "y": z["y"].astype(np.int64)}
    masks = {f: np.asarray(refs[r], bool).reshape(n, T) for f, r in families.items() if r in refs and len(refs[r]) == n * T}
    if not masks:
        return [], {}
    any_ev = np.zeros((n, T), bool)
    for m in masks.values():
        any_ev |= m
    has = np.flatnonzero(np.stack([m.any(1) for m in masks.values()], 1).any(1) & (np.arange(n) >= 1) & (np.arange(n) < n - 1))
    pick = np.sort(rng.choice(has, min(max_ep, len(has)), replace=False)) if len(has) else []
    w = w or max(int(getattr(stagers[0]["eeg"], "seq_len", 1)), 10)
    # replay check against the dump
    p_full = stage_probs(stagers, lat, 0, n, dev)
    replay = float(np.abs(p_full - np.exp(z["logp_eeg"].astype(np.float32))).max()) if "logp_eeg" in z.files else None
    x = z["x_eeg"].astype(np.float32)
    xp = x[reflect_index(n, codec.half)]
    enc = codec.encode(xp, np.asarray(pick) + codec.half) if len(pick) else None
    rows = []
    for i, e in enumerate(pick):
        lo, hi = window(n, e, w)
        p0 = stage_probs(stagers, lat, lo, hi, dev)[e - lo]
        pred = int(p0.argmax())
        P_ctx = torch.from_numpy(lat["P"][e - 1:e + 2].copy())  # (3, T, d)
        silent_ctx = ~any_ev[e - 1:e + 2].reshape(-1)
        base_dec = codec.decode(enc["N"][i:i + 1], enc["E"][i:i + 1], enc["P"][i:i + 1])[0] * scale
        det0 = detect(base_dec, fs)
        for fam, m in masks.items():
            if not m[e].any():
                continue
            ev_ctx = np.zeros((3, T), bool)
            ev_ctx[1] = dilate(m[e], 1)
            ev_ctx = ev_ctx.reshape(-1)
            edits = epoch_edits(P_ctx, ev_ctx, silent_ctx, rng)
            if edits is None:
                continue
            row = {"night": key, "epoch": int(e), "stage": int(z["y"][e]), "pred": pred, "p0": p0.round(4).tolist(),
                   "family": fam, "n_tokens": int(m[e].sum()), "frac_before": float(det0[DET[fam]].mean()), "edits": {}}
            for name, Pn in edits.items():
                Pq = codec.requant(Pn)
                lat2 = {**lat, "P": lat["P"].copy()}
                lat2["P"][e - 1:e + 2] = Pq.numpy()
                p1 = stage_probs(stagers, lat2, lo, hi, dev)[e - lo]
                r = {"p1": p1.round(4).tolist(), "dp_pred": float(p0[pred] - p1[pred]), "flip": bool(p1.argmax() != pred)}
                if name == "remove":
                    dec = codec.decode(enc["N"][i:i + 1], enc["E"][i:i + 1], Pq[1:2])[0] * scale
                    r["frac_after"] = float(detect(dec, fs)[DET[fam]].mean())
                row["edits"][name] = r
            rows.append(row)
    return rows, {"replay_max_abs_dp": replay, "n_epochs": int(len(pick)), "window": int(w)}


def summarize(rows):
    """Per predicted stage and family: mean drop of p(pred) for the removal and the controls, flip rates,
    removal efficacy on the decoded (fraction after / before), counts."""
    out = {}
    fams = sorted({r["family"] for r in rows})
    for st in range(5):
        for fam in fams:
            rs = [r for r in rows if r["pred"] == st and r["family"] == fam and "remove" in r["edits"]]
            if len(rs) < 5:
                continue
            cell = {"n": len(rs)}
            for name in ("remove", "ctrl_silent", "ctrl_random"):
                v = [r["edits"][name]["dp_pred"] for r in rs if name in r["edits"]]
                f = [r["edits"][name]["flip"] for r in rs if name in r["edits"]]
                if v:
                    cell[name] = {"dp_mean": float(np.mean(v)), "dp_q50": float(np.median(v)), "flip": float(np.mean(f)), "n": len(v)}
            fb = np.array([r["frac_before"] for r in rs])
            fa = np.array([r["edits"]["remove"]["frac_after"] for r in rs])
            ok = fb > 0
            cell["efficacy"] = {"frac_before": float(fb.mean()), "frac_after": float(fa.mean()),
                                "removed": float(np.mean(fa[ok] / fb[ok] <= 0.5)) if ok.any() else None}
            cell["specific"] = float(cell["remove"]["dp_mean"] - cell.get("ctrl_silent", {}).get("dp_mean", np.nan))
            out.setdefault(STAGES[st], {})[fam] = cell
    return out


def main(argv=None):
    import torch

    from physioex.train.bin.concept_counterfactual import Codec
    from physioex.train.bin.concept_scan import load_refs
    from physioex.train.bin.pretrain_combo import load_stagers

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--combo-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-nights", type=int, default=24)
    ap.add_argument("--max-ep", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    scale = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = float(meta["fs"]["eeg"])
    stagers = load_stagers(os.path.join(a.combo_dir, meta["stager"]), ["eeg"], dev)
    codec = Codec(a.combo_dir, dev)
    rng = np.random.default_rng(a.seed)
    pool = sorted(k for k, e in meta["nights"].items() if e["unit"] not in ("dreams", "mass_ss01")
                  and scale.get(k, {}).get("eeg", {}).get("scale") and os.path.exists(os.path.join(a.dump, "eq", f"{k}.npz")))
    by_unit = {}
    for k in pool:
        by_unit.setdefault(meta["nights"][k]["unit"], []).append(k)
    per = max(1, a.n_nights // len(by_unit))
    keys = sorted(k for ks in by_unit.values() for k in rng.choice(ks, min(per, len(ks)), replace=False))
    rows, info = [], {}
    for k in keys:
        z = np.load(os.path.join(a.dump, f"{k}.npz"))
        q = np.load(os.path.join(a.dump, "eq", f"{k}.npz"))
        refs = load_refs(os.path.join(a.dump, "refs", f"{k}.npz"))
        r, inf = explain_night(k, z, q, refs, float(scale[k]["eeg"]["scale"]), fs, stagers, codec, dev, rng, a.max_ep)
        rows += r
        info[k] = inf
        print(f"[explain] {k}: {len(r)} rows, replay {inf.get('replay_max_abs_dp')}, {time.time() - t0:.0f} s", flush=True)
    out = {"nights": keys, "info": info, "summary": summarize(rows), "rows": rows, "minutes": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"[explain] wrote {a.out} ({out['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
