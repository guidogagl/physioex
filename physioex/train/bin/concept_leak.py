"""D3 of the critical evaluation: is each factor confined to one level of the codec (N night, E epoch, P token)?
A counterfactual needs factors that move one at a time: the subject / site in N, the stage context in E, the
events in P. Two tables.
**Probes** (one row per epoch, pool nights): linear readers from N, E, P (mean and sd of the tokens), their pairs
and the three together, towards the site (balanced accuracy, nights held out), the night inside its site (first
half of every night trains, the second half tests: the subject signature), the stage (kappa, leave one site out)
and the epoch fraction of the reference events (Spearman, leave one site out). The epoch spectrum is the
label-free reference row.
**Swaps** (``--combo-dir``): on sampled epochs the codec decodes the original latents and the same with one
level taken from a donor: N from an epoch of another night, E from an epoch of the same night in another stage
(and in the same stage as control), P from an epoch of the same night in another stage, P permuted inside the
epoch. On the decoded signal the reference detectors give the slow-wave, spindle and alpha fractions, the
K-complex presence and the log RMS; ``transfer`` = (feature after - before) / (donor - before), clipped to
[-1, 2]: 1 = the swapped level carries that feature entirely, 0 = not at all.
``python -m physioex.train.bin.concept_leak --dump DIR --out JSON [--combo-dir DIR --n-nights 12 --n-ep 40]``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.train.bin import concept_rationale as RA

T = RA.T
EVENTS = ("k_complex_rule", "spindle_rule", "arousal_ann", "delta")
LEVELS = {"N": ["N"], "E": ["E"], "P": ["P"], "N+E": ["N", "E"], "E+P": ["E", "P"], "N+P": ["N", "P"],
          "N+E+P": ["N", "E", "P"], "spec_ep": ["spec"]}


# ---------------------------------------------------------------------------- probes
def epoch_views(D):
    """Per-epoch arrays: ``N`` / ``E`` (the epoch latents as loaded, PCA-rotated by ``finish``), ``P`` (mean and
    sd of the 60 tokens), ``spec`` (epoch spectrum), ``y``, ``night``, ``gname``, ``set`` and the event fractions."""
    P = D["P"].reshape(-1, T, D["P"].shape[1])
    V = {"P": np.hstack([P.mean(1), P.std(1)]).astype(np.float32), "spec": D["spec_ep"][::T],
         "y": D["y"][::T], "night": D["night"][::T], "gname": D["gname"][::T], "set": D["set"][::T]}
    for s, k in (("N", "Nq"), ("E", "Eq")):
        if k in D:
            V[s] = D[k][::T]
    V["frac"] = {}
    for ev in EVENTS:
        if ev in D["refs"]:
            r, h = D["refs"][ev]
            V["frac"][ev] = (r.reshape(-1, T).mean(1), h.reshape(-1, T).all(1))
    return V


def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _multiclass(X, yc, tr, te, seed=0, max_fit=60_000):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    sub = tr[rng.choice(len(tr), min(max_fit, len(tr)), replace=False)]
    a, b = _std(X[sub], X[te])
    m = LogisticRegression(max_iter=500, C=0.5).fit(a, yc[sub])
    return m.predict(b)


def _kappa(a, b):
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b)) if len(np.unique(a)) > 1 and len(np.unique(b)) > 1 else np.nan


def _balanced_acc(a, b):
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(a, b))


def probes(V, seed=0, n_folds=5):
    """Level x target table on the pool epochs."""
    from scipy.stats import spearmanr

    pool = np.flatnonzero(V["set"] == "pool")
    y, night, gname = V["y"][pool], V["night"][pool], V["gname"][pool]
    sites = sorted(set(gname.tolist()))
    rng = np.random.default_rng(seed)
    nights = np.unique(night)
    fold_of = dict(zip(nights, rng.permutation(len(nights)) % n_folds))
    nf = np.array([fold_of[n] for n in night])
    half = np.zeros(len(pool), bool)  # first half of every night
    for n in nights:
        ix = np.flatnonzero(night == n)
        half[ix[: len(ix) // 2]] = True
    n_in_site = [len(np.unique(night[gname == g])) for g in sites]
    out = {"n_epochs": int(len(pool)), "n_nights": int(len(nights)), "sites": sites,
           "chance": {"site_balanced_acc": 1 / len(sites), "night_in_site_acc": float(np.mean([1 / k for k in n_in_site])),
                      "stage_kappa": 0.0}, "levels": {}}
    for lev, parts in LEVELS.items():
        if any(p not in V for p in parts):
            continue
        X = np.hstack([V[p][pool] for p in parts]).astype(np.float32)
        fin = np.isfinite(X).all(1)
        row = {"dim": int(X.shape[1])}
        # site: nights held out (5 folds)
        pred = np.full(len(pool), "", object)
        for f in range(n_folds):
            tr, te = np.flatnonzero((nf != f) & fin), np.flatnonzero((nf == f) & fin)
            if len(te) and len(np.unique(gname[tr])) > 1:
                pred[te] = _multiclass(X, gname, tr, te, seed)
        ok = pred != ""
        row["site_balanced_acc"] = _balanced_acc(gname[ok], pred[ok].astype(str)) if ok.any() else None
        # night inside its site: first half trains, second half tests
        acc = []
        for g in sites:
            ig = np.flatnonzero((gname == g) & fin)
            if len(np.unique(night[ig])) < 2:
                continue
            tr, te = ig[half[ig]], ig[~half[ig]]
            if len(tr) and len(te):
                p = _multiclass(X, night.astype(str), tr, te, seed)
                acc.append(float(np.mean(p == night[te].astype(str))))
        row["night_in_site_acc"] = float(np.mean(acc)) if acc else None
        # stage: leave one site out
        pred = np.full(len(pool), -1)
        for g in sites:
            tr, te = np.flatnonzero((gname != g) & fin), np.flatnonzero((gname == g) & fin)
            if len(te) and len(np.unique(y[tr])) > 1:
                pred[te] = _multiclass(X, y, tr, te, seed)
        ok = pred >= 0
        row["stage_kappa"] = _kappa(y[ok], pred[ok]) if ok.any() else None
        # events: ridge on the epoch fraction, leave one site out, Spearman
        for ev, (fr, h) in V["frac"].items():
            t = fr[pool]
            hv = h[pool] & fin
            s = RA.fold_probe(X[hv], [], t[hv], gname[hv], sites, max_fit=100_000, seed=seed, ridge=True)
            ok = np.isfinite(s)
            row[f"event_{ev}_spearman"] = float(spearmanr(s[ok], t[hv][ok])[0]) if ok.sum() > 10 and np.ptp(t[hv][ok]) > 0 else None
        out["levels"][lev] = row
        print(f"[leak] probes {lev}: {row}", flush=True)
    return out


# ---------------------------------------------------------------------------- swaps
def features(u, fs):
    from physioex.train.bin.concept_counterfactual import detect

    d = detect(u, fs)
    return {"swa": float(d["delta"].mean()), "spindle": float(d["spindle"].mean()), "alpha": float(d["alpha"].mean()),
            "kc": float(d["kc"].any()), "log_rms": float(np.log(np.sqrt((u ** 2).mean()) + 1e-6))}


def _transfer(before, after, donor):
    den = donor - before
    if abs(den) < 1e-6:
        return None
    return float(np.clip((after - before) / den, -1, 2))


def swaps(codec, dump, meta, scale, keys, fs, n_ep=40, seed=0):
    """Swap rows over the sampled nights: one row per (epoch, swap) with the features before, after and of the
    donor."""
    import torch

    from physioex.train.bin.concept_counterfactual import reflect_index

    rng = np.random.default_rng(seed)
    rows = []
    Z = {}
    for k in keys:
        z = np.load(os.path.join(dump, f"{k}.npz"))
        x = z["x_eeg"].astype(np.float32)
        y = z["y"]
        n = len(x)
        pick = np.sort(rng.choice(n, min(n_ep, n), replace=False))
        lat = codec.encode(x[reflect_index(n, codec.half)], pick + codec.half)
        Z[k] = {"pick": pick, "y": y[pick], "lat": lat, "scale": float(scale[k]["eeg"]["scale"]),
                "site": meta["nights"][k]["unit"]}
    for k in keys:
        zk = Z[k]
        lat, y, sc = zk["lat"], zk["y"], zk["scale"]
        base = codec.decode(lat["N"], lat["E"], lat["P"])
        fb = [features(base[i] * sc, fs) for i in range(len(y))]
        others = [o for o in keys if o != k and Z[o]["site"] != zk["site"]] or [o for o in keys if o != k]
        for i in range(len(y)):
            cands = {}
            o = others[int(rng.integers(len(others)))]
            cands["N_other_night"] = (o, int(rng.integers(len(Z[o]["y"]))), "N")
            diff = np.flatnonzero(y != y[i])
            same = np.flatnonzero((y == y[i]) & (np.arange(len(y)) != i))
            if len(diff):
                j = int(rng.choice(diff))
                cands["E_other_stage"] = (k, j, "E")
                cands["P_other_stage"] = (k, j, "P")
            if len(same):
                cands["E_same_stage"] = (k, int(rng.choice(same)), "E")
            for name, (dk, j, lev) in cands.items():
                dl = Z[dk]["lat"]
                n_, e_, p_ = lat["N"][i:i + 1], lat["E"][i:i + 1], lat["P"][i:i + 1]
                if lev == "N":
                    n_ = dl["N"][j:j + 1]
                elif lev == "E":
                    e_ = dl["E"][j:j + 1]
                else:
                    p_ = dl["P"][j:j + 1]
                w = codec.decode(n_, e_, p_)[0]
                fa = features(w * sc, fs)
                donor_dec = codec.decode(dl["N"][j:j + 1], dl["E"][j:j + 1], dl["P"][j:j + 1])[0]
                fd = features(donor_dec * Z[dk]["scale"], fs)
                rows.append({"night": k, "epoch": int(zk["pick"][i]), "stage": int(y[i]), "swap": name,
                             "donor_stage": int(Z[dk]["y"][j]), "before": fb[i], "after": fa, "donor": fd})
            perm = lat["P"][i][torch.randperm(T, generator=torch.Generator().manual_seed(int(i)))][None]
            w = codec.decode(lat["N"][i:i + 1], lat["E"][i:i + 1], perm)[0]
            rows.append({"night": k, "epoch": int(zk["pick"][i]), "stage": int(y[i]), "swap": "P_perm",
                         "donor_stage": int(y[i]), "before": fb[i], "after": features(w * sc, fs), "donor": fb[i]})
        print(f"[leak] swaps {k}: {len(rows)} rows", flush=True)
    return rows


def summarize_swaps(rows):
    """Per swap and feature: median |after - before|, median |donor - before| (the natural distance) and the
    median transfer where the donor differs."""
    out = {}
    names = sorted({r["swap"] for r in rows})
    feats = ("swa", "spindle", "alpha", "kc", "log_rms")
    for s in names:
        rs = [r for r in rows if r["swap"] == s]
        out[s] = {"n": len(rs)}
        for f in feats:
            d_ab = np.array([abs(r["after"][f] - r["before"][f]) for r in rs])
            d_db = np.array([abs(r["donor"][f] - r["before"][f]) for r in rs])
            tr = [t for t in (_transfer(r["before"][f], r["after"][f], r["donor"][f]) for r in rs) if t is not None]
            out[s][f] = {"moved": float(np.median(d_ab)), "natural": float(np.median(d_db)),
                         "transfer": float(np.median(tr)) if tr else None, "n_transfer": len(tr)}
    return out


# ---------------------------------------------------------------------------- command
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--combo-dir", default="")
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--n-nights", type=int, default=12)
    ap.add_argument("--n-ep", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    RA.PCA_DIM = 128  # keep the epoch latents whole (the PCA of ``finish`` is then a rotation)
    D, keys, sets = RA.load(a.dump, "eeg", list(EVENTS), frac=a.frac, n_ss01=0, seed=a.seed)
    print(f"[leak] loaded {len(keys)} nights, {len(D['y'])} tokens, {time.time() - t0:.0f} s", flush=True)
    out = {"frac": a.frac, "nights": {s: sum(v == s for v in sets.values()) for s in sorted(set(sets.values()))},
           "probes": probes(epoch_views(D), seed=a.seed)}
    if a.combo_dir:
        import torch

        from physioex.train.bin.concept_counterfactual import Codec

        meta = json.load(open(os.path.join(a.dump, "meta.json")))
        scale = json.load(open(os.path.join(a.dump, "scale.json")))
        rng = np.random.default_rng(a.seed)
        pool = [k for k in keys if sets[k] == "pool" and scale.get(k, {}).get("eeg", {}).get("scale")]
        pick = [pool[i] for i in sorted(rng.choice(len(pool), min(a.n_nights, len(pool)), replace=False))]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        codec = Codec(a.combo_dir, dev)
        rows = swaps(codec, a.dump, meta, scale, pick, float(meta["fs"]["eeg"]), n_ep=a.n_ep, seed=a.seed)
        out["swaps"] = summarize_swaps(rows)
        out["swap_nights"] = pick
        out["swap_rows"] = rows
    out["minutes"] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"[leak] wrote {a.out} ({out['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
