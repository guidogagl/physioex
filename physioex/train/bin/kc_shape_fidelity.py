"""Does the codec keep the SHAPE of a true K-complex, or only its delta energy? (C10, V0 on the experts)

Events, on the DREAMS K-complex excerpts (the experts marked every KC there):
- ``kc``: K-complexes of scorer 1 (``vis1``, type ``k_complex``), tagged ``cons`` when scorer 2 overlaps;
- ``sw``: AASM slow waves (0.5-2 Hz half-wave pair, >= 75 uV p-p) of the same excerpts more than 1.5 s
  away from any expert K-complex;
- optional ``sw_pool``: slow waves of N3 epochs of pool nights (another site: fidelity only, no classifier).

Each night is encoded and decoded whole (N+E+P; controls N+E with P zeroed and N+E+P with the P tokens
permuted inside the epoch). Around the negative peak of the original (+-1.5 s) both signals give:
- fidelity: Pearson r in 0.3-30 Hz and in 0.3-4 Hz, p-p ratio decoded / original, shift of the negative peak;
- shape features (0.3-4 Hz for the slow component, 0.3-8 Hz for slopes): p-p, negative and positive peak,
  log |neg| / pos, steepest down- and up-slope, negative half-wave duration, negative-to-positive delay,
  sharpness = down-slope / p-p, prominence = p-p / median p-p of 1.5 s windows in the +-10 s around.

Decisive test: a logistic classifier ``kc`` vs ``sw`` trained on the ORIGINAL features (leave one excerpt
out) and applied to the DECODED features of the held-out excerpt. If the AUC holds, the codec keeps what
tells a K-complex from a slow wave; if it drops to the N+E level, the codec keeps energy, not shape. Run
with all features, with shape only (no absolute amplitudes), and on amplitude-matched pairs.

While running, writes per-excerpt PNGs (original / decoded overlay) and ``progress.json`` in ``--out``.

``python -m physioex.train.bin.kc_shape_fidelity --combo-dir DIR --dump DIR --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from physioex.explain import concept_refs as CRF
from physioex.train.bin.concept_counterfactual import Codec, reflect_index

EP = 30.0
W_S = 1.5  # analysis half-window (s)
SHOW_S = 3.0  # plot half-window (s)
FULL = ["ptp", "neg", "pos", "log_np", "down", "up", "neg_dur", "n2p", "sharp", "prom"]
SHAPE = ["log_np", "neg_dur", "n2p", "sharp", "up_rel", "prom"]
STAGES = ["W", "N1", "N2", "N3", "R"]


def feats(low, mid, c, fs):
    """Shape features around sample ``c`` (negative peak of the original) of one signal; None if out of range."""
    w = int(W_S * fs)
    if c - int(10 * fs) < 0 or c + int(10 * fs) > len(low):
        return None
    lo = low[c - w:c + w]
    k0 = int(0.5 * fs)
    neg_i = c - w + (w - k0) + int(np.argmin(low[c - k0:c + k0]))
    neg = float(low[neg_i])
    pos_seg = low[neg_i:neg_i + int(fs)]
    pos_i = neg_i + int(np.argmax(pos_seg))
    pos = float(low[pos_i])
    ptp = pos - neg
    d = np.diff(mid) * fs
    down = float(-d[max(0, neg_i - k0):neg_i].min()) if neg_i > 0 else 0.0
    up = float(d[neg_i:max(pos_i, neg_i + 1)].max())
    s = np.sign(low)
    a = neg_i
    while a > 0 and s[a - 1] < 0:
        a -= 1
    b = neg_i
    while b < len(low) - 1 and s[b + 1] < 0:
        b += 1
    bg = []
    for t0 in range(c - int(10 * fs), c + int(10 * fs) - w, w):
        if abs(t0 + w // 2 - c) > w:
            bg.append(np.ptp(low[t0:t0 + w]))
    prom = ptp / (np.median(bg) + 1e-6)
    return {"ptp": ptp, "neg": neg, "pos": pos, "log_np": float(np.log((abs(neg) + 1) / (abs(pos) + 1))),
            "down": down, "up": up, "neg_dur": (b - a) / fs, "n2p": (pos_i - neg_i) / fs,
            "sharp": down / (ptp + 1e-6), "up_rel": up / (ptp + 1e-6), "prom": float(prom), "neg_i": int(neg_i),
            "lo_ptp": float(np.ptp(lo))}


def fidelity(o_low, o_br, d_low, d_br, c, fs, fo, fd):
    w = int(W_S * fs)
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return {"r_broad": r(o_br[c - w:c + w], d_br[c - w:c + w]), "r_low": r(o_low[c - w:c + w], d_low[c - w:c + w]),
            "ptp_ratio": fd["lo_ptp"] / (fo["lo_ptp"] + 1e-6), "lat_ms": 1000 * (fd["neg_i"] - fo["neg_i"]) / fs}


def plot(path, o_br, dec_br, c, fs, title, marks=()):
    """``marks``: (start s, end s, colour, label) relative to the negative peak, e.g. the expert intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = int(SHOW_S * fs)
    t = (np.arange(c - h, c + h) - c) / fs
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=100)
    ax.axvspan(-W_S, W_S, color="#eceff1", lw=0)
    for k, (m0, m1, col, lab) in enumerate(marks):
        y0, y1 = (0.0, 0.5) if k else (0.5, 1.0)
        ax.axvspan(m0, m1, ymin=y0 if len(marks) > 1 else 0, ymax=y1 if len(marks) > 1 else 1, color=col, alpha=0.22, lw=0, label=lab)
    ax.plot(t, dec_br["NE"][c - h:c + h], color="#8c959f", lw=0.8, ls="--", label="N+E (P zeroed)")
    ax.plot(t, o_br[c - h:c + h], color="#1f2328", lw=1.1, label="original")
    ax.plot(t, dec_br["NEP"][c - h:c + h], color="#e8590c", lw=1.1, label="decoded N+E+P")
    ax.axhline(0, color="#d0d7de", lw=0.6)
    ax.set_xlim(-SHOW_S, SHOW_S)
    ax.set_xlabel("s from the negative peak of the original (0.3-30 Hz); shaded = analysis window")
    ax.set_ylabel("µV (positive up)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decode_night(codec, x, idx=None):
    import torch

    n = len(x)
    idx = np.arange(n) if idx is None else idx
    xp = x[reflect_index(n, codec.half)]
    z = codec.encode(xp, idx + codec.half)
    perm = torch.stack([zp[torch.randperm(zp.shape[0], generator=torch.Generator().manual_seed(int(i)))]
                        for i, zp in enumerate(z["P"])])
    return {"NEP": codec.decode(z["N"], z["E"], z["P"]), "NE": codec.decode(z["N"], z["E"], torch.zeros_like(z["P"])),
            "NEPperm": codec.decode(z["N"], z["E"], perm)}


def events_night(meta_n, o_low, n_samp, fs):
    kc = [e for e in meta_n["events"] if e.get("type") == "k_complex" and e["concept"] in ("vis1", "vis2")]
    v1 = [(e["onset"], e["onset"] + e["dur"]) for e in kc if e["concept"] == "vis1"]
    v2 = [(e["onset"], e["onset"] + e["dur"]) for e in kc if e["concept"] == "vis2"]
    ov = lambda a, b, lst, pad=0.0: any(a - pad < y and x < b + pad for x, y in lst)
    out = []
    base = v1 if v1 else v2
    for a, b in base:
        i0, i1 = int(a * fs), int(b * fs)
        if i1 <= i0 or i1 >= n_samp:
            continue
        c = i0 + int(np.argmin(o_low[i0:i1]))
        out.append({"kind": "kc", "t": c / fs, "c": c, "cons": bool(v1 and ov(a, b, v2)), "scorer": "vis1" if v1 else "vis2",
                    "iv": [a, b], "iv2": [[x, y] for x, y in v2 if v1 and a < y and x < b]})
    allkc = v1 + v2
    for a, b in CRF.detect_delta_waves(o_low_raw_cache[0], fs):
        ta, tb = a / fs, b / fs
        if ov(ta, tb, allkc, pad=1.5):
            continue
        c = a + int(np.argmin(o_low[a:b]))
        out.append({"kind": "sw", "t": c / fs, "c": c, "cons": False, "scorer": "-"})
    return out


o_low_raw_cache = [None]  # raw uV night for the slow-wave detector (set per night)


def auc_lono(X, y, g, Xt_list):
    """Leave-one-group-out logistic: out-of-fold AUC on X and on each matrix of Xt_list (same rows)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pr = np.full(len(y), np.nan)
    pt = [np.full(len(y), np.nan) for _ in Xt_list]
    for gg in np.unique(g):
        te, tr = g == gg, g != gg
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        m.fit(X[tr], y[tr])
        pr[te] = m.predict_proba(X[te])[:, 1]
        for k, Xt in enumerate(Xt_list):
            pt[k][te] = m.predict_proba(Xt[te])[:, 1]
    ok = np.isfinite(pr)
    a = lambda p: float(roc_auc_score(y[ok], p[ok])) if len(np.unique(y[ok])) == 2 else None
    return a(pr), [a(p) for p in pt]


def match_amp(rows, rng):
    """KC / slow-wave pairs with original p-p within 20 % (greedy, nearest first)."""
    kc = [r for r in rows if r["kind"] == "kc"]
    sw = [r for r in rows if r["kind"] == "sw"]
    used, out = set(), []
    for r in rng.permutation(len(kc)):
        k = kc[r]
        best, bd = None, 0.2
        for j, s in enumerate(sw):
            if j in used:
                continue
            d = abs(np.log(s["orig"]["ptp"] / max(k["orig"]["ptp"], 1e-6)))
            if d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            out += [k, sw[best]]
    return out


def classify(rows, rng):
    res = {}
    for subset_name, rs in (("all", rows), ("amp_matched", match_amp(rows, rng))):
        rs = [r for r in rs if r["kind"] in ("kc", "sw")]
        if len({r["kind"] for r in rs}) < 2:
            continue
        y = np.array([r["kind"] == "kc" for r in rs], int)
        g = np.array([r["night"] for r in rs])
        out = {"n_kc": int(y.sum()), "n_sw": int((1 - y).sum())}
        for fname, F in (("full", FULL), ("shape", SHAPE), ("ptp_only", ["ptp"])):
            M = lambda key: np.array([[r[key][f] for f in F] for r in rs], float)
            auc_o, (auc_nep, auc_ne, auc_perm) = auc_lono(M("orig"), y, g, [M("NEP"), M("NE"), M("NEPperm")])
            auc_dd, _ = auc_lono(M("NEP"), y, g, [])
            out[fname] = {"orig_to_orig": auc_o, "orig_to_NEP": auc_nep, "orig_to_NE": auc_ne,
                          "orig_to_NEPperm": auc_perm, "NEP_to_NEP": auc_dd}
        res[subset_name] = out
    return res


def summarize(rows):
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        for tag, sel in (("", lambda r: True), ("_cons", lambda r: r.get("cons")), ("_N2", lambda r: r["stage"] == "N2"),
                         ("_N3", lambda r: r["stage"] == "N3")):
            rs = [r for r in rows if r["kind"] == kind and sel(r)]
            if not rs:
                continue
            d = {"n": len(rs)}
            for k in ("NEP", "NE", "NEPperm"):
                for m in ("r_broad", "r_low", "ptp_ratio", "lat_ms"):
                    v = np.array([r["fid"][k][m] for r in rs], float)
                    d[f"{k}_{m}_median"] = float(np.nanmedian(v))
                d[f"{k}_abs_lat_ms_median"] = float(np.nanmedian([abs(r["fid"][k]["lat_ms"]) for r in rs]))
            for f in SHAPE + ["ptp"]:
                o = np.array([r["orig"][f] for r in rs])
                dd = np.array([r["NEP"][f] for r in rs])
                d[f"shape_{f}_orig_median"] = float(np.median(o))
                d[f"shape_{f}_NEP_median"] = float(np.median(dd))
                d[f"shape_{f}_spearman"] = float(_spear(o, dd))
            out[kind + tag] = d
    return out


def _spear(a, b):
    from scipy.stats import spearmanr

    return spearmanr(a, b).correlation if len(a) > 2 else np.nan


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--combo-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-nights", type=int, default=10)
    ap.add_argument("--png-per-kind", type=int, default=6)
    ap.add_argument("--png-kinds", default="kc,sw,sw_pool", help="event kinds that get PNGs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(os.path.join(a.out, "examples"), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = Codec(a.combo_dir, dev)
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    fs = float(meta["fs"]["eeg"])
    rng = np.random.default_rng(a.seed)
    keys = sorted(meta["nights"], key=lambda k: (len(k), k))
    kcx = [k for k in keys if meta["nights"][k]["unit"] == "dreams" and "kc_excerpt" in k]
    pool = [k for k in keys if meta["nights"][k]["unit"] not in ("dreams", "mass_ss01")]
    pool = [pool[i] for i in sorted(rng.choice(len(pool), min(a.pool_nights, len(pool)), replace=False))]
    rows, prog = [], {"examples": [], "nights": [], "started": time.strftime("%F %T")}

    def save_prog():
        prog["n_events"] = {k: sum(r["kind"] == k for r in rows) for k in ("kc", "sw", "sw_pool")}
        prog["running"] = summarize(rows) if rows else {}
        prog["elapsed_min"] = (time.time() - t0) / 60
        json.dump(prog, open(os.path.join(a.out, "progress.json"), "w"), indent=1, default=float)

    for setname, ks in (("dreams", kcx), ("pool", pool)):
        for k in ks:
            z = np.load(os.path.join(a.dump, f"{k}.npz"))
            s = sc.get(k, {}).get("eeg", {}).get("scale")
            if s is None or "x_eeg" not in z.files:
                continue
            x = z["x_eeg"].astype(np.float32)
            y = z["y"]
            n = len(x)
            ep_len = x.shape[1]
            if setname == "pool":
                idx = np.flatnonzero(y == 3)
                if len(idx) == 0:
                    continue
                idx = np.sort(rng.permutation(idx)[:120])
                dec = decode_night(codec, x, idx)
                xo = x[idx]
            else:
                idx = np.arange(n)
                dec = decode_night(codec, x)
                xo = x
            o_uv = xo.ravel() * float(s)
            d_uv = {kk: v.ravel() * float(s) for kk, v in dec.items()}
            o_low, o_br = CRF.bandpass(o_uv, 0.3, 4, fs), CRF.bandpass(o_uv, 0.3, 30, fs)
            d_low = {kk: CRF.bandpass(v, 0.3, 4, fs) for kk, v in d_uv.items()}
            d_br = {kk: CRF.bandpass(v, 0.3, 30, fs) for kk, v in d_uv.items()}
            o_mid = CRF.bandpass(o_uv, 0.3, 8, fs)
            d_mid = {kk: CRF.bandpass(v, 0.3, 8, fs) for kk, v in d_uv.items()}
            o_low_raw_cache[0] = o_uv
            if setname == "dreams":
                evs = events_night(meta["nights"][k], o_low, len(o_uv), fs)
            else:
                evs = []
                for aa, bb in CRF.detect_delta_waves(o_uv, fs):
                    c = aa + int(np.argmin(o_low[aa:bb]))
                    ep_i = c // ep_len
                    # window must stay inside one decoded epoch: the selected N3 epochs are not contiguous
                    if (c - int(10 * fs)) // ep_len != ep_i or (c + int(10 * fs)) // ep_len != ep_i:
                        continue
                    evs.append({"kind": "sw_pool", "t": c / fs, "c": c, "cons": False, "scorer": "-"})
            n_png = {"kc": 0, "sw": 0, "sw_pool": 0}
            kept = 0
            for ev in rng.permutation(len(evs)) if evs else []:
                ev = evs[ev]
                c = ev["c"]
                fo = feats(o_low, o_mid, c, fs)
                if fo is None:
                    continue
                fd = {kk: feats(d_low[kk], d_mid[kk], c, fs) for kk in dec}
                if any(v is None for v in fd.values()):
                    continue
                ep_local = c // ep_len
                stage_i = int(y[idx[ep_local]])
                stage = STAGES[stage_i] if 0 <= stage_i < 5 else "?"
                cross = (c - int(W_S * fs)) // ep_len != (c + int(W_S * fs)) // ep_len
                fid = {kk: fidelity(o_low, o_br, d_low[kk], d_br[kk], c, fs, fo, fd[kk]) for kk in dec}
                row = {"night": k, "set": setname, "kind": ev["kind"], "t": round(ev["t"], 2), "cons": ev["cons"],
                       "scorer": ev["scorer"], "stage": stage, "crosses_epoch": bool(cross), "orig": fo, **fd, "fid": fid}
                rows.append(row)
                kept += 1
                if ev["kind"] in a.png_kinds.split(",") and n_png[ev["kind"]] < a.png_per_kind:
                    n_png[ev["kind"]] += 1
                    lab = {"kc": "K-complex (expert" + (", both scorers" if ev["cons"] else f", {ev['scorer']}") + ")",
                           "sw": "slow wave, no expert KC", "sw_pool": "slow wave, pool N3"}[ev["kind"]]
                    f = fid["NEP"]
                    title = (f"{lab} · {k} t={ev['t']:.1f}s · stage {stage} · r={f['r_broad']:.2f} (0.3-4 Hz {f['r_low']:.2f})"
                             f" · p-p dec/orig {f['ptp_ratio']:.2f} · peak shift {f['lat_ms']:+.0f} ms · N+E r={fid['NE']['r_broad']:.2f}")
                    fn = f"{len(prog['examples']):03d}_{ev['kind']}_{k.split('/')[-1]}_{int(ev['t'])}.png"
                    marks = []
                    if ev.get("iv"):
                        marks.append((ev["iv"][0] - c / fs, ev["iv"][1] - c / fs, "#0969da", f"marked by {ev['scorer']}"))
                        marks += [(x - c / fs, y - c / fs, "#8250df", "marked by vis2") for x, y in ev["iv2"][:1]]
                    plot(os.path.join(a.out, "examples", fn), o_br, d_br, c, fs, title, marks)
                    prog["examples"].append({"file": fn, "kind": ev["kind"], "night": k, "t": ev["t"], "stage": stage,
                                             "cons": ev["cons"], "fid_NEP": f, "fid_NE": fid["NE"],
                                             "orig": {kk: fo[kk] for kk in SHAPE + ["ptp"]}, "scorer": ev["scorer"],
                                             "NEP": {kk: fd["NEP"][kk] for kk in SHAPE + ["ptp"]}})
            prog["nights"].append({"night": k, "set": setname, "events": kept})
            save_prog()
            print(f"[kcfid] {setname} {k} events={kept} total={len(rows)} {time.time() - t0:.0f}s", flush=True)

    res = {"summary": summarize(rows), "classifier": classify([r for r in rows if r["set"] == "dreams"], rng),
           "classifier_no_epoch_cross": classify([r for r in rows if r["set"] == "dreams" and not r["crosses_epoch"]], rng),
           "n_rows": len(rows), "minutes": (time.time() - t0) / 60}
    from physioex.train.bin.concept_compose import _jsonable

    json.dump(_jsonable(res), open(os.path.join(a.out, "kc_shape_fidelity.json"), "w"), indent=1)
    json.dump(_jsonable(rows), open(os.path.join(a.out, "rows.json"), "w"))
    prog["done"] = True
    prog["result"] = _jsonable(res)
    save_prog()
    print(f"[kcfid] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
