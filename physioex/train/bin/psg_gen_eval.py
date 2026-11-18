"""Blocks 4b and 5 of the psg_gen comparison (sleep-lens #5) on a ``concept_scan`` dump.

``rules``: training-free AASM-like staging from the decomposition of every scored epoch (EEG only,
so N1 and REM are merged: REM needs EOG and EMG). Order of the rules on one epoch:

1. N3 if the slow-wave events (non-periodic) cover >= 6 s (AASM: >= 20 % of the epoch);
2. W if the alpha band dominates theta (background log power) in >= 50 % of the windows, or the
   alpha bursts (8-11 Hz) cover >= 15 s (AASM: alpha rhythm > 50 % of the epoch);
3. N2 if there is a transient (K-complex shape) or a sigma burst (11-16 Hz, spindle);
4. else N1/REM.

The reference row applies the same rules to the rule detectors of the dump on the raw signal
(``delta_epoch``, ``alpha_epoch``, ``k_complex_rule`` / ``spindle_rule`` tokens). Cohen's kappa on
the 4 classes (W, N1+REM, N2, N3) per set, bootstrap over nights.

``lexicon``: per atom, a per-token score (the largest |amplitude| of its events whose support covers
the 0.5 s token, 0 elsewhere), AP against the token references (rule detectors on the pool, the two
DREAMS experts on the excerpts). Per reference: best atom, its class and frequency, and the number of
atoms within 90 % of its AP (redundancy).

``python -m physioex.train.bin.psg_gen_eval rules|lexicon --combo-dir DIR --dump DIR --out JSON``
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

TOK = 60                       # tokens per epoch (0.5 s)
STAGE4 = np.array([0, 1, 2, 3, 1])   # W, N1, N2, N3, R -> W, N1+R, N2, N3
LEX_REFS = {"k_complex_rule": "pool", "spindle_rule": "pool", "slow_wave": "pool", "alpha": "pool",
            "vertex": "pool", "k_complex_vis1": "dreams", "spindle_vis1": "dreams"}


def _nights(dump):
    meta = json.load(open(os.path.join(dump, "meta.json")))
    out = []
    for k in sorted(meta["nights"]):
        u = meta["nights"][k]["unit"]
        out.append((k, "dreams" if u == "dreams" else ("ss01" if u == "mass_ss01" else "pool")))
    return out


def _decompose_night(m, x, dev, batch=128):
    """Events (numpy dict, epoch = index in the night) and background logP (n, W, nb) of every epoch."""
    import torch

    from physioex.train.bin.concept_counterfactual import reflect_index

    half = m.sequence_length // 2 if hasattr(m, "sequence_length") else 20
    n = len(x)
    xp = x[reflect_index(n, half)].astype(np.float32)
    ev, logP = {k: [] for k in ("epoch", "atom", "cls", "onset", "dur", "amp", "periodic")}, []
    with torch.no_grad():
        for j in range(0, n, batch):
            c = np.arange(j, min(n, j + batch)) + half
            w = torch.from_numpy(np.stack([xp[t - half:t + half + 1] for t in c]))[:, :, None].to(dev)
            dec = m.encode(w)
            e = dec["events"]
            for k in ev:
                v = getattr(e, k).detach().cpu().numpy()
                ev[k].append(v + j if k == "epoch" else v)
            logP.append(dec["logP"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in ev.items()}, np.concatenate(logP)


def _load(combo_dir, dev):
    from physioex.train.bin.pretrain_combo import load_s2

    run, m = load_s2(os.path.join(combo_dir, "eeg", "s2"), "eeg", dev)
    m.sequence_length = run["args"]["seq"]
    return m


def _kappa(a, b):
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b)) if len(np.unique(np.r_[a, b])) > 1 else float("nan")


# ---------------------------------------------------------------------------- rules
def stage_model(ev, logP, n, d, fs=100):
    ok = ~ev["periodic"]
    sw = np.zeros(n)
    sel = ok & (ev["cls"] == 2)
    np.add.at(sw, ev["epoch"][sel], ev["dur"][sel])
    freq = d.freq.numpy()[ev["atom"]]
    burst = ok & (ev["cls"] == 0)
    al_b = np.zeros(n)
    sa = burst & (freq >= 8) & (freq < 11)
    np.add.at(al_b, ev["epoch"][sa], ev["dur"][sa])
    n2 = np.zeros(n, bool)
    n2[ev["epoch"][ok & ((ev["cls"] == 1) | (burst & (freq >= 11) & (freq < 16)))]] = True
    alpha_dom = (logP[..., 3] > logP[..., 2]).mean(1)
    st = np.full(n, 1)
    st[n2] = 2
    st[(alpha_dom >= 0.5) | (al_b >= 15)] = 0
    st[sw >= 6] = 3
    return st, {"sw_sec": sw, "alpha_dom": alpha_dom, "n2": n2}


def stage_refs(rf, n):
    tok = lambda k: rf[k][: n * TOK].reshape(n, TOK).any(1) if k in rf else np.zeros(n, bool)
    st = np.full(n, 1)
    st[tok("spindle_rule") | tok("k_complex_rule")] = 2
    if "alpha_epoch" in rf:
        st[rf["alpha_epoch"][:n]] = 0
    if "delta_epoch" in rf:
        st[rf["delta_epoch"][:n]] = 3
    return st


def rules(a):
    import torch

    from physioex.train.bin.concept_scan import load_refs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load(a.combo_dir, dev)
    d = m.d
    per = {}
    for k, s in _nights(a.dump):
        z = np.load(os.path.join(a.dump, f"{k}.npz"))
        x, y = z["x_eeg"].astype(np.float32), STAGE4[z["y"].astype(int)]
        rf = load_refs(os.path.join(a.dump, "refs", f"{k}.npz"))
        ev, logP = _decompose_night(m, x, dev)
        sm, _ = stage_model(ev, logP, len(x), d)
        per[k] = {"set": s, "y": y, "model": sm, "refs": stage_refs(rf, len(x))}
        print(f"[rules] {k}: kappa model {_kappa(y, sm):.3f} refs {_kappa(y, per[k]['refs']):.3f}", flush=True)
    rng = np.random.default_rng(0)
    out = {"classes": ["W", "N1+R", "N2", "N3"], "sets": {}}
    for s in ("pool", "ss01", "dreams"):
        ks = [k for k in per if per[k]["set"] == s]
        if not ks:
            continue
        row = {"n_nights": len(ks)}
        for src in ("model", "refs"):
            y = np.concatenate([per[k]["y"] for k in ks])
            p = np.concatenate([per[k][src] for k in ks])
            boot = []
            for _ in range(200):
                b = rng.choice(len(ks), len(ks))
                boot.append(_kappa(np.concatenate([per[ks[i]]["y"] for i in b]), np.concatenate([per[ks[i]][src] for i in b])))
            cm = np.zeros((4, 4), int)
            np.add.at(cm, (y, p), 1)
            row[src] = {"kappa": _kappa(y, p), "q05": float(np.nanquantile(boot, 0.05)),
                        "q95": float(np.nanquantile(boot, 0.95)),
                        "recall": [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(4)], "confusion": cm.tolist()}
        out["sets"][s] = row
        print(f"[rules] {s}: " + json.dumps({k: (v["kappa"], v["q05"], v["q95"]) if isinstance(v, dict) else v
                                              for k, v in row.items()}), flush=True)
    return out


# ---------------------------------------------------------------------------- lexicon
def _ap_sparse(idx, score, pos_idx, n_total):
    """AP of a score that is 0 on every token except ``idx`` (score > 0), against the positive tokens
    ``pos_idx`` of a population of ``n_total`` tokens (the zero tokens are one tied threshold)."""
    P = len(pos_idx)
    if P == 0:
        return None
    pos = np.isin(idx, pos_idx)
    o = np.argsort(-score, kind="stable")
    sc, ps = score[o], pos[o]
    ap, tp, fp, i = 0.0, 0, 0, 0
    while i < len(sc):
        j = i
        while j < len(sc) and sc[j] == sc[i]:
            j += 1
        dtp = int(ps[i:j].sum())
        tp, fp = tp + dtp, fp + (j - i) - dtp
        if dtp:
            ap += dtp / P * tp / (tp + fp)
        i = j
    rest_pos = P - tp
    if rest_pos:
        ap += rest_pos / P * P / n_total            # everything selected at the tied zero threshold
    return float(ap)


def lexicon(a):
    import torch

    from physioex.train.bin.concept_scan import load_refs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _load(a.combo_dir, dev)
    d = m.d
    K = len(d)
    sc = {s: [{} for _ in range(K)] for s in ("pool", "dreams")}        # atom -> {global token: score}
    pos = {r: [] for r in LEX_REFS}
    n_tok = {"pool": 0, "dreams": 0}
    for k, s in _nights(a.dump):
        if s == "ss01":
            continue
        z = np.load(os.path.join(a.dump, f"{k}.npz"))
        x = z["x_eeg"].astype(np.float32)
        n = len(x)
        rf = load_refs(os.path.join(a.dump, "refs", f"{k}.npz"))
        ev, _ = _decompose_night(m, x, dev)
        off = n_tok[s]
        for r, where in LEX_REFS.items():
            if where == s and r in rf:
                pos[r].append(off + np.flatnonzero(rf[r][: n * TOK]))
        ok = ~ev["periodic"]
        t0 = np.floor((ev["onset"] - ev["dur"] / 2) / 0.5).astype(int).clip(0, TOK - 1)
        t1 = np.floor((ev["onset"] + ev["dur"] / 2) / 0.5).astype(int).clip(0, TOK - 1)
        for i in np.flatnonzero(ok):
            at, base, v = int(ev["atom"][i]), off + int(ev["epoch"][i]) * TOK, abs(float(ev["amp"][i]))
            dct = sc[s][at]
            for t in range(base + t0[i], base + t1[i] + 1):
                if v > dct.get(t, 0.0):
                    dct[t] = v
        n_tok[s] += n * TOK
        print(f"[lexicon] {k}: {int(ok.sum())} events", flush=True)
    cls, freq, names = d.cls.numpy(), d.freq.numpy(), list(d.names)
    out = {"n_atoms": K, "n_tokens": n_tok, "refs": {}}
    for r, s in LEX_REFS.items():
        p = np.concatenate(pos[r]) if pos[r] else np.zeros(0, int)
        if not len(p):
            out["refs"][r] = {"skipped": "no positives"}
            continue
        aps = []
        for at in range(K):
            dct = sc[s][at]
            if not dct:
                aps.append(None)
                continue
            idx = np.fromiter(dct.keys(), int, len(dct))
            val = np.fromiter(dct.values(), float, len(dct))
            aps.append(_ap_sparse(idx, val, p, n_tok[s]))
        v = np.array([x if x is not None else np.nan for x in aps])
        best = int(np.nanargmax(v)) if np.isfinite(v).any() else None
        out["refs"][r] = {"set": s, "n_pos": int(len(p)), "prevalence": float(len(p) / n_tok[s]),
                          "best_atom": names[best] if best is not None else None,
                          "best_class": int(cls[best]) if best is not None else None,
                          "best_freq": float(freq[best]) if best is not None else None,
                          "best_ap": float(v[best]) if best is not None else None,
                          "n_within_90": int(np.nansum(v >= 0.9 * v[best])) if best is not None else 0,
                          "top5": [[names[i], float(v[i])] for i in np.argsort(-np.nan_to_num(v, nan=-1))[:5]]}
        print(f"[lexicon] {r}: {json.dumps(out['refs'][r])}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("rules", "lexicon"))
    ap.add_argument("--combo-dir", required=True, help="combo dir with eeg/s2 -> the psg_gen run")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = rules(a) if a.cmd == "rules" else lexicon(a)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
