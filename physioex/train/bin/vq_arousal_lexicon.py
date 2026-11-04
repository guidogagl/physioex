"""Arousal words of the frozen P lexicon of an MSAE-VQ run, named on MASS SS01 (expert MicroArousal).

Third event of the explainability workflow (``WORKFLOW_SPIEGABILITA.md``), after the K-complexes
and the spindles named on DREAMS. Same rules: the bases and the thresholds come from DCSM, never
from the target; the expert labels only choose the words (at most three) and score them,
leave-one-night-out. What changes for an arousal:

- it lasts 3-15 s (median 8 s, 16 tokens) and is a change of the high-frequency activity (theta,
  alpha, > 16 Hz) against the preceding sleep: the words are energies of a few directions of the
  ``pq`` x3 bases averaged over a centred window of ``win`` tokens (3, 7, 11), raw or as the log ratio
  to the median of the 20 tokens (10 s) before the window (``rel``);
- the events follow the AASM duration rule: runs of firing tokens with gaps of up to 2 tokens merged
  and at least 6 tokens (3 s) kept;
- arousals are scored in sleep only: the primary stratum is the scored sleep epochs (N1-REM, expert
  hypnogram) without the flat epochs (RMS < 0.1 x the night median); the scored epochs with wake are
  reported too.

Families of words: ``label_free`` (single directions, the 4 most kurtotic, the 10 most co-active
pairs of z^2, all directions; chosen on DCSM), ``stage_informed`` (the directions with the largest
W/N2 ratio of z^2 on DCSM: they use the DCSM hypnogram, never an event label) and ``signal`` (the
same windows and thresholds on the band power of the clean input signal: a rule-based detector
used as reference, not part of the lexicon). References: a supervised probe on ``pq`` x3 (linear
and MLP, 5 folds by night) for what the codes contain; nulls: random rotations of the whitened
``pq`` x3 in place of ICA, and the expert events circularly shifted inside each night.

``python -m physioex.train.bin.vq_arousal_lexicon readout --run RUN --out DIR`` (GPU)
``python -m physioex.train.bin.vq_arousal_lexicon lexicon --run RUN --lexicon LEX --dcsm-fit NPZ --out DIR`` (CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from physioex.models.vq_lexicon import Lexicon, ctx3, rel_change, token_runs, window_mean
from physioex.train.bin.vq_event_lexicon2 import match_count, merge_refs, prf

FS, TOK, EPOCH_S, T = 100, 0.5, 30.0, 60
TOL = 0.5
GAP, MIN_TOK = 2, 6
WINS, RELS, QS = (3, 7, 11), (0, 20), (0.9, 0.95, 0.975, 0.99)
COV_MIN = 0.5
MAX_WORDS = 3
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30), "gamma": (30, 45)}
SIGNAL_GROUPS = {"theta-gamma": [1, 2, 4, 5], "alpha-gamma": [2, 4, 5], "beta-gamma": [4, 5], "alpha": [2], "theta": [1]}
STAGES = ("W", "N1", "N2", "N3", "REM")


# ---------------------------------------------------------------------------- readout (GPU)
def token_bandpower(x: np.ndarray) -> np.ndarray:
    """log10 band power of 1 s Hann windows centred on every 0.5 s token: ``(n_tok, len(BANDS))``."""
    from numpy.lib.stride_tricks import sliding_window_view

    n_tok = len(x) // 50
    xp = np.pad(x, (25, 75), mode="reflect")
    w = sliding_window_view(xp, 100)[::50][:n_tok] * np.hanning(100)
    p = np.abs(np.fft.rfft(w, axis=1)) ** 2
    f = np.fft.rfftfreq(100, 1 / FS)
    return np.log10(np.stack([p[:, (f >= a) & (f < b)].sum(1) for a, b in BANDS.values()], 1) + 1e-9)


def arousal_refs(events) -> List[tuple]:
    """Expert MicroArousal events (C3 and C4 annotations merged when they overlap)."""
    ev = sorted((e.onset_sec, e.onset_sec + e.duration_sec) for e in events if e.type == "arousal")
    return merge_refs(ev, [], "union")


def readout(a):
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines
    from physioex.train.bin.vq_event_readout import readout_recording

    run, dcsm_ds, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    mds = get_dataset("mass")(cohort=1, channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                              sequence_length=1, cache_dir=ra.get("cache_dir"))
    od = os.path.join(a.out, "readout")
    os.makedirs(od, exist_ok=True)
    meta = {"run": a.run, "vq": a.vq, "mass": {}, "dcsm": {}}

    def one(ds, spec, tag):
        sid = spec.subject_id
        n = ds._n_epochs[sid]
        x = next(iter(ds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        y = np.asarray(ds._get_labels(spec))[:n].astype(np.int8)
        rms = np.sqrt((x ** 2).mean(1))
        flat = rms < 0.1 * np.median(rms)
        r = readout_recording(model, x, ra["seq"])
        np.savez(os.path.join(od, f"{tag}_{sid}.npz"), pq=r["pq"], zE=r["zE"], stage=y, flat=flat,
                 bp=token_bandpower(x.reshape(-1)).astype(np.float32))
        m = {"n_epochs": int(n), "n_flat": int(flat.sum())}
        if tag == "mass":
            m["arousals"] = arousal_refs(ds._load_or_compute_events(spec))
        meta[tag][sid] = m
        print(f"[arousal readout] {tag} {sid} {n} epochs {int(flat.sum())} flat {len(m.get('arousals', []))} arousals", flush=True)

    for spec in mds._subjects:
        one(mds, spec, "mass")
    have = {s.subject_id: s for s in dcsm_ds._subjects}
    for sid in [s for s in run["splits"]["train"] if s in have][:a.dcsm_nights]:
        one(dcsm_ds, have[sid], "dcsm")
    json.dump(meta, open(os.path.join(od, "meta.json"), "w"))
    print("DONE", flush=True)


# ---------------------------------------------------------------------------- events and counts
def mask_events(m: np.ndarray, gap: int = GAP, min_tok: int = MIN_TOK) -> List[tuple]:
    return [(s * TOK - TOL, (e + 1) * TOK + TOL) for s, e in token_runs(m, gap, min_tok)]


def ev_stage(ev, stage):
    k = int(0.5 * (ev[0] + ev[1]) // EPOCH_S)
    return int(stage[k]) if 0 <= k < len(stage) else -1


def acounts(m: np.ndarray, ref: List[tuple], stage=None, s=None, gap=GAP, min_tok=MIN_TOK) -> np.ndarray:
    pe = mask_events(m, gap, min_tok)
    if s is not None:
        pe = [e for e in pe if ev_stage(e, stage) == s]
        ref = [e for e in ref if ev_stage(e, stage) == s]
    return np.array([match_count(pe, ref), len(pe), len(ref)], np.int64)


# ---------------------------------------------------------------------------- data held by the workers
D: dict = {}


def load(a):
    rd = os.path.join(a.out, "readout")
    meta = json.load(open(os.path.join(rd, "meta.json")))
    lex = Lexicon.load(a.lexicon)
    nights = sorted(meta["mass"])
    dn = sorted(meta["dcsm"])
    Z, Zd = {b: {} for b in lex.bases}, {b: [] for b in lex.bases}
    Z["bp"], Zd["bp"] = {}, []
    valid, valid_all, stage, refs, refs_all, pq3, dstage = {}, {}, {}, {}, {}, {}, []
    for sid in nights:
        z = np.load(os.path.join(rd, f"mass_{sid}.npz"))
        pq = z["pq"].astype(np.float32)
        for b in lex.bases:
            Z[b][sid] = lex.coords(pq, b).astype(np.float32)
        Z["bp"][sid] = (10.0 ** z["bp"]).astype(np.float32)
        st = np.full(len(z["flat"]), -1)
        st[:len(z["stage"])] = z["stage"][:len(st)]  # labels can be one epoch short
        ok = ~z["flat"]
        stage[sid] = np.where(ok, st, -1)
        valid[sid] = np.repeat(ok & (st >= 1) & (st <= 4), T)
        valid_all[sid] = np.repeat(ok & (st >= 0), T)
        r = [tuple(e) for e in meta["mass"][sid]["arousals"]]
        refs[sid] = [e for e in r if ev_stage(e, stage[sid]) in (1, 2, 3, 4)]
        refs_all[sid] = [e for e in r if ev_stage(e, stage[sid]) >= 0]
        pq3[sid] = pq
    for sid in dn:
        z = np.load(os.path.join(rd, f"dcsm_{sid}.npz"))
        pq = z["pq"].astype(np.float32)
        st = np.full(len(z["flat"]), -1)
        st[:len(z["stage"])] = z["stage"][:len(st)]  # labels can be one epoch short
        keep = np.repeat((~z["flat"]) & (st >= 0), T)
        for b in lex.bases:
            Zd[b].append((lex.coords(pq, b).astype(np.float32), keep))
        Zd["bp"].append(((10.0 ** z["bp"]).astype(np.float32), keep))
        dstage.append(np.repeat(st, T)[keep])
    D.update(lex=lex, nights=nights, Z=Z, Zd=Zd, valid=valid, valid_all=valid_all, stage=stage, refs=refs,
             refs_all=refs_all, pq=pq3, dstage=np.concatenate(dstage), meta=meta, dcsm_nights=dn)


def groups_of(basis: str, family_lf=True, family_si=True) -> List[dict]:
    """Direction groups chosen on DCSM (continuous nights): label-free and stage-informed."""
    from scipy.stats import kurtosis

    Zc = np.concatenate([z[k] for z, k in D["Zd"][basis]])
    out = []
    if family_lf:
        out += [{"basis": basis, "name": f"d{d}", "dirs": [d], "family": "label_free"} for d in range(Zc.shape[1])]
        kt = np.argsort(-kurtosis(Zc, axis=0))
        out.append({"basis": basis, "name": "kurt4", "dirs": [int(d) for d in kt[:4]], "family": "label_free"})
        sq = Zc[:: max(1, len(Zc) // 50000)] ** 2
        C = np.corrcoef(sq.T)
        iu = np.triu_indices(len(C), 1)
        for k in np.argsort(-C[iu])[:10]:
            i, j = int(iu[0][k]), int(iu[1][k])
            out.append({"basis": basis, "name": f"d{i}+d{j}", "dirs": [i, j], "family": "label_free"})
        out.append({"basis": basis, "name": "all", "dirs": list(range(Zc.shape[1])), "family": "label_free"})
    if family_si:
        ds = D["dstage"]
        ratio = (Zc[ds == 0] ** 2).mean(0) / np.maximum((Zc[ds == 2] ** 2).mean(0), 1e-9)
        top = np.argsort(-ratio)
        for k in (2, 4, 8):
            out.append({"basis": basis, "name": f"wake{k}", "dirs": [int(d) for d in top[:k]], "family": "stage_informed"})
    return out


def values(g: dict, win: int, rel: int, Zs: np.ndarray) -> np.ndarray:
    e = Zs[:, g["dirs"]].sum(1) if g["basis"] in ("bp", "tex") else (Zs[:, g["dirs"]] ** 2).sum(1)
    if win > 1:
        e = window_mean(e, win, len(e))
    if rel:
        e = rel_change(e, rel, win, len(e))
    return e


def word_name(g, win, rel, q):
    return f"{g['basis']}:E[{g['name']}]w{win}{'r' if rel else ''}>=q{q}"


def group_words(g: dict) -> List[dict]:
    """All words of one group: DCSM thresholds, counts per night (sleep stratum), firing by stage."""
    out = []
    for win in WINS:
        for rel in RELS:
            dv = np.concatenate([values(g, win, rel, z)[k] for z, k in D["Zd"][g["basis"]]])
            mv = {sid: values(g, win, rel, D["Z"][g["basis"]][sid]) for sid in D["nights"]}
            for q in QS:
                thr = float(np.quantile(dv, q))
                fire = dv >= thr
                c = np.stack([acounts((mv[s] >= thr) & D["valid"][s], D["refs"][s]) for s in D["nights"]])
                out.append({**g, "win": win, "rel": rel, "q": q, "threshold": thr, "name": word_name(g, win, rel, q),
                            "counts": c, "dcsm_stage_mix": [float((D["dstage"][fire] == i).mean()) for i in range(5)],
                            "dcsm_rate": [float(fire[D["dstage"] == i].mean()) for i in range(5)]})
    return out


def build_words(groups: List[dict], n_jobs: int) -> List[dict]:
    import multiprocessing as mp

    with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(group_words, groups, chunksize=1))
    return [w for r in res for w in r]


# ---------------------------------------------------------------------------- selection
class Masks:
    """Word masks per night, computed on demand and cached (the greedy search reuses few words)."""

    def __init__(self, words, valid):
        self.words, self.valid, self.c = words, valid, {}

    def __call__(self, w: int, sid: str) -> np.ndarray:
        k = (w, sid)
        if k not in self.c:
            x = self.words[w]
            self.c[k] = (values(x, x["win"], x["rel"], D["Z"][x["basis"]][sid]) >= x["threshold"]) & self.valid[sid]
        return self.c[k]

    def union(self, ws, sid):
        return np.any([self(w, sid) for w in ws], 0)


def select(words, M: Masks, train: List[str], refs, cand_idx=None, n_cand: int = 30) -> List[int]:
    """Greedy lexicon of <= MAX_WORDS words on the training nights (rule of S4b): the purest word with
    coverage >= COV_MIN (else the best F1), then the word that most raises the F1 of the union."""
    pos = {s: i for i, s in enumerate(D["nights"])}
    ti = [pos[s] for s in train]
    idx = np.arange(len(words)) if cand_idx is None else np.asarray(cand_idx)
    c = np.stack([words[w]["counts"][ti].sum(0) for w in idx])
    pur, cov = c[:, 0] / np.maximum(c[:, 1], 1), c[:, 0] / np.maximum(c[:, 2], 1)
    f1 = np.where(pur + cov > 0, 2 * pur * cov / np.maximum(pur + cov, 1e-9), 0)
    ok = np.flatnonzero(cov >= COV_MIN)
    first = int(idx[ok[np.argmax(pur[ok])]]) if len(ok) else int(idx[np.argmax(f1)])
    chosen, cand = [first], [int(idx[k]) for k in np.argsort(-f1)[:n_cand] if idx[k] != first]

    def uf1(ws):
        return prf(sum(acounts(M.union(ws, s), refs[s]) for s in train))["f1"]

    best = uf1(chosen)
    while len(chosen) < MAX_WORDS and cand:
        f, w = max((uf1(chosen + [w]), w) for w in cand)
        if f <= best + 1e-4:
            break
        chosen.append(w); cand.remove(w); best = f
    return chosen


def per_stage(M: Masks, ws, nights, refs) -> dict:
    out = {}
    for k in range(1, 5):
        out[STAGES[k]] = prf(sum(acounts(M.union(ws, s), refs[s], D["stage"][s], k) for s in nights))
    return out


def loo(words, cand_idx, refs, valid, log, tag) -> dict:
    M = Masks(words, valid)
    nights = D["nights"]
    tot, one, per_night, picks = np.zeros(3, np.int64), np.zeros(3, np.int64), {}, []
    stage_c = {k: np.zeros(3, np.int64) for k in range(1, 5)}
    for te in nights:
        ws = select(words, M, [s for s in nights if s != te], refs, cand_idx)
        picks.append(ws)
        c = acounts(M.union(ws, te), refs[te])
        tot += c; one += acounts(M(ws[0], te), refs[te]); per_night[te] = prf(c)
        for k in stage_c:
            stage_c[k] += acounts(M.union(ws, te), refs[te], D["stage"][te], k)
    freq = {}
    for ws in picks:
        for w in ws:
            freq[w] = freq.get(w, 0) + 1
    res = {"lexicon": prf(tot), "single_word": prf(one), "per_stage": {STAGES[k]: prf(v) for k, v in stage_c.items()},
           "per_night": per_night,
           "words_picked": [{"word": words[w]["name"], "folds": n} for w, n in sorted(freq.items(), key=lambda kv: -kv[1])[:8]]}
    log(f"[loo {tag}] {res['lexicon']} picks {res['words_picked'][:4]}")
    return res


def shift_null(M: Masks, ws, refs, n_shift: int, rng) -> dict:
    """Purity / coverage of the final words against the expert events circularly shifted inside each
    night (offset >= 60 s), refs kept in the stratum."""
    pur, cov = [], []
    for _ in range(n_shift):
        c = np.zeros(3, np.int64)
        for s in D["nights"]:
            L = len(D["stage"][s]) * EPOCH_S
            off = rng.uniform(60, L - 60)
            sh = [((a + off) % L, (b + off) % L) for a, b in refs[s]]
            sh = [e for e in sh if e[1] > e[0] and ev_stage(e, D["stage"][s]) in (1, 2, 3, 4)]
            c += acounts(M.union(ws, s), sh)
        p = prf(c)
        pur.append(p["purity"]); cov.append(p["coverage"])
    return {"purity": [float(v) for v in pur], "coverage": [float(v) for v in cov],
            "purity_q95": float(np.quantile(pur, 0.95)), "coverage_q95": float(np.quantile(cov, 0.95))}


# ---------------------------------------------------------------------------- probes (supervised reference)
def token_targets(sid) -> np.ndarray:
    n = len(D["valid"][sid])
    t0 = np.arange(n) * TOK
    y = np.zeros(n, bool)
    for a, b in D["refs"][sid]:
        y |= (t0 + TOK > a) & (t0 < b)
    return y


def probes(log, n_train: int = 300000, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    nights = D["nights"]
    X = {s: ctx3(D["pq"][s]) for s in nights}
    y = {s: token_targets(s) for s in nights}
    out = {}
    for name, make in (("linear", lambda: LogisticRegression(max_iter=300, C=1.0)),
                       ("mlp", lambda: MLPClassifier((128,), early_stopping=True, max_iter=60, random_state=seed))):
        score = {}
        for tr, te in GroupKFold(min(5, len(nights))).split(nights, groups=nights):
            trn = [nights[i] for i in tr]
            Xt = np.concatenate([X[s][D["valid"][s]] for s in trn])
            yt = np.concatenate([y[s][D["valid"][s]] for s in trn])
            sub = rng.choice(len(Xt), min(n_train, len(Xt)), replace=False)
            sc = StandardScaler().fit(Xt[sub])
            clf = make().fit(sc.transform(Xt[sub]), yt[sub])
            for i in te:
                score[nights[i]] = clf.predict_proba(sc.transform(X[nights[i]]))[:, 1]
            # threshold for the events: chosen on the training nights of this fold
            ptr = {s: clf.predict_proba(sc.transform(X[s]))[:, 1] for s in trn}
            qs = np.quantile(np.concatenate([ptr[s][D["valid"][s]] for s in trn]), np.linspace(0.8, 0.995, 40))
            best = None
            for q in qs:
                c = sum(acounts((ptr[s] >= q) & D["valid"][s], D["refs"][s]) for s in trn)
                p = prf(c)
                if p["coverage"] >= COV_MIN and (best is None or p["purity"] > best[1]):
                    best = (q, p["purity"])
            for i in te:
                score[nights[i] + "__thr"] = best[0] if best else qs[0]
        ap = average_precision_score(np.concatenate([y[s][D["valid"][s]] for s in nights]),
                                     np.concatenate([score[s][D["valid"][s]] for s in nights]))
        c = sum(acounts((score[s] >= score[s + "__thr"]) & D["valid"][s], D["refs"][s]) for s in nights)
        base = float(np.mean(np.concatenate([y[s][D["valid"][s]] for s in nights])))
        out[name] = {"token_ap": float(ap), "base_rate": base, "events": prf(c)}
        log(f"[probe {name}] AP {ap:.3f} (base {base:.3f}) events {out[name]['events']}")
    return out


# ---------------------------------------------------------------------------- null bases
def rotation_null(a, seeds, n_jobs, log) -> List[dict]:
    from physioex.train.bin.vq_event_lexicon2 import RandomRotation

    dc = np.load(a.dcsm_fit)
    X3 = ctx3(dc["pq"])
    k = D["lex"].bases["pq3_ica"]["unmix"].shape[0]
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rot = RandomRotation(X3[rng.choice(len(X3), min(120000, len(X3)), replace=False)], k, seed)
        b = f"rot{seed}"
        D["Z"][b] = {s: rot.transform(ctx3(D["pq"][s])).astype(np.float32) for s in D["nights"]}
        D["Zd"][b] = [(rot.transform(ctx3(np.load(os.path.join(a.out, "readout", f"dcsm_{s}.npz"))["pq"].astype(np.float32))).astype(np.float32), k_)
                      for s, (_, k_) in zip(D["dcsm_nights"], D["Zd"]["pq3_ica"])]
        words = build_words(groups_of(b, True, False), n_jobs)
        M = Masks(words, D["valid"])
        ws = select(words, M, D["nights"], D["refs"])
        p = prf(sum(acounts(M.union(ws, s), D["refs"][s]) for s in D["nights"]))
        out.append({"seed": seed, "in_sample": p, "words": [words[w]["name"] for w in ws]})
        log(f"[null rot{seed}] {p}")
        del D["Z"][b], D["Zd"][b]
    return out


# ---------------------------------------------------------------------------- lexicon (CPU)
def lexicon(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "arousal_lexicon.log"), "a")

    def log(m):
        m = f"[{time.time() - t0:7.0f}s] {m}"
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    load(a)
    nights = D["nights"]
    res = {"run": a.run, "lexicon": a.lexicon, "nights": nights, "dcsm_nights": D["dcsm_nights"],
           "rules": {"tol_s": TOL, "gap_tok": GAP, "min_tok": MIN_TOK, "wins": WINS, "rels": RELS, "qs": QS,
                     "cov_min": COV_MIN, "max_words": MAX_WORDS, "bands": BANDS, "signal_groups": SIGNAL_GROUPS},
           "n_ref": {"sleep": int(sum(len(D["refs"][s]) for s in nights)),
                     "scored": int(sum(len(D["refs_all"][s]) for s in nights)),
                     "all": int(sum(len(D["meta"]["mass"][s]["arousals"]) for s in nights))},
           "ref_stage": {STAGES[k]: int(sum(ev_stage(e, D["stage"][s]) == k for s in nights for e in D["refs_all"][s])) for k in range(5)},
           "tokens": {"sleep": int(sum(D["valid"][s].sum() for s in nights)), "scored": int(sum(D["valid_all"][s].sum() for s in nights))}}
    log(f"refs {res['n_ref']} by stage {res['ref_stage']}")

    groups = [g for b in D["lex"].bases for g in groups_of(b)]
    groups += [{"basis": "bp", "name": k, "dirs": v, "family": "signal"} for k, v in SIGNAL_GROUPS.items()]
    words = build_words(groups, a.jobs)
    log(f"{len(groups)} groups, {len(words)} words")
    fam = {f: [i for i, w in enumerate(words) if w["family"] == f] for f in ("label_free", "stage_informed", "signal")}
    latent = fam["label_free"] + fam["stage_informed"]

    # word table: in-sample counts of every word (for the page and for the record)
    tab = []
    for i, w in enumerate(words):
        p = prf(w["counts"].sum(0))
        tab.append({"i": i, "name": w["name"], "family": w["family"], "basis": w["basis"], "dirs": w["dirs"], "win": w["win"],
                    "rel": w["rel"], "q": w["q"], "threshold": w["threshold"], "purity": p["purity"], "coverage": p["coverage"],
                    "f1": p["f1"], "n_pred": p["n_pred"], "dcsm_stage_mix": w["dcsm_stage_mix"], "dcsm_rate": w["dcsm_rate"]})
    res["word_table_top"] = sorted(tab, key=lambda r: -r["f1"])[:60]
    res["best_single"] = {f: max((tab[i] for i in idx if tab[i]["coverage"] >= COV_MIN), key=lambda r: r["purity"], default=None)
                          for f, idx in fam.items()}

    # leave-one-night-out
    res["loo"] = {"latent": loo(words, latent, D["refs"], D["valid"], log, "latent"),
                  "label_free": loo(words, fam["label_free"], D["refs"], D["valid"], log, "label_free"),
                  "signal": loo(words, fam["signal"], D["refs"], D["valid"], log, "signal")}

    # final lexicon (all nights) and its checks
    M = Masks(words, D["valid"])
    ws = select(words, M, nights, D["refs"], latent)
    fin = {"words": [{k: words[w][k] for k in ("name", "family", "basis", "dirs", "win", "rel", "q", "threshold",
                                               "dcsm_stage_mix", "dcsm_rate")} for w in ws]}
    fin["in_sample"] = prf(sum(acounts(M.union(ws, s), D["refs"][s]) for s in nights))
    fin["per_stage"] = per_stage(M, ws, nights, D["refs"])
    fin["word_alone"] = [prf(sum(acounts(M(w, s), D["refs"][s]) for s in nights)) for w in ws]
    fin["default_rule"] = prf(sum(acounts(M.union(ws, s), D["refs"][s], gap=1, min_tok=1) for s in nights))
    Ma = Masks(words, D["valid_all"])
    fin["scored_with_wake"] = prf(sum(acounts(Ma.union(ws, s), D["refs_all"][s]) for s in nights))
    fin["shift_null"] = shift_null(M, ws, D["refs"], a.null_shift, np.random.default_rng(1))
    # firing rate per stage on all non-flat tokens, MASS vs DCSM (domain, L2)
    raw = Masks(words, {s: np.ones(len(D["valid"][s]), bool) for s in nights})
    rate = {}
    for k in range(5):
        tok = {s: np.repeat(D["stage"][s] == k, T) for s in nights}
        rate[STAGES[k]] = float(sum(raw.union(ws, s)[tok[s]].sum() for s in nights) / max(sum(tok[s].sum() for s in nights), 1))
    fin["mass_rate"] = rate
    log(f"[final] {fin['in_sample']} words {[w['name'] for w in fin['words']]}")
    res["final"] = fin

    # events of the final lexicon per night (page: timelines, examples)
    ev = {}
    for s in nights:
        m = M.union(ws, s)
        ev[s] = {"lexicon": [[a_, b_, [words[w]["name"] for w in ws if M(w, s)[int((a_ + TOL) / TOK):int((b_ - TOL) / TOK)].any()]]
                             for a_, b_ in mask_events(m)],
                 "expert": D["refs"][s], "stages": D["stage"][s].tolist()}
    json.dump(ev, open(os.path.join(a.out, "arousal_events.json"), "w"))

    # frozen reader: bases of the loaded lexicon + the arousal words + the event rule
    lexw = [{"name": words[w]["name"], "event": "arousal", "basis": words[w]["basis"], "kind": "energy", "dirs": words[w]["dirs"],
             "threshold": words[w]["threshold"], "side": 1, "win": words[w]["win"], "rel": words[w]["rel"]} for w in ws]
    used = {w["basis"] for w in lexw}
    L = Lexicon({b: v for b, v in D["lex"].bases.items() if b in used}, lexw,
                {"event_rules": {"arousal": {"gap": GAP, "min_tok": MIN_TOK}}, "named_on": "mass_ss01 MicroArousal (expert)",
                 "fit_on": "dcsm", "source_lexicon": a.lexicon})
    L.save(os.path.join(a.out, "lexicon"))
    # the frozen reader must give the same events as the selection code (all tokens, no stratum)
    s0 = nights[0]
    lev = [(e["start_s"] - TOL, e["stop_s"] + TOL) for e in L.events(D["pq"][s0], continuous=True)["arousal"]]
    mine = mask_events(raw.union(ws, s0))
    res["reader_check"] = {"night": s0, "n_reader": len(lev), "n_selection": len(mine),
                           "equal": bool(len(lev) == len(mine) and np.allclose(lev, mine))}
    log(f"reader check {res['reader_check']}")

    res["probes"] = probes(log)
    res["null_rotation"] = rotation_null(a, a.null_seeds, a.jobs, log)
    lf_ws = select(words, M, nights, D["refs"], fam["label_free"])
    res["label_free_in_sample"] = prf(sum(acounts(M.union(lf_ws, s), D["refs"][s]) for s in nights))
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(a.out, "arousal_lexicon.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("DONE")


# ---------------------------------------------------------------------------- decoder words (texture)
TEX_GROUPS = {"beta-gamma": ["17-31", "31-51"], "beta": ["17-31"], "alpha": ["9-13"], "alpha-gamma": ["9-13", "13-17", "17-31", "31-51"],
              "theta-gamma": ["5-9", "9-13", "13-17", "17-31", "31-51"], "theta": ["5-9"]}


def tex_batch(model, w: np.ndarray) -> np.ndarray:
    """Per-token log10 amplitude of the texture bands decoded from the quantized N, E, P of the
    centre epoch (same path as ``MSAE.decode`` + ``noise_texture``, without the noise)."""
    import torch

    dev = next(model.parameters()).device
    x = torch.from_numpy(w).to(dev).float()
    with torch.no_grad():
        n = model.encode_context(x)
        if model.use_n:
            n, _, _ = model._q("N", n)
        h = model.features(x[:, model.c])
        g, b = model.e_film(n).chunk(2, -1)
        he = h * (1 + g[:, None]) + b[:, None]
        e = torch.tanh(model.e_out(model.e_pool(model.e_query.expand(len(h), -1, -1), he, he, need_weights=False)[0][:, 0]))
        e, _, _ = model._q("E", e)
        g, b = model.p_film(torch.cat([n, e], -1)).chunk(2, -1)
        p = torch.tanh(model.p_out(h * (1 + g[:, None]) + b[:, None]))
        pq, _, _ = model._q("P", p)
        z = (model.dec_n(n) + model.dec_e(e))[:, None] + model.pos_dec + model.dec_p(pq)
        lg = model.tex_head(model.dec_tok(z)).float().clamp(-6, 2)
    return lg.cpu().numpy()


def texout(a):
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines
    from physioex.train.bin.vq_event_codes import reflect_index

    run, dcsm_ds, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    assert model.texture, "the run has no texture head"
    bands = [f"{model.pow_bands[i][0]}-{model.pow_bands[i][1]}" for i in model.tex_bands]
    od = os.path.join(a.out, "readout")
    meta = json.load(open(os.path.join(od, "meta.json")))
    mds = get_dataset("mass")(cohort=1, channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                              sequence_length=1, cache_dir=ra.get("cache_dir"))
    have = {**{("mass", s.subject_id): (mds, s) for s in mds._subjects},
            **{("dcsm", s.subject_id): (dcsm_ds, s) for s in dcsm_ds._subjects}}
    seq, half = ra["seq"], ra["seq"] // 2
    for tag in ("mass", "dcsm"):
        for sid in meta[tag]:
            ds, spec = have[(tag, sid)]
            n = ds._n_epochs[sid]
            x = next(iter(ds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
            idx = reflect_index(n, half)
            out = [tex_batch(model, np.stack([x[idx[s:s + seq]] for s in range(i, min(n, i + 64))])) for i in range(0, n, 64)]
            np.save(os.path.join(od, f"tex_{tag}_{sid}.npy"), np.concatenate(out).reshape(n * T, -1).astype(np.float32))
            print(f"[arousal texout] {tag} {sid}", flush=True)
    meta["tex_bands"] = bands
    json.dump(meta, open(os.path.join(od, "meta.json"), "w"))
    print("DONE", flush=True)


def decoder(a):
    """Arousal words on the decoded texture power (``tex``): same windows, thresholds on DCSM, AASM
    rule and leave-one-night-out selection as the latent words; the input band power (``signal``)
    is rerun as reference."""
    t0 = time.time()
    load(a)
    rd = os.path.join(a.out, "readout")
    bands = D["meta"]["tex_bands"]
    D["Z"]["tex"] = {s: (10.0 ** (2 * np.load(os.path.join(rd, f"tex_mass_{s}.npy")))).astype(np.float32) for s in D["nights"]}
    D["Zd"]["tex"] = [((10.0 ** (2 * np.load(os.path.join(rd, f"tex_dcsm_{s}.npy")))).astype(np.float32), k)
                      for s, (_, k) in zip(D["dcsm_nights"], D["Zd"]["bp"])]
    groups = [{"basis": "tex", "name": k, "dirs": [bands.index(b) for b in v], "family": "decoder"} for k, v in TEX_GROUPS.items()]
    groups += [{"basis": "bp", "name": k, "dirs": v, "family": "signal"} for k, v in SIGNAL_GROUPS.items()]
    words = build_words(groups, a.jobs)
    fam = {f: [i for i, w in enumerate(words) if w["family"] == f] for f in ("decoder", "signal")}
    nights = D["nights"]
    res = {"tex_bands": bands, "groups": TEX_GROUPS, "loo": {f: loo(words, idx, D["refs"], D["valid"], print, f) for f, idx in fam.items()}}
    M = Masks(words, D["valid"])
    ws = select(words, M, nights, D["refs"], fam["decoder"])
    fin = {"words": [{k: words[w][k] for k in ("name", "dirs", "win", "rel", "q", "threshold", "dcsm_stage_mix", "dcsm_rate")} for w in ws]}
    fin["in_sample"] = prf(sum(acounts(M.union(ws, s), D["refs"][s]) for s in nights))
    fin["per_stage"] = per_stage(M, ws, nights, D["refs"])
    fin["word_alone"] = [prf(sum(acounts(M(w, s), D["refs"][s]) for s in nights)) for w in ws]
    fin["shift_null"] = shift_null(M, ws, D["refs"], a.null_shift, np.random.default_rng(1))
    Ma = Masks(words, D["valid_all"])
    fin["scored_with_wake"] = prf(sum(acounts(Ma.union(ws, s), D["refs_all"][s]) for s in nights))
    raw = Masks(words, {s: np.ones(len(D["valid"][s]), bool) for s in nights})
    fin["mass_rate"] = {STAGES[k]: float(sum(raw.union(ws, s)[np.repeat(D["stage"][s] == k, T)].sum() for s in nights)
                                         / max(sum(np.repeat(D["stage"][s] == k, T).sum() for s in nights), 1)) for k in range(5)}
    # decoded vs input power per token (sleep tokens): how much of the input band change the codes keep
    corr = {}
    for gname, v in TEX_GROUPS.items():
        ti = [bands.index(b) for b in v]
        ii = [ {"5-9": 1, "9-13": 2, "13-17": 3, "17-31": 4, "31-51": 5}[b] for b in v]
        dv = np.concatenate([np.log10(D["Z"]["tex"][s][:, ti].sum(1))[D["valid"][s]] for s in nights])
        iv = np.concatenate([np.log10(D["Z"]["bp"][s][:, ii].sum(1))[D["valid"][s]] for s in nights])
        corr[gname] = float(np.corrcoef(dv, iv)[0, 1])
    fin["tex_vs_input_corr"] = corr
    print("[decoder final]", fin["in_sample"], [words[w]["name"] for w in ws], "per stage", fin["per_stage"], "corr", corr, flush=True)
    res["final"] = fin
    ev = {s: {"lexicon": [[a_, b_] for a_, b_ in mask_events(M.union(ws, s))]} for s in nights}
    json.dump(ev, open(os.path.join(a.out, "arousal_decoder_events.json"), "w"))
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(a.out, "arousal_decoder.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("readout")
    r.add_argument("--run", required=True)
    r.add_argument("--vq", default="N8E8P6")
    r.add_argument("--dcsm-nights", type=int, default=12)
    r.add_argument("--out", required=True)
    x = sp.add_parser("lexicon")
    x.add_argument("--run", required=True)
    x.add_argument("--lexicon", required=True, help="frozen Lexicon of the run (its bases are reused)")
    x.add_argument("--dcsm-fit", required=True, help="dcsm_readout.npz the bases were fitted on (null rotations)")
    x.add_argument("--jobs", type=int, default=8)
    x.add_argument("--null-shift", type=int, default=20)
    x.add_argument("--null-seeds", type=int, nargs="+", default=[0, 1, 2])
    x.add_argument("--out", required=True)
    t = sp.add_parser("texout")
    t.add_argument("--run", required=True)
    t.add_argument("--vq", default="N8E8P6")
    t.add_argument("--out", required=True)
    d = sp.add_parser("decoder")
    d.add_argument("--run", required=True)
    d.add_argument("--lexicon", required=True)
    d.add_argument("--jobs", type=int, default=8)
    d.add_argument("--null-shift", type=int, default=20)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    {"readout": readout, "lexicon": lexicon, "texout": texout, "decoder": decoder}[a.cmd](a)


if __name__ == "__main__":
    main()
