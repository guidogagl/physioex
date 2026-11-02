"""Do expert micro-events (DREAMS K-complexes, spindles) correspond to codebook elements, or to
co-occurrences of codebook elements, of the MSAE-VQ EEG encoder?

Label-free codes, expert events used only to evaluate:

1. **Codes.** Every epoch of every DREAMS excerpt is encoded with its 41-epoch window; the windows
   of the edge epochs are completed by reflection. Stability check: on the epochs that have a full
   window, the codes are compared with the ones obtained when the same epoch sits at the start of
   the recording (window reflected at distance ``d``).
2. **Token labels.** A P token (0.5 s) is an event token if it overlaps an expert event
   (expert 1 = ``vis1``; ``vis2`` for the comparison).
3. **Single elements.** Enrichment of every level-1 P code (and of level-1+2 prefixes, and of
   consecutive level-1 bigrams): P(event | code) against the base rate, one-sided Fisher test,
   Benjamini-Hochberg.
4. **Detectors, leave-one-excerpt-out.** Logistic models on token features:
   ``P1`` (the token's level-1 code), ``P1x3`` (codes of the token and its two neighbours),
   ``P1x3+pairs`` (plus hashed bigrams: co-occurrence), ``P12`` (level 1 and 2 of the stack),
   ``…+E`` (plus the epoch's E level-1 code, the context), ``latent`` (the quantized P latents of
   3 tokens and E, continuous), ``signal`` (band powers and slow-wave peak-to-peak in ±1 s, the
   filter baseline), and the rule-based ``filter`` detectors of
   :mod:`physioex.train.bin.vq_explain_n2`. Threshold picked by inner group CV on the training
   excerpts. Token average precision and one-to-one event precision / recall / F1, bootstrap CIs over excerpts.
5. **DCSM check (label-free domain).** Stage distribution of the enriched codes in DCSM test nights.

``python -m physioex.train.bin.vq_event_codes --run VQ_RUN --vq N8E8P6 --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_explain_n2 import _bp, kcomplexes, spindles

FS = 100
TOK_S = 0.5
KINDS = {"kc_": "k_complex", "sp_": "spindle"}
STAGES = ("W", "N1", "N2", "N3", "REM")
NCODE = 4096
HASH = 1 << 16


# ---------------------------------------------------------------------------- encoding
def reflect_index(n: int, half: int) -> np.ndarray:
    i = np.arange(-half, n + half)
    i = np.abs(i)
    return np.where(i >= n, 2 * (n - 1) - i, i)


def encode_windows(model, x: np.ndarray, idx: np.ndarray, starts, seq: int, batch: int = 64):
    """Codes and quantized latents of the centre epoch of the windows ``x[idx[s:s+seq]]``."""
    import torch

    dev = next(model.parameters()).device
    out = {"N": [], "E": [], "P": [], "zE": [], "zP": []}
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            w = np.stack([x[idx[s:s + seq]] for s in starts[i:i + batch]])
            z = model.encode(torch.from_numpy(w)[:, :, None].to(dev))
            for k in ("N", "E", "P"):
                out[k].append(z["codes"][k].cpu().numpy().astype(np.int16))
            out["zE"].append(z["E"].float().cpu().numpy())
            out["zP"].append(z["P"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def encode_recording(model, x: np.ndarray, seq: int) -> dict:
    half = seq // 2
    idx = reflect_index(len(x), half)
    return encode_windows(model, x, idx, list(range(len(x))), seq)


def stability(model, x: np.ndarray, seq: int, full: dict, ds=(0, 5, 10)) -> Dict[str, Dict[str, float]]:
    """Code agreement (level 1) between the full window and the window of the same epoch placed
    ``d`` epochs from the start of a recording that begins there (reflected context)."""
    half, n = seq // 2, len(x)
    cent = list(range(half, n - half))
    res = {}
    for d in ds:
        agree = {"N": [], "E": [], "P": []}
        for t in cent:
            sub = x[t - d:]
            idx = reflect_index(len(sub), half)
            c = encode_windows(model, sub, idx, [d], seq)
            for k in agree:
                a, b = c[k][0], full[k][t]
                agree[k].append(float(np.mean(a[..., 0] == b[..., 0])))
        res[str(d)] = {k: float(np.mean(v)) for k, v in agree.items()}
    return res


# ---------------------------------------------------------------------------- labels and features
def token_labels(events, kind, concept, n_ep, T) -> np.ndarray:
    lab = np.zeros(n_ep * T, bool)
    for e in events:
        if e.type != kind or e.concept != concept:
            continue
        a = int(np.floor(e.onset_sec / TOK_S))
        b = int(np.ceil((e.onset_sec + e.duration_sec) / TOK_S))
        lab[max(0, a):min(len(lab), b)] = True
    return lab


def signal_features(x: np.ndarray, T: int) -> np.ndarray:
    """Per token: log power of delta/theta/alpha/sigma/beta and slow-wave peak-to-peak and minimum
    over the token and its two neighbours on each side (±1 s)."""
    flat = x.reshape(-1).astype(np.float64)
    seg = 3000 // T
    feats = []
    for lo, hi in ((0.3, 4), (4, 8), (8, 11), (11, 16), (16, 30)):
        b = _bp(flat, lo, hi).reshape(-1, seg)
        feats.append(np.log((b ** 2).mean(1) + 1e-8))
    s = _bp(flat, 0.3, 4.0).reshape(-1, seg)
    mx, mn = s.max(1), s.min(1)
    k = np.ones(5) / 5
    out = [np.convolve(f, k, "same") for f in feats]
    pad = lambda v, fn: np.array([fn(v[max(0, i - 2):i + 3]) for i in range(len(v))])  # noqa: E731
    out += [pad(mx, np.max) - pad(mn, np.min), pad(mn, np.min)]
    return np.stack(out, 1)


def sparse_features(rec: dict, kind: str):
    """Sparse one-hot blocks for every feature set, rows = tokens of the recording."""
    from scipy import sparse

    P = rec["codes"]["P"]  # (n, T, k)
    n, T, _ = P.shape
    c1 = P[:, :, 0].reshape(-1).astype(np.int64)
    c2 = P[:, :, 1].reshape(-1).astype(np.int64)
    e1 = np.repeat(rec["codes"]["E"][:, 0].astype(np.int64), T)
    m = len(c1)
    prev = np.r_[c1[:1], c1[:-1]]
    nxt = np.r_[c1[1:], c1[-1:]]
    rows = np.arange(m)

    def onehot(cols, width):
        return sparse.csr_matrix((np.ones(m, np.float32), (rows, cols)), shape=(m, width))

    pair = (prev * NCODE + c1) % HASH, (c1 * NCODE + nxt) % HASH
    blocks = {
        "P1": onehot(c1, NCODE),
        "P1prev": onehot(prev, NCODE), "P1next": onehot(nxt, NCODE),
        "pairs": onehot(pair[0], HASH) + onehot(pair[1], HASH),
        "P2": onehot(c2, NCODE),
        "P12": onehot((c1 * NCODE + c2) % HASH, HASH),
        "E1": onehot(e1, NCODE),
    }
    return blocks


FEATURE_SETS = {
    "P1": ["P1"],
    "P1x3": ["P1", "P1prev", "P1next"],
    "P1x3+pairs": ["P1", "P1prev", "P1next", "pairs"],
    "P12": ["P1", "P2", "P12"],
    "P1x3+pairs+E": ["P1", "P1prev", "P1next", "pairs", "E1"],
    "E": ["E1"],
}


def latent_features(rec: dict) -> np.ndarray:
    zP = rec["codes"]["zP"]  # (n, T, d_p)
    n, T, d = zP.shape
    f = zP.reshape(-1, d)
    prev = np.r_[f[:1], f[:-1]]
    nxt = np.r_[f[1:], f[-1:]]
    zE = np.repeat(rec["codes"]["zE"], T, axis=0)
    return np.concatenate([prev, f, nxt, zE], 1)


# ---------------------------------------------------------------------------- evaluation
def runs_to_events(pred: np.ndarray, gap: int = 1) -> List[tuple]:
    """Positive tokens -> events (s), merging gaps of at most ``gap`` tokens."""
    idx = np.flatnonzero(pred)
    if not len(idx):
        return []
    ev, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b <= gap + 1:
            b = i
        else:
            ev.append((a * TOK_S, (b + 1) * TOK_S)); a = b = i
    ev.append((a * TOK_S, (b + 1) * TOK_S))
    return ev


def match_one_to_one(pred: list, ref: list) -> int:
    """Greedy one-to-one matching by overlap (s): a predicted event can claim one reference event
    and vice versa, so a long prediction spanning many events counts once. Returns the matches."""
    pairs = []
    for i, (a, b) in enumerate(pred):
        for j, (c, d) in enumerate(ref):
            ov = min(b, d) - max(a, c)
            if ov > 0:
                pairs.append((ov, i, j))
    pairs.sort(reverse=True)
    used_p, used_r, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_r:
            continue
        used_p.add(i); used_r.add(j); tp += 1
    return tp


def event_scores(preds: Dict[str, np.ndarray], refs: Dict[str, list], events: Dict[str, list] = None):
    """Event precision / recall / F1 pooled over recordings, one-to-one matching. ``preds`` are
    token masks (converted to events) unless ``events`` gives the predicted events directly."""
    tp = n_p = n_r = 0
    for rid in preds if events is None else events:
        pe = runs_to_events(preds[rid]) if events is None else events[rid]
        tp += match_one_to_one(pe, refs[rid]); n_p += len(pe); n_r += len(refs[rid])
    pr = tp / n_p if n_p else 0.0
    rc = tp / n_r if n_r else 0.0
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "n_pred": n_p, "n_ref": n_r, "tp": tp}


def best_threshold(scores, refs, order) -> float:
    qs = np.unique(np.quantile(np.concatenate([scores[r] for r in order]), np.linspace(0.5, 0.995, 60)))
    best, bt = -1.0, qs[-1]
    for q in qs:
        f = event_scores({r: scores[r] >= q for r in order}, {r: refs[r] for r in order})["f1"]
        if f > best:
            best, bt = f, q
    return float(bt)


def evaluate_scores(scores: Dict[str, np.ndarray], y: Dict[str, np.ndarray], refs: Dict[str, list],
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Token AP and one-to-one event F1 of out-of-fold token scores. The threshold of a recording
    is the one with the best F1 on the other recordings' scores. Bootstrap over recordings."""
    from sklearn.metrics import average_precision_score

    ids = list(scores)
    thr = {r: best_threshold(scores, refs, [o for o in ids if o != r]) for r in ids}
    masks = {r: scores[r] >= thr[r] for r in ids}

    def stats(sel):
        yy = np.concatenate([y[r] for r in sel]); ss = np.concatenate([scores[r] for r in sel])
        ap = float(average_precision_score(yy, ss)) if yy.any() else 0.0
        ev = event_scores({r: masks[r] for r in sel}, refs)
        return ap, ev["f1"]

    ap, f1 = stats(ids)
    rng = np.random.default_rng(seed)
    boots = np.array([stats([ids[i] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"token_ap": ap, "token_ap_ci": [float(lo[0]), float(hi[0])],
            "token_base_rate": float(np.concatenate([y[r] for r in ids]).mean()),
            "vis1": event_scores(masks, refs), "f1_ci": [float(lo[1]), float(hi[1])],
            "threshold_median": float(np.median(list(thr.values())))}


def fit_predict(Xtr, ytr, Xte, dense: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if dense:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    clf = LogisticRegression(l1_ratio=1.0, C=0.5, solver="liblinear", class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def loo(recs: List[dict], feats: Dict[str, object], y: Dict[str, np.ndarray], refs: Dict[str, list], dense: bool):
    """Leave-one-excerpt-out token scores, then :func:`evaluate_scores`."""
    from scipy import sparse

    ids = [r["id"] for r in recs]
    stack = (lambda L: np.concatenate(L)) if dense else (lambda L: sparse.vstack(L).tocsr())  # noqa: E731
    out_scores = {}
    for te in ids:
        tr = [r for r in ids if r != te]
        out_scores[te] = fit_predict(stack([feats[r] for r in tr]), np.concatenate([y[r] for r in tr]), feats[te], dense)
    return evaluate_scores(out_scores, y, refs), out_scores, None


def filter_events(recs: List[dict], kind: str, kc_k: float) -> Dict[str, list]:
    """Events (s) of the rule-based detectors of ``vq_explain_n2`` (no training)."""
    out = {}
    for r in recs:
        x = r["x"]
        if kind == "k_complex":
            n23 = np.flatnonzero(np.isin(r["y"], (2, 3)))
            ref = float(np.median(np.ptp(_bp(x[n23], 0.3, 4.0), axis=1))) / 2 if len(n23) else 1.0
            out[r["id"]] = [((ep * 3000 + s) / FS, (ep * 3000 + e) / FS) for ep in n23 for s, e in kcomplexes(x[ep], ref, kc_k)]
        else:
            out[r["id"]] = [(s / FS, e / FS) for s, e in spindles(x.reshape(-1))]
    return out


def expert_refs(recs: List[dict], kind: str, concept: str) -> Dict[str, list]:
    return {r["id"]: [(e.onset_sec, e.onset_sec + e.duration_sec) for e in r["events"] if e.type == kind and e.concept == concept]
            for r in recs}


def ceilings(det: dict, refs: Dict[str, list], refs2: Dict[str, list], scores_all: Dict[str, dict],
             y: Dict[str, np.ndarray], fev: Dict[str, list]) -> dict:
    """Expert 2 against expert 1 on the excerpts that have both, and every detector restricted to
    those excerpts, against expert 1 and against expert 2."""
    both = [rid for rid in refs2 if refs2[rid]]
    if not both:
        return {}
    out = {"excerpts": both, "expert2_vs_expert1": event_scores({}, refs, events={r: refs2[r] for r in both}),
           "expert1_vs_expert2": event_scores({}, refs2, events={r: refs[r] for r in both}), "detectors": {}}
    sub = lambda d: {r: d[r] for r in both}  # noqa: E731
    for name, sc in scores_all.items():
        out["detectors"][name] = {"vs_expert1": evaluate_scores(sub(sc), sub(y), refs, n_boot=200),
                                  "vs_expert2": evaluate_scores(sub(sc), {r: token_labels_from(refs2[r], len(y[r])) for r in both}, refs2, n_boot=200)}
    out["detectors"]["filter"] = {"vs_expert1": event_scores({}, refs, events=sub(fev)), "vs_expert2": event_scores({}, refs2, events=sub(fev))}
    return out


def token_labels_from(ref: list, n: int) -> np.ndarray:
    lab = np.zeros(n, bool)
    for a, b in ref:
        lab[max(0, int(np.floor(a / TOK_S))):min(n, int(np.ceil(b / TOK_S)))] = True
    return lab


def enrichment(recs, y, key_fn, min_n=20):
    """Enrichment of discrete token keys: rows (key, n, positives, precision, lift, p, q)."""
    from scipy.stats import fisher_exact

    keys = np.concatenate([key_fn(r) for r in recs])
    lab = np.concatenate([y[r["id"]] for r in recs])
    base = lab.mean()
    P, N = lab.sum(), len(lab)
    u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=lab, minlength=len(u))
    rows = []
    for i in np.flatnonzero(cnt >= min_n):
        a, b = int(pos[i]), int(cnt[i] - pos[i])
        p = fisher_exact([[a, b], [int(P - a), int(N - P - b)]], alternative="greater")[1]
        rows.append([int(u[i]), int(cnt[i]), a, a / cnt[i], (a / cnt[i]) / base, p])
    rows.sort(key=lambda r: r[5])
    m = len(rows)
    q = np.minimum.accumulate([r[5] * m / (i + 1) for i, r in enumerate(rows)][::-1])[::-1] if m else []
    for r, qq in zip(rows, q):
        r.append(float(min(1.0, qq)))
    sig = [r for r in rows if r[6] < 0.05 and r[4] > 1]
    cover = float(sum(r[2] for r in sig) / P) if P else 0.0
    return {"base_rate": float(base), "n_tested": m, "n_significant": len(sig), "event_tokens_covered_by_significant": cover,
            "precision_of_significant": float(sum(r[2] for r in sig) / max(1, sum(r[1] for r in sig))),
            "top": [dict(zip(("key", "n", "pos", "precision", "lift", "p", "q"), r)) for r in rows[:25]]}


# ---------------------------------------------------------------------------- rescore
def rescore(a, tds, log):
    """Metrics (one-to-one events, bootstrap, ceilings) from the saved out-of-fold token scores."""
    rep = json.load(open(os.path.join(a.out, "vq_event_codes.json")))
    specs = {s.subject_id: s for s in tds._subjects}
    for pref, kind in KINDS.items():
        z = np.load(os.path.join(a.out, f"scores_{kind}.npz"))
        ids = sorted({k.split("__", 1)[1] for k in z if k.startswith("label__")})
        R = []
        for sid in ids:
            n = tds._n_epochs[sid]
            x = next(iter(tds._build_item(specs[sid], 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
            R.append({"id": sid, "x": x, "y": np.asarray(tds._get_labels(specs[sid]))[:n], "events": tds._load_or_compute_events(specs[sid])})
        y = {sid: z[f"label__{sid}"] for sid in ids}
        refs, refs2 = expert_refs(R, kind, "vis1"), expert_refs(R, kind, "vis2")
        names = sorted({k.split("__", 1)[0] for k in z if not k.startswith("label__")})
        scores_all = {n_: {sid: z[f"{n_}__{sid}"] for sid in ids} for n_ in names}
        det = {n_: evaluate_scores(sc, y, refs) for n_, sc in scores_all.items()}
        for n_, d in det.items():
            log(kind, n_, round(d["token_ap"], 3), d["token_ap_ci"], round(d["vis1"]["f1"], 3), d["f1_ci"])
        fev = filter_events(R, kind, a.kc_k)
        det["filter"] = {"vis1": event_scores({}, refs, events=fev), "k": a.kc_k if kind == "k_complex" else None}
        res = rep["events"][kind]
        res["detectors"] = det
        res["ceiling"] = ceilings(det, refs, refs2, scores_all, y, fev)
        res.pop("expert2_vs_expert1", None)
        log(kind, "filter", det["filter"]["vis1"], "ceiling", res["ceiling"].get("expert2_vs_expert1"))
    rep["rescored"] = "one-to-one event matching, bootstrap over excerpts"
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_codes.json"), "w"), indent=1)
    log("DONE")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--kc-k", type=float, default=2.0, help="threshold of the filter K-complex detector")
    ap.add_argument("--no-dcsm", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute the metrics from DIR/scores_*.npz (no GPU)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_codes]", *s, flush=True)  # noqa: E731
    run, dcsm_ds, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    seq = ra["seq"]
    tds = get_dataset(a.target)(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                sequence_length=1, cache_dir=ra.get("cache_dir"))
    if a.rescore:
        rescore(a, tds, log)
        return
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    T = model.n_tok
    recs = []
    t0 = time.time()
    for spec in tds._subjects:
        sid = spec.subject_id
        n = tds._n_epochs[sid]
        x = next(iter(tds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        recs.append({"id": sid, "x": x, "y": np.asarray(tds._get_labels(spec))[:n],
                     "events": tds._load_or_compute_events(spec), "codes": encode_recording(model, x, seq)})
    log("encoded", len(recs), "excerpts", f"{time.time() - t0:.0f}s")
    rep = {"run": a.run, "vq": a.vq, "seq": seq, "n_tok": T, "n_codes": NCODE,
           "stability": {r["id"]: stability(model, r["x"], seq, r["codes"]) for r in recs[:3]}}
    log("stability", rep["stability"])
    np.savez_compressed(os.path.join(a.out, "dreams_codes.npz"),
                        **{f"{r['id']}_{k}": r["codes"][k] for r in recs for k in ("N", "E", "P")})

    rep["events"] = {}
    for pref, kind in KINDS.items():
        R = [r for r in recs if r["id"].startswith(pref)]
        y = {r["id"]: token_labels(r["events"], kind, "vis1", len(r["x"]), T) for r in R}
        refs, refs2 = expert_refs(R, kind, "vis1"), expert_refs(R, kind, "vis2")
        res = {"n_excerpts": len(R), "n_events_vis1": int(sum(len(v) for v in refs.values())),
               "n_tokens": int(sum(len(v) for v in y.values())), "n_event_tokens": int(sum(v.sum() for v in y.values()))}
        # stage of event tokens (description only)
        st = np.concatenate([np.repeat(r["y"], T)[y[r["id"]]] for r in R])
        res["event_token_stages"] = {s: int((st == i).sum()) for i, s in enumerate(STAGES)}
        # 3. enrichment
        res["enrichment"] = {
            "P1": enrichment(R, y, lambda r: r["codes"]["P"][:, :, 0].reshape(-1)),
            "P12": enrichment(R, y, lambda r: r["codes"]["P"][:, :, 0].reshape(-1).astype(np.int64) * NCODE + r["codes"]["P"][:, :, 1].reshape(-1), min_n=10),
            "P1_bigram": enrichment(R, y, lambda r: (lambda c: c * NCODE + np.r_[c[1:], c[-1:]])(r["codes"]["P"][:, :, 0].reshape(-1).astype(np.int64)), min_n=10),
            "E1": enrichment(R, y, lambda r: np.repeat(r["codes"]["E"][:, 0], T), min_n=T),
        }
        log(kind, "enrichment", {k: (v["n_significant"], round(v["event_tokens_covered_by_significant"], 3)) for k, v in res["enrichment"].items()})
        # 4. detectors
        blocks = {r["id"]: sparse_features(r, kind) for r in R}
        from scipy import sparse
        det = {}
        scores_all = {}
        for name, parts in FEATURE_SETS.items():
            feats = {rid: sparse.hstack([b[p] for p in parts]).tocsr() for rid, b in blocks.items()}
            det[name], scores_all[name], _ = loo(R, feats, y, refs, dense=False)
            log(kind, name, det[name]["token_ap"], det[name]["vis1"]["f1"])
        for name, ff in (("latent", latent_features), ("signal", lambda r: signal_features(r["x"], T))):
            feats = {r["id"]: ff(r) for r in R}
            det[name], scores_all[name], _ = loo(R, feats, y, refs, dense=True)
            log(kind, name, det[name]["token_ap"], det[name]["vis1"]["f1"])
        fev = filter_events(R, kind, a.kc_k)
        det["filter"] = {"vis1": event_scores({}, refs, events=fev), "k": a.kc_k if kind == "k_complex" else None}
        res["ceiling"] = ceilings(det, refs, refs2, scores_all, y, fev)
        res["detectors"] = det
        rep["events"][kind] = res
        np.savez_compressed(os.path.join(a.out, f"scores_{kind}.npz"),
                            **{f"{n}__{rid}": v for n, d in scores_all.items() for rid, v in d.items()},
                            **{f"label__{rid}": v for rid, v in y.items()})

    # 5. DCSM: stage distribution of the significant level-1 codes
    if not a.no_dcsm:
        sig = {kind: [t["key"] for t in rep["events"][kind]["enrichment"]["P1"]["top"] if t["q"] < 0.05 and t["lift"] > 1]
               for kind in rep["events"]}
        cnt = np.zeros((NCODE, 5), np.int64)
        tot = np.zeros(5, np.int64)
        have = {s.subject_id: s for s in dcsm_ds._subjects}
        for sid in run["splits"]["test"]:
            if sid not in have:
                continue
            n = dcsm_ds._n_epochs[sid]
            x = next(iter(dcsm_ds._build_item(have[sid], 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
            yy = np.asarray(dcsm_ds._get_labels(have[sid]))[:n]
            c = encode_recording(model, x, seq)["P"][:, :, 0]
            for i in range(5):
                v = c[yy == i].reshape(-1)
                cnt[:, i] += np.bincount(v, minlength=NCODE)
                tot[i] += v.size
        rep["dcsm_stage_mix_all_tokens"] = (tot / tot.sum()).tolist()
        rep["dcsm_code_stages"] = {kind: {str(k): {"n": int(cnt[k].sum()), "mix": (cnt[k] / max(1, cnt[k].sum())).tolist()} for k in ks}
                                   for kind, ks in sig.items()}
        log("dcsm done")
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_codes.json"), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
