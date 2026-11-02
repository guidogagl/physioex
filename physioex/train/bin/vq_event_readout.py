"""Where does the MSAE-VQ EEG encoder lose the expert micro-events, and can a small label-free
dictionary read them?

Follows :mod:`physioex.train.bin.vq_event_codes` (no code, stack or co-occurrence of indices
corresponds to DREAMS K-complexes or spindles; the continuous latent does, for K-complexes).

1. **Representations** of every token: ``h`` (token features before FiLM, after the epoch
   transformer, 128), ``p`` (continuous P latent after FiLM and tanh, before the VQ, d_p),
   ``p0`` (the same with N and E set to zero: FiLM neutral), ``pq`` (quantized P latent, eval
   prefix), ``c1`` (level-1 code), ``zE`` (quantized E). DREAMS: all epochs (reflected windows).
   DCSM: random full-window epochs of training nights, label-free.
2. **Chain probes** on DREAMS, leave-one-excerpt-out, on windows of 3 tokens: ``h3``, ``p3``,
   ``p03``, ``pq3``, ``pq3+E`` (= the ``latent`` probe of ``vq_event_codes``, consistency check),
   ``pq1``. Signal -> h -> p -> pq says which part loses each event type.
3. **Dictionaries** fitted on the DCSM sample with k-means (k in 32..256), never on DREAMS:
   (a) word = ``pq`` of one token (a grouping of the original codes: the codebook vectors map to
   words), (b) phrase = ``pq`` of 3 consecutive tokens, (c) word on ``h``, (d) phrase on ``h``.
   Read on DREAMS with a lookup table P(event | word) estimated leave-one-excerpt-out: token AP,
   one-to-one event F1, per word purity / coverage / lift, stage mix of the word in DCSM, the
   codes it groups, and the word trigrams most frequent around the events.

``python -m physioex.train.bin.vq_event_readout --run VQ_RUN --vq N8E8P6 --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Dict, List

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_event_codes import (KINDS, NCODE, STAGES, evaluate_scores, expert_refs, loo, reflect_index,
                                                token_labels)

KS = (32, 64, 128, 256)


# ---------------------------------------------------------------------------- representations
def readout_batch(model, w: np.ndarray) -> Dict[str, np.ndarray]:
    """Replicates ``MSAEVQ.encode`` and keeps the intermediate representations of the centre epoch."""
    import torch

    dev = next(model.parameters()).device
    x = torch.from_numpy(w).to(dev).float()  # (B, L, 3000)
    with torch.no_grad():
        n = model.encode_context(x)
        if model.use_n:
            n, cn, _ = model._q("N", n)
        h = model.features(x[:, model.c])
        g, b = model.e_film(n).chunk(2, -1)
        he = h * (1 + g[:, None]) + b[:, None]
        e = torch.tanh(model.e_out(model.e_pool(model.e_query.expand(len(h), -1, -1), he, he, need_weights=False)[0][:, 0]))
        e, ce, _ = model._q("E", e)
        g, b = model.p_film(torch.cat([n, e], -1)).chunk(2, -1)
        p = torch.tanh(model.p_out(h * (1 + g[:, None]) + b[:, None]))
        g0, b0 = model.p_film(torch.zeros_like(torch.cat([n, e], -1))).chunk(2, -1)
        p0 = torch.tanh(model.p_out(h * (1 + g0[:, None]) + b0[:, None]))
        pq, cp, _ = model._q("P", p)
    f16 = lambda t: t.float().cpu().numpy().astype(np.float16)  # noqa: E731
    return {"h": f16(h), "p": f16(p), "p0": f16(p0), "pq": f16(pq), "zE": f16(e),
            "c1": cp[..., 0].cpu().numpy().astype(np.int16)}


def readout_recording(model, x: np.ndarray, seq: int, batch: int = 64) -> Dict[str, np.ndarray]:
    half = seq // 2
    idx = reflect_index(len(x), half)
    out = {}
    for i in range(0, len(x), batch):
        w = np.stack([x[idx[s:s + seq]] for s in range(i, min(len(x), i + batch))])
        for k, v in readout_batch(model, w).items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def readout_dcsm(model, ds, ids: List[str], per_night: int, seq: int, seed: int = 0):
    """Random full-window epochs of the DCSM nights; stage of every epoch kept for description only."""
    rng = np.random.default_rng(seed)
    have = {s.subject_id: s for s in ds._subjects}
    half = seq // 2
    out, stages = {}, []
    for sid in ids:
        if sid not in have:
            continue
        n = ds._n_epochs[sid]
        lo, hi = ds._main_period.get(sid, (0, n)) if hasattr(ds, "_main_period") else (0, n)
        lo, hi = max(lo, half), min(hi, n - half)
        if hi <= lo:
            continue
        x = next(iter(ds._build_item(have[sid], 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        y = np.asarray(ds._get_labels(have[sid]))[:n]
        ts = np.sort(rng.choice(np.arange(lo, hi), min(per_night, hi - lo), replace=False))
        for i in range(0, len(ts), 64):
            w = np.stack([x[t - half:t + half + 1] for t in ts[i:i + 64]])
            for k, v in readout_batch(model, w).items():
                out.setdefault(k, []).append(v)
        stages.append(y[ts])
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(stages)


# ---------------------------------------------------------------------------- features
def ctx3(f: np.ndarray) -> np.ndarray:
    """``(n, T, d) -> (n*T, 3d)``: previous, current, next token (edges repeated)."""
    n, T, d = f.shape
    v = f.reshape(-1, d).astype(np.float32)
    return np.concatenate([np.r_[v[:1], v[:-1]], v, np.r_[v[1:], v[-1:]]], 1)


PROBES = {
    "h3": lambda r: ctx3(r["h"]),
    "p3": lambda r: ctx3(r["p"]),
    "p03": lambda r: ctx3(r["p0"]),
    "pq3": lambda r: ctx3(r["pq"]),
    "pq3+E": lambda r: np.concatenate([ctx3(r["pq"]), np.repeat(r["zE"].astype(np.float32), r["pq"].shape[1], axis=0)], 1),
    "pq1": lambda r: r["pq"].reshape(-1, r["pq"].shape[-1]).astype(np.float32),
    "h1": lambda r: r["h"].reshape(-1, r["h"].shape[-1]).astype(np.float32),
}

DICTS = {"pq_word": ("pq", 1), "pq_phrase": ("pq", 3), "h_word": ("h", 1), "h_phrase": ("h", 3)}


def dict_features(rep: Dict[str, np.ndarray], key: str, span: int) -> np.ndarray:
    return ctx3(rep[key]) if span == 3 else rep[key].reshape(-1, rep[key].shape[-1]).astype(np.float32)


# ---------------------------------------------------------------------------- dictionaries
def lookup_loo(words: Dict[str, np.ndarray], y: Dict[str, np.ndarray], k: int) -> Dict[str, np.ndarray]:
    """Token score = P(event | word) estimated on the other excerpts (Laplace 0.5)."""
    ids = list(words)
    cnt = {r: (np.bincount(words[r], minlength=k), np.bincount(words[r], weights=y[r], minlength=k)) for r in ids}
    out = {}
    for te in ids:
        n = sum(cnt[r][0] for r in ids if r != te); p = sum(cnt[r][1] for r in ids if r != te)
        out[te] = ((p + 0.5) / (n + 1.0))[words[te]]
    return out


def word_table(words: Dict[str, np.ndarray], y: Dict[str, np.ndarray], k: int, dcsm_words: np.ndarray,
               dcsm_stage: np.ndarray, dcsm_c1: np.ndarray = None, top: int = 12) -> dict:
    """Per word (pooled DREAMS, description): n, event tokens, purity, lift, share of the event
    tokens; stage mix of the word in DCSM; how many original level-1 codes it groups."""
    w = np.concatenate([words[r] for r in words]); lab = np.concatenate([y[r] for r in words])
    n = np.bincount(w, minlength=k); pos = np.bincount(w, weights=lab, minlength=k)
    base = lab.mean(); P = lab.sum()
    purity = pos / np.maximum(n, 1)
    order = np.argsort(-purity * (n >= 10))
    dn = np.bincount(dcsm_words, minlength=k)
    rows = []
    for i in order[:top]:
        row = {"word": int(i), "n": int(n[i]), "event_tokens": int(pos[i]), "purity": float(purity[i]),
               "lift": float(purity[i] / base) if base else None, "share_of_event_tokens": float(pos[i] / P) if P else None,
               "dcsm_share": float(dn[i] / dn.sum()) if dn.sum() else None,
               "dcsm_stage_mix": [float((dcsm_stage[dcsm_words == i] == s).mean()) if dn[i] else None for s in range(5)]}
        if dcsm_c1 is not None:
            row["n_codes_grouped"] = int(len(np.unique(dcsm_c1[dcsm_words == i])))
        rows.append(row)
    pure = purity >= 0.5
    return {"base_rate": float(base), "n_words_pure": int((pure & (n >= 10)).sum()),
            "coverage_pure": float(pos[pure & (n >= 10)].sum() / P) if P else None, "top": rows}


def trigrams(words: Dict[str, np.ndarray], y: Dict[str, np.ndarray], refs: Dict[str, list], top: int = 10) -> list:
    """Most frequent word trigrams (t-1, t, t+1) centred on the middle token of the expert events."""
    c = Counter()
    for r, w in words.items():
        for a, b in refs[r]:
            t = int(((a + b) / 2) // 0.5)
            if 1 <= t < len(w) - 1:
                c[(int(w[t - 1]), int(w[t]), int(w[t + 1]))] += 1
    tot = sum(c.values())
    return [{"trigram": list(k), "n": v, "share": v / tot} for k, v in c.most_common(top)] if tot else []


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--dcsm-nights", type=int, default=40)
    ap.add_argument("--per-night", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from sklearn.cluster import MiniBatchKMeans

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_readout]", *s, flush=True)  # noqa: E731
    run, dcsm_ds, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    seq, T = ra["seq"], model.n_tok
    tds = get_dataset(a.target)(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                sequence_length=1, cache_dir=ra.get("cache_dir"))
    t0 = time.time()
    recs = []
    for spec in tds._subjects:
        sid = spec.subject_id
        n = tds._n_epochs[sid]
        x = next(iter(tds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        recs.append({"id": sid, "y": np.asarray(tds._get_labels(spec))[:n], "events": tds._load_or_compute_events(spec),
                     **readout_recording(model, x, seq)})
    log("dreams read", len(recs), f"{time.time() - t0:.0f}s")
    np.savez_compressed(os.path.join(a.out, "dreams_readout.npz"),
                        **{f"{r['id']}_{k}": r[k] for r in recs for k in ("h", "p", "p0", "pq", "zE", "c1")})
    dc, dstage = readout_dcsm(model, dcsm_ds, list(run["splits"]["train"])[:a.dcsm_nights], a.per_night, seq)
    log("dcsm read", dc["pq"].shape, f"{time.time() - t0:.0f}s")
    rep = {"run": a.run, "vq": a.vq, "n_tok": T, "d_p": int(dc["pq"].shape[-1]), "n_dcsm_epochs": int(len(dstage)),
           "dcsm_stage_mix": np.bincount(dstage[dstage >= 0], minlength=5).tolist(), "events": {}}
    # FiLM effect: how far p moves from p0 (relative), and codes changed
    d = np.linalg.norm(dc["p"].astype(np.float32) - dc["p0"].astype(np.float32), axis=-1)
    rep["film"] = {"rel_shift_median": float(np.median(d / (np.linalg.norm(dc["p"].astype(np.float32), axis=-1) + 1e-6)))}
    dcsm_stage_tok = np.repeat(dstage, T)
    cb = model.rvq["P"].levels[0].codebook.detach().float().cpu().numpy()

    for pref, kind in KINDS.items():
        R = [r for r in recs if r["id"].startswith(pref)]
        y = {r["id"]: token_labels(r["events"], kind, "vis1", len(r["y"]), T) for r in R}
        refs = expert_refs(R, kind, "vis1")
        res = {"probes": {}, "dictionaries": {}}
        for name, fn in PROBES.items():
            feats = {r["id"]: fn(r) for r in R}
            res["probes"][name], _, _ = loo(R, feats, y, refs, dense=True)
            log(kind, "probe", name, round(res["probes"][name]["token_ap"], 3), round(res["probes"][name]["vis1"]["f1"], 3))
        for dname, (key, span) in DICTS.items():
            X = dict_features(dc, key, span)
            res["dictionaries"][dname] = {}
            for k in KS:
                km = MiniBatchKMeans(k, n_init=3, random_state=0, batch_size=4096).fit(X)
                dw = km.labels_
                words = {r["id"]: km.predict(dict_features(r, key, span)) for r in R}
                ev = evaluate_scores(lookup_loo(words, y, k), y, refs, n_boot=300)
                entry = {**ev, "table": word_table(words, y, k, dw, dcsm_stage_tok, dc["c1"].reshape(-1) if key == "pq" else None),
                         "trigrams": trigrams(words, y, refs)}
                if dname == "pq_word":
                    entry["code_to_word"] = km.predict(cb).tolist()
                res["dictionaries"][dname][str(k)] = entry
                log(kind, dname, k, "AP", round(ev["token_ap"], 3), "F1", round(ev["vis1"]["f1"], 3),
                    "pure", entry["table"]["n_words_pure"], "cov", entry["table"]["coverage_pure"])
        rep["events"][kind] = res
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_readout.json"), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
