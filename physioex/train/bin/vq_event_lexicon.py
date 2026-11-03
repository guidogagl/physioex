"""Iteration S2 of the explainability workflow: a directional lexicon on the P latent (R1).

The k-means lexicons of :mod:`vq_event_readout` give no pure word: an event is a direction of the
latent, not a compact region. Here the lexicon is a label-free basis of the quantized P latent
fitted on DCSM (PCA and FastICA, one basis per token and one per 3-token phrase); a word is
(direction, quantile bin of the projection), 8 bins per direction from the DCSM quantiles. Every
token therefore carries one word per direction; its event score is the naive-Bayes sum of the
log-lifts of its words, lifts estimated leave-one-excerpt-out on DREAMS (the labels only name
and score the words, never shape the basis).

Reported, per event type and basis: token AP and one-to-one event F1 (same battery as
``vq_event_codes``), the direction-bins with the highest purity (lift, coverage, DCSM stage mix),
and the upper bound of the same representation read by a logistic probe (``p3`` / ``pq3`` of
``vq_event_readout``). The DCSM representations are saved for later iterations.

``python -m physioex.train.bin.vq_event_lexicon --run VQ_RUN --vq N8E8P6 --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_event_codes import KINDS, evaluate_scores, expert_refs, loo, token_labels
from physioex.train.bin.vq_event_readout import ctx3, readout_dcsm, readout_recording

NBIN = 8


def fit_basis(X: np.ndarray, kind: str, n_comp: int, seed: int = 0):
    from sklearn.decomposition import PCA, FastICA

    if kind == "pca":
        return PCA(n_comp, random_state=seed).fit(X)
    return FastICA(n_comp, whiten="unit-variance", random_state=seed, max_iter=500).fit(X)


def bin_edges(Z: np.ndarray) -> np.ndarray:
    """``(n, D) -> (D, NBIN-1)`` DCSM quantile edges per direction."""
    return np.quantile(Z, np.linspace(0, 1, NBIN + 1)[1:-1], axis=0).T


def words_of(Z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``(n, D) -> (n, D)`` bin index per direction."""
    return np.stack([np.searchsorted(edges[d], Z[:, d]) for d in range(Z.shape[1])], 1)


def nb_scores_loo(W: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Token score = sum over directions of log P(event | word) / P(event), estimated on the other excerpts."""
    ids = list(W)
    D = W[ids[0]].shape[1]
    cnt = {}
    for r in ids:
        n = np.zeros((D, NBIN)); p = np.zeros((D, NBIN))
        for d in range(D):
            n[d] = np.bincount(W[r][:, d], minlength=NBIN); p[d] = np.bincount(W[r][:, d], weights=y[r], minlength=NBIN)
        cnt[r] = (n, p, y[r].sum(), len(y[r]))
    out = {}
    for te in ids:
        n = sum(cnt[r][0] for r in ids if r != te); p = sum(cnt[r][1] for r in ids if r != te)
        base = sum(cnt[r][2] for r in ids if r != te) / sum(cnt[r][3] for r in ids if r != te)
        lift = np.log(((p + 0.5) / (n + 1.0)) / base)
        out[te] = sum(lift[d][W[te][:, d]] for d in range(D))
    return out


def word_table(W: Dict[str, np.ndarray], y: Dict[str, np.ndarray], Wd: np.ndarray, dstage: np.ndarray, top: int = 10) -> dict:
    w = np.concatenate([W[r] for r in W]); lab = np.concatenate([y[r] for r in W])
    base, P = lab.mean(), lab.sum()
    D = w.shape[1]
    rows = []
    for d in range(D):
        n = np.bincount(w[:, d], minlength=NBIN); p = np.bincount(w[:, d], weights=lab, minlength=NBIN)
        for b in range(NBIN):
            if n[b] >= 20:
                sel = Wd[:, d] == b
                rows.append({"direction": d, "bin": b, "n": int(n[b]), "event_tokens": int(p[b]), "purity": float(p[b] / n[b]),
                             "lift": float(p[b] / n[b] / base), "share_of_event_tokens": float(p[b] / P),
                             "dcsm_stage_mix": [float((dstage[sel] == s).mean()) if sel.any() else None for s in range(5)]})
    rows.sort(key=lambda r: -r["purity"])
    return {"base_rate": float(base), "n_pure": sum(r["purity"] >= 0.5 for r in rows),
            "coverage_pure": float(sum(r["event_tokens"] for r in rows if r["purity"] >= 0.5) / P) if P else None, "top": rows[:top]}


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

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_lexicon]", *s, flush=True)  # noqa: E731
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
    dc, dstage = readout_dcsm(model, dcsm_ds, list(run["splits"]["train"])[:a.dcsm_nights], a.per_night, seq)
    np.savez_compressed(os.path.join(a.out, "dcsm_readout.npz"), stage=dstage, **{k: dc[k] for k in ("h", "p", "pq", "c1", "zE")})
    log("read", len(recs), "excerpts +", dc["pq"].shape, "DCSM epochs", f"{time.time() - t0:.0f}s")
    dstage_tok = np.repeat(dstage, T)
    rep = {"run": a.run, "vq": a.vq, "n_dcsm_epochs": int(len(dstage)), "bins": NBIN, "events": {}}

    reps = {"pq1": lambda r: r["pq"].reshape(-1, r["pq"].shape[-1]).astype(np.float32), "pq3": lambda r: ctx3(r["pq"])}
    bases = {}
    for rname, fn in reps.items():
        Xd = fn(dc)
        for bname in ("pca", "ica"):
            B = fit_basis(Xd, bname, Xd.shape[1] if rname == "pq1" else 16)
            Zd = B.transform(Xd)
            bases[f"{rname}_{bname}"] = (B, bin_edges(Zd), words_of(Zd, bin_edges(Zd)), fn)
            if bname == "pca":
                rep[f"explained_variance_{rname}"] = B.explained_variance_ratio_.tolist()
    for pref, kind in KINDS.items():
        R = [r for r in recs if r["id"].startswith(pref)]
        y = {r["id"]: token_labels(r["events"], kind, "vis1", len(r["y"]), T) for r in R}
        refs = expert_refs(R, kind, "vis1")
        res = {"upper_bound_probe": {}, "lexicon": {}}
        for rname, fn in reps.items():
            res["upper_bound_probe"][rname], _, _ = loo(R, {r["id"]: fn(r) for r in R}, y, refs, dense=True)
        for name, (B, edges, Wd, fn) in bases.items():
            W = {r["id"]: words_of(B.transform(fn(r)), edges) for r in R}
            ev = evaluate_scores(nb_scores_loo(W, y), y, refs, n_boot=300)
            res["lexicon"][name] = {**ev, "table": word_table(W, y, Wd, dstage_tok)}
            log(kind, name, "AP", round(ev["token_ap"], 3), ev["token_ap_ci"], "F1", round(ev["vis1"]["f1"], 3),
                "pure", res["lexicon"][name]["table"]["n_pure"], "cov", res["lexicon"][name]["table"]["coverage_pure"])
        rep["events"][kind] = res
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_lexicon.json"), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
