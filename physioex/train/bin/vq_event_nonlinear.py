"""S4a diagnostics: is the event information inside the P latent but not linearly readable, or
is it gone?

After S3 (``WORKFLOW_SPIEGABILITA.md``): widening the bottleneck (d_p 16 -> 32 -> 64) does not bring
back the spindles in ``p`` (linear probe 0.05 at every width) although ``h`` carries them (0.29).
Two readings are separated here:

1. **Nonlinear probes** (MLP 2 x 64, leave-one-excerpt-out, same battery as the linear ones) on
   ``h`` x3, ``p`` x3, ``p0`` x3 (FiLM neutral), ``pq`` x3 and ``pq`` x3 + E. If the MLP on ``pq``
   reads the spindles far above the linear probe, the information is there but entangled.
2. **Reconstruction at the events**: the rule detectors of :mod:`vq_explain_n2` run on the DECODED
   signal (N + E + P reconstruction of every DREAMS epoch) against expert 1, next to the same
   detectors on the input; the sigma-envelope correlation input / decoded inside the expert spindles
   (and outside), and the slow-wave peak-to-peak ratio decoded / input inside the expert
   K-complexes. If the decoder puts the spindles back at the right times, the codes carry them.

``python -m physioex.train.bin.vq_event_nonlinear --run VQ_RUN --vq N8E8P6 --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from physioex.train.bin.domain_shift_report import _jsonable
from physioex.train.bin.vq_event_codes import (FS, KINDS, TOK_S, evaluate_scores, event_scores, expert_refs,
                                                filter_events, reflect_index, token_labels)
from physioex.train.bin.vq_explain_n2 import _bp
from physioex.train.bin.vq_event_readout import PROBES, readout_recording


def mlp_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, device: str, epochs: int = 12,
                    hidden: int = 64, seed: int = 0) -> np.ndarray:
    """Two-hidden-layer MLP, class-balanced BCE, no early stopping (fixed budget), standardized inputs."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.from_numpy((Xtr - mu) / sd).float().to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    net = nn.Sequential(nn.Linear(Xt.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    pos = float(yt.sum()); neg = float(len(yt) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0), device=device))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad(); loss_fn(net(Xt[idx])[:, 0], yt[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Xe = torch.from_numpy((Xte - mu) / sd).float().to(device)
        return torch.sigmoid(net(Xe)[:, 0]).cpu().numpy()


def mlp_loo(recs: List[dict], feats: Dict[str, np.ndarray], y: Dict[str, np.ndarray], refs: Dict[str, list], device: str):
    ids = [r["id"] for r in recs]
    out = {}
    for te in ids:
        tr = [r for r in ids if r != te]
        out[te] = mlp_fit_predict(np.concatenate([feats[r] for r in tr]), np.concatenate([y[r] for r in tr]), feats[te], device)
    return evaluate_scores(out, y, refs, n_boot=300)


def decode_recording(model, x: np.ndarray, seq: int, batch: int = 64) -> np.ndarray:
    """N + E + P reconstruction of every epoch (reflected windows), ``(n, 3000)`` in the scaled domain."""
    import torch

    dev = next(model.parameters()).device
    half = seq // 2
    idx = reflect_index(len(x), half)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            w = np.stack([x[idx[s:s + seq]] for s in range(i, min(len(x), i + batch))])
            o = model(torch.from_numpy(w)[:, :, None].to(dev))
            out.append(o["wave"].float().reshape(len(w), -1).cpu().numpy())
    return np.concatenate(out)


def envelope(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import hilbert

    return np.abs(hilbert(_bp(x, lo, hi)))


def event_fidelity(x_in: np.ndarray, x_dec: np.ndarray, refs: List[tuple], kind: str) -> dict:
    """Inside the expert events: sigma-envelope correlation and RMS ratio (spindles) or slow-wave
    peak-to-peak ratio decoded / input (K-complexes); the same quantities outside the events."""
    flat_in, flat_dec = x_in.reshape(-1).astype(np.float64), x_dec.reshape(-1).astype(np.float64)
    if kind == "spindle":
        e_in, e_dec = envelope(flat_in, 11, 16), envelope(flat_dec, 11, 16)
    else:
        e_in, e_dec = _bp(flat_in, 0.3, 4.0), _bp(flat_dec, 0.3, 4.0)
    mask = np.zeros(len(flat_in), bool)
    corr, ratio = [], []
    for a, b in refs:
        a, b = int(a * FS), int(b * FS)
        if b - a < 10 or b > len(flat_in):
            continue
        mask[a:b] = True
        corr.append(float(np.corrcoef(e_in[a:b], e_dec[a:b])[0, 1]))
        if kind == "spindle":
            ratio.append(float(np.sqrt((e_dec[a:b] ** 2).mean() / ((e_in[a:b] ** 2).mean() + 1e-9))))
        else:
            ratio.append(float(np.ptp(e_dec[a:b]) / (np.ptp(e_in[a:b]) + 1e-9)))
    out_ratio = float(np.sqrt((e_dec[~mask] ** 2).mean() / ((e_in[~mask] ** 2).mean() + 1e-9)))
    return {"n": len(corr), "corr_median": float(np.median(corr)) if corr else None,
            "ratio_median_inside": float(np.median(ratio)) if ratio else None, "ratio_rms_outside": out_ratio,
            "wave_corr_inside": float(np.corrcoef(flat_in[mask], flat_dec[mask])[0, 1]) if mask.any() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--kc-k", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, channel_pipelines

    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[vq_event_nonlinear]", *s, flush=True)  # noqa: E731
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run, _, model = _load_run(a.run)
    ra = run["args"]
    model.vq_levels = parse_vq_key(a.vq)
    model.to(dev).eval()
    seq, T = ra["seq"], model.n_tok
    tds = get_dataset(a.target)(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                sequence_length=1, cache_dir=ra.get("cache_dir"))
    t0 = time.time()
    recs = []
    for spec in tds._subjects:
        sid = spec.subject_id
        n = tds._n_epochs[sid]
        x = next(iter(tds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        recs.append({"id": sid, "x": x, "y": np.asarray(tds._get_labels(spec))[:n], "events": tds._load_or_compute_events(spec),
                     "dec": decode_recording(model, x, seq), **readout_recording(model, x, seq)})
    log("read + decoded", len(recs), f"{time.time() - t0:.0f}s")
    rep = {"run": a.run, "vq": a.vq, "d_p": int(recs[0]["pq"].shape[-1]), "events": {}}
    for pref, kind in KINDS.items():
        R = [r for r in recs if r["id"].startswith(pref)]
        y = {r["id"]: token_labels(r["events"], kind, "vis1", len(r["y"]), T) for r in R}
        refs = expert_refs(R, kind, "vis1")
        res = {"mlp_probes": {}, "reconstruction": {}}
        for name in ("h3", "p3", "p03", "pq3", "pq3+E"):
            res["mlp_probes"][name] = mlp_loo(R, {r["id"]: PROBES[name](r) for r in R}, y, refs, dev)
            log(kind, "mlp", name, round(res["mlp_probes"][name]["token_ap"], 3), res["mlp_probes"][name]["token_ap_ci"],
                round(res["mlp_probes"][name]["vis1"]["f1"], 3))
        # rule detectors on the input and on the decoded signal
        dec_recs = [{**r, "x": r["dec"]} for r in R]
        fin, fdec = filter_events(R, kind, a.kc_k), filter_events(dec_recs, kind, a.kc_k)
        res["reconstruction"]["detector_on_input"] = event_scores({}, refs, events=fin)
        res["reconstruction"]["detector_on_decoded"] = event_scores({}, refs, events=fdec)
        res["reconstruction"]["decoded_vs_input_detector"] = event_scores({}, fin, events=fdec)
        fid = [event_fidelity(r["x"], r["dec"], refs[r["id"]], kind) for r in R]
        res["reconstruction"]["fidelity"] = {k: float(np.nanmedian([f[k] for f in fid if f[k] is not None])) for k in fid[0] if k != "n"}
        res["reconstruction"]["fidelity"]["n_events"] = int(sum(f["n"] for f in fid))
        log(kind, "detector input F1", round(res["reconstruction"]["detector_on_input"]["f1"], 3),
            "decoded F1", round(res["reconstruction"]["detector_on_decoded"]["f1"], 3), "fidelity", res["reconstruction"]["fidelity"])
        rep["events"][kind] = res
    json.dump(_jsonable(rep), open(os.path.join(a.out, "vq_event_nonlinear.json"), "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
