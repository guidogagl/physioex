"""Reconstruction audit of an EOG or EMG MSAE / MSAE-VQ (01.11.2026), per night.

Metrics for every prefix (``N``, ``NE``, ``NEP``) and every setting (MSAE: bit depth of the
latents, ``0`` = continuous; MSAE-VQ: prefix of RVQ levels, key ``vq_N{a}E{b}P{c}``):

* ``logpow`` per band (the model's ``pow_bands``, one value per ``seg_sec`` segment) and
  ``logpow5`` (per 5 s): R^2 of log10 band power. ``logpow5`` is the gate (as for EEG);
* EOG (``recon="wave"``): ``wave_low``, R^2 of the waveform band-passed 0.3-4 Hz (eye
  movements, phase-exact part of the model);
* EMG (``recon="texture"``): ``env``, R^2 of log10 RMS per 0.5 s (tone and bursts);
* ``event_auc``: label-free event proxy, AUC of the reconstruction's feature against the
  target's top 5% of the night. EOG: rapid eye movement, max |slope| of the 0.3-5 Hz signal
  per 0.5 s. EMG: phasic burst, RMS per 0.5 s.

Every metric is also given per night (paired units of the non-inferiority test). Sleep
stages are not used.

    python -m physioex.models.recon_eval_modality --run <run dir> [--split valid|test] [--vq-grid ...]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Sequence

import numpy as np
import torch

from physioex.models.recon_eval import model_waves, r2, sample_windows


def _seg(x: np.ndarray, n: int) -> np.ndarray:
    m = x.shape[-1] // n * n
    return x[..., :m].reshape(*x.shape[:-1], -1, n)


def band_logpow(x: np.ndarray, fs: int, bands, seg_sec: float) -> np.ndarray:
    """``(M, n) -> (M, n_seg, n_bands)`` log10 band power per ``seg_sec`` segment."""
    p = np.abs(np.fft.rfft(_seg(x, int(seg_sec * fs)), axis=-1)) ** 2
    return np.stack([np.log10(p[..., int(a * seg_sec):int(b * seg_sec)].sum(-1) + 1e-6) for a, b in bands], -1)


def bandpass(x: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * ((f >= lo) & (f <= hi)), n=x.shape[-1], axis=-1)


def event_feature(x: np.ndarray, fs: int, kind: str) -> np.ndarray:
    """``(M, n) -> (M, n_tok)`` per 0.5 s: ``"eye"`` max |slope| of 0.3-5 Hz, ``"burst"`` RMS."""
    h = int(0.5 * fs)
    if kind == "eye":
        return np.abs(np.diff(_seg(bandpass(x, fs, 0.3, 5.0), h), axis=-1)).max(-1) * fs
    return np.sqrt((_seg(x, h) ** 2).mean(-1))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else float("nan")


def metrics(model, pred: np.ndarray, target: np.ndarray, night: np.ndarray) -> dict:
    fs, bands, seg = model.fs, model.pow_bands, model.seg_sec
    names = [f"{a:g}-{b:g}" for a, b in bands]
    kind = "burst" if model.recon == "texture" else "eye"
    lp, lt = band_logpow(pred, fs, bands, seg), band_logpow(target, fs, bands, seg)
    lp5, lt5 = band_logpow(pred, fs, bands, 5.0), band_logpow(target, fs, bands, 5.0)
    fp, ft = event_feature(pred, fs, kind), event_feature(target, fs, kind)
    out = {"logpow": {}, "logpow5": {}, "per_night": {"night": [], "logpow5": {n: [] for n in names}, "event_auc": []}}
    for i, n in enumerate(names):
        out["logpow"][n] = r2(lp[..., i], lt[..., i])
        out["logpow5"][n] = r2(lp5[..., i], lt5[..., i])
    if kind == "eye":
        wl, tl = bandpass(pred, fs, 0.3, 4.0), bandpass(target, fs, 0.3, 4.0)
        out["wave_low"] = r2(wl, tl)
        out["per_night"]["wave_low"] = []
    else:
        out["env"] = r2(np.log10(fp + 1e-6), np.log10(ft + 1e-6))
        out["per_night"]["env"] = []
    aucs = []
    for g in np.unique(night):
        m = night == g
        pn = out["per_night"]
        pn["night"].append(int(g))
        for i, n in enumerate(names):
            pn["logpow5"][n].append(r2(lp5[m][..., i], lt5[m][..., i]))
        lab = ft[m] > np.quantile(ft[m], 0.95)
        pn["event_auc"].append(auc(fp[m].ravel(), lab.ravel()))
        if kind == "eye":
            pn["wave_low"].append(r2(wl[m], tl[m]))
        else:
            pn["env"].append(r2(np.log10(fp[m] + 1e-6), np.log10(ft[m] + 1e-6)))
        aucs.append(pn["event_auc"][-1])
    out["event_auc"] = float(np.nanmean(aucs))
    out["event"] = kind
    return out


def evaluate(model, ds, ids: Sequence[str], per_night: int = 250, seed: int = 0, bits: Sequence[int] = (8, 0),
             vq_grid: Sequence[dict] = None) -> dict:
    from physioex.models.msae_vq import MSAEVQ, default_vq_grid, vq_key
    from physioex.models.recon_eval import spec_r2_per_night

    windows, target, night = sample_windows(ds, ids, per_night, seed, return_night=True)
    res = {"n_epochs": int(len(target)), "fs": model.fs, "recon": model.recon, "use_n": model.use_n, "model": {}}
    if isinstance(model, MSAEVQ):
        settings = [(g, vq_key(g)) for g in (vq_grid or default_vq_grid(model.max_levels))]
        res["bits_per_epoch"] = {k: model.bits_per_epoch(g) for g, k in settings}
    else:
        settings = [(b, f"bits_{b}") for b in bits]
    for s, key in settings:
        if isinstance(model, MSAEVQ):
            model.vq_levels = s
        else:
            model.latent_bits = s
        waves = model_waves(model, windows)
        rep = {k: metrics(model, waves[k], target, night) for k in ("N", "NE", "NEP")}
        rep["spec"] = spec_r2_per_night(model, windows, night)
        res["model"][key] = rep
        print(f"[recon_eval_modality] {key}: NEP logpow5 "
              + " ".join(f"{b}={v:.3f}" for b, v in rep["NEP"]["logpow5"].items())
              + f" event_auc={rep['NEP']['event_auc']:.3f}", flush=True)
    return res


def main():
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=("valid", "test"), default="valid")
    ap.add_argument("--per-night", type=int, default=250)
    ap.add_argument("--vq-grid", nargs="*", default=None, help="MSAE-VQ prefixes, e.g. N4E8P6 (default: the axis grid)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    run, ds, model = _load_run(a.run, cache_dir=a.cache_dir, root=a.root)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    grid = [parse_vq_key(k) for k in a.vq_grid] if a.vq_grid else None
    res = evaluate(model, ds, run["splits"][a.split], a.per_night, vq_grid=grid)
    res["split"] = a.split
    out = os.path.join(a.run, f"recon_modality_{a.split}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[recon_eval_modality] written {out}", flush=True)


if __name__ == "__main__":
    main()
