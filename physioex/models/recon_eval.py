"""Reconstruction audit of an EEG autoencoder (MSAE or MSVQ) on the test nights.

The overall R^2 of the waveform is dominated by the slow frequencies (with 1/f EEG,
~78% of the variance of a 1 s segment lies below 4 Hz), so it is reported per band:

* **R^2 with phase** per band: on the complex rfft of every 1 s segment;
* **R^2 of the amplitude** per band: on ``|rfft|`` per bin (phase ignored). Strict on
  noise-like EEG: above 5 Hz even a PCA at equal budget scores below 0;
* **R^2 of the band power** per 1 s segment: on ``log10`` of the power summed over the
  band's bins. It is the amplitude gate (envelope of every band, second by second).

Reference baselines at equal budget (numbers per epoch), fitted on training nights:
PCA of the 30 s epoch with as many components as the model's latent numbers, and
a k-means on raw 1 s segments. For MSAE every prefix is reported (``N``, ``N+E``,
``N+E+P``, with the gain of each scale per band; with the "istft" decoder the
prefixes are compared on the log spectrogram, the waveform only for ``N+E+P``) and the latents are rounded to
8, 6 and 4 bits (``0`` = continuous ``tanh`` latents) to measure the cost of the bit channel.

    python -m physioex.models.recon_eval --run <run dir> [--cache-dir ...] [--root ...]

reads ``run.json`` and ``model.pt`` of a run of ``physioex.train.bin.msvq`` and
writes ``recon_eval.json`` next to them. Sleep stages are not used.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch

FS = 100
BANDS = ((0, 1, "DC"), (1, 2, "1"), (2, 5, "2-4"), (5, 9, "5-8"), (9, 13, "9-12"),
         (13, 17, "13-16"), (17, 31, "17-30"), (31, 51, "31-50"))


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def band_r2(pred: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    """R^2 per band (Hz, 1 s rfft bins): with phase, of the amplitude, of the log band power.
    ``pred``, ``target``: ``(M, 3000)`` epochs at 100 Hz."""
    fp = np.fft.rfft(pred.reshape(-1, FS), axis=1)
    ft = np.fft.rfft(target.reshape(-1, FS), axis=1)
    out = {"phase": {}, "amp": {}, "logpow": {}, "logpow_bias": {}}
    for a, b, name in BANDS:
        p, t = fp[:, a:b], ft[:, a:b]
        out["phase"][name] = float(1 - (np.abs(p - t) ** 2).sum() / (np.abs(t - t.mean(0)) ** 2).sum())
        ap, at = np.abs(p), np.abs(t)
        out["amp"][name] = float(1 - ((ap - at) ** 2).sum() / ((at - at.mean(0)) ** 2).sum())
        lp, lt = np.log10((ap ** 2).sum(1) + 1e-6), np.log10((at ** 2).sum(1) + 1e-6)
        out["logpow"][name] = r2(lp, lt)
        out["logpow_bias"][name] = float((lp - lt).mean())  # log10: < 0 = reconstruction too weak in the band
    return out


def _report(pred: np.ndarray, target: np.ndarray) -> dict:
    return {"r2": r2(pred, target), **band_r2(pred, target)}


# ----------------------------------------------------------------------------
# Test windows
# ----------------------------------------------------------------------------


def sample_windows(ds, subject_ids: Sequence[str], per_night: int, seed: int = 0, centres_only: bool = False):
    """Random windows centred inside the main sleep period of each night.
    Returns ``(windows (M, L, 3000) or None, centres (M, 3000))``."""
    rng = np.random.default_rng(seed)
    have = {s.subject_id: s for s in ds._subjects}
    half = ds.sequence_length // 2
    W, C = [], []
    for sid in subject_ids:
        if sid not in have:
            continue
        n = ds._n_epochs[sid]
        lo, hi = ds._main_period.get(sid, (0, n)) if hasattr(ds, "_main_period") else (0, n)
        lo, hi = max(lo, half), min(hi, n - half)
        if hi <= lo:
            continue
        x = next(iter(ds._build_item(have[sid], 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        ts = np.sort(rng.choice(np.arange(lo, hi), min(per_night, hi - lo), replace=False))
        C.append(x[ts])
        if not centres_only:
            W.append(np.stack([x[t - half:t + half + 1] for t in ts]))
    return (np.concatenate(W) if W else None), np.concatenate(C)


# ----------------------------------------------------------------------------
# Model outputs
# ----------------------------------------------------------------------------


@torch.no_grad()
def model_waves(model, windows: np.ndarray, batch: int = 64) -> Dict[str, np.ndarray]:
    """Reconstructed centre epochs ``(M, 3000)`` per prefix: MSAE ``N``, ``NE``, ``NEP``; MSVQ ``NEP``."""
    from physioex.models.msae import MSAE

    device = next(model.parameters()).device
    model.eval()
    tf = isinstance(model, MSAE) and model.decoder == "istft"
    if tf:  # waves only from N+E+P; the N and N+E prefixes give log spectrograms (true units)
        keys = {"wave": "NEP", "tf_n": "tf_N", "tf_ne": "tf_NE", "tf": "tf_NEP"}
    elif isinstance(model, MSAE):
        keys = {"wave_n": "N", "wave_ne": "NE", "wave": "NEP"}
        if model.texture:  # NEP = deterministic + noise texture; NEP_det = the MSE wave alone
            keys["wave_det"] = "NEP_det"
    else:
        keys = {"wave": "NEP"}
    acc = {k: [] for k in keys.values()}
    for i in range(0, len(windows), batch):
        xb = torch.from_numpy(windows[i:i + batch])[:, :, None].to(device)
        out = model(xb)
        for src, dst in keys.items():
            v = out[src].float()
            if src.startswith("tf"):
                v = v * model.lm_std + model.lm_mean
            else:
                v = v.reshape(len(xb), -1)
            acc[dst].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def tf_band_r2(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """R^2 per band of the log spectrogram ``(M, frames, bins)`` (1 Hz bins), plus ``"all"``."""
    out = {"all": r2(pred, target)}
    for a, b, name in BANDS:
        out[name] = r2(pred[..., a:b], target[..., a:b])
    return out


def baselines(train_epochs: np.ndarray, test_epochs: np.ndarray, n_numbers: int, k_means: Sequence[int] = (256,),
              seed: int = 0) -> Dict[str, dict]:
    """PCA of the 30 s epoch with ``n_numbers`` components (equal numbers per epoch) and
    k-means on raw 1 s segments (``log2 K`` bits per second)."""
    from sklearn.cluster import MiniBatchKMeans

    out = {}
    mu = train_epochs.mean(0)
    k = min(n_numbers, len(train_epochs) - 1)
    _, _, vt = np.linalg.svd(train_epochs - mu, full_matrices=False)
    v = vt[:k]
    out[f"pca_{k}"] = _report((test_epochs - mu) @ v.T @ v + mu, test_epochs)
    s_tr, s_te = train_epochs.reshape(-1, FS), test_epochs.reshape(-1, FS)
    for K in k_means:
        km = MiniBatchKMeans(K, batch_size=8192, n_init=1, random_state=seed, max_iter=50).fit(s_tr)
        out[f"kmeans_{K}_1s"] = _report(km.cluster_centers_[km.predict(s_te)].reshape(test_epochs.shape), test_epochs)
    return out


def latent_numbers(model) -> int:
    """Numbers per epoch the decoder receives besides the context (MSAE: E + P)."""
    from physioex.models.msae import MSAE

    if isinstance(model, MSAE):
        return model.dim + model.n_tok * model.d_p
    return getattr(model, "dim", 64) * 30


def evaluate(model, ds, test_ids, train_ids, per_night: int = 250, per_train_night: int = 120,
             max_train_nights: int = 40, bits: Sequence[int] = (8, 6, 4, 0), seed: int = 0) -> dict:
    """Full audit (see module docstring). Returns the dict written to ``recon_eval.json``."""
    from physioex.models.msae import MSAE

    windows, target = sample_windows(ds, test_ids, per_night, seed)
    _, train_epochs = sample_windows(ds, list(train_ids)[:max_train_nights], per_train_night, seed + 1, centres_only=True)
    res = {"n_test_epochs": int(len(target)), "n_train_epochs": int(len(train_epochs)),
           "bands_hz": [b[2] for b in BANDS], "model": {}}
    n_num = latent_numbers(model)
    res["numbers_per_epoch"] = n_num
    if isinstance(model, MSAE):
        res["bits_per_epoch"] = model.bits_per_epoch()
        res["compression_numbers"] = 3000 / n_num
        res["compression_bits"] = 3000 * 16 / model.bits_per_epoch()
        b0 = model.latent_bits
        tf_target = None
        if model.decoder == "istft":
            from physioex.models.msae import tf_logmag, tf_stft

            tf_target = tf_logmag(tf_stft(torch.from_numpy(target))).numpy()
        for b in bits:
            model.latent_bits = b
            waves = model_waves(model, windows)
            if tf_target is not None:
                rep = {"NEP": _report(waves["NEP"], target)}
                for k in ("N", "NE", "NEP"):
                    rep[f"tf_{k}"] = tf_band_r2(waves[f"tf_{k}"], tf_target)
                for hi, lo in (("NE", "N"), ("NEP", "NE")):
                    rep[f"gain_{hi[-1]}"] = {band: rep[f"tf_{hi}"][band] - rep[f"tf_{lo}"][band]
                                             for band in rep[f"tf_{hi}"]}
                res["model"][f"bits_{b}"] = rep
                continue
            rep = {k: _report(v, target) for k, v in waves.items()}
            for hi, lo in (("NE", "N"), ("NEP", "NE")):
                rep[f"gain_{hi[-1]}"] = {kind: {band: rep[hi][kind][band] - rep[lo][kind][band] for band in rep[hi][kind]}
                                         for kind in ("phase", "amp", "logpow")}
                rep[f"gain_{hi[-1]}"]["r2"] = rep[hi]["r2"] - rep[lo]["r2"]
            res["model"][f"bits_{b}"] = rep
        model.latent_bits = b0
        if 0 in bits and b0 in bits:
            res["drop_r2_rounding"] = res["model"]["bits_0"]["NEP"]["r2"] - res["model"][f"bits_{b0}"]["NEP"]["r2"]
    else:
        res["model"]["default"] = {k: _report(v, target) for k, v in model_waves(model, windows).items()}
    res["baselines"] = baselines(train_epochs, target, n_num, seed=seed)
    return res


def best_checkpoint(run_dir: str):
    """Newest ``ckpt/*.pt`` of a run (the Trainer writes one on every validation improvement), or None."""
    import glob

    files = glob.glob(os.path.join(run_dir, "ckpt", "*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def load_best(model, run_dir: str):
    """Load the best validation checkpoint into ``model`` in place; return its path (None = kept as is)."""
    path = best_checkpoint(run_dir)
    if path is not None:
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    return path


def _load_run(run_dir: str, cache_dir=None, root=None, weights="best"):
    from physioex.data.datasets import get_dataset
    from physioex.data.steps import clean_pipelines

    run = json.load(open(os.path.join(run_dir, "run.json")))
    a = run["args"]
    kw = dict(channels=["EEG_clean"], pipelines=clean_pipelines(), sequence_length=a["seq"],
              cache_dir=cache_dir or a.get("cache_dir"), trim_mode="main_period")
    if root or a.get("root"):
        kw["root"] = root or a["root"]
    ds = get_dataset(a["dataset"])(**kw)
    if a.get("model", "msvq") == "msae":
        from physioex.models.msae import MSAE

        model = MSAE(a["seq"], width=a["width"], dim=a["dim"] or 128, d_p=a["d_p"], token_sec=a["token_sec"],
                     latent_bits=a["latent_bits"], decoder=a.get("decoder", "time"),
                     pow_hz=a.get("pow_hz") or a.get("low_hz", 4.0), texture=a.get("texture", False))
    else:
        from physioex.models.msvq import MSVQ

        model = MSVQ(a["seq"], a["dim"] or 64, *a["codes"], quantize=not a["no_vq"])
    if weights != "best" or load_best(model, run_dir) is None:
        model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"), strict=False)
    return run, ds, model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--per-night", type=int, default=250)
    ap.add_argument("--out", default=None, help="default: <run>/recon_eval.json")
    ap.add_argument("--weights", choices=("best", "final"), default="best",
                    help="best = newest ckpt/*.pt (best validation), final = model.pt")
    a = ap.parse_args()
    run, ds, model = _load_run(a.run, a.cache_dir, a.root, a.weights)
    if torch.cuda.is_available():
        model = model.cuda()
    res = evaluate(model, ds, run["splits"]["test"], run["splits"]["train"], per_night=a.per_night)
    with open(a.out or os.path.join(a.run, "recon_eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("model", "baselines")}), flush=True)


if __name__ == "__main__":
    main()
