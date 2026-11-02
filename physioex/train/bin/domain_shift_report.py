"""Domain shift of a dataset of clean channels against the DCSM test nights (qualitative report).

Built for DREAMS (30-min excerpts with expert K-complexes and spindles) against DCSM, the
training domain of the MSAE-VQ encoders. Everything is computed from the clean caches, label-free
except that stages describe the epochs and the expert events are the reference:

1. amplitude per stage and modality in µV (clean signal before the per-night scale): RMS,
   5-95 percentile range, peak-to-peak of 0.3-4 Hz (EEG);
2. Welch spectrum per stage (mean log10 PSD) and band powers; spindle peak frequency in N2;
3. the per-recording robust scale of the pipelines (``RobustScale``: IQR/1.349 of the nonzero
   samples) on the target recordings, on whole DCSM nights and on the most-N2 30-min window of
   every DCSM night; the N2 RMS after the scale;
4. event shape: K-complexes aligned on their trough (DREAMS: expert ``vis1``; DCSM: the
   label-free detector of :mod:`physioex.train.bin.vq_explain_n2`), spindles (sigma envelope,
   duration, frequency); the same detectors on DREAMS against the experts (precision / recall,
   and expert 1 against expert 2 as a ceiling);
5. reconstruction of the EEG encoder (MSAE-VQ run) on the centre epochs that have a full
   context window, against DCSM test windows (wave R², log band power R² over 5 s).

``python -m physioex.train.bin.domain_shift_report --target dreams --run VQ_RUN --vq N8E8P6 --out DIR``
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from physioex.train.bin.vq_explain_n2 import _bp, kcomplexes, spindles

STAGES = ("W", "N1", "N2", "N3", "REM")
CHANNELS = ["EEG_clean", "EOG_clean", "EMG_clean"]
MODS = ["eeg", "eog", "emg"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 11), "sigma": (11, 16), "beta": (16, 30)}
FS = 100
GRID = np.arange(0, 50.001, 0.25)  # Hz, common spectrum grid


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 5)
    return o


def robust_scale(x: np.ndarray) -> float:
    """The scale of :class:`physioex.data.steps.normalize.RobustScale` (IQR / 1.349, nonzero samples)."""
    v = x[x != 0]
    if len(v) < 100:
        v = x
    q1, q3 = np.percentile(v, [25, 75])
    return float((q3 - q1) / 1.349 + 1e-6)


def load_recordings(ds, ids=None) -> List[dict]:
    """Per recording: µV epochs of every clean channel ``(n, samples)``, its fs, labels, events."""
    have = {s.subject_id: s for s in ds._subjects}
    out = []
    for sid in (ids or list(have)):
        if sid not in have:
            continue
        spec = have[sid]
        n = ds._n_epochs[sid]
        item = ds._build_item(spec, 0, n)
        sig = [item["signals"][k].numpy().reshape(n, -1).astype(np.float32) for k in item["channel_order"]]
        out.append({"id": sid, "x": dict(zip(MODS, sig)), "fs": {m: s.shape[1] / 30.0 for m, s in zip(MODS, sig)},
                    "y": np.asarray(ds._get_labels(spec))[:n], "events": ds._load_or_compute_events(spec)})
    return out


def epoch_features(recs: List[dict]) -> Dict[str, dict]:
    """Amplitude and spectrum per stage and modality (all epochs of all recordings pooled)."""
    from scipy.signal import welch

    out = {}
    for m in MODS:
        acc = {s: {"rms": [], "p5": [], "p95": [], "ptp_slow": [], "psd": []} for s in STAGES}  # noqa: E501
        freqs = None
        for r in recs:
            x, fs, y = r["x"][m], r["fs"][m], r["y"]
            ok = np.abs(x).sum(1) > 0
            for st_i, st in enumerate(STAGES):
                sel = (y == st_i) & ok
                if not sel.any():
                    continue
                e = x[sel]
                acc[st]["rms"] += np.sqrt((e ** 2).mean(1)).tolist()
                acc[st]["p5"] += np.percentile(e, 5, axis=1).tolist()
                acc[st]["p95"] += np.percentile(e, 95, axis=1).tolist()
                if m == "eeg":
                    acc[st]["ptp_slow"] += np.ptp(_bp(e, 0.3, 4.0, int(fs)), axis=1).tolist()
                f_in, p = welch(e, fs=fs, nperseg=int(4 * fs), axis=-1)
                lp = np.log10(p + 1e-12)  # common 0.25 Hz grid; NaN above this recording's Nyquist
                acc[st]["psd"].append(np.stack([np.interp(GRID, f_in, row, right=np.nan) for row in lp]))
                acc[st]["fs"] = sorted(set(acc[st].get("fs", [])) | {float(fs)})
                freqs = GRID
        res = {"freqs": freqs.tolist() if freqs is not None else []}
        for st, a in acc.items():
            if not a["rms"]:
                continue
            psd = np.concatenate(a["psd"])
            f = np.asarray(freqs)
            bp = {b: float(np.nanmedian(np.log10(np.nansum(10 ** psd[:, (f >= lo) & (f < hi)], 1) * (f[1] - f[0]) + 1e-12)))
                  for b, (lo, hi) in BANDS.items()}
            res[st] = {"n": len(a["rms"]), "rms_median": float(np.median(a["rms"])),
                       "rms_iqr": np.percentile(a["rms"], [25, 75]).tolist(),
                       "p5_median": float(np.median(a["p5"])), "p95_median": float(np.median(a["p95"])),
                       "ptp_slow_median": float(np.median(a["ptp_slow"])) if a["ptp_slow"] else None,
                       "logpsd_mean": np.nanmean(psd, 0).tolist(), "fs_seen": a.get("fs", []), "log_band_power_median": bp}
            if m == "eeg" and st == "N2":
                s = (f >= 10) & (f <= 16)
                m_ = np.nanmean(psd, 0)[s]  # peak above the straight line between 10 and 16 Hz
                res[st]["sigma_peak_hz"] = float(f[s][np.argmax(m_ - np.linspace(m_[0], m_[-1], s.sum()))])
        out[m] = res
    return out


def scales(recs: List[dict], n2_window: int = 0) -> Dict[str, list]:
    """Robust scale per recording (whole) and, with ``n2_window`` epochs, on the most-N2 window."""
    out = {m: {"whole": [], "n2_window": [], "n2_rms_scaled": []} for m in MODS}
    for r in recs:
        y = r["y"]
        lo, hi = 0, len(y)
        if n2_window and len(y) > n2_window:
            c = np.convolve((y == 2).astype(float), np.ones(n2_window), "valid")
            lo = int(np.argmax(c)); hi = lo + n2_window
        for m in MODS:
            x = r["x"][m]
            s_whole = robust_scale(x.reshape(-1))
            s_win = robust_scale(x[lo:hi].reshape(-1))
            out[m]["whole"].append(s_whole)
            out[m]["n2_window"].append(s_win)
            n2 = x[(y == 2) & (np.abs(x).sum(1) > 0)]
            if len(n2):
                out[m]["n2_rms_scaled"].append(float(np.median(np.sqrt((n2 ** 2).mean(1))) / s_whole))
    return out


def kc_shapes_expert(recs, half_s=1.5, concept="vis1"):
    """Expert K-complexes (EEG µV) aligned on the trough of 0.3-4 Hz inside the scored span."""
    segs, amps, durs = [], [], []
    for r in recs:
        x = r["x"]["eeg"].reshape(-1)
        lo = _bp(x, 0.3, 4.0)
        for e in r["events"]:
            if e.type != "k_complex" or e.concept != concept:
                continue
            a, b = int(e.onset_sec * FS), int((e.onset_sec + e.duration_sec) * FS)
            if b <= a or b > len(x):
                continue
            t = a + int(np.argmin(lo[a:b]))
            if t - int(half_s * FS) < 0 or t + int(half_s * FS) > len(x):
                continue
            segs.append(x[t - int(half_s * FS):t + int(half_s * FS)])
            amps.append(float(lo[a:b].max() - lo[a:b].min())); durs.append(e.duration_sec)
    return segs, amps, durs


def kc_shapes_detector(recs, half_s=1.5, k=4.0):
    """Detector K-complexes in N2 epochs, aligned on their trough."""
    segs, amps, found = [], [], []
    for ri, r in enumerate(recs):
        x, y = r["x"]["eeg"], r["y"]
        n2 = np.flatnonzero((y == 2) & (np.abs(x).sum(1) > 0))
        if not len(n2):
            continue
        ref = float(np.median(np.ptp(_bp(x[n2], 0.3, 4.0), axis=1))) / 2
        flat = x.reshape(-1)
        for ep in n2:
            for a, b in kcomplexes(x[ep], ref, k):
                lo = _bp(x[ep], 0.3, 4.0)
                t = ep * 3000 + a + int(np.argmin(lo[a:b]))
                found.append((ri, (ep * 3000 + a) / FS, (ep * 3000 + b) / FS))
                if t - int(half_s * FS) < 0 or t + int(half_s * FS) > len(flat):
                    continue
                segs.append(flat[t - int(half_s * FS):t + int(half_s * FS)])
                amps.append(float(lo[a:b].max() - lo[a:b].min()))
    return segs, amps, found


def spindle_stats(x_flat, spans):
    """Sigma envelope amplitude (µV, RMS of 11-16 Hz x sqrt2), duration and peak frequency."""
    sig = _bp(x_flat, 11, 16)
    amp, dur, freq = [], [], []
    for a, b in spans:
        if b - a < 20 or b > len(x_flat):
            continue
        seg = x_flat[a:b]
        amp.append(float(np.sqrt(2 * (sig[a:b] ** 2).mean())))
        dur.append((b - a) / FS)
        f = np.fft.rfftfreq(256, 1 / FS)
        p = np.abs(np.fft.rfft(seg - seg.mean(), 256)) ** 2
        s = (f >= 10) & (f <= 17)
        freq.append(float(f[s][np.argmax(p[s])]))
    return amp, dur, freq


def match(pred, ref, tol=0.0):
    """Event match by overlap: precision, recall, F1 (``pred``/``ref`` lists of (start, stop) s)."""
    if not pred or not ref:
        return {"precision": None if not pred else 0.0, "recall": None if not ref else 0.0, "n_pred": len(pred), "n_ref": len(ref)}
    hit_p = sum(any(a < d + tol and c < b + tol for c, d in ref) for a, b in pred)
    hit_r = sum(any(a < d + tol and c < b + tol for a, b in pred) for c, d in ref)
    p, r = hit_p / len(pred), hit_r / len(ref)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "n_pred": len(pred), "n_ref": len(ref)}


def detector_vs_experts(recs, kind, k=4.0):
    """Our detector against expert 1 (and expert 2 against expert 1, the ceiling), pooled over excerpts.
    Only epochs scored N2 or N3 for the detector (the experts scored the whole excerpt)."""
    res = {"detector_vs_vis1": [], "detector_vs_vis2": [], "vis2_vs_vis1": [], "auto_vs_vis1": []}
    tot = {k_: {"tp_p": 0, "n_pred": 0, "tp_r": 0, "n_ref": 0} for k_ in res}
    for r in recs:
        ev = {c: [(e.onset_sec, e.onset_sec + e.duration_sec) for e in r["events"] if e.type == kind and e.concept == c]
              for c in ("vis1", "vis2", "auto")}
        x, y = r["x"]["eeg"], r["y"]
        if kind == "k_complex":
            n2 = np.flatnonzero(np.isin(y, (2, 3)) & (np.abs(x).sum(1) > 0))
            ref = float(np.median(np.ptp(_bp(x[n2], 0.3, 4.0), axis=1))) / 2 if len(n2) else 1.0
            det = [((ep * 3000 + a) / FS, (ep * 3000 + b) / FS) for ep in n2 for a, b in kcomplexes(x[ep], ref, k)]
        else:
            det = [(a / FS, b / FS) for a, b in spindles(x.reshape(-1))]
        pairs = {"detector_vs_vis1": (det, ev["vis1"]), "detector_vs_vis2": (det, ev["vis2"]),
                 "vis2_vs_vis1": (ev["vis2"], ev["vis1"]), "auto_vs_vis1": (ev["auto"], ev["vis1"])}
        for name, (p, q) in pairs.items():
            if not q or (name.endswith("vis2") and not ev["vis2"]) or (name == "vis2_vs_vis1" and not ev["vis2"]):
                continue
            mm = match(p, q)
            res[name].append({"rec": r["id"], **mm})
            tot[name]["n_pred"] += len(p); tot[name]["n_ref"] += len(q)
            tot[name]["tp_p"] += round((mm["precision"] or 0) * len(p)); tot[name]["tp_r"] += round((mm["recall"] or 0) * len(q))
    pooled = {}
    for name, t in tot.items():
        if t["n_ref"]:
            p = t["tp_p"] / t["n_pred"] if t["n_pred"] else 0.0
            r_ = t["tp_r"] / t["n_ref"]
            pooled[name] = {"precision": p, "recall": r_, "f1": 2 * p * r_ / (p + r_) if p + r_ else 0.0, **t}
    return {"pooled": pooled, "per_recording": res}


def encoder_recon(run_dir, vq, target, target_root=None, per_night=60):
    """R² of the EEG reconstruction on every target centre epoch with a full context window
    against random DCSM test windows (the run's own dataset and pipelines)."""
    import torch

    from physioex.data.datasets import get_dataset
    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import _load_run, _report, channel_pipelines, model_waves, sample_windows

    run, dcsm_ds, model = _load_run(run_dir)
    ra = run["args"]
    model.vq_levels = parse_vq_key(vq)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    kw = dict(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra), sequence_length=ra["seq"],
              cache_dir=ra.get("cache_dir"))
    if target_root:
        kw["root"] = target_root
    tds = get_dataset(target)(**kw)
    half = ra["seq"] // 2
    W, G, ids = [], [], []
    for gi, spec in enumerate(tds._subjects):
        n = tds._n_epochs[spec.subject_id]
        x = next(iter(tds._build_item(spec, 0, n)["signals"].values())).numpy().reshape(n, -1).astype(np.float32)
        for t in range(half, n - half):
            W.append(x[t - half:t + half + 1]); G.append(gi)
        ids.append(spec.subject_id)
    W, G = np.stack(W), np.array(G)
    wt = model_waves(model, W)["NEP"]
    Wd, Cd, Gd = sample_windows(dcsm_ds, list(run["splits"]["test"]), per_night, return_night=True)
    wd = model_waves(model, Wd)["NEP"]
    out = {"target": _report(wt, W[:, half], G), "dcsm": _report(wd, Cd, Gd), "target_ids": ids,
           "n_target_windows": int(len(W)), "n_dcsm_windows": int(len(Wd)),
           "scaled_rms": {"target": float(np.sqrt((W[:, half] ** 2).mean())), "dcsm": float(np.sqrt((Cd ** 2).mean()))}}
    k = min(6, len(W))
    out["_examples"] = {"target_input": W[:k, half], "target_decoded": wt[:k], "dcsm_input": Cd[:k], "dcsm_decoded": wd[:k]}
    return out


def raw_vs_clean(ds, recs, ids=("kc_excerpt1", "kc_excerpt2"), n_each=2, span_s=30.0):
    """Raw scoring channel (EDF, 0.3-35 Hz, 100 Hz) against the clean EEG, both in µV, on
    ``span_s`` windows centred on expert-1 K-complexes. Returns arrays for the page."""
    from scipy.signal import resample_poly

    from physioex.data.readers.edf import read_edf_lenient

    have = {s.subject_id: s for s in ds._subjects}
    out, meta = {}, []
    for sid in ids:
        r = next((r for r in recs if r["id"] == sid), None)
        if r is None or sid not in have:
            continue
        e = read_edf_lenient(str(have[sid].edf_path))
        ch = next(c for c in ("CZ-A1", "C3-A1") if c in e["labels"])
        i = e["labels"].index(ch)
        fs = int(round(e["fs"][i]))
        raw = resample_poly(np.asarray(e["signals"][i], np.float64), FS, fs) if fs != FS else np.asarray(e["signals"][i], np.float64)
        raw = _bp(raw, 0.3, 35.0)
        clean = r["x"]["eeg"].reshape(-1)
        kcs = [ev for ev in r["events"] if ev.type == "k_complex" and ev.concept == "vis1"]
        kcs = kcs[:: max(1, len(kcs) // n_each)][:n_each]
        for k, ev in enumerate(kcs):
            c = ev.onset_sec + ev.duration_sec / 2
            a = int(max(0, min(c - span_s / 2, len(clean) / FS - span_s)) * FS)
            b = a + int(span_s * FS)
            key = f"{sid}_{k}"
            out[f"rvc_raw_{key}"], out[f"rvc_clean_{key}"] = raw[a:b], clean[a:b]
            others = [(o.onset_sec - a / FS, o.duration_sec) for o in kcs + [x for x in r["events"] if x.type == "k_complex" and x.concept == "vis1"]
                      if a / FS <= o.onset_sec < b / FS]
            meta.append({"key": key, "rec": sid, "channel": ch, "start_s": a / FS, "kc_in_window": sorted(set(others)),
                         "stage": int(r["y"][int(c // 30)]) if int(c // 30) < len(r["y"]) else -1,
                         "corr_raw_clean": float(np.corrcoef(raw[a:b], clean[a:b])[0, 1])})
    return out, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="dreams")
    ap.add_argument("--target-root", default=None)
    ap.add_argument("--run", required=True, help="DCSM MSAE-VQ EEG run (its test split = the DCSM sample)")
    ap.add_argument("--vq", default="N8E8P6")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-dcsm-nights", type=int, default=0)
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--examples-only", action="store_true", help="only the raw-vs-clean windows (CPU, seconds)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from physioex.data.datasets import get_dataset
    os.makedirs(a.out, exist_ok=True)
    log = lambda *s: print("[domain_shift]", *s, flush=True)  # noqa: E731
    run = json.load(open(os.path.join(a.run, "run.json")))
    ra = run["args"]
    cache = a.cache_dir or ra.get("cache_dir")
    from physioex.data.pipeline import PreprocessingPipeline
    from physioex.data.steps.normalize import NanToZero

    raw = {k: PreprocessingPipeline([NanToZero()]) for k in CHANNELS}  # clean µV, no per-night scale
    tkw = dict(channels=CHANNELS, pipelines=raw, sequence_length=1, cache_dir=cache, trim_mode="excess_wake")
    if a.target_root:
        tkw["root"] = a.target_root
    tds = get_dataset(a.target)(**tkw)
    tgt = load_recordings(tds)
    log("target", a.target, len(tgt), "recordings")
    if a.examples_only:
        ex, meta = raw_vs_clean(tds, tgt)
        json.dump(_jsonable(meta), open(os.path.join(a.out, "raw_vs_clean.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(a.out, "raw_vs_clean.npz"), **{k: np.asarray(v, np.float32) for k, v in ex.items()})
        log("DONE")
        return
    dkw = dict(channels=CHANNELS, pipelines=raw, sequence_length=1, cache_dir=cache, trim_mode="main_period")
    if ra.get("root"):
        dkw["root"] = ra["root"]
    dcsm_ds = get_dataset(ra["dataset"])(**dkw)
    ids = list(run["splits"]["test"])[:a.max_dcsm_nights or None]
    dcsm = load_recordings(dcsm_ds, ids)
    log("dcsm", len(dcsm), "test nights")

    rep = {"target": a.target, "run": a.run, "vq": a.vq, "n_target": len(tgt), "n_dcsm": len(dcsm)}
    rep["features"] = {"target": epoch_features(tgt), "dcsm": epoch_features(dcsm)}
    n_ep = int(np.median([len(r["y"]) for r in tgt]))
    rep["scale"] = {"target": scales(tgt), "dcsm_whole": scales(dcsm), "dcsm_n2_window": scales(dcsm, n2_window=n_ep)}
    rep["stage_mix"] = {"target": np.bincount(np.concatenate([r["y"][r["y"] >= 0] for r in tgt]), minlength=5).tolist(),
                        "dcsm": np.bincount(np.concatenate([r["y"][r["y"] >= 0] for r in dcsm]), minlength=5).tolist()}
    log("features and scales done")

    kc_t, kc_t_amp, kc_t_dur = kc_shapes_expert([r for r in tgt if r["id"].startswith("kc_")])
    kc_d, kc_d_amp, _ = kc_shapes_detector(dcsm)
    kc_td, kc_td_amp, _ = kc_shapes_detector([r for r in tgt if r["id"].startswith("kc_")])
    rep["kcomplex"] = {
        "expert_target": {"n": len(kc_t), "ptp_slow_median": float(np.median(kc_t_amp)) if kc_t else None,
                          "duration_median": float(np.median(kc_t_dur)) if kc_t else None},
        "detector_target": {"n": len(kc_td), "ptp_slow_median": float(np.median(kc_td_amp)) if kc_td else None},
        "detector_dcsm": {"n": len(kc_d), "ptp_slow_median": float(np.median(kc_d_amp)) if kc_d else None},
        "agreement": detector_vs_experts([r for r in tgt if r["id"].startswith("kc_")], "k_complex"),
        # threshold of the detector (k x median N2 half peak-to-peak): k=4 is the one of vq_explain_n2
        "agreement_by_k": {str(k): detector_vs_experts([r for r in tgt if r["id"].startswith("kc_")], "k_complex", k)["pooled"]
                           for k in (1.5, 2.0, 2.5, 3.0)},
        "detector_dcsm_by_k": {str(k): len(kc_shapes_detector(dcsm, k=k)[2]) for k in (1.5, 2.0, 2.5, 3.0, 4.0)},
    }
    sp_recs = [r for r in tgt if r["id"].startswith("sp_")]
    sp_t = [spindle_stats(r["x"]["eeg"].reshape(-1), [(int(e.onset_sec * FS), int((e.onset_sec + e.duration_sec) * FS))
                                                       for e in r["events"] if e.type == "spindle" and e.concept == "vis1"]) for r in sp_recs]
    sp_d = []
    for r in dcsm:
        n2 = np.flatnonzero(r["y"] == 2)
        if len(n2):
            xf = r["x"]["eeg"][n2].reshape(-1)
            sp_d.append(spindle_stats(xf, spindles(xf)))
    cat = lambda L, i: np.concatenate([np.asarray(s[i]) for s in L if len(s[i])]) if any(len(s[i]) for s in L) else np.array([])  # noqa: E731
    sp_dt = [spindle_stats(r["x"]["eeg"].reshape(-1), spindles(r["x"]["eeg"].reshape(-1))) for r in sp_recs]
    rep["spindle"] = {
        "detector_target": {"n": int(len(cat(sp_dt, 0))), **{f"{k}_median": float(np.median(cat(sp_dt, i))) for i, k in enumerate(("amp_uV", "duration_s", "freq_hz")) if len(cat(sp_dt, i))}},
        "expert_target": {"n": int(len(cat(sp_t, 0))), **{f"{k}_median": float(np.median(cat(sp_t, i))) for i, k in enumerate(("amp_uV", "duration_s", "freq_hz")) if len(cat(sp_t, i))}},
        "detector_dcsm": {"n": int(len(cat(sp_d, 0))), **{f"{k}_median": float(np.median(cat(sp_d, i))) for i, k in enumerate(("amp_uV", "duration_s", "freq_hz")) if len(cat(sp_d, i))}},
        "agreement": detector_vs_experts(sp_recs, "spindle"),
    }
    log("events done")

    waves = {"kc_expert_target_mean": np.mean(kc_t, 0) if kc_t else np.zeros(300),
             "kc_detector_dcsm_mean": np.mean(kc_d, 0) if kc_d else np.zeros(300),
             "kc_detector_target_mean": np.mean(kc_td, 0) if kc_td else np.zeros(300)}
    if kc_t:
        waves["kc_expert_target_examples"] = np.stack(kc_t[:: max(1, len(kc_t) // 8)][:8])
    if kc_d:
        waves["kc_detector_dcsm_examples"] = np.stack(kc_d[:: max(1, len(kc_d) // 8)][:8])
    if not a.no_encoder:
        enc = encoder_recon(a.run, a.vq, a.target, a.target_root)
        for k, v in enc.pop("_examples").items():
            waves[f"recon_{k}"] = v
        rep["encoder"] = enc
        log("encoder done", enc["target"]["r2"], enc["dcsm"]["r2"])
    json.dump(_jsonable(rep), open(os.path.join(a.out, "domain_shift.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "domain_shift_waves.npz"), **{k: np.asarray(v, np.float32) for k, v in waves.items()})
    log("DONE")


if __name__ == "__main__":
    main()
