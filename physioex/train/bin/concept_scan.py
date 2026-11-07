"""Codebook components of every lexicon concept (plan of 07.11.2026, ``LEXICON_CONCETTI.md``).

Healthy nights only (the clinical cohorts Alzheimer / Parkinson / LSEC stay out): the test split of the
pool units, ``mass_ss01`` (expert arousals) and the DREAMS excerpts (expert K-complexes and spindles).

Sub-commands:

- ``dump`` (GPU): for every night and modality of a ``pretrain_combo`` combination, the codes of every
  scale and RVQ level, the quantized ``P`` tokens, the input signal of the scored epochs and the stage
  log-probabilities of the combination's stagers (``eeg``, ``multi``); one ``.npz`` per night plus
  ``meta.json`` (events, sampling rates). Every scored epoch of the main sleep period is kept; the
  windows at the edges of the night are completed by reflection, as ``vq_event_readout`` does.

    python -m physioex.train.bin.concept_scan dump --combo-dir RUNS/<tag> --stager stager_full \\
        --splits splits_29sites.json --cache-dir CACHE --per-unit 20 --out OUT

- ``scale`` (CPU): the dumped signals are robust-scaled per night (``RobustScale`` of the clean-channel
  pipeline), not uV, and the night cache keeps no uV copy. The night cleaner runs again on every dumped
  night and the median / scale of ``RobustScale`` are recomputed on its outputs, so that
  ``x_uV = x * scale + median``; where the model rate equals the cleaner rate the reconstruction is
  checked against the dump (``maxerr``, fp16 rounding ~1e-2). ``<out>/scale.json``.

    python -m physioex.train.bin.concept_scan scale --dump OUT --cache-dir CACHE --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

#: healthy units of the scan: test split of the pool, shhs as out-of-pool control, MASS SS01 (expert arousals)
UNITS = ("ccshs", "dcsm", "hmc", "hpap_lab_full", "mass_ss02", "mass_ss03", "mass_ss04", "mass_ss05", "mesa", "mros",
         "sleepedf", "wsc_visit1", "shhs_visit1", "mass_ss01")
MODS = ("eeg", "eog", "emg")


def _pick(ids, k, seed):
    """``k`` nights of a unit, fixed seed (``k`` <= 0: all)."""
    ids = sorted(ids)
    if k <= 0 or len(ids) <= k:
        return ids
    return sorted(np.random.default_rng(seed).choice(ids, k, replace=False).tolist())


def _reflect_night(x, ts, half):
    """Signal padded by reflection at the night borders and the shifted epoch indices, so that every
    scored epoch has a whole window (the edge windows are completed as in ``vq_event_readout``)."""
    from physioex.train.bin.vq_event_codes import reflect_index

    idx = reflect_index(len(x), half)
    return x[idx], ts + half


def _stage_logp(stagers, lat, mods, dev):
    """Stage log-probabilities per epoch of the ``eeg`` and ``multi`` stagers (missing modalities zero)."""
    from physioex.models.qformer_stager import pack_modalities, predict, standardize
    from physioex.train.bin.pretrain_combo import _empty_latents

    st_models, stats, meta = stagers
    out = {}
    if "eeg" in st_models and "eeg" in lat:
        out["eeg"] = predict(st_models["eeg"], standardize({"x": lat["eeg"]}, stats["eeg"])["x"], device=dev)[0]
    if "multi" in st_models:
        by = {m: {"x": lat.get(m) or _empty_latents(stats[m], meta["n_tok"][m])} for m in mods}
        data, _, _ = pack_modalities(by, stats=stats)
        out["multi"] = predict(st_models["multi"], data["x"], device=dev)[0]
    return {k: v.numpy().astype(np.float16) for k, v in out.items()}


def _night_arrays(models, readers, sid_of, mods, half_of):
    """Encode one night in every modality present: codes, P, signal of the scored epochs."""
    from physioex.train.bin.pretrain_combo import encode_night

    arr, lat, ts_ref, y_ref = {}, {}, None, None
    for m in mods:
        if m not in readers:
            continue
        x, y, ts = readers[m].get(sid_of[m])
        if m == "eeg":
            ts_ref, y_ref = ts, y[ts]
        elif ts_ref is not None:  # same epochs as EEG (the reference modality)
            ts = ts_ref[ts_ref < len(x)]
        if len(ts) < 21:
            continue
        xp, tp = _reflect_night(x, ts, half_of[m])
        z = encode_night(models[m], xp, tp, half_of[m])
        arr[f"x_{m}"] = x[ts].astype(np.float16)
        arr[f"P_{m}"] = z["P"]
        for k, c in z["codes"].items():
            arr[f"codes_{m}_{k}"] = c.astype(np.int16)
        lat[m] = {"N": z["N"], "E": z["E"], "P": z["P"], "y": y[ts].astype(np.int64),
                  "night": np.zeros(len(ts), np.int64), "epoch": ts.astype(np.int64)}
    return arr, lat, ts_ref, y_ref


class _Fixed:
    """Reader of one already loaded recording (``Nights.get`` interface)."""

    def __init__(self, x, y, ts):
        self.x, self.y, self.ts = x, y, ts

    def get(self, _sid):
        return self.x, self.y, self.ts


def _events(ds, sid):
    spec = {s.subject_id: s for s in ds._subjects}[sid]
    return [{"type": e.type, "concept": e.concept, "onset": float(e.onset_sec), "dur": float(e.duration_sec)}
            for e in ds._load_or_compute_events(spec)]


def dump(a):
    import torch

    from physioex.data.merged import BY_NAME, build_test
    from physioex.models.recon_eval import channel_pipelines
    from physioex.train.bin.pretrain_combo import Nights, _cache_dir, load_s2, load_stagers

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mods = a.mods.split(",")
    models, args0 = {}, {}
    for m in mods:
        run, models[m] = load_s2(os.path.join(a.combo_dir, m, "s2"), m, dev)
        args0[m] = run["args"]
    stagers = load_stagers(os.path.join(a.combo_dir, a.stager), mods, dev)
    half_of = {m: args0[m]["seq"] // 2 for m in mods}
    os.makedirs(a.out, exist_ok=True)
    mp = os.path.join(a.out, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {"nights": {}}
    meta.update({"combo_dir": os.path.abspath(a.combo_dir), "stager": a.stager, "mods": mods,
                 "fs": {m: float(args0[m].get("load_fs") or args0[m].get("fs") or 100) for m in mods},
                 "vq_levels": {m: models[m].vq_levels for m in mods}, "n_codes": {m: models[m].n_codes for m in mods}})
    t0 = time.time()

    def save(unit, sid, arr, lat, ts, y, events):
        if not arr:
            return
        lp = _stage_logp(stagers, lat, mods, dev)
        d = os.path.join(a.out, unit)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"{sid}.npz"), ts=ts.astype(np.int32), y=y.astype(np.int8),
                 **{f"logp_{k}": v for k, v in lp.items()}, **arr)
        meta["nights"][f"{unit}/{sid}"] = {"unit": unit, "site": BY_NAME.get(unit, {}).get("site", unit),
                                           "n_epochs": int(len(ts)), "mods": [m for m in mods if f"P_{m}" in arr],
                                           "events": events}
        json.dump(meta, open(mp, "w"))

    units = a.units.split(",") if a.units else list(UNITS)
    if units:
        tests = {m: build_test(a.splits, [args0[m]["channel"]], channel_pipelines(args0[m]), args0[m]["seq"],
                               _cache_dir(a.cache_dir, args0[m]), units=units, memmap_cache_size=4,
                               trim_mode="main_period") for m in mods}
        index = {m: {(tests[m].unit_names[d2], s2): d2 for d2, s2, _, _ in tests[m].test} for m in mods}
        for unit in units:
            sids = [s for (u, s) in index["eeg"] if u == unit]
            for sid in _pick(sids, a.per_unit, a.seed):
                if f"{unit}/{sid}" in meta["nights"]:
                    continue
                readers = {m: Nights(tests[m].multi._datasets[index[m][unit, sid]], args0[m])
                           for m in mods if (unit, sid) in index[m]}
                arr, lat, ts, y = _night_arrays(models, readers, {m: sid for m in mods}, mods, half_of)
                if ts is None or len(ts) < 21:
                    continue
                save(unit, sid, arr, lat, ts, y, _events(readers["eeg"].ds, sid))
                print(f"[dump] {unit} {sid} {len(ts)} epochs, mods {sorted(k[2:] for k in arr if k.startswith('P_'))}, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if a.dreams:
        from physioex.data.datasets import get_dataset

        ra = args0["eeg"]
        tds = get_dataset("dreams")(channels=[ra.get("channel", "EEG_clean")], pipelines=channel_pipelines(ra),
                                    sequence_length=ra["seq"], cache_dir=_cache_dir(a.cache_dir, ra))
        r = Nights(tds, ra)
        for spec in tds._subjects:
            sid = spec.subject_id
            if f"dreams/{sid}" in meta["nights"]:
                continue
            x, y, _ = r.get(sid)
            ts = np.flatnonzero(y >= 0)  # every scored epoch of the excerpt, edge windows by reflection
            arr, lat, ts, y2 = _night_arrays(models, {"eeg": _Fixed(x, y, ts)}, {"eeg": sid}, ["eeg"], half_of)
            save("dreams", sid, arr, lat, ts, y2, _events(tds, sid))
            print(f"[dump] dreams {sid} {len(ts)} epochs", flush=True)
    print(f"[dump] done after {(time.time() - t0) / 60:.1f} min, {len(meta['nights'])} nights -> {a.out}", flush=True)
    print("DONE", flush=True)


def robust_stats(x, k=1.349, eps=1e-6):
    """Median and scale of ``RobustScale`` (``physioex.data.steps.normalize``) on one night."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32).astype(np.float64))
    v = x.reshape(-1)
    v = v[:: max(1, len(v) // 2_000_000)]
    v = v[v != 0]
    if len(v) < 100:
        v = x.reshape(-1)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return float(med), float((q3 - q1) / k + eps)


def _scale_night(job):
    """Median / scale of every modality of one dumped night (one cleaner run gives all channels)."""
    key, dump_dir, cache_dir, chans, mods = job
    from physioex.data.datasets import get_dataset
    from physioex.models.recon_eval import channel_pipelines
    from physioex.train.bin.pretrain_combo import _cache_dir, open_unit

    unit, sid = key.split("/")
    a = chans["eeg"]
    t0 = time.time()
    if unit == "dreams":
        ds = get_dataset("dreams")(channels=[a["channel"]], pipelines=channel_pipelines(a), sequence_length=a["seq"],
                                   cache_dir=_cache_dir(cache_dir, a))
    else:
        ds = open_unit(unit, a, cache_dir)
    cn = ds._clean_night({s.subject_id: s for s in ds._subjects}[sid])
    z = np.load(os.path.join(dump_dir, f"{key}.npz"))
    out = {}
    for m in mods:
        raw, fs = cn.outputs[chans[m]["channel"]]
        med, k = robust_stats(raw)
        err = None
        spe = int(round(30 * fs))
        if f"x_{m}" in z and z[f"x_{m}"].shape[1] == spe:
            ep = np.asarray(raw, np.float32).astype(np.float64)
            ep = ep[: len(ep) // spe * spe].reshape(-1, spe)
            ts = z["ts"]
            if ts.max() < len(ep):
                err = float(np.abs(np.clip((ep[ts] - med) / k, -20, 20) - z[f"x_{m}"]).max())
        out[m] = {"median": med, "scale": k, "fs_clean": float(fs), "maxerr": err}
    return key, out, time.time() - t0


def scale(a):
    from concurrent.futures import ProcessPoolExecutor

    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    chans = {m: json.load(open(os.path.join(meta["combo_dir"], m, "s2", "run.json")))["args"] for m in meta["mods"]}
    sp = os.path.join(a.dump, "scale.json")
    done = json.load(open(sp)) if os.path.exists(sp) else {}
    jobs = [(k, a.dump, a.cache_dir, chans, v["mods"]) for k, v in meta["nights"].items() if k not in done]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (key, out, dt) in enumerate(ex.map(_scale_night, jobs)):
            done[key] = out
            bad = [m for m, o in out.items() if o["maxerr"] is not None and o["maxerr"] > 0.05]
            print(f"[scale] {i + 1}/{len(jobs)} {key} {dt:.0f}s "
                  + " ".join(f"{m}:s={o['scale']:.1f}" for m, o in out.items()) + (f" MISMATCH {bad}" if bad else ""),
                  flush=True)
            if i % 10 == 0:
                json.dump(done, open(sp, "w"))
    json.dump(done, open(sp, "w"))
    print(f"[scale] done after {(time.time() - t0) / 60:.1f} min -> {sp}", flush=True)
    print("DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump", help="codes, P, signal and stage log-probabilities per night (GPU)")
    d.add_argument("--combo-dir", required=True, help="pretrain_combo runs/<tag> (with <mod>/s2 and the stager)")
    d.add_argument("--stager", default="stager_full")
    d.add_argument("--splits", required=True)
    d.add_argument("--cache-dir", default=None)
    d.add_argument("--mods", default="eeg,eog,emg")
    d.add_argument("--units", default=None, help="comma list (default: UNITS; empty string: none)")
    d.add_argument("--per-unit", type=int, default=20, help="test nights per unit (<= 0: all)")
    d.add_argument("--dreams", action="store_true", help="also the DREAMS excerpts (EEG only)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", required=True)
    c = sp.add_parser("scale", help="per-night median / scale of the robust-scaled dump (CPU, reruns the cleaner)")
    c.add_argument("--dump", required=True)
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    {"dump": dump, "scale": scale}[a.cmd](a)


if __name__ == "__main__":
    main()
