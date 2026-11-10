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

- ``refs`` (CPU): the reference masks of every dumped night (:func:`physioex.explain.concept_refs.night_refs`,
  uV from ``scale.json``), ``<out>/refs/<unit>/<sid>.npz``.
- ``scan`` (CPU): discrete codebook components of every concept (:data:`TARGETS`). Within the concept's
  stratum (its stages, so that a stage code does not pass for a concept code) every (scale, RVQ level,
  code) gets count, hits, purity, lift; up to 3 codes are chosen greedily by token (or epoch) F1 of
  their union. Leave-one-item-out on the pool (held-out F1, how often each chosen code comes back),
  null = the same selection on references shifted circularly within every night (``--nulls``), the
  within-stage lift of the chosen union, and the transfer of the pool selection to the out-of-pool
  sets (shhs, MASS SS01, DREAMS: expert spindles / K-complexes against the rule-based selection).
  ``<out>/scan.json``.
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
#: out-of-pool units: transfer only, never in the selection
EXTERNAL = ("shhs_visit1", "mass_ss01", "dreams")
_ALL = (0, 1, 2, 3, 4)
_SLEEP = (1, 2, 3, 4)
#: concept -> (reference mask, modalities scanned, stages of the stratum (W 0, N1 1, N2 2, N3 3, R 4),
#: level "tok" (P codes, 0.5 s) or "epoch" (N / E codes), selection set "pool" or "dreams")
TARGETS = {
    "spindle": ("spindle_rule", ("eeg",), (2,), "tok", "pool"),
    "spindle_expert": ("spindle_vis1", ("eeg",), (2, 3), "tok", "dreams"),
    "k_complex": ("k_complex_vis1", ("eeg",), (2, 3), "tok", "dreams"),
    "slow_wave": ("slow_wave", ("eeg",), (2, 3), "tok", "pool"),
    "delta": ("delta", ("eeg",), (2, 3), "tok", "pool"),
    "delta_epoch": ("delta_epoch", ("eeg",), _ALL, "epoch", "pool"),
    "vertex": ("vertex", ("eeg",), (1, 2), "tok", "pool"),
    "sawtooth": ("sawtooth", ("eeg",), (4,), "tok", "pool"),
    "alpha": ("alpha", ("eeg",), (0, 1), "tok", "pool"),
    "alpha_epoch": ("alpha_epoch", ("eeg",), _ALL, "epoch", "pool"),
    "lamf": ("lamf", ("eeg",), (1, 4), "tok", "pool"),
    "arousal": ("arousal_ann", ("eeg", "emg"), _SLEEP, "tok", "pool"),
    "resp_end": ("resp_end", ("eeg", "emg"), _SLEEP, "tok", "pool"),
    "plm_arousal": ("plm_arousal", ("eeg", "emg"), _SLEEP, "tok", "pool"),
    "rem_eye": ("rem_eye", ("eog",), (0, 4), "tok", "pool"),
    "sem": ("sem", ("eog",), (0, 1), "tok", "pool"),
    "blink": ("blink", ("eog",), (0,), "tok", "pool"),
    "atonia": ("atonia", ("emg",), _ALL, "tok", "pool"),
    "atonia_epoch": ("atonia_epoch", ("emg",), _ALL, "epoch", "pool"),
    "tma": ("tma", ("emg",), (4,), "tok", "pool"),
    "emg_rise": ("emg_rise", ("emg",), _ALL, "tok", "pool"),
    "artifact": ("artifact_eeg", ("eeg",), _ALL, "tok", "pool"),
    # terms added from the scoring manuals (LEXICON_DEFINIZIONI.md, 08.11.2026)
    "k_complex_rule": ("k_complex_rule", ("eeg",), (2,), "tok", "pool"),
    "alpha_rem": ("alpha_rem", ("eeg",), (4,), "tok", "pool"),
    "alpha_nrem": ("alpha_nrem", ("eeg",), (2, 3), "tok", "pool"),
    "cap_a1": ("cap_a1", ("eeg",), (1, 2, 3), "tok", "pool"),
    "cap_a3": ("cap_a3", ("eeg",), (1, 2, 3), "tok", "pool"),
    "rmma": ("rmma", ("emg",), _SLEEP, "tok", "pool"),
}


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
        if not z["codes"]:  # continuous factors (no codebook): the epoch latents themselves
            arr[f"N_{m}"], arr[f"E_{m}"] = z["N"], z["E"]
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


#: expert references the selection is also scored against (transfer), and the scorer pair of the ceiling
EXPERT = {"spindle": ("spindle_vis1", "spindle_vis2"), "spindle_expert": ("spindle_vis2",),
          "k_complex": ("k_complex_vis2",), "k_complex_rule": ("k_complex_vis1", "k_complex_vis2")}
EXPERT_PAIR = {"spindle": ("spindle_vis1", "spindle_vis2"), "spindle_expert": ("spindle_vis1", "spindle_vis2"),
               "k_complex": ("k_complex_vis1", "k_complex_vis2")}


def _refs_night(job):
    from physioex.explain.concept_refs import night_refs

    key, dump_dir, entry = job
    z = dict(np.load(os.path.join(dump_dir, f"{key}.npz")))
    r = night_refs(z, entry)
    out = os.path.join(dump_dir, "refs", f"{key}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: np.packbits(v) for k, v in r.items()}, **{f"_len_{k}": len(v) for k, v in r.items()})
    return key, sorted(r)


def refs(a):
    from concurrent.futures import ProcessPoolExecutor

    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    sc = json.load(open(os.path.join(a.dump, "scale.json")))
    jobs = []
    for k, v in meta["nights"].items():
        if not a.force and os.path.exists(os.path.join(a.dump, "refs", f"{k}.npz")):
            continue
        e = dict(v, fs=meta["fs"], scale={m: [o["median"], o["scale"]] for m, o in sc.get(k, {}).items()})
        jobs.append((k, a.dump, e))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (key, names) in enumerate(ex.map(_refs_night, jobs, chunksize=2)):
            if i % 20 == 0:
                print(f"[refs] {i + 1}/{len(jobs)} {key} {len(names)} masks, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[refs] done after {(time.time() - t0) / 60:.1f} min", flush=True)
    print("DONE", flush=True)


def load_refs(path):
    z = np.load(path)
    return {k: np.unpackbits(z[k])[: int(z[f"_len_{k}"])].astype(bool) for k in z.files if not k.startswith("_len_")}


def _group(unit):
    from physioex.data.merged import BY_NAME

    return "dreams" if unit == "dreams" else (BY_NAME[unit].get("item") or unit)


def load_scan_data(dump_dir, mod, level, ref_names):
    """Codes, stage, night and group of every token (``level="tok"``: P codes, ``(n*60, L)``) or epoch
    (``"epoch"``: N and E codes side by side) of ``mod``, and the reference masks (``None`` where a
    night has no such reference)."""
    meta = json.load(open(os.path.join(dump_dir, "meta.json")))
    C, Y, G, NI, R = [], [], [], [], {r: [] for r in ref_names}
    groups, cols = [], None
    for i, (key, e) in enumerate(sorted(meta["nights"].items())):
        if mod not in e["mods"]:
            continue
        z = np.load(os.path.join(dump_dir, f"{key}.npz"))
        rf = load_refs(os.path.join(dump_dir, "refs", f"{key}.npz"))
        y = z["y"].astype(np.int8)
        if level == "tok":
            c = z[f"codes_{mod}_P"]
            c = c.reshape(-1, c.shape[-1])
            cols = cols or [f"P{l}" for l in range(c.shape[1])]
            y = np.repeat(y, c.shape[0] // len(y))
        else:
            sc = [k for k in ("N", "E") if f"codes_{mod}_{k}" in z.files]
            c = np.concatenate([z[f"codes_{mod}_{k}"] for k in sc], 1)
            cols = cols or [f"{k}{l}" for k in sc for l in range(z[f"codes_{mod}_{k}"].shape[1])]
        g = _group(e["unit"])
        if g not in groups:
            groups.append(g)
        C.append(c.astype(np.int16))
        Y.append(y)
        G.append(np.full(len(c), groups.index(g), np.int16))
        NI.append(np.full(len(c), i, np.int32))
        for r in ref_names:
            R[r].append(rf[r] if r in rf and len(rf[r]) == len(c) else None)
    out_r = {}
    for r, parts in R.items():
        has = np.concatenate([np.full(len(c), p is not None) for c, p in zip(C, parts)])
        val = np.concatenate([p if p is not None else np.zeros(len(c), bool) for c, p in zip(C, parts)])
        out_r[r] = (val, has)
    return {"codes": np.concatenate(C), "y": np.concatenate(Y), "group": np.concatenate(G),
            "night": np.concatenate(NI), "groups": groups, "cols": cols, "refs": out_r}


def _f1(sel_mask, ref):
    h = np.count_nonzero(sel_mask & ref)
    n, H = np.count_nonzero(sel_mask), np.count_nonzero(ref)
    return 2 * h / (n + H) if n + H else 0.0


def greedy_codes(codes, ref, n_codes, k=3, n_cand=30, min_n=50, min_gain=0.005):
    """Up to ``k`` (column, code) pairs whose union maximizes F1 against ``ref`` (tokens already
    restricted to the stratum). Candidates: the ``n_cand`` best single codes by F1 with >= ``min_n`` uses."""
    H = np.count_nonzero(ref)
    if H == 0:
        return [], 0.0
    cand = []
    for l in range(codes.shape[1]):
        n = np.bincount(codes[:, l], minlength=n_codes)
        h = np.bincount(codes[:, l], weights=ref, minlength=n_codes)
        f = np.where(n >= min_n, 2 * h / (n + H), 0)
        for c in np.argsort(f)[::-1][:n_cand]:
            if f[c] > 0:
                cand.append((float(f[c]), l, int(c)))
    cand = sorted(cand, reverse=True)[:n_cand]
    masks = {(l, c): codes[:, l] == c for _, l, c in cand}
    sel, cur, best = [], np.zeros(len(ref), bool), 0.0
    for _ in range(k):
        scores = {lc: _f1(cur | m, ref) for lc, m in masks.items() if lc not in sel}
        if not scores:
            break
        lc, f = max(scores.items(), key=lambda t: t[1])
        if f - best < min_gain:
            break
        sel.append(lc)
        cur |= masks[lc]
        best = f
    return sel, best


def union_mask(codes, sel):
    m = np.zeros(len(codes), bool)
    for l, c in sel:
        m |= codes[:, l] == c
    return m


def code_lift(codes, ref, n_codes, prior=20.0):
    """Lift of every code of every column (``(L, n_codes)``), shrunk towards 1 with ``prior`` pseudo-uses."""
    b = max(ref.mean(), 1e-9)
    out = np.ones((codes.shape[1], n_codes))
    for l in range(codes.shape[1]):
        n = np.bincount(codes[:, l], minlength=n_codes)
        h = np.bincount(codes[:, l], weights=ref, minlength=n_codes)
        out[l] = (h + prior * b) / (n + prior) / b
    return out


def average_precision(score, ref):
    """AP of ``score`` against ``ref``; tied scores form one block (no order inside a code)."""
    H = np.count_nonzero(ref)
    if H == 0:
        return float("nan")
    u, inv = np.unique(-score, return_inverse=True)
    n = np.bincount(inv, minlength=len(u))
    h = np.bincount(inv, weights=ref, minlength=len(u))
    cn, ch = np.cumsum(n), np.cumsum(h)
    return float(np.sum(h / H * ch / cn))


def nb_score(codes, lift, cols=None):
    """Token score: sum of the log-lifts of its codes over ``cols`` (naive Bayes over the RVQ levels)."""
    cols = range(codes.shape[1]) if cols is None else cols
    return sum(np.log(lift[l][codes[:, l]]) for l in cols)


def family(codes, ref, n_codes, min_lift=2.0, min_n=50, fdr=0.01):
    """The concept's code family, per column: codes with lift >= ``min_lift``, >= ``min_n`` uses and an
    enrichment that survives a one-sided binomial test (hits given uses at the base rate) with
    Benjamini-Hochberg at ``fdr`` over the column's tested codes (the rare codes of the deep RVQ levels
    pass the lift alone by chance)."""
    from scipy.stats import binom

    b = max(ref.mean(), 1e-9)
    lift = code_lift(codes, ref, n_codes)
    fam = []
    for l in range(codes.shape[1]):
        n = np.bincount(codes[:, l], minlength=n_codes)
        h = np.bincount(codes[:, l], weights=ref, minlength=n_codes).round().astype(int)
        cand = np.flatnonzero((lift[l] >= min_lift) & (n >= min_n))
        if not len(cand):
            continue
        tested = np.flatnonzero(n >= min_n)
        p = binom.sf(h[tested] - 1, n[tested], b)
        order = np.argsort(p)
        thr = fdr * np.arange(1, len(p) + 1) / len(p)
        ok = p[order] <= thr
        k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
        sig = set(tested[order[:k]].tolist())
        fam += [(l, int(c)) for c in cand if int(c) in sig]
    return fam, lift


def scan_family(codes, ref, y, key, stages, n_codes, cols, n_nulls, night, rng, min_n=50):
    """Held-out AP of the code-lift score (per column and naive Bayes over all columns), its null on
    references shifted within the nights, and the family of the whole selection set."""
    folds = [f for f in np.unique(key) if ref[key == f].any()]

    def heldout(r):
        per = {c: [] for c in cols + ["NB"]}
        for f in folds:
            tr, te = key != f, key == f
            lift = code_lift(codes[tr], r[tr], n_codes)
            for l, c in enumerate(cols):
                per[c].append(average_precision(lift[l][codes[te, l]], r[te]))
            per["NB"].append(average_precision(nb_score(codes[te], lift), r[te]))
        return {c: float(np.nanmean(v)) if v else float("nan") for c, v in per.items()}

    real = heldout(ref)
    nulls = [heldout(_shift_within_nights(ref, night, rng)) for _ in range(n_nulls)]
    fam, _ = family(codes, ref, n_codes, min_n=min_n)
    m = union_mask(codes, fam)
    base = float(ref.mean())
    out = {"ap": real, "ap_base": base,
           "ap_null_q95": {c: float(np.nanquantile([n[c] for n in nulls], 0.95)) for c in real} if nulls else None,
           "family_size": {c: sum(1 for l, _ in fam if cols[l] == c) for c in cols},
           "family": _describe(codes, ref, y, [], stages)["union"] if not fam else None}
    if fam:
        out["family"] = _describe(codes, ref, y, fam, stages)["union"]
    return out, fam


def _describe(codes, ref, y, sel, stages):
    """Purity, coverage, lift in the stratum and the within-stage lift (mean over the stratum's stages,
    weighted by the reference) of the union and of every chosen code."""
    def one(m):
        base = ref.mean() if len(ref) else 0.0
        h, n, H = np.count_nonzero(m & ref), np.count_nonzero(m), np.count_nonzero(ref)
        lw, wsum = 0.0, 0
        for s in stages:
            k = y == s
            hs = np.count_nonzero(ref & k)
            if hs == 0 or not (m & k).any():
                continue
            lw += hs * ((m & k & ref).sum() / (m & k).sum()) / (hs / k.sum())
            wsum += hs
        return {"n": int(n), "purity": h / n if n else 0.0, "coverage": h / H if H else 0.0,
                "lift": (h / n) / base if n and base else 0.0, "lift_within": lw / wsum if wsum else None,
                "f1": _f1(m, ref)}
    return {"union": one(union_mask(codes, sel)), "codes": [one(union_mask(codes, [lc])) for lc in sel]}


def _shift_within_nights(ref, night, rng):
    """Circular shift of the reference within every night (``night``: contiguous blocks)."""
    out = np.empty_like(ref)
    e = np.concatenate([[0], np.flatnonzero(np.diff(night)) + 1, [len(night)]])
    for a, b in zip(e[:-1], e[1:]):
        out[a:b] = np.roll(ref[a:b], rng.integers(1, max(2, b - a)))
    return out


def scan_concept(name, D, n_codes, n_nulls=20, seed=0, min_n=50):
    ref_name, _, stages, _, sel_set = TARGETS[name]
    ref, has = D["refs"][ref_name]
    grp = np.array(D["groups"])[D["group"]]
    ext = np.isin(grp, ["shhs", "mass_ss01", "dreams"])
    strat = np.isin(D["y"], stages) & has
    fit = strat & ((grp == "dreams") if sel_set == "dreams" else ~ext)
    if not fit.any() or not ref[fit].any():
        return {"concept": name, "ref": ref_name, "status": "no reference in the selection set"}
    codes, y, night = D["codes"], D["y"], D["night"]
    sel, f1 = greedy_codes(codes[fit], ref[fit], n_codes)
    res = {"concept": name, "ref": ref_name, "stages": list(stages), "cols": D["cols"], "selection_set": sel_set,
           "base_rate": float(ref[fit].mean()), "n_units": int(fit.sum()),
           "selected": [{"col": D["cols"][l], "code": c} for l, c in sel], "f1": f1,
           "describe": _describe(codes[fit], ref[fit], y[fit], sel, stages)}
    # held-out folds: groups of the pool (items) or DREAMS excerpts
    folds = sorted(set(grp[fit])) if sel_set == "pool" else sorted(set(night[fit]))
    key = grp if sel_set == "pool" else night
    loo, back = [], {f"{D['cols'][l]}:{c}": 0 for l, c in sel}
    for f in folds:
        tr, te = fit & (key != f), fit & (key == f)
        if not ref[te].any():
            continue
        s2, _ = greedy_codes(codes[tr], ref[tr], n_codes)
        loo.append({"fold": str(f), "f1": _f1(union_mask(codes[te], s2), ref[te]),
                    "f1_full_sel": _f1(union_mask(codes[te], sel), ref[te])})
        for l, c in s2:
            k = f"{D['cols'][l]}:{c}"
            if k in back:
                back[k] += 1
    res["loo"] = loo
    fam_res, fam = scan_family(codes[fit], ref[fit], y[fit], key[fit], stages, n_codes, D["cols"], n_nulls,
                               night[fit], np.random.default_rng(seed + 1), min_n)
    res["family"] = fam_res
    # code-lift score transferred to the out-of-pool units (AP, base rate)
    lift_full = code_lift(codes[fit], ref[fit], n_codes)
    res["stability"] = {k: v / max(1, len(loo)) for k, v in back.items()}
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_nulls):
        rs = _shift_within_nights(ref[fit], night[fit], rng)
        nulls.append(greedy_codes(codes[fit], rs, n_codes)[1])
    res["null_f1"] = {"q95": float(np.quantile(nulls, 0.95)) if nulls else None, "max": max(nulls, default=None)}
    # transfer of the selection to the out-of-pool units with this reference
    tr_out = {}
    for g in ("shhs", "mass_ss01", "dreams"):
        m = strat & (grp == g)
        if g == sel_set or not m.any() or not ref[m].any():
            continue
        tr_out[g] = _describe(codes[m], ref[m], y[m], sel, stages)["union"]
        tr_out[g].update(ap_nb=average_precision(nb_score(codes[m], lift_full), ref[m]), base=float(ref[m].mean()),
                         family=_describe(codes[m], ref[m], y[m], fam, stages)["union"])
    # expert references of the same concept (DREAMS): the selection against each scorer, and the
    # ceiling = agreement between the two scorers on the same tokens
    for e in EXPERT.get(name, ()):
        if e not in D["refs"]:
            continue
        er, eh = D["refs"][e]
        m = np.isin(y, stages) & eh
        if m.any() and er[m].any():
            tr_out[f"expert:{e}"] = _describe(codes[m], er[m], y[m], sel, stages)["union"]
            tr_out[f"expert:{e}"].update(ap_nb=average_precision(nb_score(codes[m], lift_full), er[m]),
                                         base=float(er[m].mean()))
    pair = EXPERT_PAIR.get(name)
    if pair and all(p in D["refs"] for p in pair):
        (r1, h1), (r2, h2) = D["refs"][pair[0]], D["refs"][pair[1]]
        m = np.isin(y, stages) & h1 & h2
        if m.any():
            res["ceiling_f1"] = _f1(r1[m], r2[m])
    res["transfer"] = tr_out
    return res


def scan(a):
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    names = a.concepts.split(",") if a.concepts else list(TARGETS)
    sp = os.path.join(a.dump, f"scan{'_' + a.tag if a.tag else ''}.json")
    out = json.load(open(sp)) if os.path.exists(sp) else {}
    t0 = time.time()
    jobs = {}
    for name in names:
        ref_name, mods, _, level, _ = TARGETS[name]
        for m in mods:
            jobs.setdefault((m, level), []).append((name, ref_name))
    for (m, level), items in jobs.items():
        refs_needed = {r for _, r in items} | {e for n, _ in items for e in EXPERT.get(n, ())}
        D = load_scan_data(a.dump, m, level, sorted(refs_needed))
        print(f"[scan] {m}/{level}: {len(D['codes'])} units, groups {D['groups']}, {(time.time() - t0) / 60:.1f} min", flush=True)
        for name, _ in items:
            r = scan_concept(name, D, meta["n_codes"][m]["P" if level == "tok" else "E"], a.nulls, a.seed)
            out[f"{name}/{m}"] = r
            json.dump(out, open(sp, "w"), indent=1)
            u = r.get("describe", {}).get("union", {})
            fr = r.get("family", {})
            ap, apn = fr.get("ap", {}), fr.get("ap_null_q95") or {}
            print(f"[scan] {name}/{m}: AP NB {ap.get('NB', np.nan):.3f} (null95 {apn.get('NB', np.nan):.3f}, "
                  f"base {fr.get('ap_base', np.nan):.3f}) best col "
                  f"{max(((c, v) for c, v in ap.items() if c != 'NB'), key=lambda t: t[1], default=None)} "
                  f"family {sum((fr.get('family_size') or {}).values())} codes "
                  f"lift {((fr.get('family') or {}).get('lift') or 0):.2f} within {(fr.get('family') or {}).get('lift_within')}"
                  f" transfer { {g: round(t.get('ap_nb', np.nan), 3) for g, t in r.get('transfer', {}).items()} }",
                  flush=True)
            print(f"[scan] {name}/{m}: sel {r.get('selected')} f1 {r.get('f1', 0):.3f} null95 "
                  f"{r.get('null_f1', {}).get('q95')} lift {u.get('lift', 0):.2f} within {u.get('lift_within')} "
                  f"loo {np.mean([f['f1'] for f in r.get('loo', [])] or [np.nan]):.3f} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
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
    r = sp.add_parser("refs", help="reference masks of every dumped night (CPU)")
    r.add_argument("--dump", required=True)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--force", action="store_true", help="recompute (overwrite) the nights already done")
    s_ = sp.add_parser("scan", help="codebook components of every concept (CPU)")
    s_.add_argument("--dump", required=True)
    s_.add_argument("--concepts", default=None, help="comma list of TARGETS (default: all)")
    s_.add_argument("--nulls", type=int, default=20)
    s_.add_argument("--seed", type=int, default=0)
    s_.add_argument("--tag", default="", help="scan_<tag>.json (parallel jobs on disjoint concept lists)")
    a = ap.parse_args(argv)
    {"dump": dump, "scale": scale, "refs": refs, "scan": scan}[a.cmd](a)


if __name__ == "__main__":
    main()
