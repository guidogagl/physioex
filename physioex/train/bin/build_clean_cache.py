"""Fill the clean-channel cache of a dataset and write a per-subject report.

Asks every subject for the clean channels with the standard clean pipelines
(:func:`physioex.data.steps.clean_pipelines`), so the cleaned night (``NIGHT_<tag>``) and the
pipelined arrays land in the cache as a training run would read them. The report has, per
subject: epochs, label counts, events per type/concept and the cleaning diagnostics (level,
resolved derivations, window, steps run or skipped).

``python -m physioex.train.bin.build_clean_cache --dataset dreams --out report.json``
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--channels", nargs="+", default=["EEG_clean", "EOG_clean", "EMG_clean"])
    ap.add_argument("--trim-mode", default="excess_wake", choices=("excess_wake", "main_period"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from physioex.data.datasets import get_dataset
    from physioex.data.steps import clean_pipelines

    kw = dict(channels=a.channels, pipelines=clean_pipelines(), sequence_length=1, trim_mode=a.trim_mode)
    for k in ("subset", "root", "cache_dir"):
        if getattr(a, k):
            kw[k] = getattr(a, k)
    ds = get_dataset(a.dataset)(**kw)
    rep = {"dataset": a.dataset, "channels": a.channels, "subjects": {}}
    for spec in ds._subjects:
        sid = spec.subject_id
        t0 = time.time()
        n = ds._n_epochs[sid]
        item = ds._build_item(spec, 0, n)
        y = np.asarray(ds._get_labels(spec) if hasattr(ds, "_get_labels") else [])
        ev = ds._load_or_compute_events(spec)
        r = {"n_epochs": int(n), "labels": dict(Counter(int(v) for v in y)),
             "events": dict(Counter(f"{e.type}/{e.concept}" for e in ev)),
             "signal_std": {k: float(v.float().std()) for k, v in item["signals"].items()},
             "diagnostics": ds.get_clean_diagnostics(sid), "seconds": round(time.time() - t0, 1)}
        rep["subjects"][sid] = r
        d = r["diagnostics"] or {}
        print(f"[build_clean_cache] {sid}: {n} epochs, level={d.get('level')}, resolved={d.get('resolved')}, "
              f"events={r['events']}, {r['seconds']} s", flush=True)
    json.dump(_jsonable(rep), open(a.out, "w"), indent=1, default=str)
    print("[build_clean_cache] DONE", len(rep["subjects"]), "subjects", flush=True)


if __name__ == "__main__":
    main()
