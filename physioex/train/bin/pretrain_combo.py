"""Pretraining-data study (04.11.2026): which combination of datasets generalizes best.

Every combination of 3-5 pool items (:data:`physioex.data.merged.POOL`, order and repetition
do not count) trains the three clean-channel encoders with the recipes of the DCSM models
(stage 1 MSAE, stage 2 MSAE-VQ initialized from stage 1) on the merged ProtoSleepNet train
split, with the same number of steps for every combination.

    python -m physioex.train.bin.pretrain_combo combos --sizes 3 --out combos_k3.txt
    python -m physioex.train.bin.pretrain_combo pack --combo-file combos_k3.txt --task 0 --per-gpu 2 \\
        --splits splits_29sites.json --cache-dir ... --out ROOT

    python -m physioex.train.bin.pretrain_combo stager --combo-file combos_k3.txt --task 0 --per-gpu 2 --out ROOT
    python -m physioex.train.bin.pretrain_combo eval --combo-file combos_k3.txt --task 0 --per-gpu 2 \\
        --splits splits_29sites.json --out ROOT

``pack`` runs ``--per-gpu`` combinations at once on the job's single GPU: one process per
combination and modality (stage 1 then stage 2), ``--workers-total`` data-loader workers shared
among them. Each finished stage leaves ``run.json`` in its directory and is skipped on a rerun.

``stager`` (after ``pack``): latents of the stage-2 models (eval prefixes :data:`VQ_KEY`) for
``--train-per-item`` train nights and ``--valid-per-item-nights`` val nights per item (the same
nights for every modality, a night without the channel is zero-filled), then the seqqformer of
the DCSM reference (``qformer_stager/multi/EnP_vq_dp64``) EEG-only and EEG+EOG+EMG (late fusion).
The latents stay in node-local scratch; the stager keeps the training statistics
(``stats_<mod>.npz``) so that ``eval`` can standardize new nights.

``aggregate``: matrix combination x group, fixed panel (clinical subgroups + MASS SS01, with and
without LSEC), out-of-domain and in-domain macro, bootstrap CI of the panel, OLS on the item
indicators.

``eval``: every test night of :func:`physioex.data.merged.build_test` (pool test splits, all
nights of the test-only units) is read once per modality and passed to every model of the task.
One JSONL row per (combination, night): staging (kappa, accuracy, macro F1, both stagers, zero
shot), reconstruction per modality on ``--recon-per-night`` epochs
(:func:`physioex.models.recon_eval_modality.metrics`: log band power per 5 s, low wave / EMG
envelope, event AUC), ``in_domain``. Code usage (perplexity, dead fraction per level) is pooled
per site into ``codes.json``.

Recipes (stage 1 / stage 2), copied from the DCSM runs:
  EEG  msae_w32_t05_dp64 / vq_r3_dp64 (N8E8P6, 4096 codes, whitened N)
  EOG  msae_eog_non / vq_eog_k1024 (no N, d_p 16, 1024 codes)
  EMG  msae_emg_non_r1 / vq_emg_r1_k1024 with the bands cut at 50 Hz (decision 04.11.2026:
       the cached EMG is 100-256 Hz depending on the dataset): model at 100 Hz, bands
       10-20/20-35/35-50 Hz, PSD heads 10-50 Hz, resampled when loaded (``--load-fs 100``).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

from physioex.data.merged import POOL

COMMON1 = ["--model", "msae", "--seq", "41", "--dim", "128", "--width", "32", "--token-sec", "0.5",
           "--decoder", "time", "--loss", "mse", "--low-hz", "4.0", "--low-weight", "1.0", "--latent-bits", "8",
           "--batch", "64", "--lr", "3e-4", "--grad-clip", "1.0", "--patience", "12", "--valid-ratio", "0.25",
           "--recon-per-night", "0"]
COMMON2 = ["--model", "msae_vq", "--vq-levels", "8", "8", "8", "--q-dropout", "0.5", "--restart-after", "30",
           "--batch", "64", "--lr", "1e-4", "--grad-clip", "1.0", "--patience", "12", "--valid-ratio", "0.25",
           "--recon-per-night", "0"]
RECIPES = {
    "eeg": (["--channel", "EEG_clean", "--d-p", "64", "--pow-hz", "4.5", "--texture", "--band-weight", "3.0"],
            ["--vq-codes", "4096", "--vq-codes-n", "4096", "--vq-whiten-n"]),
    "eog": (["--channel", "EOG_clean", "--d-p", "16", "--no-n", "--pow-hz", "4.5", "--texture", "--band-weight", "3.0",
             "--pow-bands", "0.3", "1", "1", "2", "2", "5", "5", "9", "9", "15", "15", "35"],
            ["--vq-codes", "1024", "--vq-codes-n", "1024"]),
    "emg": (["--channel", "EMG_clean", "--d-p", "8", "--no-n", "--fs", "100", "--load-fs", "100", "--recon", "texture",
             "--seg-sec", "0.5", "--band-weight", "1.0", "--pow-bands", "10", "20", "20", "35", "35", "50",
             "--psd-band", "10", "50"],
            ["--vq-codes", "1024", "--vq-codes-n", "1024"]),
}


#: eval prefix of RVQ levels per modality (the DCSM references)
VQ_KEY = {"eeg": "N8E8P6", "eog": "N8E8P6", "emg": "N8E8P3"}
#: seqqformer of qformer_stager/multi/EnP_vq_dp64 (DCSM reference)
STAGER = ["--arch", "seqqformer", "--d", "128", "--k", "8", "--layers", "2", "--heads", "4", "--dropout", "0.2",
          "--seq-len", "21", "--seq-layers", "2", "--epochs", "20", "--patience", "0", "--sched", "cosine",
          "--evals-per-epoch", "2", "--lr", "3e-4", "--batch", "672", "--seed", "0"]


def tag(items):
    return "+".join(sorted(items))


def combos(a):
    rows = [c for k in a.sizes for c in itertools.combinations(sorted(POOL), k)]
    with open(a.out, "w") as f:
        for c in rows:
            f.write(",".join(c) + "\n")
    print(f"{len(rows)} combinations of {sorted(POOL)} (sizes {a.sizes}) -> {a.out}")


def stage_cmd(a, items, mod, stage, out, workers):
    s1, s2 = RECIPES[mod]
    base = ["--items", ",".join(items), "--splits", a.splits, "--steps-per-epoch", str(a.steps_per_epoch),
            "--epochs", str(a.epochs), "--valid-per-item", str(a.valid_per_item), "--workers", str(workers),
            "--fold", str(a.seed), "--out", out]
    if a.cache_dir:
        base += ["--cache-dir", a.cache_dir]
    if stage == 1:
        return [sys.executable, "-m", "physioex.train.bin.msvq", *COMMON1, *s1, *base]
    # stage 2: architecture and data args come from the stage-1 run (MSAE_ARCH in msvq.py)
    return [sys.executable, "-m", "physioex.train.bin.msvq", *COMMON2, *s2, "--init", out.replace("/s2", "/s1"), *base]


def pack(a):
    """Run the combinations of this task, ``--per-gpu`` at a time, all modalities in parallel."""
    mine = _mine(a)
    mods = a.mods.split(",")
    n_proc = len(mine) * len(mods)
    workers = max(1, a.workers_total // max(1, n_proc))
    print(f"[pack] task {a.task}: {[tag(c) for c in mine]} x {mods}, {n_proc} processes, {workers} workers each",
          flush=True)
    procs = {}
    for items in mine:
        for mod in mods:
            d = os.path.join(a.out, tag(items), mod)
            os.makedirs(d, exist_ok=True)
            todo = []
            for st in (1, 2):
                out = os.path.join(d, f"s{st}")
                if os.path.exists(os.path.join(out, "run.json")):
                    continue
                todo.append(" ".join(_q(x) for x in stage_cmd(a, items, mod, st, out, workers)))
            if not todo:
                continue
            script = " && ".join(todo)
            log = open(os.path.join(d, "train.log"), "a")
            log.write(f"\n=== {time.strftime('%F %T')} job {os.environ.get('SLURM_JOB_ID')}\n{script}\n")
            log.flush()
            procs[(tag(items), mod)] = (subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT), log)
    t0 = time.time()
    fails = []
    for k, (p, log) in procs.items():
        rc = p.wait()
        log.close()
        print(f"[pack] {k} rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc:
            fails.append(k)
    for items in mine:
        d = os.path.join(a.out, tag(items))
        if all(os.path.exists(os.path.join(d, m, "s2", "run.json")) for m in mods):
            open(os.path.join(d, "TRAIN_DONE"), "w").write(json.dumps({"mods": mods, "job": os.environ.get("SLURM_JOB_ID")}))
    print(f"[pack] {'FAILED ' + str(fails) if fails else 'all ok'}", flush=True)
    print("DONE", flush=True)
    sys.exit(1 if fails else 0)


def _mine(a):
    rows = [l.strip().split(",") for l in open(a.combo_file) if l.strip()]
    return rows[a.task * a.per_gpu:(a.task + 1) * a.per_gpu] if a.task is not None else rows


# ----------------------------------------------------------------------------
# Models and whole nights
# ----------------------------------------------------------------------------


def load_s2(run_dir, mod, device):
    """Stage-2 model of a run (best validation checkpoint) with the eval prefix of ``mod``."""
    import torch

    from physioex.models.msae_vq import parse_vq_key
    from physioex.models.recon_eval import load_best, model_from_args

    run = json.load(open(os.path.join(run_dir, "run.json")))
    model = model_from_args(run["args"])
    if load_best(model, run_dir) is None:
        model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"), strict=False)
    model.vq_levels = parse_vq_key(VQ_KEY[mod])
    return run, model.to(device).eval()


def open_unit(unit, args, cache_dir=None):
    """Dataset of one unit with the channel and pipeline of a run (``args``)."""
    from physioex.data.merged import make_dataset
    from physioex.models.recon_eval import channel_pipelines

    return make_dataset(unit, [args["channel"]], channel_pipelines(args), args["seq"], cache_dir or args.get("cache_dir"),
                        memmap_cache_size=4, trim_mode="main_period")


class Nights:
    """Whole nights of one dataset at the model rate (``load_fs``: resampled). Callers pass only
    cached subject ids (from a run report or :func:`physioex.data.merged.build_test`): a night
    outside the cache would start the cleaner."""

    def __init__(self, ds, args):
        self.ds, self.have = ds, {s.subject_id: s for s in ds._subjects}
        self.fs = float(args.get("load_fs") or args.get("fs") or 100)
        self.half = args["seq"] // 2

    def get(self, sid):
        """``x`` ``(n, fs * 30)``, labels ``y`` ``(n,)`` and the scored epochs ``ts`` of the main
        sleep period whose window fits in the night."""
        from physioex.data.merged import _resample_poly

        ds = self.ds
        n = ds._n_epochs[sid]
        item = ds._build_item(self.have[sid], 0, n)
        key = next(iter(item["signals"]))
        x = item["signals"][key].numpy().reshape(n, -1).astype(np.float32)
        fs_in = float(item["channel_info"][key]["fs_out"])
        if abs(fs_in - self.fs) > 1e-3:
            x = _resample_poly(x.reshape(-1), fs_in, self.fs, n * int(round(self.fs * 30))).reshape(n, -1).astype(np.float32)
        y = item["labels"].numpy().reshape(-1)
        lo, hi = ds._main_period.get(sid, (0, n)) if hasattr(ds, "_main_period") else (0, n)
        ts = np.arange(max(lo, self.half), min(hi, n - self.half))
        return x, y, ts[y[ts] >= 0]


def encode_night(model, x, ts, half, batch=256, recon=None):
    """Latents ``N``, ``E``, ``P`` (float16) and codes per level of the epochs ``ts``; with
    ``recon`` (positions in ``ts``) also the reconstructed centre epoch ``wave`` of those."""
    import torch

    dev = next(model.parameters()).device

    def win(tb):
        return torch.from_numpy(np.stack([x[t - half:t + half + 1] for t in tb]))[:, :, None].to(dev)

    acc, codes, waves = {k: [] for k in "NEP"}, {k: [] for k in "NEP"}, []
    with torch.no_grad():
        for j in range(0, len(ts), batch):
            z = model.encode(win(ts[j:j + batch]))
            for k in "NEP":
                acc[k].append(z[k].float().cpu().numpy().astype(np.float16))
                if z["codes"].get(k) is not None:
                    codes[k].append(z["codes"][k].cpu().numpy())
        if recon is not None:
            tr = ts[recon]
            for j in range(0, len(tr), batch):
                xb = win(tr[j:j + batch])
                waves.append(model(xb)["wave"].float().reshape(len(xb), -1).cpu().numpy())
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["codes"] = {k: np.concatenate(v) for k, v in codes.items() if v}
    if waves:
        out["wave"] = np.concatenate(waves)
    return out


def _save_stats(path, st):
    np.savez(path, **{f"{k}_{i}": v[i] for k, v in st.items() for i in (0, 1)})


def _load_stats(path):
    z = np.load(path)
    return {k: (z[f"{k}_0"], z[f"{k}_1"]) for k in "NEP"}


# ----------------------------------------------------------------------------
# stager: zero-shot stagers on the latents of the combination's train nights
# ----------------------------------------------------------------------------


def stager_one(a, items):
    import shutil

    import torch

    from physioex.data.merged import units_of
    from physioex.models.qformer_stager import latent_stats

    d = os.path.join(a.out, tag(items))
    sd = os.path.join(d, "stager")
    if os.path.exists(os.path.join(sd, "STAGER_DONE")):
        print(f"[stager] {tag(items)}: done", flush=True)
        return 0
    mods = a.mods.split(",")
    assert mods[0] == "eeg", "EEG is the reference modality"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {m: json.load(open(os.path.join(d, m, "s2", "run.json")))["merged"] for m in mods}
    rng = np.random.default_rng(a.seed)
    nights = {}
    for split, key, k in (("train", "train_ids", a.train_per_item), ("valid", "val_ids", a.valid_nights_per_item)):
        rows = []
        for it in items:
            cand = [(u, s) for u in units_of([it]) for s in rep["eeg"][u][key]]
            rows += [cand[i] for i in sorted(rng.choice(len(cand), min(k, len(cand)), replace=False))]
        nights[split] = rows
    lat = os.path.join(a.scratch or os.environ.get("TMPDIR") or "/tmp", "pc_latents", tag(items))
    os.makedirs(sd, exist_ok=True)
    meta = {"items": list(items), "nights": nights, "n_tok": {}, "epochs": {}, "vq": {m: VQ_KEY[m] for m in mods}}
    t0 = time.time()
    for m in mods:
        run, model = load_s2(os.path.join(d, m, "s2"), m, dev)
        ok = {(u, s) for u, r in rep[m].items() if isinstance(r, dict) and "train_ids" in r
              for s in r["train_ids"] + r["val_ids"]}
        md = os.path.join(lat, m)
        os.makedirs(md, exist_ok=True)
        readers = {}
        for split, rows in nights.items():
            acc = {k: [] for k in ("N", "E", "P", "y", "night", "epoch")}
            for i, (u, s) in enumerate(rows):
                if (u, s) not in ok:  # e.g. Sleep-EDF without EMG: zero-filled by pack_modalities
                    continue
                r = readers.get(u) or readers.setdefault(u, Nights(open_unit(u, run["args"], a.cache_dir), run["args"]))
                x, y, ts = r.get(s)
                if not len(ts):
                    continue
                z = encode_night(model, x, ts, r.half)
                for k in "NEP":
                    acc[k].append(z[k])
                acc["y"].append(y[ts].astype(np.int8))
                acc["night"].append(np.full(len(ts), i, np.int32))
                acc["epoch"].append(ts.astype(np.int32))
            assert acc["y"], f"{tag(items)} {m} {split}: no nights"
            arr = {k: np.concatenate(v) for k, v in acc.items()}
            np.savez(os.path.join(md, f"latents_{split}.npz"), **arr)
            meta["epochs"].setdefault(m, {})[split] = int(len(arr["y"]))
            if split == "train":
                _save_stats(os.path.join(sd, f"stats_{m}.npz"), latent_stats(arr))
                meta["n_tok"][m] = int(arr["P"].shape[1])
        # the stager reads "test" once at the end: here the val nights (the real test is ``eval``)
        shutil.copyfile(os.path.join(md, "latents_valid.npz"), os.path.join(md, "latents_test.npz"))
        print(f"[stager] {tag(items)} {m}: latents {meta['epochs'][m]} after {(time.time() - t0) / 60:.1f} min", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(meta, open(os.path.join(sd, "nights.json"), "w"))
    py = [sys.executable, "-m", "physioex.models.qformer_stager", "train", *STAGER]
    jobs = {"eeg": py + ["--data", os.path.join(lat, "eeg"), "--out", os.path.join(sd, "eeg")]}
    if len(mods) > 1:
        jobs["multi"] = py + ["--mods", *[f"{m}={os.path.join(lat, m)}" for m in mods], "--fusion", "late",
                              "--out", os.path.join(sd, "multi")]
    procs = {k: subprocess.Popen(c, stdout=open(os.path.join(sd, f"{k}.log"), "w"), stderr=subprocess.STDOUT)
             for k, c in jobs.items()}
    rc = {k: p.wait() for k, p in procs.items()}
    print(f"[stager] {tag(items)}: {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    shutil.rmtree(lat, ignore_errors=True)
    if any(rc.values()):
        return 1
    open(os.path.join(sd, "STAGER_DONE"), "w").write(json.dumps({"job": os.environ.get("SLURM_JOB_ID")}))
    return 0


def stager(a):
    mine = [c for c in _mine(a) if os.path.exists(os.path.join(a.out, tag(c), "TRAIN_DONE"))]
    if a.only is not None:
        sys.exit(stager_one(a, mine[a.only]))
    print(f"[stager] task {a.task}: {[tag(c) for c in mine]}", flush=True)
    me = [sys.executable, "-m", "physioex.train.bin.pretrain_combo"] + [x for x in sys.argv[1:]]
    procs = [subprocess.Popen(me + ["--only", str(i)]) for i in range(len(mine))]
    rc = [p.wait() for p in procs]
    print(f"[stager] {'FAILED ' + str(rc) if any(rc) else 'all ok'}", flush=True)
    print("DONE", flush=True)
    sys.exit(1 if any(rc) else 0)


# ----------------------------------------------------------------------------
# eval: every test night once, every model of the task
# ----------------------------------------------------------------------------


def load_stagers(sd, mods, device):
    import torch

    from physioex.models.qformer_stager import ARCHS, MultiSeqStager

    meta = json.load(open(os.path.join(sd, "nights.json")))
    stats = {m: _load_stats(os.path.join(sd, f"stats_{m}.npz")) for m in mods}
    out = {}
    for kind in ("eeg", "multi"):
        f = os.path.join(sd, kind, "stager.json")
        if not os.path.exists(f):
            continue
        js = json.load(open(f))
        ar = js["args"]
        kw = dict(d=ar["d"], k=ar["k"], layers=ar["layers"], heads=ar["heads"], dropout=ar["dropout"],
                  seq_len=ar["seq_len"], seq_layers=ar["seq_layers"])
        if kind == "multi":
            model = MultiSeqStager(js["spec"], epoch=ar["arch"][3:], fusion=ar["fusion"], mod_dropout=ar["mod_dropout"], **kw)
        else:
            st = stats["eeg"]
            model = ARCHS[ar["arch"]](dim=st["N"][0].shape[-1], d_p=st["P"][0].shape[-1], n_tok=meta["n_tok"]["eeg"], **kw)
        model.load_state_dict(torch.load(os.path.join(sd, kind, "stager.pt"), map_location="cpu"))
        out[kind] = model.to(device).eval()
    return out, stats, meta


def _stage_metrics(pred, y):
    from sklearn.metrics import cohen_kappa_score, f1_score

    lab = list(range(5))
    return {"kappa": float(cohen_kappa_score(y, pred, labels=lab)), "acc": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=lab, average="macro", zero_division=0)),
            "f1": [float(v) for v in f1_score(y, pred, labels=lab, average=None, zero_division=0)]}


def _recon_summary(r):
    return {k: r[k] for k in ("logpow5", "wave_low", "env", "event_auc") if k in r}


def _empty_latents(stats, n_tok):
    return {"N": np.zeros((0,) + stats["N"][0].shape, np.float16), "E": np.zeros((0,) + stats["E"][0].shape, np.float16),
            "P": np.zeros((0, n_tok) + stats["P"][0].shape, np.float16), "y": np.zeros(0, np.int64),
            "night": np.zeros(0, np.int64), "epoch": np.zeros(0, np.int64)}


def _code_usage(hist, n_codes):
    """Perplexity and dead fraction per level from pooled code counts."""
    out = []
    for h in hist:
        p = h / max(1, h.sum())
        nz = p[p > 0]
        out.append({"perplexity": float(np.exp(-(nz * np.log(nz)).sum())), "dead": float((h == 0).mean()),
                    "n": int(h.sum()), "codes": int(n_codes)})
    return out


def evaluate(a):
    import torch

    from physioex.data.merged import BY_NAME, build_test
    from physioex.models.qformer_stager import pack_modalities, predict, standardize
    from physioex.models.recon_eval import channel_pipelines
    from physioex.models.recon_eval_modality import metrics as recon_metrics

    mods = a.mods.split(",")
    assert mods[0] == "eeg", "EEG is the reference modality"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mine = [c for c in _mine(a) if os.path.exists(os.path.join(a.out, tag(c), "stager", "STAGER_DONE"))]
    tags = [tag(c) for c in mine]
    items_of = {tag(c): set(c) for c in mine}
    models, args0 = {}, {}
    for c in tags:
        for m in mods:
            run, models[c, m] = load_s2(os.path.join(a.out, c, m, "s2"), m, dev)
            args0.setdefault(m, run["args"])
    stagers = {c: load_stagers(os.path.join(a.out, c, "stager"), mods, dev) for c in tags}
    units = a.units.split(",") if a.units else None
    tests, index, readers = {}, {}, {}
    for m in mods:
        ar = args0[m]
        tests[m] = build_test(a.splits, [ar["channel"]], channel_pipelines(ar), ar["seq"], a.cache_dir or ar.get("cache_dir"),
                              units=units, memmap_cache_size=4, trim_mode="main_period")
        index[m] = {(tests[m].unit_names[dd], s): dd for dd, s, _, _ in tests[m].test}
        readers[m] = {}
    ref = tests["eeg"]
    rows = ref.test
    if a.max_nights:  # smoke: the first nights of every unit
        seen = {}
        rows = [r for r in rows if seen.setdefault(r[0], []).append(r) or len(seen[r[0]]) <= a.max_nights]
    os.makedirs(os.path.join(a.out, "eval"), exist_ok=True)
    name = f"task{a.task}" if a.task is not None else "all"
    path = os.path.join(a.out, "eval", f"{name}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done.add((r["combo"], r["unit"], r["sid"]))
    print(f"[eval] task {a.task}: {tags} x {mods}, {len(rows)} test nights, {len(done)} rows already done; "
          + ", ".join(f"{n} {r['test_nights']}" for n, r in ref.report.items()), flush=True)
    hist = {}  # (combo, mod, site, scale) -> [counts per level]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    with open(path, "a") as fo:
        for i, (dd, sid, item, sub) in enumerate(rows):
            unit = ref.unit_names[dd]
            todo = [c for c in tags if (c, unit, sid) not in done]
            if not todo:
                continue
            site = BY_NAME[unit]["site"]
            lat = {c: {} for c in todo}
            rec = {c: {} for c in todo}
            n_ep = None
            for m in mods:
                if (unit, sid) not in index[m]:
                    continue
                dm = index[m][unit, sid]
                r = readers[m].get(dm) or readers[m].setdefault(dm, Nights(tests[m].multi._datasets[dm], args0[m]))
                x, y, ts = r.get(sid)
                if m == "eeg":
                    n_ep = len(ts)
                    if n_ep < 21:  # shorter than the stager sequence
                        break
                if not len(ts):
                    continue
                ridx = np.sort(rng.choice(len(ts), min(a.recon_per_night, len(ts)), replace=False))
                for c in todo:
                    model = models[c, m]
                    z = encode_night(model, x, ts, r.half, recon=ridx)
                    lat[c][m] = {"N": z["N"], "E": z["E"], "P": z["P"], "y": y[ts].astype(np.int64),
                                 "night": np.zeros(len(ts), np.int64), "epoch": ts.astype(np.int64)}
                    rec[c][m] = _recon_summary(recon_metrics(model, z["wave"], x[ts[ridx]], np.zeros(len(ridx))))
                    for k, cd in z["codes"].items():
                        cd = cd.reshape(-1, cd.shape[-1])
                        h = hist.setdefault((c, m, site, k), [np.zeros(model.n_codes[k], np.int64) for _ in range(cd.shape[1])])
                        for lv in range(cd.shape[1]):
                            h[lv] += np.bincount(cd[:, lv], minlength=model.n_codes[k])
            if n_ep is None or n_ep < 21:
                continue
            for c in todo:
                st_models, stats, meta = stagers[c]
                y = lat[c]["eeg"]["y"]
                stg = {}
                if "eeg" in st_models:
                    logits, _ = predict(st_models["eeg"], standardize({"x": lat[c]["eeg"]}, stats["eeg"])["x"], device=dev)
                    stg["eeg"] = _stage_metrics(logits.argmax(1).numpy(), y)
                if "multi" in st_models:
                    by = {m: {"x": lat[c].get(m) or _empty_latents(stats[m], meta["n_tok"][m])} for m in mods}
                    data, _, missing = pack_modalities(by, stats=stats)
                    logits, _ = predict(st_models["multi"], data["x"], device=dev)
                    stg["multi"] = _stage_metrics(logits.argmax(1).numpy(), y)
                    stg["multi"]["missing"] = missing["x"]
                fo.write(json.dumps({"combo": c, "unit": unit, "site": site, "subgroup": sub, "item": item, "sid": sid,
                                     "in_domain": item in items_of[c], "n_epochs": int(n_ep), "staging": stg,
                                     "recon": rec[c]}) + "\n")
            fo.flush()
            if i % 50 == 0:
                print(f"[eval] {i + 1}/{len(rows)} nights, {(time.time() - t0) / 60:.1f} min", flush=True)
    codes = {}
    for (c, m, site, k), h in hist.items():
        codes.setdefault(c, {}).setdefault(m, {}).setdefault(site, {})[k] = _code_usage(h, len(h[0]))
    cp = os.path.join(a.out, "eval", f"{name}_codes.json")
    if os.path.exists(cp) and done:  # resumed run: the counts cover only the new nights
        cp = cp.replace(".json", f"_resume{int(time.time())}.json")
    json.dump(codes, open(cp, "w"))
    print(f"[eval] done after {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)
    print("DONE", flush=True)


# ----------------------------------------------------------------------------
# aggregate: combination x group matrix, panel, trend
# ----------------------------------------------------------------------------

#: groups of the fixed always-out-of-domain panel (never in training)
PANEL = ("alzheimers_ad", "alzheimers_hc", "parkinsons_night_pd", "parkinsons_night_hoa", "parkinsons_nap_pd",
         "parkinsons_nap_hoa", "lsec_control", "lsec_wce", "lsec_dre", "mass_ss01")


def _scores(rows):
    """Per-night scores of one JSONL row: staging kappa of both stagers and the mean NEP log
    band power R^2 (5 s, mean over bands) per modality."""
    out = {}
    for k, v in rows["staging"].items():
        out[f"kappa_{k}"] = v["kappa"]
        out[f"f1_{k}"] = v["macro_f1"]
    for m, r in rows["recon"].items():
        if r.get("logpow5"):
            out[f"logpow5_{m}"] = float(np.mean(list(r["logpow5"].values())))
    return out


def aggregate(a):
    """Matrix combination x group (pool: item; test-only: subgroup) of the mean per-night scores;
    per combination the panel macro (with and without LSEC), the out-of-domain macro over the pool
    items not in training and the in-domain macro; bootstrap CI over nights (within group) of the
    panel; OLS of every macro on the item indicators (marginal contribution per item)."""
    import glob
    from collections import defaultdict

    rows = defaultdict(lambda: defaultdict(list))  # combo -> group -> [scores]
    for f in sorted(glob.glob(os.path.join(a.out, "eval", "*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            g = r["item"] if r["item"] is not None else r["subgroup"]
            rows[r["combo"]][g].append(_scores(r))
    keys = sorted({k for c in rows.values() for g in c.values() for s in g for k in s})
    rng = np.random.default_rng(a.seed)
    res = {"keys": keys, "panel": list(PANEL), "pool": sorted(POOL), "combos": {}}
    for c, groups in sorted(rows.items()):
        items = set(c.split("+"))
        mean = {g: {k: float(np.nanmean([s.get(k, np.nan) for s in v])) for k in keys} | {"n": len(v)}
                for g, v in groups.items()}
        macro = {}
        for name, gs in (("panel", [g for g in PANEL if g in groups]),
                         ("panel_no_lsec", [g for g in PANEL if g in groups and not g.startswith("lsec")]),
                         ("ood", [g for g in POOL if g in groups and g not in items]),
                         ("in_domain", [g for g in POOL if g in groups and g in items])):
            macro[name] = {k: float(np.nanmean([mean[g][k] for g in gs])) if gs else None for k in keys}
            macro[name]["groups"] = gs
        # bootstrap over nights within each panel group (the groups stay fixed)
        boot = {}
        for k in keys:
            gs = [g for g in PANEL if g in groups]
            if not gs:
                continue
            vals = [np.asarray([s.get(k, np.nan) for s in groups[g]], float) for g in gs]
            bs = [np.nanmean([np.nanmean(v[rng.integers(0, len(v), len(v))]) for v in vals]) for _ in range(a.n_boot)]
            boot[k] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        res["combos"][c] = {"items": sorted(items), "groups": mean, "macro": macro, "panel_ci95": boot}
    # trend: OLS of each macro on the inclusion indicators of the pool items (intercept = baseline)
    combos = sorted(res["combos"])
    pool = sorted(POOL)
    X = np.asarray([[1.0] + [float(it in res["combos"][c]["items"]) for it in pool] for c in combos])
    trend = {}
    for name in ("panel", "panel_no_lsec", "ood", "in_domain"):
        for k in keys:
            y = np.asarray([res["combos"][c]["macro"][name][k] if res["combos"][c]["macro"][name][k] is not None
                            else np.nan for c in combos])
            ok = np.isfinite(y)
            if ok.sum() <= X.shape[1]:
                continue
            beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            trend[f"{name}:{k}"] = {"intercept": float(beta[0]), **{it: float(b) for it, b in zip(pool, beta[1:])},
                                    "n_combos": int(ok.sum())}
    res["trend"] = trend
    out = os.path.join(a.out, "aggregate.json")
    json.dump(res, open(out, "w"), indent=1)
    best = sorted(combos, key=lambda c: -(res["combos"][c]["macro"]["panel"].get("kappa_multi") or -1))[:10]
    for c in best:
        mc = res["combos"][c]["macro"]
        print(f"{c:40s} panel kappa_multi {mc['panel'].get('kappa_multi')} ood {mc['ood'].get('kappa_multi')} "
              f"in {mc['in_domain'].get('kappa_multi')}", flush=True)
    print(f"[aggregate] {len(combos)} combinations -> {out}", flush=True)


def _q(x):
    x = str(x)
    return x if x and all(c.isalnum() or c in "-_./=+,:" for c in x) else "'" + x.replace("'", "'\\''") + "'"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("combos")
    c.add_argument("--sizes", type=int, nargs="+", default=[3])
    c.add_argument("--out", required=True)
    p = sp.add_parser("pack")
    p.add_argument("--combo-file", required=True)
    p.add_argument("--task", type=int, default=None, help="array index: rows task*per_gpu ... (default: all rows)")
    p.add_argument("--per-gpu", type=int, default=2)
    p.add_argument("--mods", default="eeg,eog,emg")
    p.add_argument("--workers-total", type=int, default=22)
    p.add_argument("--splits", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--steps-per-epoch", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--valid-per-item", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    for name, fn_help in (("stager", "zero-shot stagers of every finished combination"), ("eval", "every test night, every model")):
        q = sp.add_parser(name, help=fn_help)
        q.add_argument("--combo-file", required=True)
        q.add_argument("--task", type=int, default=None)
        q.add_argument("--per-gpu", type=int, default=2)
        q.add_argument("--mods", default="eeg,eog,emg")
        q.add_argument("--cache-dir", default=None)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--out", required=True)
    q = sp.choices["stager"]
    q.add_argument("--train-per-item", type=int, default=100, help="train nights per item (latents of the stager)")
    q.add_argument("--valid-nights-per-item", type=int, default=20)
    q.add_argument("--scratch", default=None, help="latents directory (default $TMPDIR)")
    q.add_argument("--only", type=int, default=None, help=argparse.SUPPRESS)
    q = sp.choices["eval"]
    q.add_argument("--splits", required=True)
    q.add_argument("--units", default=None, help="comma list of units (default: all)")
    q.add_argument("--max-nights", type=int, default=0, help="smoke: first nights per unit (0 = all)")
    q.add_argument("--recon-per-night", type=int, default=250)
    q = sp.add_parser("aggregate", help="combination x group matrix, panel, trend")
    q.add_argument("--out", required=True)
    q.add_argument("--n-boot", type=int, default=1000)
    q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"combos": combos, "pack": pack, "stager": stager, "eval": evaluate, "aggregate": aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
