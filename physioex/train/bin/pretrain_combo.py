"""Pretraining-data study (04.11.2026): which combination of datasets generalizes best.

Every combination of 3-5 pool items (:data:`physioex.data.merged.POOL`, order and repetition
do not count) trains the three clean-channel encoders with the recipes of the DCSM models
(stage 1 MSAE, stage 2 MSAE-VQ initialized from stage 1) on the merged ProtoSleepNet train
split, with the same number of steps for every combination.

    python -m physioex.train.bin.pretrain_combo combos --sizes 3 --out combos_k3.txt
    python -m physioex.train.bin.pretrain_combo pack --combo-file combos_k3.txt --task 0 --per-gpu 2 \\
        --splits splits_29sites.json --cache-dir ... --out ROOT

``pack`` runs ``--per-gpu`` combinations at once on the job's single GPU: one process per
combination and modality (stage 1 then stage 2), ``--workers-total`` data-loader workers shared
among them. Each finished stage leaves ``run.json`` in its directory and is skipped on a rerun.

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
    rows = [l.strip().split(",") for l in open(a.combo_file) if l.strip()]
    mine = rows[a.task * a.per_gpu:(a.task + 1) * a.per_gpu] if a.task is not None else rows
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
    a = ap.parse_args()
    {"combos": combos, "pack": pack}[a.cmd](a)


if __name__ == "__main__":
    main()
