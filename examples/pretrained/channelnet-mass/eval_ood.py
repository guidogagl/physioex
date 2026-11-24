"""Out-of-domain evaluation of models trained on one source (MASS: #25; SHHS-1: #30, xmai-lab/sleep-mechinterp).

Each model is rebuilt from its run folder (run.json config + best.pt): the multimodal model (``--arch mm``; the
branch backbone, ChannelNet or SleepTransformer, is read from the branch run.json) or a 3-channel baseline on clean
channels (``--arch ssn3ch`` / ``st3ch``); L comes from the model run.json. Every model sees the same nights (clean
cache v1, all three modalities, same loader as training) and is scored with the voting protocol of the in-domain test.

Per dataset: pooled metrics (kappa, acc, macro-F1, per-class F1, confusion) and per-night kappa for every model, plus
paired comparisons against the first model (per-night delta kappa, Wilcoxon signed-rank, bootstrap 95 % CI over nights
of the pooled delta kappa). Large datasets are subsampled to --max-nights nights (fixed seed, same nights for all
models).

    python examples/pretrained/channelnet-mass/eval_ood.py --models S3F=<run> ssn3ch=<run> --root <cache v1> \
        --datasets dcsm hmc ... --out <folder>
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

from physioex.data.clean_cache import load_multimodal_night
from physioex.models.channelnet_mm import MultiChannelNet
from physioex.models.seqsleepnet_3ch_clean import SeqSleepNet3chClean
from physioex.models.sleeptransformer_clean import SleepTransformer3chClean
from physioex.train.bin.train_channelnet import _metrics, git_sha
from physioex.train.bin.train_channelnet_mm import load_branches, voting


def build(run_dir: Path, device):
    cfg = json.loads((run_dir / "run.json").read_text())["config"]
    mods = cfg.get("modalities", ["EEG", "EOG", "EMG"])
    if cfg.get("arch", "mm") == "mm":
        branches, _ = load_branches(Path(cfg["branches"]), mods, 0.0, cfg.get("branch_prefix", ""))
        model = MultiChannelNet(branches, cfg.get("points", []), hook=cfg.get("hook", "identity"))
    elif cfg["arch"] == "st3ch":
        model = SleepTransformer3chClean(mods)
    else:
        model = SeqSleepNet3chClean(mods)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.to(device).eval(), cfg.get("seq_len", 20), mods


def kappa(yt, yp):
    return float(cohen_kappa_score(yt, yp)) if len(np.unique(np.concatenate([yt, yp]))) > 1 else float("nan")


def bootstrap_delta(pairs_a, pairs_b, n_boot=1000, seed=0):
    """95 % CI of pooled kappa(a) - kappa(b), resampling nights."""
    rng = np.random.default_rng(seed)
    n, out = len(pairs_a), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = np.concatenate([pairs_a[i][0] for i in idx])
        out.append(kappa(yt, np.concatenate([pairs_a[i][1] for i in idx]))
                   - kappa(yt, np.concatenate([pairs_b[i][1] for i in idx])))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=run_folder; the first is the reference")
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--max-nights", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pipe", default="702cb90947921189",
                    help="epoching pipeline used where a channel has several (the one of the MASS training data)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--issue", default="xmai-lab/sleep-mechinterp#25")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = (Path(path).expanduser(), *build(Path(path).expanduser(), device))
    names = list(models)
    mods = models[names[0]][3]
    root = Path(args.root)

    for ds in args.datasets:
        f_out = out / f"{ds}.json"
        if f_out.exists():
            print(f"{ds}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        subjects = sorted(p.name for p in (root / ds / "signals").iterdir())
        order = np.random.default_rng(args.seed).permutation(len(subjects))
        preds = {n: [] for n in names}
        used, skipped = [], {}
        for j in order:
            if len(used) >= args.max_nights:
                break
            try:
                night, why = load_multimodal_night(root, ds, subjects[j], mods, pipe=args.pipe)
            except Exception as e:                      # unreadable night: count it, do not stop the dataset
                night, why = None, f"error {type(e).__name__}"
            if night is None or (night.y >= 0).sum() == 0:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            res = {}
            for n in names:
                _, model, L, _ = models[n]
                r = voting(model, [night], device, L, return_preds=True) if len(night.y) >= L else None
                if r is None:
                    break
                res[n] = r["preds"][0]
            if len(res) != len(names):
                skipped["shorter than L"] = skipped.get("shorter than L", 0) + 1
                continue
            for n in names:
                preds[n].append((res[n][0], res[n][1]))
            used.append(subjects[j])
        if not used:
            print(f"{ds}: no usable night ({skipped})", flush=True)
            continue
        yt = np.concatenate([p[0] for p in preds[names[0]]])
        result = {"dataset": ds, "nights": len(used), "subjects_available": len(subjects), "skipped": skipped,
                  "used": used, "models": {}, "paired": {}}
        for n in names:
            m = _metrics(yt, np.concatenate([p[1] for p in preds[n]]))
            m["per_night_kappa"] = [kappa(a, b) for a, b in preds[n]]
            result["models"][n] = m
        ref = names[0]
        for n in names[1:]:
            ka, kb = np.array(result["models"][ref]["per_night_kappa"]), np.array(result["models"][n]["per_night_kappa"])
            ok = ~(np.isnan(ka) | np.isnan(kb))
            d = ka[ok] - kb[ok]
            p = float(wilcoxon(d).pvalue) if ok.sum() > 5 and np.any(d != 0) else float("nan")
            result["paired"][f"{ref}-{n}"] = {
                "pooled_delta_kappa": result["models"][ref]["kappa"] - result["models"][n]["kappa"],
                "pooled_delta_ci95": bootstrap_delta(preds[ref], preds[n]),
                "mean_night_delta": float(d.mean()), "median_night_delta": float(np.median(d)),
                "wins": int((d > 0).sum()), "n": int(ok.sum()), "wilcoxon_p": p}
        result.update({"minutes": round((time.time() - t0) / 60, 2), "git_sha": git_sha(), "issue": args.issue,
                       "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                       "model_runs": {n: str(models[n][0]) for n in names}})
        f_out.write_text(json.dumps(result, indent=1))
        line = " | ".join(f"{n} κ {result['models'][n]['kappa']:.4f}" for n in names)
        print(f"{ds} ({len(used)} nights, {result['minutes']} min): {line}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
