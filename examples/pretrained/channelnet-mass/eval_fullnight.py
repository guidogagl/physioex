"""Full-night inference vs the training protocol (xmai-lab/sleep-mechinterp#31). No training.

MASS fold-0 test nights, multimodal model trained with L. Three inference conditions:
  vote : voting(L), the training test protocol;
  tile : non-overlapping tiles of L, no voting (last partial tile = last L epochs, only uncovered epochs kept);
  full : the whole night in one forward pass (batch of 1 night, exact: no padding).
Metrics per condition (pooled, per-night kappa); paired full-vote and tile-vote (per-night delta, Wilcoxon,
bootstrap CI of the pooled delta kappa). Labels < 0 excluded.

    python eval_fullnight.py --model <run folder> --root <cache v1> --split-file mass_fold0_split.json --out <folder>
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from eval_ood import bootstrap_delta, build, kappa                      # noqa: E402
from physioex.data.clean_cache import CleanCacheMultiNights, load_multimodal_night   # noqa: E402
from physioex.train.bin.train_channelnet import _metrics, git_sha        # noqa: E402
from physioex.train.bin.train_channelnet_mm import voting                # noqa: E402


def _fwd(model, xs, nt, device, bf16=False):
    x = {m: v.unsqueeze(0).to(device) for m, v in xs.items()}
    off = {m: torch.full((1,), nt.offset_db[m], device=device) for m in x}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        logits, _, _ = model(x, off, None)
    return logits.reshape(-1, 5).float()


@torch.no_grad()
def tiles(model, nt, device, L, bs=64):
    xs = {m: torch.from_numpy(np.asarray(v, dtype=np.float32)) for m, v in nt.x.items()}
    n = len(nt.y)
    starts = list(range(0, n - L + 1, L))
    if starts[-1] != n - L:
        starts.append(n - L)
    pred = np.full(n, -1)
    for i in range(0, len(starts), bs):
        ss = starts[i:i + bs]
        xb = {m: torch.stack([v[s:s + L] for s in ss]).to(device) for m, v in xs.items()}
        off = {m: torch.full((len(ss),), nt.offset_db[m], device=device) for m in xb}
        p = model(xb, off, None)[0].argmax(-1).cpu().numpy()
        for s, row in zip(ss, p):
            sel = pred[s:s + L] < 0              # keep only epochs not yet covered
            pred[s:s + L][sel] = row[sel]
    assert (pred >= 0).all()
    return pred


@torch.no_grad()
def whole(model, nt, device):
    xs = {m: torch.from_numpy(np.asarray(v, dtype=np.float32)) for m, v in nt.x.items()}
    try:
        lg, bf = _fwd(model, xs, nt, device), False
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        lg, bf = _fwd(model, xs, nt, device, bf16=True), True
    return lg.argmax(-1).cpu().numpy(), bf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--datasets", nargs="+", default=[f"mass_ss0{i}" for i in range(1, 6)])
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--pipe", default="702cb90947921189")
    ap.add_argument("--max-nights", type=int, default=0, help="debug only")
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--issue", default="xmai-lab/sleep-mechinterp#31")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.model).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model, L, mods = build(run, device)
    model.disabled = set()
    t0 = time.time()
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text())
        nights = CleanCacheMultiNights(args.root, args.datasets, mods, split=split, pipe=args.pipe).nights["test"]
    else:                                                    # debug: first nights of the datasets
        nights = []
        for ds in args.datasets:
            for s in sorted(p.name for p in (Path(args.root) / ds / "signals").iterdir()):
                nt, _ = load_multimodal_night(Path(args.root), ds, s, mods, pipe=args.pipe)
                if nt is not None:
                    nights.append(nt)
    if args.max_nights:
        nights = nights[:args.max_nights]
    nights = [nt for nt in nights if len(nt.y) >= L and (nt.y >= 0).any()]
    print(f"{len(nights)} test nights, L={L}", flush=True)

    P = {"vote": [], "tile": [], "full": []}
    lengths, bf16_nights = [], []
    for nt in nights:
        msk = nt.y >= 0
        yt = nt.y[msk]
        P["vote"].append((yt, voting(model, [nt], device, L, return_preds=True)["preds"][0][1]))
        P["tile"].append((yt, tiles(model, nt, device, L)[msk]))
        pf, bf = whole(model, nt, device)
        P["full"].append((yt, pf[msk]))
        lengths.append(len(nt.y))
        if bf:
            bf16_nights.append(f"{nt.dataset}/{nt.subject}")
        print(f"{nt.dataset}/{nt.subject} n={len(nt.y)} " + " ".join(f"{c} {kappa(*P[c][-1]):.3f}" for c in P), flush=True)

    yt = np.concatenate([p[0] for p in P["vote"]])
    res = {"nights": len(nights), "L": L, "lengths": lengths, "length_min": min(lengths), "length_max": max(lengths),
           "bf16_fallback_nights": bf16_nights, "subjects": [f"{nt.dataset}/{nt.subject}" for nt in nights],
           "models": {}, "paired": {}}
    for c in P:
        m = _metrics(yt, np.concatenate([p[1] for p in P[c]]))
        m["per_night_kappa"] = [kappa(a, b) for a, b in P[c]]
        res["models"][c] = m
    for c in ("full", "tile"):
        ka, kb = np.array(res["models"][c]["per_night_kappa"]), np.array(res["models"]["vote"]["per_night_kappa"])
        ok = ~(np.isnan(ka) | np.isnan(kb))
        d = ka[ok] - kb[ok]
        res["paired"][f"{c}-vote"] = {
            "pooled_delta_kappa": res["models"][c]["kappa"] - res["models"]["vote"]["kappa"],
            "pooled_delta_ci95": bootstrap_delta(P[c], P["vote"]),
            "mean_night_delta": float(d.mean()), "median_night_delta": float(np.median(d)),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": int(ok.sum()),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if ok.sum() > 5 and np.any(d != 0) else float("nan")}
    rec = json.loads((run / "run.json").read_text())["final"]["test"]["kappa"]
    res.update({"recorded_test_kappa": rec, "minutes": round((time.time() - t0) / 60, 2), "git_sha": git_sha(),
                "issue": args.issue, "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                "model_run": str(run)})
    (out / "results.json").write_text(json.dumps(res, indent=1))
    for c in P:
        r = res["models"][c]
        print(f"{c}: kappa {r['kappa']:.4f} acc {r['acc']:.4f} mF1 {r['macro_f1']:.4f} F1 "
              + " ".join(f"{v:.3f}" for v in r["f1_per_class"]), flush=True)
    print(json.dumps(res["paired"]), flush=True)
    if not args.no_sanity and abs(res["models"]["vote"]["kappa"] - rec) > 0.002:
        print(f"SANITY FAIL: vote kappa {res['models']['vote']['kappa']:.4f} vs recorded {rec:.4f}", flush=True)
        sys.exit(2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
