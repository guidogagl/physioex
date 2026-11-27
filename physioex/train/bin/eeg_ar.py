"""Train and score the next-token model on patch codes (:mod:`physioex.models.eeg_ar`, xmai-lab/sleep-lens#20).

    python -m physioex.train.bin.eeg_ar fit --dump DUMP --out RUN [--levels L] [--max-hours 4.5] [--batch 16]
    python -m physioex.train.bin.eeg_ar score --run RUN [--dump DUMP] [--groups eval] [--out RUN/surprise]

``fit``: the train nights of a code dump (``patch_codec dump``, or its spectral twin): one night in 40 is
held out for the validation, the rest is sampled uniformly (night, one of the 4 patch grids, start) in
windows of ``--window`` patches (150 = 5 min), ``--fim`` of them in the FIM layout with a middle of 1 to
``--max-span`` patches. AdamW, warm-up then cosine; with ``--steps 0`` the number of steps is set from
``--max-hours`` after timing the first steps. The validation NLL (plain layout) selects ``model.pt``.
Writes ``run.json``, ``log.jsonl``, ``model.pt`` and ``eval.json`` (validation NLL per level, FIM middle NLL,
held-out NLL and perplexity per unit on the dump's ``eval`` nights).

``score``: surprise of every code of the chosen nights (:func:`physioex.models.eeg_ar.score_codes`, each
grid separately), the patch surprise ``S = sum_l (nll - H)`` and its mean over the 4 grids on the 0.5 s
token grid. One ``<out>/<unit>/<sid>.npz`` per night: ``s`` (tokens), ``S<o>`` (patches of grid ``o``),
``nll<o>`` and ``ent<o>`` (``(P, L)`` float16), ``e0``, ``n``, ``y``; ``<out>/meta.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from physioex.models.eeg_ar import BOS, EEGAR, MID, build_from_args, collate, make_example, score_codes, token_scores

GRIDS = ("c0", "c50", "c100", "c150")


def _git(*cmd):
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), *cmd],
                                       text=True).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dump", required=True, help="code dump folder (meta.json + <unit>/<sid>.npz)")
    f.add_argument("--out", required=True)
    f.add_argument("--levels", type=int, default=0, help="first levels of the codes; 0: all of the dump")
    f.add_argument("--window", type=int, default=150, help="patches per training window (150 = 5 min)")
    f.add_argument("--fim", type=float, default=0.2)
    f.add_argument("--max-span", type=int, default=10, help="longest FIM middle, patches")
    f.add_argument("--d-model", type=int, default=512)
    f.add_argument("--layers", type=int, default=12)
    f.add_argument("--heads", type=int, default=8)
    f.add_argument("--dropout", type=float, default=0.0)
    f.add_argument("--batch", type=int, default=16)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--weight-decay", type=float, default=0.1)
    f.add_argument("--warmup", type=int, default=1000)
    f.add_argument("--steps", type=int, default=0, help="0: from --max-hours")
    f.add_argument("--max-hours", type=float, default=4.5)
    f.add_argument("--eval-every", type=int, default=1000)
    f.add_argument("--val-windows", type=int, default=256)
    f.add_argument("--heldout-per-night", type=int, default=8)
    f.add_argument("--max-nights", type=int, default=0, help="smoke tests; 0: all")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--issue", type=int, default=20)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--dump", default=None, help="default: the run's")
    s.add_argument("--groups", default="eval", help="comma-separated dump groups (train, eval)")
    s.add_argument("--hop", type=int, default=75)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--max-nights", type=int, default=0)
    s.add_argument("--out", default=None, help="default: <run>/surprise")
    a = ap.parse_args(argv)
    return _fit(a) if a.cmd == "fit" else _score(a)


# ---------------------------------------------------------------------------- data
def load_nights(dump: str, keys, levels: int):
    """``{key: [4 arrays (P_o, levels) int16]}`` for the night keys ``unit/sid`` of a dump."""
    out = {}
    for k in keys:
        unit, sid = k.split("/", 1)
        z = np.load(os.path.join(dump, unit, f"{sid}.npz"))
        out[k] = [np.ascontiguousarray(z[g][:, :levels]) for g in GRIDS]
    return out


def split_train_val(meta: dict, every: int = 40):
    tr = sorted(k for k, v in meta["nights"].items() if v["group"] == "train")
    return [k for i, k in enumerate(tr) if i % every], [k for i, k in enumerate(tr) if not i % every]


def sample_examples(rng, nights: dict, keys, n: int, window: int, n_codes: int, fim: float, max_span: int):
    out = []
    for _ in range(n):
        arr = nights[keys[rng.integers(len(keys))]][rng.integers(len(GRIDS))]
        s = rng.integers(0, len(arr) - window + 1)
        span = None
        if rng.random() < fim:
            ln = int(rng.integers(1, max_span + 1))
            a = int(rng.integers(1, window - ln))
            span = (a, a + ln)
        out.append(make_example(arr[s: s + window], n_codes, span))
    return out


@torch.no_grad()
def evaluate(model, examples, batch: int, device):
    """NLL per level (plain targets) and mean NLL of FIM middle codes over a fixed list of examples."""
    model.eval()
    L = model.levels
    s_l, n_l, s_mid, n_mid = np.zeros(L), np.zeros(L), 0.0, 0
    for i in range(0, len(examples), batch):
        ex = examples[i: i + batch]
        b = {k: v.to(device) for k, v in collate(ex).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            nll, _ = model.nll_entropy(b)
        nll = nll.float().cpu().numpy()
        for k, e in enumerate(ex):
            ok = e["tgt"] >= 0
            if e["inp"][0] != BOS:  # FIM layout: the targets from the MID sentinel on are the middle
                mid = ok & (np.arange(len(ok)) >= np.flatnonzero(e["inp"] == MID)[0])
                s_mid += nll[k][mid].sum()
                n_mid += int(mid.sum())
            else:
                np.add.at(s_l, e["lev"][ok], nll[k][ok])
                np.add.at(n_l, e["lev"][ok], 1)
    model.train()
    per = s_l / np.maximum(n_l, 1)
    return {"nll": float(s_l.sum() / max(n_l.sum(), 1)), "nll_per_level": per.round(4).tolist(),
            "nll_fim_middle": float(s_mid / max(n_mid, 1)) if n_mid else None}


# ---------------------------------------------------------------------------- fit
def _fit(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    n_codes = int(meta["n_codes"])
    levels = a.levels or int(meta["levels"])
    tr, va = split_train_val(meta)
    if a.max_nights:
        tr, va = tr[: a.max_nights], va[: max(1, a.max_nights // 10)]
    nights = load_nights(a.dump, tr + va, levels)
    tr = [k for k in tr if min(len(x) for x in nights[k]) >= a.window]
    va = [k for k in va if min(len(x) for x in nights[k]) >= a.window]
    print(f"[eeg_ar] {len(tr)} train / {len(va)} validation nights, {levels} levels x {n_codes} codes "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    vrng = np.random.default_rng(12345)
    val = (sample_examples(vrng, nights, va, a.val_windows // 2, a.window, n_codes, 0.0, a.max_span)
           + sample_examples(vrng, nights, va, a.val_windows // 2, a.window, n_codes, 1.0, a.max_span))
    model = EEGAR(n_codes=n_codes, levels=levels, d_model=a.d_model, layers=a.layers, heads=a.heads,
                  dropout=a.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    args = {k: v for k, v in vars(a).items()}
    args.update({"n_codes": n_codes, "levels": levels, "model": "eeg_ar"})
    run = {"issue": f"xmai-lab/sleep-lens#{a.issue}", "args": args, "git_sha": _git("rev-parse", "HEAD"),
           "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "n_params": n_par,
           "dump": os.path.abspath(a.dump), "dump_run": meta.get("run"), "dump_git_sha": meta.get("git_sha"),
           "n_train_nights": len(tr), "n_val_nights": len(va), "val_nights": va}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(f"[eeg_ar] {n_par / 1e6:.1f} M parameters", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    total, step, best, t_start = a.steps or 10 ** 9, 0, float("inf"), time.time()
    budget = a.max_hours * 3600 - (time.time() - t0)

    def lr_at(i):
        if i < a.warmup:
            return a.lr * (i + 1) / a.warmup
        p = min(1.0, (i - a.warmup) / max(1, total - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    model.train()
    run_loss, run_n = 0.0, 0
    while step < total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ex = sample_examples(rng, nights, tr, a.batch, a.window, n_codes, a.fim, a.max_span)
        b = {k: v.to(device, non_blocking=True) for k, v in collate(ex).items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, _ = model(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss, run_n, step = run_loss + float(loss), run_n + 1, step + 1
        if step == 50 and not a.steps:  # time the steps, then fix the schedule to the budget
            per = (time.time() - t_start) / 50
            evals = 1.15  # validation and save overhead
            total = max(a.warmup + 100, int(budget * 0.93 / (per * evals)))
            print(f"[eeg_ar] {per:.3f} s/step -> {total} steps in {a.max_hours} h", flush=True)
            run["args"]["steps_auto"] = total
            json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)
        if step % a.eval_every == 0 or step == total:
            ev = evaluate(model, val, a.batch, device)
            rec = {"step": step, "train_loss": run_loss / max(run_n, 1), "lr": lr_at(step - 1),
                   "minutes": (time.time() - t0) / 60, **ev}
            run_loss, run_n = 0.0, 0
            if ev["nll"] < best:
                best = ev["nll"]
                torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
                rec["best"] = True
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[eeg_ar] step {step}/{total} train {rec['train_loss']:.4f} val {ev['nll']:.4f} "
                  f"fim-mid {ev['nll_fim_middle']:.4f} ({rec['minutes']:.1f} min){' *' if rec.get('best') else ''}",
                  flush=True)
        if time.time() - t0 > a.max_hours * 3600:
            print("[eeg_ar] time budget reached", flush=True)
            break
    if not os.path.exists(os.path.join(a.out, "model.pt")):
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    # held-out nights of the dump (pool test, MASS SS01, DREAMS): NLL per unit with the selected model
    model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=device))
    ho = sorted(k for k, v in meta["nights"].items() if v["group"] == "eval")
    if a.max_nights:
        ho = ho[: a.max_nights]
    hrng, res = np.random.default_rng(777), {}
    by_unit = {}
    for k in ho:
        by_unit.setdefault(k.split("/", 1)[0], []).append(k)
    for unit, keys in sorted(by_unit.items()):
        hn = load_nights(a.dump, keys, levels)
        keys = [k for k in keys if min(len(x) for x in hn[k]) >= a.window]
        if not keys:
            continue
        ex = sample_examples(hrng, hn, keys, a.heldout_per_night * len(keys), a.window, n_codes, 0.0, a.max_span)
        ev = evaluate(model, ex, a.batch, device)
        res[unit] = {"nights": len(keys), "nll": ev["nll"], "ppl": float(np.exp(ev["nll"])),
                     "nll_per_level": ev["nll_per_level"]}
    final = {"best_val_nll": best, "val": evaluate(model, val, a.batch, device), "heldout": res,
             "steps": step, "minutes": (time.time() - t0) / 60, "log_k": math.log(n_codes)}
    json.dump(final, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print(f"[eeg_ar] best val NLL {best:.4f} (log K = {math.log(n_codes):.3f}); held-out "
          + ", ".join(f"{u} {r['nll']:.3f}" for u, r in res.items()), flush=True)
    print("[eeg_ar] DONE", flush=True)


# ---------------------------------------------------------------------------- score
def load_model(run_dir: str, device):
    run = json.load(open(os.path.join(run_dir, "run.json")))
    model = build_from_args(run["args"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    return run, model.to(device).eval()


def _score(a):
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run, model = load_model(a.run, device)
    dump = a.dump or run["dump"]
    meta = json.load(open(os.path.join(dump, "meta.json")))
    groups = set(a.groups.split(","))
    keys = sorted(k for k, v in meta["nights"].items() if v["group"] in groups)
    if a.max_nights:
        keys = keys[: a.max_nights]
    out = a.out or os.path.join(a.run, "surprise")
    os.makedirs(out, exist_ok=True)
    mp = os.path.join(out, "meta.json")
    om = json.load(open(mp)) if os.path.exists(mp) else {"nights": {}}
    om.update({"run": os.path.abspath(a.run), "dump": os.path.abspath(dump), "git_sha": _git("rev-parse", "HEAD"),
               "levels": model.levels, "window": run["args"]["window"], "hop": a.hop, "offsets": meta["offsets"]})
    W, L = run["args"]["window"], model.levels
    for i, k in enumerate(keys):
        path = os.path.join(out, f"{k}.npz")
        if k in om["nights"] and os.path.exists(path):
            continue
        unit, sid = k.split("/", 1)
        z = np.load(os.path.join(dump, unit, f"{sid}.npz"))
        arr, per_grid = {}, {}
        for g, o in zip(GRIDS, meta["offsets"]):
            nll, ent = score_codes(model, np.ascontiguousarray(z[g][:, :L]), W, a.hop, a.batch, device)
            per_grid[o] = (nll - ent).sum(1)
            arr[f"nll{o}"], arr[f"ent{o}"], arr[f"S{o}"] = nll.astype(np.float16), ent.astype(np.float16), per_grid[o]
        s = token_scores(per_grid, int(z["n"]), meta["patch"], 50).astype(np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, s=s, e0=z["e0"], n=z["n"], y=z["y"], **arr)
        om["nights"][k] = {**meta["nights"][k], "n_tokens": len(s)}
        if (i + 1) % 25 == 0 or i + 1 == len(keys):
            json.dump(om, open(mp, "w"))
            print(f"[eeg_ar score] {i + 1}/{len(keys)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump(om, open(mp, "w"))
    print("[eeg_ar score] DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
