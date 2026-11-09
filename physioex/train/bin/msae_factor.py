"""Train the factorized PSG encoder (:mod:`physioex.models.msae_factor`) on the merged train split
of ``--items``, clean EEG only (phase 2 bis, ``PIANO_FASE_2B.md``).

    python -m physioex.train.bin.msae_factor --items ccshs,dcsm,mass --splits S.json --cache-dir C --out RUN
        [--l-rule 0.5]  # weak rule supervision (declared); 0 = label-free

Writes ``ckpt/`` (best validation), ``model.pt``, ``eval.json`` (validation terms of the best
checkpoint), ``run.json`` (``model: msae_factor`` so :func:`physioex.models.recon_eval.model_from_args`
rebuilds it for the dump tools). ``--local-cache`` copies the train/val cache files of the items to
the node's disk first (Sofia GPFS, see ``pretrain_combo pack``).
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from physioex.models.msae_factor import MSAEFactor, MSAEFactorTrainer
from physioex.models.recon_eval import load_best
from physioex.train.bin.msvq import _loader, _merged_data, _merged_loaders


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", required=True, help="comma-separated pool items (physioex.data.merged)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--local-cache", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="EEG_clean")
    ap.add_argument("--seq", type=int, default=41)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--valid-per-item", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--valid-ratio", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-subjects", type=int, default=0, help="smoke tests: keep only N nights per item")
    # architecture
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-atoms", type=int, default=64)
    ap.add_argument("--atom-sec", type=float, default=2.0)
    ap.add_argument("--sub", type=int, default=2, help="atom positions per token")
    ap.add_argument("--top-k", type=int, default=2, help="atoms kept per token (0 = all)")
    ap.add_argument("--d-s", type=int, default=64)
    ap.add_argument("--d-n", type=int, default=64)
    ap.add_argument("--token-sec", type=float, default=0.5)
    ap.add_argument("--no-n", action="store_true")
    # objective
    ap.add_argument("--low-hz", type=float, default=4.0)
    ap.add_argument("--pow-hz", type=float, default=4.5)
    ap.add_argument("--band-weight", type=float, default=3.0)
    ap.add_argument("--low-weight", type=float, default=1.0)
    ap.add_argument("--l-sparse", type=float, default=0.05)
    ap.add_argument("--l-ctr", type=float, default=0.5)
    ap.add_argument("--l-adv", type=float, default=0.2)
    ap.add_argument("--l-bg", type=float, default=1.0)
    ap.add_argument("--l-rule", type=float, default=0.0, help="weak AASM rule supervision of the atoms (0 = off)")
    ap.add_argument("--temperature", type=float, default=0.1)
    a = ap.parse_args(argv)
    a.model, a.dataset, a.load_fs, a.fs, a.root, a.resample = "msae_factor", None, None, None, None, None
    a.pow_bands = a.psd_band = a.recon = a.seg_sec = None  # read by channel_pipelines / modality_kwargs
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.fold)
    t0 = time.time()
    local = None
    if a.local_cache:
        from physioex.data.merged import stage_local

        src = a.cache_dir or os.environ.get("PHYSIOEX_CACHE_DIR")
        local = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                             f"physioex_cache_{os.environ.get('SLURM_JOB_ID', os.getpid())}_factor")
        rep = stage_local(a.items.split(","), a.splits, src, local, [a.channel])
        print(f"[factor] local cache {local}: {rep['_total']['files']} files, {rep['_total']['bytes'] / 2**30:.1f} GiB "
              f"in {(time.time() - t0) / 60:.1f} min", flush=True)
        a.cache_dir = local
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        _run(a, t0)
    finally:
        if local:
            shutil.rmtree(local, ignore_errors=True)


def _run(a, t0):
    md, ds, train_ids, valid_ids, _ = _merged_data(a)
    if a.max_subjects:  # smoke: first nights per item in the train / valid index
        from physioex.data.sampling import _window_owner

        for name in ("train_idx", "valid_idx"):
            idx = np.asarray(getattr(md, name))
            d, s = _window_owner(md.multi, idx)
            keep = s < a.max_subjects
            setattr(md, name, idx[keep])
    tr, n_tr, va, n_va = _merged_loaders(a, md, ds)
    sites = sorted(set(it if it is not None else "_none" for it in md.item_of_dataset))
    site_of_dataset = [sites.index(it if it is not None else "_none") for it in md.item_of_dataset]
    print(f"[factor] windows: train {n_tr}, valid {n_va}; sites {sites}", flush=True)
    model = MSAEFactor(a.seq, width=a.width, dim=a.dim, n_atoms=a.n_atoms, atom_sec=a.atom_sec, sub=a.sub,
                       top_k=a.top_k, d_s=a.d_s, d_n=a.d_n, token_sec=a.token_sec, low_hz=a.low_hz, pow_hz=a.pow_hz,
                       band_weight=a.band_weight, low_weight=a.low_weight, l_sparse=a.l_sparse, l_ctr=a.l_ctr,
                       l_adv=a.l_adv, l_bg=a.l_bg, l_rule=a.l_rule, n_sites=len(sites), temperature=a.temperature,
                       use_n=not a.no_n)
    model.site_of_dataset = site_of_dataset
    a.n_sites = len(sites)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[factor] {n_par / 1e6:.2f} M parameters ({model.n_tok} tokens x {model.d_p} amplitudes)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_val = a.epochs * max(1, round(1 / a.valid_ratio))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_val, eta_min=a.lr / 20)
    model = MSAEFactorTrainer.train(model, (tr, va), checkpoint_path=os.path.join(a.out, "ckpt"), max_epochs=a.epochs,
                                    lr=a.lr, weight_decay=1e-4, valid_interval_ratio=a.valid_ratio,
                                    log_confusion_matrix=False, early_stopping_patience=a.patience,
                                    logger="tensorboard", run_name="msae_factor", seed=a.fold,
                                    gpu_id=0 if torch.cuda.is_available() else None, grad_clip=a.grad_clip or None,
                                    optimizer=opt, scheduler=sched)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    best = load_best(model, a.out)
    print(f"[factor] evaluating {best or 'final weights'}", flush=True)
    device = next(model.parameters()).device
    model.eval()
    acc = {}
    with torch.no_grad():
        for batch in va:
            _, _, extra = MSAEFactorTrainer._step(model, batch, None, device)
            for k, v in extra.items():
                acc.setdefault(k, []).append(v)
    ev = {k: float(np.mean(v)) for k, v in acc.items()}
    ev["n_params"] = n_par
    json.dump(ev, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print("[factor] valid " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()), flush=True)

    def _git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=os.path.dirname(__file__), text=True).strip()
        except Exception:
            return None

    import physioex
    run = {"args": vars(a), "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
           "physioex_version": getattr(physioex, "__version__", None), "host": socket.gethostname(),
           "job_id": os.environ.get("SLURM_JOB_ID"), "sites": sites, "merged": md.report,
           "splits": {"train": train_ids, "valid": valid_ids, "test": []},
           "windows": {"train": n_tr, "valid": n_va}, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(run, open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print("[factor] DONE", flush=True)


if __name__ == "__main__":
    main()
