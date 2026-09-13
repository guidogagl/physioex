"""LRP vs post-hoc explanations of the pretrained single-channel SeqSleepNet.

Explains the N3 logit of the pretrained ``seqsleepnet-phan`` (MASS, EEG 1 ch) on
training-set sequences whose central epoch is N3, side by side with Saliency,
Input×Gradient and Integrated Gradients, and shows how the LRP **rule
assignment** changes the attribution (gate rule, filterbank input rule,
attention handling, stabiliser ε), always reporting the conservation ratio
``ΣR / f``.  CPU is enough (a few minutes; the physioex cache makes re-runs
instant).

Requirements: ``pip install "physioex[explain]"``, the MASS EDFs under
``$PHYSIOEX_DATA/MASS/Original/SS0{1..5}`` and a writable ``PHYSIOEX_CACHE_DIR``.
Set ``SEQSLEEPNET_DIR`` to a folder holding ``config.json`` + ``model.pt`` to run
offline (otherwise the weights are pulled from the HF hub).

Usage::

    python examples/explain/lrp_seqsleepnet_n3.py --out ./n3_out [--n 8] [--eps 1e-2]
"""

import argparse
import importlib
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from physioex.explain.lrp import LRPGRU, LRPLSTM, ModelLRP
from physioex.explain.lrp import model as lrpmodel
from physioex.explain.lrp.pooling import _PoolAdapter
from physioex.explain.posthoc import InputXGradient, IntegratedGradients, Saliency
from physioex.explain.posthoc.functionizer import SeqFunct

CLASSES = ["W", "N1", "N2", "N3", "REM"]
FBQ = "physioex.models.seqsleepnet.LearnableFilterbank"
ATQ = "physioex.models.seqsleepnet.AttentionLayer"


def load_model(model_dir):
    """Pretrained SeqSleepNet from a local folder (config.json + model.pt) or the hub."""
    if model_dir:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        mod, cls = cfg["model_class"].split(":")
        model = getattr(importlib.import_module(mod), cls)(**cfg["model_kwargs"])
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.eval(), cfg
    from huggingface_hub import hf_hub_download
    from physioex.models import load_from_pretrained

    cfg = json.load(open(hf_hub_download("4rooms/physioex", "seqsleepnet-phan/config.json")))
    return load_from_pretrained("seqsleepnet-phan", device="cpu").eval(), cfg


def n3_training_sequences(cfg, n, target_idx, log):
    """``n`` training-split sequences (fold 0, all cohorts) whose target epoch is N3."""
    from physioex.data.datasets import MultiDataset, get_dataset

    MASS = get_dataset("mass")
    L = cfg["training"]["sequence_length"]
    cohorts = []
    for c in cfg["training"]["dataset_cohorts"]:
        cohorts.append(MASS(cohort=c, channels=cfg["training"]["channels"],
                            pipelines=cfg["training"]["pipeline_preset"], sequence_length=L))
        log(f"cohort {c} indexed ({len(cohorts[-1])} sequences)")
    ds = MultiDataset(cohorts)
    train_idx = np.array(ds.split(fold=cfg["training"]["fold"])[0])
    order = np.random.default_rng(0).permutation(len(train_idx))
    xs, ys = [], []
    for k in order:
        item = ds[int(train_idx[k])]
        if int(item["labels"][target_idx]) != CLASSES.index("N3"):
            continue
        xs.append(item["signals"][item["channel_order"][0]].unsqueeze(1).float())  # (L, 1, T, F)
        ys.append(item["labels"].clone())
        if len(xs) == n:
            break
    return torch.stack(xs), torch.stack(ys)


# --- alternative input-layer rules for the filterbank (demo of a *wrong* choice) ---


class _FBLinearRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Wfb, mode):
        ctx.save_for_backward(Wfb)
        ctx.mode = mode
        return torch.matmul(x, Wfb)

    @staticmethod
    def backward(ctx, R):
        (Wfb,) = ctx.saved_tensors
        if ctx.mode == "w2":  # squared weights, independent of the input value
            W2 = Wfb ** 2
            W2 = W2 / (W2.sum(dim=-2, keepdim=True) + 1e-12)
            return torch.matmul(R, W2.transpose(-1, -2)), None, None
        F = Wfb.shape[-2]  # flat: uniform over the frequency bins
        return R.sum(-1, keepdim=True).expand(*R.shape[:-1], F) / F, None, None


class FilterbankAltRule(_PoolAdapter):
    def __init__(self, orig, mode):
        super().__init__(orig)
        self.mode = mode

    def forward(self, x):
        with torch.no_grad():
            Wfb = torch.mul(torch.sigmoid(self.orig.W), self.orig.S)
        return _FBLinearRule.apply(x, Wfb, self.mode)


def run_lrp(model, x, target_idx, epsilon, gate_rule="signal_take", fb_mode=None, attn_naive=False):
    saved = dict(lrpmodel._BUILTIN_BY_QUALNAME)
    try:
        if fb_mode:
            lrpmodel._BUILTIN_BY_QUALNAME[FBQ] = lambda m, e, _mode=fb_mode: FilterbankAltRule(m, _mode)
        if attn_naive:
            lrpmodel._BUILTIN_BY_QUALNAME.pop(ATQ, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = ModelLRP(model, in_index=target_idx, out_index=CLASSES.index("N3"), epsilon=epsilon)
        for m in explainer.model.modules():
            if isinstance(m, (LRPLSTM, LRPGRU)):
                m.gate_rule = gate_rule
        rel, report = explainer(x, return_report=True)
        return rel.detach(), report.ratio.tolist(), explainer.uncovered
    finally:
        lrpmodel._BUILTIN_BY_QUALNAME.clear()
        lrpmodel._BUILTIN_BY_QUALNAME.update(saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="n3_out")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--model-dir", default=os.environ.get("SEQSLEEPNET_DIR"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)  # noqa: E731

    model, cfg = load_model(args.model_dir)
    L = cfg["training"]["sequence_length"]
    mid, n3 = L // 2, CLASSES.index("N3")
    x, y = n3_training_sequences(cfg, args.n, mid, log)
    with torch.no_grad():
        logits = model(x)
    log("x", tuple(x.shape), "N3 logits", [round(v, 2) for v in logits[:, mid, n3].tolist()])

    f = SeqFunct(model, in_index=mid, out_index=n3)
    results = {
        "Saliency": Saliency(f)(x).detach(),
        "Input x Gradient": InputXGradient(f)(x).detach(),
        "Integrated Gradients (32 steps)": IntegratedGradients(f, steps=32)(x).detach(),
    }
    conserv = {}
    variants = {
        "LRP recommended": {},
        "LRP gate uniform": {"gate_rule": "uniform"},
        "LRP filterbank w2": {"fb_mode": "w2"},
        "LRP filterbank flat": {"fb_mode": "flat"},
        "LRP attention not ruled": {"attn_naive": True},
    }
    for eps in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        variants[f"LRP eps={eps:g}"] = {"epsilon": eps}
    for name, kw in variants.items():
        rel, ratio, uncovered = run_lrp(model, x, mid, kw.pop("epsilon", args.eps), **kw)
        results[name], conserv[name] = rel, ratio
        log(f"{name}: sumR/f={[round(v, 3) for v in ratio]} uncovered={uncovered}")

    freqs = np.arange(x.shape[-1]) * cfg["model_kwargs"]["fs"] / cfg["model_kwargs"]["nfft"]
    torch.save({"x": x, "labels": y, "logits": logits, "results": results, "conservation": conserv,
                "freqs": freqs, "target_index": mid}, os.path.join(args.out, "attributions.pt"))
    json.dump({k: v for k, v in conserv.items()}, open(os.path.join(args.out, "conservation.json"), "w"), indent=1)
    log("saved", args.out)


if __name__ == "__main__":
    main()
