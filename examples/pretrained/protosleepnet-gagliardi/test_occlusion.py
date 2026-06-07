"""Channel occlusion test for ProtoSleepNet models.

Loads ProtoSleepNet via factory method + checkpoint, runs occlusion scenarios.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/test_occlusion.py \
        --backbone seq --checkpoint /path/to/model.pt --dataset mass --seq_len 20 \
        --scenario clean --gpu_id 0 --output_dir /path/to/results
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.collate import stack_channels
from physioex.models.protosleepnet import ProtoSleepNet
from physioex.train.trainer import Trainer

SCENARIOS = {
    "clean": {"mode": None},
    "random_25": {"mode": "random", "p": 0.25},
    "random_50": {"mode": "random", "p": 0.50},
    "no_eeg": {"mode": "fixed", "channels_to_occlude": [0]},
    "no_eog_emg": {"mode": "fixed", "channels_to_occlude": [1, 2]},
}

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

FACTORY = {
    "seq": "from_seq_sleep_net",
    "st": "from_sleep_transformer",
}

MIXER_KWARGS = {
    "use_channel_mixer": True,
    "cdropout": 0.5,
    "cm_n_heads": 4,
    "cm_d_ff": 256,
    "cm_n_layers": 1,
}


class ChannelOcclusionWrapper(nn.Module):
    def __init__(self, model, mode=None, p=0.0, channels_to_occlude=None):
        super().__init__()
        self.model = model
        self.mode = mode
        self.p = p
        self.channels_to_occlude = channels_to_occlude or []

    def forward(self, x):
        if self.mode is None:
            return self.model(x)

        if self.mode == "random":
            B, L, C, T, F_dim = x.shape
            keep = torch.rand(B, L, C, 1, 1, device=x.device) >= self.p
            all_dropped = ~keep.any(dim=2, keepdim=True)
            if all_dropped.any():
                rescue = torch.zeros_like(keep)
                rescue[:, :, torch.randint(C, (1,)).item()] = True
                keep = keep | (all_dropped & rescue)
            x = x * keep.float()

        elif self.mode == "fixed":
            x = x.clone()
            for ch in self.channels_to_occlude:
                x[:, :, ch] = 0.0

        return self.model(x)


@torch.no_grad()
def evaluate_subject(model, inputs, L, device):
    inputs = inputs.to(device)
    night_length = inputs.shape[1]

    if night_length < L:
        y = model(inputs)
        return F.softmax(y.squeeze(0), dim=-1).cpu()

    probe = model(inputs[:, :L])
    n_classes = probe.shape[-1]

    votes = torch.zeros(1, night_length, n_classes, device=device, dtype=probe.dtype)
    counts = torch.zeros(1, night_length, device=device, dtype=torch.float32)

    for offset in range(L):
        x = inputs[:, offset:]
        usable = x.shape[1] - (x.shape[1] % L)
        if usable == 0:
            continue
        x = x[:, :usable]
        num_windows = usable // L
        rest_dims = x.shape[2:]
        x = x.reshape(num_windows, L, *rest_dims)
        y = model(x)
        y = y.reshape(1, num_windows * L, n_classes)
        votes[:, offset : offset + usable] += y
        counts[:, offset : offset + usable] += 1

    safe_counts = counts.clamp(min=1).unsqueeze(-1)
    logits = votes / safe_counts
    return F.softmax(logits.squeeze(0), dim=-1).cpu()


def compute_metrics(all_proba, all_targets, ignore_index=-1):
    preds = torch.cat(all_proba, dim=0)
    targets = torch.cat(all_targets, dim=0)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    pred_labels = preds.argmax(dim=1)
    n_classes = preds.shape[1]

    acc = (pred_labels == targets).float().mean().item()
    per_class_f1 = []
    for c in range(n_classes):
        tp = ((pred_labels == c) & (targets == c)).sum().float()
        fp = ((pred_labels == c) & (targets != c)).sum().float()
        fn = ((pred_labels != c) & (targets == c)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        per_class_f1.append((2 * prec * rec / (prec + rec + 1e-8)).item())
    f1_macro = sum(per_class_f1) / n_classes

    pe = 0.0
    for c in range(n_classes):
        pe += (pred_labels == c).float().mean().item() * (targets == c).float().mean().item()
    kappa = (acc - pe) / (1 - pe + 1e-8)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "cohen_kappa": kappa,
        "f1_per_class": {CLASS_NAMES[c]: per_class_f1[c] for c in range(n_classes)},
    }


def build_dataset(dataset_name, channels, pipeline, seq_len):
    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset
        MASS = get_dataset("mass")
        cohorts = []
        for c in [1, 2, 3, 4, 5]:
            ds = MASS(cohort=c, channels=channels, pipelines=pipeline,
                      sequence_length=seq_len)
            if ds.get_n_subjects() > 0:
                cohorts.append(ds)
        return MultiDataset(cohorts)
    else:
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = {"visit": 1} if dataset_name == "shhs" else {}
        return DatasetClass(channels=channels, pipelines=pipeline,
                            sequence_length=seq_len, **ds_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Occlusion test for ProtoSleepNet")
    parser.add_argument("--backbone", type=str, required=True, choices=["seq", "st"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mass")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(SCENARIOS.keys()),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Build model
    factory = getattr(ProtoSleepNet, FACTORY[args.backbone])
    model = factory(n_channels=args.n_channels, **MIXER_KWARGS)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    channels = ["EEG", "EOG", "EMG"] if args.n_channels == 3 else ["EEG"]
    dataset = build_dataset(args.dataset, channels, "seqsleepnet", args.seq_len)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, fold=0,
    )
    print(f"Dataset: {args.dataset}, test subjects: {len(test_loader)}")

    # Run scenarios
    scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS

    all_metrics = {}
    for scenario_name, scenario_cfg in scenarios_to_run.items():
        print(f"\n{'='*60}\nScenario: {scenario_name}\n{'='*60}")

        wrapped = ChannelOcclusionWrapper(
            model,
            mode=scenario_cfg.get("mode"),
            p=scenario_cfg.get("p", 0.0),
            channels_to_occlude=scenario_cfg.get("channels_to_occlude"),
        )

        subject_predictions = []
        all_proba = []
        all_targets = []

        for subj_idx, batch in enumerate(tqdm(test_loader, desc=scenario_name)):
            if isinstance(batch, dict) and "signals" in batch:
                inputs = stack_channels(batch)
                targets = batch["labels"]
            else:
                inputs, targets = batch

            proba = evaluate_subject(wrapped, inputs, args.seq_len, device)
            targets_flat = targets.reshape(-1)
            subject_predictions.append({
                "subject_idx": subj_idx,
                "proba": proba.tolist(),
                "labels": targets_flat.tolist(),
            })
            all_proba.append(proba)
            all_targets.append(targets_flat)

        metrics = compute_metrics(all_proba, all_targets)
        all_metrics[scenario_name] = metrics

        print(f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}  kappa={metrics['cohen_kappa']:.4f}")
        print(f"  per-class F1: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics["f1_per_class"].items()))

        pred_path = os.path.join(args.output_dir, f"occlusion_predictions_{scenario_name}.json")
        with open(pred_path, "w") as f:
            json.dump(subject_predictions, f)

        metrics_path = os.path.join(args.output_dir, f"occlusion_metrics_{scenario_name}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    if len(all_metrics) > 1:
        print(f"\n{'='*80}\nSummary\n{'='*80}")
        for sn, m in all_metrics.items():
            print(f"{sn:12s}  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  kappa={m['cohen_kappa']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
