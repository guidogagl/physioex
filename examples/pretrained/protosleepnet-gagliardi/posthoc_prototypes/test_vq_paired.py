"""Paired statistical test: baseline vs VQ-quantized model.

For each test subject, computes accuracy with both the original model
and the VQ-quantized model. Then runs a Wilcoxon signed-rank test on
the paired per-subject accuracies.

Usage:
    python test_vq_paired.py \
        --model_dir /path/to/pretrained/st-baseline \
        --codebook_path /path/to/codebook_vq_m48.npy \
        --gpu_id 0
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.collate import dict_collate_fn, stack_channels
from physioex.train.trainer import Trainer

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def load_model(model_dir, device):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    module_path, class_name = config["model_class"].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)
    model = ModelClass(**config["model_kwargs"])
    weights_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, config


class VQWrappedModel(nn.Module):
    def __init__(self, model, codebook):
        super().__init__()
        self.model = model
        self.register_buffer("codebook", torch.from_numpy(codebook).float())

    def _epoch_encode(self, x):
        N, C, T, F = x.shape
        if hasattr(self.model, "epoch_encoder") and hasattr(self.model, "in_chan"):
            x_flat = x.reshape(N * C, 1, T, F)
            embs = self.model.epoch_encoder(x_flat)
            embs = embs.reshape(N, C, -1)
            return embs.mean(dim=1)
        if hasattr(self.model, "epoch_encoder"):
            return self.model.epoch_encoder(x)
        if hasattr(self.model, "filterbank") and hasattr(self.model, "seqn1"):
            z = self.model.filterbank(x)
            z = z.permute(0, 2, 1, 3)
            z = z.reshape(z.shape[0], z.shape[1], -1)
            z, _ = self.model.seqn1(z)
            z = self.model.attention(z)
            return z
        raise ValueError(f"Unknown model type: {type(self.model).__name__}")

    def _quantize(self, z):
        z_sq = (z ** 2).sum(dim=1, keepdim=True)
        c_sq = (self.codebook ** 2).sum(dim=1, keepdim=True).T
        dist = z_sq + c_sq - 2 * (z @ self.codebook.T)
        return self.codebook[dist.argmin(dim=1)]

    def _sequence_encode_and_classify(self, z, B, L):
        z = z.reshape(B, L, -1)
        if hasattr(self.model, "sequence_encoder") and hasattr(self.model, "classifier"):
            z = self.model.sequence_encoder(z)
            z = z.reshape(B * L, -1)
            return self.model.classifier(z).reshape(B, L, -1)
        if hasattr(self.model, "seqn2") and hasattr(self.model, "classifier"):
            z, _ = self.model.seqn2(z)
            z = z.reshape(B * L, -1)
            return self.model.classifier(z).reshape(B, L, -1)
        if hasattr(self.model, "seqn2") and hasattr(self.model, "clf"):
            z, _ = self.model.seqn2(z)
            z = z.reshape(B * L, -1)
            return self.model.clf(z).reshape(B, L, -1)
        raise ValueError(f"Unknown model type: {type(self.model).__name__}")

    def forward(self, x):
        B, L, C, T, F_dim = x.shape
        x_flat = x.reshape(B * L, C, T, F_dim)
        z = self._epoch_encode(x_flat)
        z_q = self._quantize(z)
        return self._sequence_encode_and_classify(z_q, B, L)


@torch.no_grad()
def evaluate_subject(model, inputs, L, device):
    """Sliding-window voting for one subject. Returns per-epoch predictions."""
    inputs = inputs.to(device)
    night_length = inputs.shape[1]

    if night_length < L:
        y = model(inputs)
        return y.squeeze(0).argmax(dim=-1).cpu()

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
    return logits.squeeze(0).argmax(dim=-1).cpu()


def subject_accuracy(preds, targets):
    """Per-subject accuracy ignoring unscored epochs."""
    mask = targets >= 0
    if mask.sum() == 0:
        return float("nan")
    return float((preds[mask] == targets[mask]).float().mean())


def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical test: baseline vs VQ"
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--codebook_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    model_name = os.path.basename(args.model_dir)

    # Load model
    model, config = load_model(args.model_dir, device)

    # Load codebook and wrap
    codebook = np.load(args.codebook_path)
    vq_model = VQWrappedModel(model, codebook).to(device).eval()
    print(f"Model: {model_name}, Codebook: {codebook.shape}")

    # Dataset
    ds_kwargs = {"visit": 1} if args.dataset == "shhs" else {}
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=CHANNELS, pipelines=PIPELINE,
        sequence_length=SEQ_LEN, **ds_kwargs,
    )
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, pin_memory=False, fold=args.fold,
    )
    print(f"Test subjects: {len(test_loader)}")

    # Evaluate both models per subject
    baseline_accs = []
    vq_accs = []

    for batch in tqdm(test_loader, desc="Paired eval"):
        if isinstance(batch, dict) and "signals" in batch:
            inputs = stack_channels(batch)
            targets = batch["labels"]
        else:
            inputs, targets = batch

        targets_flat = targets.reshape(-1)

        # Baseline
        preds_base = evaluate_subject(model, inputs, SEQ_LEN, device)
        acc_base = subject_accuracy(preds_base, targets_flat)

        # VQ
        preds_vq = evaluate_subject(vq_model, inputs, SEQ_LEN, device)
        acc_vq = subject_accuracy(preds_vq, targets_flat)

        if not (np.isnan(acc_base) or np.isnan(acc_vq)):
            baseline_accs.append(acc_base)
            vq_accs.append(acc_vq)

    baseline_accs = np.array(baseline_accs)
    vq_accs = np.array(vq_accs)
    diffs = baseline_accs - vq_accs

    # Statistics
    print(f"\n{'='*60}")
    print(f"Paired test: {model_name} (N={len(baseline_accs)} subjects)")
    print(f"{'='*60}")
    print(f"Baseline:  mean={baseline_accs.mean():.4f} std={baseline_accs.std():.4f}")
    print(f"VQ M={codebook.shape[0]}:   mean={vq_accs.mean():.4f} std={vq_accs.std():.4f}")
    print(f"Diff:      mean={diffs.mean():.4f} std={diffs.std():.4f}")
    print(f"           median={np.median(diffs):.4f}")
    print(f"           min={diffs.min():.4f} max={diffs.max():.4f}")

    # Wilcoxon signed-rank test (H0: no difference)
    stat, p_value = stats.wilcoxon(baseline_accs, vq_accs, alternative="greater")
    print(f"\nWilcoxon signed-rank test (baseline > VQ):")
    print(f"  statistic = {stat:.1f}")
    print(f"  p-value   = {p_value:.2e}")
    if p_value < 0.001:
        print(f"  => Significant at p<0.001")
    elif p_value < 0.01:
        print(f"  => Significant at p<0.01")
    elif p_value < 0.05:
        print(f"  => Significant at p<0.05")
    else:
        print(f"  => NOT significant at p<0.05")

    # How many subjects are worse / same / better with VQ
    n_worse = (diffs > 0).sum()
    n_same = (diffs == 0).sum()
    n_better = (diffs < 0).sum()
    print(f"\nPer-subject: {n_worse} worse, {n_same} same, {n_better} better with VQ")

    # Effect size (rank-biserial correlation)
    n = len(diffs)
    r = 1 - (2 * stat) / (n * (n + 1) / 2)
    print(f"Effect size (rank-biserial r): {r:.4f}")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "model_name": model_name,
            "n_prototypes": int(codebook.shape[0]),
            "n_subjects": int(len(baseline_accs)),
            "baseline_mean_acc": float(baseline_accs.mean()),
            "vq_mean_acc": float(vq_accs.mean()),
            "diff_mean": float(diffs.mean()),
            "diff_std": float(diffs.std()),
            "diff_median": float(np.median(diffs)),
            "wilcoxon_statistic": float(stat),
            "wilcoxon_p_value": float(p_value),
            "n_worse": int(n_worse),
            "n_same": int(n_same),
            "n_better": int(n_better),
            "effect_size_r": float(r),
            "per_subject_baseline": baseline_accs.tolist(),
            "per_subject_vq": vq_accs.tolist(),
        }
        path = os.path.join(
            args.output_dir,
            f"paired_test_{model_name}_vq_m{codebook.shape[0]}.json",
        )
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
