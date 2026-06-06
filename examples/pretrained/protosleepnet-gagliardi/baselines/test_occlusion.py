"""Channel occlusion robustness test for baseline 3ch models.

Tests how the original Phan models (SleepTransformer, SeqSleepNet) with
3 channels handle missing/occluded channels at inference time.

Evaluates on the test split with 5 scenarios (clean + 4 occlusion types).
Supports SHHS (SleepTransformer) and MASS (SeqSleepNet) datasets.

Usage:
    # SleepTransformer on SHHS
    python examples/pretrained/protosleepnet-gagliardi/baselines/test_occlusion.py \
        --model_dir /path/to/sleeptransformer-phan-3ch --gpu_id 0

    # SeqSleepNet on MASS
    python examples/pretrained/protosleepnet-gagliardi/baselines/test_occlusion.py \
        --model_dir /path/to/seqsleepnet-phan-3ch --dataset mass --seq_len 20 --gpu_id 0
"""
import argparse
import importlib
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.train.trainer import Trainer

# ── Occlusion scenarios ──────────────────────────────────────────────

SCENARIOS = {
    "clean": {"mode": None},
    "random_25": {"mode": "random", "p": 0.25},
    "random_50": {"mode": "random", "p": 0.50},
    "no_eeg": {"mode": "fixed", "channels_to_occlude": [0]},
    "no_eog_emg": {"mode": "fixed", "channels_to_occlude": [1, 2]},
}

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"


# ── Channel occlusion wrapper ───────────────────────────────────────


class ChannelOcclusionWrapper(nn.Module):
    """Wraps a model to apply channel occlusion (zeroing) at input level.

    Modes:
        None:     no occlusion (passthrough)
        "random": zero each channel independently with probability p per epoch
                  (at least 1 channel kept per sample per epoch)
        "fixed":  zero specific channels for the entire night
    """

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


# ── Per-subject voting evaluation ────────────────────────────────────


@torch.no_grad()
def evaluate_subject(model, inputs, L, device):
    """Sliding-window voting evaluation for a single subject."""
    inputs = inputs.to(device)
    night_length = inputs.shape[1]

    if night_length < L:
        y = model(inputs)
        return F.softmax(y.squeeze(0), dim=-1).cpu()

    with torch.no_grad():
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

    proba = F.softmax(logits.squeeze(0), dim=-1)
    return proba.cpu()


# ── Metrics ─────────────────────────────────────────────────────────


def compute_metrics(all_proba, all_targets, ignore_index=-1):
    """Compute metrics from aggregated probabilities and targets."""
    preds = torch.cat(all_proba, dim=0)
    targets = torch.cat(all_targets, dim=0)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    pred_labels = preds.argmax(dim=1)
    n_classes = preds.shape[1]

    correct = (pred_labels == targets).float()
    acc = correct.mean().item()

    per_class_f1 = []
    per_class_prec = []
    per_class_rec = []
    for c in range(n_classes):
        tp = ((pred_labels == c) & (targets == c)).sum().float()
        fp = ((pred_labels == c) & (targets != c)).sum().float()
        fn = ((pred_labels != c) & (targets == c)).sum().float()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        per_class_f1.append(f1.item())
        per_class_prec.append(prec.item())
        per_class_rec.append(rec.item())

    f1_macro = sum(per_class_f1) / n_classes
    prec_macro = sum(per_class_prec) / n_classes
    rec_macro = sum(per_class_rec) / n_classes

    n = len(targets)
    pe = 0.0
    for c in range(n_classes):
        p_pred = (pred_labels == c).float().mean().item()
        p_true = (targets == c).float().mean().item()
        pe += p_pred * p_true
    kappa = (acc - pe) / (1 - pe + 1e-8)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "cohen_kappa": kappa,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_per_class": {CLASS_NAMES[c]: per_class_f1[c] for c in range(n_classes)},
    }


# ── Model loading ──────────────────────────────────────────────────


def load_model(model_dir, device):
    """Load model from config.json + model.pt."""
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


# ── Dataset loading ────────────────────────────────────────────────


def build_dataset(dataset_name, channels, pipeline, seq_len):
    """Build dataset, handling MASS multi-cohort case."""
    if dataset_name == "mass":
        from physioex.data.multi import MultiDataset

        MASS = get_dataset("mass")
        cohort_datasets = []
        for cohort in [1, 2, 3, 4, 5]:
            ds = MASS(
                cohort=cohort,
                channels=channels,
                pipelines=pipeline,
                sequence_length=seq_len,
            )
            n = ds.get_n_subjects()
            print(f"  MASS SS{cohort:02d}: {n} subjects")
            if n > 0:
                cohort_datasets.append(ds)
        return MultiDataset(cohort_datasets)
    else:
        DatasetClass = get_dataset(dataset_name)
        ds_kwargs = {"visit": 1} if dataset_name == "shhs" else {}
        return DatasetClass(
            channels=channels,
            pipelines=pipeline,
            sequence_length=seq_len,
            **ds_kwargs,
        )


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Channel occlusion robustness test")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
    parser.add_argument("--seq_len", type=int, default=21)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--scenario", type=str, default=None,
        choices=list(SCENARIOS.keys()),
        help="Run a single scenario (default: all)",
    )
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load model
    model, config = load_model(args.model_dir, device)
    model_name = os.path.basename(args.model_dir)
    print(f"Model: {model_name} ({config['model_class']})")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    dataset = build_dataset(args.dataset, CHANNELS, PIPELINE, args.seq_len)

    # Build test dataloader (batch_size=1, one subject per batch)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        fold=args.fold,
    )
    print(f"Dataset: {args.dataset}, seq_len: {args.seq_len}, test subjects: {len(test_loader)}")

    # Run scenarios
    all_predictions = {}
    all_metrics = {}

    scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS

    for scenario_name, scenario_cfg in scenarios_to_run.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*60}")

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
                from physioex.data.collate import stack_channels
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
        all_predictions[scenario_name] = subject_predictions

        print(
            f"  acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_macro']:.4f}  "
            f"kappa={metrics['cohen_kappa']:.4f}"
        )
        print(
            f"  per-class F1: "
            + "  ".join(f"{k}={v:.3f}" for k, v in metrics["f1_per_class"].items())
        )

    # Compute deltas from clean
    if "clean" in all_metrics:
        clean = all_metrics["clean"]
        for scenario_name, metrics in all_metrics.items():
            if scenario_name == "clean":
                metrics["delta"] = {k: 0.0 for k in ["accuracy", "f1_macro", "cohen_kappa", "precision_macro", "recall_macro"]}
                metrics["delta"]["f1_per_class"] = {c: 0.0 for c in CLASS_NAMES}
            else:
                metrics["delta"] = {
                    "accuracy": metrics["accuracy"] - clean["accuracy"],
                    "f1_macro": metrics["f1_macro"] - clean["f1_macro"],
                    "cohen_kappa": metrics["cohen_kappa"] - clean["cohen_kappa"],
                    "precision_macro": metrics["precision_macro"] - clean["precision_macro"],
                    "recall_macro": metrics["recall_macro"] - clean["recall_macro"],
                    "f1_per_class": {
                        c: metrics["f1_per_class"][c] - clean["f1_per_class"][c]
                        for c in CLASS_NAMES
                    },
                }

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Summary: {model_name}")
    print(f"{'='*80}")
    header = f"{'Scenario':12s} {'Acc':>7s} {'F1':>7s} {'Kappa':>7s} {'Prec':>7s} {'Rec':>7s}"
    header += "  " + "  ".join(f"{c:>5s}" for c in CLASS_NAMES)
    print(header)
    print("-" * 80)
    for scenario_name in scenarios_to_run:
        m = all_metrics[scenario_name]
        d = m.get("delta", {})
        row = f"{scenario_name:12s} {m['accuracy']:7.4f} {m['f1_macro']:7.4f} {m['cohen_kappa']:7.4f} {m['precision_macro']:7.4f} {m['recall_macro']:7.4f}"
        row += "  " + "  ".join(f"{m['f1_per_class'][c]:5.3f}" for c in CLASS_NAMES)
        print(row)
        if scenario_name != "clean" and d:
            delta_row = f"{'  delta':12s} {d['accuracy']:+7.4f} {d['f1_macro']:+7.4f} {d['cohen_kappa']:+7.4f} {d['precision_macro']:+7.4f} {d['recall_macro']:+7.4f}"
            delta_row += "  " + "  ".join(f"{d['f1_per_class'][c]:+5.3f}" for c in CLASS_NAMES)
            print(delta_row)
    print("=" * 80)

    # Save predictions and metrics
    suffix = f"_{args.scenario}" if args.scenario else ""

    pred_path = os.path.join(args.model_dir, f"occlusion_predictions{suffix}.json")
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"\nSaved predictions to {pred_path}")

    metrics_path = os.path.join(args.model_dir, f"occlusion_metrics{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
