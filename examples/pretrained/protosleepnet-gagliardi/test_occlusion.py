"""Channel occlusion robustness test.

Tests how trained models handle missing/occluded channels at inference time.
Evaluates on SHHS test split with 5 scenarios (clean + 4 occlusion types).

For each scenario, saves per-subject predict_proba and computes metrics
(acc, f1, kappa, precision, recall, per-class f1) with deltas from clean.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/test_occlusion.py \
        --model_dir /path/to/pretrained/st-baseline --gpu_id 0
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
SEQ_LEN = 21


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
            # Ensure at least 1 channel per sample per epoch
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
    """Sliding-window voting evaluation for a single subject.

    Args:
        model: nn.Module, forward(x) -> (B, L, n_classes)
        inputs: (1, night_length, C, T, F) tensor
        L: sequence length
        device: torch device

    Returns:
        proba: (night_length, n_classes) softmax probabilities
    """
    inputs = inputs.to(device)
    night_length = inputs.shape[1]

    if night_length < L:
        y = model(inputs)
        return F.softmax(y.squeeze(0), dim=-1).cpu()

    # Probe n_classes
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


# ── Batched voting evaluation (unused, kept for reference) ───────────


def _collate_nights(nights_list, targets_list):
    """Pad variable-length nights to the same length for batched processing.

    Args:
        nights_list: list of (night_len_i, C, T, F) tensors
        targets_list: list of (night_len_i,) tensors

    Returns:
        padded: (B, max_len, C, T, F)
        targets_padded: (B, max_len) with -1 for padding
        lengths: (B,) actual night lengths
    """
    lengths = [n.shape[0] for n in nights_list]
    max_len = max(lengths)
    B = len(nights_list)
    rest = nights_list[0].shape[1:]  # (C, T, F)

    padded = torch.zeros(B, max_len, *rest, dtype=nights_list[0].dtype)
    targets_padded = torch.full((B, max_len), -1, dtype=targets_list[0].dtype)

    for i, (n, t, l) in enumerate(zip(nights_list, targets_list, lengths)):
        padded[i, :l] = n
        targets_padded[i, :l] = t

    return padded, targets_padded, torch.tensor(lengths, dtype=torch.long)


@torch.no_grad()
def evaluate_batch(model, padded, lengths, L, device, max_chunk=2048):
    """Batched sliding-window voting evaluation.

    Uses Tensor.unfold for vectorized window extraction and scatter_add_
    for vote accumulation. Processes multiple subjects at once.

    Args:
        model: nn.Module, forward(x) -> (B, L, n_classes)
        padded: (B, max_len, C, T, F) padded input tensor
        lengths: (B,) actual night lengths
        L: sequence length
        device: torch device
        max_chunk: max windows per forward pass

    Returns:
        list of (night_len_i, n_classes) softmax probability tensors
    """
    padded = padded.to(device)
    B, max_len = padded.shape[0], padded.shape[1]

    # Unfold: extract all L-length windows with stride 1
    # (B, max_len, C, T, F) -> unfold dim=1 -> (B, N, C, T, F, L)
    N = max_len - L + 1
    windows = padded.unfold(1, L, 1)  # (B, N, C, T, F, L)
    windows = windows.permute(0, 1, 5, 2, 3, 4).contiguous()  # (B, N, L, C, T, F)

    # Flatten to (B*N, L, C, T, F) for forward pass
    rest_dims = windows.shape[3:]
    flat_windows = windows.reshape(B * N, L, *rest_dims)

    # Forward in chunks
    all_outputs = []
    for i in range(0, B * N, max_chunk):
        chunk = flat_windows[i : i + max_chunk]
        out = model(chunk)
        all_outputs.append(out)
    outputs = torch.cat(all_outputs, dim=0)  # (B*N, L, n_classes)
    n_classes = outputs.shape[-1]
    outputs = outputs.reshape(B, N, L, n_classes)

    # Pre-compute scatter indices (same for all subjects in batch)
    win_idx = torch.arange(N, device=device).unsqueeze(1) + torch.arange(L, device=device).unsqueeze(0)  # (N, L)
    flat_win_idx = win_idx.reshape(-1)  # (N*L,)
    flat_win_idx_cls = flat_win_idx.unsqueeze(-1).expand(-1, n_classes)  # (N*L, n_classes)
    ones = torch.ones(N * L, device=device)

    # Scatter-add per subject and extract valid predictions
    results = []
    for i in range(B):
        night_len = lengths[i].item()
        n_valid = night_len - L + 1

        # Only use windows from valid (non-padded) region
        valid_out = outputs[i, :n_valid].reshape(-1, n_classes)  # (n_valid*L, n_classes)
        valid_idx = flat_win_idx_cls[:n_valid * L]
        valid_flat = flat_win_idx[:n_valid * L]

        votes = torch.zeros(night_len, n_classes, device=device, dtype=outputs.dtype)
        votes.scatter_add_(0, valid_idx, valid_out)

        counts = torch.zeros(night_len, device=device)
        counts.scatter_add_(0, valid_flat, ones[:n_valid * L])

        proba = F.softmax(votes / counts.clamp(min=1).unsqueeze(-1), dim=-1)
        results.append(proba.cpu())

    return results


# ── Metrics ─────────────────────────────────────────────────────────


def compute_metrics(all_proba, all_targets, ignore_index=-1):
    """Compute metrics from aggregated probabilities and targets."""
    preds = torch.cat(all_proba, dim=0)  # (N, n_classes)
    targets = torch.cat(all_targets, dim=0)  # (N,)

    # Filter out ignored indices
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    pred_labels = preds.argmax(dim=1)
    n_classes = preds.shape[1]

    # Overall metrics
    correct = (pred_labels == targets).float()
    acc = correct.mean().item()

    # Per-class F1, precision, recall
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

    # Cohen's kappa
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


# ── Main ────────────────────────────────────────────────────────────


def load_model(model_dir, device):
    """Load model from config.json + model.pt."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Resolve model class
    model_class_str = config["model_class"]
    module_path, class_name = model_class_str.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)

    model = ModelClass(**config["model_kwargs"])
    weights_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Handle both formats: raw state_dict or {"model_state_dict": ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Channel occlusion robustness test")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
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

    # Load dataset and get test split
    ds_kwargs = {"visit": 1} if args.dataset == "shhs" else {}
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=SEQ_LEN,
        **ds_kwargs,
    )

    # Build test dataloader (batch_size=1, one subject per batch)
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        fold=args.fold,
    )
    print(f"Dataset: {args.dataset}, test subjects: {len(test_loader)}")

    # Run all scenarios
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

            # inputs: (1, night_length, C, T, F), targets: (1, night_length)
            proba = evaluate_subject(wrapped, inputs, SEQ_LEN, device)

            targets_flat = targets.reshape(-1)
            subject_predictions.append({
                "subject_idx": subj_idx,
                "proba": proba.tolist(),
                "labels": targets_flat.tolist(),
            })
            all_proba.append(proba)
            all_targets.append(targets_flat)

        # Compute metrics
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

    # Save predictions and metrics with scenario suffix if single scenario
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
