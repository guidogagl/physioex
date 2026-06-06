"""Evaluate single-channel SeqSleepNet-Phan on MASS (in-domain).

Loads the pretrained model from HuggingFace and evaluates on the MASS
test split using per-subject voting (L=20). Saves per-subject predictions
and aggregated metrics.

Usage:
    python examples/pretrained/protosleepnet-gagliardi/baselines/test_seq_1ch.py --gpu_id 0
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.data.multi import MultiDataset
from physioex.models import load_from_pretrained
from physioex.train.trainer import Trainer

MODEL_NAME = "seqsleepnet-phan"
CHANNELS = ["EEG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 20
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


@torch.no_grad()
def evaluate_subject(model, inputs, L, device):
    """Sliding-window voting evaluation for a single subject."""
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
    """Compute metrics from aggregated probabilities and targets."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SeqSleepNet-Phan 1ch on MASS"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default="pretrained_output/seqsleepnet-phan-1ch-eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ── Load model from HF ──────────────────────────────────────────
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = load_from_pretrained(MODEL_NAME, device=str(device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}, params: {n_params:,}")

    # ── Dataset: MASS all cohorts ────────────────────────────────────
    MASS = get_dataset("mass")
    cohort_datasets = []
    for cohort in [1, 2, 3, 4, 5]:
        ds = MASS(
            cohort=cohort,
            channels=CHANNELS,
            pipelines=PIPELINE,
            sequence_length=SEQ_LEN,
        )
        n = ds.get_n_subjects()
        print(f"  MASS SS{cohort:02d}: {n} subjects")
        if n > 0:
            cohort_datasets.append(ds)

    dataset = MultiDataset(cohort_datasets)
    print(f"Combined: {dataset}")

    # ── Build test dataloader ────────────────────────────────────────
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        fold=0,
    )
    print(f"Test subjects: {len(test_loader)}")

    # ── Per-subject voting evaluation ────────────────────────────────
    subject_predictions = []
    all_proba = []
    all_targets = []

    for subj_idx, batch in enumerate(tqdm(test_loader, desc="eval")):
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels
            inputs = stack_channels(batch)
            targets = batch["labels"]
        else:
            inputs, targets = batch

        proba = evaluate_subject(model, inputs, SEQ_LEN, device)

        targets_flat = targets.reshape(-1)
        subject_predictions.append({
            "subject_idx": subj_idx,
            "proba": proba.tolist(),
            "labels": targets_flat.tolist(),
        })
        all_proba.append(proba)
        all_targets.append(targets_flat)

    # ── Metrics ──────────────────────────────────────────────────────
    metrics = compute_metrics(all_proba, all_targets)

    print(f"\nResults: acc={metrics['accuracy']:.4f}, "
          f"f1={metrics['f1_macro']:.4f}, kappa={metrics['cohen_kappa']:.4f}")
    print("Per-class F1: " + "  ".join(
        f"{k}={v:.3f}" for k, v in metrics["f1_per_class"].items()))

    # ── Save ─────────────────────────────────────────────────────────
    pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(subject_predictions, f)
    print(f"Saved predictions to {pred_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
