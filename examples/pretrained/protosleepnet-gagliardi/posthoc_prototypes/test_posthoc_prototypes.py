"""Test accuracy loss when constraining baseline models to use post-hoc prototypes.

Evaluates two approaches:
  - NMF: nearest-prototype classification (no sequence encoder)
  - VQ:  quantized epoch embeddings -> frozen sequence encoder + classifier

Also runs the baseline (no prototypes) for comparison.

Usage:
    # Baseline (reference)
    python test_posthoc_prototypes.py --model_dir /path/to/st-baseline --method baseline

    # NMF
    python test_posthoc_prototypes.py --model_dir /path/to/st-baseline --method nmf \
        --prototypes_path /path/to/prototypes_nmf_k10.npy \
        --emb_dir /path/to/embeddings/st-baseline

    # VQ
    python test_posthoc_prototypes.py --model_dir /path/to/st-baseline --method vq \
        --codebook_path /path/to/codebook_vq_m50.npy
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from physioex.data.datasets import get_dataset
from physioex.train.trainer import Trainer
from physioex.explain.prototypes.posthoc import (
    load_epoch_embeddings,
    nearest_prototype_classify,
    quantize_embeddings,
    evaluate_metrics,
)

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"
SEQ_LEN = 21
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


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


# ── VQ-wrapped model ───────────────────────────────────────────────


class VQWrappedModel(nn.Module):
    """Wraps a model to quantize epoch embeddings before sequence encoding.

    Pipeline: x -> per-channel epoch_encoder -> mean pool -> VQ -> seq_encoder -> clf

    Supports ProtoSleepTransformer, ProtoSeqSleepNet, SleepTransformer, SeqSleepNet.
    """

    def __init__(self, model, codebook):
        super().__init__()
        self.model = model
        self.register_buffer("codebook", torch.from_numpy(codebook).float())

    def _epoch_encode(self, x):
        """Extract h(x): per-epoch embeddings from epoch encoder.

        Args:
            x: (N, C, T, F) input spectrograms.

        Returns:
            (N, d_model) epoch embeddings.
        """
        N, C, T, F = x.shape

        # ProtoSleepTransformer / ProtoSeqSleepNet: per-channel + mean pool
        if hasattr(self.model, "epoch_encoder") and hasattr(self.model, "in_chan"):
            x_flat = x.reshape(N * C, 1, T, F)
            embs = self.model.epoch_encoder(x_flat)
            embs = embs.reshape(N, C, -1)
            return embs.mean(dim=1)

        # Plain SleepTransformer
        if hasattr(self.model, "epoch_encoder"):
            return self.model.epoch_encoder(x)

        # Plain SeqSleepNet
        if hasattr(self.model, "filterbank") and hasattr(self.model, "seqn1"):
            z = self.model.filterbank(x)
            z = z.permute(0, 2, 1, 3)
            z = z.reshape(z.shape[0], z.shape[1], -1)
            z, _ = self.model.seqn1(z)
            z = self.model.attention(z)
            return z

        raise ValueError(f"Unknown model type: {type(self.model).__name__}")

    def _quantize(self, z):
        """Quantize to nearest codebook entry."""
        z_sq = (z ** 2).sum(dim=1, keepdim=True)
        c_sq = (self.codebook ** 2).sum(dim=1, keepdim=True).T
        dist = z_sq + c_sq - 2 * (z @ self.codebook.T)
        idx = dist.argmin(dim=1)
        return self.codebook[idx]

    def _sequence_encode_and_classify(self, z, B, L):
        """Run sequence encoder + classifier."""
        z = z.reshape(B, L, -1)

        # ProtoSleepTransformer / SleepTransformer
        if hasattr(self.model, "sequence_encoder") and hasattr(self.model, "classifier"):
            z = self.model.sequence_encoder(z)
            z = z.reshape(B * L, -1)
            return self.model.classifier(z).reshape(B, L, -1)

        # ProtoSeqSleepNet
        if hasattr(self.model, "seqn2") and hasattr(self.model, "classifier"):
            z, _ = self.model.seqn2(z)
            z = z.reshape(B * L, -1)
            return self.model.classifier(z).reshape(B, L, -1)

        # Plain SeqSleepNet
        if hasattr(self.model, "seqn2") and hasattr(self.model, "clf"):
            z, _ = self.model.seqn2(z)
            z = z.reshape(B * L, -1)
            return self.model.clf(z).reshape(B, L, -1)

        raise ValueError(f"Unknown model type: {type(self.model).__name__}")

    def forward(self, x):
        B, L, C, T, F_dim = x.shape
        x_flat = x.reshape(B * L, C, T, F_dim)
        z = self._epoch_encode(x_flat)      # (B*L, d_model)
        z_q = self._quantize(z)             # (B*L, d_model)
        return self._sequence_encode_and_classify(z_q, B, L)


# ── Sliding-window voting ──────────────────────────────────────────


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
    return F.softmax(logits.squeeze(0), dim=-1).cpu()


# ── Test methods ────────────────────────────────────────────────────


def test_nmf(args):
    """Test NMF nearest-prototype classification on pre-extracted embeddings."""
    print("=" * 60)
    print("NMF Prototype Test")
    print("=" * 60)

    Z_test, Y_test = load_epoch_embeddings(args.emb_dir, split="test")
    valid = Y_test >= 0
    Z_test, Y_test = Z_test[valid], Y_test[valid]
    print(f"Test: {len(Z_test)} valid epochs")

    prototypes = np.load(args.prototypes_path)
    proto_labels_path = args.prototypes_path.replace("prototypes_", "proto_labels_")
    proto_labels = np.load(proto_labels_path)
    print(f"Prototypes: {prototypes.shape}")

    Y_pred = nearest_prototype_classify(Z_test, prototypes, proto_labels, metric="cosine")
    metrics = evaluate_metrics(Y_test, Y_pred, CLASS_NAMES)
    _print_metrics(metrics, "NMF Prototype")
    return metrics


def test_vq(args, device):
    """Test VQ-quantized model with frozen sequence encoder + classifier."""
    print("=" * 60)
    print("VQ Prototype Test")
    print("=" * 60)

    if args.model_name:
        from physioex.models import load_from_pretrained
        kwargs = {"device": str(device)}
        if args.repo_id:
            kwargs["repo_id"] = args.repo_id
        model = load_from_pretrained(args.model_name, **kwargs)
        model.eval()
    else:
        model, config = load_model(args.model_dir, device)
    codebook = np.load(args.codebook_path)
    print(f"Codebook: {codebook.shape}")

    vq_model = VQWrappedModel(model, codebook).to(device).eval()

    channels = args.channels if args.channels else CHANNELS
    ds_kwargs = {"visit": 1} if args.dataset == "shhs" else {}
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=channels, pipelines=PIPELINE,
        sequence_length=args.seq_len, **ds_kwargs,
    )
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, pin_memory=False, fold=args.fold,
    )
    print(f"Test subjects: {len(test_loader)}")

    subject_predictions = []
    all_proba, all_targets = [], []
    for subj_idx, batch in enumerate(tqdm(test_loader, desc="VQ eval")):
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels
            inputs = stack_channels(batch)
            targets = batch["labels"]
        else:
            inputs, targets = batch
        proba = evaluate_subject(vq_model, inputs, args.seq_len, device)
        targets_flat = targets.reshape(-1)

        subject_predictions.append({
            "subject_idx": subj_idx,
            "proba": proba.tolist(),
            "labels": targets_flat.tolist(),
        })
        all_proba.append(proba)
        all_targets.append(targets_flat)

    preds = torch.cat(all_proba, dim=0)
    targets = torch.cat(all_targets, dim=0)
    valid = targets >= 0
    Y_pred = preds[valid].argmax(dim=1).numpy()
    Y_true = targets[valid].numpy()

    metrics = evaluate_metrics(Y_true, Y_pred, CLASS_NAMES)
    _print_metrics(metrics, "VQ Prototype")
    return metrics, subject_predictions


def test_baseline(args, device):
    """Test baseline model without prototypes (reference accuracy)."""
    print("=" * 60)
    print("Baseline Test")
    print("=" * 60)

    if args.model_name:
        from physioex.models import load_from_pretrained
        kwargs = {"device": str(device)}
        if args.repo_id:
            kwargs["repo_id"] = args.repo_id
        model = load_from_pretrained(args.model_name, **kwargs)
        model.eval()
    else:
        model, config = load_model(args.model_dir, device)

    channels = args.channels if args.channels else CHANNELS
    ds_kwargs = {"visit": 1} if args.dataset == "shhs" else {}
    DatasetClass = get_dataset(args.dataset)
    dataset = DatasetClass(
        channels=channels, pipelines=PIPELINE,
        sequence_length=args.seq_len, **ds_kwargs,
    )
    _, _, test_loader = Trainer.build_dataloaders(
        dataset=dataset, train_batch_size=1, eval_batch_size=1,
        num_workers=0, pin_memory=False, fold=args.fold,
    )
    print(f"Test subjects: {len(test_loader)}")

    all_proba, all_targets = [], []
    for batch in tqdm(test_loader, desc="Baseline eval"):
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels
            inputs = stack_channels(batch)
            targets = batch["labels"]
        else:
            inputs, targets = batch
        proba = evaluate_subject(model, inputs, args.seq_len, device)
        all_proba.append(proba)
        all_targets.append(targets.reshape(-1))

    preds = torch.cat(all_proba, dim=0)
    targets = torch.cat(all_targets, dim=0)
    valid = targets >= 0
    Y_pred = preds[valid].argmax(dim=1).numpy()
    Y_true = targets[valid].numpy()

    metrics = evaluate_metrics(Y_true, Y_pred, CLASS_NAMES)
    _print_metrics(metrics, "Baseline")
    return metrics


# ── Helpers ─────────────────────────────────────────────────────────


def _print_metrics(metrics, label):
    print(f"\n{label} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"  Kappa:     {metrics['kappa']:.4f}")
    print(f"  Per-class F1:")
    for name, f1 in metrics["per_class_f1"].items():
        print(f"    {name}: {f1:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test accuracy loss with post-hoc prototypes"
    )
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with config.json + model.pt")
    parser.add_argument("--model_name", type=str, default=None,
                        help="HF model name for load_from_pretrained")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HF repo ID")
    parser.add_argument("--method", type=str, required=True,
                        choices=["nmf", "vq", "baseline"])
    parser.add_argument("--emb_dir", type=str, default=None,
                        help="Directory with pre-extracted embeddings (for nmf)")
    parser.add_argument("--prototypes_path", type=str, default=None,
                        help="Path to prototypes_nmf_kN.npy (for nmf)")
    parser.add_argument("--codebook_path", type=str, default=None,
                        help="Path to codebook_vq_mN.npy (for vq)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="shhs")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Channel list (default: EEG EOG EMG)")
    parser.add_argument("--seq_len", type=int, default=21,
                        help="Sequence length for voting (default: 21)")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.model_name:
        model_name = args.model_name
    elif args.model_dir:
        model_name = os.path.basename(args.model_dir)
    else:
        parser.error("--model_dir or --model_name required")
    print(f"Model: {model_name}")

    subject_predictions = None

    if args.method == "nmf":
        if not args.emb_dir or not args.prototypes_path:
            parser.error("--emb_dir and --prototypes_path required for nmf")
        metrics = test_nmf(args)
    elif args.method == "vq":
        if not args.codebook_path:
            parser.error("--codebook_path required for vq")
        metrics, subject_predictions = test_vq(args, device)
    elif args.method == "baseline":
        metrics = test_baseline(args, device)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Determine suffix from method + params
        if args.method == "vq" and args.codebook_path:
            # Extract M from codebook filename (codebook_vq_m48.npy -> m48)
            cb_stem = os.path.splitext(os.path.basename(args.codebook_path))[0]
            suffix = cb_stem.replace("codebook_", "")  # vq_m48
        elif args.method == "nmf" and args.prototypes_path:
            p_stem = os.path.splitext(os.path.basename(args.prototypes_path))[0]
            suffix = p_stem.replace("prototypes_", "")
        else:
            suffix = args.method

        # Metrics JSON
        metrics_path = os.path.join(
            args.output_dir,
            f"metrics_{model_name}_{suffix}.json",
        )
        result = {
            "model_name": model_name,
            "method": args.method,
            "metrics": metrics,
        }
        if args.prototypes_path:
            result["prototypes_path"] = args.prototypes_path
        if args.codebook_path:
            result["codebook_path"] = args.codebook_path
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Per-subject predictions JSON
        if subject_predictions is not None:
            preds_path = os.path.join(
                args.output_dir,
                f"predictions_{model_name}_{suffix}.json",
            )
            with open(preds_path, "w") as f:
                json.dump(subject_predictions, f)
            print(f"Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()
