"""Learn a per-channel roto-translation on STFT spectrograms to adapt
frozen sleep staging models to the Alzheimer dataset.

All models are frozen. A shared SpectrogramTransform (3 matrices A + 3 biases b)
is optimized to minimize the sum of CE losses across all models on the HC
(Healthy Controls) train split. Evaluation on HC test + AD splits.

Models:
  - seqsleepnet-phan (1ch EEG, MASS)
  - sleeptransformer-phan (1ch EEG, SHHS)
  - seqsleepnet-phan-3ch (3ch, SHHS)
  - sleeptransformer-phan-3ch (3ch, SHHS)
  - protosleepnet-seq-3ch-mixer (3ch, MASS)
  - protosleepnet-st-3ch-mixer (3ch, SHHS)

Usage:
    python train_roto_translation.py --gpu_id 0 --output_dir /path/to/results
"""
import argparse
import importlib
import json
import os

import torch
import torch.nn as nn

from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.models.protosleepnet import ProtoSleepNet
from physioex.train.trainer import Trainer
from physioex.train.metrics import accuracy_score

CHANNELS = ["EEG", "EOG", "EMG"]
PIPELINE = "seqsleepnet"

# Model registry: (name, type, channels, seq_len, load_fn)
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    os.path.expanduser("~/experiments/protosleepnet/models"),
)


# ── SpectrogramTransform ─────────────────────────────────────────────


class SpectrogramTransform(nn.Module):
    """Learnable per-channel band-diagonal transform on STFT spectrograms.

    Each frequency bin interacts only with its bandwidth nearest neighbors.
    Initialized to identity (diagonal=1, off-diagonal=0).

    Parameters per channel: F * (2*bandwidth+1) + F (bias) ≈ F*7 for bw=3.
    Total for 3 channels with F=129, bw=3: 3 * (129*7 + 129) = 3,096 params.
    """

    def __init__(self, n_channels=3, F=129, bandwidth=3):
        super().__init__()
        self.n_channels = n_channels
        self.F = F
        self.bandwidth = bandwidth

        # Store band-diagonal entries as (F, 2*bandwidth+1) per channel
        # Initialized: center (diagonal) = 1, off-diags = 0
        w = 2 * bandwidth + 1
        self.bands = nn.ParameterList()
        for _ in range(n_channels):
            band = torch.zeros(F, w)
            band[:, bandwidth] = 1.0  # diagonal = 1 (identity)
            self.bands.append(nn.Parameter(band))

        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(F)) for _ in range(n_channels)]
        )

    def _build_matrix(self, band):
        """Build sparse (F, F) matrix from band-diagonal entries (F, 2*bw+1)."""
        F = self.F
        bw = self.bandwidth
        A = torch.zeros(F, F, device=band.device, dtype=band.dtype)
        for k in range(-bw, bw + 1):
            col = k + bw  # index in band tensor
            if k >= 0:
                A[range(F - k), range(k, F)] = band[:F - k, col]
            else:
                A[range(-k, F), range(F + k)] = band[-k:, col]
        return A

    def forward(self, x):
        """Transform spectrograms: x_out[c] = x_in[c] @ A[c] + b[c]

        A[c] is a band-diagonal matrix with bandwidth=self.bandwidth.

        Args:
            x: (B, L, C, T, F) spectrograms.
        Returns:
            (B, L, C, T, F) transformed spectrograms.
        """
        x = x.clone()
        C = min(x.shape[2], self.n_channels)
        for c in range(C):
            A = self._build_matrix(self.bands[c])
            x[:, :, c] = x[:, :, c] @ A + self.b[c]
        return x


# ── MultiModelWrapper ────────────────────────────────────────────────


class MultiModelWrapper(nn.Module):
    """Wraps transform + all frozen models."""

    def __init__(self, transform, models_1ch, models_3ch):
        super().__init__()
        self.transform = transform
        self.models_1ch = nn.ModuleList(models_1ch)
        self.models_3ch = nn.ModuleList(models_3ch)
        # Freeze all models
        for m in list(self.models_1ch) + list(self.models_3ch):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def _forward_all(self, x):
        """Returns list of logits from all models (for per-model loss)."""
        x_t = self.transform(x)  # (B, L, 3, T, F)

        all_logits = []
        for m in self.models_3ch:
            all_logits.append(m(x_t))
        for m in self.models_1ch:
            all_logits.append(m(x_t[:, :, 0:1]))  # EEG only

        return all_logits

    def forward(self, x):
        """Returns mean logits across all models (Trainer-compatible)."""
        all_logits = self._forward_all(x)
        return torch.stack(all_logits).mean(dim=0)


# ── SingleModelWithTransform ─────────────────────────────────────────


class SingleModelWithTransform(nn.Module):
    """Wraps transform + single model for evaluation."""

    def __init__(self, transform, model, n_channels=3):
        super().__init__()
        self.transform = transform
        self.model = model
        self.n_channels = n_channels

    def forward(self, x):
        x_t = self.transform(x)
        if self.n_channels == 1:
            x_t = x_t[:, :, 0:1]
        return self.model(x_t)


# ── MultiModelTrainer ────────────────────────────────────────────────


class MultiModelTrainer(Trainer):
    """Trainer that sums CE loss from all models in MultiModelWrapper."""

    @staticmethod
    def _step(model, batch, loss_fn, device):
        if isinstance(batch, dict) and "signals" in batch:
            from physioex.data.collate import stack_channels
            inputs = stack_channels(batch).to(device)
            targets = batch["labels"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

        with torch.autocast(device.type if "cuda" in device.type else "cpu"):
            all_logits = model._forward_all(inputs)

        targets_flat = targets.reshape(-1)
        n_classes = all_logits[0].shape[-1]

        total_loss = torch.tensor(0.0, device=device)
        for logits in all_logits:
            logits_flat = logits.reshape(-1, n_classes)
            total_loss = total_loss + loss_fn(logits_flat, targets_flat)

        # Accuracy from mean logits
        mean_logits = torch.stack(all_logits).mean(dim=0)
        acc = accuracy_score(
            mean_logits.reshape(-1, n_classes),
            targets_flat,
            ignore_index=getattr(loss_fn, "ignore_index", None),
        )

        return total_loss, acc


# ── Model loading ────────────────────────────────────────────────────


def load_model_from_dir(model_dir, device):
    """Load model from config.json + model.pt."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    module_path, class_name = config["model_class"].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)

    if "factory" in config:
        factory = getattr(ModelClass, config["factory"])
        model = factory(**config.get("factory_kwargs", {}))
    else:
        model = ModelClass(**config["model_kwargs"])

    weights_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model.to(device).eval()


def load_all_models(models_dir, device):
    """Load all 6 models. Returns (models_1ch, models_3ch, model_names)."""
    models_1ch = []
    models_3ch = []
    names_1ch = []
    names_3ch = []

    # 1ch models from HuggingFace
    for name in ["seqsleepnet-phan", "sleeptransformer-phan"]:
        print(f"Loading {name} (1ch) from HF...")
        m = load_from_pretrained(name, device=str(device))
        m.eval()
        models_1ch.append(m)
        names_1ch.append(name)

    # 3ch models from local dirs
    for name in [
        "seqsleepnet-phan-3ch",
        "sleeptransformer-phan-3ch",
        "protosleepnet-seq-3ch-mixer",
        "protosleepnet-st-3ch-mixer",
    ]:
        model_dir = os.path.join(models_dir, name)
        print(f"Loading {name} (3ch) from {model_dir}...")
        m = load_model_from_dir(model_dir, device)
        models_3ch.append(m)
        names_3ch.append(name)

    return models_1ch, models_3ch, names_1ch, names_3ch


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Learn roto-translation for Alzheimer domain adaptation"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="fix_alzheimer_output")
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=21)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ── Load models ──────────────────────────────────────────────
    models_1ch, models_3ch, names_1ch, names_3ch = load_all_models(
        args.models_dir, device
    )
    all_names = names_3ch + names_1ch
    all_nch = [3] * len(names_3ch) + [1] * len(names_1ch)
    all_models = list(models_3ch) + list(models_1ch)

    print(f"\nLoaded {len(all_models)} models: {all_names}")

    # ── Dataset: HC for training ─────────────────────────────────
    DatasetClass = get_dataset("alzheimers")
    hc_dataset = DatasetClass(
        channels=CHANNELS,
        pipelines=PIPELINE,
        sequence_length=args.seq_len,
        subset="HC",
    )
    print(f"HC dataset: {hc_dataset.get_n_subjects()} subjects")

    # ── Build wrapper ────────────────────────────────────────────
    transform = SpectrogramTransform(n_channels=3, F=129)
    wrapper = MultiModelWrapper(transform, models_1ch, models_3ch)
    wrapper = wrapper.to(device)

    n_transform_params = sum(p.numel() for p in transform.parameters())
    n_frozen_params = sum(p.numel() for p in wrapper.parameters() if not p.requires_grad)
    print(f"Transform params: {n_transform_params:,}")
    print(f"Frozen params: {n_frozen_params:,}")

    # ── Train transform ──────────────────────────────────────────
    optimizer = torch.optim.Adam(transform.parameters(), lr=args.lr)

    print(f"\n{'='*60}")
    print("Training roto-translation on HC train split")
    print(f"{'='*60}")

    wrapper = MultiModelTrainer.train(
        model=wrapper,
        dataset=hc_dataset,
        max_epochs=args.max_epochs,
        train_batch_size=args.batch_size,
        fold=0,
        gpu_id=args.gpu_id,
        optimizer=optimizer,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=args.patience,
    )

    # ── Save transform ───────────────────────────────────────────
    transform_path = os.path.join(args.output_dir, "transform.pt")
    torch.save(transform.state_dict(), transform_path)
    print(f"\nSaved transform to {transform_path}")

    # ── Evaluate each model ──────────────────────────────────────
    results = {}

    for model, name, nch in zip(all_models, all_names, all_nch):
        print(f"\n{'='*60}")
        print(f"Evaluating {name} ({nch}ch)")
        print(f"{'='*60}")

        wrapped = SingleModelWithTransform(transform, model, n_channels=nch)

        # HC test split
        print(f"\n--- HC test ---")
        hc_results = Trainer.voting_evaluate(
            model=wrapped,
            dataset=hc_dataset,
            L=args.seq_len,
            fold=0,
            gpu_id=args.gpu_id,
        )
        hc_metrics = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in hc_results.items()
        }
        print(f"  HC: acc={hc_results['accuracy']:.4f}, "
              f"f1={hc_results['f1_score']:.4f}, kappa={hc_results['cohen_kappa']:.4f}")

        # AD (all subjects)
        print(f"\n--- AD ---")
        ad_dataset = DatasetClass(
            channels=CHANNELS,
            pipelines=PIPELINE,
            sequence_length=args.seq_len,
            subset="AD",
        )
        ad_results = Trainer.voting_evaluate(
            model=wrapped,
            dataset=ad_dataset,
            L=args.seq_len,
            fold=0,
            gpu_id=args.gpu_id,
        )
        ad_metrics = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in ad_results.items()
        }
        print(f"  AD: acc={ad_results['accuracy']:.4f}, "
              f"f1={ad_results['f1_score']:.4f}, kappa={ad_results['cohen_kappa']:.4f}")

        results[name] = {"hc_test": hc_metrics, "ad": ad_metrics}

    # ── Save all results ─────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("Summary (with roto-translation)")
    print(f"{'='*80}")
    print(f"{'Model':40s} {'HC ACC':>8s} {'HC κ':>8s} {'AD ACC':>8s} {'AD κ':>8s}")
    print("-" * 80)
    for name in all_names:
        r = results[name]
        hc = r["hc_test"]
        ad = r["ad"]
        print(
            f"{name:40s} "
            f"{hc['accuracy']:8.4f} {hc['cohen_kappa']:8.4f} "
            f"{ad['accuracy']:8.4f} {ad['cohen_kappa']:8.4f}"
        )


if __name__ == "__main__":
    main()
