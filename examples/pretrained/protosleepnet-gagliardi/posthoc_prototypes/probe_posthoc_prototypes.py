"""Linear probe on VQ-quantized epoch embeddings passed through frozen sequence encoder.

Pipeline per subject:
  1. Load epoch embeddings (N, d_model)
  2. Quantize to nearest codebook entry
  3. Sliding window voting through frozen sequence encoder -> (N, D_out)
  4. 5-fold subject-wise CV linear probe on contextualized embeddings

Output format matches ``physioex.models.embed.linear_probe()``:
  - probe_results.json: per-fold + pooled + mean±std metrics
  - probe_predictions.json: per-subject softmax proba + labels

Usage:
    python probe_posthoc_prototypes.py \
        --model_dir /path/to/st-baseline \
        --codebook_path /path/to/codebook_vq_m48.npy \
        --emb_dir /path/to/posthoc_embeddings/st-baseline/hmc/all \
        --output_dir /path/to/results \
        --gpu_id 0
"""
import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from physioex.explain.prototypes.posthoc.vq import quantize_embeddings
from physioex.train.metrics import (
    accuracy_score as _accuracy_score,
    f1_score as _f1_score,
    cohen_kappa_score as _cohen_kappa_score,
    confusion_matrix as _confusion_matrix,
    _per_class_f1,
)

SLEEP_CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
SEQ_LEN = 21


# ── Model loading ──────────────────────────────────────────────────


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
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_sequence_encoder_fn(model):
    """Return a function that takes (B, L, d_model) -> (B, L, D_out)."""

    # ProtoSleepTransformer / SleepTransformer
    if hasattr(model, "sequence_encoder"):
        def fn(z):
            return model.sequence_encoder(z)
        return fn

    # ProtoSeqSleepNet / SeqSleepNet
    if hasattr(model, "seqn2"):
        def fn(z):
            out, _ = model.seqn2(z)
            return out
        return fn

    raise ValueError(f"Unknown model type: {type(model).__name__}")


# ── VQ + Sequence encoding per subject ─────────────────────────────


@torch.no_grad()
def encode_subject_vq(
    Z_subj: np.ndarray,
    codebook: np.ndarray,
    seq_encoder_fn,
    L: int,
    device: torch.device,
) -> np.ndarray:
    """Quantize epoch embeddings and contextualize through sequence encoder.

    Uses sliding window voting with L offsets, same as model evaluation.

    Args:
        Z_subj: (N, d_model) epoch embeddings for one subject.
        codebook: (M, d_model) VQ codebook.
        seq_encoder_fn: (B, L, d_model) -> (B, L, D_out).
        L: Sequence length for sliding window.
        device: Torch device.

    Returns:
        (N, D_out) contextualized embeddings after voting.
    """
    # Quantize
    Z_q, _ = quantize_embeddings(Z_subj, codebook)
    Z_q = torch.from_numpy(Z_q).float().to(device)

    N = Z_q.shape[0]

    if N < L:
        # Pad to L
        pad = torch.zeros(L - N, Z_q.shape[1], device=device, dtype=Z_q.dtype)
        padded = torch.cat([Z_q, pad], dim=0)
        out = seq_encoder_fn(padded.unsqueeze(0))  # (1, L, D_out)
        return out[0, :N].cpu().float().numpy()

    # Probe D_out
    probe = seq_encoder_fn(Z_q[:L].unsqueeze(0))  # (1, L, D_out)
    D_out = probe.shape[-1]

    votes = torch.zeros(N, D_out, device=device, dtype=probe.dtype)
    counts = torch.zeros(N, device=device, dtype=torch.float32)

    for offset in range(L):
        z = Z_q[offset:]
        usable = z.shape[0] - (z.shape[0] % L)
        if usable == 0:
            continue
        z = z[:usable]
        n_win = usable // L
        z = z.reshape(n_win, L, -1)

        out = seq_encoder_fn(z)  # (n_win, L, D_out)
        out = out.reshape(usable, D_out)

        votes[offset : offset + usable] += out
        counts[offset : offset + usable] += 1

    safe_counts = counts.clamp(min=1).unsqueeze(-1)
    averaged = votes / safe_counts

    return averaged.cpu().float().numpy()


# ── Loading helpers ────────────────────────────────────────────────


def load_subjects_from_dir(emb_dir: str) -> list:
    """Load per-subject embeddings + labels from a directory.

    Supports both flat layout ({subj}_embeddings.npy) and
    subdirectory layout ({subj}/embeddings.npy).

    Returns list of dicts with "id", "embeddings", "labels".
    """
    emb_dir = Path(emb_dir)
    subjects = []

    # Try flat layout first (posthoc_embeddings style)
    emb_files = sorted(emb_dir.glob("*_embeddings.npy"))
    if emb_files:
        for ef in emb_files:
            subj_id = ef.stem.replace("_embeddings", "")
            lf = ef.parent / f"{subj_id}_labels.npy"
            subjects.append({
                "id": subj_id,
                "embeddings": np.load(str(ef)).astype(np.float32),
                "labels": np.load(str(lf)).astype(np.int64),
            })
        return subjects

    # Try subdirectory layout (embed.py style)
    for subj_dir in sorted(emb_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        ef = subj_dir / "embeddings.npy"
        lf = subj_dir / "labels.npy"
        if ef.exists() and lf.exists():
            subjects.append({
                "id": subj_dir.name,
                "embeddings": np.load(str(ef)).astype(np.float32),
                "labels": np.load(str(lf)).astype(np.int64),
            })

    return subjects


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe on VQ-quantized epoch embeddings + sequence encoder"
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None,
                        help="HF model name for load_from_pretrained")
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--codebook_path", type=str, required=True)
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Directory with per-subject epoch embeddings")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=21,
                        help="Sequence length for voting (default: 21)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load model (frozen, for sequence encoder only)
    if args.model_name:
        from physioex.models import load_from_pretrained
        model_name = args.model_name
        kwargs = {"device": str(device)}
        if args.repo_id:
            kwargs["repo_id"] = args.repo_id
        model = load_from_pretrained(args.model_name, **kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    elif args.model_dir:
        model_name = os.path.basename(args.model_dir)
        model = load_model(args.model_dir, device)
    else:
        parser.error("--model_dir or --model_name required")
    seq_encoder_fn = build_sequence_encoder_fn(model)

    # Load codebook
    codebook = np.load(args.codebook_path)
    M = codebook.shape[0]
    print(f"Codebook: {codebook.shape}")

    # Load epoch embeddings
    print(f"Loading epoch embeddings from {args.emb_dir}")
    subjects = load_subjects_from_dir(args.emb_dir)
    if not subjects:
        raise ValueError(f"No subjects found in {args.emb_dir}")
    print(f"  {len(subjects)} subjects loaded")

    # Step 1: VQ + sequence encode all subjects
    print(f"\nVQ quantization + sequence encoding (L={args.seq_len})...")
    for subj in tqdm(subjects, desc="Encoding"):
        subj["embeddings"] = encode_subject_vq(
            subj["embeddings"], codebook, seq_encoder_fn, args.seq_len, device
        )

    # Determine n_classes
    all_labels = np.concatenate([s["labels"] for s in subjects])
    valid_labels = all_labels[all_labels >= 0]
    n_classes = int(valid_labels.max()) + 1
    class_names = (
        SLEEP_CLASS_NAMES if n_classes == 5 else [str(i) for i in range(n_classes)]
    )

    D_out = subjects[0]["embeddings"].shape[1]
    print(f"  D_out={D_out}, n_classes={n_classes}")

    # Step 2: 5-fold subject-wise CV linear probe
    n_subjects = len(subjects)
    n_folds = args.n_folds
    rng = np.random.RandomState(42)
    indices = np.arange(n_subjects)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    per_fold = []
    all_logits_list = []
    all_targets_list = []
    subject_predictions = []

    print(
        f"\nLinear probe: {n_subjects} subjects, {n_folds}-fold CV, "
        f"{n_classes} classes ({', '.join(class_names)})"
    )

    for fold_idx in range(n_folds):
        test_set = set(folds[fold_idx].tolist())
        train_idx = [i for i in range(n_subjects) if i not in test_set]
        test_idx = folds[fold_idx].tolist()

        train_embs = np.concatenate([subjects[i]["embeddings"] for i in train_idx])
        train_lbls = np.concatenate([subjects[i]["labels"] for i in train_idx])
        test_embs = np.concatenate([subjects[i]["embeddings"] for i in test_idx])
        test_lbls = np.concatenate([subjects[i]["labels"] for i in test_idx])

        train_mask = train_lbls >= 0
        test_mask = test_lbls >= 0
        train_embs, train_lbls = train_embs[train_mask], train_lbls[train_mask]
        test_embs, test_lbls = test_embs[test_mask], test_lbls[test_mask]

        # Standard scaling
        mean = train_embs.mean(axis=0)
        std = train_embs.std(axis=0) + 1e-8
        train_embs = (train_embs - mean) / std
        test_embs = (test_embs - mean) / std

        X_train = torch.from_numpy(train_embs)
        y_train = torch.from_numpy(train_lbls)
        X_test = torch.from_numpy(test_embs)
        y_test = torch.from_numpy(test_lbls)

        # Train linear classifier
        probe_model = nn.Linear(D_out, n_classes).to(device)
        optimizer = torch.optim.Adam(
            probe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=args.batch_size, shuffle=True, drop_last=False,
        )

        probe_model.train()
        for _ in range(args.max_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(probe_model(xb), yb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Evaluate
        probe_model.eval()
        test_logits_chunks = []
        with torch.no_grad():
            for i in range(0, len(X_test), args.batch_size):
                chunk = X_test[i : i + args.batch_size].to(device)
                test_logits_chunks.append(probe_model(chunk).cpu())
        test_logits = torch.cat(test_logits_chunks, dim=0)

        # Per-subject predictions
        offset = 0
        for subj_i in test_idx:
            subj = subjects[subj_i]
            subj_labels = subj["labels"]
            valid_mask_subj = subj_labels >= 0
            n_valid = int(valid_mask_subj.sum())
            subj_logits = test_logits[offset : offset + n_valid]
            subj_proba = torch.nn.functional.softmax(subj_logits, dim=-1)
            subject_predictions.append({
                "subject_id": subj["id"],
                "fold": fold_idx,
                "proba": subj_proba.tolist(),
                "labels": subj_labels[valid_mask_subj].tolist(),
            })
            offset += n_valid

        # Metrics
        acc = _accuracy_score(test_logits, y_test, ignore_index=None)
        mf1 = _f1_score(test_logits, y_test, ignore_index=None)
        kappa = _cohen_kappa_score(test_logits, y_test, ignore_index=None)
        cm = _confusion_matrix(test_logits, y_test, ignore_index=None)
        preds = test_logits.argmax(dim=-1)
        pcf1, support = _per_class_f1(preds, y_test, n_classes)

        fold_result = {
            "fold": fold_idx,
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "kappa": round(kappa, 4),
            "per_class_f1": {
                name: round(pcf1[i], 4) for i, name in enumerate(class_names)
            },
            "support": {
                name: int(support[i]) for i, name in enumerate(class_names)
            },
            "n_train_subjects": len(train_idx),
            "n_test_subjects": len(test_idx),
            "n_train_epochs": int(train_mask.sum()),
            "n_test_epochs": int(test_mask.sum()),
            "confusion_matrix": cm.tolist(),
        }
        per_fold.append(fold_result)
        all_logits_list.append(test_logits)
        all_targets_list.append(y_test)

        pcf1_str = "  ".join(
            f"{name}={pcf1[i]:.2f}" for i, name in enumerate(class_names)
        )
        print(
            f"  Fold {fold_idx}: ACC={acc:.4f}  MF1={mf1:.4f}  "
            f"\u03ba={kappa:.4f}  [{pcf1_str}]"
        )

    # Pooled metrics
    all_logits = torch.cat(all_logits_list)
    all_targets = torch.cat(all_targets_list)

    pooled_acc = _accuracy_score(all_logits, all_targets, ignore_index=None)
    pooled_mf1 = _f1_score(all_logits, all_targets, ignore_index=None)
    pooled_kappa = _cohen_kappa_score(all_logits, all_targets, ignore_index=None)
    pooled_cm = _confusion_matrix(all_logits, all_targets, ignore_index=None)
    pooled_preds = all_logits.argmax(dim=-1)
    pooled_pcf1, pooled_support = _per_class_f1(pooled_preds, all_targets, n_classes)

    fold_accs = [f["accuracy"] for f in per_fold]
    fold_mf1s = [f["macro_f1"] for f in per_fold]
    fold_kappas = [f["kappa"] for f in per_fold]

    results = {
        "model_name": model_name,
        "codebook_path": args.codebook_path,
        "n_prototypes": int(M),
        "emb_dir": args.emb_dir,
        "n_folds": n_folds,
        "n_classes": n_classes,
        "class_names": class_names,
        "n_subjects": n_subjects,
        "probe_config": {
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
        "per_fold": per_fold,
        "pooled": {
            "accuracy": round(pooled_acc, 4),
            "macro_f1": round(pooled_mf1, 4),
            "kappa": round(pooled_kappa, 4),
            "per_class_f1": {
                name: round(pooled_pcf1[i], 4)
                for i, name in enumerate(class_names)
            },
            "support": {
                name: int(pooled_support[i])
                for i, name in enumerate(class_names)
            },
            "confusion_matrix": pooled_cm.tolist(),
        },
        "mean_std": {
            "accuracy": {
                "mean": round(float(np.mean(fold_accs)), 4),
                "std": round(float(np.std(fold_accs)), 4),
            },
            "macro_f1": {
                "mean": round(float(np.mean(fold_mf1s)), 4),
                "std": round(float(np.std(fold_mf1s)), 4),
            },
            "kappa": {
                "mean": round(float(np.mean(fold_kappas)), 4),
                "std": round(float(np.std(fold_kappas)), 4),
            },
        },
    }

    # Print summary
    pcf1_str = "  ".join(
        f"{name}={pooled_pcf1[i]:.2f}" for i, name in enumerate(class_names)
    )
    print(
        f"\n  Pooled:   ACC={pooled_acc:.4f}  MF1={pooled_mf1:.4f}  "
        f"\u03ba={pooled_kappa:.4f}"
    )
    print(f"  Per-class F1: {pcf1_str}")
    print(
        f"  Mean\u00b1SD: ACC={np.mean(fold_accs):.4f}\u00b1{np.std(fold_accs):.4f}  "
        f"MF1={np.mean(fold_mf1s):.4f}\u00b1{np.std(fold_mf1s):.4f}  "
        f"\u03ba={np.mean(fold_kappas):.4f}\u00b1{np.std(fold_kappas):.4f}"
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    predictions_path = os.path.join(args.output_dir, "probe_predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(subject_predictions, f)
    print(f"  Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
