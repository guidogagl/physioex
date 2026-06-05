"""Generic embedding extraction for any PhysioEx model with encode().

Extracts contextualized per-epoch embeddings using sliding-window voting
(same approach as Trainer.voting_evaluate), caches them to disk, and
provides a loader for downstream use.

Works with any ``nn.Module`` that has an ``encode(x) -> (B, L, D)``
method: SeqSleepNet, TinySleepNet, SleepTransformer, LSeqSleepNet, etc.

Cache layout::

    {cache_root}/embeddings/{model_name}/{dataset_name}/
        metadata.json
        linear_probe_results.json
        {subject_id}/
            embeddings.npy       (n_epochs, D)
            embeddings.meta.json
            labels.npy           (n_epochs,)
            labels.meta.json

HuggingFace repos: ``4rooms/{model_name}-embeddings`` (one dataset repo per
model) mirror the same structure.
``load_embeddings()`` downloads from HF automatically if not in local cache.
``linear_probe()`` evaluates embedding quality via 5-fold subject-wise CV.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from physioex.data.cache import (
    ChannelCache,
    recommended_dtype,
    cast_to_cache_dtype,
)

HF_EMBEDDINGS_ORG = "4rooms"


def _cache_root(cache_dir: Optional[str] = None) -> Path:
    root = cache_dir or os.environ.get(
        "PHYSIOEX_CACHE_DIR", os.path.expanduser("~/.cache/physioex")
    )
    return Path(root) / "embeddings"


@torch.no_grad()
def _extract_subject_sliding(
    model: torch.nn.Module,
    signals: torch.Tensor,
    L: int,
    device: torch.device,
) -> np.ndarray:
    """Extract contextualized embeddings for one full-night recording.

    Uses the same sliding-window approach as Trainer.voting_evaluate:
    slides L-sized windows at L different offsets, encodes each window
    via model.encode(), and averages overlapping embeddings.

    Args:
        model: Model with encode(x) -> (B, L, D).
        signals: (1, N, C, ...) full-night signal tensor.
        L: Sequence length the model was trained with.
        device: CUDA or CPU device.

    Returns:
        (N, D) numpy array of averaged embeddings.
    """
    signals = signals.to(device)
    N = signals.shape[1]

    if N < L:
        pad_len = L - N
        pad = torch.zeros(
            1, pad_len, *signals.shape[2:], device=device, dtype=signals.dtype
        )
        padded = torch.cat([signals, pad], dim=1)
        emb = model.encode(padded)  # (1, L, D)
        return emb[0, :N].cpu().float().numpy()

    # Probe embedding dimension
    probe = model.encode(signals[:, :L])  # (1, L, D)
    D = probe.shape[-1]

    votes = torch.zeros(1, N, D, device=device, dtype=probe.dtype)
    counts = torch.zeros(1, N, device=device, dtype=torch.float32)

    for offset in range(L):
        x = signals[:, offset:]
        usable = x.shape[1] - (x.shape[1] % L)
        if usable == 0:
            continue
        x = x[:, :usable]
        num_windows = usable // L
        rest_dims = x.shape[2:]
        x = x.reshape(num_windows, L, *rest_dims)

        emb = model.encode(x)  # (num_windows, L, D)
        emb = emb.reshape(1, num_windows * L, D)

        votes[:, offset : offset + usable] += emb
        counts[:, offset : offset + usable] += 1

    safe_counts = counts.clamp(min=1).unsqueeze(-1)
    averaged = votes / safe_counts  # (1, N, D)

    return averaged[0].cpu().float().numpy()


def _hf_repo_id(model_name: str) -> str:
    """Return the HF dataset repo id for a model's embeddings."""
    return f"{HF_EMBEDDINGS_ORG}/{model_name}-embeddings"


def _upload_to_hf(out_dir: Path, model_name: str, dataset_name: str) -> None:
    """Upload embeddings for one dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = _hf_repo_id(model_name)

    # Ensure repo exists
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    # Upload entire directory in a single commit
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=dataset_name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {dataset_name} embeddings",
    )

    print(f"Uploaded to {repo_id}/{dataset_name}/")


def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    model_name: str,
    dataset_name: str,
    L: int,
    device: str = "cpu",
    overwrite: bool = False,
    upload: bool = False,
    cache_dir: Optional[str] = None,
) -> Path:
    """Extract and cache contextualized embeddings for all subjects.

    For each subject in the dataset, extracts per-epoch embeddings using
    sliding-window encoding (same as voting evaluation) and saves them
    to disk.  Optionally uploads to HuggingFace Hub.

    Args:
        model: Model with ``encode(x) -> (B, L, D)`` method.
        dataset: A ``BasePhysioDataset`` instance.
        model_name: Identifier for cache directory (e.g. ``"seqsleepnet-phan"``).
        dataset_name: Dataset name for cache (e.g. ``"sleepedf"``).
        L: Sequence length the model was trained with.
        device: Device string (``"cpu"`` or ``"cuda:0"``).
        overwrite: If True, re-extract even if cached.
        upload: If True, upload all embeddings to HuggingFace Hub.
        cache_dir: Override cache root directory.

    Returns:
        Path to the embeddings directory.
    """
    out_dir = _cache_root(cache_dir) / model_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    model = model.to(dev).eval()

    cache = ChannelCache(cache_dir)

    subjects = dataset.get_subjects()
    n_extracted = 0
    embedding_dim = None

    for subj_idx, subject_id in enumerate(subjects):
        subj_dir = out_dir / subject_id
        emb_path = subj_dir / "embeddings.npy"

        if emb_path.exists() and not overwrite:
            if embedding_dim is None:
                existing = np.load(str(emb_path), mmap_mode="r")
                embedding_dim = existing.shape[1]
            n_extracted += 1
            continue

        # Load full recording for this subject
        try:
            spec = next(s for s in dataset._subjects if s.subject_id == subject_id)
            n_epochs = dataset._n_epochs[subject_id]
            item = dataset._build_item(spec, 0, n_epochs)

            # Stack channels: (n_epochs, C, ...)
            ch_tensors = [item["signals"][ch] for ch in item["channel_order"]]
            signals = torch.stack(ch_tensors, dim=1)  # (n_epochs, C, ...)
            signals = signals.unsqueeze(0)  # (1, n_epochs, C, ...)

            labels = item["labels"].numpy()

            # Extract embeddings
            embeddings = _extract_subject_sliding(model, signals, L, dev)
        except Exception as e:
            print(f"  [SKIP] {subject_id}: {e}")
            continue

        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]

        # Save
        subj_dir.mkdir(parents=True, exist_ok=True)
        dtype_name = recommended_dtype()

        cache.atomic_save_array(
            emb_path,
            cast_to_cache_dtype(embeddings, dtype_name),
            meta={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "subject_id": subject_id,
                "embedding_dim": int(embeddings.shape[1]),
                "n_epochs": int(embeddings.shape[0]),
            },
        )

        lbl_path = subj_dir / "labels.npy"
        cache.atomic_save_array(
            lbl_path,
            labels.astype(np.int16),
            meta={
                "subject_id": subject_id,
                "n_epochs": int(labels.shape[0]),
            },
        )

        n_extracted += 1
        print(
            f"  [{n_extracted}/{len(subjects)}] {subject_id}: "
            f"{embeddings.shape[0]} epochs, dim={embeddings.shape[1]}"
        )

    # Save global metadata
    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "embedding_dim": int(embedding_dim) if embedding_dim else 0,
        "n_subjects": len(subjects),
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {n_extracted} subjects to {out_dir}")

    if upload:
        _upload_to_hf(out_dir, model_name, dataset_name)

    return out_dir


def load_embeddings(
    model_name: str,
    dataset_name: str,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """Load cached embeddings, downloading from HuggingFace if needed.

    Checks the local cache first. If embeddings are not found locally,
    downloads them from ``4rooms/{model_name}-embeddings`` on HuggingFace
    Hub (one dataset repo per model).

    Args:
        model_name: Model identifier (e.g. ``"seqsleepnet-phan"``).
        dataset_name: Dataset name (e.g. ``"sleepedf"``).
        cache_dir: Override cache root directory.
        verbose: If True, print embedding metadata and linear probe
            results (if available).

    Returns:
        Path to the local embeddings directory containing per-subject
        subdirectories with ``embeddings.npy`` and ``labels.npy``.

    Example::

        from physioex.models import load_embeddings

        path = load_embeddings("seqsleepnet-phan", "sleepedf", verbose=True)
        # path / "SC4001E0" / "embeddings.npy"  ->  (n_epochs, 128)
    """
    out_dir = _cache_root(cache_dir) / model_name / dataset_name
    meta_path = out_dir / "metadata.json"

    # Check if already cached locally
    if meta_path.exists():
        if verbose:
            _print_embedding_info(out_dir)
        return out_dir

    # Download from HuggingFace
    from huggingface_hub import snapshot_download

    repo_id = _hf_repo_id(model_name)
    model_cache = _cache_root(cache_dir) / model_name

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{dataset_name}/**",
            local_dir=str(model_cache),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"No embeddings for {model_name}/{dataset_name} on HuggingFace "
            f"({repo_id}). Extract them first with "
            f"extract_embeddings(). Error: {e}"
        )

    print(f"Downloaded {model_name}/{dataset_name} from {repo_id}")

    if verbose:
        _print_embedding_info(out_dir)

    return out_dir


def _print_embedding_info(emb_dir: Path) -> None:
    """Print metadata and linear probe results for a cached embedding dir."""
    meta_path = emb_dir / "metadata.json"
    probe_path = emb_dir / "linear_probe_results.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n{'=' * 60}")
        print(f"Embeddings: {meta.get('model_name', '?')}/{meta.get('dataset_name', '?')}")
        print(f"  Subjects:      {meta.get('n_subjects', '?')}")
        print(f"  Embedding dim: {meta.get('embedding_dim', '?')}")
        print(f"  Cache:         {emb_dir}")

    if probe_path.exists():
        with open(probe_path) as f:
            probe = json.load(f)

        pooled = probe.get("pooled", {})
        mean_std = probe.get("mean_std", {})
        class_names = probe.get("class_names", [])

        print(f"\nLinear probe ({probe.get('n_folds', '?')}-fold subject-wise CV):")

        # Pooled metrics
        print(
            f"  Pooled:   ACC={pooled.get('accuracy', 0):.4f}  "
            f"MF1={pooled.get('macro_f1', 0):.4f}  "
            f"\u03ba={pooled.get('kappa', 0):.4f}"
        )

        # Mean +/- std
        acc_ms = mean_std.get("accuracy", {})
        mf1_ms = mean_std.get("macro_f1", {})
        kap_ms = mean_std.get("kappa", {})
        print(
            f"  Mean\u00b1SD: ACC={acc_ms.get('mean', 0):.4f}\u00b1{acc_ms.get('std', 0):.4f}  "
            f"MF1={mf1_ms.get('mean', 0):.4f}\u00b1{mf1_ms.get('std', 0):.4f}  "
            f"\u03ba={kap_ms.get('mean', 0):.4f}\u00b1{kap_ms.get('std', 0):.4f}"
        )

        # Per-class F1
        pcf1 = pooled.get("per_class_f1", {})
        if pcf1:
            pcf1_str = "  ".join(
                f"{name}={pcf1.get(name, 0):.2f}" for name in class_names
            )
            print(f"  Per-class F1: {pcf1_str}")

        # Support
        support = pooled.get("support", {})
        if support:
            sup_str = "  ".join(
                f"{name}={int(support.get(name, 0))}" for name in class_names
            )
            print(f"  Support:      {sup_str}")

        print(f"{'=' * 60}")
    else:
        print("  Linear probe: not yet computed")
        print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

SLEEP_CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]


def _load_npy_as_float32(npy_path: Path) -> np.ndarray:
    """Load a .npy file as float32, handling bfloat16 (|V2) transparently."""
    arr = np.load(str(npy_path))
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.astype(np.float32)
    if arr.dtype == np.int16 or arr.dtype == np.int64:
        return arr
    # bfloat16 stored as void (|V2): convert via raw bytes → torch → numpy
    if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
        flat = torch.frombuffer(arr.tobytes(), dtype=torch.bfloat16)
        return flat.float().numpy().reshape(arr.shape)
    # Fallback: try direct cast
    return arr.astype(np.float32)


def linear_probe(
    model_name: str,
    dataset_name: str,
    n_folds: int = 5,
    max_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    device: str = "cpu",
    upload: bool = False,
    cache_dir: Optional[str] = None,
    save_predictions: bool = False,
) -> dict:
    """5-fold subject-wise cross-validated linear probe on cached embeddings.

    Loads per-subject embeddings and labels from the local cache, splits
    subjects into *n_folds* folds, trains a ``nn.Linear`` classifier per
    fold, and reports pooled and per-fold metrics.

    Metrics reported (standard in sleep staging literature):

    * **Accuracy (ACC)** — overall classification accuracy.
    * **Macro F1 (MF1)** — unweighted mean of per-class F1; handles class
      imbalance.
    * **Cohen's Kappa (κ)** — chance-corrected agreement; the standard
      metric in sleep medicine.
    * **Per-class F1** — shows per-stage performance (N1 is typically
      the hardest).

    Results are saved to ``linear_probe_results.json`` in the embeddings
    cache directory.  If *upload* is True, the entire directory (embeddings
    + results) is uploaded to HuggingFace Hub.

    Args:
        model_name: Model identifier matching cache directory.
        dataset_name: Dataset identifier matching cache directory.
        n_folds: Number of cross-validation folds (default 5).
        max_epochs: Training epochs per fold (default 100).
        lr: Learning rate (default 1e-3).
        weight_decay: L2 regularization (default 1e-4).
        batch_size: Training batch size (default 512).
        device: Device string (``"cpu"`` or ``"cuda:0"``).
        upload: If True, upload results to HuggingFace Hub.
        cache_dir: Override cache root directory.
        save_predictions: If True, save per-subject softmax probabilities
            and labels to ``linear_probe_predictions.json``. Each entry
            contains subject_id, fold index, proba (n_epochs, n_classes),
            and labels (n_epochs,).

    Returns:
        Results dict with ``per_fold``, ``pooled``, and ``mean_std`` keys.
    """
    from physioex.train.metrics import (
        accuracy_score as _accuracy_score,
        f1_score as _f1_score,
        cohen_kappa_score as _cohen_kappa_score,
        confusion_matrix as _confusion_matrix,
        _per_class_f1,
    )

    emb_dir = _cache_root(cache_dir) / model_name / dataset_name
    if not emb_dir.exists():
        raise FileNotFoundError(f"No embeddings found at {emb_dir}")

    # ------------------------------------------------------------------
    # 1. Load all subject embeddings & labels
    # ------------------------------------------------------------------
    subjects = []
    for subj_dir in sorted(emb_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        emb_path = subj_dir / "embeddings.npy"
        lbl_path = subj_dir / "labels.npy"
        if emb_path.exists() and lbl_path.exists():
            subjects.append(
                {
                    "id": subj_dir.name,
                    "embeddings": _load_npy_as_float32(emb_path),
                    "labels": np.load(str(lbl_path)).astype(np.int64),
                }
            )

    if not subjects:
        raise ValueError(f"No subject data found in {emb_dir}")

    all_labels = np.concatenate([s["labels"] for s in subjects])
    valid_labels = all_labels[all_labels >= 0]
    n_classes = int(valid_labels.max()) + 1
    class_names = (
        SLEEP_CLASS_NAMES if n_classes == 5 else [str(i) for i in range(n_classes)]
    )

    # ------------------------------------------------------------------
    # 2. Create deterministic fold assignments
    # ------------------------------------------------------------------
    n_subjects = len(subjects)
    rng = np.random.RandomState(42)
    indices = np.arange(n_subjects)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    dev = torch.device(device)

    per_fold = []
    all_logits_list = []
    all_targets_list = []
    subject_predictions = []  # for save_predictions

    print(
        f"\nLinear probe: {n_subjects} subjects, {n_folds}-fold CV, "
        f"{n_classes} classes ({', '.join(class_names)})"
    )

    # ------------------------------------------------------------------
    # 3. Train & evaluate per fold
    # ------------------------------------------------------------------
    for fold_idx in range(n_folds):
        test_set = set(folds[fold_idx].tolist())
        train_idx = [i for i in range(n_subjects) if i not in test_set]
        test_idx = folds[fold_idx].tolist()

        # Concatenate per-subject arrays
        train_embs = np.concatenate([subjects[i]["embeddings"] for i in train_idx])
        train_lbls = np.concatenate([subjects[i]["labels"] for i in train_idx])
        test_embs = np.concatenate([subjects[i]["embeddings"] for i in test_idx])
        test_lbls = np.concatenate([subjects[i]["labels"] for i in test_idx])

        # Drop unscored epochs (label == -1)
        train_mask = train_lbls >= 0
        test_mask = test_lbls >= 0
        train_embs, train_lbls = train_embs[train_mask], train_lbls[train_mask]
        test_embs, test_lbls = test_embs[test_mask], test_lbls[test_mask]

        # Standard scaling: fit on train, transform both
        mean = train_embs.mean(axis=0)
        std = train_embs.std(axis=0) + 1e-8
        train_embs = (train_embs - mean) / std
        test_embs = (test_embs - mean) / std

        X_train = torch.from_numpy(train_embs)
        y_train = torch.from_numpy(train_lbls)
        X_test = torch.from_numpy(test_embs)
        y_test = torch.from_numpy(test_lbls)

        # Linear classifier
        D = X_train.shape[1]
        probe_model = torch.nn.Linear(D, n_classes).to(dev)
        optimizer = torch.optim.Adam(
            probe_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # --- Train ---
        probe_model.train()
        for _epoch in range(max_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = loss_fn(probe_model(xb), yb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # --- Evaluate ---
        probe_model.eval()
        test_logits_chunks = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                chunk = X_test[i : i + batch_size].to(dev)
                test_logits_chunks.append(probe_model(chunk).cpu())
        test_logits = torch.cat(test_logits_chunks, dim=0)

        # Collect per-subject predictions if requested
        if save_predictions:
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

        # Metrics (functions expect logits and do argmax internally)
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

    # ------------------------------------------------------------------
    # 4. Pooled metrics (all folds concatenated)
    # ------------------------------------------------------------------
    all_logits = torch.cat(all_logits_list)
    all_targets = torch.cat(all_targets_list)

    pooled_acc = _accuracy_score(all_logits, all_targets, ignore_index=None)
    pooled_mf1 = _f1_score(all_logits, all_targets, ignore_index=None)
    pooled_kappa = _cohen_kappa_score(all_logits, all_targets, ignore_index=None)
    pooled_cm = _confusion_matrix(all_logits, all_targets, ignore_index=None)
    pooled_preds = all_logits.argmax(dim=-1)
    pooled_pcf1, pooled_support = _per_class_f1(
        pooled_preds, all_targets, n_classes
    )

    # Mean +/- std across folds
    fold_accs = [f["accuracy"] for f in per_fold]
    fold_mf1s = [f["macro_f1"] for f in per_fold]
    fold_kappas = [f["kappa"] for f in per_fold]

    results = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "n_folds": n_folds,
        "n_classes": n_classes,
        "class_names": class_names,
        "n_subjects": n_subjects,
        "probe_config": {
            "max_epochs": max_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
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

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6. Save & upload
    # ------------------------------------------------------------------
    results_path = emb_dir / "linear_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    if save_predictions:
        predictions_path = emb_dir / "linear_probe_predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(subject_predictions, f)
        print(f"  Predictions saved to {predictions_path}")

    if upload:
        _upload_to_hf(emb_dir, model_name, dataset_name)

    return results
