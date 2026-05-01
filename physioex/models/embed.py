"""Generic embedding extraction for any PhysioEx model with encode().

Extracts contextualized per-epoch embeddings using sliding-window voting
(same approach as Trainer.voting_evaluate), caches them to disk, and
provides a loader for downstream use.

Works with any ``nn.Module`` that has an ``encode(x) -> (B, L, D)``
method: SeqSleepNet, TinySleepNet, SleepTransformer, LSeqSleepNet, etc.

Cache layout::

    {cache_root}/embeddings/{model_name}/{dataset_name}/
        metadata.json
        {subject_id}/
            embeddings.npy       (n_epochs, D)
            embeddings.meta.json
            labels.npy           (n_epochs,)
            labels.meta.json

HuggingFace repo: ``4rooms/physioex-embeddings`` mirrors the same structure.
``load_embeddings()`` downloads from HF automatically if not in local cache.
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

HF_EMBEDDINGS_REPO = "4rooms/physioex-embeddings"


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


def _upload_to_hf(out_dir: Path, model_name: str, dataset_name: str) -> None:
    """Upload all embeddings in out_dir to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{model_name}/{dataset_name}"

    # Upload metadata.json
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=f"{prefix}/metadata.json",
            repo_id=HF_EMBEDDINGS_REPO,
            repo_type="model",
        )

    # Upload per-subject files
    for subj_dir in sorted(out_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject_id = subj_dir.name
        for fname in [
            "embeddings.npy",
            "embeddings.meta.json",
            "labels.npy",
            "labels.meta.json",
        ]:
            fpath = subj_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"{prefix}/{subject_id}/{fname}",
                    repo_id=HF_EMBEDDINGS_REPO,
                    repo_type="model",
                )

    print(f"Uploaded embeddings to {HF_EMBEDDINGS_REPO}/{prefix}/")


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
) -> Path:
    """Load cached embeddings, downloading from HuggingFace if needed.

    Checks the local cache first. If embeddings are not found locally,
    downloads them from ``4rooms/physioex-embeddings`` on HuggingFace Hub.

    Args:
        model_name: Model identifier (e.g. ``"seqsleepnet-phan"``).
        dataset_name: Dataset name (e.g. ``"sleepedf"``).
        cache_dir: Override cache root directory.

    Returns:
        Path to the local embeddings directory containing per-subject
        subdirectories with ``embeddings.npy`` and ``labels.npy``.

    Example::

        from physioex.models import load_embeddings

        path = load_embeddings("seqsleepnet-phan", "sleepedf")
        # path / "SC4001E0" / "embeddings.npy"  ->  (n_epochs, 128)
    """
    out_dir = _cache_root(cache_dir) / model_name / dataset_name
    meta_path = out_dir / "metadata.json"

    # Check if already cached locally
    if meta_path.exists():
        return out_dir

    # Download from HuggingFace
    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"{model_name}/{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata first
    try:
        hf_hub_download(
            repo_id=HF_EMBEDDINGS_REPO,
            filename=f"{prefix}/metadata.json",
            local_dir=str(_cache_root(cache_dir)),
            repo_type="model",
        )
    except Exception as e:
        raise FileNotFoundError(
            f"No embeddings for {model_name}/{dataset_name} on HuggingFace "
            f"({HF_EMBEDDINGS_REPO}). Extract them first with "
            f"extract_embeddings(). Error: {e}"
        )

    # Read metadata to discover subjects
    with open(meta_path) as f:
        metadata = json.load(f)

    # List all files under the prefix on HF
    api = HfApi()
    all_files = api.list_repo_files(repo_id=HF_EMBEDDINGS_REPO, repo_type="model")
    prefix_files = [f for f in all_files if f.startswith(prefix + "/")]

    # Download all embedding files
    n_downloaded = 0
    for hf_path in prefix_files:
        if hf_path.endswith("/metadata.json"):
            continue  # already downloaded
        hf_hub_download(
            repo_id=HF_EMBEDDINGS_REPO,
            filename=hf_path,
            local_dir=str(_cache_root(cache_dir)),
            repo_type="model",
        )
        n_downloaded += 1

    print(
        f"Downloaded {n_downloaded} files for {model_name}/{dataset_name} "
        f"from {HF_EMBEDDINGS_REPO}"
    )

    return out_dir
