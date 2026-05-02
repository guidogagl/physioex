"""Chambon et al. (2018) deep learning architecture for sleep staging.

Reference: "A deep learning architecture for temporal sleep stage
classification using multivariate and multimodal time series"
(IEEE TNSRE 2018, arXiv:1707.03321).

Architecture (from the paper):
  Per-epoch: spatial_conv -> 2x(conv + ReLU + maxpool) -> flatten
  Sequence:  concatenate all L epoch features -> dropout -> linear -> softmax

The model classifies the **central epoch** of a sequence of L epochs,
using temporal context from surrounding epochs.  This matches the
paper's approach where features from multiple epochs are concatenated
before the final classification layer.

Uses braindecode's SleepStagerChambon2018 as the epoch encoder (official
implementation matching the paper's architecture).
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from braindecode.models import SleepStagerChambon2018
from torch import nn
from torchinfo import summary

from physioex.train.trainer import Trainer
from physioex.data.dataset import PhysioExDataset


class Chambon2018Net(nn.Module):
    """Chambon et al. (2018) sleep staging network.

    Encodes each epoch independently via braindecode's SleepStagerChambon2018,
    then concatenates all L epoch features and classifies the **central epoch**.

    Input:  (B, L, C, T) — batch, sequence_length, channels, time_samples
    Output: (B, 1, n_classes) — logit for the central epoch only

    This matches the paper: "the temporal context of each 30s window of data"
    is exploited by concatenating features from surrounding epochs.

    Args:
        n_classes: Number of sleep stages (default 5).
        in_channels: Number of input channels (default 1).
        sf: Sampling frequency in Hz (default 100).
        n_times: Samples per epoch (default 3000 = 30s @ 100Hz).
        dropout: Dropout before the classifier (default 0.25, as in paper).
    """

    def __init__(
        self,
        n_classes: int = 5,
        in_channels: int = 1,
        sf: int = 100,
        n_times: int = 3000,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.epoch_encoder = SleepStagerChambon2018(
            n_chans=in_channels,
            sfreq=sf,
            n_outputs=n_classes,
            n_times=n_times,
            return_feats=True,
        )

        # The paper concatenates features from ALL L epochs and classifies
        # with a single linear layer.  We don't know L at init time, so
        # we use a lazy linear that infers input size on first forward.
        self.drop = nn.Dropout(dropout)
        self.clf = nn.LazyLinear(n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each epoch independently into a feature vector.

        Args:
            x: (B, L, C, T) input tensor.

        Returns:
            (B, L, D) epoch-level feature embeddings.
        """
        batch_size, seqlen, nchan, nsamp = x.size()
        x = x.reshape(-1, nchan, nsamp)
        x = self.epoch_encoder(x)
        x = x.reshape(batch_size, seqlen, -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode all epochs, concatenate, classify central.

        Args:
            x: (B, L, C, T) input tensor.

        Returns:
            (B, 1, n_classes) logits for the central epoch.
        """
        # Encode each epoch independently
        feats = self.encode(x)  # (B, L, D)

        # Concatenate all L epoch features into one vector
        batch_size = feats.size(0)
        concat = feats.reshape(batch_size, -1)  # (B, L*D)

        # Classify the central epoch
        out = self.drop(concat)
        out = self.clf(out)  # (B, n_classes)

        # Return as (B, 1, n_classes) for consistency with other models
        return out.unsqueeze(1)


@torch.no_grad()
def _extract_subject_direct(
    model: Chambon2018Net,
    signals: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract per-epoch embeddings without sliding window.

    Chambon2018's epoch encoder is independent (no inter-epoch context),
    so each epoch can be encoded directly.  This is ~L times faster than
    the generic sliding-window approach.

    Args:
        model: A Chambon2018Net with encode().
        signals: (1, N, C, T) full-night signal tensor.
        device: CUDA or CPU device.
        batch_size: Epochs per forward pass.

    Returns:
        (N, D) numpy array of per-epoch embeddings.
    """
    signals = signals.squeeze(0).to(device)  # (N, C, T)
    N = signals.shape[0]

    embeddings = []
    for i in range(0, N, batch_size):
        chunk = signals[i : i + batch_size].unsqueeze(0)  # (1, chunk, C, T)
        # encode() expects (B, L, C, T) → returns (B, L, D)
        # We use B=1, L=chunk_size
        emb = model.encode(chunk)  # (1, chunk, D)
        embeddings.append(emb.squeeze(0).cpu().float())

    return torch.cat(embeddings, dim=0).numpy()  # (N, D)


def extract_embeddings(
    model: Chambon2018Net,
    dataset,
    model_name: str,
    dataset_name: str,
    device: str = "cpu",
    overwrite: bool = False,
    upload: bool = False,
    cache_dir: Optional[str] = None,
) -> Path:
    """Extract and cache per-epoch embeddings for Chambon2018.

    Functionally identical to ``physioex.models.embed.extract_embeddings``
    but uses direct per-epoch encoding instead of sliding-window voting,
    since Chambon2018's epoch encoder has no inter-epoch context.

    Args:
        model: A Chambon2018Net instance.
        dataset: A BasePhysioDataset instance.
        model_name: Identifier for cache directory.
        dataset_name: Dataset name for cache.
        device: Device string.
        overwrite: If True, re-extract even if cached.
        upload: If True, upload to HuggingFace Hub.
        cache_dir: Override cache root directory.

    Returns:
        Path to the embeddings directory.
    """
    from physioex.data.cache import ChannelCache, recommended_dtype, cast_to_cache_dtype
    from physioex.models.embed import _cache_root, _upload_to_hf

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

        try:
            spec = next(s for s in dataset._subjects if s.subject_id == subject_id)
            n_epochs = dataset._n_epochs[subject_id]
            item = dataset._build_item(spec, 0, n_epochs)

            ch_tensors = [item["signals"][ch] for ch in item["channel_order"]]
            signals = torch.stack(ch_tensors, dim=1)  # (n_epochs, C, ...)
            signals = signals.unsqueeze(0)  # (1, n_epochs, C, ...)

            labels = item["labels"].numpy()

            embeddings = _extract_subject_direct(model, signals, dev)
        except Exception as e:
            print(f"  [SKIP] {subject_id}: {e}")
            continue

        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]

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


if __name__ == "__main__":

    dataset = PhysioExDataset(datasets=["sleepedf"])

    model = Chambon2018Net(
        in_channels=dataset.get_num_channels(),
    )

    print("Chambon2018Net summary:")
    summary(model, (32, 21, dataset.get_num_channels(), 3000))

    print("\nTraining Chambon2018Net...")
    model = Trainer.train(
        model=model,
        dataset=dataset,
        max_epochs=20,
        lr=1e-3,
    )

    print("\nEvaluating Chambon2018Net...")
    results = Trainer.evaluate(
        model=model,
        dataset=dataset,
    )
    print(results)
