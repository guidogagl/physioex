"""Tsinalis et al. (2016) CNN for sleep staging from raw EEG.

Reference: "Automatic Sleep Stage Scoring with Single-Channel EEG
Using Convolutional Neural Networks" (arXiv:1610.01683).

Includes ``extract_embeddings()`` for efficient per-epoch embedding
extraction using a centered sliding window (L=5).

Architecture (Table 3 of the paper):
  Input: 5 concatenated 30s epochs (15000 samples at 100Hz)
  C1: 20 filters, kernel=200, stride=1, ReLU
  P1: max-pool kernel=20, stride=10
  S1: stack (reshape 20 channels into 2D image)
  C2: 400 filters, kernel=(20,30), stride=1, ReLU
  P2: max-pool kernel=(1,10), stride=(1,2)
  F1: 500 units, ReLU, dropout
  F2: 500 units, ReLU, dropout
  Output: 5-class softmax

PhysioEx integration:
  Input from dataset: (B, 5, 1, 3000) — 5 epochs × 1 channel × 3000 samples
  Internally reshaped to: (B, 1, 15000) — concatenated signal
  Output: (B, 1, n_classes) — prediction for the CENTRAL epoch only

This is compatible with the standard Trainer when paired with a
target_transform that extracts the central epoch label.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


class TsinalisCNN(nn.Module):
    """CNN for single-channel sleep staging (Tsinalis et al. 2016).

    Accepts the standard PhysioEx sequence format ``(B, L, C, T)`` with
    ``L=5, C=1, T=3000``, concatenates the 5 epochs into a single
    15000-sample signal, and predicts the sleep stage of the **central**
    (3rd) epoch.

    Args:
        n_classes: Number of output classes (default 5: W, N1, N2, N3, REM).
        sfreq:     Sampling frequency in Hz (default 100).
        n_filters_c1: Number of filters in first conv layer (default 20).
        n_filters_c2: Number of filters in second conv layer (default 400).
        fc_size:   Hidden units in FC layers (default 500).
        dropout:   Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_classes: int = 5,
        sfreq: int = 100,
        n_filters_c1: int = 20,
        n_filters_c2: int = 400,
        fc_size: int = 500,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Input length: 5 epochs × 30s × sfreq
        n_times = 5 * 30 * sfreq  # 15000

        # C1: Long temporal filters (2 seconds = sfreq*2 samples)
        c1_kernel = sfreq * 2  # 200 for 100Hz
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_filters_c1, kernel_size=c1_kernel, stride=1),
            nn.ReLU(inplace=True),
        )

        # P1: Max-pool
        self.pool1 = nn.MaxPool1d(kernel_size=20, stride=10)

        # Compute size after C1 + P1
        c1_out = n_times - c1_kernel + 1  # 15000 - 200 + 1 = 14801
        p1_out = (c1_out - 20) // 10 + 1  # (14801 - 20) // 10 + 1 = 1479

        # C2: Cross-filter convolution (2D)
        c2_kernel = (n_filters_c1, 30)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filters_c2, kernel_size=c2_kernel, stride=1),
            nn.ReLU(inplace=True),
        )

        # P2: Temporal pooling
        c2_time = p1_out - c2_kernel[1] + 1  # 1479 - 30 + 1 = 1450
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 2))
        p2_time = (c2_time - 10) // 2 + 1  # (1450 - 10) // 2 + 1 = 721

        flat_size = n_filters_c2 * 1 * p2_time

        # FC layers (paper: F1=500, F2=500)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size, n_classes),
        )

    def _features(self, x):
        """Extract feature vector before the classifier.

        Args:
            x: (B, 5, 1, 3000) or (B, 1, 15000).

        Returns:
            (B, fc_size) feature vector.
        """
        if x.dim() == 4:
            B = x.shape[0]
            x = x.reshape(B, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pool1(x)
        x = x.unsqueeze(1)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.flatten(1)

        # Pass through FC layers up to (but not including) the final linear
        # classifier[0]=Dropout, [1]=Linear, [2]=ReLU, [3]=Dropout,
        # [4]=Linear, [5]=ReLU, [6]=Linear(fc_size, n_classes)
        for layer in list(self.classifier)[:-1]:
            x = layer(x)
        return x  # (B, fc_size)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 5, 1, 3000) from PhysioEx dataset (5-epoch sequence),
               or (B, 1, 15000) pre-concatenated signal.

        Returns:
            (B, 1, n_classes) logits for the central epoch.
        """
        feats = self._features(x)  # (B, fc_size)
        out = list(self.classifier)[-1](feats)  # final Linear -> (B, n_classes)
        return out.unsqueeze(1)


@torch.no_grad()
def _extract_subject_centered(
    model: TsinalisCNN,
    signals: torch.Tensor,
    L: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract per-epoch embeddings using centered L-epoch windows.

    For each epoch, builds a window of L epochs centered on it (padding
    with zeros at the edges), and extracts the feature vector via
    ``model._features()``.  Each epoch gets exactly one embedding —
    no sliding-window voting needed.

    Args:
        model: A TsinalisCNN with ``_features()``.
        signals: (1, N, C, T) full-night signal tensor.
        L: Sequence length (5 for Tsinalis).
        device: CUDA or CPU device.
        batch_size: Windows per forward pass.

    Returns:
        (N, D) numpy array of per-epoch embeddings.
    """
    signals = signals.squeeze(0)  # (N, C, T)
    N, C, T = signals.shape
    half = L // 2

    # Pad beginning and end with zeros
    pad = torch.zeros(half, C, T, dtype=signals.dtype)
    padded = torch.cat([pad, signals, pad], dim=0)  # (N+2*half, C, T)

    embeddings = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        windows = []
        for j in range(i, end):
            # Window centered on epoch j (in padded coords: j+half)
            w = padded[j : j + L]  # (L, C, T)
            windows.append(w)
        batch = torch.stack(windows).to(device)  # (batch, L, C, T)
        feats = model._features(batch)  # (batch, D)
        embeddings.append(feats.cpu().float())

    return torch.cat(embeddings, dim=0).numpy()  # (N, D)


def extract_embeddings(
    model: TsinalisCNN,
    dataset,
    model_name: str,
    dataset_name: str,
    L: int = 5,
    device: str = "cpu",
    overwrite: bool = False,
    upload: bool = False,
    cache_dir: Optional[str] = None,
) -> Path:
    """Extract and cache per-epoch embeddings for Tsinalis.

    Uses centered windows: for each epoch, builds a window of L=5 epochs
    centered on it and extracts the CNN feature vector.  Functionally
    identical to ``physioex.models.embed.extract_embeddings`` but
    avoids redundant sliding-window voting.

    Args:
        model: A TsinalisCNN instance.
        dataset: A BasePhysioDataset instance.
        model_name: Identifier for cache directory.
        dataset_name: Dataset name for cache.
        L: Sequence length (default 5).
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
            signals = torch.stack(ch_tensors, dim=1).unsqueeze(0)  # (1, N, C, T)
            labels = item["labels"].numpy()

            embeddings = _extract_subject_centered(model, signals, L, dev)
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
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {n_extracted} subjects to {out_dir}")

    if upload:
        _upload_to_hf(out_dir, model_name, dataset_name)

    return out_dir


if __name__ == "__main__":
    model = TsinalisCNN(n_classes=5, sfreq=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TsinalisCNN: {n_params:,} parameters")

    # Test with PhysioEx format: (B, 5, 1, 3000)
    x = torch.randn(4, 5, 1, 3000)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (4, 1, 5)

    # Verify gradient flow
    y.sum().backward()
    print("Gradient flow: OK")
