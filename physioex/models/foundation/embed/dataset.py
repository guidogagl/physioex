"""EmbeddingDataset: reads cached foundation model embeddings as a PyTorch Dataset.

Same interface as BasePhysioDataset for split/fold/sequencing, but returns
pre-extracted embeddings (L, D) instead of raw signals (L, C, T).

Embeddings are read as read-only numpy memmaps — only the requested sequence
window is loaded into RAM (bfloat16 → float32 conversion on the slice only).

Usage::

    from physioex.models.foundation.embed.dataset import EmbeddingDataset

    ds = EmbeddingDataset("cbramod", "hmc", sequence_length=21)
    train_idx, val_subj, test_subj = ds.split(fold=0)
    item = ds[0]
    # item["embeddings"].shape == (21, 200)
    # item["labels"].shape == (21,)
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from physioex.data.cache import ChannelCache
from physioex.models.foundation._checkpoints import get_embeddings_dir

logger = logging.getLogger("physioex.foundation.embed")


def embedding_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for EmbeddingDataset items."""
    return {
        "embeddings": torch.stack([b["embeddings"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class EmbeddingDataset(Dataset):
    """Dataset that reads cached foundation model embeddings.

    Provides the same split/fold/sequencing interface as BasePhysioDataset,
    but each ``__getitem__`` returns a dict with ``embeddings`` (L, D)
    instead of multi-channel signals.

    Embeddings are read via numpy memmap — only the requested window is
    loaded into RAM, keeping memory usage proportional to batch size,
    not dataset size.

    Args:
        model_name: foundation model slug (e.g. "cbramod")
        dataset_name: dataset slug (e.g. "hmc")
        cache_root: root cache directory (default: ~/.cache/physioex)
        sequence_length: number of epochs per sample (default 21).
            Set to 0 for full-recording mode (one item per subject).
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        sequence_length: int = 21,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.sequence_length = sequence_length

        from physioex.models.foundation._checkpoints import _cache_root

        self._cache_root = str(_cache_root())
        self._cache = ChannelCache(self._cache_root)

        emb_dir = get_embeddings_dir(model_name, dataset_name)
        if not emb_dir.exists():
            raise FileNotFoundError(
                f"No cached embeddings found at {emb_dir}. "
                f"Run extract_embeddings('{model_name}', '{dataset_name}') first."
            )

        # Load global metadata
        meta_path = emb_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
        else:
            self.metadata = {}
        self.embedding_dim = self.metadata.get("embedding_dim", None)

        # Discover subjects: each subdirectory with embeddings.npy
        self._subjects: List[str] = []
        self._emb_memmaps: Dict[str, np.ndarray] = {}
        self._lab_memmaps: Dict[str, np.ndarray] = {}
        self._n_epochs: Dict[str, int] = {}

        for subj_dir in sorted(emb_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            emb_path = subj_dir / "embeddings.npy"
            lab_path = subj_dir / "labels.npy"
            if not emb_path.exists() or not lab_path.exists():
                continue

            subject_id = subj_dir.name
            emb_mm, emb_meta = self._cache.load_memmap(emb_path)
            lab_mm, lab_meta = self._cache.load_memmap(lab_path)

            self._subjects.append(subject_id)
            self._emb_memmaps[subject_id] = emb_mm
            self._lab_memmaps[subject_id] = lab_mm
            self._n_epochs[subject_id] = int(emb_mm.shape[0])

            if self.embedding_dim is None:
                self.embedding_dim = int(emb_mm.shape[1]) if emb_mm.ndim > 1 else 1

        if not self._subjects:
            raise FileNotFoundError(f"No valid subject embeddings found in {emb_dir}")

        logger.info(
            f"[EmbeddingDataset] {model_name}/{dataset_name}: "
            f"{len(self._subjects)} subjects, embedding_dim={self.embedding_dim}"
        )

        # Build flat index (same logic as BasePhysioDataset._build_index)
        self._subject_ranges: List[Tuple[str, int, int]] = []
        running = 0
        for sid in self._subjects:
            n = self._n_epochs[sid]
            if self.sequence_length > 0:
                count = max(0, n - self.sequence_length + 1)
            else:
                count = 1  # recording mode: one item per subject
            if count == 0:
                continue
            self._subject_ranges.append((sid, running, running + count))
            running += count
        self._length = running

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        # Find subject + local offset
        subject_id, local_offset = self._find_subject(idx)
        emb_mm = self._emb_memmaps[subject_id]
        lab_mm = self._lab_memmaps[subject_id]

        if self.sequence_length > 0:
            start = local_offset
            end = start + self.sequence_length
        else:
            start = 0
            end = self._n_epochs[subject_id]

        # Slice from memmap — only these epochs are loaded into RAM
        emb_slice = np.array(emb_mm[start:end], dtype=np.float32)
        lab_slice = np.array(lab_mm[start:end], dtype=np.int64)

        return {
            "embeddings": torch.from_numpy(emb_slice),
            "labels": torch.from_numpy(lab_slice),
            "subject_id": subject_id,
        }

    def _find_subject(self, flat_idx: int) -> Tuple[str, int]:
        for sid, start, end in self._subject_ranges:
            if start <= flat_idx < end:
                return sid, flat_idx - start
        raise IndexError(f"Flat index {flat_idx} not in any subject range")

    # ── Split / fold logic ──────────────────────────────────────────

    def get_splits(self, fold: int = 0) -> Tuple[List[str], List[str], List[str]]:
        """Return (train, valid, test) subject_id lists.

        Uses the same random 70/15/15 split with seed 42+fold as
        BasePhysioDataset.get_splits(), ensuring identical partitions.
        """
        rng = random.Random(42 + int(fold))
        ids = list(self._subjects)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)
        return ids[:n_train], ids[n_train : n_train + n_valid], ids[n_train + n_valid :]

    def split(
        self, fold: int = 0
    ) -> Tuple[np.ndarray, List[Tuple[int, str]], List[Tuple[int, str]]]:
        """Return (train_indices, valid_subjects, test_subjects).

        Compatible with Trainer.build_dataloaders().
        """
        train_ids, valid_ids, test_ids = self.get_splits(fold)
        train_flat = self._subject_ids_to_flat_indices(train_ids)
        valid_subjects = [(0, sid) for sid in valid_ids]
        test_subjects = [(0, sid) for sid in test_ids]
        return np.asarray(train_flat, dtype=np.int64), valid_subjects, test_subjects

    def _subject_ids_to_flat_indices(self, ids: List[str]) -> List[int]:
        id_set = set(ids)
        out: List[int] = []
        for sid, start, end in self._subject_ranges:
            if sid in id_set:
                out.extend(range(start, end))
        return out
