"""MultiDataset: wraps multiple BasePhysioDataset instances into a single unified dataset."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from physioex.data.base import BasePhysioDataset

logger = logging.getLogger("physioex.data")


class MultiDataset(Dataset):
    """Wraps multiple BasePhysioDataset instances into a single unified dataset.

    Provides ConcatDataset-like flat indexing across all datasets, with:
    - Unified __getitem__ returning the same dict format as BasePhysioDataset
    - Cross-dataset split coordination via split(fold)
    - dataset_idx tracking in the returned dict's subject metadata
    - Proportional memmap cache distribution across constituent datasets
    """

    def __init__(
        self, datasets: List[BasePhysioDataset], memmap_cache_size: int = 1000
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("MultiDataset requires at least one dataset")
        self._datasets = list(datasets)
        
        # all the datasets should have the same sequence length
        assert all(ds.sequence_length == self._datasets[0].sequence_length for ds in self._datasets), "All datasets must have the same sequence length for MultiDataset"
        self.sequence_length = self._datasets[0].sequence_length
        
        # Build cumulative offset table for flat indexing.
        # Each entry is (dataset_index, cumulative_start, cumulative_end).
        self._offsets: List[Tuple[int, int, int]] = []
        running = 0
        for i, ds in enumerate(self._datasets):
            length = len(ds)
            self._offsets.append((i, running, running + length))
            running += length
        self._length = running

        # Distribute memmap cache proportionally across datasets
        if memmap_cache_size > 0:
            self._distribute_memmap_cache(memmap_cache_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        ds_idx, local_idx = self._resolve_index(idx)
        item = self._datasets[ds_idx][local_idx]

        # Inject / overwrite dataset_idx in the subject metadata dict so
        # downstream code can identify which dataset a sample came from.
        if "subject" in item and isinstance(item["subject"], dict):
            item["subject"]["dataset_idx"] = ds_idx
        return item

    def split(
        self, fold: int = 0
    ) -> Tuple[np.ndarray, List[Tuple[int, str]], List[Tuple[int, str]]]:
        """Return train/valid/test indices coordinated across all datasets.

        Training: flat integer indices into this MultiDataset's ``__getitem__``.
        Valid/Test: list of ``(dataset_idx, subject_id)`` tuples, where
        ``dataset_idx`` is the position of the originating dataset in the
        ``datasets`` list passed to the constructor.
        """
        all_train: List[int] = []
        all_valid: List[Tuple[int, str]] = []
        all_test: List[Tuple[int, str]] = []

        for ds_idx, offset_start, offset_end in self._offsets:
            ds = self._datasets[ds_idx]
            train_flat, valid_subjects, test_subjects = ds.split(fold=fold)

            # Shift the per-dataset flat training indices by the cumulative
            # offset so they point into this MultiDataset's flat index space.
            if len(train_flat) > 0:
                shifted = np.asarray(train_flat, dtype=np.int64) + offset_start
                all_train.extend(shifted.tolist())

            # Re-tag valid/test subject tuples with the multi-dataset index
            # (the per-dataset split always returns dataset_idx=0).
            for _unused_idx, sid in valid_subjects:
                all_valid.append((ds_idx, sid))
            for _unused_idx, sid in test_subjects:
                all_test.append((ds_idx, sid))

        return np.asarray(all_train, dtype=np.int64), all_valid, all_test

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def datasets(self) -> List[BasePhysioDataset]:
        """Return the wrapped dataset list (read-only reference)."""
        return list(self._datasets)

    def get_n_subjects(self) -> int:
        """Return total number of subjects across all datasets."""
        return sum(ds.get_n_subjects() for ds in self._datasets)

    def get_subjects(self) -> List[str]:
        """Return all subject IDs across all datasets.

        Note: IDs are not guaranteed to be unique across datasets. If
        disambiguation is needed, use ``get_subjects_with_dataset_idx``.
        """
        out: List[str] = []
        for ds in self._datasets:
            out.extend(ds.get_subjects())
        return out

    def get_subjects_with_dataset_idx(self) -> List[Tuple[int, str]]:
        """Return ``(dataset_idx, subject_id)`` pairs for every subject."""
        out: List[Tuple[int, str]] = []
        for i, ds in enumerate(self._datasets):
            for sid in ds.get_subjects():
                out.append((i, sid))
        return out

    def available_channels(self) -> Dict[str, int]:
        """Return union of available channels across all datasets."""
        from collections import Counter

        merged: Counter = Counter()
        for ds in self._datasets:
            merged.update(ds.available_channels())
        return dict(merged.most_common())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Map a flat index to ``(dataset_index, local_index)``."""
        for ds_idx, start, end in self._offsets:
            if start <= idx < end:
                return ds_idx, idx - start
        raise IndexError(f"Flat index {idx} not in any dataset range")

    def __repr__(self) -> str:
        parts = ", ".join(f"{ds.DATASET_NAME}({len(ds)})" for ds in self._datasets)
        return f"MultiDataset(total={self._length}, datasets=[{parts}])"

    # ------------------------------------------------------------------
    # Memmap cache management
    # ------------------------------------------------------------------

    def _distribute_memmap_cache(self, total_cache_size: int) -> None:
        """Distribute memmap cache proportionally across constituent datasets.

        Each dataset gets a share proportional to its number of subjects.
        This ensures that larger datasets get more cache while respecting
        the total cache budget.

        Args:
            total_cache_size: Total number of memmap files to cache across all datasets.
        """
        if total_cache_size <= 0:
            return

        total_subjects = sum(ds.get_n_subjects() for ds in self._datasets)
        if total_subjects == 0:
            logger.warning("MultiDataset: no subjects found, skipping cache distribution")
            return

        for i, ds in enumerate(self._datasets):
            n_subjects = ds.get_n_subjects()
            # Proportional allocation: ensure at least 1 for non-empty datasets
            if n_subjects > 0:
                share = max(1, int(n_subjects / total_subjects * total_cache_size))
                ds._memmap_cache_size = share
                logger.info(
                    f"MultiDataset: {ds.DATASET_NAME} (n={n_subjects}) "
                    f"allocated memmap_cache_size={share}"
                )

    def memmap_cache_stats(self) -> Dict[str, Any]:
        """Aggregate memmap cache statistics across all constituent datasets.

        Returns:
            Dict with aggregated stats (total_hits, total_misses, hit_rate, etc.)
            and per-dataset breakdown.
        """
        total_hits = 0
        total_misses = 0
        per_dataset = {}

        for i, ds in enumerate(self._datasets):
            if hasattr(ds, "memmap_cache_stats"):
                stats = ds.memmap_cache_stats()
                total_hits += stats["hits"]
                total_misses += stats["misses"]
                per_dataset[ds.DATASET_NAME] = stats

        total = total_hits + total_misses
        hit_rate = (total_hits / total * 100) if total > 0 else 0.0

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_requests": total,
            "hit_rate_percent": hit_rate,
            "per_dataset": per_dataset,
        }

    def close(self) -> None:
        """Close all memmap caches in constituent datasets."""
        for ds in self._datasets:
            if hasattr(ds, "close"):
                ds.close()

    def __del__(self) -> None:
        """Cleanup memmap caches on deletion."""
        try:
            self.close()
        except Exception:
            pass
