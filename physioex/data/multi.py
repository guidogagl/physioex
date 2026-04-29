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
    """

    def __init__(self, datasets: List[BasePhysioDataset]) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("MultiDataset requires at least one dataset")
        self._datasets = list(datasets)

        # Build cumulative offset table for flat indexing.
        # Each entry is (dataset_index, cumulative_start, cumulative_end).
        self._offsets: List[Tuple[int, int, int]] = []
        running = 0
        for i, ds in enumerate(self._datasets):
            length = len(ds)
            self._offsets.append((i, running, running + length))
            running += length
        self._length = running

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
