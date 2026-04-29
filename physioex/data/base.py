"""BasePhysioDataset: abstract lazy-loading PyTorch Dataset with on-disk caching."""
from __future__ import annotations

import logging
import os
import random
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from physioex.data.pipeline import PreprocessingPipeline, CompiledPipeline
from physioex.data.steps import Identity
from physioex.data.cache import (
    ChannelCache,
    recommended_dtype,
    cast_to_cache_dtype,
    _encode_physical,
)
from physioex.data.readers.edf import (
    EDFHeader,
    ResolvedChannel,
    ChannelNotAvailableError,
    DEFAULT_PREFERENCES,
    probe_edf_header,
    resolve_channels,
    read_channel,
)
from physioex.data.presets import get_preset
from physioex.data.events import (
    SleepEvent,
    map_events_to_epochs,
    events_to_dicts,
    dicts_to_events,
)

logger = logging.getLogger("physioex.data")


# AASM 5-class sleep staging: W=0, N1=1, N2=2, N3=3, REM=4. -1 marks
# unscored / corrupt / missing epochs. Any label value outside this set
# is considered invalid and will be coerced to -1 by ``_sanitize_labels``.
AASM_VALID_LABELS = (-1, 0, 1, 2, 3, 4)


@dataclass
class SubjectSpec:
    """Metadata describing how to locate a single subject's raw data.

    Returned by ``BasePhysioDataset._list_subjects()``.  Lightweight -- must
    not itself load signal data.
    """

    subject_id: str
    edf_path: Path
    label_path: Optional[Path] = None
    external_meta: Dict[str, Any] = field(default_factory=dict)


class BasePhysioDataset(Dataset):
    """Abstract base class for lazy-loading EDF-based sleep datasets.

    Subclasses must implement:
      - ``_list_subjects()`` -> list[SubjectSpec]
      - ``_read_subject_labels(spec)`` -> np.ndarray of int per-epoch labels

    Subclasses typically override:
      - ``DATASET_NAME`` (class attribute, used as cache directory name)
      - ``CHANNEL_PREFERENCES`` (class attribute, per-modality preference lists)

    Optional overrides:
      - ``_read_edf_header(spec)`` to inject external metadata
      - ``_read_subject_channel(spec, resolved)`` for non-EDF formats
    """

    # Subclasses must set this
    DATASET_NAME: str = "base"

    # Default preferences (can be overridden per-subclass)
    CHANNEL_PREFERENCES: Dict[str, List] = DEFAULT_PREFERENCES

    # Epoch length in seconds (30 for humans, 4 for mouse)
    DEFAULT_EPOCH_LENGTH_SEC: float = 30.0

    def __init__(
        self,
        root: str,
        channels: Optional[List[Union[str, Dict]]] = None,
        pipelines: Union[
            PreprocessingPipeline, str, Dict[str, Union[PreprocessingPipeline, str]]
        ] = "raw",
        sequence_length: int = 21,
        subset: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True,
        epoch_length_sec: Optional[float] = None,
        label_transform: Optional[Callable] = None,
        skip_corrupt: bool = True,
        skip_subjects_without_channels: bool = True,
        stage_map: Optional[Dict] = None,
        trim_excess_wake: bool = True,
    ):
        super().__init__()
        self.root = str(root) if root is not None else None
        # channels=None means "load ALL channels found across the dataset".
        # Resolved after header probing in _init_headers_and_resolution.
        self._channels_request = channels  # raw user input (may be None)
        self.channels = list(channels) if channels is not None else []
        self.sequence_length = int(sequence_length)
        self.subset = subset
        self.epoch_length_sec = (
            float(epoch_length_sec)
            if epoch_length_sec is not None
            else self.DEFAULT_EPOCH_LENGTH_SEC
        )
        self.label_transform = label_transform
        self.skip_corrupt = bool(skip_corrupt)
        self.skip_subjects_without_channels = bool(skip_subjects_without_channels)
        self.stage_map = stage_map
        # If True (default), excess pre-/post-sleep wake epochs are marked as -1
        # so they are ignored by the loss without being dropped from the index.
        self.trim_excess_wake = bool(trim_excess_wake)
        # When False, all disk I/O for headers, labels, and signals is bypassed.
        # Data is computed on-the-fly and returned in-memory without reading or
        # writing to the cache directory.
        self.cache_enabled = bool(cache_enabled)
        # In-memory cache for label arrays AFTER runtime transforms (trim, etc.).
        # Keeps disk cache stable while allowing runtime flags to change without
        # invalidation.
        self._labels_cache: Dict[str, np.ndarray] = {}
        self._events_cache: Dict[str, List[SleepEvent]] = {}

        self.cache = ChannelCache(cache_dir)

        # Normalize pipelines into either a single pipeline (applied to all
        # channels) or a dict[request_key -> pipeline].  Request keys can be
        # modality names or physical channel strings; resolution is attempted
        # first by modality/request, then by physical name after channel
        # resolution.
        self._pipelines_spec = self._normalize_pipelines(pipelines)

        # Discover subjects
        self._subjects: List[SubjectSpec] = self._list_subjects()
        if subset is not None:
            self._subjects = self._filter_subset(self._subjects, subset)

        if not self._subjects:
            warnings.warn(
                f"[{self.DATASET_NAME}] No subjects found under root={self.root!r} "
                f"(subset={subset!r})"
            )

        # Probe all headers (uses disk cache) and pre-resolve channels per
        # subject.
        self._headers: Dict[str, EDFHeader] = {}
        self._resolved: Dict[str, List[ResolvedChannel]] = {}
        self._compiled_pipelines: Dict[Tuple[str, float, str], CompiledPipeline] = {}
        # Per-subject label arrays (loaded lazily)
        self._n_epochs: Dict[str, int] = {}

        self._init_headers_and_resolution()

        # Build flat index mapping for sequence mode
        self._build_index()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _list_subjects(self) -> List[SubjectSpec]:
        """Return the subject manifest.  Lightweight: must NOT load signal data."""
        raise NotImplementedError

    @abstractmethod
    def _read_subject_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Return per-epoch label array (int, -1 for unscored)."""
        raise NotImplementedError

    def _read_edf_header(self, spec: SubjectSpec) -> EDFHeader:
        """Default EDF header probe.  Subclasses may override to add external
        metadata."""
        return probe_edf_header(spec.edf_path)

    def _read_subject_channel(
        self, spec: SubjectSpec, resolved: ResolvedChannel
    ) -> Tuple[np.ndarray, float]:
        """Default: open EDF via pyedflib and read the channel.  Override for
        non-EDF formats."""
        import pyedflib

        with pyedflib.EdfReader(str(spec.edf_path)) as f:
            return read_channel(f, resolved)

    def _filter_subset(
        self, subjects: List[SubjectSpec], subset: str
    ) -> List[SubjectSpec]:
        """Default: return subjects unchanged.  Dataset subclasses can override
        for subsetting."""
        return subjects

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        if self.sequence_length > 0:
            return self._get_sequence_item(idx)
        else:
            return self._get_recording_item(idx)

    def split(
        self, fold: int = 0
    ) -> Tuple[np.ndarray, List[Tuple[int, str]], List[Tuple[int, str]]]:
        """Return train/valid/test indices compatible with
        ``physioex/data/dataset.py:PhysioExDataset.split``.

        Training: flat integer indices into the sequence-mode ``__getitem__``.
        Valid/Test: list of ``(dataset_idx, subject_id)`` tuples matching the
        legacy API.  The convention is: one dataset here -> ``dataset_idx=0``
        for all subjects.
        """
        train_ids, valid_ids, test_ids = self.get_splits(fold=fold)
        train_flat = self._subject_ids_to_flat_indices(train_ids)
        valid_subjects = [(0, sid) for sid in valid_ids]
        test_subjects = [(0, sid) for sid in test_ids]
        return np.asarray(train_flat, dtype=np.int64), valid_subjects, test_subjects

    def get_splits(self, fold: int = 0) -> Tuple[List[str], List[str], List[str]]:
        """Return ``(train, valid, test)`` subject_id lists.

        Default: random 70/15/15 split with seed ``42 + fold``.
        Subclasses can override to provide custom benchmark folds.
        """
        rng = random.Random(42 + int(fold))
        ids = [s.subject_id for s in self._subjects]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(0.70 * n)
        n_valid = int(0.15 * n)
        train = ids[:n_train]
        valid = ids[n_train : n_train + n_valid]
        test = ids[n_train + n_valid :]
        return train, valid, test

    def probe(self, subject: Union[int, str, None] = None) -> Dict[str, Any]:
        """Return discovered channels + metadata for a subject (int index,
        string id, or ``None`` = first)."""
        if subject is None:
            spec = self._subjects[0]
        elif isinstance(subject, int):
            spec = self._subjects[subject]
        else:
            matches = [s for s in self._subjects if s.subject_id == subject]
            if not matches:
                raise KeyError(f"Subject {subject!r} not found")
            spec = matches[0]
        header = self._headers.get(spec.subject_id)
        if header is None:
            header = self._load_or_probe_header(spec)
        return {
            "subject_id": spec.subject_id,
            "available_channels": list(header.available_channels),
            "channel_fs": dict(header.channel_fs),
            "channel_units": dict(header.channel_units),
            "patient_meta": dict(header.patient_meta),
            "duration_sec": header.duration_sec,
        }

    def available_channels(self) -> Dict[str, int]:
        """Return all unique channel names found across the dataset with their
        occurrence count. Useful for discovering what channels to request.

        Returns:
            dict mapping ``physical_channel_name`` -> number of subjects
            that have it.
        """
        from collections import Counter

        counts: Counter = Counter()
        for spec in self._subjects:
            header = self._headers.get(spec.subject_id)
            if header is not None:
                counts.update(header.available_channels)
        return dict(counts.most_common())

    def get_n_subjects(self) -> int:
        """Return the number of subjects in this dataset."""
        return len(self._subjects)

    # ------------------------------------------------------------------
    # Internal: label sanitization + safe slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_labels(labels: np.ndarray) -> np.ndarray:
        """Coerce a per-epoch label array to AASM 5-class + ``-1``.

        Any value not in ``AASM_VALID_LABELS`` (i.e. outside ``{-1, 0..4}``)
        is replaced with ``-1``. This is the contract consumers rely on:
        every epoch in the dataset either has a valid AASM stage or is
        marked as unscored (``-1``) -- never a foreign value.
        """
        labels = np.asarray(labels, dtype=np.int16)
        if labels.size == 0:
            return labels
        valid_mask = np.isin(labels, AASM_VALID_LABELS)
        if valid_mask.all():
            return labels
        n_invalid = int((~valid_mask).sum())
        logger.warning(
            f"_sanitize_labels: {n_invalid}/{labels.size} labels out of "
            f"AASM 5-class range; coercing to -1"
        )
        return np.where(valid_mask, labels, -1).astype(np.int16)

    @staticmethod
    def _safe_slice(arr, start: int, end: int, fill_value):
        """Slice ``arr[start:end]``; pad with ``fill_value`` if the source is
        shorter than the requested range.

        Guarantees the returned array has exactly ``end - start`` entries on
        the leading axis. Used to keep labels and signals aligned even when
        an EDF is truncated or a label array is shorter than expected.
        """
        length = int(end - start)
        if length <= 0:
            raise ValueError(f"_safe_slice requires end > start, got {start}..{end}")
        arr = np.asarray(arr)
        # Clamp start and end to valid array bounds
        actual_start = max(0, min(start, arr.shape[0]))
        actual_end = max(actual_start, min(end, arr.shape[0]))
        taken = arr[actual_start:actual_end]
        missing = length - taken.shape[0]
        if missing <= 0:
            return taken[:length]  # extra safety: truncate if somehow too long
        # Infer trailing shape from array (not from taken, which may be empty)
        trailing = tuple(arr.shape[1:]) if arr.ndim > 1 else ()
        pad_shape = (missing,) + trailing
        pad = np.full(pad_shape, fill_value, dtype=arr.dtype)
        if taken.shape[0] > 0:
            return np.concatenate([taken, pad], axis=0)
        return pad

    @staticmethod
    def _trim_excess_wake(
        labels: np.ndarray,
        wake_value: int = 0,
        keep_minutes: float = 30.0,
        epoch_sec: float = 30.0,
    ) -> np.ndarray:
        """Mark excess pre-/post-sleep wake epochs as ``-1``.

        Keeps up to *keep_minutes* of contiguous wake before the first
        non-wake epoch and after the last non-wake epoch.  Everything
        beyond that window is set to ``-1`` so that
        ``CrossEntropyLoss(ignore_index=-1)`` ignores it during training.

        Epochs are **not** dropped -- they stay in the index so that
        sequence indices remain stable.

        Args:
            labels: 1D integer array with values in ``{-1, 0, 1, 2, 3, 4}``.
            wake_value: the label integer for Wake (default ``0``).
            keep_minutes: minutes of wake to keep at each boundary (default 30).
            epoch_sec: duration of a single epoch in seconds (default 30).
        Returns:
            A new array (copy) with excess wake positions set to ``-1``.
        """
        labels = np.asarray(labels, dtype=np.int16).copy()
        n = labels.shape[0]
        if n == 0:
            return labels

        keep_epochs = int(keep_minutes * 60 / epoch_sec)

        # Find the first and last non-wake, non-ignored epoch
        non_wake = np.where((labels != wake_value) & (labels >= 0))[0]
        if len(non_wake) == 0:
            return labels  # all wake or all ignored -- nothing to trim

        first_sleep = non_wake[0]
        last_sleep = non_wake[-1]

        # Mark leading wake beyond the keep window
        trim_start = max(0, first_sleep - keep_epochs)
        if trim_start > 0:
            labels[:trim_start] = -1

        # Mark trailing wake beyond the keep window
        trim_end = min(n, last_sleep + 1 + keep_epochs)
        if trim_end < n:
            labels[trim_end:] = -1

        return labels

    def _get_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Return labels for a subject with all runtime transforms applied.

        Order: disk-load (sanitized) -> ``_trim_excess_wake`` if enabled.
        Result is cached in memory for the lifetime of the dataset instance.
        """
        if spec.subject_id in self._labels_cache:
            return self._labels_cache[spec.subject_id]
        labels = self._load_or_compute_labels(spec)
        if self.trim_excess_wake:
            labels = self._trim_excess_wake(labels)
        self._labels_cache[spec.subject_id] = labels
        return labels

    # ------------------------------------------------------------------
    # Events: per-epoch temporal event metadata
    # ------------------------------------------------------------------

    def _read_subject_events(self, spec: SubjectSpec) -> List[SleepEvent]:
        """Return temporal events for a subject. Override in subclasses that have event data."""
        return []

    def _load_or_compute_events(self, spec: SubjectSpec) -> List[SleepEvent]:
        """Load events from cache or parse from source."""
        if spec.subject_id in self._events_cache:
            return self._events_cache[spec.subject_id]
        if self.cache_enabled:
            ep = self.cache.events_path(self.DATASET_NAME, spec.subject_id)
            if ep.exists():
                try:
                    import json

                    with open(ep) as f:
                        cached = json.load(f)
                    events = dicts_to_events(cached)
                    self._events_cache[spec.subject_id] = events
                    return events
                except Exception:
                    pass
        events = self._read_subject_events(spec)
        if self.cache_enabled and events:
            try:
                self.cache.save_json(
                    self.cache.events_path(self.DATASET_NAME, spec.subject_id),
                    events_to_dicts(events),
                )
            except Exception as exc:
                logger.warning(
                    "Could not cache events for %s: %s", spec.subject_id, exc
                )
        self._events_cache[spec.subject_id] = events
        return events

    # ------------------------------------------------------------------
    # Public metadata API
    # ------------------------------------------------------------------

    def get_subjects(self) -> List[str]:
        """Return list of all subject IDs."""
        return [s.subject_id for s in self._subjects]

    def get_subject_metadata(self, subject_id: str) -> Dict[str, Any]:
        """Return subject-level metadata without loading signals."""
        matches = [s for s in self._subjects if s.subject_id == subject_id]
        if not matches:
            raise KeyError(f"Subject {subject_id!r} not found")
        spec = matches[0]
        header = self._headers.get(spec.subject_id)
        meta: Dict[str, Any] = {"id": spec.subject_id, "dataset": self.DATASET_NAME}
        if header is not None:
            meta.update({k: v for k, v in header.patient_meta.items() if v is not None})
        meta.update(spec.external_meta)
        return meta

    def get_all_subject_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata for all subjects keyed by subject_id."""
        return {
            s.subject_id: self.get_subject_metadata(s.subject_id)
            for s in self._subjects
        }

    def get_subject_events(self, subject_id: str) -> List[SleepEvent]:
        """Return all temporal events for a subject without loading signals."""
        matches = [s for s in self._subjects if s.subject_id == subject_id]
        if not matches:
            raise KeyError(f"Subject {subject_id!r} not found")
        return self._load_or_compute_events(matches[0])

    # ------------------------------------------------------------------
    # Internal: pipelines
    # ------------------------------------------------------------------

    def _normalize_pipelines(self, p) -> Dict[str, PreprocessingPipeline]:
        """Normalize the user's ``pipelines`` arg into a
        ``dict[request_key -> PreprocessingPipeline]``.

        Accepts:
          - ``PreprocessingPipeline`` -- stored under ``'__default__'``
          - ``str`` preset name -- resolved via ``get_preset``; if the preset
            returns a dict (bundle preset like ``"time_domain"``), that dict
            is used directly, otherwise stored under ``'__default__'``
          - ``dict`` -- each value resolved to a ``PreprocessingPipeline``
        """
        # Resolve string presets first -- may expand to a dict bundle
        if isinstance(p, str):
            p = get_preset(p)

        if isinstance(p, dict):
            return {k: self._to_pipeline(v) for k, v in p.items()}
        return {"__default__": self._to_pipeline(p)}

    def _to_pipeline(self, p) -> PreprocessingPipeline:
        if isinstance(p, PreprocessingPipeline):
            return p
        if isinstance(p, str):
            resolved = get_preset(p)
            if isinstance(resolved, PreprocessingPipeline):
                return resolved
            raise TypeError(
                f"preset {p!r} returned a {type(resolved).__name__}; "
                f"dict presets cannot be nested inside a dict entry"
            )
        raise TypeError(
            f"pipelines entry must be a PreprocessingPipeline or preset name; "
            f"got {type(p)}"
        )

    def _pipeline_for_channel(self, resolved: ResolvedChannel) -> PreprocessingPipeline:
        """Pick the right pipeline for a resolved channel.

        Priority: physical name > modality name > ``__default__``.
        """
        physical_key = _encode_physical(resolved.physical)
        if physical_key in self._pipelines_spec:
            return self._pipelines_spec[physical_key]
        # Also try without encoding (for user-friendly names like "C4-M2")
        if (
            isinstance(resolved.physical, str)
            and resolved.physical in self._pipelines_spec
        ):
            return self._pipelines_spec[resolved.physical]
        if resolved.modality and resolved.modality in self._pipelines_spec:
            return self._pipelines_spec[resolved.modality]
        if "__default__" in self._pipelines_spec:
            return self._pipelines_spec["__default__"]
        # Fall back to identity if no explicit default was provided
        return PreprocessingPipeline([Identity()])

    # ------------------------------------------------------------------
    # Internal: headers, resolution, cache compilation
    # ------------------------------------------------------------------

    def _load_or_probe_header(self, spec: SubjectSpec) -> EDFHeader:
        """Return the EDF header for a subject, using on-disk cache when
        valid.  When ``self.cache_enabled`` is False, the header is always
        probed from the source file and no disk I/O is performed."""
        if self.cache_enabled:
            hpath = self.cache.header_path(self.DATASET_NAME, spec.subject_id)
            src_mtime = None
            try:
                src_mtime = spec.edf_path.stat().st_mtime
            except Exception:
                pass

            if hpath.exists():
                cached = self.cache.load_json(hpath)
                if cached is not None:
                    try:
                        if (
                            src_mtime is None
                            or abs(float(cached.get("source_mtime", 0)) - src_mtime)
                            < 1e-3
                        ):
                            return EDFHeader.from_dict(cached)
                    except Exception:
                        pass

        header = self._read_edf_header(spec)

        if self.cache_enabled:
            try:
                self.cache.save_json(hpath, header.to_dict())
            except Exception as exc:
                logger.warning(f"Could not cache header for {spec.subject_id}: {exc}")
        return header

    def _init_headers_and_resolution(self) -> None:
        """Probe all subjects and resolve channels.

        If ``channels`` was ``None`` at construction time, ALL unique
        physical channel names across every subject are collected and
        used as explicit channel requests (each is a specific-name
        request, not a modality shortcut). This makes every channel
        unconditionally available; subjects missing a channel get it
        zero-filled at access time.

        Subjects are **never dropped** due to missing channels.
        Only genuinely corrupt EDFs (header unreadable) are skipped
        when ``skip_corrupt=True``.
        """
        # Phase 1: probe all headers
        kept = []
        for spec in self._subjects:
            try:
                header = self._load_or_probe_header(spec)
            except Exception as exc:
                if self.skip_corrupt:
                    warnings.warn(
                        f"[{self.DATASET_NAME}] Skipping corrupt subject "
                        f"{spec.subject_id}: {exc}"
                    )
                    continue
                raise
            self._headers[spec.subject_id] = header
            kept.append(spec)
        self._subjects = kept

        # Phase 1b: if channels=None ("all"), collect the union of every
        # physical channel across the dataset and use those as requests.
        if self._channels_request is None:
            from collections import Counter

            counts: Counter = Counter()
            for spec in self._subjects:
                h = self._headers[spec.subject_id]
                counts.update(h.available_channels)
            # Order by frequency (most common first) for determinism
            self.channels = [ch for ch, _ in counts.most_common()]
            logger.info(
                f"[{self.DATASET_NAME}] channels=None -> auto-discovered "
                f"{len(self.channels)} channels across {len(self._subjects)} "
                f"subjects"
            )

        # Phase 2: resolve channels per subject
        for spec in self._subjects:
            header = self._headers[spec.subject_id]

            resolved = resolve_channels(
                self.channels,
                header.available_channels,
                header.channel_fs,
                preferences=self.CHANNEL_PREFERENCES,
                allow_missing=True,
            )

            # Log missing channels (informational, not fatal)
            missing = [self.channels[i] for i, rc in enumerate(resolved) if rc is None]
            if missing:
                logger.info(
                    f"[{self.DATASET_NAME}] {spec.subject_id}: channels "
                    f"{missing} not available (will be zero-filled)"
                )

            self._resolved[spec.subject_id] = resolved

    # ------------------------------------------------------------------
    # Internal: labels + cache
    # ------------------------------------------------------------------

    def _load_or_compute_labels(self, spec: SubjectSpec) -> np.ndarray:
        """Load labels from cache if present, otherwise parse and cache.

        When ``self.cache_enabled`` is False, always computes labels from the
        source and returns them in-memory without any disk read or write.
        """
        if self.cache_enabled:
            lp = self.cache.labels_path(self.DATASET_NAME, spec.subject_id)
            lmeta = self.cache.labels_meta_path(self.DATASET_NAME, spec.subject_id)

            if lp.exists() and lmeta.exists():
                try:
                    memmap, meta = self.cache.load_memmap(lp)
                    return np.asarray(memmap).astype(np.int64, copy=False)
                except Exception as exc:
                    logger.warning(
                        f"Corrupt label cache for {spec.subject_id}: {exc}; "
                        f"rebuilding"
                    )

        labels = self._read_subject_labels(spec)
        labels = self._sanitize_labels(np.asarray(labels, dtype=np.int16))

        if self.cache_enabled:
            meta = {
                "dataset": self.DATASET_NAME,
                "subject_id": spec.subject_id,
                "n_epochs": int(labels.shape[0]),
                "sanitized_to": "AASM_5_class",
            }
            try:
                self.cache.atomic_save_array(lp, labels, meta)
            except Exception as exc:
                logger.warning(f"Could not cache labels for {spec.subject_id}: {exc}")

        return labels.astype(np.int64, copy=False)

    def _load_or_compute_channel(
        self, spec: SubjectSpec, resolved: ResolvedChannel
    ) -> np.ndarray:
        """Load a cached preprocessed channel, or compute + cache it.

        When ``self.cache_enabled`` is False, always computes the signal from
        the source and returns it in-memory (float32) without any disk I/O.
        """
        pipeline = self._pipeline_for_channel(resolved)
        pipeline_hash = pipeline.hash()

        if self.cache_enabled:
            signal_path = self.cache.signal_path(
                self.DATASET_NAME,
                spec.subject_id,
                resolved.physical,
                pipeline_hash,
            )

            # If cache file exists, try to use it.
            if signal_path.exists():
                try:
                    memmap, meta = self.cache.load_memmap(signal_path)
                    return memmap
                except Exception as exc:
                    logger.warning(
                        f"Corrupt signal cache {signal_path}: {exc}; recomputing"
                    )

        # Compile pipeline for this subject's channel fs
        fs_in = resolved.fs_in
        compile_key = (pipeline_hash, fs_in, spec.subject_id)
        compiled = self._compiled_pipelines.get(compile_key)
        if compiled is None:
            compiled = pipeline.compile(fs_in)
            self._compiled_pipelines[compile_key] = compiled

        # Read raw signal
        raw, fs = self._read_subject_channel(spec, resolved)

        # Epoch the signal and run through the pipeline
        processed = self._epoch_and_run_pipeline(raw, fs, compiled, resolved)

        if self.cache_enabled:
            # Cast to cache dtype and atomic save
            dtype_name = recommended_dtype()
            casted = cast_to_cache_dtype(processed, dtype_name)
            meta = {
                "dataset": self.DATASET_NAME,
                "subject_id": spec.subject_id,
                "channel_request": repr(resolved.request),
                "channel_physical": repr(resolved.physical),
                "modality": resolved.modality,
                "pipeline_hash": pipeline_hash,
                "pipeline_spec": pipeline.spec(),
                "fs_in": fs_in,
                "fs_out": compiled.fs_out,
            }
            try:
                self.cache.atomic_save_array(signal_path, casted, meta)
            except Exception as exc:
                logger.warning(f"Could not cache signal {signal_path}: {exc}")

            # Reopen via memmap for consistency
            try:
                memmap, _ = self.cache.load_memmap(signal_path)
                return memmap
            except Exception:
                # Fallback: return in-memory array
                return casted
        else:
            # Return in-memory array directly (float32, not cast to cache dtype)
            return processed.astype(np.float32)

    def _epoch_and_run_pipeline(
        self,
        raw: np.ndarray,
        fs_in: float,
        compiled: CompiledPipeline,
        resolved: ResolvedChannel,
    ) -> np.ndarray:
        """Epoch-segment the raw signal, then run the compiled pipeline per
        epoch.

        Strategy per compiled step:
          - Non-zero ``fs_out``: apply to full 1D signal; update working fs.
          - Zero ``fs_out`` (domain-changing, e.g. spectrogram): epoch the
            signal first (if not already epoched), then apply.

        After all steps, if the signal is still 1D it is reshaped to
        ``(n_epochs, samples_per_epoch)``.
        """
        x = np.asarray(raw, dtype=np.float32)
        fs = fs_in
        epoched = False

        for step in compiled.steps:
            fs_out_step = step.fs_out
            if fs_out_step == 0:
                # Domain-changing step (e.g., spectrogram) -- epoch first if
                # not already
                if not epoched:
                    sps = int(round(self.epoch_length_sec * fs))
                    n_full = (x.shape[-1] // sps) * sps
                    x = x[..., :n_full].reshape(-1, sps)
                    epoched = True
                x = step.apply(x)
            else:
                x = step.apply(x)
                fs = fs_out_step

        if not epoched:
            # No domain-changing step; epoch at the end
            sps = int(round(self.epoch_length_sec * fs))
            n_full = (x.shape[-1] // sps) * sps
            x = x[..., :n_full].reshape(-1, sps)

        return np.ascontiguousarray(x)

    # ------------------------------------------------------------------
    # Internal: index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Populate ``n_epochs`` per subject and compute ``self._length``.

        The authoritative epoch count is ``n_epochs = max(labels_len,
        signal_epochs_from_header)``: we index every physical epoch of
        the recording. Gaps between the two are handled at access time
        by ``_safe_slice`` -- missing label positions return ``-1`` and
        missing signal positions return zeros, so nothing is ever
        dropped.
        """
        # Load labels for all subjects (cached after first call; runtime
        # transforms like trim_excess_wake applied via _get_labels).
        for spec in self._subjects:
            try:
                labels = self._get_labels(spec)
            except Exception as exc:
                if self.skip_corrupt:
                    warnings.warn(
                        f"[{self.DATASET_NAME}] Cannot read labels for "
                        f"{spec.subject_id}: {exc}; skipping"
                    )
                    continue
                raise

            header = self._headers.get(spec.subject_id)
            n_from_header = (
                int(header.duration_sec // self.epoch_length_sec)
                if header is not None
                else 0
            )
            n_from_labels = int(labels.shape[0])
            # Authoritative: largest consistent count. Takes every physical
            # epoch into account. _safe_slice fills gaps at access time.
            n_epochs = max(n_from_header, n_from_labels)

            if (
                n_from_labels != n_from_header
                and n_from_header > 0
                and n_from_labels > 0
            ):
                logger.info(
                    f"[{self.DATASET_NAME}] {spec.subject_id}: "
                    f"labels_len={n_from_labels} vs signal_epochs={n_from_header}; "
                    f"using n_epochs={n_epochs} (gaps padded with -1 / zeros)"
                )

            self._n_epochs[spec.subject_id] = n_epochs

        # Drop subjects with 0 epochs or too few for sequence_length
        kept = []
        for spec in self._subjects:
            n = self._n_epochs.get(spec.subject_id, 0)
            if n <= 0:
                continue
            if self.sequence_length > 0 and n < self.sequence_length:
                warnings.warn(
                    f"[{self.DATASET_NAME}] Subject {spec.subject_id} has "
                    f"{n} epochs < sequence_length={self.sequence_length}; "
                    f"skipping"
                )
                continue
            kept.append(spec)
        self._subjects = kept

        # Compute flat index
        self._subject_ranges: List[Tuple[str, int, int]] = []
        running = 0
        for spec in self._subjects:
            n = self._n_epochs[spec.subject_id]
            if self.sequence_length > 0:
                count = n - self.sequence_length + 1
            else:
                count = 1  # one recording per subject
            self._subject_ranges.append((spec.subject_id, running, running + count))
            running += count
        self._length = running

    def _find_subject_for_flat_idx(self, idx: int) -> Tuple[SubjectSpec, int]:
        """Map a flat index into ``(subject, local_offset_within_subject)``."""
        for sid, start, end in self._subject_ranges:
            if start <= idx < end:
                spec = next(s for s in self._subjects if s.subject_id == sid)
                return spec, idx - start
        raise IndexError(f"Flat index {idx} not in any subject range")

    def _subject_ids_to_flat_indices(self, ids: List[str]) -> List[int]:
        """Convert a list of ``subject_id`` values to all the flat training
        indices that belong to them."""
        id_set = set(ids)
        out: List[int] = []
        for sid, start, end in self._subject_ranges:
            if sid in id_set:
                out.extend(range(start, end))
        return out

    # ------------------------------------------------------------------
    # Internal: dict assembly
    # ------------------------------------------------------------------

    def _get_sequence_item(self, flat_idx: int) -> Dict[str, Any]:
        spec, epoch_start = self._find_subject_for_flat_idx(flat_idx)
        epoch_end = epoch_start + self.sequence_length
        return self._build_item(spec, epoch_start, epoch_end)

    def _get_recording_item(self, flat_idx: int) -> Dict[str, Any]:
        # flat_idx is directly a subject index in recording mode
        spec = self._subjects[flat_idx]
        n = self._n_epochs[spec.subject_id]
        return self._build_item(spec, 0, n)

    def _build_item(
        self,
        spec: SubjectSpec,
        epoch_start: int,
        epoch_end: int,
    ) -> Dict[str, Any]:
        resolved_list = self._resolved[spec.subject_id]
        header = self._headers[spec.subject_id]

        # Labels -- safe slice pads with -1 if the label array is shorter
        # than the requested range (subject has more physical epochs than
        # scored ones). After _sanitize_labels every value is in AASM 5-class
        # range or -1. _get_labels additionally applies trim_excess_wake.
        labels_full = self._get_labels(spec)
        labels_slice = self._safe_slice(
            labels_full,
            epoch_start,
            epoch_end,
            fill_value=-1,
        )
        if self.label_transform is not None:
            labels_slice = self.label_transform(labels_slice)
        # Re-sanitize after any user-provided transform (cheap) to enforce
        # the AASM contract at the tensor boundary.
        labels_slice = self._sanitize_labels(np.asarray(labels_slice, dtype=np.int16))
        labels_tensor = torch.from_numpy(np.asarray(labels_slice, dtype=np.int64))

        # Signals (per channel) -- safe slice pads with 0.0 if the cached
        # signal is shorter than the requested range (e.g. truncated EDF).
        # After slicing, all channels are normalized to the SAME epoch count
        # (the maximum across channels and labels). Shorter channels get
        # zero-padded; labels for padded positions are set to -1.
        requested_len = int(epoch_end - epoch_start)
        signals_raw: Dict[str, np.ndarray] = {}
        channel_info: Dict[str, Dict[str, Any]] = {}
        channel_order: List[str] = []

        # Reference shape: needed to zero-fill channels that are missing
        # for this subject. We get the shape from the FIRST resolved channel,
        # or fall back to a sensible default.
        _ref_shape: Optional[Tuple[int, ...]] = None

        for i, rc in enumerate(resolved_list):
            req = self.channels[i] if i < len(self.channels) else f"ch{i}"

            if rc is None:
                # Channel not available for this subject — zero-fill later
                key = f"__missing_{req}_{i}"
                channel_order.append(key)
                channel_info[key] = {
                    "request": req,
                    "physical": None,
                    "modality": req if isinstance(req, str) else None,
                    "fs_in": 0,
                    "fs_out": 0,
                    "is_differential": False,
                    "unit": "",
                    "pipeline_hash": "",
                    "available": False,
                }
                continue

            key = _encode_physical(rc.physical)
            channel_order.append(key)

            signal_data = self._load_or_compute_channel(spec, rc)
            sig_slice = self._safe_slice(
                signal_data,
                epoch_start,
                epoch_end,
                fill_value=0.0,
            )
            signals_raw[key] = np.asarray(sig_slice, dtype=np.float32)
            if _ref_shape is None:
                _ref_shape = tuple(signals_raw[key].shape)

            pipeline = self._pipeline_for_channel(rc)
            compiled = pipeline.compile(rc.fs_in)
            channel_info[key] = {
                "request": (
                    rc.request
                    if not isinstance(rc.request, tuple)
                    else list(rc.request)
                ),
                "physical": (
                    rc.physical
                    if not isinstance(rc.physical, tuple)
                    else list(rc.physical)
                ),
                "modality": rc.modality,
                "fs_in": rc.fs_in,
                "fs_out": compiled.fs_out,
                "is_differential": rc.is_differential,
                "unit": header.channel_units.get(
                    header.available_channels[rc._indices[0]] if rc._indices else "",
                    "",
                ),
                "pipeline_hash": pipeline.hash(),
                "available": True,
            }

        # Fill in missing channels with zeros (same shape as resolved ones)
        if _ref_shape is None:
            _ref_shape = (requested_len,)
        for key in channel_order:
            if key not in signals_raw:
                signals_raw[key] = np.zeros(_ref_shape, dtype=np.float32)

        # ---- Cross-channel length normalization ----
        # Different channels may produce different epoch counts (e.g. due to
        # different native fs + resampling rounding). Normalize ALL channels
        # and labels to the SAME length (max across all). Zero-pad shorter
        # signals; set labels to -1 for padded positions.
        actual_lengths = [s.shape[0] for s in signals_raw.values()]
        actual_lengths.append(labels_tensor.shape[0])
        max_len = max(actual_lengths) if actual_lengths else requested_len

        signals: Dict[str, torch.Tensor] = {}
        for key in channel_order:
            arr = signals_raw[key]
            if arr.shape[0] < max_len:
                trailing = tuple(arr.shape[1:])
                pad_arr = np.zeros(
                    (max_len - arr.shape[0],) + trailing, dtype=arr.dtype
                )
                arr = np.concatenate([arr, pad_arr], axis=0)
            elif arr.shape[0] > max_len:
                arr = arr[:max_len]
            signals[key] = torch.from_numpy(arr)

        if labels_tensor.shape[0] < max_len:
            pad_lbl = torch.full(
                (max_len - labels_tensor.shape[0],), -1, dtype=labels_tensor.dtype
            )
            labels_tensor = torch.cat([labels_tensor, pad_lbl])
        elif labels_tensor.shape[0] > max_len:
            labels_tensor = labels_tensor[:max_len]

        subject_meta = {
            "id": spec.subject_id,
            "dataset": self.DATASET_NAME,
            "dataset_idx": 0,
            **spec.external_meta,
            **{k: v for k, v in header.patient_meta.items() if v is not None},
        }

        # Events -- map to epochs, slice to match window
        all_events = self._load_or_compute_events(spec)
        if all_events:
            n_total = self._n_epochs.get(spec.subject_id, max_len)
            epoch_events_full = map_events_to_epochs(
                all_events, n_total, self.epoch_length_sec
            )
            events_slice = epoch_events_full[epoch_start : epoch_start + max_len]
            while len(events_slice) < max_len:
                events_slice.append([])
        else:
            events_slice = [[] for _ in range(max_len)]

        return {
            "signals": signals,
            "channel_order": channel_order,
            "channel_info": channel_info,
            "labels": labels_tensor,
            "subject": subject_meta,
            "epoch_indices": torch.arange(
                epoch_start, epoch_start + max_len, dtype=torch.long
            ),
            "recording_length": max_len,
            "events": events_slice,
        }
