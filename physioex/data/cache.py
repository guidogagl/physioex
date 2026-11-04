"""On-disk cache manager for preprocessed per-channel signals.

Cache key: (SCHEMA_VERSION, dataset, subject_id, physical_channel, pipeline_hash)

Layout (under cache_root, which defaults to $PHYSIOEX_CACHE_DIR or ~/.cache/physioex):

    {cache_root}/v1/{dataset}/
        headers/{subject_id}.json          # EDF header probe cache
        subjects/{subject_id}.json          # Optional subject metadata
        labels/{subject_id}.npy             # Raw labels, no pipeline applied
        labels/{subject_id}.meta.json
        signals/{subject_id}/{physical}/{pipeline_hash}/
            signal.npy                      # np.memmap data
            signal.meta.json                # sidecar with fs_out, shape, dtype, pipeline_spec

Multi-worker safety: atomic rename (no locks).
Dataset names with "/" are flattened to "__" in filesystem paths.
Differential channel pairs like ("C4", "M1") are encoded as "C4__M1".
"""
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Dict, Tuple

import numpy as np

# bfloat16 support via ml_dtypes (follows physioex/data/datareader.py convention)
try:
    import torch
    from ml_dtypes import bfloat16 as _bfloat16

    _BFLOAT16_SUPPORTED = (
        torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    )
except Exception:
    _bfloat16 = None
    _BFLOAT16_SUPPORTED = False


SCHEMA_VERSION = "v1"


DTYPE_MAP: Dict[str, object] = {
    "float16": np.float16,  # clean-channel caches v2/v3 (protosleepnet-v3 cachefill)
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}
if _bfloat16 is not None:
    DTYPE_MAP["bfloat16"] = _bfloat16


def _sanitize_name(name: str) -> str:
    """Flatten filesystem-unsafe chars. Slash -> double-underscore."""
    return str(name).replace("/", "__").replace("\\", "__")


def _encode_physical(physical) -> str:
    """Encode a physical channel name (string or tuple) into a fs-safe directory name."""
    if isinstance(physical, (tuple, list)):
        return "__".join(_sanitize_name(p) for p in physical)
    return _sanitize_name(physical)


def _dtype_name(dtype) -> str:
    """Stable string name for a numpy dtype (supports ml_dtypes.bfloat16)."""
    if hasattr(dtype, "name"):
        return dtype.name
    return str(dtype)


def _resolve_dtype(name: str):
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype name {name!r}")
    return DTYPE_MAP[name]


class ChannelCache:
    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, cache_root: Optional[str] = None):
        if cache_root is None:
            cache_root = os.environ.get(
                "PHYSIOEX_CACHE_DIR", os.path.expanduser("~/.cache/physioex")
            )
        self.cache_root = Path(cache_root)

    # ----- Path resolution -----

    def dataset_root(self, dataset: str) -> Path:
        return self.cache_root / self.SCHEMA_VERSION / _sanitize_name(dataset)

    def header_path(self, dataset: str, subject_id: str) -> Path:
        return (
            self.dataset_root(dataset)
            / "headers"
            / f"{_sanitize_name(subject_id)}.json"
        )

    def subject_meta_path(self, dataset: str, subject_id: str) -> Path:
        return (
            self.dataset_root(dataset)
            / "subjects"
            / f"{_sanitize_name(subject_id)}.json"
        )

    def labels_path(self, dataset: str, subject_id: str) -> Path:
        return (
            self.dataset_root(dataset) / "labels" / f"{_sanitize_name(subject_id)}.npy"
        )

    def labels_meta_path(self, dataset: str, subject_id: str) -> Path:
        return (
            self.dataset_root(dataset)
            / "labels"
            / f"{_sanitize_name(subject_id)}.meta.json"
        )

    def events_path(self, dataset: str, subject_id: str) -> Path:
        return (
            self.dataset_root(dataset) / "events" / f"{_sanitize_name(subject_id)}.json"
        )

    def signal_dir(
        self, dataset: str, subject_id: str, physical, pipeline_hash: str
    ) -> Path:
        return (
            self.dataset_root(dataset)
            / "signals"
            / _sanitize_name(subject_id)
            / _encode_physical(physical)
            / pipeline_hash
        )

    def signal_path(
        self, dataset: str, subject_id: str, physical, pipeline_hash: str
    ) -> Path:
        return (
            self.signal_dir(dataset, subject_id, physical, pipeline_hash) / "signal.npy"
        )

    def signal_meta_path(
        self, dataset: str, subject_id: str, physical, pipeline_hash: str
    ) -> Path:
        return (
            self.signal_dir(dataset, subject_id, physical, pipeline_hash)
            / "signal.meta.json"
        )

    # ----- Existence checks -----

    def exists(self, path: Path) -> bool:
        return Path(path).exists()

    # ----- Atomic writes -----

    def atomic_save_array(
        self, path: Union[str, Path], data: np.ndarray, meta: Dict
    ) -> None:
        """Write `data` to `path` atomically, plus {stem}.meta.json sidecar.

        Algorithm: write to a temp file in the same directory, flush, then
        os.replace (POSIX-atomic on same filesystem, safe on NFSv3+).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Data file
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".tmp.",
            suffix=f".{os.getpid()}",
        )
        try:
            os.close(fd)
            # Use np.save without pickle (raw .npy) -- simplest portable format.
            # np.save always appends .npy when the path lacks it, creating a second
            # file alongside the original (empty) mkstemp fd. We handle both cases.
            np.save(tmp_path, data, allow_pickle=False)
            npy_tmp = tmp_path + ".npy"
            if os.path.exists(npy_tmp):
                # np.save wrote to the .npy copy; remove the original empty stub
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                actual_tmp = npy_tmp
            else:
                actual_tmp = tmp_path
            os.replace(actual_tmp, path)
        except Exception:
            for candidate in (tmp_path, tmp_path + ".npy"):
                if os.path.exists(candidate):
                    try:
                        os.unlink(candidate)
                    except OSError:
                        pass
            raise

        # Sidecar metadata
        meta_path = path.parent / (path.stem + ".meta.json")
        meta_full = dict(meta)
        meta_full.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        meta_full.setdefault("shape", list(data.shape))
        meta_full.setdefault("dtype", _dtype_name(data.dtype))
        meta_full.setdefault("schema_version", self.SCHEMA_VERSION)
        # atomic: another process (DataLoader worker, concurrent job) may build the same subject
        # and read the sidecar as soon as the .npy exists; a plain open("w") exposes an empty file
        self.save_json(meta_path, meta_full)

    def save_json(self, path: Union[str, Path], obj: Dict) -> None:
        """Atomic JSON write."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".tmp.",
            suffix=f".{os.getpid()}",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    def load_json(self, path: Union[str, Path]) -> Optional[Dict]:
        path = Path(path)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # ----- Memmap reads -----

    def load_memmap(self, data_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Open a cached .npy as a read-only memmap. Returns (memmap, meta).

        For non-native numpy dtypes (e.g. ml_dtypes.bfloat16), np.load returns a
        void type (``|V2``). We detect this via the sidecar metadata and reconstruct
        the memmap with the correct dtype and shape, computing the .npy header
        offset from the file.
        """
        data_path = Path(data_path)
        meta_path = data_path.parent / (data_path.stem + ".meta.json")
        meta = self.load_json(meta_path)
        if meta is None:
            raise FileNotFoundError(
                f"Sidecar missing for {data_path}: expected {meta_path}"
            )

        dtype_name = meta.get("dtype", "float32")
        dtype = _resolve_dtype(dtype_name)
        shape = tuple(meta["shape"])

        # Header parsed on an unbuffered file, then np.memmap at the data offset. np.load(mmap_mode)
        # opens the file buffered: on GPFS the buffer is the file-system block (16 MiB), so every
        # cold open read ~30 MB to parse a ~128 byte header; with many workers and nights this
        # saturated the client (smoke 04.11). Non-native dtypes (bfloat16) take the dtype and shape
        # from the sidecar, because the .npy header stores them as void (|V2).
        with open(data_path, "rb", buffering=0) as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                hshape, fortran, hdtype = np.lib.format.read_array_header_1_0(f)
            else:
                hshape, fortran, hdtype = np.lib.format.read_array_header_2_0(f)
            offset = f.tell()
        if dtype_name in ("bfloat16",):
            mm = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        else:
            mm = np.memmap(data_path, dtype=hdtype, mode="r", offset=offset, shape=tuple(hshape),
                           order="F" if fortran else "C")
        return mm, meta

    # ----- Cache invalidation -----

    def clear(
        self,
        dataset: Optional[str] = None,
        subject: Optional[str] = None,
        pipeline_hash: Optional[str] = None,
    ) -> None:
        """Remove cached data. Omit args to clear broader scopes.

        - clear() -> removes entire cache_root / SCHEMA_VERSION
        - clear(dataset) -> removes that dataset's subtree
        - clear(dataset, subject) -> removes subject's subtree within dataset
        - clear(dataset, subject, pipeline_hash) -> removes that pipeline hash across all channels
        """
        import shutil

        if dataset is None:
            root = self.cache_root / self.SCHEMA_VERSION
            if root.exists():
                shutil.rmtree(root)
            return

        ds_root = self.dataset_root(dataset)
        if subject is None:
            if ds_root.exists():
                shutil.rmtree(ds_root)
            return

        subj_sig = ds_root / "signals" / _sanitize_name(subject)
        subj_hdr = ds_root / "headers" / f"{_sanitize_name(subject)}.json"
        subj_lbl_data = ds_root / "labels" / f"{_sanitize_name(subject)}.npy"
        subj_lbl_meta = ds_root / "labels" / f"{_sanitize_name(subject)}.meta.json"
        subj_meta = ds_root / "subjects" / f"{_sanitize_name(subject)}.json"

        if pipeline_hash is None:
            for p in [subj_sig]:
                if p.exists():
                    shutil.rmtree(p)
            for p in [subj_hdr, subj_lbl_data, subj_lbl_meta, subj_meta]:
                if p.exists():
                    p.unlink()
            return

        # Clear a specific pipeline across all channels for this subject
        if subj_sig.exists():
            for ch_dir in subj_sig.iterdir():
                hp = ch_dir / pipeline_hash
                if hp.exists():
                    shutil.rmtree(hp)


def recommended_dtype() -> str:
    """Returns 'bfloat16' if CUDA+bfloat16 is supported at call time, else 'float32'."""
    return "bfloat16" if _BFLOAT16_SUPPORTED and _bfloat16 is not None else "float32"


def cast_to_cache_dtype(
    data: np.ndarray, dtype_name: Optional[str] = None
) -> np.ndarray:
    """Cast data to the specified cache dtype (defaults to `recommended_dtype()`)."""
    if dtype_name is None:
        dtype_name = recommended_dtype()
    dt = _resolve_dtype(dtype_name)
    if data.dtype == dt:
        return data
    return data.astype(dt)
