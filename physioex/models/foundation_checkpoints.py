"""Centralized checkpoint management for foundation models.

Checkpoints are cached under:

    {cache_root}/v1/foundation_models/{model_name}/
        {filename}          # the checkpoint file(s)

where ``cache_root`` defaults to ``$PHYSIOEX_CACHE_DIR`` or ``~/.cache/physioex``.

Resolution order (per model):
  1. If the caller passes an explicit ``checkpoint_path`` → use it directly,
     no caching (avoids overwriting downloaded models).
  2. If the checkpoint is already in cache → return the cached path.
  3. Otherwise download it (from HuggingFace Hub, GitHub, etc.) and cache it.

Models that live entirely on HuggingFace as ``from_pretrained()``-style repos
(CBraMod, BENDR, SJEPA) return ``None`` from ``ensure_checkpoint()`` — the
wrapper itself calls ``from_pretrained()`` with the repo id.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("physioex.foundation.checkpoints")


def _cache_root() -> Path:
    return Path(
        os.environ.get("PHYSIOEX_CACHE_DIR", os.path.expanduser("~/.cache/physioex"))
    )


def _models_dir() -> Path:
    return _cache_root() / "v1" / "foundation_models"


def _checkpoint_dir(model_name: str) -> Path:
    return _models_dir() / model_name / "checkpoint"


def _embeddings_dir(model_name: str, dataset_name: str) -> Path:
    return _models_dir() / model_name / "embeddings" / dataset_name


def _probes_dir(model_name: str, dataset_name: str, fold: int) -> Path:
    return _models_dir() / model_name / "probes" / dataset_name / f"fold_{fold}"


# ── Registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckpointSource:
    """Describes where to get a model's checkpoint."""

    # "huggingface_repo"  — full model directory, use snapshot_download / from_pretrained
    # "huggingface_file"  — single file in a HF repo, use hf_hub_download()
    # "github"            — shallow-clone a GitHub repo and extract a file
    # "none"              — no checkpoint needed (handled by from_pretrained inside wrapper)
    source_type: str

    # HF repo_id or GitHub URL
    repo_id: str = ""

    # For huggingface_file: remote filename(s) to download (may include subdirs)
    filenames: List[str] = field(default_factory=list)

    # For github: path inside the cloned repo to the checkpoint file
    github_internal_path: str = ""

    # Local filename to save as (defaults to basename of the first filename/github path)
    local_filename: str = ""

    # For huggingface_repo: required files that must exist in the cached directory
    required_files: List[str] = field(default_factory=list)

    # If True, the result is a directory (for from_pretrained-style models)
    is_directory: bool = False


# Registry: model_name → how to get its checkpoint.
# These are the default sources; explicit checkpoint_path overrides everything.
CHECKPOINT_REGISTRY: Dict[str, CheckpointSource] = {
    # ── from_pretrained models (no file download needed) ──
    "cbramod": CheckpointSource(
        source_type="none",
        repo_id="braindecode/cbramod-pretrained",
    ),
    "bendr": CheckpointSource(
        source_type="none",
        repo_id="braindecode/braindecode-bendr",
    ),
    "sjepa": CheckpointSource(
        source_type="none",
        repo_id="braindecode/SignalJEPA-pretrained",
    ),
    # ── HuggingFace repo (directory download) ──
    "reve": CheckpointSource(
        source_type="huggingface_repo",
        repo_id="brain-bzh/reve-base",
        is_directory=True,
        required_files=["config.json", "model.safetensors"],
    ),
    # ── HuggingFace single file ──
    "neurolm": CheckpointSource(
        source_type="huggingface_file",
        repo_id="Weibang/NeuroLM",
        filenames=["checkpoints/VQ.pt"],
        local_filename="VQ.pt",
    ),
    # ── GitHub clone (shallow clone + extract file) ──
    "biot": CheckpointSource(
        source_type="github",
        repo_id="https://github.com/ycq091044/BIOT",
        github_internal_path="EEG-PREST-16-channels.ckpt",
        local_filename="EEG-PREST-16-channels.ckpt",
    ),
    "sleepfm": CheckpointSource(
        source_type="github",
        repo_id="https://github.com/zou-group/sleepfm-clinical",
        github_internal_path="sleepfm/checkpoints/model_base/best.pt",
        local_filename="best.pt",
    ),
    "tfc": CheckpointSource(
        source_type="github",
        repo_id="https://github.com/mims-harvard/TFC-pretraining",
        github_internal_path="code/experiments_logs/SleepEEG_2_Epilepsy/run1/"
        "pre_train_seed_42_2layertransformer/saved_models/ckp_last.pt",
        local_filename="pretrain_sleepEEG.pt",
    ),
    "labram": CheckpointSource(
        source_type="huggingface_file",
        repo_id="braindecode/Labram-Braindecode",
        filenames=["braindecode_labram_base.pt"],
        local_filename="labram-base.pth",
    ),
}

# Position bank for REVE (separate model)
REVE_POSITIONS_SOURCE = CheckpointSource(
    source_type="huggingface_repo",
    repo_id="brain-bzh/reve-positions",
    is_directory=True,
    required_files=["config.json", "model.safetensors"],
)


# ── Public API ───────────────────────────────────────────────────────


def get_checkpoint_dir(model_name: str) -> Path:
    """Return the checkpoint cache directory for a model (may not exist yet)."""
    return _checkpoint_dir(model_name)


def get_embeddings_dir(model_name: str, dataset_name: str) -> Path:
    """Return the embeddings cache directory for a model+dataset."""
    return _embeddings_dir(model_name, dataset_name)


def get_probes_dir(model_name: str, dataset_name: str, fold: int) -> Path:
    """Return the probes cache directory for a model+dataset+fold."""
    return _probes_dir(model_name, dataset_name, fold)


def ensure_checkpoint(
    model_name: str,
    checkpoint_path: Optional[str] = None,
) -> Optional[str]:
    """Resolve a checkpoint path for a foundation model.

    Args:
        model_name: registered model name (e.g., "biot", "cbramod").
        checkpoint_path: explicit path from the user. If provided, returned
            as-is (no caching, no downloading).

    Returns:
        - The local path to the checkpoint file or directory.
        - ``None`` for models that use ``from_pretrained()`` internally
          (cbramod, bendr, sjepa) when no explicit path is given.

    Raises:
        KeyError: if model_name is not in the registry.
        FileNotFoundError: if explicit checkpoint_path doesn't exist.
        RuntimeError: if download fails.
    """
    # 1. Explicit path — use directly
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Explicit checkpoint_path does not exist: {checkpoint_path}"
            )
        logger.info(f"[{model_name}] Using explicit checkpoint: {checkpoint_path}")
        return checkpoint_path

    # 2. Look up registry
    if model_name not in CHECKPOINT_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. "
            f"Available: {sorted(CHECKPOINT_REGISTRY)}"
        )

    source = CHECKPOINT_REGISTRY[model_name]

    # 3. "none" — wrapper handles it internally (from_pretrained)
    if source.source_type == "none":
        return None

    # 4. Check cache
    model_dir = get_checkpoint_dir(model_name)

    if source.source_type == "huggingface_file":
        local_name = source.local_filename or Path(source.filenames[0]).name
        primary = model_dir / local_name
        if primary.exists():
            logger.info(f"[{model_name}] Checkpoint cached at {primary}")
            return str(primary)
        return _download_hf_file(model_name, source, model_dir)

    elif source.source_type == "huggingface_repo":
        if source.required_files and all(
            (model_dir / f).exists() for f in source.required_files
        ):
            logger.info(f"[{model_name}] Checkpoint cached at {model_dir}")
            return str(model_dir)
        return _download_hf_repo(model_name, source, model_dir)

    elif source.source_type == "github":
        local_name = source.local_filename or Path(source.github_internal_path).name
        primary = model_dir / local_name
        if primary.exists():
            logger.info(f"[{model_name}] Checkpoint cached at {primary}")
            return str(primary)
        return _download_github(model_name, source, model_dir)

    else:
        raise ValueError(f"Unknown source_type: {source.source_type!r}")


def ensure_reve_positions() -> str:
    """Ensure the REVE position bank is cached. Returns the local directory."""
    model_dir = _models_dir() / "reve" / "positions"
    source = REVE_POSITIONS_SOURCE
    if source.required_files and all(
        (model_dir / f).exists() for f in source.required_files
    ):
        return str(model_dir)
    return _download_hf_repo("reve-positions", source, model_dir)


def get_repo_id(model_name: str) -> Optional[str]:
    """Return the HuggingFace repo_id for a model, or None."""
    source = CHECKPOINT_REGISTRY.get(model_name)
    if source is None:
        return None
    return source.repo_id or None


# ── Download helpers ─────────────────────────────────────────────────


def _download_hf_file(
    model_name: str, source: CheckpointSource, model_dir: Path
) -> str:
    """Download file(s) from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    local_name = source.local_filename or Path(source.filenames[0]).name
    target = model_dir / local_name

    for filename in source.filenames:
        logger.info(f"[{model_name}] Downloading {filename} from {source.repo_id}...")
        try:
            token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get(
                "HF_TOKEN"
            )
            downloaded = hf_hub_download(
                repo_id=source.repo_id,
                filename=filename,
                token=token,
            )
            # Copy to our cache (hf_hub_download caches in its own dir)
            dst = model_dir / (source.local_filename or Path(filename).name)
            shutil.copy2(downloaded, dst)
            logger.info(f"[{model_name}] Cached at {dst}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {filename} from {source.repo_id}: {e}\n"
                f"You can manually place the checkpoint at {target}"
            ) from e

    return str(target)


def _download_hf_repo(
    model_name: str, source: CheckpointSource, model_dir: Path
) -> str:
    """Download an entire HuggingFace repo (for from_pretrained-style models)."""
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{model_name}] Downloading repo {source.repo_id}...")
    try:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        snapshot_download(
            repo_id=source.repo_id,
            local_dir=str(model_dir),
            token=token,
        )
        logger.info(f"[{model_name}] Cached at {model_dir}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download repo {source.repo_id}: {e}\n"
            f"You can manually place the model files at {model_dir}"
        ) from e

    return str(model_dir)


def _download_github(model_name: str, source: CheckpointSource, model_dir: Path) -> str:
    """Shallow-clone a GitHub repo and extract a specific file."""
    model_dir.mkdir(parents=True, exist_ok=True)
    local_name = source.local_filename or Path(source.github_internal_path).name
    target = model_dir / local_name

    repo_url = source.repo_id
    if not repo_url.startswith("http"):
        repo_url = f"https://github.com/{repo_url}"
    if not repo_url.endswith(".git"):
        repo_url = repo_url + ".git"

    logger.info(
        f"[{model_name}] Cloning {repo_url} (shallow) to extract "
        f"{source.github_internal_path}..."
    )

    with tempfile.TemporaryDirectory(prefix="physioex_dl_") as tmpdir:
        clone_dir = os.path.join(tmpdir, "repo")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, clone_dir],
                check=True,
                capture_output=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to clone {repo_url}: {e.stderr.decode()}\n"
                f"You can manually place the checkpoint at {target}"
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                "git is not installed. Install git or manually place the "
                f"checkpoint at {target}"
            )

        src = Path(clone_dir) / source.github_internal_path
        if not src.exists():
            # Try glob search as fallback
            candidates = list(
                Path(clone_dir).rglob(Path(source.github_internal_path).name)
            )
            if candidates:
                src = candidates[0]
            else:
                raise RuntimeError(
                    f"Expected checkpoint at {source.github_internal_path} "
                    f"in {repo_url} but not found after cloning.\n"
                    f"You can manually place the checkpoint at {target}"
                )

        shutil.copy2(str(src), str(target))

        # Copy companion files if present (config.json, LICENSE)
        for extra in ("config.json", "LICENSE", "LICENSE.md"):
            extra_src = src.parent / extra
            extra_dst = model_dir / extra
            if extra_src.exists() and not extra_dst.exists():
                shutil.copy2(str(extra_src), str(extra_dst))

    logger.info(f"[{model_name}] Cached at {target}")
    return str(target)


# ── Utilities ────────────────────────────────────────────────────────


def list_cached_models() -> Dict[str, Optional[str]]:
    """Return {model_name: cached_path_or_None} for all registered models."""
    result = {}
    for name in CHECKPOINT_REGISTRY:
        model_dir = get_checkpoint_dir(name)
        source = CHECKPOINT_REGISTRY[name]
        if source.source_type == "none":
            result[name] = None  # from_pretrained, always available
            continue
        if source.is_directory:
            if source.required_files and all(
                (model_dir / f).exists() for f in source.required_files
            ):
                result[name] = str(model_dir)
            else:
                result[name] = None
        else:
            local_name = source.local_filename or (
                Path(source.filenames[0]).name
                if source.filenames
                else Path(source.github_internal_path).name
            )
            p = model_dir / local_name
            result[name] = str(p) if p.exists() else None
    return result


def clear_cache(model_name: Optional[str] = None) -> None:
    """Remove cached checkpoints. If model_name is None, clear all."""
    if model_name:
        d = get_checkpoint_dir(model_name)
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"Cleared cache for {model_name}")
    else:
        root = _models_dir()
        if root.exists():
            shutil.rmtree(root)
            logger.info("Cleared all foundation model caches")
