"""Load pretrained PhysioEx models from HuggingFace Hub.

Models are stored in the ``4rooms/physioex`` HF repo as sub-folders::

    4rooms/physioex/
      seqsleepnet-huy/
        config.json      # model_class, model_kwargs, training info
        model.pt         # torch state_dict

The ``config.json`` carries everything needed to reconstruct the model
(class path, constructor kwargs) so the caller does not need to know
the architecture in advance.

Cached locally under ``$PHYSIOEX_CACHE_DIR/pretrained/{name}/``
(defaults to ``~/.cache/physioex/pretrained/{name}/``).
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import torch

HF_REPO_ID = "4rooms/physioex"


def _cache_dir(name: str) -> Path:
    root = os.environ.get("PHYSIOEX_CACHE_DIR", os.path.expanduser("~/.cache/physioex"))
    return Path(root) / "pretrained" / name


def _resolve_class(spec: str):
    """Import a class from a ``'module.path:ClassName'`` string."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def load_from_pretrained(name: str, device: str = "cpu") -> torch.nn.Module:
    """Load a pretrained model from HuggingFace Hub.

    The model class and constructor kwargs are read from ``config.json``
    stored alongside the weights on HuggingFace.  The caller only needs
    to know the model identifier (e.g. ``"seqsleepnet-huy"``).

    Args:
        name: Model identifier matching a sub-folder in the HF repo
            (e.g. ``"seqsleepnet-huy"``).
        device: Device to load the model onto (default ``"cpu"``).

    Returns:
        The pretrained ``nn.Module`` in eval mode on *device*.

    Example::

        from physioex.models import load_from_pretrained

        model = load_from_pretrained("seqsleepnet-huy")
    """
    from huggingface_hub import hf_hub_download

    cache = _cache_dir(name)
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Download and read config
    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"{name}/config.json",
        local_dir=str(cache),
    )
    with open(config_path) as f:
        config = json.load(f)

    # 2. Resolve model class and kwargs
    model_cls = _resolve_class(config["model_class"])
    model_kwargs = config["model_kwargs"]

    # 3. Download weights
    weights_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"{name}/model.pt",
        local_dir=str(cache),
    )

    # 4. Instantiate and load
    model = model_cls(**model_kwargs)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model.to(device).eval()
