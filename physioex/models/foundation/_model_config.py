"""Model configuration loader for foundation model wrappers.

Each model has a YAML config in foundation/configs/ defining:
- target_layout: the channel layout the model expects (or null for variable)
- channel_aliases: name aliases (e.g., T3→T7)
- missing_strategy: what to do with missing channels (zero_pad, skip)
- normalization: preprocessing normalization type
- normalization_params: parameters for normalization
- sampling_rate: target sampling rate
- embedding_dim: output embedding dimension
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


_CONFIG_DIR = Path(__file__).parent / "configs"
_CACHE: Dict[str, Dict[str, Any]] = {}


def load_model_config(model_name: str) -> Dict[str, Any]:
    """Load the YAML config for a foundation model.

    Cached after first load.
    """
    if model_name in _CACHE:
        return _CACHE[model_name]

    config_path = _CONFIG_DIR / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found for model {model_name!r} at {config_path}. "
            f"Available: {[f.stem for f in _CONFIG_DIR.glob('*.yaml')]}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    _CACHE[model_name] = config
    return config


def get_target_layout(model_name: str) -> Optional[List[str]]:
    """Return the target channel layout, or None if variable."""
    cfg = load_model_config(model_name)
    return cfg.get("target_layout")


def get_channel_aliases(model_name: str) -> Dict[str, str]:
    """Return channel name aliases for this model."""
    cfg = load_model_config(model_name)
    return cfg.get("channel_aliases", {})


def get_normalization(model_name: str) -> str:
    """Return normalization type: 'none', 'mean_center', 'clip100', 'zscore', 'q95', 'minmax'."""
    cfg = load_model_config(model_name)
    return cfg.get("normalization", "none")


def get_normalization_params(model_name: str) -> Dict[str, Any]:
    """Return normalization-specific parameters."""
    cfg = load_model_config(model_name)
    return cfg.get("normalization_params", {})
