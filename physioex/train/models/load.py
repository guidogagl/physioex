import importlib
import os

import pandas as pd
import torch


def _get_registry():
    """Load the model registry from check_table.csv."""
    registry_path = os.path.join(os.path.dirname(__file__), "check_table.csv")
    return pd.read_csv(registry_path)


def load_model(
    model_class,
    model_kwargs: dict,
    ckpt_path: str = None,
    model_name: str = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a model with dual-format checkpoint support.

    Supports three checkpoint formats:
    - Lightning .ckpt (has "state_dict" key with "nn." prefixed keys)
    - New .pt format (has "model_state_dict" key)
    - Raw state_dict (plain dict of parameter tensors)

    Args:
        model_class: A torch.nn.Module subclass, or a string "module:Class".
        model_kwargs: Constructor kwargs for the model class.
        ckpt_path: Path to checkpoint file. If None, uses registry lookup.
        model_name: Name for registry lookup (e.g. "seqsleepnet").
        device: Device to load the model onto.

    Returns:
        The loaded model in eval mode on the specified device.
    """
    # --- 1. Resolve model class ---
    if isinstance(model_class, str) and ":" in model_class:
        module_path, class_name = model_class.split(":")
        model_class = getattr(importlib.import_module(module_path), class_name)

    # --- 2. Resolve checkpoint path ---
    if ckpt_path is not None:
        pass  # use it directly
    elif model_name is not None:
        table = _get_registry()

        # Determine in_channels from model_kwargs
        in_channels = model_kwargs.get("in_channels", None)
        if in_channels is None:
            selected_channels = model_kwargs.get("selected_channels", None)
            if selected_channels is not None:
                in_channels = len(selected_channels)

        seq_len = model_kwargs.get("sequence_length", None)

        # Build filter
        mask = table["name"] == model_name
        if seq_len is not None:
            mask = mask & (table["sequence_length"] == seq_len)
        if in_channels is not None:
            mask = mask & (table["in_channels"] == in_channels)

        filtered = table[mask]
        if len(filtered) == 0:
            raise ValueError(
                f"No registry entry found for model_name='{model_name}' "
                f"with sequence_length={seq_len}, in_channels={in_channels}"
            )

        row = filtered.iloc[0]
        ckpt_filename = row["checkpoint"]

        # Resolve to absolute path relative to the checkpoints dir
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)

        # Auto-download from HuggingFace Hub if file is missing
        if not os.path.isfile(ckpt_path):
            from huggingface_hub import hf_hub_download

            os.makedirs(checkpoints_dir, exist_ok=True)
            ckpt_path = hf_hub_download(
                repo_id="4rooms/physioex",
                filename=os.path.basename(ckpt_path),
                local_dir=os.path.dirname(ckpt_path),
            )
    else:
        raise ValueError("Either ckpt_path or model_name must be provided")

    # --- 3. Load checkpoint (dual format) ---
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Lightning format -- strip "nn." prefix from keys that have it
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            new_k = k[3:] if k.startswith("nn.") else k
            state_dict[new_k] = v
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")

    # --- 4. Instantiate and load ---
    model = model_class(**model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[Warning - load_model]: missing keys: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
        print(
            f"[Warning - load_model]: unexpected keys: "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )

    return model.to(device).eval()
