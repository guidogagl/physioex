"""Load pretrained PhysioEx models from HuggingFace Hub.

Models are stored in the ``4rooms/physioex`` HF repo as sub-folders::

    4rooms/physioex/
      seqsleepnet-huy/
        config.json      # model_class, model_kwargs, training info
        model.pt         # torch state_dict
        metrics.json     # per-dataset evaluation results

The ``config.json`` carries everything needed to reconstruct the model
(class path, constructor kwargs) so the caller does not need to know
the architecture in advance.

Cached locally under the standard HuggingFace cache directory.
"""
from __future__ import annotations

import importlib
import json

import torch

HF_REPO_ID = "4rooms/physioex"


def _resolve_class(spec: str):
    """Import a class from a ``'module.path:ClassName'`` string."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def load_from_pretrained(
    name: str, device: str = "cpu", verbose: bool = False,
    repo_id: str = None,
) -> torch.nn.Module:
    """Load a pretrained model from HuggingFace Hub.

    The model class and constructor kwargs are read from ``config.json``
    stored alongside the weights on HuggingFace.  The caller only needs
    to know the model identifier (e.g. ``"seqsleepnet-huy"``).

    Args:
        name: Model identifier matching a sub-folder in the HF repo
            (e.g. ``"seqsleepnet-huy"``).
        device: Device to load the model onto (default ``"cpu"``).
        verbose: If True, print model info and per-dataset evaluation
            metrics stored on HuggingFace.

    Returns:
        The pretrained ``nn.Module`` in eval mode on *device*.

    Example::

        from physioex.models import load_from_pretrained

        model = load_from_pretrained("seqsleepnet-huy", verbose=True)
    """
    from huggingface_hub import hf_hub_download

    repo = repo_id or HF_REPO_ID

    # 1. Download and read config
    config_path = hf_hub_download(
        repo_id=repo,
        filename=f"{name}/config.json",
    )
    with open(config_path) as f:
        config = json.load(f)

    # 2. Resolve model class
    model_cls = _resolve_class(config["model_class"])

    # 3. Download weights
    weights_path = hf_hub_download(
        repo_id=repo,
        filename=f"{name}/model.pt",
    )

    # 4. Instantiate: via a factory classmethod if the config specifies one
    #    (e.g. ProtoSleepNet.from_sleep_transformer), else plain constructor.
    if "factory" in config:
        model = getattr(model_cls, config["factory"])(**config.get("factory_kwargs", {}))
    else:
        model = model_cls(**config.get("model_kwargs", {}))
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # 5. Verbose: print model info and stored metrics
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model:     {name}")
        print(f"Class:     {config['model_class']}")
        print(f"Params:    {n_params:,}")
        if "reference" in config:
            print(f"Reference: {config['reference']}")
        if "training" in config:
            t = config["training"]
            print(
                f"Trained on: {t.get('dataset', '?')}, "
                f"channels={t.get('channels', '?')}, "
                f"L={t.get('sequence_length', '?')}, "
                f"epochs={t.get('max_epochs', '?')}"
            )

        # Download and display per-dataset metrics
        try:
            metrics_path = hf_hub_download(
                repo_id=repo,
                filename=f"{name}/metrics.json",
            )
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Detect format: per-dataset dict or flat (legacy)
            first_val = next(iter(metrics.values()), None)
            if isinstance(first_val, dict):
                # Per-dataset format
                print(
                    f"\n{'Dataset':15s} {'Acc':>7s} {'F1':>7s} "
                    f"{'Kappa':>7s} {'Prec':>7s} {'Rec':>7s}"
                )
                print("-" * 60)
                for ds_name, ds_metrics in sorted(metrics.items()):
                    print(
                        f"{ds_name:15s} "
                        f"{ds_metrics.get('accuracy', 0):7.4f} "
                        f"{ds_metrics.get('f1_score', 0):7.4f} "
                        f"{ds_metrics.get('cohen_kappa', 0):7.4f} "
                        f"{ds_metrics.get('precision', 0):7.4f} "
                        f"{ds_metrics.get('recall', 0):7.4f}"
                    )
            else:
                # Legacy flat format
                print(
                    f"\nAccuracy: {metrics.get('accuracy', 0):.4f}, "
                    f"F1: {metrics.get('f1_score', 0):.4f}, "
                    f"Kappa: {metrics.get('cohen_kappa', 0):.4f}"
                )
        except Exception:
            pass  # metrics.json not available, silently skip

    return model
