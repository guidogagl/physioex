"""Shared CLI helpers for the train / finetune / test_model entry points.

All three commands use the SAME raw-EDF data layer (``physioex.data.datasets``)
so a model trained with a given ``--dataset/--channels/--pipelines`` spec can be
fine-tuned and evaluated with the identical spec.

Legacy flag names (``--datasets``, ``--selected_channels``, ``--preprocessing``,
``--seqlen``) are kept as aliases of the canonical flags so existing scripts keep
working.
"""
from __future__ import annotations

import argparse
import importlib
import json
from typing import Tuple

import yaml


def import_class(spec: str):
    """Import a class from a ``'module.path:ClassName'`` spec."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def parse_kwargs(raw: str) -> dict:
    """Parse a JSON (preferred) or YAML string into a dict."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return yaml.safe_load(raw) or {}


def add_dataset_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the canonical dataset flags shared by all three CLIs.

    Canonical flags (with legacy aliases writing the same dest):
      --dataset/--datasets, --channels/--selected_channels,
      --pipelines/--preprocessing, --sequence_length/--seqlen.
    """
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="dataset",
        nargs="+",
        required=True,
        help="One or more dataset names (e.g. hmc sleepedf dcsm). "
        "Multiple names are merged via MultiDataset.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Override default data root directory (else PHYSIOEX_DATA).",
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        default="{}",
        help="JSON/YAML extra constructor kwargs applied to every dataset "
        "(e.g. '{\"cohort\": 2}' for MASS).",
    )
    parser.add_argument(
        "--channels",
        "--selected_channels",
        dest="channels",
        nargs="+",
        default=["EEG", "EOG", "EMG"],
        help="Channels to load (modality names or physical names).",
    )
    parser.add_argument(
        "--pipelines",
        "--preprocessing",
        dest="pipelines",
        type=str,
        default="time_domain",
        help="Preset pipeline name: raw, time_domain, time_frequency, "
        "seqsleepnet, eeg, emg, etc.",
    )
    parser.add_argument(
        "--sequence_length",
        "--seqlen",
        dest="sequence_length",
        type=int,
        default=21,
        help="Epoch sequence length (L). -1 for full recordings.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Override default cache directory.",
    )


def apply_config_overlay(args: argparse.Namespace) -> None:
    """Overlay a YAML ``--config`` file over parsed args (non-None wins)."""
    if getattr(args, "config", None) is None:
        return
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}
    for k, v in config.items():
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)


def build_dataset_from_args(args: argparse.Namespace):
    """Build a single dataset or a MultiDataset from the parsed CLI args.

    Returns ``(dataset, n_channels)``. Uses the new raw-EDF data layer for all
    CLIs (no legacy PhysioExDataset).
    """
    from physioex.data.datasets import get_dataset
    from physioex.data.multi import MultiDataset

    names = args.dataset if isinstance(args.dataset, list) else [args.dataset]
    extra = parse_kwargs(getattr(args, "dataset_kwargs", "{}"))

    ds_kwargs = dict(
        channels=args.channels,
        pipelines=args.pipelines,
        sequence_length=args.sequence_length,
        cache_dir=args.cache_dir,
    )
    if getattr(args, "dataset_root", None) is not None:
        ds_kwargs["root"] = args.dataset_root

    built = [get_dataset(name)(**ds_kwargs, **extra) for name in names]
    dataset = built[0] if len(built) == 1 else MultiDataset(built)

    n_channels = len(args.channels) if args.channels is not None else 0
    return dataset, n_channels


def inject_in_chan(model_kwargs: dict, n_channels: int) -> dict:
    """Auto-inject ``in_chan`` from the channel count if not set explicitly.

    Accepts either ``in_chan`` or ``in_channels`` as already-provided; the
    canonical injected key is ``in_chan`` (matches the model zoo constructors).
    """
    if n_channels and "in_chan" not in model_kwargs and "in_channels" not in model_kwargs:
        model_kwargs["in_chan"] = n_channels
    return model_kwargs
