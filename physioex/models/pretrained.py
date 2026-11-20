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

Third-party cardiorespiratory models (wav2sleep, SleepPPG-Net, Jones ECG)
live in their authors' repos; they are registered in
``THIRD_PARTY_PRETRAINED`` below and downloaded on demand to the
foundation-model cache.  ``load_from_pretrained`` routes by name: registry
names build the third-party model, anything else falls through to the
4rooms/physioex behaviour.  ``list_pretrained()`` lists both families.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

import torch

from physioex.models.foundation_checkpoints import (
    CheckpointSource,
    ensure_checkpoint,
    _register_checkpoint_source,
)

logger = logging.getLogger("physioex.models")

HF_REPO_ID = "4rooms/physioex"

# ── Third-party pretrained models ────────────────────────────────────

#: Licenses that carry no known usage restriction and need no warning.
_PERMISSIVE_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-3-Clause"})


def _resolve_class(spec: str):
    """Import a class from a ``'module.path:ClassName'`` string."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _load_wav2sleep(paths: list[str]) -> torch.nn.Module:
    """Build Wav2Sleep from downloaded ``config.yaml`` + ``state_dict.pth``."""
    from physioex.models.wav2sleep import Wav2Sleep

    return Wav2Sleep.from_original(paths[0], paths[1])


def _load_wav2sleep_eog(paths: list[str]) -> torch.nn.Module:
    """Build Wav2Sleep (EOG variant) from ``config.yaml`` + ``state_dict.pth``."""
    from physioex.models.wav2sleep import Wav2Sleep

    return Wav2Sleep.from_original(paths[0], paths[1])


def _load_sleepppgnet(paths: list[str]) -> torch.nn.Module:
    """Build SleepPPGNet from the authors' replica weights (``class_order="physioex"``)."""
    from physioex.models.sleepppgnet import SleepPPGNet

    return SleepPPGNet.from_original(paths[0], class_order="physioex")


def _load_jones_ecg(paths: list[str], variant: str = "primary") -> torch.nn.Module:
    """Build JonesECGNet from a downloaded checkpoint (``variant`` = family name)."""
    from physioex.models.jones_ecg import JonesECGNet

    return JonesECGNet.from_original(paths[0], variant)


# Base folder for the Jones et al. 2024 checkpoints (adammj/ecg-sleep-staging).
_JONES_RAW_BASE = "https://raw.githubusercontent.com/adammj/ecg-sleep-staging/main/models"

# (variant in physioex registry, Jones checkpoint filename, sub-folder in the repo)
_JONES_VARIANTS = (
    ("primary", "epoch_0647_kappa_0.724.pt", "1_paper_models/primary_model"),
    ("realtime", "epoch_0637_kappa_0.703.pt", "1_paper_models/real-time_model"),
    ("no-demographics", "epoch_0552_kappa_0.718.pt", "2_post-paper_models/primary_model_no_demographics"),
    ("no-time", "epoch_0490_kappa_0.724.pt", "2_post-paper_models/primary_model_no_time"),
    ("only-ecg", "epoch_0369_kappa_0.721.pt", "2_post-paper_models/primary_model_only_ecg"),
    ("cyclic", "epoch_0548_kappa_0.725.pt", "3_cyclic-time_models/primary_model"),
    ("cyclic-realtime", "epoch_0472_kappa_0.703.pt", "3_cyclic-time_models/real-time_model"),
)

#: physioex's 5-class AASM order (W, N1, N2, N3, REM).
_AASM_CLASS_NAMES = ("W", "N1", "N2", "N3", "REM")


def _jones_spec(variant: str, filename: str, folder: str) -> CheckpointSource:
    url = f"{_JONES_RAW_BASE}/{folder}/{filename}"
    return CheckpointSource(
        source_type="raw_url",
        repo_id="adammj/ecg-sleep-staging",
        filenames=[filename],
        raw_urls=[url],
        local_filename=filename,
    )


def _third_party_registry() -> dict:
    """Build ``THIRD_PARTY_PRETRAINED`` (the seven Jones variants expand inline)."""
    registry: dict = {
        "wav2sleep": {
            "source": CheckpointSource(
                source_type="huggingface_file",
                repo_id="joncarter/wav2sleep",
                filenames=["config.yaml", "state_dict.pth"],
            ),
            "loader": _load_wav2sleep,
            "license": "MIT",
            "license_note": "MIT license from the authors' repository.",
            "reference": (
                "Carter, J. F. & Tarassenko, L. (2024). wav2sleep: A Unified "
                "Multi-Modal Approach to Sleep Stage Classification from "
                "Physiological Signals. arXiv:2411.04644. "
                "https://github.com/joncarter1/wav2sleep"
            ),
            "n_classes": 4,
            "class_names": ("W", "Light", "Deep", "REM"),
            "input": (
                "Whole night (1200 x 30-s epochs) of any subset of ECG, PPG "
                "(1024 samples/epoch) and ABD, THX (256 samples/epoch); absent "
                "signals = -inf; preset \"wav2sleep\""
            ),
        },
        "wav2sleep-eog": {
            "source": CheckpointSource(
                source_type="huggingface_file",
                repo_id="joncarter/wav2sleep-eog",
                filenames=["config.yaml", "state_dict.pth"],
            ),
            "loader": _load_wav2sleep_eog,
            "license": "MIT",
            "license_note": "MIT license from the authors' repository.",
            "reference": (
                "Carter, J. F. & Tarassenko, L. (2024), wav2sleep EOG release. "
                "https://huggingface.co/joncarter/wav2sleep-eog"
            ),
            "n_classes": 5,
            "class_names": _AASM_CLASS_NAMES,
            "input": (
                "Whole night (1200 x 30-s epochs) of EOG-L, EOG-R at 4096 "
                "samples/epoch; preset \"wav2sleep\""
            ),
        },
        "sleepppgnet-replica": {
            "source": CheckpointSource(
                source_type="huggingface_file",
                repo_id="sleep-is-all-you-need/SPPPPG-Net",
                filenames=["modelStatebe14cbce-76c1a81b-dd25c9a2-SleepPPGNet-"
                           "857c0582-b900d29c-6864dcd6-39dc88ff-0--current"],
                local_filename="sleepppgnet-replica.pt",
            ),
            "loader": _load_sleepppgnet,
            "license": "fair-noncommercial-research",
            "license_note": (
                "Third-party replica weights; non-commercial research use only, "
                "see the SPPPPG-Net repository."
            ),
            "reference": (
                "Architecture: Kotzen, K. et al. (2023). SleepPPG-Net: a deep "
                "learning algorithm for robust sleep staging from continuous "
                "photoplethysmography. IEEE JBHI, arXiv:2202.05735. Weights: "
                "third-party replica https://huggingface.co/sleep-is-all-you-need/SPPPPG-Net"
            ),
            "n_classes": 4,
            "class_names": ("W", "Light", "Deep", "REM"),
            "input": (
                "Whole night PPG, 1 channel, 1200 x 1024 samples (34.13 Hz); "
                "preset \"sleepppgnet\""
            ),
        },
    }
    for variant, filename, folder in _JONES_VARIANTS:
        registry[f"jones-ecg-{variant}"] = {
            "source": _jones_spec(variant, filename, folder),
            "loader": (lambda paths, v=variant: _load_jones_ecg(paths, v)),
            "license": "AGPL-3.0",
            "license_note": (
                "AGPL-3.0 per the authors' repository; note that AGPL copyleft "
                "may propagate to derivative works."
            ),
            "reference": (
                "Jones, A. M., Itti, L. & Sheth, B. R. (2024). Expert-level sleep "
                "staging using an electrocardiography-only feed-forward neural "
                "network. Computers in Biology and Medicine. "
                "doi:10.1016/j.compbiomed.2024.108545. "
                "https://github.com/adammj/ecg-sleep-staging"
            ),
            "n_classes": 5,
            "class_names": _AASM_CLASS_NAMES,
            "input": (
                "Whole night ECG, 1 channel at 256 Hz (7680 samples/epoch) plus "
                "per-epoch additional features; preset \"jones_ecg\""
            ),
        }
    # Alias: the paper's primary checkpoint.
    registry["jones-ecg"] = dict(
        registry["jones-ecg-primary"], alias="jones-ecg-primary"
    )
    # Register all sources in the download-cache registry so ensure_checkpoint
    # knows where (and whether) each checkpoint lives.
    for name, spec in registry.items():
        _register_checkpoint_source(name, spec["source"])
    return registry


THIRD_PARTY_PRETRAINED: dict = _third_party_registry()


def _third_party_names() -> tuple:
    """Names in the registry, following aliases to their canonical target."""
    names = []
    for name in THIRD_PARTY_PRETRAINED:
        spec = THIRD_PARTY_PRETRAINED[name]
        target = spec.get("alias") or name
        names.append(target if target in THIRD_PARTY_PRETRAINED else name)
    return tuple(names)


def _license_is_restrictive(license_: str) -> bool:
    """True unless the license is in :data:`_PERMISSIVE_LICENSES`."""
    return license_.strip().lower() not in {l.lower() for l in _PERMISSIVE_LICENSES}


# (name, reference). License is "physioex" (permissive, Apache-2.0) for the
# 4rooms family; n_classes=None means the model takes it from config.json.
_4ROOMS_PRETRAINED = (
    ("seqsleepnet-huy", "Huy et al. (2022)"),
    ("seqsleepnet-phan", "Phan et al. (2021)"),
    ("lseqsleepnet-phan", "Phan et al. (2022)"),
    ("sleeptransformer-phan", "Phan et al. (2021)"),
    ("tinysleepnet-supratak", "Supratak et al. (2017)"),
    ("chambon2018", "Chambon et al. (2018)"),
    ("tsinalis-2016", "Tsinalis et al. (2016)"),
    ("coresleep-kontras", "Kontras et al. (2024)"),
    ("protosleepnet-gagliardi", "Gagliardi et al. (2026)"),
    ("protosleeptransformer-gagliardi", "Gagliardi et al. (2026)"),
)


def list_pretrained() -> list[dict]:
    """List the pretrained models available through ``load_from_pretrained``.

    Covers the third-party registry (wav2sleep, SleepPPG-Net, Jones ECG) and a
    static list of the 4rooms/physioex models; no network access.

    Returns:
        One dict per model with keys ``name``, ``source``, ``license``,
        ``reference``, ``n_classes`` (and ``class_names`` for the
        third-party models).
    """
    entries: list[dict] = []
    for name in THIRD_PARTY_PRETRAINED:
        spec = THIRD_PARTY_PRETRAINED[name]
        src = spec["source"]
        if src.source_type == "raw_url":
            source = f"github: {src.repo_id} ({src.filenames[0]})"
        elif src.source_type == "huggingface_file":
            source = f"hf: {src.repo_id} ({', '.join(src.filenames)})"
        else:
            source = f"hf: {src.repo_id}"
        entries.append({
            "name": name,
            "source": source,
            "license": spec["license"],
            "reference": spec["reference"],
            "n_classes": spec["n_classes"],
            "class_names": list(spec["class_names"]),
        })
    for name, reference in _4ROOMS_PRETRAINED:
        entries.append({
            "name": name,
            "source": f"hf: {HF_REPO_ID}/{name}",
            "license": "Apache-2.0",
            "reference": reference,
            "n_classes": None,
            "class_names": None,
        })
    return entries


def _load_third_party(
    name: str, device: str, verbose: bool, repo_id: str | None
) -> torch.nn.Module:
    """Download (if needed) and build a third-party registered model."""
    spec = THIRD_PARTY_PRETRAINED[name]
    target = spec.get("alias", name)
    if target != name:
        spec = THIRD_PARTY_PRETRAINED[target]
        name = target
    source = spec["source"]

    if source.source_type == "huggingface_file":
        # ensure_checkpoint returns the first file; all files land in the
        # same directory (see _download_hf_file).
        primary = Path(ensure_checkpoint(name))
        if len(source.filenames) > 1:
            paths = [str(primary.parent / f) for f in source.filenames]
        else:
            paths = [str(primary)]
    elif source.source_type == "raw_url":
        paths = ensure_checkpoint(name)
        if isinstance(paths, str):
            paths = [paths]
    else:
        raise ValueError(
            f"Unsupported source_type {source.source_type!r} for {name!r}"
        )

    model = spec["loader"](paths)

    if _license_is_restrictive(spec["license"]):
        logger.warning(
            "%s: weights under %s — %s",
            name, spec["license"], spec["license_note"],
        )

    model = model.to(device).eval()

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model:     {name}")
        print(f"Class:     {type(model).__module__}.{type(model).__qualname__}")
        print(f"Params:    {n_params:,}")
        print(f"Reference: {spec['reference']}")
        print(f"License:   {spec['license']} ({spec['license_note']})")
        print(f"Classes:   {', '.join(spec['class_names'])}")
        print(f"Input:     {spec['input']}")

    return model


# (key in metrics.json, column header, width). A column is printed only if at
# least one dataset reports it, so a metrics.json without precision/recall does
# not show made-up zeros.
_METRIC_COLUMNS = (
    ("accuracy", "Acc", 7),
    ("f1_score", "F1", 7),
    ("cohen_kappa", "Kappa", 7),
    ("precision", "Prec", 7),
    ("recall", "Rec", 7),
    ("n_subjects", "N", 5),
)


def _format_dataset_metrics(metrics: dict) -> list:
    """Table lines for a per-dataset ``metrics.json`` (``{dataset: {metric: value}}``)."""
    cols = [c for c in _METRIC_COLUMNS if any(c[0] in m for m in metrics.values())]
    name_w = max([15] + [len(k) for k in metrics])
    header = f"{'Dataset':{name_w}s}" + "".join(f" {h:>{w}s}" for _, h, w in cols)
    lines = [header, "-" * max(60, len(header))]
    for name, m in sorted(metrics.items()):
        cells = []
        for key, _, w in cols:
            v = m.get(key)
            if v is None:
                cells.append(f" {'-':>{w}s}")
            elif key == "n_subjects":
                cells.append(f" {int(v):>{w}d}")
            else:
                cells.append(f" {v:{w}.4f}")
        lines.append(f"{name:{name_w}s}" + "".join(cells))
    return lines


def load_from_pretrained(
    name: str, device: str = "cpu", verbose: bool = False,
    repo_id: str = None,
) -> torch.nn.Module:
    """Load a pretrained model, downloading the weights if needed.

    If ``name`` is a registered third-party model (see
    :data:`THIRD_PARTY_PRETRAINED` — wav2sleep, wav2sleep-eog,
    sleepppgnet-replica, jones-ecg-*) and ``repo_id`` is not set, the
    checkpoint is downloaded to the foundation-model cache
    (``$PHYSIOEX_CACHE_DIR/v1/foundation_models/<name>/``) and the model is
    built with the authors' ``from_original`` factory.  Licenses that are
    not MIT/Apache/BSD are always logged as a warning.

    Otherwise the 4rooms/physioex behaviour applies: the model class and
    constructor kwargs are read from the ``config.json`` stored in the HF
    repo (e.g. ``"seqsleepnet-huy"``).

    Args:
        name: Model identifier — a third-party registry name or a sub-folder
            in the HF repo.
        device: Device to load the model onto (default ``"cpu"``).
        verbose: If True, print model info; for third-party models also the
            reference, license, classes and expected input.
        repo_id: Override the HF repo for the 4rooms-style load.

    Returns:
        The pretrained ``nn.Module`` in eval mode on *device*.

    Example::

        from physioex.models import load_from_pretrained

        model = load_from_pretrained("seqsleepnet-huy", verbose=True)
        model = load_from_pretrained("wav2sleep", verbose=True)
    """
    if name in THIRD_PARTY_PRETRAINED and repo_id is None:
        return _load_third_party(name, device, verbose, repo_id)

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
                print("\n" + "\n".join(_format_dataset_metrics(metrics)))
            else:
                # Legacy flat format
                print(
                    f"\nAccuracy: {metrics.get('accuracy', 0):.4f}, "
                    f"F1: {metrics.get('f1_score', 0):.4f}, "
                    f"Kappa: {metrics.get('cohen_kappa', 0):.4f}"
                )
        except Exception as exc:
            # metrics.json is optional; note the reason at debug level.
            logger.debug("No metrics.json for %s (%s)", name, exc)

    return model
