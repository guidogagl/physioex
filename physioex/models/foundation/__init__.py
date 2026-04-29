"""Foundation model wrappers for sleep staging.

Each model wraps a pretrained encoder (from braindecode, transformers, or
vendored code) with a trainable classification head. The encoder is frozen;
only the head trains. All preprocessing is pure PyTorch — differentiable
and GPU-compatible, with zero imports from EEGBenchmarks.

    model = get_foundation_model("cbramod")(n_classes=5, in_chan=3)
    Trainer.train(model, dataset, max_epochs=20)

Available models:
    biot      — BIOT (Yang et al., NeurIPS 2023)
    bendr     — BENDR (Kostas et al., 2021)
    cbramod   — CBraMod (Jiang et al., ICLR 2025)
    labram    — LaBraM (Jiang et al., ICLR 2024)
    sleepfm   — SleepFM (Thapa et al., ICML 2024)
    tfc       — TF-C (Zhang et al., NeurIPS 2022)
    reve      — REVE (Transformer with positional encoding bank)
    sjepa     — SJEPA (Spatial-JEPA)
    neurolm   — NeuroLM (VQ-tokenized encoder)
"""
from physioex.models.foundation.biot import BIOTSleepNet
from physioex.models.foundation.bendr import BENDRSleepNet
from physioex.models.foundation.cbramod import CBraModSleepNet
from physioex.models.foundation.labram import LaBraMSleepNet
from physioex.models.foundation.sleepfm import SleepFMSleepNet
from physioex.models.foundation.tfc import TFCSleepNet
from physioex.models.foundation.reve import REVESleepNet
from physioex.models.foundation.sjepa import SJEPASleepNet
from physioex.models.foundation.neurolm import NeuroLMSleepNet
from physioex.models.foundation._datasets import (
    DATASET_CONFIGS,
    get_dataset_config,
    available_dataset_configs,
)
from physioex.models.foundation._checkpoints import (
    ensure_checkpoint,
    get_checkpoint_dir,
    get_embeddings_dir,
    get_probes_dir,
    CHECKPOINT_REGISTRY,
)

FOUNDATION_MODELS = {
    "biot": BIOTSleepNet,
    "bendr": BENDRSleepNet,
    "cbramod": CBraModSleepNet,
    "labram": LaBraMSleepNet,
    "sleepfm": SleepFMSleepNet,
    "tfc": TFCSleepNet,
    "reve": REVESleepNet,
    "sjepa": SJEPASleepNet,
    "neurolm": NeuroLMSleepNet,
}


def get_foundation_model(name: str):
    if name not in FOUNDATION_MODELS:
        raise KeyError(
            f"Unknown foundation model {name!r}. "
            f"Available: {sorted(FOUNDATION_MODELS)}"
        )
    return FOUNDATION_MODELS[name]


def available_foundation_models() -> list[str]:
    return sorted(FOUNDATION_MODELS.keys())


__all__ = [
    "BIOTSleepNet",
    "BENDRSleepNet",
    "CBraModSleepNet",
    "LaBraMSleepNet",
    "SleepFMSleepNet",
    "TFCSleepNet",
    "REVESleepNet",
    "SJEPASleepNet",
    "NeuroLMSleepNet",
    "FOUNDATION_MODELS",
    "get_foundation_model",
    "available_foundation_models",
    "DATASET_CONFIGS",
    "get_dataset_config",
    "available_dataset_configs",
    "ensure_checkpoint",
    "get_checkpoint_dir",
    "get_embeddings_dir",
    "get_probes_dir",
    "CHECKPOINT_REGISTRY",
]
