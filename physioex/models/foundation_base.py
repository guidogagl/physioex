"""Base class for foundation model encoders.

Adapts a pretrained single-epoch encoder to return embeddings:
    (B, L, C, T) -> (B, L, D)

No classification head - embeddings are evaluated via linear_probe() in embed.py.

Each model declares two class attributes that describe its data requirements:

- ``PIPELINE_PRESET``   — resampling preset name (e.g. ``"biot"`` → 200 Hz)
- ``CHANNEL_STRATEGY``  — how the model handles channels internally:
    - ``"first"`` — uses only the first channel (BIOT, TF-C)
    - ``"pad_fixed"`` — zero-pads to a fixed count (SleepFM → 10)
    - ``"layout"`` — maps to a named channel layout (BENDR → 20)
    - ``"all"`` — uses all input channels as-is (CBraMod, LaBraM)

Channel selection is **per-dataset**, not per-model. The ``_datasets.py``
registry (mirroring EEGBenchmarks' DATASET_REGISTRY) specifies exactly which
EEG channels to request from each dataset. All models receive the same
channels for a given dataset; each model then handles them internally
according to its CHANNEL_STRATEGY.

Usage::

    from physioex.models import CBraModEncoder

    # get_dataset handles everything: pipeline, channels, root path
    dataset = CBraModEncoder.get_dataset("hmc")

    # in_chan = number of channels the dataset provides
    model = CBraModEncoder(in_chan=len(dataset.channels))

    # Extract embeddings
    embeddings = model(signals)  # (B, L, D)

    # Evaluate via linear probe
    from physioex.models.embed import extract_embeddings, linear_probe
    extract_embeddings(model, dataset, "cbramod", "hmc", L=21)
    linear_probe("cbramod", "hmc")
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("physioex.foundation")


class FoundationEncoder(nn.Module):
    """Abstract adapter: frozen encoder that returns embeddings.

    Subclasses implement:
      - ``_build_encoder(**kwargs)``  -> nn.Module (the pretrained encoder)
      - ``_get_embedding_dim()``      -> int
      - ``_load_pretrained(path)``    -> None (load checkpoint into self.encoder)
      - ``_encode(x)``               -> (B, D) embeddings from (B, C', T') input
      - ``_preprocess(x)``           -> (B, C', T') (optional: resampling, channel mapping)

    The wrapper handles:
      - Reshaping (B, L, C, T) -> (B*L, C, T) before encoding
      - Freezing all encoder parameters
      - Reshaping output back to (B, L, D)
    """

    # ── Subclass class attributes ────────────────────────────────────

    MODEL_NAME: str = "foundation"

    # Preset name for data resampling (registered in physioex/data/presets.py)
    PIPELINE_PRESET: Optional[str] = None

    # How this model handles channels internally:
    #   "first"     — uses only channel 0, ignores rest (BIOT, TF-C)
    #   "pad_fixed" — zero-pads to a fixed count (SleepFM → 10, BIOT → 16)
    #   "layout"    — maps to named layout (BENDR → 20-ch 10-20 system)
    #   "all"       — uses all channels as provided (CBraMod, LaBraM)
    CHANNEL_STRATEGY: str = "all"

    def __init__(
        self,
        in_chan: int,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.in_chan = in_chan

        # Subclass builds the encoder
        self.encoder = self._build_encoder(in_chan=in_chan, **kwargs)
        self.embedding_dim = self._get_embedding_dim()

        # Load pretrained weights
        self._load_pretrained(checkpoint_path)

        # Freeze encoder
        self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C, T) -> (B, L, D) embeddings."""
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        x = self._preprocess(x)
        with torch.no_grad():
            emb = self._encode(x)  # (B*L, D)
        return emb.reshape(B, L, -1)  # (B, L, D)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() - compatibility with embed.py."""
        return self.forward(x)

    # ── Subclass hooks ──────────────────────────────────────────────

    def _build_encoder(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    def _get_embedding_dim(self) -> int:
        raise NotImplementedError

    def _load_pretrained(self, checkpoint_path: str | None) -> None:
        raise NotImplementedError

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C', T') -> (B, D) embeddings."""
        raise NotImplementedError

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C', T') with resampling / channel mapping.

        Default: mean-center only.
        """
        return x - x.mean(dim=-1, keepdim=True)

    # ── Data loading helpers ────────────────────────────────────────

    @classmethod
    def get_pipeline(cls) -> "PreprocessingPipeline":
        """Return the preprocessing pipeline for this model.

        The pipeline is a resample-only pipeline matching the model's native
        sampling rate. Its hash is identical to the one used in EEGBenchmarks,
        so cached preprocessed data is shared between the two systems.

        Subclasses must set ``PIPELINE_PRESET`` class attribute.
        """
        from physioex.data.presets import get_preset

        preset = getattr(cls, "PIPELINE_PRESET", None)
        if preset is None:
            raise NotImplementedError(
                f"{cls.__name__} must define a PIPELINE_PRESET class attribute "
                f"(one of: biot, bendr, cbramod, labram, sleepfm, tfc)"
            )
        return get_preset(preset)

    @classmethod
    def get_dataset(
        cls,
        dataset_name: str,
        root: str = None,
        channels: list = None,
        sequence_length: int = 21,
        **kwargs,
    ) -> "BasePhysioDataset":
        """Create a dataset with the correct pipeline AND channels for this model.

        Automatically selects:
        - The right resampling pipeline (via ``PIPELINE_PRESET``)
        - The right EEG channels for this dataset (via ``_datasets.py`` registry)
        - The right constructor kwargs (cohort, visit, subset, ...)

        The user doesn't need to know what channels each dataset has or what
        each model expects — ``get_dataset()`` handles both automatically.

        The channel registry mirrors EEGBenchmarks' ``DATASET_REGISTRY`` exactly,
        so the same cache files are reused across both systems.

        Args:
            dataset_name: registered name (hmc, sleepedf, mass, wsc, mesa, ...).
                See ``available_dataset_configs()`` for full list.
            root: override data root. If None, uses the default path from the
                dataset config registry.
            channels: override channel requests. If None (recommended), uses the
                dataset-specific channels from the registry.
            sequence_length: number of epochs per sample (default 21)
            **kwargs: extra args merged with the dataset config's extra_kwargs.

        Returns:
            A BasePhysioDataset instance ready for this model.

        Example::

            # Everything is automatic — pipeline, channels, root:
            dataset = CBraModEncoder.get_dataset("hmc")

            # Override root only:
            dataset = CBraModEncoder.get_dataset("hmc", root="/my/data/hmc")

            # Full manual control:
            dataset = CBraModEncoder.get_dataset(
                "hmc", channels=["EEG C4-M1"], root="/my/data/hmc"
            )
        """
        from physioex.models.foundation_datasets import get_dataset_config

        config = get_dataset_config(dataset_name)
        pipeline = cls.get_pipeline()

        # Import the dataset class
        mod = __import__(config.module_path, fromlist=[config.class_name])
        ds_cls = getattr(mod, config.class_name)

        # Channels: dataset-specific defaults, overridable by user
        effective_channels = channels if channels is not None else config.channels
        effective_root = root if root is not None else config.default_root

        # Merge extra kwargs: dataset config defaults + user overrides
        ds_kwargs = dict(config.extra_kwargs)
        ds_kwargs.update(kwargs)
        ds_kwargs.update(
            root=effective_root,
            channels=effective_channels,
            pipelines=pipeline,
            sequence_length=sequence_length,
        )

        dataset = ds_cls(**ds_kwargs)

        logger.info(
            f"[{cls.MODEL_NAME}] dataset={dataset_name!r}, "
            f"channels={effective_channels} ({len(effective_channels)}ch), "
            f"pipeline={cls.PIPELINE_PRESET!r} (hash={pipeline.hash()}), "
            f"subjects={dataset.get_n_subjects()}"
        )

        return dataset
