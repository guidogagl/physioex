"""Base class for foundation model wrappers.

Adapts a pretrained single-epoch encoder to the test/ training system's
sequence-based interface: (B, L, C, T) -> (B, L, n_classes).

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

    from physioex.models.foundation import get_foundation_model

    model_cls = get_foundation_model("cbramod")

    # get_dataset handles everything: pipeline, channels, root path, extra kwargs
    dataset = model_cls.get_dataset("hmc")

    # in_chan = number of channels the dataset provides
    model = model_cls(n_classes=5, in_chan=len(dataset.channels))

    # Train
    from physioex.train.trainer import Trainer
    Trainer.train(model, dataset, max_epochs=20)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("physioex.foundation")


class FoundationModelWrapper(nn.Module):
    """Abstract adapter: frozen encoder + trainable classification head.

    Subclasses implement:
      - ``_build_encoder(**kwargs)``  -> nn.Module (the pretrained encoder)
      - ``_get_embedding_dim()``      -> int
      - ``_load_pretrained(path)``    -> None (load checkpoint into self.encoder)
      - ``_encode(x)``               -> (B, D) embeddings from (B, C', T') input
      - ``_preprocess(x)``           -> (B, C', T') (optional: resampling, channel mapping)

    The wrapper handles:
      - Reshaping (B, L, C, T) -> (B*L, C, T) before encoding
      - Freezing all encoder parameters
      - Trainable LayerNorm + Linear head
      - Reshaping output back to (B, L, n_classes)
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
        n_classes: int,
        in_chan: int,
        sequence_length: int = 1,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.in_chan = in_chan
        self.sequence_length = sequence_length

        # Subclass builds the encoder
        self.encoder = self._build_encoder(in_chan=in_chan, **kwargs)
        self.embedding_dim = self._get_embedding_dim()

        # Load pretrained weights
        self._load_pretrained(checkpoint_path)

        # Freeze encoder — only the head trains
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Trainable classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, n_classes),
        )

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_head = sum(p.numel() for p in self.head.parameters())
        logger.info(
            f"[{self.MODEL_NAME}] encoder={n_enc:,} params (frozen), "
            f"head={n_head:,} params (trainable)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C, T) -> (B, L, n_classes)"""
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        x = self._preprocess(x)
        with torch.no_grad():
            emb = self._encode(x)  # (B*L, D)
        logits = self.head(emb)  # (B*L, n_classes)
        return logits.reshape(B, L, -1)

    def train(self, mode: bool = True):
        """Override: encoder stays in eval mode even when the wrapper trains."""
        super().train(mode)
        self.encoder.eval()
        return self

    # ── Probe loading ───────────────────────────────────────────────

    def load_probe(self, probe_path: str) -> None:
        """Load a trained linear probe checkpoint into this model's head.

        Accepts checkpoints saved by the Trainer (``model_state_dict`` key)
        or raw state dicts.

        Args:
            probe_path: path to a ``.pt`` checkpoint file.

        Example::

            model = CBraModSleepNet(n_classes=5, in_chan=4)
            model.load_probe("checkpoints/train_1/epoch=16-step=724-val_acc=0.71%.pt")
            # model.head now contains the trained weights
        """
        import torch

        ckpt = torch.load(str(probe_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        # Strip "head." prefix if present (Trainer saves full model keys)
        stripped = {}
        for k, v in state.items():
            stripped[k.removeprefix("head.")] = v
        self.head.load_state_dict(stripped, strict=True)
        logger.info(f"[{self.MODEL_NAME}] Loaded probe from {probe_path}")

    @classmethod
    def from_probe(
        cls,
        dataset_name: str,
        probe_path: str,
        checkpoint_path: str = None,
        root: str = None,
        n_classes: int = 5,
        **kwargs,
    ) -> "FoundationModelWrapper":
        """Create a model with a pre-trained probe head loaded.

        Convenience method that combines model creation + probe loading
        in one call. The model is ready for inference.

        Args:
            dataset_name: dataset slug (used to determine ``in_chan``)
            probe_path: path to the trained probe ``.pt`` checkpoint
            checkpoint_path: encoder checkpoint path (if required)
            root: data root override
            n_classes: number of classes

        Returns:
            Model with frozen encoder + trained head, ready for inference.

        Example::

            model = CBraModSleepNet.from_probe(
                "hmc",
                probe_path="checkpoints/train_1/epoch=16-step=724-val_acc=0.71%.pt",
            )
            # Ready for inference: model(x) uses trained head
        """
        from physioex.models.foundation._datasets import get_dataset_config

        config = get_dataset_config(dataset_name)
        model = cls(
            n_classes=n_classes,
            in_chan=len(config.channels),
            checkpoint_path=checkpoint_path,
            channel_names=list(config.channels),
            channel_map=config.channel_map,
            **kwargs,
        )
        model.load_probe(probe_path)
        model.eval()
        return model

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
            dataset = CBraModSleepNet.get_dataset("hmc")

            # Override root only:
            dataset = CBraModSleepNet.get_dataset("hmc", root="/my/data/hmc")

            # Full manual control:
            dataset = CBraModSleepNet.get_dataset(
                "hmc", channels=["EEG C4-M1"], root="/my/data/hmc"
            )
        """
        from physioex.models.foundation._datasets import get_dataset_config

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

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        n_classes: int = 5,
        checkpoint_path: str = None,
        root: str = None,
        channels: list = None,
        sequence_length: int = 21,
        **kwargs,
    ) -> tuple:
        """Create both the dataset AND the model, wired together correctly.

        This is the recommended entry point. It ensures that:
        - The dataset uses the right pipeline and channels
        - The model receives the right ``in_chan`` count
        - Channel metadata (names) is forwarded to models that need it (BENDR)

        Returns:
            ``(model, dataset)`` tuple, ready for ``Trainer.train(model, dataset)``.

        Example::

            model, dataset = CBraModSleepNet.from_dataset("hmc", n_classes=5)
            Trainer.train(model, dataset, max_epochs=20)
        """
        from physioex.models.foundation._datasets import get_dataset_config

        config = get_dataset_config(dataset_name)
        effective_channels = channels if channels is not None else config.channels

        dataset = cls.get_dataset(
            dataset_name,
            root=root,
            channels=channels,
            sequence_length=sequence_length,
            **kwargs,
        )

        # Resolve actual channel names from the dataset (what was resolved
        # from the EDF headers). These are needed by BENDR for alias mapping.
        # For most models this is ignored.
        resolved_names = list(dataset.channels)

        model_kwargs = dict(
            n_classes=n_classes,
            in_chan=len(effective_channels),
            sequence_length=sequence_length,
            checkpoint_path=checkpoint_path,
            channel_names=resolved_names,
            channel_map=config.channel_map,
        )
        model = cls(**model_kwargs)

        return model, dataset
