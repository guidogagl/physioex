# API Reference

This reference is generated automatically from the `physioex` docstrings, so it
always mirrors the installed code (the `physioex-dev` API). For the high-level
picture of how these pieces fit together, see the
[library architecture](../pages/architecture/library/overview.md) pages.

## `physioex.data`

The raw-EDF, lazy-loading dataset layer, the preprocessing pipeline, the
on-disk cache, EDF/annotation readers and sleep-event helpers.

```{eval-rst}
.. currentmodule:: physioex.data
.. autosummary::
   :toctree: _generated
   :nosignatures:

   BasePhysioDataset
   MultiDataset
   SubjectSpec
   PreprocessingStep
   PreprocessingPipeline
   CompiledPipeline
   CompiledStep
   Identity
   BandpassFilter
   NotchFilter
   Resample
   ZScoreNormalize
   XSleepNetSpectrogram
   get_preset
   available_presets
   ChannelCache
   dict_collate_fn
   stack_channels
   is_dict_batch
   EDFHeader
   ResolvedChannel
   probe_edf_header
   resolve_channels
   SleepEvent
   map_events_to_epochs
   events_to_dicts
   dicts_to_events
```

Datasets are resolved by name from the registry:

```{eval-rst}
.. currentmodule:: physioex.data.datasets
.. autosummary::
   :toctree: _generated
   :nosignatures:

   get_dataset
   available_datasets
```

## `physioex.models`

Foundation encoders (a uniform `(B, L, C, T) -> (B, L, D)` wrapper around
heterogeneous pretrained backbones), embedding/pretrained helpers, and the
classic sleep-staging architectures.

```{eval-rst}
.. currentmodule:: physioex.models
.. autosummary::
   :toctree: _generated
   :nosignatures:

   FoundationEncoder
   CBraModEncoder
   BENDREncoder
   LaBraMEncoder
   BIOTEncoder
   SleepFMEncoder
   TFCEncoder
   REVEEncoder
   SJEEncoder
   NeuroLMEncoder
   load_from_pretrained
   extract_embeddings
   load_embeddings
   linear_probe
```

Classic architectures:

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   physioex.models.chambon2018.Chambon2018Net
```

## `physioex.train`

A hand-written PyTorch training/evaluation loop (not Lightning), with metric and
statistics helpers, pluggable logging, and the three console-script entry points.

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   physioex.train.trainer.Trainer
   physioex.train.trainer.seed_everything
   physioex.train.multidevicetrainer.MultiDeviceTrainer
   physioex.train.logger.Logger
   physioex.train.logger.NoOpLogger
   physioex.train.logger.build_logger
   physioex.train.models.load.load_model
```

Metrics, statistics and progress helpers are documented at module level:

```{eval-rst}
.. autosummary::
   :toctree: _generated

   physioex.train.metrics
   physioex.train.stats
   physioex.train.progress
```

## `physioex.explain`

Post-hoc attribution, the foundational Conceptual Spectral Decomposition (CSD),
and prototype discovery.

```{eval-rst}
.. autosummary::
   :toctree: _generated

   physioex.explain.foundational
   physioex.explain.prototypes.posthoc
```
