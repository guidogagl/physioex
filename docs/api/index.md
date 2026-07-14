# API Reference

This reference is generated automatically from the `physioex` docstrings, so it
always mirrors the installed code (the `physioex-dev` API). For the high-level
picture of how these pieces fit together, see the
[library architecture](../pages/architecture/library/overview.md) pages.

## `physioex.data`

The raw-EDF, lazy-loading dataset layer, the preprocessing pipeline, the
on-disk cache, EDF/annotation readers and sleep-event helpers.

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   physioex.data.BasePhysioDataset
   physioex.data.MultiDataset
   physioex.data.SubjectSpec
   physioex.data.PreprocessingStep
   physioex.data.PreprocessingPipeline
   physioex.data.CompiledPipeline
   physioex.data.CompiledStep
   physioex.data.Identity
   physioex.data.BandpassFilter
   physioex.data.NotchFilter
   physioex.data.Resample
   physioex.data.ZScoreNormalize
   physioex.data.XSleepNetSpectrogram
   physioex.data.get_preset
   physioex.data.available_presets
   physioex.data.ChannelCache
   physioex.data.dict_collate_fn
   physioex.data.stack_channels
   physioex.data.is_dict_batch
   physioex.data.EDFHeader
   physioex.data.ResolvedChannel
   physioex.data.probe_edf_header
   physioex.data.resolve_channels
   physioex.data.SleepEvent
   physioex.data.map_events_to_epochs
   physioex.data.events_to_dicts
   physioex.data.dicts_to_events
```

Datasets are resolved by name from the registry:

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   physioex.data.datasets.get_dataset
   physioex.data.datasets.available_datasets
```

## `physioex.models`

Foundation encoders (a uniform `(B, L, C, T) -> (B, L, D)` wrapper around
heterogeneous pretrained backbones), embedding/pretrained helpers, and the
classic sleep-staging architectures.

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   physioex.models.FoundationEncoder
   physioex.models.CBraModEncoder
   physioex.models.BENDREncoder
   physioex.models.LaBraMEncoder
   physioex.models.BIOTEncoder
   physioex.models.SleepFMEncoder
   physioex.models.TFCEncoder
   physioex.models.REVEEncoder
   physioex.models.SJEEncoder
   physioex.models.NeuroLMEncoder
   physioex.models.load_from_pretrained
   physioex.models.extract_embeddings
   physioex.models.load_embeddings
   physioex.models.linear_probe
```

Classic architectures:

```{eval-rst}
.. autosummary::
   :toctree: _generated
   :nosignatures:

   physioex.models.chambon2018.Chambon2018Net
   physioex.models.tinysleepnet.TinySleepNet
   physioex.models.seqsleepnet.SeqSleepNet
   physioex.models.sleeptransformer.SleepTransformer
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
   physioex.train.multidevicetrainer.Trainer
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
