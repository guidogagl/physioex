# Model architectures

Models in PhysioEx are plain `torch.nn.Module` architectures — there is no
Lightning `SleepModule` base class. They fall into two families, both living
under `physioex.models`:

- **Classic sleep-staging architectures** — self-contained networks for supervised
  sleep staging.
- **Foundation encoders** — pretrained backbones adapted to a uniform interface
  (the `FoundationEncoder` family).

All architectures take epoch sequences of 30-second sleep epochs as input. The
preprocessing they expect is expressed through a matching pipeline preset (see
the [Preprocessing](../preprocess.md) page); for foundation encoders this is
declared on the class as `PIPELINE_PRESET`.

For the full class diagram and per-model contract see the
[models architecture](../architecture/library/models.md) page.

## Classic architectures

Each is a `torch.nn.Module` following the common contract
`__init__(n_classes, in_chan, ...)`, `forward(x) -> (B, L, n_classes)` logits,
and (for most) `encode(x)` returning per-epoch features. They are imported from
their own modules and referenced by dotted `module:Class` spec in the CLIs:

| Model | Class spec |
|---|---|
| Chambon 2018 | `physioex.models.chambon2018:Chambon2018Net` |
| TinySleepNet | `physioex.models.tinysleepnet:TinySleepNet` |
| SeqSleepNet | `physioex.models.seqsleepnet:SeqSleepNet` |
| L-SeqSleepNet | `physioex.models.lseqsleepnet:LSeqSleepNet` |
| SleepTransformer | `physioex.models.sleeptransformer:SleepTransformer` |
| Tsinalis CNN | `physioex.models.tsinalis:TsinalisCNN` |
| CoRe-Sleep | `physioex.models.coresleep:CoReSleep` |
| ProtoSleepNet | `physioex.models.protosleepnet:ProtoSleepNet` |

`ProtoSleepNet` additionally provides prototype quantization and factory
constructors from a `SleepTransformer`/`SeqSleepNet` epoch encoder.

## Foundation encoders

The `FoundationEncoder` family wraps heterogeneous pretrained backbones behind a
uniform `encode: (B, L, C, T) -> (B, L, D)` interface, each declaring its
`MODEL_NAME`, `PIPELINE_PRESET` and `CHANNEL_STRATEGY`. The encoders are
re-exported from `physioex.models`: `BENDREncoder`, `BIOTEncoder`,
`CBraModEncoder`, `LaBraMEncoder`, `NeuroLMEncoder`, `REVEEncoder`,
`SJEEncoder`, `SleepFMEncoder`, `TFCEncoder`. Some require the optional
`physioex[foundation]` extra.

## Loading a model

`physioex.train.models.load.load_model` reconstructs a model from a class (or
`"module:Class"` string) plus constructor kwargs, and loads weights from a
checkpoint. It supports Lightning `.ckpt`, the new `.pt` format, and raw
state-dicts, and can look a model up in the built-in registry.

```{eval-rst}
.. autofunction:: physioex.train.models.load.load_model
```

Pretrained models published on the Hugging Face Hub can be reconstructed with
`physioex.models.load_from_pretrained` (see the
[models architecture](../architecture/library/models.md) page and the
[API Reference](../../api/index.md)).

## Adding your own model

Any `torch.nn.Module` that follows the common contract
(`__init__(n_classes, in_chan, ...)` and `forward(x) -> (B, L, n_classes)`) is
compatible with `Trainer` and the CLIs — pass it as a `module:Class` spec to
`--model`. No base class or module wrapper is required.

```{toctree}
:hidden:

networks/chambon2018
```
