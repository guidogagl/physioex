# `physioex.models` — class diagram

Two model families — **foundation encoders** (pretrained backbones adapted to a
uniform interface) and **classic sleep-staging architectures** — plus embedding
extraction, pretrained loading and the VQ sleep tokenizer. Source: `physioex/models/`.

## Foundation encoders

```mermaid
classDiagram
    class Module {
        <<torch.nn>>
    }
    class FoundationEncoder {
        <<abstract>>
        +str MODEL_NAME
        +str PIPELINE_PRESET
        +str CHANNEL_STRATEGY
        +__init__(in_chan, checkpoint_path, **kwargs)
        +forward(x)
        +encode(x)
        +get_pipeline()$
        +get_dataset(name, root, channels, sequence_length)$
    }
    Module <|-- FoundationEncoder
    FoundationEncoder <|-- BENDREncoder
    FoundationEncoder <|-- BIOTEncoder
    FoundationEncoder <|-- CBraModEncoder
    FoundationEncoder <|-- LaBraMEncoder
    FoundationEncoder <|-- NeuroLMEncoder
    FoundationEncoder <|-- REVEEncoder
    FoundationEncoder <|-- SJEEncoder
    FoundationEncoder <|-- SleepFMEncoder
    FoundationEncoder <|-- TFCEncoder
```

Every encoder declares `MODEL_NAME`, `PIPELINE_PRESET` (the matching preprocessing
preset) and `CHANNEL_STRATEGY`, and adapts a heterogeneous backbone to
`encode: (B,L,C,T) -> (B,L,D)`. Support modules:

- `foundation_base.py` — the ABC + `get_pipeline()`/`get_dataset()` template.
- `foundation_datasets.py` — `DatasetConfig` + `get_dataset_config`/`available_dataset_configs` (PHYSIOEX_DATA-relative roots, channel maps).
- `foundation_checkpoints.py` — `CheckpointSource` + `ensure_checkpoint`, `ensure_reve_positions`, `get_checkpoint_dir`, cache helpers.
- `foundation_channel_mapping.py` — `pad_or_repeat_channels`, `map_to_bendr_layout`.
- `foundation_preproc.py` — ~18 differentiable normalization/channel ops (`mean_center`, `zscore*`, `q95_normalize`, `strip_zero_channels`, `compute_channel_stats`, ...).

## Classic architectures

```mermaid
classDiagram
    class Module {
        <<torch.nn>>
    }
    class SeqSleepNet {
        +__init__(n_classes, in_chan, F, D, nfft, sf, ...)
        +encode(x)
        +forward(x)
    }
    class TinySleepNet {
        +__init__(n_classes, in_chan, ...)
        +encode(x)
        +forward(x)
    }
    class SleepTransformer {
        +__init__(n_classes, in_chan, d_model, ...)
        +encode(x)
        +forward(x)
    }
    class LSeqSleepNet {
        +encode(x)
        +forward(x)
    }
    class Chambon2018Net {
        +encode(x)
        +forward(x)
    }
    class TsinalisCNN {
        +forward(x)
    }
    class CoReSleep {
        +encode(x)
        +forward(x)
    }
    class ProtoSleepNet {
        +encode(x, quantize)
        +forward(x, quantize)
        +from_sleep_transformer(...)$
        +from_seq_sleep_net(...)$
    }
    Module <|-- SeqSleepNet
    Module <|-- TinySleepNet
    Module <|-- SleepTransformer
    Module <|-- LSeqSleepNet
    Module <|-- Chambon2018Net
    Module <|-- TsinalisCNN
    Module <|-- CoReSleep
    Module <|-- ProtoSleepNet
```

Common contract: `__init__(n_classes, in_chan, ...)`, `forward(x)` and (most)
`encode(x)`. `ProtoSleepNet` adds prototype quantization and factory
constructors from `SleepTransformer`/`SeqSleepNet` epoch encoders; its training
uses `ProtoSleepNetTrainer(Trainer)`.

## Sleep tokenizer & helpers

```mermaid
classDiagram
    class Module { <<torch.nn>> }
    class ChannelUNet
    class SAB
    class PMA
    class PrototypicalClassifier {
        +update_centroids(embeddings, labels)
        +forward(embeddings)
    }
    class SleepTokenizer {
        +__init__(n_classes, d_model, T, sf, ...)
        +forward(x, modality_ids)
        +encode(x, modality_ids)
        +fit_prototypes(embeddings, K)
        +tokenize(x, modality_ids)
        +get_prototypes()
    }
    Module <|-- ChannelUNet
    Module <|-- SAB
    Module <|-- PMA
    Module <|-- PrototypicalClassifier
    Module <|-- SleepTokenizer
    SleepTokenizer *-- ChannelUNet
    SleepTokenizer *-- SAB
    SleepTokenizer *-- PMA
    SleepTokenizer *-- PrototypicalClassifier
```

## Embedding & pretrained API (module-level)

- `embed.py` — `extract_embeddings(model, dataset, ...)`, `load_embeddings(model_name, dataset_name, ...)`, `linear_probe(model_name, dataset_name, n_folds, ...)` (HF-backed embedding cache).
- `pretrained.py` — `load_from_pretrained(name, device, repo_id)` reconstructs any model from a HF `config.json` (`model_class` + `model_kwargs`) in `4rooms/physioex`.
- `models/__init__.py` re-exports `load_from_pretrained`, the embed helpers, `FoundationEncoder` and the 9 encoders. Classic architectures are imported from their own modules.

## Class reference (responsibilities)

- **`FoundationEncoder` (ABC)** — uniform adapter over pretrained backbones; subclasses only build/load the backbone and rely on the base for `forward`, `get_pipeline`, `get_dataset` (which resolves a `DatasetConfig` and builds the matching `BasePhysioDataset`). The `transformers` import in `REVEEncoder` is lazy (`physioex[foundation]`).
- **Classic architectures** — self-contained `nn.Module`s for sleep staging; `encode` returns per-epoch features, `forward` returns `(B, L, n_classes)` logits.
- **`SleepTokenizer`** — VQ tokenizer: `ChannelUNet` feature extractor + set-transformer blocks (`SAB`/`PMA`) + `PrototypicalClassifier`; `fit_prototypes`/`tokenize` produce discrete codes. Helpers `infer_modality`, `build_modality_ids`, `modality_dropout`.
- **Support modules** — checkpoint/dataset/channel/preproc utilities that keep encoders declarative.
