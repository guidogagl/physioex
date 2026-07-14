---
orphan: true
---

# Library architecture — overview

This section is the **living class-diagram documentation** of the PhysioEx
library (Diagram A). It is extracted directly from the source and is kept in
sync with the code: every class, its public methods and its relationships
mirror `physioex/`. Each subpackage has its own page:

- [Data](data.md) — datasets, preprocessing pipeline, cache, readers, events.
- [Models](models.md) — foundation encoders, classic architectures, embeddings, pretrained loading.
- [Train](train.md) — training/evaluation loop, metrics, stats, logging, CLIs.
- [Explain](explain.md) — post-hoc attribution, foundational (CSD), prototypes.

## Subsystem map

```mermaid
flowchart LR
    subgraph data["physioex.data"]
        REG["datasets.REGISTRY\nget_dataset()"]
        BPD["BasePhysioDataset (ABC)"]
        PIPE["PreprocessingPipeline\n+ steps"]
        CACHE["ChannelCache"]
        RD["readers (EDF / annotations)"]
    end
    subgraph models["physioex.models"]
        FE["FoundationEncoder (ABC)"]
        ARCH["classic architectures\n(SeqSleepNet, TinySleepNet, ...)"]
        EMB["embed / pretrained"]
    end
    subgraph train["physioex.train"]
        TR["Trainer\n(+ multidevice)"]
        MET["metrics / stats"]
        LOG["Logger (ABC)"]
        CLI["bin: train / finetune / test_model"]
    end
    subgraph explain["physioex.explain"]
        PH["posthoc (gradients / DFT / STFT)"]
        FO["foundational (CSD)"]
        PR["prototypes (NMF / VQ)"]
    end

    RD --> BPD
    PIPE --> BPD
    CACHE --> BPD
    REG --> BPD
    BPD --> TR
    FE --> BPD
    ARCH --> TR
    FE --> TR
    TR --> MET
    TR --> LOG
    CLI --> TR
    TR --> EMB
    ARCH --> PH
    FE --> FO
    EMB --> PR
```

## Principal types and cross-cutting relationships

```mermaid
classDiagram
    class BasePhysioDataset {
        <<abstract>>
        +split(fold)
        +__getitem__(idx)
        #_list_subjects()*
        #_read_subject_labels(spec)*
    }
    class MultiDataset
    class PreprocessingPipeline
    class ChannelCache
    class FoundationEncoder {
        <<abstract>>
        +encode(x)
        +get_dataset(name)
    }
    class Trainer {
        +train(model, dataset)$
        +evaluate(model, dataset)$
        +voting_evaluate(model, dataset)$
    }
    class Logger {
        <<abstract>>
    }

    BasePhysioDataset "1" *-- "1" PreprocessingPipeline : preprocess
    BasePhysioDataset "1" *-- "1" ChannelCache : memmap cache
    MultiDataset o-- BasePhysioDataset : wraps N
    FoundationEncoder ..> BasePhysioDataset : get_dataset()
    Trainer ..> BasePhysioDataset : consumes .split()
    Trainer ..> Logger : logs to
```

## Design patterns used

- **Abstract base classes**: `BasePhysioDataset`, `PreprocessingStep`, `FoundationEncoder`, `Logger`, `SpecificityStrategy` define the extension points.
- **String registry**: `physioex/data/datasets/__init__.py` maps names → dataset classes (`get_dataset`, `available_datasets`); `foundation_checkpoints`/`foundation_datasets` do the same for models.
- **Dynamic factory** `"module.path:ClassName"`: resolved by `physioex/train/bin/_common.py:import_class` and `physioex/train/models/load.py`.
- **Strategy**: `SpecificityStrategy` subclasses in `explain/foundational/specificity.py`; the `CHANNEL_STRATEGY` class attribute on encoders.
- **Template method / adapter**: `FoundationEncoder` wraps heterogeneous pretrained backbones behind a uniform `(B,L,C,T) -> (B,L,D)` `encode`.
- **Content-addressed caching**: `PreprocessingPipeline.hash()` keys the on-disk `ChannelCache`.

## Main flows

- **Data → tensor**: `get_dataset(name)(channels, pipelines, sequence_length)` → `_list_subjects` + header probe + channel resolution → `__getitem__` reads channels, applies the compiled `PreprocessingPipeline` (cached in `ChannelCache`), returns a dict `{signals, labels, channel_order, subject, events, ...}` collated by `dict_collate_fn`.
- **Training**: `Trainer.train(model, dataset)` → `build_dataloaders` (uses `dataset.split(fold)`) → hand-written epoch loop → `evaluate`/`voting_evaluate` → metrics/stats, logged via a `Logger` backend. The three CLIs (`train`/`finetune`/`test_model`) share this path through `bin/_common.py`.
- **Foundation embeddings**: `FoundationEncoder.get_dataset(name)` builds the matching preprocessing → `encode` → `extract_embeddings`/`linear_probe`.
- **Explainability**: post-hoc gradient/DFT/STFT attributions over a trained classifier; `ConceptualSpectralDecomposition` over foundation embeddings; prototype discovery (NMF/VQ) over cached embeddings.