---
orphan: true
---

# `physioex.data` — class diagram

The data layer turns raw EDF recordings into model-ready tensors through a
lazy, content-addressed pipeline. Source: `physioex/data/`.

## Dataset hierarchy

```mermaid
classDiagram
    class Dataset {
        <<torch.utils.data>>
    }
    class BasePhysioDataset {
        <<abstract>>
        +str DATASET_NAME
        +dict CHANNEL_PREFERENCES
        +float DEFAULT_EPOCH_LENGTH_SEC
        +__init__(root, channels, pipelines, sequence_length, ...)
        +__len__()
        +__getitem__(idx)
        +split(fold)
        +get_splits(fold)
        +probe(subject)
        +available_channels()
        +get_n_subjects()
        +get_subjects()
        +get_subject_metadata(id)
        +get_subject_events(id)
        +close()
        #_list_subjects()*
        #_read_subject_labels(spec)*
    }
    class _NSRRBaseDataset {
        +float DEFAULT_EPOCH_LENGTH_SEC
    }
    class SubjectSpec {
        +str subject_id
        +Path edf_path
        +Path label_path
        +dict external_meta
    }
    class MultiDataset {
        +__init__(datasets, memmap_cache_size)
        +split(fold)
        +datasets
        +get_n_subjects()
        +available_channels()
        +close()
    }

    Dataset <|-- BasePhysioDataset
    Dataset <|-- MultiDataset
    BasePhysioDataset <|-- _NSRRBaseDataset
    BasePhysioDataset <|-- HMCDataset
    BasePhysioDataset <|-- SleepEDFDataset
    BasePhysioDataset <|-- DCSMDataset
    BasePhysioDataset <|-- MASSDataset
    BasePhysioDataset <|-- WSCDataset
    BasePhysioDataset <|-- STAGESDataset
    BasePhysioDataset <|-- AlzheimersDataset
    BasePhysioDataset <|-- ParkinsonsDataset
    _NSRRBaseDataset <|-- MESADataset
    _NSRRBaseDataset <|-- MrOSDataset
    _NSRRBaseDataset <|-- SHHSDataset
    _NSRRBaseDataset <|-- HPAPDataset
    MultiDataset o-- "N" BasePhysioDataset
    BasePhysioDataset ..> SubjectSpec : yields
```

The 12 concrete datasets are resolved by name from a **string registry** in
`physioex/data/datasets/__init__.py` — `get_dataset(name)` / `available_datasets()`;
keys: `hmc, sleepedf, dcsm, mesa, mros, hpap, wsc, mass, alzheimers, parkinsons, shhs, stages`.

### Legacy (deprecated)

```mermaid
classDiagram
    class PhysioExDataset {
        <<deprecated>>
        +__init__(datasets, preprocessing, selected_channels, seqlen, ...)
        +split(fold)
        +get_scaling()
    }
    class DataReader {
        <<deprecated>>
        +__init__(data_folder, dataset, preprocessing, seqlen, channels_index, offset)
    }
    PhysioExDataset ..> DataReader : reads scaling.npz + table.csv
```

`PhysioExDataset`/`DataReader` (`data/dataset.py`, `data/datareader.py`) are the
old preprocessed-array layer; both emit a `DeprecationWarning` on instantiation
and are not re-exported from `data/__init__`.

## Preprocessing pipeline

```mermaid
classDiagram
    class PreprocessingStep {
        <<abstract>>
        +spec()*
        +compile(fs_in)*
    }
    class CompiledStep {
        +Callable apply
        +float fs_out
    }
    class PreprocessingPipeline {
        +__init__(steps)
        +spec()
        +hash()
        +compile(fs_in)
    }
    class CompiledPipeline {
        +__init__(steps, fs_out)
        +__call__(signal)
    }

    PreprocessingStep <|-- Identity
    PreprocessingStep <|-- BandpassFilter
    PreprocessingStep <|-- HighPassFilter
    PreprocessingStep <|-- NotchFilter
    PreprocessingStep <|-- Resample
    PreprocessingStep <|-- ZScoreNormalize
    PreprocessingStep <|-- XSleepNetSpectrogram
    PreprocessingPipeline o-- "N" PreprocessingStep
    PreprocessingStep ..> CompiledStep : compile()
    PreprocessingPipeline ..> CompiledPipeline : compile(fs_in)
```

Named presets (`data/presets.py`) build ready pipelines: `get_preset(name)` /
`available_presets()` — e.g. `raw`, `time_domain`, `time_frequency`,
`seqsleepnet`, per-modality (`eeg`/`eog`/`emg`/`ecg`) and per foundation model.

## Cache, readers, events, support

```mermaid
classDiagram
    class ChannelCache {
        +str SCHEMA_VERSION
        +signal_path(dataset, subject, physical, hash)
        +atomic_save_array(path, data, meta)
        +load_memmap(path)
        +clear(dataset, subject, hash)
    }
    class EDFHeader {
        +list available_channels
        +dict channel_fs
        +dict patient_meta
        +float duration_sec
        +to_dict()
        +from_dict(d)$
    }
    class ResolvedChannel {
        +str request
        +str physical
        +ModalityType modality
        +float fs_in
        +bool is_differential
    }
    class ChannelNotAvailableError
    class SleepEvent {
        +str type
        +str concept
        +float onset_sec
        +float duration_sec
        +dict extra
    }
    class ModalityType {
        <<IntEnum>>
        EEG EOG EMG ECG RESP SPO2 HR ...
    }
    ValueError <|-- ChannelNotAvailableError
    BasePhysioDataset ..> ChannelCache : caches signals
    BasePhysioDataset ..> EDFHeader : probe()
    BasePhysioDataset ..> ResolvedChannel : channel resolution
    BasePhysioDataset ..> SleepEvent : get_subject_events()
    ResolvedChannel ..> ModalityType
```

## Class reference (responsibilities)

- **`BasePhysioDataset` (ABC, `base.py`)** — central `torch.utils.data.Dataset`.
  Concrete datasets implement two hooks: `_list_subjects()` (discover subjects →
  `SubjectSpec`) and `_read_subject_labels(spec)` (hypnogram → AASM int array).
  The base handles: header probing + channel resolution, split logic
  (`split`/`get_splits`, 70/15/15 by seed `42+fold`), excess-wake trimming,
  label sanitization, event loading, memmap caching, and `__getitem__` assembling
  the batch dict (`signals, channel_order, channel_info, labels, subject,
  epoch_indices, recording_length, events`). Module fn `get_data_root()` reads
  `$PHYSIOEX_DATA`.
- **`SubjectSpec`** — lightweight locator (`subject_id`, `edf_path`, `label_path`,
  `external_meta`); returned by `_list_subjects`.
- **Concrete datasets** — each sets `DATASET_NAME`, `DATASET_SUBDIR`,
  `CHANNEL_PREFERENCES`, `DEFAULT_EPOCH_LENGTH_SEC`, and constructor extras where
  relevant: `MASSDataset(cohort)`, `SHHSDataset(visit)`, `WSCDataset(visit)`,
  `STAGESDataset(site, recording)`, `AlzheimersDataset(subset)`,
  `ParkinsonsDataset(recording, group)`, `HPAPDataset(subset)`. The four NSRR
  datasets subclass `_NSRRBaseDataset` (shared XML annotation handling).
- **`MultiDataset` (`multi.py`)** — concatenates N `BasePhysioDataset` into one
  flat index with coordinated `split(fold)` (returns `(dataset_idx, subject_id)`
  tuples for valid/test), proportional cache distribution, unified `__getitem__`.
- **`PreprocessingStep` (ABC) / `PreprocessingPipeline` / `Compiled*`
  (`pipeline.py`, `steps/`)** — each step exposes a canonical `spec()` (hashable)
  and `compile(fs_in) -> CompiledStep`. The pipeline hashes to a `pipeline_hash`
  used as the cache key. Steps: `Identity, BandpassFilter, HighPassFilter,
  NotchFilter, Resample, ZScoreNormalize, XSleepNetSpectrogram`.
- **`ChannelCache` (`cache.py`)** — on-disk memmap cache keyed by
  `(SCHEMA_VERSION, dataset, subject, physical_channel, pipeline_hash)` with atomic
  writes; bf16 support via `ml_dtypes` (`recommended_dtype`, `cast_to_cache_dtype`).
- **Readers (`readers/edf.py`, `readers/annotations.py`)** — `probe_edf_header`,
  `resolve_channels` (+ `EDFHeader`, `ResolvedChannel`, `ChannelNotAvailableError`,
  `DEFAULT_PREFERENCES`), and annotation parsers `parse_nsrr_xml`,
  `parse_tsv_annotations`, `parse_nsrr_xml_events`, `parse_stages_csv`.
- **Events (`events.py`)** — `SleepEvent` dataclass + `map_events_to_epochs`,
  `events_to_dicts`/`dicts_to_events` (round-trippable, cached as JSON).
- **Collate (`collate.py`)** — `dict_collate_fn` (deterministic channel-order by
  sorted union — no per-item validation), `stack_channels`, `is_dict_batch`.
- **Modality (`modality.py`)** — `ModalityType` IntEnum + `infer_channel_modality`.