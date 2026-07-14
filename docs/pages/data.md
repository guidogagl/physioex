# Data

The `physioex.data` module turns raw EDF recordings into model-ready tensors
through a **lazy, content-addressed pipeline**. There is no more preprocessed
"array" layer and no Lightning data-module: datasets read raw EDF on demand,
preprocess per channel on the fly, and cache the result to disk.

The public entry points are:

- **`physioex.data.BasePhysioDataset`** — the abstract `torch.utils.data.Dataset`
  base that all concrete datasets inherit from.
- **`physioex.data.MultiDataset`** — concatenates several datasets into one.
- **`physioex.data.dict_collate_fn`** — the collate function that batches the
  dict samples for a `DataLoader`.

Concrete datasets are resolved **by name** from a string registry rather than
imported directly. For the full class diagram and how the pieces fit together,
see the [data-layer architecture](architecture/library/data.md) page.

## Resolving a dataset by name

`physioex.data.datasets` exposes a registry of the 12 concrete datasets:

```python
from physioex.data.datasets import get_dataset, available_datasets

available_datasets()
# ['alzheimers', 'dcsm', 'hmc', 'hpap', 'mass', 'mesa', 'mros',
#  'parkinsons', 'shhs', 'sleepedf', 'stages', 'wsc']

HMCDataset = get_dataset("hmc")   # the class, not an instance
```

## Instantiating a dataset

Every concrete dataset shares the `BasePhysioDataset` constructor. The most
common arguments are the data `root` (falls back to the `PHYSIOEX_DATA`
environment variable when omitted), the `channels` to load, a preprocessing
`pipelines` preset, and the `sequence_length`:

```python
from physioex.data.datasets import get_dataset

data = get_dataset("hmc")(
    root=None,                     # None -> uses $PHYSIOEX_DATA
    channels=["EEG", "EOG", "EMG"],
    pipelines="time_domain",       # a preset name (see the Preprocessing page)
    sequence_length=21,
)

len(data)              # number of indexable epoch-sequences
data.get_n_subjects()  # number of subjects discovered
data.available_channels()
```

Each item is a **dict** (not a `(signal, label)` tuple). The keys are:

| Key | Meaning |
|---|---|
| `signals` | stacked channel tensor for the sequence |
| `channel_order` | ordered list of channel names |
| `channel_info` | per-channel resolution metadata |
| `labels` | per-epoch AASM labels (`-1` = unscored) |
| `subject` | subject metadata dict (incl. `dataset_idx` for `MultiDataset`) |
| `epoch_indices` | indices of the epochs in the recording |
| `recording_length` | number of epochs in the full recording |
| `events` | sleep events overlapping the sequence |

```python
item = data[0]
item["signals"].shape   # (sequence_length, n_channels, ...)
item["labels"].shape    # (sequence_length,)
```

The trailing dimensions of `signals` depend on the preprocessing preset:
time-domain presets yield `(L, C, T)` samples, while spectrogram presets
(`seqsleepnet`, `time_frequency`) yield `(L, C, T, F)`.

## Splitting

`BasePhysioDataset.split(fold)` and `get_splits(fold)` produce reproducible
train/validation/test subject splits (70/15/15, seeded by `42 + fold`). See the
API reference for the exact return types.

## Merging datasets

`MultiDataset` concatenates several `BasePhysioDataset` instances into a single
flat index with coordinated splitting. All constituent datasets must share the
same `sequence_length`.

```python
from physioex.data.datasets import get_dataset
from physioex.data import MultiDataset

hmc = get_dataset("hmc")(channels=["EEG"], pipelines="raw", sequence_length=21)
dcsm = get_dataset("dcsm")(channels=["EEG"], pipelines="raw", sequence_length=21)

merged = MultiDataset([hmc, dcsm])
```

## Building a DataLoader

Because items are dicts, use `dict_collate_fn` as the loader's `collate_fn`:

```python
from torch.utils.data import DataLoader
from physioex.data import dict_collate_fn

loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=dict_collate_fn)
batch = next(iter(loader))
batch["signals"].shape   # (32, sequence_length, n_channels, ...)
```

:::{admonition} Training
:class: tip
For actual training and evaluation you rarely wire the `DataLoader` yourself:
`physioex.train.trainer.Trainer.build_dataloaders` detects a
`BasePhysioDataset`/`MultiDataset` and attaches `dict_collate_fn` automatically.
See the [Training](train/train.md) page and the
[train CLIs](train/cli.md).
:::

## API

```{eval-rst}
.. autoclass:: physioex.data.BasePhysioDataset
   :no-index:
   :members: __getitem__, __len__, split, get_splits, probe,
             available_channels, get_n_subjects, get_subjects,
             get_subject_events, close
   :show-inheritance:

.. autoclass:: physioex.data.MultiDataset
   :no-index:
   :members: split, available_channels, get_n_subjects, close
   :show-inheritance:

.. autofunction:: physioex.data.dict_collate_fn
   :no-index:
```

See the full auto-generated [API Reference](../api/index.md) for the complete
`physioex.data` surface (readers, cache, events, modality helpers).
