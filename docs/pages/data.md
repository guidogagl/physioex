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

`physioex.data.datasets` exposes a registry of the 15 concrete datasets:

```python
from physioex.data.datasets import get_dataset, available_datasets

available_datasets()
# ['alzheimers', 'ccshs', 'cfs', 'chat', 'dcsm', 'hmc', 'hpap', 'mass', 'mesa',
#  'mros', 'parkinsons', 'shhs', 'sleepedf', 'stages', 'wsc']

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

## Whole nights and cardiorespiratory signals

With `sequence_length=0` (or any value `<= 0`) an item is one whole night, and
item `i` is subject `i`. Channels a subject lacks are zero-filled and flagged
with `channel_info[key]["available"] = False`. This mode is used by the
cardiorespiratory models (wav2sleep, Jones et al. 2024, SleepPPG-Net), which
stage the full night at once.

- **Channels.** Besides EEG/EOG/EMG, the NSRR datasets resolve `ECG`, `PPG`
  (its own modality, no longer grouped with SpO2) and the respiratory
  effort belts `ABD` and `THX`.
- **Labels.** `label_transform="aasm4"` maps the AASM labels to four classes
  (W, Light = N1+N2, Deep = N3, REM); `-1` stays unscored. The default keeps
  the five AASM classes.
- **Visits.** `shhs`, `mros` and `wsc` take a `visit` argument; each visit is
  a separate dataset instance with its own cache.
- **Subject metadata.** NSRR datasets return
  `{"age", "sex", "recording_start"}` from `subject_metadata(subject_id)`.
- **Published splits.** Besides the 70/15/15 folds, `get_splits(split=...)`
  returns the lists used by published work: `"wav2sleep"` (the per-dataset
  train/validation/test sessions of Carter & Tarassenko, 2024),
  `"wav2sleep-census"` (the census-balanced held-out set of Jones et al., 2024)
  and `"sleepppgnet-replica"` (the 204 MESA test nights of the SleepPPG-Net
  replica).

```python
from physioex.data.datasets import get_dataset
from physioex.train.replication import iter_nights

data = get_dataset("shhs")(
    channels=["ECG", "THX", "ABD"],
    pipelines="wav2sleep",
    sequence_length=0,
    label_transform="aasm4",
    trim_excess_wake=False,
    visit=2,
)
test = data.get_splits(split="wav2sleep")[2]
for subject, signals, labels, metadata in iter_nights(data, test):
    signals["ECG"].shape   # (n_epochs, 1024)
```

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

### Balancing sites with very different sizes

When the merged datasets differ a lot in size, uniform sampling lets the largest one dominate. `make_site_sampler`
gives every *site* the same total probability mass and, inside a site, every *subject* the same mass:
`p(window) ∝ 1 / (N_s · W_u)` with `N_s` the number of training subjects of the site and `W_u` the number of windows of
the subject. Several wrapped datasets can form one site (for example the visits of one cohort): give them the same
name in `site_of_dataset`.

```python
from torch.utils.data import DataLoader, Subset
from physioex.data import dict_collate_fn, make_site_sampler

train_idx = ...                                    # flat MultiDataset indices of the training windows
sampler, report = make_site_sampler(
    merged, train_idx, site_of_dataset=["hmc", "dcsm"], num_samples=100_000, seed=0
)
loader = DataLoader(Subset(merged, train_idx), batch_size=32, sampler=sampler, collate_fn=dict_collate_fn)
report["hmc"]   # {'subjects': ..., 'windows': ..., 'mass': 0.5}
```

`site_balanced_weights(merged, train_idx, site_of_dataset)` returns the weights (summing to 1) if you want to build
your own sampler.

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
