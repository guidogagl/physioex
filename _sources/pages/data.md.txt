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

## Supported datasets

PhysioEx reads the original recordings; obtain them from the source of each
dataset and point `PHYSIOEX_DATA` at the directory that holds them.

### Public datasets

- **Sleep-EDF Expanded** (`sleepedf`, [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)).
  197 whole-night polysomnographic recordings with EEG, EOG, chin EMG and event markers.
- **HMC, Haaglanden Medisch Centrum** (`hmc`, [PhysioNet](https://physionet.org/content/hmc-sleep-staging/1.1/)).
  151 whole-night recordings from 85 men and 66 women: four EEG channels (F4/M1, C4/M1, O2/M1, C3/M2),
  two EOG channels (E1/M2, E2/M2) and one bipolar chin EMG, all at 256 Hz.
- **DCSM, Danish Center for Sleep Medicine** (`dcsm`, [ERDA](https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/published-archive.html)).
  255 randomly selected, fully anonymized overnight lab recordings of patients referred for non-specific
  sleep disorders; EEG, EOG and EMG at 256 Hz.

### NSRR datasets

Available from the [National Sleep Research Resource](https://sleepdata.org) after a data access request.

- **SHHS, Sleep Heart Health Study** (`shhs`). A multi-center cohort on the cardiovascular consequences of
  sleep-disordered breathing: 5,793 participants aged 40 or older at visit 1, recorded at home. C3/A2 and
  C4/A1 EEG at 125 Hz, left and right EOG at 50 Hz, bipolar submental EMG at 125 Hz.
- **MESA, Multi-Ethnic Study of Atherosclerosis** (`mesa`). 2,237 ethnically diverse men and women aged
  45 to 84 from six US communities, recorded at home: central C4-M1 EEG, bilateral EOG and chin EMG at
  256 Hz, scored by certified polysomnologists.
- **MrOS, Osteoporotic Fractures in Men Study** (`mros`). 2,911 home recordings of men aged 65 or older
  from six clinical centers: C3/A2 and C4/A1 EEG, chin EMG and left and right EOG at 256 Hz. The `visit`
  argument selects the visit.
- **WSC, Wisconsin Sleep Cohort** (`wsc`). A longitudinal study of sleep disorders with in-lab recordings
  every four years from a baseline sample of 1,500 participants: C3/M2 EEG, EMG and left and right EOG
  at 200 Hz. The `visit` argument selects the visit.
- **STAGES, Stanford Technology Analytics and Genomics in Sleep** (`stages`). 1,914 recordings from 13
  clinical sites with a bipolar EEG montage (C4-M1 / C3-M2 equivalent), EOG and chin EMG, plus respiratory,
  arousal and limb-movement events. Selectable by site and by recording (`first`, `second`, `all`).
- **CCSHS, Cleveland Children's Sleep and Health Study** (`ccshs`). 515 children and adolescents with one
  home recording each (NSRR `ccshs-trec`): C3/A2 and C4/A1 EEG and LOC/ROC EOG at 128 Hz, chin EMG at
  256 Hz; stages scored with Rechtschaffen & Kales and merged to N3. Eleven recordings whose hypnogram
  does not match the NSRR summary table are excluded by default (`exclude_subjects=[]` keeps them).
- **CFS, Cleveland Family Study** (`cfs`). The visit-5 in-lab recordings of a family-based study of sleep
  apnea.
- **CHAT, Childhood Adenotonsillectomy Trial** (`chat`). Recordings of children at baseline and follow-up,
  plus the non-randomized baseline group.
- **HomePAP** (`hpap`). A positive-airway-pressure study with three subsets, `lab-full`, `lab-split` and
  `home`, selected with the `subset` argument.

The `cfs` and `chat` readers follow the other NSRR readers but have not yet been
tested on the real recordings.

### Other datasets

- **MASS, Montreal Archive of Sleep Studies** (`mass`, [CEAMS](http://ceams-carsm.ca/mass/)). 200 complete
  lab nights from 97 men and 103 women aged 18 to 76, at 256 Hz, with 4 to 20 EEG channels, EOG and EMG.
- **Alzheimer's disease dataset** (`alzheimers`, UZ Leuven). 69 participants (37 with Alzheimer's disease,
  32 healthy controls), one night each: average-reference EEG, differential EOG, chin EMG and ECG at 200 Hz.
  The `subset` argument selects patients or controls.
- **Parkinson's disease dataset** (`parkinsons`, UZ Leuven). 87 participants (48 with Parkinson's disease,
  40 healthy older adults) with up to 157 recordings (night and nap): bipolar EEG, EOG, chin EMG and ECG
  at 500 Hz.

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
