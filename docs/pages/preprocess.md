# Preprocessing

Preprocessing in PhysioEx is **lazy and on-the-fly**. There is no `preprocess`
command and no `Preprocessor` class that writes a preprocessed copy of a
dataset to disk. Instead, a `BasePhysioDataset` reads raw EDF, applies a
per-channel **preprocessing pipeline** the first time each channel is touched,
and caches the compiled result in an on-disk `ChannelCache` (keyed by a hash of
the pipeline, so identical pipelines share a cache).

You describe preprocessing in one of two ways:

1. **A named preset** — the common case; pass a preset name as the `pipelines`
   argument of a dataset.
2. **A hand-built `PreprocessingPipeline`** — compose the individual steps
   yourself for full control.

For the full class diagram (`PreprocessingStep`, `PreprocessingPipeline`,
`CompiledPipeline`, `ChannelCache`) see the
[data-layer architecture](architecture/library/data.md) page.

## Presets

Presets are the quickest path. `physioex.data` re-exports `get_preset`,
`available_presets` and the `PRESETS` registry:

```python
from physioex.data import available_presets, get_preset

available_presets()
# includes: 'raw', 'seqsleepnet', 'identity', 'time_domain', 'time_frequency',
#           per-modality ('eeg', 'eog', 'emg', 'ecg'), 'coresleep',
#           and foundation-model presets ('biot', 'bendr', 'cbramod',
#           'labram', 'sleepfm', 'tfc', 'reve', 'sjepa', 'neurolm')

pipe = get_preset("seqsleepnet")   # a PreprocessingPipeline
```

A preset resolves to either a single `PreprocessingPipeline` applied to every
channel (uniform presets such as `raw`, `seqsleepnet`) or a **dict of
per-modality pipelines** (bundle presets such as `time_domain` and
`time_frequency`, which filter EEG/EOG/EMG/ECG differently). In practice you
usually pass the preset name straight to the dataset:

```python
from physioex.data.datasets import get_dataset

# uniform: minimal bandpass + resample on every channel
raw_ds = get_dataset("hmc")(channels=["EEG"], pipelines="raw", sequence_length=21)

# per-modality time-domain filtering (default in the CLIs)
td_ds = get_dataset("hmc")(channels=["EEG", "EOG", "EMG"],
                           pipelines="time_domain", sequence_length=21)

# time-frequency: time_domain + XSleepNet spectrogram -> (L, C, T, F) samples
tf_ds = get_dataset("hmc")(channels=["EEG"], pipelines="time_frequency",
                           sequence_length=21)
```

## Composing a pipeline manually

A `PreprocessingPipeline` is an ordered list of `PreprocessingStep`s. The
available steps are re-exported from `physioex.data`:
`Identity`, `BandpassFilter`, `NotchFilter`, `Resample`, `ZScoreNormalize`,
`XSleepNetSpectrogram` (plus `HighPassFilter` in `physioex.data.steps`).

```python
from physioex.data import (
    PreprocessingPipeline, BandpassFilter, Resample, XSleepNetSpectrogram,
)

pipe = PreprocessingPipeline([
    BandpassFilter(low=0.3, high=40.0, order=5),
    Resample(target_fs=100.0),
    XSleepNetSpectrogram(nperseg=200, noverlap=100, nfft=256, window="hamming"),
])
```

The pipeline (or a dict of pipelines keyed by modality/physical name) can be
passed directly as the `pipelines` argument of any dataset in place of a preset
name. The dataset compiles it against each channel's native sampling rate and
caches the output.

:::{admonition} Custom modality dispatch
:class: note
When you pass a `dict` for `pipelines`, the dataset resolves each channel to a
pipeline by precedence: physical channel name, then modality, then a
`__default__` fallback. This is how the `time_domain`/`time_frequency` bundle
presets apply modality-specific filtering.
:::

## API

```{eval-rst}
.. autoclass:: physioex.data.PreprocessingPipeline
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: physioex.data.PreprocessingStep
   :no-index:
   :members:
   :show-inheritance:

.. autofunction:: physioex.data.get_preset
   :no-index:

.. autofunction:: physioex.data.available_presets
   :no-index:
```

The individual steps, the `ChannelCache`, and the compiled-pipeline classes are
documented in the auto-generated [API Reference](../api/index.md).
