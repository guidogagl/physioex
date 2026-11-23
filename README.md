<p align="center">
<img src="https://raw.githubusercontent.com/guidogagl/physioex/refs/heads/main/docs/assets/images/logo.svg" width="250px" alt="PhysioEx Logo">
</p>

# PhysioEx

**PhysioEx (Physiological Signal Explainer)** is a Python library for explainable sleep staging with deep learning.

[![PyPI Version](https://img.shields.io/pypi/v/physioex?cacheSeconds=3600)](https://pypi.org/project/physioex/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/physioex/)
[![Tests](https://github.com/guidogagl/physioex/actions/workflows/test.yml/badge.svg)](https://github.com/guidogagl/physioex/actions/workflows/test.yml)
[![Docs](https://github.com/guidogagl/physioex/actions/workflows/ci.yml/badge.svg)](https://guidogagl.github.io/physioex/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/guidogagl/physioex/blob/main/LICENSE)

## Overview

PhysioEx provides a standard, reproducible way to train and evaluate
state-of-the-art sleep-staging models, so that the effort can go into
understanding what the models learn rather than into re-implementing them.

- **Data.** Fifteen sleep datasets are read directly from the raw recordings;
  preprocessing runs lazily, per channel, and is cached on disk.
- **Models.** Classic sleep-staging architectures, cardiorespiratory models and
  pretrained foundation encoders share one interface; published weights are
  loaded with `load_from_pretrained`.
- **Training.** A PyTorch training and evaluation loop, also available from the
  command line (`train`, `finetune`, `test_model`).
- **Foundation encoders.** Embedding extraction and linear probing on top of
  pretrained EEG and biosignal backbones.
- **Explainability.** Gradient-based attribution (Saliency, Input×Gradient,
  Integrated and Expected Gradients, and their DFT/STFT variants), concept and
  prototype methods, and Layer-wise Relevance Propagation for CNN, recurrent
  and transformer models, with conservation diagnostics.

The full documentation is at <https://guidogagl.github.io/physioex/>.

## Installation

Install the latest release from PyPI (Python 3.11 or later):

```bash
pip install physioex
```

Optional extras add dependencies that only some features need:

| Extra | Enables |
|---|---|
| `foundation` | the REVE encoder (`transformers`) |
| `datasets` | the HomePAP EDF fallback reader (`mne`) |
| `tracking` | TensorBoard and Weights & Biases logging |
| `explain` | Layer-wise Relevance Propagation (`zennit`) |
| `gpu-monitor` | GPU monitoring during training (`nvitop`) |

```bash
pip install "physioex[explain]"
```

To install from source, which includes changes not yet released on PyPI:

```bash
git clone https://github.com/guidogagl/physioex.git
cd physioex
pip install -e .
```

> **CPU-only installs and `torchvision`.** The `explain` extra depends on
> `zennit`, which requires `torchvision`. If `torch` comes from the PyTorch CPU
> index, install `torchvision` from the same index before the extra; otherwise
> pip installs a CUDA build that fails at import with
> `operator torchvision::nms does not exist`:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install "physioex[explain]"
> ```

## Quickstart

Set `PHYSIOEX_DATA` to the directory that holds your datasets.

### Train a model

```python
import os
os.environ["PHYSIOEX_DATA"] = "/path/to/datasets"

from physioex.data.datasets import get_dataset, available_datasets
from physioex.models.tinysleepnet import TinySleepNet
from physioex.train.trainer import Trainer

print(available_datasets())  # -> ['alzheimers', 'ccshs', 'cfs', 'chat', 'dcsm', ...]

dataset = get_dataset("hmc")(
    channels=["EEG", "EOG", "EMG"],
    pipelines="time_domain",   # preset preprocessing pipeline
    sequence_length=21,
)

model = TinySleepNet(in_chan=3, n_classes=5)
model = Trainer.train(model=model, dataset=dataset, max_epochs=20)
```

The same dataset specification drives the command-line tools; see the CLI page
in the documentation.

### Load a pretrained model or extract embeddings

```python
from physioex.models import load_from_pretrained, extract_embeddings, linear_probe
from physioex.models import CBraModEncoder

model = load_from_pretrained("protosleepnet-gagliardi")  # list_pretrained() lists all names
encoder = CBraModEncoder()
ds = encoder.get_dataset("hmc")                          # matched preprocessing
emb = extract_embeddings(encoder, ds)
scores = linear_probe(emb)
```

### Stage whole nights from cardiorespiratory signals

With `sequence_length=0`, one dataset item is one whole night.

```python
from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from physioex.train import replication as rep

model = load_from_pretrained("wav2sleep", verbose=True)
data = get_dataset("mesa")(
    channels=["ECG", "PPG", "ABD", "THX"],
    pipelines="wav2sleep",        # the authors' preprocessing
    sequence_length=0,
    label_transform="aasm4",      # W, Light, Deep, REM
    trim_excess_wake=False,
)
test = data.get_splits(split="wav2sleep")[2]
for subject, signals, labels, metadata in rep.iter_nights(data, test[:1]):
    stages = model.predict_night(signals)
```

## Models

All models live in `physioex.models`.

### Classic sleep-staging architectures

Trained from scratch on epoch sequences.

| Model | Class | Input |
|---|---|---|
| [Chambon et al. 2018](https://ieeexplore.ieee.org/document/8307462) | `Chambon2018Net` | raw signals |
| [TinySleepNet](https://github.com/akaraspt/tinysleepnet) | `TinySleepNet` | raw signals |
| Tsinalis et al. 2016 | `TsinalisCNN` | raw signals |
| [SeqSleepNet](https://arxiv.org/abs/1809.10932) | `SeqSleepNet` | time-frequency images |
| L-SeqSleepNet | `LSeqSleepNet` | time-frequency images, long sequences |
| [SleepTransformer](https://arxiv.org/abs/2105.11043) | `SleepTransformer` | time-frequency images |
| CoRe-Sleep | `CoReSleep` | multimodal |
| ProtoSleepNet | `ProtoSleepNet` | prototype-based, interpretable by design |

### Cardiorespiratory models

Whole-night staging from ECG, PPG and respiratory effort. The architectures are
re-implemented in PhysioEx; the published weights are downloaded from the
original sources, and each model has a matching preprocessing preset.

| Model | Class | Input | `load_from_pretrained` name | Weights | License |
|---|---|---|---|---|---|
| [wav2sleep](https://arxiv.org/abs/2411.04644) | `Wav2Sleep` | any subset of ECG, PPG, ABD, THX (or EOG) | `wav2sleep`, `wav2sleep-eog` | authors (Hugging Face) | MIT |
| [Jones et al. 2024](https://doi.org/10.1016/j.compbiomed.2024.108545) | `JonesECGNet` | ECG, age, sex, start time | `jones-ecg` and six variants | authors (GitHub) | AGPL-3.0 |
| [SleepPPG-Net](https://arxiv.org/abs/2202.05735) | `SleepPPGNet` | PPG | `sleepppgnet-replica` | third-party replica | non-commercial |

`examples/pretrained/<model>/verify.py` compares each model with its paper on
the published test splits. The public wav2sleep checkpoint is a later version
than the paper's model (about 1 % lower, as confirmed by the authors), so its
results are reported next to the paper's numbers rather than matched.

### Foundation encoders

Pretrained backbones behind a uniform `encode: (B, L, C, T) -> (B, L, D)`
interface, for embedding extraction and linear probing.

| Encoder | Class |
|---|---|
| CBraMod | `CBraModEncoder` |
| BENDR | `BENDREncoder` |
| LaBraM | `LaBraMEncoder` |
| BIOT | `BIOTEncoder` |
| SleepFM | `SleepFMEncoder` |
| TF-C | `TFCEncoder` |
| REVE | `REVEEncoder` |
| S-JEPA | `SJEEncoder` |
| NeuroLM | `NeuroLMEncoder` |

## Datasets

Datasets are resolved by name with `get_dataset(name)`; `available_datasets()`
lists them. PhysioEx reads the original recordings, which you obtain from the
source below. Descriptions of each dataset are in the
[Data](https://guidogagl.github.io/physioex/pages/data.html) page of the
documentation.

| Name | Dataset | Source | Size |
|---|---|---|---|
| `sleepedf` | [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) | PhysioNet | 197 recordings |
| `hmc` | [Haaglanden Medisch Centrum](https://physionet.org/content/hmc-sleep-staging/1.1/) | PhysioNet | 151 recordings |
| `dcsm` | [Danish Center for Sleep Medicine](https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/published-archive.html) | ERDA | 255 recordings |
| `shhs` | [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs) | NSRR | 5,793 participants at visit 1 |
| `mesa` | [Multi-Ethnic Study of Atherosclerosis](https://sleepdata.org/datasets/mesa) | NSRR | 2,237 participants |
| `mros` | [Osteoporotic Fractures in Men Study](https://sleepdata.org/datasets/mros) | NSRR | 2,911 recordings |
| `wsc` | [Wisconsin Sleep Cohort](https://sleepdata.org/datasets/wsc) | NSRR | 1,500 participants at baseline, several visits |
| `stages` | [Stanford Technology Analytics and Genomics in Sleep](https://sleepdata.org/datasets/stages) | NSRR | 1,914 recordings, 13 sites |
| `ccshs` | [Cleveland Children's Sleep and Health Study](https://sleepdata.org/datasets/ccshs) | NSRR | 515 children |
| `cfs` | [Cleveland Family Study](https://sleepdata.org/datasets/cfs) | NSRR | visit-5 recordings |
| `chat` | [Childhood Adenotonsillectomy Trial](https://sleepdata.org/datasets/chat) | NSRR | baseline and follow-up recordings |
| `hpap` | [HomePAP](https://sleepdata.org/datasets/homepap) | NSRR | three subsets (lab-full, lab-split, home) |
| `mass` | [Montreal Archive of Sleep Studies](http://ceams-carsm.ca/mass/) | CEAMS | 200 recordings |
| `alzheimers` | Alzheimer's disease dataset | UZ Leuven | 69 participants |
| `parkinsons` | Parkinson's disease dataset | UZ Leuven | 87 participants, up to 157 recordings |

## Citation

If you use PhysioEx, please cite:

```bib
@article{10.1088/1361-6579/adaf73,
  author  = {Gagliardi, Guido and Alfeo, Luca and Cimino, Mario G C A and Valenza, Gaetano and De Vos, Maarten},
  title   = {PhysioEx, a new Python library for explainable sleep staging through deep learning},
  journal = {Physiological Measurement},
  url     = {http://iopscience.iop.org/article/10.1088/1361-6579/adaf73},
  year    = {2025},
}
```

## License

PhysioEx is released under the [MIT License](https://github.com/guidogagl/physioex/blob/main/LICENSE).
Pretrained weights from third parties keep their own licenses, listed in the
model tables above and printed when they are loaded.
