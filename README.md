<p align="center">
<img src="https://raw.githubusercontent.com/guidogagl/physioex/refs/heads/main/docs/assets/images/logo.svg" width="250px" alt="PhysioEx Logo">
</p>

# PhysioEx

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
[![PyPI Version](https://img.shields.io/pypi/v/physioex)](https://pypi.org/project/physioex/)

**PhysioEx ( Physiological Signal Explainer )** is a versatile python library tailored for building, training, and explaining deep learning models for physiological signal analysis. 

The main purpose of the library is to propose a standard and fast methodology to train and evalutate state-of-the-art deep learning architectures for physiological signal analysis, to shift the attention from the architecture building task to the explainability task. 

With PhysioEx you can simulate a state-of-the-art experiment just running the `train`, `test_model`  and `finetune` commands; evaluating and saving the trained model; and start focusing on the explainability task!

Beyond training classic architectures from scratch, PhysioEx wraps a family of
**pretrained foundation encoders** behind a uniform interface, so you can extract
embeddings, run linear probes, and apply the explainability tools on top of them.

## Supported deep learning architectures

PhysioEx ships two families of models under `physioex.models`.

**Classic sleep-staging architectures** (trained from scratch):

- [Chambon2018](https://ieeexplore.ieee.org/document/8307462) — raw time series as input.
- [TinySleepNet](https://github.com/akaraspt/tinysleepnet) — raw time series as input.
- Tsinalis (`TsinalisCNN`) — raw time series as input.
- [SeqSleepNet](https://arxiv.org/pdf/1809.10932.pdf) — time-frequency images as input.
- L-SeqSleepNet (`LSeqSleepNet`) — long-sequence time-frequency modelling.
- [SleepTransformer](https://arxiv.org/pdf/2105.11043) — time-frequency images as input.
- CoReSleep — multimodal architecture.
- ProtoSleepNet — prototype-based, interpretable-by-design architecture.

**Foundation encoders** (pretrained backbones exposed through a uniform
`(B, L, C, T) -> (B, L, D)` `encode()` interface, for embedding extraction and
linear probing): `CBraMod`, `BENDR`, `LaBraM`, `BIOT`, `SleepFM`, `TFC`, `REVE`,
`S-JEPA`, `NeuroLM`.

## Supported datasets

All datasets are resolved by name via `get_dataset(name)`; `available_datasets()` returns:
`alzheimers`, `dcsm`, `hmc`, `hpap`, `mass`, `mesa`, `mros`, `parkinsons`, `shhs`, `sleepedf`, `stages`, `wsc`.

### Publicly Available:

These datasets are supported out of the box — PhysioEx reads the raw recordings directly (preprocessing is applied lazily, on the fly).

- [Sleep-EDF(78)](https://physionet.org/content/sleep-edfx/1.0.0/), The sleep-edf database contains 197 whole-night PolySomnoGraphic sleep recordings, containing EEG, EOG, chin EMG, and event markers.
- [HMC (Haaglanden Medisch Centrum)](https://physionet.org/content/hmc-sleep-staging/1.1/), is a collection of 151 whole-night PSG recordings from 85 men and 66 women, gathered at the Haaglanden Medisch Centrum sleep center. The PSG data includes 4 EEG channels (F4/M1, C4/M1, O2/M1, and C3/M2), two EOG channels (E1/M2 and E2/M2), and one bipolar chin EMG, with all signals sampled at 256 Hz.
- [DCSM (Danish Center for Sleep Medicine)](https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/published-archive.html), is a collection of 255 randomly selected and fully anonymized overnight lab-based PSG recordings from patients seeking diagnosis for non-specific sleep-related disorders at the DCSM. The PSG setup included EEG, EOG, and EMG channels, all sampled at 256 Hz.


### [NSRR](https://sleepdata.org) Datasets

These datasets can be obtained from the NSRR archive. Once downloaded, point PhysioEx at the dataset root and it will read the recordings directly.

- [SHHS (Sleep Heart Health Study)](https://sleepdata.org/datasets/shhs), is a multi-center cohort study designed to investigate the cardiovascular and other consequences of sleep-disordered breathing. At visit 1, it included 5,793 participants aged 40 years or older. PSG recordings were typically conducted in the subjects' homes by trained and certified technicians. The recording montage included C3/A2 and C4/A1 EEGs sampled at 125 Hz, right and left EOGs sampled at 50 Hz, and a bipolar submental EMG sampled at 125 Hz.
- [MESA (Multi-Ethnic Study of Atherosclerosis)](https://sleepdata.org/datasets/mesa), is a multi-center prospective study of 2.237 ethnically diverse men and women aged 45-84 from six communities in the United States. PSGs recordings were obtained using in-home settings including central C4-M1 EEG, bilateral EOG and chin EMG sampled at 256Hz. PSGs were scored by one of 3 MESA certified, registered polysomnologists.
- [MrOS (The Osteoporotic Fractures in Men Study)](https://sleepdata.org/datasets/mros), is a multicenter study comprising 2,911 PSG recordings from men aged 65 years or older, enrolled at six clinical centers. PSG recordings were conducted in home settings and included C3/A2 and C4/A1 EEGs, chin EMG, and left-right EOG, all sampled at 256 Hz.
- [WSC (The Wisconsin Sleep Cohort)](https://sleepdata.org/datasets/wsc) A longitudinal study of the causes, consequences, and natural history of sleep disorders using overnight in-laboratory sleep recordings gathered at the University of Wisconsin, United States, with a baseline sample of 1,500 subjects assessed at four-year intervals. The study consists of multiple visits with overnight PSG data acquisition. PSG recordings included C3/M2 EEG, EMG, and left-right EOG, all sampled at 200 Hz.
- [STAGES (Stanford Technology Analytics and Genomics in Sleep)](https://sleepdata.org/datasets/stages), 1,914 recordings across 13 clinical sites with a bipolar EEG montage (C4-M1 / C3-M2 equivalent), EOG and chin EMG, plus rich respiratory / arousal / PLM event annotations. Selectable by site and by recording (`first` / `second` / `all`).
- [HomePAP](https://sleepdata.org/datasets/homepap) (`hpap`), a positive-airway-pressure study in NSRR format, with three subsets — `lab-full`, `lab-split` and `home` — selectable via the `subset` parameter.

### Others

- [MASS (Montreal Archive of Sleep Studies)](http://ceams-carsm.ca/mass/), is an open-access collaborative database containing laboratory-based PSG recordings. It includes 200 complete nights recorded from 97 men and 103 women, aged 18 to 76 years. All recordings have a sampling frequency of 256 Hz and feature an EEG montage of 4–20 channels, along with standard EOG and EMG.
- **Alzheimer's disease dataset** (`alzheimers`, UZ Leuven), 69 subjects (37 Alzheimer's + 32 healthy controls), one night each with an average-reference EEG montage, differential EOG, chin EMG and ECG at 200 Hz. The `subset` parameter selects only AD or HC subjects.
- **Parkinson's disease dataset** (`parkinsons`, UZ Leuven), 87 subjects (48 Parkinson's + 40 healthy older adults) with up to 157 recordings (night and nap), bipolar EEG montage, EOG, chin EMG and ECG at 500 Hz.


## Installation guidelines

### Create a Virtual Environment (Optional but Recommended)

```bash
$ conda create -n physioex python==3.11
$ conda activate physioex
$ conda install pip
$ pip install --upgrade pip  # On Windows, use `venv\Scripts\activate`
```

### Install from source ( Recommended )
1. **Clone the Repository:**
```bash
$ git clone https://github.com/guidogagl/physioex.git
$ cd physioex
```

2. **Install Dependencies and Package in Development Mode**
```bash
$ pip install -e .
```

### Install via pip

1. **Install PhysioEx from PyPI:**
```bash
$ pip install physioex
```

Note: the github version of the library is kept updated weekly, the PiPy version may be outdated depending on the last commit of the github version. We recommend to use the github version if possible.  

**Optional extras** (heavyweight, feature-specific deps installed lazily):

```bash
$ pip install "physioex[foundation]"   # transformers, for the REVE encoder
$ pip install "physioex[datasets]"     # mne, for the HOMEPAP EDF fallback reader
$ pip install "physioex[tracking]"     # tensorboard / wandb logging
```

## Quickstart

Set `PHYSIOEX_DATA` to the directory holding your datasets, then use the raw-EDF
data layer, the model zoo, and the training loop:

```python
import os
os.environ["PHYSIOEX_DATA"] = "/path/to/datasets"

from physioex.data.datasets import get_dataset, available_datasets
from physioex.models.tinysleepnet import TinySleepNet
from physioex.train.trainer import Trainer

print(available_datasets())  # -> ['hmc', 'sleepedf', 'dcsm', 'mass', ...]

dataset = get_dataset("hmc")(
    channels=["EEG", "EOG", "EMG"],
    pipelines="time_domain",   # preset preprocessing pipeline
    sequence_length=21,
)

model = TinySleepNet(in_chan=3, n_classes=5)
model = Trainer.train(model=model, dataset=dataset, max_epochs=20)
```

Load a pretrained model or extract foundation-model embeddings:

```python
from physioex.models import load_from_pretrained, extract_embeddings, linear_probe
from physioex.models import CBraModEncoder

model = load_from_pretrained("protosleepnet")           # from the HF model zoo
encoder = CBraModEncoder()
ds = encoder.get_dataset("hmc")                          # matched preprocessing
emb = extract_embeddings(encoder, ds)
scores = linear_probe(emb)
```

The same dataset spec drives the CLI (`train` / `finetune` / `test_model`) — see
the CLI page in the [documentation](https://guidogagl.github.io/physioex/).

## Cite Us!
```bib
@article{10.1088/1361-6579/adaf73,
	author={Gagliardi, Guido and Alfeo, Luca and Cimino, Mario G C A and Valenza, Gaetano and De Vos, Maarten},
	title={PhysioEx, a new Python library for explainable sleep staging through deep learning},
	journal={Physiological Measurement},
	url={http://iopscience.iop.org/article/10.1088/1361-6579/adaf73},
	year={2025},
}
```
