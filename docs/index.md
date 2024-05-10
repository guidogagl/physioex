<p align="center">
<img src="assets/images/logo.svg" width = "250px", alt="PhysioEx Logo">

<h1>PhysioEx</h1>
</p>

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![PyPI Version](https://badge.fury.io/py/physioex.svg)

**PhysioEx ( Physiological Signal Explainer )** is a versatile python library tailored for building, training, and explaining deep learning models for physiological signal analysis. 

The main purpose of the library is to propose a standard and fast methodology to train and evalutate state-of-the-art deep learning architectures for physiological signal analysis, to shift the attention from the architecture building task to the explainability task. 

With PhysioEx you can simulate a state-of-the-art experiment just running the `train` command; evaluating and saving the trained model; and start focusing on the explainability task! The `train` command will also take charge of downloading and processing the specified dataset if unavailable.

## Supported deep learning architectures
- [Chambon2018](https://ieeexplore.ieee.org/document/8307462) model for sleep stage classification.
- [TinySleepNet](https://github.com/akaraspt/tinysleepnet) model for sleep stage classification.
- [SeqSleepNet](https://arxiv.org/pdf/1809.10932.pdf) model for sleep stage classification (time-frequency images as input).
- SeqECGnet model for ECG arrythmia classifiaction ( 5-AAMI classes ).
## Supported datasets
- [SleepEDF (version 2018-2013)](https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/) sleep staging dataset.
- [Dreem (version DODO-DODH)](https://github.com/Dreem-Organization/dreem-learning-open) sleep staging dataset.
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) dataset for ECG analysis.

## Installation guidelines

1. **Clone the Repository:**
```bash
   git clone https://github.com/guidogagl/physioex.git
   cd physioex
```
2. **Create a Virtual Environment (Optional but Recommended)**
```bash
    conda create -n physioex python==3.10
    conda activate physioex
    conda install pip
    pip install --upgrade pip  # On Windows, use `venv\Scripts\activate`
```
3. **Install Dependencies and Package in Development Mode**
```bash
    pip install -e .
```

