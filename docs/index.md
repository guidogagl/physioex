<p align="center">
<img src="assets/images/logo.svg" width = "250px", alt="PhysioEx Logo">

<h1>PhysioEx</h1>
</p>

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![PyPI Version](https://badge.fury.io/py/physioex.svg)

**PhysioEx ( Physiological Signal Explainer )** is a versatile python library tailored for building, training, and explaining deep learning models for physiological signal analysis. 

The main purpose of the library is to propose a standard and fast methodology to train and evalutate state-of-the-art deep learning architectures for physiological signal analysis, to shift the attention from the architecture building task to the explainability task. 

With PhysioEx you can simulate a state-of-the-art experiment just running the `train`, `test_model`  and `finetune` commands; evaluating and saving the trained model; and start focusing on the explainability task! 

## Supported deep learning architectures

- [Chambon2018](https://ieeexplore.ieee.org/document/8307462) model for sleep stage classification ( raw time series as input).
- [TinySleepNet](https://github.com/akaraspt/tinysleepnet) model for sleep stage classification (raw time series as input).
- [SeqSleepNet](https://arxiv.org/pdf/1809.10932.pdf) model for sleep stage classification (time-frequency images as input).

## Supported datasets

- [SHHS (Sleep Heart Health Study)](https://sleepdata.org/datasets/shhs): A multi-center cohort study designed to investigate the cardiovascular consequences of sleep-disordered breathing.
- [MROS (MrOS Sleep Study)](https://sleepdata.org/datasets/mros): A study focusing on the outcomes of sleep disorders in older men.
- [MESA (Multi-Ethnic Study of Atherosclerosis)](https://sleepdata.org/datasets/mesa): A study examining the prevalence, correlates, and progression of subclinical cardiovascular disease.
- [DCSM (Dreem Challenge Sleep Monitoring)](https://physionet.org/content/dreem/1.0.0/): A dataset from the Dreem Challenge for automatic sleep staging.
- [MASS (Montreal Archive of Sleep Studies)](https://massdb.herokuapp.com/en/): A comprehensive collection of polysomnographic sleep recordings.
- [HMC (Home Monitoring of Cardiorespiratory Health)](https://physionet.org/content/hmc-kinematics/1.0.0/): A dataset for the study of cardiorespiratory health using home monitoring devices.

For the public available datasets ( DCSM, HMC ) PhysioEx takes care of automatically download the data thanks to the `preprocess` command. The other datasets needs to be acquired first ( mostly on [NSSR](https://sleepdata.org) ) and then fetched by PhysioEx via the `preprocess` command.

## Installation guidelines

### Create a Virtual Environment (Optional but Recommended)

```bash
    conda create -n physioex python==3.10
    conda activate physioex
    conda install pip
    pip install --upgrade pip  # On Windows, use `venv\Scripts\activate`
```
### Install via pip

1. **Install PhysioEx from PyPI:**
```bash
pip install physioex
```

### Install from source
1. **Clone the Repository:**
```bash
   git clone https://github.com/guidogagl/physioex.git
   cd physioex
```

2. **Install Dependencies and Package in Development Mode**
```bash
    pip install -e .
```