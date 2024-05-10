# Train state-of-the-art Pytorch Models

PhysioEx provides a fast and customizable way to train, evaluate and save state-of-the-art models for different physiological signal analysis tasks with different physiological signal datasets. This functionality is provided by the `train` command provided by this repository.

## Setup

Before using the `train` command, you need to set up a virtual environment and install the package in development mode. Here are the steps:

1. Make sure to have anaconda or miniconda correctly installed in your machine, then start installing a new virtual enviroment
```bash
    conda create -n myenv python==3.10
```    

2. Now jump into the enviroment and upgrade pip
```bash
    conda activate myenv
    conda install pip
    pip install --upgrade pip
```

3. Last but not least install PhysioEx in development mode
```bash
    git clone https://github.com/guidogagl/physioex.git
    cd physioex
    pip install -e .
```    

## Experiments
- `chambon2018`: This experiment uses the [Chambon2018](https://ieeexplore.ieee.org/document/8307462) model for sleep stage classification.
- `tinysleepnet`: This experiment uses the [TinySleepNet](https://github.com/akaraspt/tinysleepnet) model for sleep stage classification.
- `seqsleepnet`: This experiment uses the [SeqSleepNet](https://arxiv.org/pdf/1809.10932.pdf) model for sleep stage classification (time-frequency images as input).

To run an experiment, use the `-e` or `--experiment` argument followed by the name of the experiment. For example:

```bash
train --experiment chambon2018
```
### Dataset-experiment compatibility
|                     | SleepPhysioNet | Dreem |
|---------------------|:--------------:|:-----:|
| chambon2018         |       ✔        |   ✔️   |
| tinysleepnet        |       ✔        |   ✔️   |
| seqsleepnet         |       ✔        |   ✔️   |
| contr_chambon2018   |       ✔        |   ✔️   |
| contr_tinysleepnet  |       ✔        |   ✔️   |
| contr_seqsleepnet   |       ✔        |   ✔️   |

## Train Command
The train command is used to train models. Here are the available arguments:

- `-e`, `--experiment`: Specify the experiment to run. Expected `type: str`. `Default: "chambon2018"`.
- `-ckpt`, `--chekpoint`:  Specify where to save the checkpoint. Expected `type: str`. `Default: None`    
- `-d`, `--dataset`: Specify the dataset to use. Expected `type: str`. `Default: "SleepPhysionet"`.
- `-v`, `--version`: Specify the version of the dataset. Expected `type: str`. `Default: "2018"`.
- `-c`, `--use_cache`: Specify whether to use cache for the dataset. Expected `type: bool`. `Default: True`.
- `-sl`, `--sequence_lenght`: Specify the sequence length for the model. Expected `type: int`. `Default: 3`.
- `-me`, `--max_epoch`: Specify the maximum number of epochs for training. Expected `type: int`. `Default: 20`.
- `-vci`, `--val_check_interval`: Specify the validation check interval during training. Expected `type: int`. `Default: 300`.
- `-bs`, `--batch_size`: Specify the batch size for training. Expected `type: int`. `Default: 32`.
- `-nj`, `--n_jobs`: Specify the number of jobs for parallelization. Expected `type: int`. `Default: 10`
- `-imb`, `--imbalance`:  -me "Specify rather or not to use f1 score instead of accuracy to save the checkpoints. Expected `type: bool`. `Default: False`

## Experimental Results
### Sequence Lenght: 3

=== "Standard models"
    ![results table](evaluations/ccl_seqlen=3.svg){ width="75%" }

=== "Similarity models"
    ![results table](evaluations/scl_seqlen=3.svg){ width="75%" }
