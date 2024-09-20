# Train state-of-the-art Pytorch Models

PhysioEx provides a fast and customizable way to train, evaluate and save state-of-the-art models for different physiological signal analysis tasks with different physiological signal datasets. This functionality is provided by the `train` command provided by this repository.

## Setup

Before using the `train` command, you need to set up a virtual environment and install the package in development mode. Here are the steps:

1. Make sure to have anaconda or miniconda correctly installed in your machine, then start installing a new virtual enviroment, in this case the new venv will be called `physioex`
```bash
    conda create -n physioex python==3.12
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

## Command-Line Arguments

The `train` command accepts several arguments to customize the training process. Below is a detailed description of each argument:

- `-m, --model`: Specify the model to train. This can be a registered model name or a path to a YAML configuration file. Default: `"chambon2018"`.
- `-ck, --checkpoint_dir`: Directory to save the checkpoint. Default: `None`.
- `-d, --datasets`: List of datasets to train the model on. Default: `["mass"]`.
- `-sc, --selected_channels`: List of channels to train the model. Default: `["EEG"]`.
- `-sl, --sequence_length`: Sequence length for the model. Default: `21`.
- `-l, --loss`: Loss function to use. Default: `"cel"` (Cross Entropy Loss).
- `-me, --max_epoch`: Maximum number of epochs for training. Default: `20`.
- `-nv, --num_validations`: Number of validation steps per epoch. Default: `10`.
- `-bs, --batch_size`: Batch size for training. Default: `32`.
- `--data_folder, -df`: Absolute path of the directory where the PhysioEx datasets are stored. Default: `None`.
- `--test, -t`: Test the model after training. Default: `False`.
- `--aggregate, -a`: Aggregate the results of the test. Default: `False`.
- `--hpc, -hpc`: Use high-performance computing setups. Default: `False`.
- `--num_nodes, -nn`: Number of nodes for distributed training. Default: `1`.
- `--config, -c`: Path to the configuration file for training options. Default: `None`.

You can use the `train -h --help` command to access the command documentation.

### Using a YAML Configuration File

You can specify a custom model configuration using a YAML file. Below is an example of how to structure your YAML file and use it with the `train` command. The model package can point to a custom python file into your working directory.

#### Example YAML Configuration (`my_model_config.yaml`)

```yaml
model_package: physioex.train.networks.seqsleepnet
model_class: SeqSleepNet
module_config:
  seq_len: 21
  in_channels: 1
  loss_call: cel
  loss_params: {}
preprocessing: xsleepnet
target_transform: get_mid_label
```

#### Running the Training with a YAML Configuration

To run the training using the above YAML configuration file, use the following command:

```bash
train --model my_model_config.yaml --datasets mass hmc --checkpoint_dir ./checkpoints --max_epoch 20 --batch_size 32
```

### Basic Usage

```bash
train --model chambon2018 --datasets mass --checkpoint_dir ./checkpoints --max_epoch 20 --batch_size 32
```
#### Testing the Model After Training

```bash
train --model chambon2018 --datasets mass --checkpoint_dir ./checkpoints --max_epoch 20 --batch_size 32 --test
```

#### Using High-Performance Computing

```bash
srun train --model chambon2018 --datasets mass --checkpoint_dir ./checkpoints --max_epoch 20 --batch_size 32 --hpc --num_nodes 4
```

Note that in this case the number of nodes needs to be properly setted up according to slurm or torque, i.e. you need to setup `--ntasks-per-core` or `ppn` value equal to the number of nodes you want to train the model on.  