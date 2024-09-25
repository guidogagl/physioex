# Train state-of-the-art Pytorch Models

PhysioEx provides a fast and customizable way to train, evaluate and save state-of-the-art models for different physiological signal analysis tasks with different physiological signal datasets. This functionality is provided by the `train`, `test_model` and `finetune` commands provided by this repository.

::: physioex.train.bin.train
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: physioex.train.bin.finetune
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

## Train CLI

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

## Test CLI

The `test_model` command is used to evaluate the performance of a trained model on specified datasets. The command supports various arguments to customize the testing process. Below is a detailed description of the arguments and their usage.

## Test Command

The `test` command is used to evaluate the performance of a trained model on specified datasets. The command supports various arguments to customize the testing process. Below is a detailed description of the arguments and their usage.

- `-m, --model`: Specifies the model to test. This can be a string representing a registered model or a path to a `.yaml` file if the model is not registered. The expected type is `str`. The default value is `"chambon2018"`.

- `-ck, --checkpoint_path`: Specifies the path to the model checkpoint. The expected type is `str`. The default value is `None`.

- `-d, --datasets`: Specifies the list of datasets on which to test the model. The expected type is `list`. The default value is `["mass"]`.

- `-sc, --selected_channels`: Specifies the channels to use for testing the model. The expected type is `list`. The default value is `["EEG"]`.

- `-sl, --sequence_length`: Specifies the sequence length for the model. The expected type is `int`. The default value is `21`.

- `-l, --loss`: Specifies the loss function to use. The expected type is `str`. The default value is `"cel"` (Cross Entropy Loss).

- `-bs, --batch_size`: Specifies the batch size for testing. The expected type is `int`. The default value is `32`.

- `--data_folder, -df`: Specifies the absolute path of the directory where the PhysioEx datasets are stored. If `None`, the home directory is used. The expected type is `str`. This argument is optional. The default value is `None`.

- `--aggregate, -a`: Aggregates the results of the test. The expected type is `bool`. This argument is optional. The default value is `False`.

- `--hpc, -hpc`: Indicates whether to use high-performance computing setups. This should be called when datasets have been compressed into `.h5` format with the `compress_datasets` command. The expected type is `bool`. This argument is optional. The default value is `False`.

- `--num_nodes, -nn`: Specifies the number of nodes to be used for distributed testing. This is only used when `hpc` is `True`. Note that in `slurm`, this value needs to be coherent with `--ntasks-per-node` or `ppn` in `torque`. The expected type is `int`. The default value is `1`.

- `--config, -c`: Specifies the path to the configuration file where the options to test the model are stored. The expected type is `str`. The default value is `None`.

## Finetune CLI


The `finetune` command is used to further train a pre-trained model on specified datasets. This command supports various arguments to customize the finetuning process. Below is a detailed description of the arguments and their usage.

- `-m, --model`: Specifies the model to finetune. This can be a string representing a registered model or a path to a `.yaml` file if the model is not registered. The expected type is `str`. The default value is `"chambon2018"`.

- `-lr, --learning_rate`: Specifies the learning rate for the model. The expected type is `float`. The default value is `1e-7`.

- `-ck, --checkpoint_path`: Specifies the path to the model checkpoint. If `None`, PhysioEx searches among its pre-trained models. The expected type is `str`. The default value is `None`.

- `-ck_dir, --checkpoint_dir`: Specifies the directory where the new finetuned model checkpoints will be stored. The expected type is `str`. The default value is `None`.

- `-d, --datasets`: Specifies the list of datasets on which to finetune the model. The expected type is `list`. The default value is `["mass"]`.

- `-sc, --selected_channels`: Specifies the channels to use for finetuning the model. The expected type is `list`. The default value is `["EEG"]`.

- `-sl, --sequence_length`: Specifies the sequence length for the model. The expected type is `int`. The default value is `21`.

- `-l, --loss`: Specifies the loss function to use. The expected type is `str`. The default value is `"cel"` (Cross Entropy Loss).

- `-me, --max_epoch`: Specifies the maximum number of epochs for finetuning. The expected type is `int`. The default value is `20`.

- `-nv, --num_validations`: Specifies the number of validation steps to be performed in each epoch. The expected type is `int`. The default value is `10`.

- `-bs, --batch_size`: Specifies the batch size for finetuning. The expected type is `int`. The default value is `32`.

- `--data_folder, -df`: Specifies the absolute path of the directory where the PhysioEx datasets are stored. If `None`, the home directory is used. The expected type is `str`. This argument is optional. The default value is `None`.

- `--test, -t`: Tests the model after finetuning. The expected type is `bool`. This argument is optional. The default value is `False`.

- `--aggregate, -a`: Aggregates the results of the test. The expected type is `bool`. This argument is optional. The default value is `False`.

- `--hpc, -hpc`: Indicates whether to use high-performance computing setups. This should be called when datasets have been compressed into `.h5` format with the `compress_datasets` command. The expected type is `bool`. This argument is optional. The default value is `False`.

- `--num_nodes, -nn`: Specifies the number of nodes to be used for distributed finetuning. This is only used when `hpc` is `True`. Note that in `slurm`, this value needs to be coherent with `--ntasks-per-node` or `ppn` in `torque`. The expected type is `int`. The default value is `1`.

- `--config, -c`: Specifies the path to the configuration file where the options to finetune the model are stored. The expected type is `str`. The default value is `None`.

