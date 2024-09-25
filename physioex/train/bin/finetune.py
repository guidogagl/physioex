import argparse
import importlib
from pathlib import Path

import yaml
from loguru import logger

from physioex.train.utils.finetune import finetune
from physioex.train.utils.test import test

from physioex.train.networks import config as networks_config
from physioex.train.networks.utils.loss import config as loss_config

def finetune_script():
    """
    Finetuning script for training and testing a model.

    This script allows you to fine-tune a pre-trained model using specified configurations and parameters.

    Usage:
       `$ finetune [PARAMS]`
       You can use the `finetune -h --help` command to access the command documentation.
       
    Args:
        --model (str, optional): Specify the model to train, can be a yaml file if the model is not registered. Defaults to "chambon2018".
            If a yaml file is provided, it should contain the model configuration details.
        --learning_rate (float, optional): Specify the learning rate for the model. Defaults to 1e-7.
            Note: A smaller learning rate is often used for fine-tuning to avoid large updates that could disrupt the pre-trained weights.
        --checkpoint_path (str, optional): Specify the model checkpoint, if None physioex searches into its pretrained models. Defaults to None.
            Note: Provide the path to a specific checkpoint file to resume training from a saved state.
        --checkpoint_dir (str, optional): Specify the checkpoint directory where to store the new finetuned model checkpoints. Defaults to None.
            Note: This directory will be used to save checkpoints during training.
        --datasets (list, optional): Specify the datasets list to train the model on. Defaults to ['mass'].
            Note: Provide a list of dataset names to be used for training.
        --selected_channels (list, optional): Specify the channels to train the model. Defaults to ['EEG'].
            Note: Channels refer to the data modalities (e.g., EEG, EOG) used for training.
        --sequence_length (int, optional): Specify the sequence length for the model. Defaults to 21.
            Note: Sequence length refers to the number of time steps in each input sequence.
        --loss (str, optional): Specify the loss function to use. Defaults to "cel".
            Note: The loss function determines how the model's performance is measured during training.
        --max_epoch (int, optional): Specify the maximum number of epochs for training. Defaults to 20.
            Note: An epoch is one complete pass through the training dataset.
        --num_validations (int, optional): Specify the number of validations steps to be done in each epoch. Defaults to 10.
            Note: Validation steps are used to evaluate the model's performance on a validation set during training.
        --batch_size (int, optional): Specify the batch size for training. Defaults to 32.
            Note: Batch size refers to the number of samples processed before the model's weights are updated.
        --data_folder (str, optional): The absolute path of the directory where the physioex dataset are stored, if None the home directory is used. Defaults to None.
            Note: Provide the path to the directory containing the datasets.
        --test (bool, optional): Test the model after training. Defaults to False.
            Note: If specified, the model will be tested on the validation set after training.
        --aggregate (bool, optional): Aggregate the results of the test. Defaults to False.
            Note: If specified, the test results will be aggregated across multiple datasets.
        --hpc (bool, optional): Using high performance computing setups or not, need to be called when datasets have been compressed into .h5 format with the compress_datasets command. Defaults to False.
            Note: Use this option if you are running the script on a high-performance computing cluster.
        --num_nodes (int, optional): Specify the number of nodes to be used for distributed training, only used when hpc is True. Defaults to 1.
            Note: In slurm this value needs to be coherent with '--ntasks-per-node' or 'ppn' in torque. This option is relevant for distributed training setups.
        --config (str, optional): Specify the path to the configuration file where to store the options to train the model with. Defaults to None.
            Note: The configuration file can override command line arguments.
 

    Example:
        ```bash
        $ finetune --model tinysleepnet --loss cel --sequence_length 21 --selected_channels EEG --checkpoint_path /path/to/checkpoint
        ```
        This command fine-tunes the `tinysleepnet` model using the CrossEntropy Loss (`cel`), with a sequence length of 21 and the `EEG` channel, starting from the specified checkpoint.

    Notes:
        - Ensure that the datasets are properly formatted and stored in the specified data folder using the preprocess script.
        - The script supports both single-node and multi-node training setups.
        - The configuration file, if provided, should be in YAML format and contain valid key-value pairs for the script options.
    """
    parser = argparse.ArgumentParser(description="Finetuning script")

    parser.add_argument(
        "-m",
        "--model",
        default="chambon2018",
        type=str,
        help='Specify the model to train, can be a .yaml file if the model is not registered. Expected type: str. Default: "chambon2018"',
    )
    
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-7,
        type=float,
        help="Specify the learning rate for the model. Expected type: float. Default: 1e-7",
    )
    
    parser.add_argument(
        "-ck",
        "--checkpoint_path",
        default=None,
        type=str,
        help="Specify the model checkpoint, if None physioex searchs into its pretrained models. Expected type: str. Default: None",
    )
    parser.add_argument(
        "-ck_dir",
        "--checkpoint_dir",
        default=None,
        type=str,
        help="Specify the checkpoint directory where to store the new finetuned model checkpoints. Expected type: str. Default: None",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        help="Specify the datasets list to train the model on. Expected type: list. Default: ['mass']",
        nargs="+",
        default=["mass"],
    )            

    parser.add_argument(
        "-sc",
        "--selected_channels",
        default=['EEG'],
        nargs="+",
        help="Specify the channels to train the model. Expected type: list. Default: 'EEG'",
    )

    # sequence
    parser.add_argument(
        "-sl",
        "--sequence_length",
        default=21,
        type=int,
        help="Specify the sequence length for the model. Expected type: int. Default: 21",
    )
    
    parser.add_argument(
        "-l",
        "--loss",
        default="cel",
        type=str,
        help='Specify the loss function to use. Expected type: str. Default: "cel" (Cross Entropy Loss)',
    )
    
    # trainer
    parser.add_argument(
        "-me",
        "--max_epoch",
        default=20,
        type=int,
        help="Specify the maximum number of epochs for training. Expected type: int. Default: 20",
    )
    
    parser.add_argument(
        "-nv",
        "--num_validations",
        default=10,
        type=int,
        help="Specify the number of validations steps to be done in each epoch. Expected type: int. Default: 10",
    )
    
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="Specify the batch size for training. Expected type: int. Default: 32",
    )

    parser.add_argument(
        "--data_folder",
        "-df",
        type=str,
        default=None,
        required=False,
        help="The absolute path of the directory where the physioex dataset are stored, if None the home directory is used. Expected type: str. Optional. Default: None",
    )

    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Test the model after training. Expected type: bool. Optional. Default: False",
    )
    
    parser.add_argument(
        "--aggregate",
        "-a",
        action="store_true",
        help="Aggregate the results of the test. Expected type: bool. Optional. Default: False",
    )
    
    parser.add_argument(
        "--hpc",
        "-hpc",
        action="store_true",
        help="Using high performance computing setups or not, need to be called when datasets have been compressed into .h5 format with the compress_datasets command. Expected type: bool. Optional. Default: False",
    )
    
    parser.add_argument(
        "--num_nodes",
        "-nn",
        default=1,
        type=int,
        help="Specify the number of nodes to be used for distributed training, only used when hpc is True, note: in slurm this value needs to be coherent with '--ntasks-per-node' or 'ppn' in torque. Expected type: int. Default: 1",
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Specify the path to the configuration file where to store the options to train the model with. Expected type: str. Default: None",
    )
    
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            # Override command line arguments with config file values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)

    # check if the checkpoint path is provided
    
    if args.checkpoint_path is None:    
        # we can load_pretrained model from the available models
        model = {
            "name" : args.model,
            "in_channels" : len(args.selected_channels),
            "sequence_length" : args.sequence_length,
            "loss" : args.loss,            
        }
        target_transform = networks_config[args.model]["target_transform"]
        preprocessing = networks_config[args.model]["input_transform"]
        model_class, model_config, model_checkpoint = None, None, None
    else:
    # check if the experiment is a yaml file
        if args.model.endswith(".yaml") or args.model.endswith(".yml"):
            import yaml
            import importlib
            
            # the user can specify a custom model implementation in a yaml file
            # the custom model must extend the SleepModule class and take a dictionary "module_config" as input
            
            # the yaml file must contain the following keys:
            # - model_package: the class of the model to be trained ex. path.to.some.module.ModelClass
            # - model_class: the class of the model inside model_package to be trained
            # - module_config: the configuration dictionary for the model
            # - preprocessing: the preprocessing keyword arguments to be passed to the PhysioExDataModule
            # - target_transform: the name of the target transform function inside the model_package to be used in the PhysioexDataModule, can be None
            
            with open(args.model, "r") as file:
                
                config = yaml.safe_load(file)
                
                # import the model class from the specified package
                model_package = config["model_package"]
                model_class = config["model_class"]
                model_config = config["module_config"]
                
                model_config["loss_call"] = loss_config[args.loss]
                model_config["loss_params"] = dict()
                model_config["seq_len"] = args.sequence_length
                model_config["in_channels"] = len(args.selected_channels)
                
                model = getattr(importlib.import_module(model_package), model_class).load_from_checkpoint(args.checkpoint_path, module_config=model_config)
                
                preprocessing = config["preprocessing"]
                target_transform = getattr(importlib.import_module(model_package), config["target_transform"])

            model_class, model_config, model_checkpoint = None, None, None

        else:
        # the model is already registered in PhysioEx, but it's not from the pretrained models
            model = None
        
            model_class = networks_config[args.model]["module"]
            model_config = networks_config[args.model]["module_config"]
            
            model_config["loss_call"] = loss_config[args.loss]
            model_config["loss_params"] = dict()
            model_config["seq_len"] = args.sequence_length
            model_config["in_channels"] = len(args.selected_channels)
            
            target_transform = networks_config[args.model]["target_transform"]
            preprocessing = networks_config[args.model]["input_transform"]
            
            model_checkpoint = args.checkpoint_path
        
    datamodule_kwargs = {
        "selected_channels" : args.selected_channels,
        "sequence_length" : args.sequence_length,
        "target_transform" : target_transform,
        "preprocessing" : preprocessing,
        "data_folder" : args.data_folder,
    }
    
    train_kwargs = {
        "datasets": args.datasets,
        "datamodule_kwargs": datamodule_kwargs,
        "model_class": None,
        "model_config": None,
        "batch_size": args.batch_size,
        "fold": -1,
        "hpc": args.hpc,
        "num_validations": args.num_validations,
        "checkpoint_path": args.checkpoint_dir,
        "max_epochs": args.max_epoch,
        "num_nodes": args.num_nodes,
        "resume": True,
    }
    
    best_checkpoint = finetune(
        model = model,
        model_class = model_class,
        model_config = model_config,
        model_checkpoint = model_checkpoint,
        learning_rate = args.learning_rate,
        train_kwargs = train_kwargs,
    )
    
    if args.test:
        
        if args.model.endswith(".yaml") or args.model.endswith(".yml"):
            # if the model is a yaml file we need to get the model class and config from the yaml file
            with open(args.model, "r") as file:
            
                config = yaml.safe_load(file)
            
                # import the model class from the specified package
                model_package = config["model_package"]
                model_class = config["model_class"]
                model_config = config["module_config"]
                
                model_config["loss_call"] = loss_config[args.loss]
                model_config["loss_params"] = dict()
                model_config["seq_len"] = args.sequence_length
                model_config["in_channels"] = len(args.selected_channels)
            
                model = getattr(importlib.import_module(model_package), model_class).load_from_checkpoint(best_checkpoint, module_config=model_config)
        else:
            model_class = networks_config[args.model]["module"]
            model_config = networks_config[args.model]["module_config"]
            
            model_config["loss_call"] = loss_config[args.loss]
            model_config["loss_params"] = dict()
            model_config["seq_len"] = args.sequence_length
            model_config["in_channels"] = len(args.selected_channels)
            
            model = model_class.load_from_checkpoint(best_checkpoint, module_config=model_config)
            
        test(
            datasets = args.datasets,
            datamodule_kwargs = datamodule_kwargs,
            model = model,
            batch_size = args.batch_size,
            checkpoint_path = best_checkpoint,
            hpc = args.hpc,
            num_nodes = args.num_nodes,
            results_path = args.checkpoint_dir,
            aggregate_datasets  = args.aggregate,
        )

if __name__ == "__main__":
    finetune_script()