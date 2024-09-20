import argparse
import importlib
from pathlib import Path

import yaml
from loguru import logger

from physioex.train.utils.train import train
from physioex.train.utils.test import test

from physioex.train.networks import config as networks_config
from physioex.train.networks.utils.loss import config as loss_config

def test_script():
    parser = argparse.ArgumentParser(description="Testing script")

    # experiment arguments
    parser.add_argument(
        "-m",
        "--model",
        default="chambon2018",
        type=str,
        help='Specify the model to test, can be a .yaml file if the model is not registered. Expected type: str. Default: "chambon2018"',
    )
    
    parser.add_argument(
        "-ck",
        "--checkpoint_path",
        default=None,
        type=str,
        help="Specify the model checkpoint. Expected type: str. Default: None",
    )
    
    parser.add_argument(
        "-d",
        "--datasets",
        help="Specify the datasets list to test the model on. Expected type: list. Default: ['mass']",
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
        
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="Specify the batch size for testing. Expected type: int. Default: 32",
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
        help="Specify the path to the configuration file where to store the options to test the model with. Expected type: str. Default: None",
    )
    
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            # Override command line arguments with config file values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)

    # check if the experiment is a yaml file
    if args.model.endswith(".yaml") or args.model.endswith(".yml"):
        import yaml
        import importlib
        
        # the user can specify a custom model implementation in a yaml file
        # the custom model must extend the SleepModule class and take a dictionary "module_config" as input
        
        # the yaml file must contain the following keys:
        # - model_package: the class of the model to be tested ex. path.to.some.module.ModelClass
        # - model_class: the class of the model inside model_package to be tested
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
        
    else:
        # the model is already registered in PhysioEx

        
        model_class = networks_config[args.model]["module"]
        model_config = networks_config[args.model]["module_config"]
        
        model_config["loss_call"] = loss_config[args.loss]
        model_config["loss_params"] = dict()
        model_config["seq_len"] = args.sequence_length
        model_config["in_channels"] = len(args.selected_channels)
        
        target_transform = networks_config[args.model]["target_transform"]
        preprocessing = networks_config[args.model]["input_transform"]
        
        model = model_class.load_from_checkpoint(args.checkpoint_path, module_config=model_config)
        
    datamodule_kwargs = {
        "selected_channels" : args.selected_channels,
        "sequence_length" : args.sequence_length,
        "target_transform" : target_transform,
        "preprocessing" : preprocessing,
        "data_folder" : args.data_folder,
    }
    
    # get the dir_path of the checkpoint_path
    checkpoint_dir = Path(args.checkpoint_path).parent
            
    test(
        datasets = args.datasets,
        datamodule_kwargs = datamodule_kwargs,
        model = model,
        batch_size = args.batch_size,
        hpc = args.hpc,
        num_nodes = args.num_nodes,
        results_path = checkpoint_dir,
        aggregate_datasets  = args.aggregate,
    )
