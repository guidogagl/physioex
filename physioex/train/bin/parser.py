import importlib
import os
from argparse import ArgumentParser

import yaml

from physioex.train.networks import config as network_config
from typing import Union


def read_config(args: ArgumentParser) -> dict:

    args = vars(args)

    if args["config"] is not None:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)

        args.update(config)

    return args


def deep_update(dict1, dict2):
    """
    Aggiorna dict1 con i valori di dict2 ricorsivamente.
    """
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            deep_update(dict1[key], value)
        else:
            dict1[key] = value


def parse_model(parser: dict) -> dict:

    model = parser["model"]

    default_config = network_config["default"].copy()

    if model.endswith(".yaml"):

        with open(model, "r") as file:
            config = yaml.safe_load(file)

    elif model in network_config.keys():
        config = network_config[model]
    else:
        raise ValueError(
            f"Model {model} not found in the registered models or not a .yaml file"
        )

    deep_update(default_config, config)
    config = default_config

    config["model_kwargs"]["in_channels"] = len(parser["selected_channels"])
    config["model_kwargs"]["sequence_length"] = parser["sequence_length"]

    module, class_name = config["model"].split(":")
    config["model"] = getattr(importlib.import_module(module), class_name)

    # import the target_transform function if it exists
    if config["target_transform"] is not None:
        target_transform_package, target_transform_class = config[
            "target_transform"
        ].split(":")
        config["target_transform"] = getattr(
            importlib.import_module(target_transform_package), target_transform_class
        )

    deep_update(parser, config)

    return parser


class PhysioExParser:

    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Specify the path to the configuration file where to store the options to train the model with. Expected type: str. Default: None",
    )

    ##### Model arguments #####
    parser.add_argument(
        "-m",
        "--model",
        default="chambon2018",
        type=str,
        help='Specify the model to train, can be a .yaml file if the model is not registered or the model name. Expected type: str. Default: "chambon2018"',
    )

    ###### Data arguments #####
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
        default=["EEG"],
        nargs="+",
        help="Specify the channels to train the model. Expected type: list. Default: 'EEG'",
    )

    parser.add_argument(
        "-sl",
        "--sequence_length",
        type=int,
        default=21,
        help="Specify the sequence length for the model. Expected type: int. Default: 21",
    )

    parser.add_argument(
        "--model_task",
        "-mt",
        type=str,
        default="sleep",
        required=False,
        help="The task to be performed by the model can be either 'sleep' for sleep staging or 'sex' for sex classification. Expected type: str. Optional. Default: 'sleep'",
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
        "--num_workers",
        "-nw",
        type=int,
        default=0,
        help="Specify the number of workers for the dataloader. Expected type: int. Default: 0",
    )

    ##### Trainer arguments #####

    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="Specify the batch size for training. Expected type: int. Default: 32",
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
        "--aggregate",
        "-a",
        action="store_true",
        help="Aggregate the results of the test. Expected type: bool. Optional. Default: False",
    )

    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Test the model after training. Expected type: bool. Optional. Default: False",
    )

    parser.add_argument(
        "-rp",
        "--results_path",
        default=None,
        type=str,
        help="Specify the path where to store the results. Expected type: str. Default: None",
    )

    parser.add_argument(
        "--monitor",
        "-mon",
        default="val_acc",
        type=str,
        help="Specify the metric to monitor for the checkpoint callaback. Expected type: str. Default: 'val_acc'",
    )

    parser.add_argument(
        "--mode",
        "-mod",
        default="max",
        type=str,
        help="Specify the mode for the checkpoint callback. Expected type: str. Default: 'max'",
    )

    @classmethod
    def train_parser(cls) -> dict:

        cls.parser.add_argument(
            "-ck",
            "--checkpoint_dir",
            default=None,
            type=str,
            help="Specify where to save the checkpoint. Expected type: str. Default: None",
        )

        cls.parser.add_argument(
            "-me",
            "--max_epoch",
            default=20,
            type=int,
            help="Specify the maximum number of epochs for training. Expected type: int. Default: 20",
        )

        cls.parser.add_argument(
            "-nv",
            "--num_validations",
            default=10,
            type=int,
            help="Specify the number of validations steps to be done in each epoch. Expected type: int. Default: 10",
        )

        parser = cls.parser.parse_args()
        parser = read_config(parser)
        parser = parse_model(parser)

        return parser

    @classmethod
    def test_parser(cls):

        cls.parser.add_argument(
            "-ck_path",
            "--checkpoint_path",
            default=None,
            type=str,
            help="Specify the model checkpoint, if None a pretrained model is loaded. Expected type: str. Default: None",
        )

        parser = cls.parser.parse_args()
        parser = read_config(parser)
        parser = parse_model(parser)

        return parser

    @classmethod
    def finetune_parser(cls):

        cls.parser.add_argument(
            "-ck",
            "--checkpoint_dir",
            default=None,
            type=str,
            help="Specify where to save the checkpoint. Expected type: str. Default: None",
        )

        cls.parser.add_argument(
            "-me",
            "--max_epoch",
            default=20,
            type=int,
            help="Specify the maximum number of epochs for training. Expected type: int. Default: 20",
        )

        cls.parser.add_argument(
            "-nv",
            "--num_validations",
            default=10,
            type=int,
            help="Specify the number of validations steps to be done in each epoch. Expected type: int. Default: 10",
        )

        cls.parser.add_argument(
            "-lr",
            "--learning_rate",
            default=1e-7,
            type=float,
            help="Specify the learning rate for the model. Expected type: float. Default: 1e-7",
        )

        cls.parser.add_argument(
            "-ckp_path",
            "--checkpoint_path",
            default=None,
            type=str,
            help="Specify the model checkpoint, if None physioex searchs into its pretrained models. Expected type: str. Default: None",
        )

        parser = cls.parser.parse_args()
        parser = read_config(parser)
        parser = parse_model(parser)

        return parser
