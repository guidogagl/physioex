import argparse
import importlib
from pathlib import Path

import yaml
from loguru import logger

from physioex.train import Trainer
from physioex.train.networks import config, register_experiment


def main():
    parser = argparse.ArgumentParser(description="Training script")

    # experiment arguments
    parser.add_argument(
        "-e",
        "--experiment",
        default="chambon2018",
        type=str,
        help='Specify the experiment to run. Expected type: str. Default: "chambon2018"',
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        default=None,
        type=str,
        help="Specify where to save the checkpoint. Expected type: str. Default: None",
    )
    parser.add_argument(
        "-l",
        "--loss",
        default="cel",
        type=str,
        help='Specify the loss function to use. Expected type: str. Default: "cel" (Cross Entropy Loss)',
    )

    # dataset args
    parser.add_argument(
        "-d",
        "--dataset",
        default="mass",
        type=str,
        help='Specify the dataset to use. Expected type: str. Default: "MASS dataset"',
    )
    parser.add_argument(
        "-v",
        "--version",
        default=None,
        type=str,
        help='Specify the version of the dataset. Expected type: str. Default: "None"',
    )
    parser.add_argument(
        "-p",
        "--picks",
        default="EEG",
        type=str,
        help="Specify the signal electrodes to pick to train the model. Expected type: list. Default: 'EEG'",
    )

    # sequence
    parser.add_argument(
        "-sl",
        "--sequence_lenght",
        default=21,
        type=int,
        help="Specify the sequence length for the model. Expected type: int. Default: 3",
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
        "-vci",
        "--val_check_interval",
        default=3,
        type=int,
        help="Specify the validation check interval during training. Expected type: int. Default: 300",
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
        "--config",
        "-c",
        type=str,
        required=False,
        help="Path to the configuration file in YAML format",
    )

    parser.add_argument(
        "--random_fold",
        "-rf",
        type=bool,
        default=False,
        required=False,
        help="Weather or not to perform the training on a random fold. Expected type: bool. Optional. Default: False",
    )
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            # Override command line arguments with config file values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    
    # check if the experiment is a yaml file
    if args.experiment.endswith(".yaml") or args.experiment.endswith(".yml"):
        args.experiment = register_experiment(args.experiment)

    # check if the dataset is a yaml file

    # convert the datasets into a list by diving by " "
    datasets = args.dataset.split(" ")
    versions = args.version.split(" ") if args.version is not None else None
    picks = args.picks.split(" ")
    
    print(datasets)
    print(versions)
    
    Trainer(
        model_name=args.experiment,
        datasets = datasets,
        versions = versions,
        ckp_path = args.checkpoint,
        loss_name = args.loss,
        selected_channels= picks,
        sequence_length=args.sequence_lenght,
        max_epoch=args.max_epoch,
        val_check_interval=args.val_check_interval,
        batch_size=args.batch_size,
        random_fold = args.random_fold,
    ).run()


if __name__ == "__main__":
    main()
