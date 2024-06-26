import argparse
import importlib
from pathlib import Path

import yaml
from loguru import logger

from physioex.data import register_dataset, set_data_folder
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
        default="sleep_edf",
        type=str,
        help='Specify the dataset to use. Expected type: str. Default: "SleepPhysionet"',
    )
    parser.add_argument(
        "-v",
        "--version",
        default="2018",
        type=str,
        help='Specify the version of the dataset. Expected type: str. Default: "2018"',
    )
    parser.add_argument(
        "-p",
        "--picks",
        default="Fpz-Cz",
        type=str,
        help="Specify the signal electrodes to pick to train the model. Expected type: list. Default: 'Fpz-Cz'",
    )

    # sequence
    parser.add_argument(
        "-sl",
        "--sequence_lenght",
        default=3,
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
        default=300,
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
        "-nj",
        "--n_jobs",
        default=10,
        type=int,
        help="Specify the number of jobs for parallelization. Expected type: int. Default: 10",
    )

    parser.add_argument(
        "-imb",
        "--imbalance",
        default=False,
        type=bool,
        help="Specify rather or not to use f1 score instead of accuracy to save the checkpoints. Expected type: bool. Default: False",
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
        "--wandb",
        "-wdb",
        action="store_true",
        help="Enables the logging on Weights and Biases. If the argument is present, it is set to True. Otherwise, it is set to False.",
    )

    args = parser.parse_args()

    if args.data_folder is not None:
        set_data_folder(args.data_folder)

    # check if the experiment is a yaml file
    if args.experiment.endswith(".yaml") or args.experiment.endswith(".yml"):
        args.experiment = register_experiment(args.experiment)

    # check if the dataset is a yaml file
    if args.dataset.endswith(".yaml") or args.dataset.endswith(".yml"):
        args.dataset = register_dataset(args.dataset)

    Trainer(
        model_name=args.experiment,
        dataset_name=args.dataset,
        ckp_path=args.checkpoint,
        loss_name=args.loss,
        version=args.version,
        picks=args.picks,
        sequence_length=args.sequence_lenght,
        max_epoch=args.max_epoch,
        val_check_interval=args.val_check_interval,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        imbalance=args.imbalance,
        use_wandb=args.wandb,
    ).run()


if __name__ == "__main__":
    main()
