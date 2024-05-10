import argparse

from physioex.train import Trainer


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
        default="sleep_physionet",
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

    args = parser.parse_args()

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
    ).run()


if __name__ == "__main__":
    main()
