import os
import uuid

from physioex.train.bin.parser import PhysioExParser
from physioex.train.utils.test import test
from physioex.train.utils.train import train


def train_script():

    parser = PhysioExParser.train_parser()

    datamodule_kwargs = {
        "selected_channels": parser["selected_channels"],
        "sequence_length": parser["sequence_length"],
        "target_transform": parser["target_transform"],
        "preprocessing": parser["preprocessing"],
        "task": parser["model_task"],
        "data_folder": parser["data_folder"],
        "num_workers": parser["num_workers"],
    }

    train_kwargs = {
        "datasets": parser["datasets"],
        "datamodule_kwargs": datamodule_kwargs,
        "model_class": parser["model"],
        "model_config": parser["model_kwargs"],
        "model": None,
        "batch_size": parser["batch_size"],
        "fold": -1,
        "hpc": parser["hpc"],
        "num_validations": parser["num_validations"],
        "checkpoint_path": (
            parser["checkpoint_dir"]
            if parser["checkpoint_dir"] is not None
            else os.path.join("models", str(uuid.uuid4()))
        ),
        "max_epochs": parser["max_epoch"],
        "num_nodes": parser["num_nodes"],
        "resume": True,
        "monitor": parser["monitor"],
        "mode": parser["mode"],
    }

    best_checkpoint = train(**train_kwargs)

    best_checkpoint = os.path.join(train_kwargs["checkpoint_path"], best_checkpoint)

    if parser["test"]:

        test(
            datasets=parser["datasets"],
            datamodule_kwargs=datamodule_kwargs,
            model=None,
            model_class=parser["model"],
            model_config=parser["model_kwargs"],
            batch_size=parser["batch_size"],
            hpc=parser["hpc"],
            num_nodes=parser["num_nodes"],
            checkpoint_path=best_checkpoint,
            results_path=parser["results_path"],
            aggregate_datasets=parser["aggregate"],
        )

if __name__ == "__main__":
    train_script()