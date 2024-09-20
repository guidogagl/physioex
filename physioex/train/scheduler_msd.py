from pathlib import Path
from itertools import combinations

import os

import pandas as pd

from physioex.train.networks.finetuned import FineTunedModule
from physioex.train.networks import get_config
from physioex.data import PhysioExDataModule, PhysioExDataset
from physioex.train.networks.utils.loss import config as loss_config

from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from loguru import logger


import torch
from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)


############### **** ----------    model setup   ---------- **** ###############

model_name = "seqsleepnet"
preprocessing = "xsleepnet"
target_transform = None

module_config = get_config()[model_name]

selected_channels = ["EEG", "EOG", "EMG"]
sequence_length = 21

module_config.update(
    {
        "model_name": model_name,
        "seq_len": sequence_length,
        "in_channels": len(selected_channels),
        "n_train": 0,
        "n_classes": 5,
        "loss_call": loss_config["cel"],
        "loss_params": {},
    }
)

############### **** ----------    training setup   ---------- **** ###############

HPC = False

datasets = ["mass", "hmc", "mros", "dcsm", "mesa"]

train_datasets = [
    combo for r in range(1, len(datasets) + 1) for combo in combinations(datasets, r)
]

max_epochs = 5
VCI = 10
batch_size = 128
data_folder = "/mnt/guido-data/"
# data_folder= "bin/"
############### **** ----------    train/test utilities   ---------- **** ###############


def test_model(model_checkpoint, train_datasets):
    # load the model
    model = FineTunedModule.load_from_checkpoint(
        model_checkpoint, module_config=module_config
    )

    # test the model on each dataset
    multi_source_results = []

    trainer = Trainer(
        devices="auto",
        deterministic=True,
    )

    for test_dataset in datasets:

        test_datamodule = PhysioExDataModule(
            datasets=[test_dataset],
            batch_size=batch_size,
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            data_folder=data_folder,
            preprocessing=preprocessing,
            target_transform=target_transform,
            folds=-1,
            hpc=HPC,
        )

        multi_source_result = trainer.test(model, datamodule=test_datamodule)[0]

        multi_source_result["dataset"] = test_dataset

        if test_dataset in train_datasets:
            multi_source_result["split"] = "test"
        else:
            multi_source_result["split"] = "train"

        multi_source_results.append(multi_source_result)

    multi_source_results = pd.DataFrame(multi_source_results)
    checkpoint_dir = os.path.dirname(model_checkpoint)
    multi_source_results.to_csv(f"{checkpoint_dir}/msd_results.csv", index=False)


def train_model(model, train_datasets, checkpoint_path):

    train_datamodule = PhysioExDataModule(
        datasets=train_datasets,
        batch_size=batch_size,
        selected_channels=selected_channels,
        sequence_length=sequence_length,
        data_folder=data_folder,
        preprocessing=preprocessing,
        target_transform=target_transform,
        folds=-1,
        hpc=HPC,
    )

    num_steps = train_datamodule.dataset.__len__() * 0.7 // batch_size
    val_check_interval = max(1, num_steps // VCI)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath=checkpoint_path,
        filename="%d-{epoch}-{step}-{val_acc:.2f}",
        save_weights_only=False,
    )

    progress_bar_callback = RichProgressBar()

    my_logger = CSVLogger(save_dir=checkpoint_path)

    trainer = Trainer(
        devices="auto",
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        callbacks=[checkpoint_callback, progress_bar_callback],
        deterministic=True,
        logger=my_logger,
    )

    trainer.fit(model, train_datamodule)


############### **** ----------    main   ---------- **** ###############


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_folder", type=str, default="/mnt/guido-data/")

    args = argparser.parse_args()

    data_folder = args.data_folder

    for train_combo in train_datasets:

        logger.info(f"Setup for {train_combo}")

        k = len(train_combo)
        module_config["n_train"] = k

        combo_name = "_".join(train_combo)
        checkpoint_path = (
            f"models/multi-source-domain/finetunedmodel/k={k}/{combo_name}/"
        )
        # create the checkpoint path if it does not exist
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        check_files = os.listdir(checkpoint_path)

        if "msd_results.csv" in check_files:
            logger.info(f"Results for {combo_name} already exist")
            continue

        check_files = [file for file in check_files if ".ckpt" in file]

        if len(check_files) > 0:
            # check if the model is currently in training on another node of the cluster

            # get the last lightning log file metrics.csv into the last lightning_logs/version_x folder

            metrics_file = os.listdir(os.path.join(checkpoint_path, "lightning_logs"))
            metrics_file = os.path.join(
                checkpoint_path, "lightning_logs", metrics_file[0], "metrics.csv"
            )

            # check if the training reached the max epochs
            metrics_file = pd.read_csv(metrics_file)
            if metrics_file["epoch"].max() < max_epochs - 1:
                logger.info(f"Model for {combo_name} already in training")
                continue

            logger.info(f"Model for {combo_name} already exists, testing it ...")
            test_model(os.path.join(checkpoint_path, check_files[0]), train_combo)
            continue

        logger.info(f"Training model for {combo_name}")

        if k == 1:
            model = FineTunedModule(module_config=module_config)
            train_datasets = train_combo
        else:
            # search the best performing model on the k-1 dataset combo
            best_model = None
            best_acc = 0
            best_combo = None
            for combo in combinations(train_combo, k - 1):
                combo_name = "_".join(combo)
                combo_path = (
                    f"models/multi-source-domain/finetunedmodel/k={k-1}/{combo_name}/"
                )
                results_file = os.path.join(combo_path, "msd_results.csv")
                try:
                    results_file = pd.read_csv(results_file)
                except:
                    continue

                # drop the rows of the dataframe not related to the train datasets ( in combo )
                results_file = results_file[results_file["dataset"].isin(train_combo)]

                accuracy = results_file["test_acc"].mean().item()
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model = combo_path
                    best_combo = combo

            logger.info(f"Best model found for {train_combo} is {best_combo}")

            # get the checkpoint of the best model

            best_checkpoint = [
                file for file in os.listdir(best_model) if ".ckpt" in file
            ][0]
            best_checkpoint = os.path.join(best_model, best_checkpoint)

            model = FineTunedModule.load_from_checkpoint(
                best_checkpoint, module_config=module_config
            )
            train_datasets = [
                dataset for dataset in train_combo if dataset not in best_combo
            ]

        train_model(model, train_datasets, checkpoint_path)

        model_checkpoint = [
            file for file in os.listdir(checkpoint_path) if ".ckpt" in file
        ][0]

        test_model(os.path.join(checkpoint_path, model_checkpoint), train_datasets)
