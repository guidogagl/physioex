import copy
import json
import uuid
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import Parallel, delayed
from lightning.pytorch import seed_everything
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from physioex.data import CombinedTimeDistributedModule
from physioex.data import MultiSourceDomain as MSD
from physioex.data import TimeDistributedModule
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config


def calculate_combinations(elements):
    combinations_list = []
    for k in range(1, len(elements) + 1):
        combinations_list.extend(combinations(elements, k))
    return combinations_list


torch.set_float32_matmul_precision("medium")

model_dataset = "seqsleepnet"

target_domain = [
    {
        "dataset": "mass",
        "version": None,
        "picks": ["EEG"],
    },
    {
        "dataset": "dreem",
        "version": "dodh",
        "picks": ["C3-M2"],
    },
    {
        "dataset": "dreem",
        "version": "dodo",
        "picks": ["C3-M2"],
    },
    {
        "dataset": "sleep_physionet",
        "version": "2018",
        "picks": ["Fpz-Cz"],
    },
]

ckp_path = "models/ssd/pretrained/fold=0-epoch=43-step=639890-val_acc=0.87.ckpt"

max_epoch = 20
batch_size = 256
imbalance = False

val_check_interval = 300
num_folds = 2


class MultiSourceDomain:
    def __init__(
        self,
        model_dataset: str = model_dataset,
        msd_domain: List[Dict] = target_domain,
        sequence_length: int = 21,
        max_epoch: int = max_epoch,
        batch_size: int = batch_size,
        val_check_interval: int = val_check_interval,
        ckp_path: str = ckp_path,
        imbalance: bool = imbalance,
        num_folds: int = num_folds,
    ):

        seed_everything(42, workers=True)

        self.msd_domain = msd_domain

        self.model_call = config[model_dataset]["module"]

        self.input_transform = config[model_dataset]["input_transform"]
        self.target_transform = config[model_dataset]["target_transform"]
        self.sequence_length = sequence_length
        self.num_folds = num_folds
        self.module_config = config[model_dataset]["module_config"]
        self.module_config["seq_len"] = sequence_length
        self.module_config["in_channels"] = len(msd_domain[0]["picks"])

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval

        self.imbalance = imbalance

        self.ckp_path = ckp_path

        self.module_config["loss_call"] = loss_config["cel"]
        self.module_config["loss_params"] = dict()

    def train_evaluate(self, train_dataset, test_dataset, fold, ckp_path):

        logger.info("Splitting datasets into train, validation and test sets")
        train_dataset.split(fold)

        if test_dataset is not None:
            test_dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=train_dataset,
            batch_size=self.batch_size,
            fold=fold,
        )

        module = self.model_call.load_from_checkpoint(
            self.ckp_path, module_config=self.module_config
        )

        progress_bar_callback = RichProgressBar()

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            dirpath=ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold,
        )

        logger.info("Trainer setup")
        # Configura il trainer con le callback
        trainer = pl.Trainer(
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True,
        )

        # check if the model is already in the ckp_path
        # list the files inside the ckp_path
        # if the model is already there, load it
        files = list(Path(ckp_path).rglob("*.ckpt"))
        if len(files) > 0:
            files = [str(file) for file in files if f"fold={fold}" in str(file)]
            if len(files) > 0:
                ckp_path = files[0]
                logger.info(f"Loading model from {ckp_path}")
                module = self.model_call.load_from_checkpoint(
                    ckp_path, module_config=self.module_config
                )
            else:
                logger.warning("No model checkpoint found for the specified fold")
                logger.info("Training model")
                trainer.fit(module, datamodule=datamodule)

        else:
            logger.warning("No model checkpoints found in the specified directory")
            logger.info("Training model")
            trainer.fit(module, datamodule=datamodule)

        logger.info("Evaluating model on train domain")
        train_results = trainer.test(module, datamodule=datamodule)[0]
        train_results["fold"] = fold

        logger.info("Evaluating model on target domain")
        if test_dataset is not None:
            target_datamodule = CombinedTimeDistributedModule(
                dataset=test_dataset,
                batch_size=self.batch_size,
            )

            logger.info("Evaluating model")
            target_results = trainer.test(module, datamodule=target_datamodule)[0]
            target_results["fold"] = fold

            return {"train_results": train_results, "test_results": target_results}

        return {"train_results": train_results, "test_results": train_results}

    def run(self):

        domains_id = list(range(len(self.msd_domain)))
        domains_combinations = calculate_combinations(domains_id)

        for combination in domains_combinations:
            k = len(combination)

            if k < 4:
                continue

            train_domain = [self.msd_domain[idx] for idx in combination]
            test_domain = [
                self.msd_domain[idx] for idx in domains_id if idx not in combination
            ]

            train_domain_names = ""
            for domain in train_domain:
                train_domain_names += domain["dataset"]
                train_domain_names += (
                    "v." + domain["version"] if domain["version"] is not None else ""
                )

                if domain != train_domain[-1]:
                    train_domain_names += "-"

            test_domain_names = ""
            for domain in test_domain:
                test_domain_names += (
                    domain["dataset"] + " v. " + domain["version"]
                    if domain["version"] is not None
                    else "None" + " "
                )
                test_domain_names += "; "

            logger.info("Training on domains: %s" % train_domain_names)
            logger.info("Testing on domains: %s" % test_domain_names)

            train_dataset = MSD(
                domains=train_domain,
                preprocessing=self.input_transform,
                sequence_length=self.sequence_length,
                target_transform=self.target_transform,
                num_folds=self.num_folds,
            )

            if len(test_domain) == 0:
                test_dataset = None
            else:
                test_dataset = MSD(
                    domains=test_domain,
                    preprocessing=self.input_transform,
                    sequence_length=self.sequence_length,
                    target_transform=self.target_transform,
                    num_folds=self.num_folds,
                )

            ckp_path = f"models/msd/k={k}/{train_domain_names}/"
            Path(ckp_path).mkdir(parents=True, exist_ok=True)

            with open(ckp_path + "domains_setup.txt", "w") as f:
                train_line = "Train domain: " + train_domain_names
                if test_domain is not None:
                    test_line = "Test domain: " + test_domain_names
                else:
                    test_line = "Test domain: None"

                f.write(train_line + "\n" + test_line)

            results = [
                self.train_evaluate(train_dataset, test_dataset, fold, ckp_path)
                for fold in range(self.num_folds)
            ]

            train_results = [result["train_results"] for result in results]
            pd.DataFrame(train_results).to_csv(
                ckp_path + "train_results.csv", index=False
            )

            test_results = [result["test_results"] for result in results]

            pd.DataFrame(test_results).to_csv(
                ckp_path + "test_results.csv", index=False
            )

            logger.info("Results successfully saved in %s" % self.ckp_path)


if __name__ == "__main__":
    ssd = MultiSourceDomain()
    ssd.run()
