import copy
import uuid
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from joblib import Parallel, delayed
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from physioex.data import TimeDistributedModule, datasets
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config

import json
import numpy as np

import torch
from loguru import logger

torch.set_float32_matmul_precision("medium")

model_name = "seqsleepnet"

train_domain = {
    "name": "shhs",
    "version": None,
    "picks": ["EEG"],
    "sequence_length": 21,
}

target_domain = [
    {
        "name": "dreem",
        "version": "dodh",
        "picks": ["C3-M2"],
        "sequence_length": 21,
    },
    {
        "name": "dreem",
        "version": "dodo",
        "picks": ["C3-M2"],
        "sequence_length": 21,
    },
    {
        "name": "sleep_physionet",
        "version": "2018",
        "picks": ["Fpz-Cz"],
        "sequence_length": 21,
    },
    {
        "name": "sleep_physionet",
        "version": "2013",
        "picks": ["Fpz-Cz"],
        "sequence_length": 21,
    },
]

ckp_path = "models/ssd/" + str(uuid.uuid4()) + "/"

max_epoch = 100
batch_size = 256
imbalance = False

val_check_interval = 300


class SingleSourceDomain:
    def __init__(
        self,
        model_name: str = model_name,
        train_domain: dict = train_domain,
        target_domain: dict = target_domain,
        max_epoch: int = max_epoch,
        batch_size: int = batch_size,
        val_check_interval: int = val_check_interval,
        ckp_path: str = ckp_path,
        imbalance: bool = imbalance,
    ):

        seed_everything(42, workers=True)

        self.target_domain = target_domain

        self.train_call = datasets[train_domain["name"]]

        self.train_args = {
            "version": train_domain["version"],
            "picks": train_domain["picks"],
            "sequence_length": train_domain["sequence_length"],
        }

        self.model_call = config[model_name]["module"]

        self.input_transform = config[model_name]["input_transform"]
        self.target_transform = config[model_name]["target_transform"]
        self.module_config = config[model_name]["module_config"]
        self.module_config["seq_len"] = train_domain["sequence_length"]
        self.module_config["in_channels"] = len(train_domain["picks"])

        self.train_args["target_transform"] = self.target_transform
        self.train_args["preprocessing"] = self.input_transform

        logger.info("Loading training dataset")
        self.train_dataset = self.train_call(**self.train_args)
        logger.info("Dataset loaded")

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval

        self.imbalance = imbalance

        self.ckp_path = ckp_path
        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)

        self.folds = list(range(self.train_dataset.get_num_folds()))

        self.module_config["loss_call"] = loss_config["cel"]
        self.module_config["loss_params"] = dict()

    def train_evaluate(self, fold: int = 0):

        dataset = self.train_dataset

        logger.info("Splitting dataset into train, validation and test sets")
        dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=dataset,
            batch_size=self.batch_size,
            fold=fold,
        )

        module = self.model_call(module_config=self.module_config)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            dirpath=self.ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold,
        )

        progress_bar_callback = RichProgressBar()

        logger.info("Trainer setup")
        # Configura il trainer con le callback
        trainer = pl.Trainer(
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True,
        )

        logger.info("Training model")
        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=datamodule)

        logger.info("Evaluating model on single source domain")
        ssd_results = trainer.test(ckpt_path="best", datamodule=datamodule)

        logger.info("Evaluating model on target domains")
        target_results = {}
        for target_domain in self.target_domain:

            target_call = datasets[target_domain["name"]]
            logger.info(
                f"Loading target dataset {target_domain['name']} version {target_domain['version']}"
            )

            target_args = {
                "version": target_domain["version"],
                "picks": target_domain["picks"],
                "sequence_length": target_domain["sequence_length"],
                "target_transform": self.target_transform,
                "preprocessing": self.input_transform,
            }

            target_dataset = target_call(**target_args)

            target_folds = list(range(target_dataset.get_num_folds()))
            
            # check if the key exists in case create it
            if target_domain["name"] not in target_results:
                target_results[target_domain["name"]] = {}
                
            if target_domain["version"] not in target_results[target_domain["name"]]:
                target_results[target_domain["name"]][target_domain["version"]] = {}

            for tf in target_folds:
                target_dataset.split(fold)

                target_datamodule = TimeDistributedModule(
                    dataset=target_dataset,
                    batch_size=self.batch_size,
                    fold=fold,
                )

                logger.info("Evaluating model on fold %d" % tf)

                target_results[target_domain["name"]][target_domain["version"]][tf] = (
                    trainer.test(ckpt_path="best", datamodule=target_datamodule)
                )

        return {"ssd_results": ssd_results, "msd_results": target_results}

    def run(self):

        # ssd has only one fold : 0
        results = self.train_evaluate(0)

        ssd_results = results["ssd_results"]
        msd_results = results["msd_results"]

        pd.DataFrame(ssd_results).to_csv(self.ckp_path + "ssd_results.csv", index=False)

        # save msd_results as a json file

        with open(self.ckp_path + "msd_results.json", "w") as f:
            json.dump(msd_results, f)

        # msd results saving
        targets = list(msd_results.keys())

        target_df = []
        for target in targets:
            versions = list(msd_results[target].keys())

            for version in versions:

                results_dict = msd_results[target][version]
                results_folds = list(results_dict.keys())

                for fold in results_folds:
                    target_df.append(pd.DataFrame(results_dict[fold]))
                    target_df[-1]["fold"] = fold
                    target_df[-1]["version"] = version
                    target_df[-1]["target"] = target

        target_df = pd.concat(target_df)

        target_df.to_csv(self.ckp_path + "msd_results.csv", index=False)

        logger.info("Results successfully saved in %s" % self.ckp_path)


if __name__ == "__main__":
    ssd = SingleSourceDomain()
    ssd.run()
