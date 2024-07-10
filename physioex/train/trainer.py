import copy
import uuid
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

# from joblib import Parallel, delayed
from lightning.pytorch import seed_everything
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger

import wandb
from physioex.data import TimeDistributedModule, get_datasets
from physioex.train.networks import get_config
from physioex.train.networks.utils.loss import config as loss_config

torch.set_float32_matmul_precision("medium")


class Trainer:
    def __init__(
        self,
        model_name: str = "chambon2018",
        dataset_name: str = "sleep_physioex",
        version: str = "2018",
        sequence_length: int = 3,
        picks: list = ["Fpz-Cz"],
        loss_name: str = "cel",
        ckp_path: str = None,
        max_epoch: int = 20,
        val_check_interval: int = 300,
        batch_size: int = 32,
        n_jobs: int = 10,
        imbalance: bool = False,
        use_wandb: bool = False,
    ):

        seed_everything(42, workers=True)

        datasets = get_datasets()
        config = get_config()

        self.use_wandb = use_wandb

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.loss_name = loss_name

        self.dataset_call = datasets[dataset_name]
        self.model_call = config[model_name]["module"]
        self.input_transform = config[model_name]["input_transform"]
        self.target_transform = config[model_name]["target_transform"]
        self.module_config = config[model_name]["module_config"]
        self.module_config["seq_len"] = sequence_length

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval
        self.version = version
        self.n_jobs = n_jobs

        self.imbalance = imbalance

        if ckp_path is None:
            self.ckp_path = "models/" + str(uuid.uuid4()) + "/"
        else:
            self.ckp_path = ckp_path

        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)

        picks = picks.split(" ")
        self.module_config["in_channels"] = len(picks)

        logger.info("Loading dataset")
        self.dataset = self.dataset_call(
            version=self.version,
            picks=picks,
            preprocessing=self.input_transform,
            sequence_length=sequence_length,
            target_transform=self.target_transform,
        )

        logger.info("Dataset loaded")

        self.folds = list(range(self.dataset.get_num_folds()))

        self.module_config["loss_call"] = loss_config[loss_name]
        self.module_config["loss_params"] = dict()

    def train_evaluate(self, fold: int = 0):

        dataset = self.dataset

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=dataset,
            batch_size=self.batch_size,
            fold=fold,
        )

        module = self.model_call(module_config=self.module_config)

        # Definizione delle callback
        if self.imbalance:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_f1",
                save_top_k=1,
                mode="max",
                dirpath=self.ckp_path,
                filename="fold=%d-{epoch}-{step}-{val_f1:.2f}" % fold,
                save_weights_only=False,
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_acc",
                save_top_k=1,
                mode="max",
                dirpath=self.ckp_path,
                filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold,
                save_weights_only=False,
            )

        progress_bar_callback = RichProgressBar()

        if self.use_wandb:
            my_logger = WandbLogger(
                group=self.dataset_name,
                project=self.model_name,
                name=f"{self.model_name}-{self.dataset_name}-v.{self.version}-fold={fold}",
                log_model="False",
                entity="ggagliar-sleep",
            )
        else:
            my_logger = CSVLogger(save_dir=self.ckp_path)

        # Configura il trainer con le callback
        trainer = pl.Trainer(
            devices="auto",
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True,
            logger=my_logger,
            num_sanity_val_steps = -1
        )

        if self.use_wandb:
            my_logger.log_hyperparams(
                {
                    "model": self.model_name,
                    "dataset": self.dataset_name,
                    "version": self.version,
                    "monitor": "val_acc" if not self.imbalance else "val_f1",
                    "fold": fold,
                    "batch_size": self.batch_size,
                    "loss": self.loss_name,
                }
            )

        logger.info("JOB:%d-Training model" % fold)
        #trainer.validate(module, datamodule.val_dataloader())
        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=datamodule)

        # Salva il modello come un artefatto
        if self.use_wandb:
            artifact = wandb.Artifact(
                f"{self.model_name}-{self.dataset_name}-v.{self.version}",
                type=self.model_name,
            )
            artifact.add_file(checkpoint_callback.best_model_path)
            my_logger.experiment.log_artifact(artifact)

        logger.info("JOB:%d-Evaluating model" % fold)
        val_results = trainer.test(
            ckpt_path="best", dataloaders=datamodule.val_dataloader()
        )
        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

        return {"val_results": val_results, "test_results": test_results}

    def run(self):
        logger.info("Jobs pool spawning")

        results = [self.train_evaluate(fold) for fold in self.folds]

        logger.info("Results successfully collected from jobs pool")

        try:
            val_results = pd.DataFrame([result["val_results"][0] for result in results])
            test_results = pd.DataFrame(
                [result["test_results"][0] for result in results]
            )

            val_results.to_csv(self.ckp_path + "val_results.csv", index=False)
            test_results.to_csv(self.ckp_path + "test_results.csv", index=False)
        except Exception as e:
            logger.error(f"Error while saving results: {e}")
            raise e

        logger.info("Results successfully saved in %s" % self.ckp_path)
