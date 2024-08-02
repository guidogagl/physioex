from typing import List

import uuid
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from lightning.pytorch import seed_everything
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from physioex.data import PhysioExDataModule, PhysioExDataset, get_datasets
from physioex.train.networks import get_config
from physioex.train.networks.utils.loss import config as loss_config

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)

class Trainer:
    def __init__(
        self,
        datasets: List[str] = ["mass"],
        versions: List[str] = None,
        batch_size: int = 32,
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        #task: str = "sleep",
        data_folder: str = None,
        
        random_fold : bool = False,
        
        model_name: str = "chambon2018",
        
        loss_name: str = "cel",
        
        ckp_path: str = None,
        max_epoch: int = 20,
        val_check_interval: int = 3,
    ):   
        ###### module setup ######
        network_config = get_config()[model_name]

        module_config = network_config["module_config"]
        module_config["seq_len"] = sequence_length
        module_config["loss_call"] = loss_config[loss_name] 
        module_config["loss_params"] = dict()
        module_config["in_channels"] = len(selected_channels)
        
        self.model_call = network_config["module"]
        self.module_config = module_config    
        
        ###### datamodule setup ######
        
        if random_fold :
            self.folds = [ -1 ]
        else:
            self.folds = PhysioExDataset(
                datasets=datasets,
                versions=versions,
                preprocessing=network_config["input_transform"],
                selected_channels=selected_channels,
                sequence_length=sequence_length,
                target_transform = network_config["target_transform"],
                data_folder=data_folder,
            ).get_num_folds()
        
            self.folds = list(range(self.folds))
        
        num_steps = PhysioExDataset(
            datasets=datasets,
            versions=versions,
            preprocessing=network_config["input_transform"],
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            target_transform = network_config["target_transform"],
            data_folder=data_folder,
        ).__len__() // batch_size
        
        val_check_interval = max(1, num_steps // val_check_interval)
        
        self.datasets = datasets
        self.versions = versions
        self.batch_size = batch_size
        self.preprocessing = network_config["input_transform"]
        self.selected_channels = selected_channels
        self.sequence_length = sequence_length
        self.data_folder = data_folder
        self.target_transform = network_config["target_transform"] 
                
        ##### trainer setup #####
    

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval
        
        self.ckp_path = ckp_path if ckp_path is not None else "models/" + str(uuid.uuid4()) + "/"
        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)
        
        #############################



    def train_evaluate(self, fold: int = 0):

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        
        #### datamodules setup ####
        
        train_datamodule = PhysioExDataModule(
            datasets=self.datasets,
            versions=self.versions,
            folds=fold,
            batch_size=self.batch_size,
            selected_channels=self.selected_channels,
            sequence_length=self.sequence_length,
            data_folder=self.data_folder,
            preprocessing = self.preprocessing,
            target_transform= self.target_transform,
        )
        
        ###### module setup ######
        
        module = self.model_call(module_config=self.module_config)

        
        ###### trainer setup ######
        
        # Definizione delle callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            dirpath=self.ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold,
            save_weights_only=False,
        )

        progress_bar_callback = RichProgressBar()

        my_logger = CSVLogger(save_dir=self.ckp_path)

        # Configura il trainer con le callback
        trainer = pl.Trainer(
            devices="auto",
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True,
            logger=my_logger,
            # num_sanity_val_steps = -1
        )

        ###### training ######
        
        logger.info("JOB:%d-Training model" % fold)
        # trainer.validate(module, datamodule.val_dataloader())
        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=train_datamodule)

        logger.info("JOB:%d-Evaluating model" % fold)
        val_results = trainer.test(
            ckpt_path="best", dataloaders=train_datamodule.val_dataloader()
        )[0]
        
        val_results["fold"] = fold
        
        test_results = trainer.test(ckpt_path="best", datamodule=train_datamodule)[0]

        return {"val_results": val_results, "test_results": test_results}

    def run(self):

        results = [self.train_evaluate(fold) for fold in self.folds]

        val_results = pd.DataFrame([result["val_results"] for result in results])
        test_results = pd.DataFrame(
            [result["test_results"] for result in results]
        )

        val_results.to_csv(self.ckp_path + "val_results.csv", index=False)
        test_results.to_csv(self.ckp_path + "test_results.csv", index=False)

        logger.info("Results successfully saved in %s" % self.ckp_path)
