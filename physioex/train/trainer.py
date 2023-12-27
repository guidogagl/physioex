import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch import seed_everything
from pathlib import Path

from physioex.train.networks import config 
from physioex.data import datasets, TimeDistributedModule

import uuid

from joblib import Parallel, delayed
import pandas as pd

import copy

folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


from ast import literal_eval
from loguru import logger

import torch
torch.set_float32_matmul_precision('medium')


class Trainer():
    def __init__(self, 
            model_name : str = "chambon2018", 
            dataset_name : str = "sleep_physioex", 
            ckp_path : str = None,
            version : str = "2018", 
            use_cache : bool = True, 
            sequence_lenght : int = 3, 
            max_epoch : int = 20, 
            val_check_interval : int = 300, 
            batch_size : int = 32,
            n_jobs : int = 10
        ):

        seed_everything(42, workers=True)

        self.dataset_call = datasets[dataset_name]
        
        self.model_call = config[model_name]["module"]
        self.input_transform = config[model_name]["input_transform"]
        self.target_transform = config[model_name]["target_transform"]
        self.module_config = config[model_name]["module_config"]
        self.module_config["seq_len"] = sequence_lenght

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval
        self.version = version
        self.use_cache = use_cache 
        self.n_jobs = n_jobs

        if ckp_path is None:
            self.ckp_path = "models/" + str(uuid.uuid4()) + "/"
        else:
            self.ckp_path = ckp_path
             
        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading dataset")
        self.dataset = self.dataset_call( version=self.version, use_cache=self.use_cache)
        logger.info("Dataset loaded")

    def train_evaluate(self, fold : int = 0):

        module = self.model_call( module_config = self.module_config )

        dataset = self.dataset

        logger.info("JOB:%d-Splitting dataset into train, validation and test sets" % fold)
        dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset = dataset, 
            sequence_lenght = self.module_config["seq_len"], 
            batch_size = self.batch_size, 
            transform = self.input_transform, 
            target_transform = self.target_transform
        )

        # Definizione delle callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            save_top_k=1,
            mode="max",
            dirpath=self.ckp_path,
            filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold
        )

        progress_bar_callback = RichProgressBar()

        # Configura il trainer con le callback
        trainer = pl.Trainer(
            max_epochs=self.max_epoch,
            val_check_interval=self.val_check_interval,
            callbacks=[checkpoint_callback, progress_bar_callback],
            deterministic=True
        )

        logger.info("JOB:%d-Training model" % fold)
        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=datamodule)
        
        logger.info("JOB:%d-Evaluating model" % fold)
        val_results = trainer.test(ckpt_path="best", dataloaders=datamodule.val_dataloader())
        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

        return {'val_results': val_results, 'test_results': test_results} 
    
    def run(self):
        logger.info("Jobs pool spawning")
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_evaluate)(fold) for fold in folds
        )    
        
        logger.info("Results successfully collected from jobs pool")

        try:
            val_results = pd.DataFrame([result["val_results"][0]    for result in results]) 
            test_results = pd.DataFrame([result["test_results"][0]  for result in results])

            val_results.to_csv(self.ckp_path + 'val_results.csv', index=False)
            test_results.to_csv(self.ckp_path + 'test_results.csv', index=False)
        except Exception as e:
            logger.error(f"Error while saving results: {e}")
            raise e

        logger.info("Results successfully saved in %s" % self.ckp_path)
