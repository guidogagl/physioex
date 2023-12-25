import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch import seed_everything

from pathlib import Path

from physioex.train.networks import models, module_config, input_transform, target_transform
from physioex.data import datasets, TimeDistributedModule

import uuid

from concurrent.futures import ProcessPoolExecutor
import pandas as pd


folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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

        self.model_call = models[model_name]
        self.dataset_call = datasets[dataset_name]
        self.input_transform = input_transform[model_name]
        self.target_transform = target_transform[model_name]

        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.val_check_interval = val_check_interval
        self.version = version
        self.use_cache = use_cache 
        self.n_jobs = n_jobs

        self.module_config = dict( module_config )
        self.module_config["seq_len"] = sequence_lenght

        if ckp_path is None:
            self.ckp_path = "models/" + str(uuid.uuid4()) + "/"
        else:
            self.ckp_path = ckp_path
             
        Path(self.ckp_path).mkdir(parents=True, exist_ok=True)

    def train_evaluate(self, fold : int = 0):

        module = self.model_call( module_config = self.module_config )
        dataset = self.dataset_call( version=self.version, use_cache=self.use_cache)
        dataset.split( fold )

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

        # Addestra il modello utilizzando il trainer e il DataModule
        trainer.fit(module, datamodule=datamodule)
        val_results = trainer.test(ckpt_path="best", dataloaders=datamodule.val_dataloader())
        test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

        return val_results, test_results
    
    def run(self):
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(self.train_evaluate, fold) for fold in folds]
            
        # Raccogli i risultati da ogni future
        results = [future.result() for future in futures]

        # Salva i risultati di validazione e test in due file csv
        val_results = pd.DataFrame([result['val_results'] for result in results])
        test_results = pd.DataFrame([result['test_results'] for result in results])

        val_results.to_csv(self.ckp_path + 'val_results.csv', index=False)
        test_results.to_csv(self.ckp_path + 'test_results.csv', index=False)
