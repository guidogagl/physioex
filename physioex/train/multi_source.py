import itertools
import subprocess
import time
from itertools import combinations

from loguru import logger


import os
from pathlib import Path
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch.optim as optim
from loguru import logger

from physioex.data import PhysioExDataModule

from physioex.models import load_pretrained_model
from physioex.train.networks import get_config, register_experiment
from physioex.train.networks.base import SleepModule
from physioex.train.networks.utils.loss import config as loss_config
from physioex.train.trainer import Trainer


class FineTunedModule(SleepModule):
    def __init__(self, module_config: Dict):

        model_params = module_config["model_params"]

        model = load_pretrained_model(
            name=model_params["name"],
            in_channels=model_params["in_channels"],
            sequence_length=model_params["seq_len"],
        ).train()
        # check if it is a string

        if isinstance(module_config["loss_call"], str):
            module_config["loss_call"] = loss_config[module_config["loss_call"]]

        super().__init__(model.nn, module_config)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-7, weight_decay=1e-6)

        scheduler_exp = optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
        scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.1, patience=10, verbose=True
        )

        # Restituisci entrambi gli scheduler in una lista di dizionari
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler_exp,
                "interval": "epoch",  # Esegui scheduler_exp ad ogni epoca
                "frequency": 1,
            },
            "lr_scheduler_plateau": {
                "scheduler": scheduler_plateau,
                "interval": "epoch",  # Esegui scheduler_plateau ad ogni epoca
                "frequency": 1,
                "monitor": "val_loss",  # Necessario per ReduceLROnPlateau
            },
        }


def main(train_dataset=None):
    import yaml

    train_dataset = list(train_dataset)
    
    test_datasets = ["mass", "hpap", "dcsm", "mesa", "mros"]

    with open("multi-source-domain.yaml", "r") as f:
        config = yaml.safe_load(f)

    k = len(train_dataset)

    if k > 0 :

        train_dataset_name = "_".join(train_dataset)
        checkpoint_dir = f"{config['checkpoint']}k={k}/{train_dataset_name}/"
        
        try:
            # check if there is a checkpoint in the directory
            checkpoint = [
                file for file in os.listdir(checkpoint_dir) if file.endswith(".ckpt")
            ][0]
            logger.info(f"Checkpoint found: {checkpoint}")
        except:
            logger.info(f"No checkpoint found in {checkpoint_dir}")
            checkpoint = None
        
        if checkpoint is None:
            # train the model on the training datasets

            exp = register_experiment("multi-source-domain.yaml")
            
            Trainer(
                model_name = exp,
                datasets = train_dataset,
                batch_size = config["batch_size"],
                sequence_length=config["experiment"]["module_config"]["model_params"][
                        "seq_len"
                    ],
                ckp_path=checkpoint_dir,
                max_epoch=config["max_epoch"],
                
                val_check_interval=config["val_check_interval"],
                
                random_fold = True,
            ).run()  

            checkpoint = [
                file for file in os.listdir(checkpoint_dir) if file.endswith(".ckpt")
            ][0]
        
        
        model = FineTunedModule.load_from_checkpoint(
                checkpoint, module_config=config["experiment"]["module_config"]
            )
    
    else :
        
        checkpoint_dir = f"{config['checkpoint']}k={k}/"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        model = load_pretrained_model(
            name="seqsleepnet",
            in_channels=1,
            sequence_length=21,
        )

    trainer = pl.Trainer(
        deterministic=True,
    )

    results = []

    for dts in test_datasets:
        test_datamodule = PhysioExDataModule(
            datasets = dts,
            batch_size= config["batch_size"],
            selected_channels= ["EEG"],
            sequence_length=config["experiment"]["module_config"]["model_params"][
                        "seq_len"
                    ],
            preprocessing = config["experiment"]["module_config"]["input_transform"],
        )

        results.append(trainer.test(model, datamodule=test_datamodule)[0])
        results[-1]["dataset"] = dts
        results[-1]["split"] = "train" if dts in train_dataset else "test"

    results = pd.DataFrame(results)
    results.to_csv(f"{checkpoint_dir}/results.csv", index=False)



def chunked_iterable(iterable, size):
    """Divide un iterable in chunk di data dimensione."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


if __name__ == "__main__":

    # datasets = ["hpap", "mass", "dcsm", "sleep_edf", "svuh", "isruc", "hmc"]
    datasets = ["hpap", "mass", "dcsm", "sleep_edf", "hmc"]
    N = 10  # Numero di sessioni di screen da eseguire in parallelo

    train_datasets = [
        combo
        for r in range(1, len(datasets) + 1)
        for combo in combinations(datasets, r)
    ]
    logger.info(f"Generated {len(train_datasets)} combinations.")

    # take the first half of the combinations
    # train_datasets = train_datasets[:len(train_datasets) // 2]
    # take the second half of the combinations
    # train_datasets = train_datasets[len(train_datasets) // 2:]

    for group in chunked_iterable(train_datasets, N):
        screen_sessions = []
        for train_dataset in group:
            screen_name = "train_" + "_".join(train_dataset)
            command = f'screen -dmS {screen_name} python -c "from physioex.train.multi_source import main; main(train_dataset={train_dataset})"'
            subprocess.run(command, shell=True)
            screen_sessions.append(screen_name)
            logger.info(f"Launched main in screen session: {screen_name}")

        # Aspetta che tutte le sessioni di questo gruppo terminino
        all_done = False
        while not all_done:
            all_done = True
            for session in screen_sessions:
                # Controlla se la sessione è ancora attiva
                result = subprocess.run(
                    f"screen -list | grep {session}", shell=True, stdout=subprocess.PIPE
                )
                if result.stdout:
                    all_done = False
                    break  # Una sessione è ancora attiva, quindi aspetta e poi controlla di nuovo
            if not all_done:
                time.sleep(60)  # Aspetta 5 secondi prima di controllare di nuovo

        logger.info(f"All sessions in group completed.")
