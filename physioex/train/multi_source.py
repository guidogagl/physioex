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
import torch
from loguru import logger

from physioex.data import PhysioExDataModule

from physioex.models import load_pretrained_model
from physioex.train.networks import get_config, register_experiment
from physioex.train.networks.base import SleepModule
from physioex.train.networks.utils.loss import config as loss_config
from physioex.train.trainer import Trainer


class MultiSourceLayer( torch.nn.Module ):
    
    def __init__( self, single_source_layer : torch.nn.Module ):
        super().__init__()
        import copy

        self.single_source_layer = single_source_layer
 
        # add a new layer to the model equal to the single source layer by cloning it
        self.multi_source_layer =  copy.deepcopy(single_source_layer)
        # initialize the weights of the new layer
        for param in self.multi_source_layer.parameters():
            torch.nn.init.zeros_(param)
                    
        # freeze the weights of the single source layer
        for param in self.single_source_layer.parameters():
            param.requires_grad = False 
    
    def forward(self, x ):
        # apply the single source
        with torch.no_grad():
            single_source_output = self.single_source_layer(x)
        # apply the multi source
        multi_source_output = self.multi_source_layer(x)        
        return single_source_output + multi_source_output



class MultiSourceModule(FineTunedModule):
    def __init__(self, module_config: Dict, module_checkpoint : str = None):
        super().__init__(module_config)
        
        # if module_checkpoint is None the model is just a FineTunedModule
        
        # else
        
        if module_checkpoint is not None:
            # load the checkpoint
            checkpoint = torch.load(module_checkpoint)
            # load the weights of the model
            self.load_state_dict(checkpoint["state_dict"])
            # freeze the weights of the model
            for param in self.parameters():
                param.requires_grad = False
        
        self.nn.epoch_encoder = MultiSourceLayer( self.nn.epoch_encoder )



def main(train_dataset=None):
    import yaml

    train_dataset = list(train_dataset)
    
    test_datasets = ["mass", "hmc", "dcsm", "mesa", "mros"]

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

    datasets = ["mass", "hmc", "dcsm", "mesa", "mros"]
    #datasets = ["mass", "hmc"]
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
            log_file = f"log/train_multi_source_{train_dataset}.log"
            screen_name = "train_" + "_".join(train_dataset)
            log_dir = f"log/multi_source/k={len(train_dataset)}/{'_'.join(train_dataset)}"
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file = f"{log_dir}/output.log"
            command = f'screen -dmS {screen_name} bash -c "python -c \\"from physioex.train.multi_source import main; main(train_dataset={train_dataset})\\" > {log_file} 2>&1"'
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
