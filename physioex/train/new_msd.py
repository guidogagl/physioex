import torch
import torch.optim as optim
from loguru import logger

import pytorch_lightning as pl

import os 

from typing import List, Tuple

from physioex.train.networks.base import SleepModule
from physioex.train.trainer import Trainer
from physioex.train.networks import register_experiment

from physioex.models import load_pretrained_model

from physioex.data.msd.msd import MultiSourceDomain as MSD
from physioex.data import datasets as dataset_config
from pysioex.data.base import TimeDistributedModule

import pandas as pd

class FineTunedModule( SleepModule ):
    def __init__( self, module_config : Dict ):
        
        model_params = module_config['model_params']
        
        model = load_pretrained_model(
            name  = model_params["name"],
            in_channels = model_params["in_channels"],
            sequence_length= model_params["seq_len"],
        )
        
        super().__init__( model.nn, module_config )
    
    def on_train_start(self):
            # Esegui un passo di validazione all'inizio del training
            self.trainer.validate(self, dataloaders=self.val_dataloader())
    
    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.nn.parameters(),
            lr=1e-8,
            weight_decay=1e-7,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="max",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": self.opt,
            "lr_scheduler": {"scheduler": self.scheduler, "monitor": "val_acc"},
        }
    
if __name__ == "__main__":
    import yaml
    import argparse

    with open("multi-source-domain.yaml", "r") as f:
        experiment_config = yaml.safe_load(f)

    datasets = [ "hpap", "mass", "dcsm" ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-datasets", type=str, required=True)
    args = parser.parse_args()

    train_datasets = args.train_datasets.split("-")

    k = len( train_datasets )
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    def MultiSourceDomain( version, picks, preprocessing, sequence_length, target_transform ):
        return MSD( 
            domains = [ { "dataset" : dts, "version": version, "picks": picks } for dts in train_datasets ],
            sequence_length = sequence_length,
            target_transform = target_transform,
            preprocessing = preprocessing,        
        )

    dataset_config[ "multi-source-domain"] = MultiSourceDomain
    
    checkpoint_dir = f"{config['checkpoint']}k={k}/{args.train_datasets}/"
    
    try:
        # check if there is a checkpoint in the directory
        checkpoint = [ file for file in os.listdir(checkpoint_dir) if file.endswith(".ckpt") ][0]        
        logger.info( f"Checkpoint found: {checkpoint}" )    
    except:
        logger.info( f"No checkpoint found in {checkpoint_dir}" )
        
        exp = register_experiment( "multi-source-domain.yaml" )
        
        Trainer(
            model_name = exp,
            dataset_name = "multi-source-domain",
            version = None,
            sequence_length = config["experiment"]["module_config"]["model_params"]["seq_len"],
            picks = ["EEG"],
            ckp_path = checkpoint_dir,
            max_epoch = config["max_epoch"],
            val_check_interval = config["val_check_interval"],
            batch_size = config["batch_size"],
            n_jobs = 1,
        ).run()
    
    
    checkpoint = [ file for file in os.listdir(checkpoint_dir) if file.endswith(".ckpt") ][0] 
    checkpoint = os.path.join( checkpoint_dir, checkpoint )
    
    logger.info( f"Checkpoint: {checkpoint}" )
    model = FineTunedModule.load_from_checkpoint( checkpoint, module_config = config["experiment"]["module_config"] )
    
    trainer = pl.Trainer(
        deterministic=True,
    )
    
    results = []
    
    for dts in datasets:
        dataset = dataset_config[ dts ](
            version = None,
            picks = ["EEG"],
            preprocessing = config["experiment"]["module_config"]["input_transform"],
            sequence_length = config["experiment"]["module_config"]["model_params"]["seq_len"],
            target_transform = config["experiment"]["module_config"]["target_transform"],
        )
        
        dataset.split(0)
        
        dataset = TimeDistributedModule( 
            dataset = dataset, 
            batch_size = config["batch_size"],
            fold = 0
            )


        dataloader = dataset.test_dataloader()
        
        results.append( trainer.test( model, dataloaders = dataloader )[0] )
        results[-1]["dataset"] = dts
        results[-1]["split"] = "train" if dts in train_datasets else "test"
    
    results = pd.DataFrame( results )
    results.to_csv( f"{checkpoint_dir}/results.csv", index = False )   
        
    