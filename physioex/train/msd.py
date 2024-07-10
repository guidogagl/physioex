import torch
import torch.optim as optim
from loguru import logger

import pytorch_lightning as pl

import os 

from typing import Dict

from physioex.train.networks.base import SleepModule
from physioex.train.trainer import Trainer
from physioex.train.networks import register_experiment, get_config

from physioex.models import load_pretrained_model

from physioex.data.msd.msd import MultiSourceDomain as MSD
from physioex.data import datasets as dataset_config
from physioex.data.base import TimeDistributedModule

from physioex.train.networks.utils.loss import config as loss_config
import pandas as pd

from pathlib import Path
import numpy as np

from tqdm import tqdm


class FineTunedModule( SleepModule ):
    def __init__( self, module_config : Dict ):
        
        model_params = module_config['model_params']
        
        model = load_pretrained_model(
            name  = model_params["name"],
            in_channels = model_params["in_channels"],
            sequence_length= model_params["seq_len"],
        ).train()
        # check if it is a string
    
        if isinstance( module_config["loss_call"], str):
            module_config["loss_call"] = loss_config[ module_config["loss_call"] ]
        
        super().__init__( model.nn, module_config )

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        
        scheduler1 = optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
        
        # Restituisci entrambi gli scheduler in una lista di dizionari
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler1,
                "interval": "epoch",  # Esegui scheduler1 ad ogni epoca
                "frequency": 1,
            },
        }
        
class _MultiSourceDomain( MSD ):
    def __init__(self, version, picks, preprocessing, sequence_length, target_transform, train_datasets = None):

        train_datasets.insert( 0, "shhs" )
        domains = [ { "dataset" : dts, "version": version, "picks": picks } for dts in train_datasets ]
        super().__init__( 
            domains = domains,
            sequence_length = sequence_length,
            target_transform = target_transform,
            preprocessing = preprocessing,        
        )

    def get_sets(self):
        total_train_idx = []
        total_valid_idx = []
        total_test_idx = []

        for i, dataset in enumerate(self.datasets):
            train_idx, valid_idx, test_idx = dataset.get_sets()

            total_valid_idx.append(valid_idx + self.offsets[i])

            if i!=0 :
                total_train_idx.append(train_idx + self.offsets[i])
                total_test_idx.append(test_idx + self.offsets[i])

        return (
            np.concatenate(total_train_idx),
            np.concatenate(total_valid_idx),
            np.concatenate(total_test_idx),
        )


if __name__ == "__main__":
    import yaml
    import argparse
    
    from physioex.data.constant import set_data_folder
    
    set_data_folder( "/home/guido/shared/")

    with open("multi-source-domain.yaml", "r") as f:
        experiment_config = yaml.safe_load(f)

    datasets = [ "hpap", "mass", "dcsm", "sleep_edf", "svuh", "isruc", "hmc" ]
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-datasets", type=str, default = "", required = False)
    args = parser.parse_args()

    if args.train_datasets == "":
        train_datasets = []
        k = 0
    else:    
        train_datasets = args.train_datasets.split("-")
        k = len( train_datasets )
    
    with open("multi-source-domain.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if k > 0:
        def MultiSourceDomain( version, picks, preprocessing, sequence_length, target_transform ):
            return _MultiSourceDomain( 
                version = version,
                picks = picks,
                sequence_length = sequence_length,
                target_transform = target_transform,
                preprocessing = preprocessing,        
                train_datasets = train_datasets,
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
            
            # calcola il val_check_interval in modo che ci siano sempre almeno 3 valutazioni
            
            logger.info( "Computing val_check_interval" )
            dataset = MultiSourceDomain(
                version=None,
                picks = ["EEG"],
                preprocessing = config["experiment"]["input_transform"],
                sequence_length = config["experiment"]["module_config"]["model_params"]["seq_len"],
                target_transform = config["experiment"]["target_transform"],
            ) 
            dataset.split(0)
            dataset = TimeDistributedModule( 
                dataset = dataset, 
                batch_size = config["batch_size"],
                fold = 0
                )
            
            num_train_steps = len( dataset.train_dataloader() )
            val_check_interval = max( 1, num_train_steps // 3 )
            
            Trainer(
                model_name = exp,
                dataset_name = "multi-source-domain",
                version = None,
                sequence_length = config["experiment"]["module_config"]["model_params"]["seq_len"],
                picks = "EEG",
                ckp_path = checkpoint_dir,
                max_epoch = config["max_epoch"],
                val_check_interval = val_check_interval,
                batch_size = config["batch_size"],
                n_jobs = 1,
            ).run()

        checkpoint = [ file for file in os.listdir(checkpoint_dir) if file.endswith(".ckpt") ][0] 
        checkpoint = os.path.join( checkpoint_dir, checkpoint )
        
        logger.info( f"Checkpoint: {checkpoint}" )
        model = FineTunedModule.load_from_checkpoint( checkpoint, module_config = config["experiment"]["module_config"] )

    else:
    
        checkpoint_dir = f"{config['checkpoint']}k={k}/"
        Path( checkpoint_dir ).mkdir( parents = True, exist_ok = True)
        
        model = load_pretrained_model(
            name  = "seqsleepnet",
            in_channels = 1,
            sequence_length= 21,
        )
    
    
    trainer = pl.Trainer(
        deterministic=True,
    )
    
    results = []
    
    for dts in datasets:
        dataset = dataset_config[ dts ](
            version = None,
            picks = ["EEG"],
            preprocessing = config["experiment"]["input_transform"],
            sequence_length = config["experiment"]["module_config"]["model_params"]["seq_len"],
            target_transform = config["experiment"]["target_transform"],
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
        
    