import copy
import torch

from physioex.train.networks.base import SleepModule
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.models import load_pretrained_model

from physioex.train.networks.base import SleepModule

module_config = dict()

class MultiSourceLayer( torch.nn.Module ):
    
    def __init__( self, single_source_layer : torch.nn.Module ):
        super().__init__()

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



class MultiSourceModule(SleepModule):
    def __init__(self, module_config: Dict):

        model = load_pretrained_model(
            name=module_config["model_name"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).train()
        
        super().__init__(model.nn, module_config)                
        
        if module_config[ "module_checkpoint" ] is not None:
            # load the checkpoint
            checkpoint = torch.load(module_config[ "module_checkpoint" ])
            # load the weights of the model
            self.load_state_dict(checkpoint["state_dict"])
                
        self.nn.epoch_encoder = MultiSourceLayer( self.nn.epoch_encoder )

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
