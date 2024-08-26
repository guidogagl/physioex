from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from physioex.models import load_pretrained_model

from physioex.train.networks.base import SleepModule

module_config = dict()

class FineTunedModule(SleepModule):
    def __init__(self, module_config: Dict):
        
        self.n_train = ( module_config["n_train"] - 1 )  / 4

        model = load_pretrained_model(
            name=module_config["model_name"],
            in_channels=module_config["in_channels"],
            sequence_length=module_config["seq_len"],
        ).train()

        super().__init__(model.nn, module_config)

    def configure_optimizers(self):
        
        # depending on how big is n_train the learning rate and weight decay are adjusted
        # min is 1e-7 and max is 1e-4
        
        lr = 1e-7 + ( (1e-4 - 1e-7) * self.n_train )
        weight_decay = lr / 10
                  
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        
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
