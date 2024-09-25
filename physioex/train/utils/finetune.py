from typing import List, Union, Dict, Type
import os
from pathlib import Path
import torch

import uuid

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from physioex.data import PhysioExDataModule
from physioex.models import load_pretrained_model
from physioex.train.networks.base import SleepModule
from physioex.train.utils.train import train

# the finetune function is similar to the train function but it has a few differences:
# - models can't be finetuned from scratch, they must be loaded from a checkpoint, or being a pretrained model from physioex.models
# tipycally when fine-tuning a model you want to setup the learning rate

def finetune(
    model: Union[SleepModule, Dict] = None,  # if passed model_class, model_config and resume are ignored
    model_class: Type[SleepModule] = None,
    model_config: Dict = None,
    model_checkpoint: str = None,
    learning_rate: float = 1e-7, # if None not updated
    weight_decay: Union[str, float] = "auto", # if None not updated
    train_kwargs: Dict = {},
    ) -> str:

    if model is None:
        # if model is None, model_class and model_config must be passed
        if model_class is None or model_config is None or model_checkpoint is None:
            raise ValueError("model, model_class and model_config must be passed if model is None")
        
        model = model_class.load_from_checkpoint(model_checkpoint, module_config=model_config)
    
    else:
        
        if isinstance( model, dict):
            # the model is a dictionary so it links to a pretrained model of the library
            model=load_pretrained_model(**model)
        elif isinstance(model, SleepModule):
            # the model is a torch.nn.Module
            pass
        else:
            raise ValueError("model must be a dictionary or a torch.nn.Module")
        
    # setup the learning rate of the model    
    if learning_rate is not None:
        model.learning_rate = learning_rate
    
    if weight_decay is not None:
        if weight_decay == "auto":
            model.weight_decay = model.learning_rate * 0.1
        else:
            model.weight_decay = weight_decay    
    
    model.configure_optimizers()
         
    train_kwargs["resume"] = False
    
    return train(
        model=model,
        **train_kwargs
    )