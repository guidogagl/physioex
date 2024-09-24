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
    """
    Fine-tunes a pre-trained model using the provided datasets and configuration.

    Args:
        datasets (Union[List[str], str, PhysioExDataModule]): The datasets to be used for fine-tuning. Can be a list of dataset names, a single dataset name, or a PhysioExDataModule instance.
        datamodule_kwargs (dict, optional): Additional keyword arguments to be passed to the PhysioExDataModule. Defaults to {}.
        model (Union[dict, SleepModule], optional): The model to be fine-tuned. If provided, `model_class`, `model_config`, and `model_checkpoint` are ignored. Defaults to None.
        model_class (type, optional): The class of the model to be fine-tuned. Required if `model` is not provided. Defaults to None.
        model_config (dict, optional): The configuration dictionary for the model. Required if `model` is not provided. Defaults to None.
        model_checkpoint (str, optional): The path to the checkpoint from which to load the model. Required if `model` is not provided. Defaults to None.
        learning_rate (float, optional): The learning rate to be set for fine-tuning. If `None`, the learning rate is not updated. Default is 1e-7.
        weight_decay (Union[str, float], optional): The weight decay to be set for fine-tuning. If `None`, the weight decay is not updated. If "auto", it is set to 10% of the learning rate. Default is "auto".
        train_kwargs (Dict, optional): Additional keyword arguments to be passed to the `train` function. Defaults to {}.

    Returns:
        str: The path of the best model checkpoint.

    Raises:
        ValueError: If `model` is `None` and any of `model_class`, `model_config`, or `model_checkpoint` are also `None`.
        ValueError: If `model` is not a dictionary or a `SleepModule`.

    Notes:
        - Models cannot be fine-tuned from scratch; they must be loaded from a checkpoint or be a pre-trained model from `physioex.models`.
        - Typically, when fine-tuning a model, you want to set up the learning rate.
    """
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