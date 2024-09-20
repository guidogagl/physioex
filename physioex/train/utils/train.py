from typing import List, Union
import os
from pathlib import Path
import torch
from loguru import logger
import uuid

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from physioex.data import PhysioExDataModule
from physioex.train.networks.base import SleepModule

import pandas as pd 

def train(
    datasets : Union[ List[str], str, PhysioExDataModule],
    datamodule_kwargs : dict = {},
    model : SleepModule = None, # if passed model_class, model_config and resume are ignored
    model_class = None,
    model_config : dict = None,
    batch_size : int = 128,
    fold : int = -1,
    hpc : bool = False,
    num_validations : int = 10,
    checkpoint_path : str = None,
    max_epochs : int = 10,
    num_nodes : int = 1,
    resume: bool = True,
    ) -> str:
    
    """
    Trains a model using the provided datasets and configuration.

    Parameters
    ----------
    datasets : Union[List[str], str, PhysioExDataModule]
        The datasets to be used for training. Can be a list of dataset names, a single dataset name, or a PhysioExDataModule instance.
    datamodule_kwargs : dict, optional
        Additional keyword arguments to be passed to the PhysioExDataModule.
    model : SleepModule, optional
        The model to be trained. If provided, `model_class`, `model_config`, and `resume` are ignored.
    model_class : type, optional
        The class of the model to be trained. Required if `model` is not provided.
    model_config : dict, optional
        The configuration dictionary for the model. Required if `model` is not provided.
    batch_size : int, optional
        The batch size to be used for training. Default is 128.
    fold : int, optional
        The fold index for cross-validation. Default is -1.
    hpc : bool, optional
        Whether to use high-performance computing (HPC) settings. Default is False.
    num_validations : int, optional
        The number of validation steps per epoch. Default is 10.
    checkpoint_path : str, optional
        The path to save the model checkpoints. If None, a new path is generated. Default is None.
    max_epochs : int, optional
        The maximum number of epochs for training. Default is 10.
    num_nodes : int, optional
        The number of nodes to be used for distributed training. Default is 1.
    resume : bool, optional
        Whether to resume training from the last checkpoint. Default is True.

    Returns
    -------
    str
        The path to the best model checkpoint.

    Raises
    ------
    ValueError
        If `datasets` is not a list, a string, or a PhysioExDataModule instance.
        If `model` is None and any of `model_class` or `model_config` are also None.

    Notes
    -----
    - The function sets up the data module, model, and trainer, and then starts the training process.
    - If `resume` is True and a checkpoint is found, training resumes from the last checkpoint.
    - The function returns the path to the best model checkpoint based on validation accuracy.
    """
    
    datamodule_kwargs["batch_size"] = batch_size
    datamodule_kwargs["hpc"] = hpc
    datamodule_kwargs["folds"] = fold
    
    if checkpoint_path is None:
        checkpoint_path = "models/" + str(uuid.uuid4())
    
    # check if the path exists and in case create it
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    
    ##### DataModule Setup #####
    if isinstance(datasets, PhysioExDataModule):
        datamodule = datasets
    elif isinstance(datasets, str):
        datamodule = PhysioExDataModule(
            datasets=[datasets],
            **datamodule_kwargs,
        )
    elif isinstance(datasets, list):
        datamodule = PhysioExDataModule(
            datasets=datasets,
            **datamodule_kwargs,
        ) 
    else:
        raise ValueError("datasets must be a list, a string or a PhysioExDataModule")
        
    
    
    ########### Resuming Model if needed else instantiate it ############:
    if resume and (model is None):
        chekpoints = list(Path(checkpoint_path).glob("*.ckpt"))
        if len(chekpoints) > 0:
            # read the lightning_logs/version_XX/metrics.csv file
            metrics = os.listdir(os.path.join(checkpoint_path, "lightning_logs"))
            # find the last version
            version = sorted(metrics)[-1]
            metrics = pd.read_csv(os.path.join(checkpoint_path, "lightning_logs", version, "metrics.csv"))
            
            # get the max_epoch from the metrics file
            interruption_epoch = max(metrics["epoch"])
            
            if interruption_epoch < max_epochs:
                max_epochs = max_epochs - interruption_epoch
            else:
                return chekpoints[0]
            
            logger.info(f"Resuming training from epoch {interruption_epoch}")
            
            model = model_class.load_from_checkpoint(chekpoints[0], module_config=model_config)
    
    if model is None:
        model = model_class(module_config=model_config)

    ########### Callbacks ############    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath=checkpoint_path,
        filename="fold=%d{epoch}-{step}-{val_acc:.2f}" % fold,
        save_weights_only=False,
    )
    progress_bar_callback = RichProgressBar()
    my_logger = CSVLogger(save_dir=checkpoint_path)
    
    
    ########### Trainer Setup ############
    num_steps = datamodule.dataset.__len__() * 0.7 // batch_size
    val_check_interval = max(1, num_steps // num_validations)

    trainer = Trainer(
        devices="auto",
        strategy="ddp" if hpc and num_nodes > 1 else "auto",
        num_nodes=num_nodes if hpc else 1,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        callbacks=[checkpoint_callback, progress_bar_callback],
        deterministic=True,
        logger=my_logger,
    )
    
    # Start training
    trainer.fit(model, datamodule=datamodule)
    
    return checkpoint_callback.best_model_path