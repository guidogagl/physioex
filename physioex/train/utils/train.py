import os
import uuid
from pathlib import Path
from typing import List, Type, Union

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision

from physioex.data import PhysioExDataModule
from physioex.train.networks.base import SleepModule


def train(
    datasets: Union[List[str], str, PhysioExDataModule],
    datamodule_kwargs: dict = {},
    model: SleepModule = None,  # if passed model_class, model_config and resume are ignored
    model_class: Type[SleepModule] = None,
    model_config: dict = None,
    batch_size: int = 128,
    fold: int = -1,
    hpc: bool = False,
    num_validations: int = 10,
    checkpoint_path: str = None,
    max_epochs: int = 10,
    num_nodes: int = 1,
    resume: bool = True,
    monitor: str = "val_acc",
    mode: str = "max",
) -> str:

    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    datamodule_kwargs["batch_size"] = batch_size
    datamodule_kwargs["folds"] = fold
    datamodule_kwargs["num_nodes"] = num_nodes

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
        
            try:
                # read the lightning_logs/version_XX/metrics.csv file
                metrics = os.listdir(os.path.join(checkpoint_path, "lightning_logs"))
                # find the last version
                version = sorted(metrics)[-1]
                metrics = pd.read_csv(
                    os.path.join(checkpoint_path, "lightning_logs", version, "metrics.csv")
                )

                interruption_epoch = max(metrics["epoch"])

                if interruption_epoch < max_epochs:
                    max_epochs = max_epochs - interruption_epoch

                logger.info(f"Resuming training from epoch {interruption_epoch}")
            
            except:
                logger.info(f"No logging metric found, setting max_epoch to {max_epochs}")    

            model = model_class.load_from_checkpoint(
                chekpoints[0], module_config=model_config
            )

    if model is None:
        model = model_class(module_config=model_config)

    ########### Callbacks ############
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        save_top_k=1,
        mode=mode,
        dirpath=checkpoint_path,
        filename="fold=%d-{epoch}-{step}-{val_acc:.2f}" % fold,
        save_weights_only=False,
    )
    
    lr_callback = LearningRateMonitor(logging_interval="step")
    
    #dvc_callback = DeviceStatsMonitor()
    
    # progress_bar_callback = RichProgressBar()
    my_logger = [
        TensorBoardLogger(save_dir=checkpoint_path),
        CSVLogger(save_dir=checkpoint_path),
    ]

    ########### Trainer Setup ############
    from lightning.pytorch.accelerators import find_usable_cuda_devices

    try :
        devices = find_usable_cuda_devices(-1)
        logger.info( f"Available devices: {devices}")
    except :
        devices = "auto"
    
    effective_batch_size = batch_size * num_nodes * len(devices)

    num_steps = datamodule.dataset.__len__() * 0.7 // effective_batch_size
    val_check_interval = max(1, num_steps // num_validations)
        
    trainer = Trainer(
        devices=devices,
        strategy = "ddp" if (num_nodes > 1 or len(devices) > 1) else "auto",
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        callbacks=[checkpoint_callback, lr_callback ], # dvc_callback, progress_bar_callback],
        deterministic=True,
        logger=my_logger,
    )

    # setup the model in training mode if needed
    model = model.train()
    # Start training
    trainer.fit(model, datamodule=datamodule)
    
    return checkpoint_callback.best_model_path
