import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision
import seaborn as sns
import matplotlib.pyplot as plt

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule

from loguru import logger

def test(
    datasets: Union[List[str], str, PhysioExDataModule],
    datamodule_kwargs: dict = {},
    model: SleepModule = None,  # if passed model_class, model_config and resume are ignored
    model_class=None,
    model_config: dict = None,
    batch_size: int = 128,
    fold: int = -1,
    hpc: bool = False,
    checkpoint_path: str = None,
    results_path: str = None,
    num_nodes: int = 1,
    aggregate_datasets: bool = False,
) -> pd.DataFrame:

    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    datamodule_kwargs["batch_size"] = batch_size
    # datamodule_kwargs["hpc"] = hpc
    datamodule_kwargs["folds"] = fold
    datamodule_kwargs["num_nodes"] = num_nodes

    ##### DataModule Setup #####
    if isinstance(datasets, PhysioExDataModule):
        datamodule = [datasets]
    elif isinstance(datasets, str):
        datamodule = [
            PhysioExDataModule(
                datasets=[datasets],
                **datamodule_kwargs,
            )
        ]
    elif isinstance(datasets, list):
        if aggregate_datasets:
            logger.info( "Building the aggregation dataset...")
            datamodule = PhysioExDataModule(
                datasets=datasets,
                **datamodule_kwargs,
            )
        else:
            datamodule = []
            for dataset in datasets:
                datamodule.append(
                    PhysioExDataModule(
                        datasets=[dataset],
                        **datamodule_kwargs,
                    )
                )
    else:
        raise ValueError("datasets must be a list, a string or a PhysioExDataModule")

    ########### Resuming Model if needed else instantiate it ############:
    if model is None:
        model = load_model(
            model=model_class,
            model_kwargs=model_config,
            ckpt_path=checkpoint_path,
        )

    ########### Callbacks ############
    # progress_bar_callback = RichProgressBar()

    ########### Trainer Setup ############
    from lightning.pytorch.accelerators import find_usable_cuda_devices

    if results_path is None:
        if checkpoint_path is not None:
            results_path = os.path.dirname( checkpoint_path )
        else :
            results_path = os.path.join( os.getcwd(), "temp" )
            os.makedirs( results_path, exist_ok=True )
    
    try :
        devices = find_usable_cuda_devices(-1)
        logger.info( f"Available devices: {devices}")

    except :
        devices = "auto"
        
    my_logger = [
        TensorBoardLogger(save_dir=results_path + "/test_logs/"),
        CSVLogger(save_dir=results_path + "/test_logs/"),
    ]
    
    trainer = Trainer(
        devices=devices,
        strategy="ddp" if (num_nodes > 1 or len(devices) > 1) else "auto",
        num_nodes=num_nodes,
        # callbacks=[progress_bar_callback],
        deterministic=True,
        logger=my_logger
    )
    
    results = []
    cm_dict = {}
    for _, test_datamodule in enumerate(datamodule):
        logger.info( f"Testing on {test_datamodule.datasets_id[0]}")
        results += [trainer.test(model, datamodule=test_datamodule)[0]]
        results[-1]["dataset"] = (
            test_datamodule.datasets_id[0] if not aggregate_datasets else "aggregate"
        )
        results[-1]["fold"] = fold
        
    results = pd.DataFrame(results)

    if results_path is not None:
        results.to_csv(os.path.join(results_path, "results.csv"))

    return results
