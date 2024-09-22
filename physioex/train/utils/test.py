from typing import List, Union
import os
from pathlib import Path
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from physioex.data import PhysioExDataModule
from physioex.train.networks.base import SleepModule

import pandas as pd

def test(
    datasets : Union[ List[str], str, PhysioExDataModule],
    datamodule_kwargs : dict = {},
    model : SleepModule = None, # if passed model_class, model_config and resume are ignored
    model_class = None,
    model_config : dict = None,
    batch_size : int = 128,
    fold : int = -1,
    hpc : bool = False,
    checkpoint_path : str = None,
    results_path : str = None,
    num_nodes : int = 1,
    aggregate_datasets : bool = False,
    ) -> pd.DataFrame:
    """
    Tests a model using the provided datasets and configuration.

    Parameters
    ----------
    datasets : Union[List[str], str, PhysioExDataModule]
        The datasets to be used for testing. Can be a list of dataset names, a single dataset name, or a PhysioExDataModule instance.
    datamodule_kwargs : dict, optional
        Additional keyword arguments to be passed to the PhysioExDataModule.
    model : SleepModule, optional
        The model to be tested. If provided, `model_class`, `model_config`, and `resume` are ignored.
    model_class : type, optional
        The class of the model to be tested. Required if `model` is not provided.
    model_config : dict, optional
        The configuration dictionary for the model. Required if `model` is not provided.
    batch_size : int, optional
        The batch size to be used for testing. Default is 128.
    fold : int, optional
        The fold index for cross-validation. Default is -1.
    hpc : bool, optional
        Whether to use high-performance computing (HPC) settings. Default is False.
    checkpoint_path : str, optional
        The path to the checkpoint from which to load the model. Required if `model` is not provided.
    results_path : str, optional
        The path to save the test results. If None, results are not saved. Default is None.
    num_nodes : int, optional
        The number of nodes to be used for distributed testing. Default is 1.
    aggregate_datasets : bool, optional
        Whether to aggregate the datasets for testing. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the test results.

    Raises
    ------
    ValueError
        If `datasets` is not a list, a string, or a PhysioExDataModule instance.
        If `model` is None and any of `model_class` or `model_config` are also None.

    Notes
    -----
    - The function sets up the data module, model, and trainer, and then starts the testing process.
    - The function returns a DataFrame containing the test results for each dataset.
    - If `results_path` is provided, the results are saved as a CSV file in the specified path.
    """
    datamodule_kwargs["batch_size"] = batch_size
    datamodule_kwargs["hpc"] = hpc
    datamodule_kwargs["folds"] = fold
    datamodule_kwargs["num_nodes"] = num_nodes
    
    ##### DataModule Setup #####
    if isinstance(datasets, PhysioExDataModule):
        datamodule = [ datasets ]
    elif isinstance(datasets, str):
        datamodule = [ PhysioExDataModule(
            datasets=[datasets],
            **datamodule_kwargs,
        ) ]
    elif isinstance(datasets, list):
        if aggregate_datasets:
            datamodule = PhysioExDataModule(
                datasets=datasets,
                **datamodule_kwargs,
            )
        else:
            datamodule = []
            for dataset in datasets:
                datamodule.append(PhysioExDataModule(
                    datasets=[dataset],
                    **datamodule_kwargs,
                )) 
    else:
        raise ValueError("datasets must be a list, a string or a PhysioExDataModule")
        
    
    
    ########### Resuming Model if needed else instantiate it ############:
    if model is None:            
        model = model_class.load_from_checkpoint(checkpoint_path, module_config=model_config)
    
    ########### Callbacks ############    
    #progress_bar_callback = RichProgressBar()    
    
    ########### Trainer Setup ############

    trainer = Trainer(
        devices="auto",
        strategy="ddp" if hpc and num_nodes > 1 else "auto",
        num_nodes=num_nodes if hpc else 1,
        #callbacks=[progress_bar_callback],
        deterministic=True,
    )
    results =[]
    for _,  test_datamodule in enumerate(datamodule):
        results += [ trainer.test(model, datamodule=test_datamodule)[0] ]
        results[-1]["dataset"] = test_datamodule.datasets_id[0] if not aggregate_datasets else "aggregate"
        results[-1]["fold"] = fold
    
    results = pd.DataFrame(results)
    
    if results_path is not None:
        results.to_csv( os.path.join(results_path, "results.csv") )
        
    return results