import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import set_float32_matmul_precision
import seaborn as sns
import matplotlib.pyplot as plt

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule


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

    devices = find_usable_cuda_devices(-1)

    trainer = Trainer(
        devices=devices,
        strategy="ddp" if (num_nodes > 1 or len(devices) > 1) else "auto",
        num_nodes=num_nodes,
        # callbacks=[progress_bar_callback],
        deterministic=True,
    )
    
    results = []
    cm_dict = {}
    for _, test_datamodule in enumerate(datamodule):
        results += [trainer.test(model, datamodule=test_datamodule)[0]]
        results[-1]["dataset"] = (
            test_datamodule.datasets_id[0] if not aggregate_datasets else "aggregate"
        )
        results[-1]["fold"] = fold
        
        for key, value in model.cm.items():
            cm = value.compute()
            
            if key == 'cm5':
                cm = cm[:3, :]
                
            cm_dict[f"{key}_{results[-1]['dataset']}_{fold}"] = cm
            value.reset()

    results = pd.DataFrame(results)

    if results_path is not None:
        results.to_csv(os.path.join(results_path, "results.csv"))
        
        for key, value in cm_dict.items():
            if value.shape == (3, 3):
                true_labels = ['W', 'N', 'R']
                pred_labels = true_labels
            elif value.shape == (5, 5):
                true_labels = ['W', 'N1', 'N2', 'N3', 'R']
                pred_labels = true_labels
            elif value.shape == (3, 5):
                true_labels = ['W', 'N', 'R']
                pred_labels = ['W', 'N1', 'N2', 'N3', 'R']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(value.cpu(), annot=True, fmt="d", cmap="Blues",
                        xticklabels=pred_labels, yticklabels=true_labels, cbar=False)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.gca().legend('off')
            plt.savefig(os.path.join(results_path, key+".png"), dpi=300, bbox_inches="tight") 

    return results
