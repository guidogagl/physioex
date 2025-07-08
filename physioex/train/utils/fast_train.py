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

from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.tuner import Tuner
class FastTrainDataset(Dataset):
    def __init__(self, 
        datasets : List[str], 
        preprocess : str, 
        L : int, 
        indexed_channels : List[int],  
        data_folder : str = "./"
    ):
        
        self.X, self.y = [], [] 
        
        for d in datasets:
            logger.info(f"Loading FastTrainDataset for {d} with preprocess {preprocess}")

            X, y = torch.load(
                f"{data_folder}.tmp/{d}/{preprocess}/train_dataset.pt",
                map_location=torch.device("cpu")
            )

            self.X.append(X)
            self.y.append(y)

        self.X = torch.cat(self.X, dim=0)
        self.y = torch.cat(self.y, dim=0)

        logger.info(f"Loaded X: {self.X.shape}, y: {self.y.shape}")

        self.X = self.X[:, indexed_channels]

        self.L = L

    def __len__(self):
        return len(self.y) - self.L + 1

    def __getitem__(self, idx):
        return self.X[idx:idx+self.L].float(), self.y[idx:idx+self.L].long(), 0, 0

import pytorch_lightning as pl
class FastDataModule(pl.LightningDataModule):
    def __init__(self, 
        batch_size,
        train_dataset = None, 
        val_dataset = None, 
        test_dataset = None,
        num_workers = os.cpu_count()
        ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
    )

    def val_dataloader(self):
        return DataLoader(
        self.val_dataset,
        batch_size=1,
        num_workers=self.num_workers,
    )    

    def test_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=1,
        num_workers=self.num_workers,
    )

class FastEvalDataset(Dataset):
    def __init__(self, 
        datasets : List[str], 
        preprocess : str, 
        indexed_channels : List[int], 
        split = "eval",
        data_folder : str = "./" 
    ):

        self.X, self.y = [], []

        for d in datasets:
            logger.info(f"Loading FastEvalDataset for {d} with preprocess {preprocess}")

            X, Y = torch.load(
                f"{data_folder}.tmp/{d}/{preprocess}/{split}_dataset.pt",
                map_location=torch.device("cpu")
            )

            for x, y in zip(X, Y):
                self.X.append(x)
                self.y.append(y)

        self.indexed_channels = indexed_channels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx][:, self.indexed_channels].float(), self.y[idx].long(), 0, 0


def train(
    datasets: List[str],
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

    indexed_channels = ["EEG", "EOG", "EMG", "ECG"]
    channels_index = [indexed_channels.index(ch) for ch in datamodule_kwargs["selected_channels"]]

    
    ## custom fast train dataloader
    train_dataset = FastTrainDataset(
        datasets = datasets,
        preprocess = datamodule_kwargs["preprocessing"],
        L = datamodule_kwargs["sequence_length"],
        indexed_channels = channels_index,
        data_folder=datamodule_kwargs["data_folder"],
    )

    ### custom fast eval dataloader

    eval_dataset = FastEvalDataset(
        datasets = datasets,
        preprocess = datamodule_kwargs["preprocessing"],
        indexed_channels = channels_index,
        split = "eval",
        data_folder=datamodule_kwargs["data_folder"],
    )

    datamodule = FastDataModule(
        train_dataset = train_dataset,
        val_dataset = eval_dataset,
        batch_size = datamodule_kwargs["batch_size"],
        num_workers = datamodule_kwargs["num_workers"],
    )

    ###
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

    #logger.info(f"Number of workers: {os.cpu_count()}")
    
    ########### Callbacks ############
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        save_top_k=1,
        mode=mode,
        dirpath=checkpoint_path,
        filename="fold=%d-{epoch}-{step}-{val_acc:.3f}" % fold,
        save_weights_only=False,
    )
    
    lr_callback = LearningRateMonitor(logging_interval="step")
    
    dvc_callback = DeviceStatsMonitor()
    
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
        effective_batch_size = batch_size * num_nodes * len(devices)

    except :
        devices = "auto"
        effective_batch_size = batch_size * num_nodes

    num_steps = len(train_dataset) // effective_batch_size
    val_check_interval = max(1, num_steps // num_validations)
    
    if devices == "auto":
        strategy = "auto"
    elif num_nodes > 1 or len(devices) > 1:
        strategy = "ddp"
    else:
        strategy = "auto"
            
    trainer = Trainer(
        devices=devices,
        strategy = strategy,
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        callbacks=[checkpoint_callback, lr_callback, dvc_callback], #, progress_bar_callback],
        deterministic=True,
        logger=my_logger,
    )

    # Start training
    trainer.fit(model.train(), datamodule)
    return checkpoint_callback.best_model_path
