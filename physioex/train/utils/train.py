import uuid
from pathlib import Path
from typing import List, Type, Union

import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision

from physioex.data import PhysioExDataModule
from physioex.train.networks.base import SleepModule


def train(
    datasets: Union[List[str], str, PhysioExDataModule],
    datamodule_kwargs: dict = None,
    model: SleepModule = None,  # if passed model_class, model_config and resume are ignored
    model_class: Type[SleepModule] = None,
    model_config: dict = None,
    batch_size: int = 128,
    fold: int = 0,
    num_validations: int = 10,
    checkpoint_path: str = None,
    max_epochs: int = 10,
    num_nodes: int = 1,
    resume: bool = True,
    monitor: str = "val/acc",
    mode: str = "max",
) -> str:
    seed_everything(42, workers=True)
    set_float32_matmul_precision("medium")

    if datamodule_kwargs is None:
        datamodule_kwargs = {}

    datamodule_kwargs["batch_size"] = batch_size
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
        checkpoints = list(Path(checkpoint_path).glob("*.ckpt"))
        if len(checkpoints) == 0:
            logger.warning(
                f"No checkpoints found in {checkpoint_path}. Instantiating a new model."
            )
        else:
            logger.info(f"Resuming from checkpoint: {checkpoints[0]}")
            model = model_class.load_from_checkpoint(
                checkpoints[0], module_config=model_config
            )

    if model is None:
        model = model_class(module_config=model_config)

    ########### Callbacks ############
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        save_top_k=1,
        mode=mode,
        dirpath=checkpoint_path,
        filename=f"fold={fold}-" + "{epoch}-{step}-{val_acc:.2%}",
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
        effective_batch_size = batch_size * num_nodes *  len(devices)

    except :
        devices = "auto"
        effective_batch_size = batch_size * num_nodes
    num_steps = datamodule.__len__() // effective_batch_size
    val_check_interval = max(1, num_steps // num_validations)

    if devices == "auto":
        strategy = "auto"
    elif num_nodes > 1 or len(devices) > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = Trainer(
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            dvc_callback,
        ],  # , progress_bar_callback],
        deterministic="warn",
        logger=my_logger,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "32-true",
    )

    # setup the model in training mode if needed
    model = model.train()

    # Start training
    trainer.fit(model, datamodule=datamodule)

    return checkpoint_callback.best_model_path
