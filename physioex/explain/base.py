import os
import re
from abc import ABC, abstractmethod

import torch
from loguru import logger
from pytorch_lightning import LightningModule

from physioex.data import TimeDistributedModule, datasets
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config


class PhysioExplainer(ABC):
    def __init__(
        self,
        model_name: str = "chambon2018",
        dataset_name: str = "sleep_physioex",
        loss_name: str = "cel",
        ckp_path: str = None,
        version: str = "2018",
        use_cache: bool = True,
        sequence_lenght: int = 3,
        batch_size: int = 32,
    ):

        assert ckp_path is not None, "ckp_path must be provided"
        assert os.path.isdir(
            ckp_path
        ), "ckp_path must be a valid directory containing at least one checkpoint"

        self.model_name = model_name
        self.model_call = config[model_name]["module"]
        self.input_transform = config[model_name]["input_transform"]
        self.target_transform = config[model_name]["target_transform"]
        self.module_config = config[model_name]["module_config"]
        self.module_config["seq_len"] = sequence_lenght

        self.module_config["loss_call"] = loss_config[loss_name]
        self.module_config["loss_params"] = dict()

        self.batch_size = batch_size
        self.version = version
        self.use_cache = use_cache

        logger.info("Scanning checkpoint directory...")
        self.checkpoints = {}
        for elem in os.scandir(ckp_path):
            if elem.is_file() and elem.name.endswith(".ckpt"):
                try:
                    fold = int(re.search(r"fold=(\d+)", elem.name).group(1))
                except Exception as e:
                    logger.warning(
                        "Could not parse fold number from checkpoint name: %s. Skipping..."
                        % elem.name
                    )
                    continue
                self.checkpoints[fold] = elem.path

        logger.info("Found %d checkpoints" % len(self.checkpoints))

        self.ckpt_path = ckp_path

        logger.info("Loading dataset")
        self.dataset_call = datasets[dataset_name]
        self.dataset = self.dataset_call(version=self.version, use_cache=self.use_cache)
        logger.info("Dataset loaded")

    @abstractmethod
    def explain(self):
        pass
