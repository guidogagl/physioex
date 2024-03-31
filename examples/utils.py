import os
import re
import subprocess

import torch
from loguru import logger

from physioex.data import TimeDistributedModule, datasets
from physioex.train.networks import config
from physioex.train.networks.utils.loss import config as loss_config


class PhysioLoader:
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
        if ckp_path is None:
            raise ValueError("ckp_path must be provided")
        if not os.path.isdir(ckp_path):
            raise ValueError(
                "ckp_path must be a valid directory containing at least one checkpoint"
            )

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
        self.checkpoints = {
            int(re.search(r"fold=(\d+)", elem.name).group(1)): elem.path
            for elem in os.scandir(ckp_path)
            if elem.is_file()
            and elem.name.endswith(".ckpt")
            and re.search(r"fold=(\d+)", elem.name)
        }

        logger.info(f"Found {len(self.checkpoints)} checkpoints")

        self.ckpt_path = ckp_path

        logger.info("Loading dataset")
        self.dataset_call = datasets[dataset_name]
        self.dataset = self.dataset_call(version=self.version, use_cache=self.use_cache)
        logger.info("Dataset loaded")

    def get_fold(self, fold: int = 0):

        logger.info(
            "JOB:%d-Splitting dataset into train, validation and test sets" % fold
        )
        self.dataset.split(fold)

        datamodule = TimeDistributedModule(
            dataset=self.dataset,
            sequence_lenght=self.module_config["seq_len"],
            batch_size=self.batch_size,
            transform=self.input_transform,
            target_transform=self.target_transform,
        )

        self.module_config["loss_params"]["class_weights"] = datamodule.class_weights()

        model = self.model_call.load_from_checkpoint(
            self.checkpoints[fold], module_config=self.module_config
        ).eval()

        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = model.to(model_device)
        train = datamodule.train_dataloader()
        test = datamodule.test_dataloader()
        valid = datamodule.val_dataloader()

        return model, train, valid, test


def set_root():
    # Ottieni il percorso corrente
    current_path = os.getcwd()

    # Esegui il comando git per trovare la root del repository
    git_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=current_path
    )

    # Decodifica l'output per ottenere una stringa
    git_root_path = git_root.decode("utf-8").strip()

    # Cambia la directory corrente alla root del repository git
    os.chdir(git_root_path)

    # Stampa il nuovo percorso corrente
    print(f"Current working directory: {os.getcwd()}")

    # Aggiungi il percorso alla variabile d'ambiente PATH
    os.environ["PATH"] += os.pathsep + git_root_path
