import importlib
import os
from typing import Type, Union

import pandas as pd
import pkg_resources as pkg
import torch
from loguru import logger

from physioex.train.networks import config as network_config
from physioex.train.networks.base import SleepModule
import gdown

def get_models_table():
    check_table_path = pkg.resource_filename(
        "physioex", os.path.join("train", "models", "check_table.csv")
    )
    check_table = pd.read_csv(check_table_path)
    return check_table


def load_model(
    model: Union[str, Type[SleepModule]],
    model_kwargs: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ckpt_path: str = None,
    softmax: bool = False,
    summary: bool = False,
):
    assert (
        "sequence_length" in model_kwargs.keys()
    ), "Sequence length must be provided in the model_kwargs"
    assert (
        "in_channels" in model_kwargs.keys()
    ), "Number of input channels must be provided in the model_kwargs"

    seq_len = model_kwargs["sequence_length"]
    in_channels = model_kwargs["in_channels"]

    default_kwargs = network_config["default"]["model_kwargs"].copy()

    if isinstance(model, str) and ":" in model:
        # import the module and the class
        module, class_name = model.split(":")
        model = getattr(importlib.import_module(module), class_name)

        assert (
            ckpt_path is not None
        ), "Checkpoint path must be provided when model is passed as a string path.to.module:Class"
    elif isinstance(model, str):
        assert (
            model in network_config.keys()
        ), f"Model {model} not found in the registered models"

        table = get_models_table()
        assert (
            model in table["name"].values
        ), f"Model {model} not found in the models table"

        table = table[
            (table["name"] == model)
            & (table["sequence_length"] == seq_len)
            & (table["in_channels"] == in_channels)
        ]

        if "model_kwargs" in network_config[model]:
            default_kwargs.update(network_config[model]["model_kwargs"])

        module, class_name = table["module"].values[0].split(":")
        model = getattr(importlib.import_module(module), class_name)

        ckpt_path = table["checkpoint"].values[0]
        ckpt_path = pkg.resource_filename(
            "physioex", os.path.join("train", "models", "checkpoints", ckpt_path)
        )
    elif isinstance(model, type(SleepModule)) and ckpt_path is None:
        if model.__name__ == "SeqSleepNet":
            model_name = "seqsleepnet"
        elif model.__name__ == "TinySleepNet":
            model_name = "tinysleepnet"
        elif model.__name__ == "Chambon2018Net":
            model_name = "chambon2018"
        elif model.__name__ == "SleepTransformer":
            model_name = "sleeptransformer"
        elif model.__name__ == "MiceTransformer":
            model_name = "micetransformer"
        elif model.__name__ == "ProtoSleepNet":
            model_name = "protosleepnet"
        else:
            raise ValueError(
                f"Model {model.__name__} not found in the registered pretrained models"
            )

        table = get_models_table()

        assert (
            model_name in table["name"].values
        ), f"Model {model_name} not found in the models table"

        table = table[
            (table["name"] == model_name)
            & (table["sequence_length"] == seq_len)
            & (table["in_channels"] == in_channels)
        ]

        
        if "model_kwargs" in network_config[model_name]:
            default_kwargs.update(network_config[model_name]["model_kwargs"])

        
        ckpt_path = table["checkpoint"].values[0]
        ckpt_path = pkg.resource_filename(
            "physioex", os.path.join("train", "models", "checkpoints", ckpt_path)
        )
    else:
        pass
    
    if not os.path.isfile( ckpt_path ):
        
        from huggingface_hub import hf_hub_download
        
        # Scarica il modello dal repository Hugging Face
        model_name = table["name"].values[0]
        filename = os.path.basename(ckpt_path)
        ckpt_path = hf_hub_download(
            repo_id="4rooms/physioex",
            filename=filename,
            local_dir=pkg.resource_filename(
                "physioex", os.path.join("train", "models", "checkpoints")
            ),
        )
    
    default_kwargs.update(model_kwargs)
    model_kwargs = default_kwargs

    model = (
        model.load_from_checkpoint(ckpt_path, module_config=model_kwargs)
        .to(device)
        .eval()
    )

    if softmax:
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))

    if summary:
        logger.info(model)

    return model
