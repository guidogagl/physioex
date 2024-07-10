import importlib

import pkg_resources as pkg
import yaml
from loguru import logger

import physioex as physioex
import physioex.train.networks.utils.target_transform as target_transform
from physioex.train.networks.chambon2018 import Chambon2018Net
from physioex.train.networks.seqsleepnet import SeqSleepNet
from physioex.train.networks.sleeptransformer import SleepTransformer
from physioex.train.networks.tinysleepnet import TinySleepNet

# from physioex.train.networks.seqecgnet import SeqECGnet


def read_config(model_name: str):
    config_file = pkg.resource_filename(__name__, "config/" + model_name + ".yaml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


config = {
    "chambon2018": {
        "module_config": read_config("chambon2018"),
        "module": Chambon2018Net,
        "input_transform": "raw",
        "target_transform": target_transform.get_mid_label,
    },
    "seqsleepnet": {
        "module_config": read_config("seqsleepnet"),
        "module": SeqSleepNet,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "sleeptransformer": {
        "module_config": read_config("sleeptransformer"),
        "module": SleepTransformer,
        "input_transform": "xsleepnet",
        "target_transform": None,
    },
    "tinysleepnet": {
        "module_config": read_config("tinysleepnet"),
        "module": TinySleepNet,
        "input_transform": "raw",
        "target_transform": None,
    },
}


def get_config():
    return config


def register_experiment(experiment: str = None):
    global config

    logger.info(f"Registering experiment {experiment}")

    try:
        with open(experiment, "r") as f:
            experiment = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Experiment {experiment} not found")

    experiment = experiment["experiment"]

    experiment_name = experiment["name"]

    config[experiment_name] = dict()

    module = importlib.import_module(experiment["module"])

    config[experiment_name]["module"] = getattr(module, experiment["class"])
    config[experiment_name]["module_config"] = experiment["module_config"]
    config[experiment_name]["input_transform"] = experiment["input_transform"]

    if experiment["target_transform"] is not None:
        if experiment["module"] != experiment["target_transform"]["module"]:
            module = importlib.import_module(experiment["target_transform"]["module"])

        config[experiment_name]["target_transform"] = getattr(
            module, experiment["target_transform"]["function"]
        )
    else:
        logger.warning(f"Target transform not found for {experiment_name}")
        config[experiment_name]["target_transform"] = None

    return experiment_name
