import pkg_resources as pkg
import yaml

import physioex as physioex
import physioex.train.networks.utils.target_transform as target_transform
from physioex.train.networks.chambon2018 import Chambon2018Net
from physioex.train.networks.seqsleepnet import SeqSleepNet
from physioex.train.networks.tinysleepnet import TinySleepNet
from physioex.train.networks.seqecgnet import SeqECGnet


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
    "tinysleepnet": {
        "module_config": read_config("tinysleepnet"),
        "module": TinySleepNet,
        "input_transform": "raw",
        "target_transform": None,
    },
    "seqecgnet": {
        "module_config": read_config("seqecgnet"),
        "module": SeqECGnet,
        "input_transform": None,
        "target_transform": None,
    },
}
