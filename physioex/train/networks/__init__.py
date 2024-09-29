import os

import pkg_resources as pkg
import yaml

from physioex.train.networks.chambon2018 import Chambon2018Net
from physioex.train.networks.seqsleepnet import SeqSleepNet
from physioex.train.networks.tinysleepnet import TinySleepNet

config_file = pkg.resource_filename(
    "physioex", os.path.join("train", "networks", "config.yaml")
)
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Network configuration file not found: {config_file}")


with open(config_file, "r") as file:
    config = yaml.safe_load(file)
