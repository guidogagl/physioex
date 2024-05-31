import importlib
from pathlib import Path

import yaml
from loguru import logger

from physioex.data.dreem.dreem import Dreem
from physioex.data.mass.mass import Mass
from physioex.data.msd.msd import MultiSourceDomain
from physioex.data.shhs.shhs import Shhs
from physioex.data.sleep_edf.sleep_edf import SleepEDF

data_folder = str(Path.home())


def get_data_folder():
    return data_folder


def set_data_folder(new_data_folder):
    global data_folder
    data_folder = new_data_folder

datasets = {
    "sleep_physionet": SleepEDF,
    "dreem": Dreem,
    "shhs": Shhs,
    "mass": Mass,
    "MSD": MultiSourceDomain,
}

def get_datasets():
    return datasets

def add_dataset(name, dataset):
    global datasets
    datasets[name] = dataset
    

@logger.catch
def register_dataset(dataset: str = None):

    logger.info(f"Registering dataset {dataset}")

    try:
        with open(dataset, "r") as f:
            dataset = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {dataset} not found")

    dataset = dataset["dataset"]
    dataset_name = dataset["name"]
    module = importlib.import_module(dataset["module"])

    add_dataset(dataset_name, getattr(module, dataset["class"]))

    return dataset_name
    