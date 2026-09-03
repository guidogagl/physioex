"""Dataset registry.

As more datasets are implemented, they get added here. Users can instantiate by
string name via ``get_dataset(name)``.
"""
from physioex.data.datasets.hmc import HMCDataset
from physioex.data.datasets.sleepedf import SleepEDFDataset
from physioex.data.datasets.dcsm import DCSMDataset
from physioex.data.datasets.mesa import MESADataset
from physioex.data.datasets.mros import MrOSDataset
from physioex.data.datasets.hpap import HPAPDataset
from physioex.data.datasets.wsc import WSCDataset
from physioex.data.datasets.mass import MASSDataset
from physioex.data.datasets.alzheimers import AlzheimersDataset
from physioex.data.datasets.parkinsons import ParkinsonsDataset
from physioex.data.datasets.shhs import SHHSDataset
from physioex.data.datasets.stages import STAGESDataset
from physioex.data.datasets.vitaldb import VitalDBDataset

REGISTRY = {
    "hmc": HMCDataset,
    "sleepedf": SleepEDFDataset,
    "dcsm": DCSMDataset,
    "mesa": MESADataset,
    "mros": MrOSDataset,
    "hpap": HPAPDataset,
    "wsc": WSCDataset,
    "mass": MASSDataset,
    "alzheimers": AlzheimersDataset,
    "parkinsons": ParkinsonsDataset,
    "shhs": SHHSDataset,
    "stages": STAGESDataset,
    "vitaldb": VitalDBDataset,
}


def get_dataset(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available_datasets() -> list:
    return sorted(REGISTRY.keys())


__all__ = [
    "HMCDataset",
    "SleepEDFDataset",
    "DCSMDataset",
    "MESADataset",
    "MrOSDataset",
    "HPAPDataset",
    "WSCDataset",
    "MASSDataset",
    "AlzheimersDataset",
    "ParkinsonsDataset",
    "SHHSDataset",
    "STAGESDataset",
    "REGISTRY",
    "get_dataset",
    "available_datasets",
]
