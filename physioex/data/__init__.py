from pathlib import Path

from physioex.data.base import (CombinedTimeDistributedModule, PhysioExDataset,
                                TimeDistributedModule)
from physioex.data.dreem.dreem import Dreem
from physioex.data.mass.mass import Mass
from physioex.data.msd.msd import MultiSourceDomain
from physioex.data.shhs.shhs import Shhs
from physioex.data.sleep_edf.sleep_edf import SleepEDF

preprocess = {
    "sleep_physionet": "physioex.data.sleep_edf.preprocess",
    "dreem": "physioex.data.dreem.preprocess",
    "shhs": "physioex.data.shhs.preprocess",
    "mass": "physioex.data.mass.preprocess",
}
