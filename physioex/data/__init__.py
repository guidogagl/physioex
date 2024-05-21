from physioex.data.base import (
    PhysioExDataset,
    TimeDistributedModule,
    CombinedTimeDistributedModule,
)
from physioex.data.dreem.dreem import Dreem
from physioex.data.sleep_edf.sleep_edf import SleepEDF
from physioex.data.shhs.shhs import Shhs
from physioex.data.mass.mass import Mass
from physioex.data.msd.msd import MultiSourceDomain

datasets = {
    "sleep_physionet": SleepEDF,
    "dreem": Dreem,
    "shhs": Shhs,
    "mass": Mass,
    "MSD": MultiSourceDomain,
}
