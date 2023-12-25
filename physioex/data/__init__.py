from physioex.data.sleep_physionet import SleepPhysionet
from physioex.data.dreem import Dreem
from physioex.data.base import PhysioExDataset, TimeDistributedModule


datasets = {
    "sleep_physionet": SleepPhysionet,
    "dreem": Dreem
}