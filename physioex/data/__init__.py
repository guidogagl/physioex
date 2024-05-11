from physioex.data.base import PhysioExDataset, TimeDistributedModule
from physioex.data.dreem.dreem import Dreem
from physioex.data.sleep_edf.sleep_edf import SleepEDF
from physioex.data.shhs.shhs import Shhs

datasets = {"sleep_physionet": SleepEDF, "dreem": Dreem, "shhs": Shhs}
