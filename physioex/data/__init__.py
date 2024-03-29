from physioex.data.base import PhysioExDataset, TimeDistributedModule
from physioex.data.dreem import Dreem
from physioex.data.mitdb import MITBIH
from physioex.data.sleep_physionet import SleepPhysionet

datasets = {"sleep_physionet": SleepPhysionet, "dreem": Dreem, "mitdb": MITBIH}
