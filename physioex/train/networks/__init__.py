from physioex.train.networks.tinysleepnet import TinySleepNet, ContrTinySleepNet,target_transforms as tiny_target, inpunt_transforms as tiny_input
from physioex.train.networks.chambon2018 import Chambon2018Net, ContrChambon2018Net, target_transforms as chambon2018_target, inpunt_transforms as chambon2018_input

import physioex as physioex

module_config = {
    "n_classes": 5,
    "n_channels": 1,
    "sfreq": 100,
    "n_times": 3000,
    "seq_len": 3,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "latent_space_dim": 32
}

models = {
    "tinysleepnet": TinySleepNet,
    "contr_tinysleepnet": ContrTinySleepNet,
    "chambon2018": Chambon2018Net,
    "contr_chambon2018": ContrChambon2018Net
}

target_transform = {
    "tinysleepnet": tiny_target,
    "contr_tinysleepnet": tiny_target,
    "chambon2018": chambon2018_target,
    "contr_chambon2018": chambon2018_target
}

input_transform = {
    "tinysleepnet": tiny_input,
    "contr_tinysleepnet": tiny_input,
    "chambon2018": chambon2018_input,
    "contr_chambon2018": chambon2018_input,
}