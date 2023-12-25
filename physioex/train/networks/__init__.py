from physioex.train.networks.tinysleepnet import TinySleepNet, ContrTinySleepNet
from physioex.train.networks.chambon2018 import Chambon2018Net, ContrChambon2018Net

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
    "tinysleepnet": physioex.train.networks.tinysleepnet.target_transform,
    "contr_tinysleepnet": physioex.train.networks.tinysleepnet.target_transform,
    "chambon2018": physioex.train.networks.chambon2018.target_transform,
    "contr_chambon2018": physioex.train.networks.chambon2018.target_transform
}

input_transform = {
    "tinysleepnet": physioex.train.networks.tinysleepnet.input_transform,
    "contr_tinysleepnet": physioex.train.networks.tinysleepnet.input_transform,
    "chambon2018": physioex.train.networks.chambon2018.input_transform,
    "contr_chambon2018": physioex.train.networks.chambon2018.input_transform,
}