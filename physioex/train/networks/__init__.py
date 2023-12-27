import yaml
import pkg_resources as pkg

from physioex.train.networks.tinysleepnet import TinySleepNet, ContrTinySleepNet
from physioex.train.networks.chambon2018 import Chambon2018Net, ContrChambon2018Net
from physioex.train.networks.seqtoseqnet import SeqtoSeqSleepNet, ContrSeqtoSeqSleepNet

import physioex.train.networks.input_transform as input_transform
import physioex.train.networks.target_transform as target_transform


import physioex as physioex


def read_config(model_name : str):
    config_file =  pkg.resource_filename(__name__, 'config/' + model_name + '.yaml')
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config

config = {
    "chambon2018": { 
        "module_config" : read_config("chambon2018"),
        "module" : Chambon2018Net,
        "input_transform" : None,
        "target_transform" : target_transform.get_mid_label,
    },
    "contr_chambon2018": { 
        "module_config" : read_config("chambon2018"),
        "module" : ContrChambon2018Net,
        "input_transform" : None,
        "target_transform" : target_transform.get_mid_label,
    },
    "seqtoseqsleepnet": { 
        "module_config" : read_config("seqtoseqsleepnet"),
        "module" : SeqtoSeqSleepNet,
        "input_transform" : input_transform.xsleepnet_transform,
        "target_transform" : None,
    },
    "contr_seqtoseqsleepnet": { 
        "module_config" : read_config("seqtoseqsleepnet"),
        "module" : ContrSeqtoSeqSleepNet,
        "input_transform" : input_transform.xsleepnet_transform,
        "target_transform" : None,
    },
    "tinysleepnet": { 
        "module_config" : read_config("tinysleepnet"),
        "module" : TinySleepNet,
        "input_transform" : None,
        "target_transform" : None,
    },
    "contr_tinysleepnet": { 
        "module_config" : read_config("tinysleepnet"),
        "module" : ContrTinySleepNet,
        "input_transform" : None,
        "target_transform" : None,
    },
}
