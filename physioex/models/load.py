import pandas as pd
import pkg_resources as pkg
import torch
from loguru import logger

from physioex.train.networks import get_config as get_networks
from physioex.train.networks import register_experiment
from physioex.train.networks.utils.loss import config as losses


def load_pretrained_model(
    name : str = None,
    in_channels : int = 1,
    sequence_length : int = 21,
    loss : str = "cel",
    loss_params : dict = None,
    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ckpt_path : str = None, 
    softmax : bool = False,
    summary : bool = False
    ):
    
    
    # check if the experiment is a yaml file
    if name.endswith(".yaml") or name.endswith(".yml"):
        logger.info("Registering experiment from configuration file")
        name = register_experiment( name )    
    
    networks = get_networks()
    
    try:
        model = networks[name]
    except KeyError:
        raise ValueError(f"Model {name} not found in the available models")
    
    # set model config
    model["module_config"]["seq_len"] = sequence_length
    model["module_config"]["in_channels"] = in_channels
    model["module_config"]["loss_call"] = losses[ loss ]
    model["module_config"]["loss_params"] = {"params": loss_params}

    if ckpt_path is None:
        # read the checkpoint path table from the current package directory
        check_table_path = pkg.resource_filename('physioex', 'models/check_table.csv')
        check_table = pd.read_csv(check_table_path)
        
        # se esiste il modello con quella configurazione  nella tabella del checkpoint prendi il path del checkpoint
        try:
            ckpt_path = check_table[ check_table["name"] == name ]
            ckpt_path = ckpt_path[ check_table["sequence_length"] == sequence_length ]
            ckpt_path = ckpt_path[ check_table["in_channels"] == in_channels ]
            ckpt_path = ckpt_path[ check_table["loss"] == loss ]
            ckpt_path = ckpt_path["checkpoint"].values[0]
            ckpt_path = pkg.resource_filename('physioex', f"models/checkpoints/{ckpt_path}")
        except IndexError:
            raise ValueError("No checkpoint found for the specified model configuration")
                    
    model = (
        model["module"]
        .load_from_checkpoint(ckpt_path, module_config=model["module_config"])
        .to(device)
        .eval()
    )
    
    if softmax:
        # add a softmax activation function at the end of the model to get probabilities ( care with Seq to Seq )
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim = -1))
    
    if summary:
        logger.info(model)
    
    return model