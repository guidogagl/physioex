import importlib
from typing import Dict
import yaml
import os 
import uuid

from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

from jsonargparse import ArgumentParser, ActionConfigFile

from physioex.data.base import TimeDistributedModule, get_datasets   

from physioex.train.networks import get_config as get_model_config
from physioex.train.networks.utils.loss import config as loss_config

class CLIDataModule:
    def __init__(self,        
        dataset,
        version,
        picks,
        sequence_lenght,
        preprocessing,
        target_transform,
        task,
        batch_size, 
        fold 
        ):
        
        dataset = get_datasets()[dataset](
            version=version,
            picks=picks,
            sequence_lenght=sequence_lenght,
            preprocessing=preprocessing,
            target_transform=target_transform,
            task=task
        )
        
        return TimeDistributedModule( dataset, batch_size, fold )


class CLIModel:
    def __init__(self,
        model, 
        picks,
        sequence_lenght,
        loss_name,
        task
        ):
        
        model_config = get_model_config()
    
        model_class = model_config[model]["module"]
        model_args = model_config[model]["module_config"]
        model_args[ "in_channels" ] = len( picks.split(" ") )
        model_args[ "seq_len" ] = sequence_lenght

        if task == "sleep":
            model_args["n_classes"] = 5
        else:
            raise ValueError(f"Task {task} not supported")    
        
        model_args["loss_call"] = loss_config[loss_name]
        model_args["loss_params"] = dict()

        return model_class( model_args )

class PhysioexCLI( LightningCLI ):
    def add_arguments_to_parser(self, parser: ArgumentParser) -> None:

        parser.link_arguments( "data.sequence_lenght", "model.sequence_lenght", apply_on="instantiate" )
        parser.link_arguments( "data.picks", "model.picks", apply_on="instantiate" )
        parser.link_arguments( "data.task", "model.task", apply_on="instantiate" )
    

    def before_instantiate_classes(self) -> None:
        # Recupera il nome del modello dagli argomenti della CLI
        model_name = self.config['model']['model']
        
        # Recupera il model_config utilizzando il nome del modello
        model_config = get_model_config()
        
        # Assicurati che il modello esista nel model_config
        if model_name not in model_config:
            raise ValueError(f"Modello {model_name} non trovato nel model_config.")
        
        # Recupera il parametro preprocessing dal model_config
        preprocessing = model_config[model_name].get('input_transform', None)
        target_transform = model_config[model_name].get('target_transform', None)
        
        # Aggiorna la configurazione della classe dei dati con il parametro preprocessing
        self.config['data']['preprocessing'] = preprocessing
        self.config['data']['target_transform'] = target_transform 
        
        ckpt = self.config["trainer"]["callbacks"][0]["init_args"] 
        
        ckpt["dirpath"] = f"models/{self.config["data"]["dataset"]}/{self.config["model"]["model"]}/"
        ckpt["filename"] = f"{self.config["data"]["fold"]}-{epoch}-{step}-{val_acc:.2f}"
        
        super().before_instantiate_classes()
        
        
        
if __name__ == "__main__":
    
    cli = PhysioexCLI(
        model_class= CLIModel,
        datamodule_class= CLIDataModule,
        default_config_file_path = os.path.join( os.path.dirname(__file__), "config.yaml" )
    )
