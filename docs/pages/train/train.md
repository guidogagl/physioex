# Train Module Overview

The `physioex.train` module provides comprehensive tools for training, fine-tuning, and evaluating deep learning models on sleep staging datasets. 

## Usage Example

Below is an example of how to use the training module:

```python
from physioex.train.utils import train, test, finetune
from physioex.data import PhysioExDataModule

from physioex.train.networks.utils.loss import config as loss_config
from physioex.train.networks import config as network_config

checkpoint_path = "/path/to/your/checkpoint/dir/"

# first configure the model

# set the model configuration dictonary

# in case your model is from physioex.train.networks
# you can load its configuration

your_model_config = network_config["tinysleepnet"] 

your_model_config["loss_call"] = loss_config["cel"] # CrossEntropy Loss
your_model_config["loss_params"] = dict()
your_model_config["seq_len"] = 21 # needs to be the same as the DataModule
your_model_config["in_channels"] = 1 # needs to be the same as the DataModule

#your_model_config["n_classes"] = 5  needs to be set if you are loading a custom SleepModule
```

Here we loaded che configuration setup to train a `physioex.train.networks.TinySleepNet` model. Best practices is to have a .yaml file where to store the model configuration, both in case you are using a custom  `SleepModule` or in case you are using one of the models provided by PhysioEx. 

Here is an example of a possible .yaml configuration file and how to read it properly:


=== ":fontawesome-solid-file-code: .yaml"
    ```yaml
    module_config: 
        loss_call : physioex.train.networks.utils.loss:CrossEntropyLoss
        loss_params : {}
        seq_len : 21
        in_channels : 1
        n_classes : 5
        ... # your additional model configuration parameters should be provided here
    module: physioex.train.networks:TinySleepNet # can be any model extends SleepModule
    model_name: "tinysleepnet" # can be avoided if loading your custom SleepModule
    input_transform: "raw"
    target_transform: null
    checkpoint_path : "/path/to/your/checkpoint/dir/"
    ```
=== ":fontawesome-brands-python: .py"
    ```python
    import yaml

    with open("my_network_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    your_module_config = config["module_config"]

    # load the loss function 
    import importlib
    loss_package, loss_class = your_module_config["loss_call"].split(":")
    your_model_config = getattr(importlib.import_module(loss_package), loss_class)

    # in case you provide model_name the system loads the additional model parameters from the library
    if "model_name" in config:
        model_name = config["model_name"]
        module_config = networks_config[model_name]["module_config"]
        your_model_config.update(module_config)
 
        config["input_transform"] = networks_config[model_name]["input_transform"]
        config["target_transform"] = networks_config[model_name]["target_transform"]

    # load the model class
    model_package, model_class = config["module"].split(":")
    model_class = getattr(importlib.import_module(model_package), model_class)
    ```
    ??? note
        In case you are using a model provided by the library, "input_transform" and "target_transform" can be loaded from the network_config ( line 20-21 )

Now we need to set up the datamodule arguments and we can start training the model:

```python

datamodule_kwargs = {
    "selected_channels" : ["EEG"], # needs to match in_channels
    "sequence_length" : your_model_config["seq_len"],
    "target_transform" : your_model_config["target_transform"]
    "preprocessing" : your_model_config["input_transform"],
    "data_folder" : "/your/data/folder",
}

model_config = your_model_config

# Train the model
best_checkpoint = train(
    datasets = "hmc", # can be a list or a PhysioExDataModule
    datamodule_kwargs = datamodule_kwargs,
    model = model,
    model_class = model_class,
    model_config = model_config,
    checkpoint_path = checkpoint_path
    batch_size = 128,
    max_epochs = 10
)

# Test the model
results_dataframe = test(
    datasets = "hmc",
    datamodule_kwargs = datamodule_kwargs,
    model_class = model_class,
    model_config = your_model_config,
    chekcpoint_path = os.path.join( checkpoint_path, best_checkpoint ),
    batch_size = 128,
    results_dir = checkpoint_path,  # if you want to save the test results 
                                    # in your checkpoint directory
)
```
Even in this case, the best practice should be to save the datamodule_kwargs into the .yaml configuration file, at least the non-dynamic ones. 

Now imagine that we want to fine-tune the trained model on a new dataset.

```python

train_kwargs = {
    "dataset" = "dcsm",
    "datamodule" = datamodule_kwargs,
    "batch_size" = 128,
    "max_epochs" = 10,
    "checkpoint_path" = checkpoint_path,
}

new_best_checkpoint = finetune(
    model_class = model_class,
    model_config = model_config,
    model_checkpoint = os.path.join( checkpoint_path, best_checkpoint),
    learning_rate = 1e-7, # slower the learning rate to avoid losing prior training info.
    train_kwargs = train_kwargs,
) 
```

## Documentation
`train` 
::: train.train
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

---
`test` 
::: train.test
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

---
`finetune` 
::: train.finetune
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
