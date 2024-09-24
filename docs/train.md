# Train Module Overview

The `physioex.train` module provides comprehensive tools for training, fine-tuning, and evaluating deep learning models on sleep staging datasets. 

## Usage Example

Below is an example of how to use the training module:

```python
from physioex.train.utils import train, test
from physioex.data import PhysioExDataModule

from physioex.train.networks.utils.loss import config as loss_config

checkpoint_path = "/path/to/your/checkpoint/dir/"

# first configure the model
# set the model configuration dictonary
your_model_config["loss_call"] = loss_config["cel"] # CrossEntropy Loss
your_model_config["loss_params"] = dict()
your_model_config["seq_len"] = 21 # needs to be the same as the DataModule
your_model_config["in_channels"] = 1 # needs to be the same as the DataModule

datamodule_kwargs = {
    "selected_channels" : ["EEG"],
    "sequence_length" : 21,
    "target_transform" : None, # if sequence-to-sequence you can set it to None
    "preprocessing" : "raw",
    "data_folder" : "/your/data/folder",
}

# 1. you can use your own defined model, 
#   if it extends physioex.train.networks.base:SleepModule

class YourSleepModule(SleepModule):
    def __init__(self, module_config : dict )
        pass

model = YourSleepModule( module_config = your_model_config )
model_class, model_config = None, None # you can set them at None in case

# 2. or you want to resume the traing of an already saved model
model_class =  YourModelClass
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

# ( optional ) load your model from memory
model = YourModelClass.load_from_checkpoint( best_checkpoint, module_config = your_module_config )

# if not you can pass the class, config and checkpoint_path
model = None # set it to None in case

# Test the model
results_dataframe = test(
    datasets = "hmc",
    datamodule_kwargs = datamodule_kwargs,
    model = model,
    model_class = YourModelClass,
    model_config = your_model_config,
    chekcpoint_path = os.path.join( checkpoint_path, best_checkpoint ),
    batch_size = 128,
    results_dir = checkpoint_path,  # if you want to save the test results 
                                    # on your checkpoint directory
)
```

## CLI


## Documentation

::: physioex.train.utils.train
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: physioex.train.utils.test
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

::: physioex.train.utils.finetune
    handler: python
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3
