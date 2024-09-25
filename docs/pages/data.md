# Data Module 

The `physioex.data` module provides the API to read the data from the disk once the raw datasets have been processed by the `Preprocess` module. It consists of two classes: 

- `physioex.data.PhysioExDataset` which serialize the disk processed version of the dataset into a `PyTorch Dataset`
- `physioex.data.PhysioExDataModule` which transforms the datasets to `PyTorch DataLoaders` ready for training. 

### Example of Usage

#### PhysioExDataset

The `PhysioExDataset` class is automatically handled by the `PhysioExDataModule` class when you need to use it for training or testing purposes. In most of the cases you don't need to interact with the `PhysioExDataset` class.

The class is instead really helpfull when you need to visualize your data, or you need to get some samples of your data to provide them as input to Explainable AI algorithms.

In these cases you need to instantiate a `PhysioExDataset`:

```python
from physioex.data import PhysioExDataset

data = PhysioExDataset(
    datasets = ["hmc"], # you can read different datasets merged together in this way
    preprocessing = "raw",  
    selected_channels = ["EEG", "EOG", "EMG"],     
    data_folder = "/your/data/path/",
)

# you can now access any sequence of epochs in the dataset
signal, label = data[0]

signal.shape # will be [21 (default sequence lenght), 3, 3000]
label.shape # will be [21]
```

Then you can use a python plotting library to plot visualize the data

!!! example
	```python
	import seaborn as sns
	import numpy as np 

	hypnogram = np.ones((21, 3000)) * label.numpy().reshape(-1, 1)

	# plot a subfigure with one column for each element of the sequence (21)
	fig, ax = plt.subplots(4, 1, figsize = (21, 8), sharex="col", sharey="row")

	hypnogram = hypnogram.reshape( -1 )
	signals = signal.numpy().transpose(1, 0, 2).reshape(3, -1)

	# set tytle for each subplot
	sns.lineplot( x = range(3000*21), y = hypnogram, ax = ax[0], color = "blue")
	# then the channels:
	sns.lineplot( x = range(3000*21), y = signals[ 0], ax = ax[1], color = "red")
	sns.lineplot( x = range(3000*21), y = signals[ 1], ax = ax[2], color = "green")
	sns.lineplot( x = range(3000*21), y = signals[ 2], ax = ax[3], color = "purple")    

	# check the examples notebook "visualize_data.ipynb" to see how to customize the plot properly

	plt.tight_layout()
	```

	![png](assets/images/data/sequence_viz.png)


#### PhysioExDataModule

The `PhysioExDataModule` class is designed to transform datasets into `PyTorch DataLoaders` ready for training. It handles the batching, shuffling, and splitting of the data into training, validation, and test sets.

To use the `PhysioExDataModule`, you need to instantiate it with the required parameters:

```python
from physioex.data import PhysioExDataModule

datamodule = PhysioExDataModule(
    datasets=["hmc", "mass"],  # list of datasets to be used
    batch_size=64,             # batch size for the DataLoader
    preprocessing="raw",       # preprocessing method
    selected_channels=["EEG", "EOG", "EMG"],  # channels to be selected
    sequence_length=21,        # length of the sequence
    data_folder="/your/data/path/",  # path to the data folder
)

# get the DataLoaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

PhysiEx is built on `pytorch_lightning` for model training and testig, hence you can use `PhysioExDataModule` in combination with `pl.Trainer`

```python
from pytorch_lightning import Trainer

model = SomePytorchModel()

trainer = Trainer(
    devices="auto"
    max_epochs=10,
    deterministic=True,
)
    
# setup the model in training mode if needed
model = model.train()
# Start training
trainer.fit(model, datamodule=datamodule)
results = trainer.test( model, datamodule = datamodule)
```

## Documentation
`PhysioExDataset` 
::: data.PhysioExDataset
    handler: python
    options:
      members:
        - __init__
        - __getitem__
        - __len__
        - split
        - get_num_folds
        - get_sets
      show_root_heading: false
      show_source: false
	  heading_level: 3

---

`PhysioExDataModule`
::: data.PhysioExDataModule
    handler: python
    options:
      members:
        - __init__
        - train_dataloader
        - valid_dataloader
        - test_dataloader
      show_root_heading: false
      show_source: false
	  heading_level: 3