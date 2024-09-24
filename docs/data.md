# Data Module 

The `physioex.data` module provides the API to read the data from the disk once the raw datasets have been processed by the `Preprocess` module. It consists of two classes: 
- `physioex.data.PhysioExDataset` which serialize the disk processed version of the dataset into a `PyTorch Dataset`
- `physioex.data.PhysioExDataModule` which transforms the datasets to `PyTorch DataLoaders` ready for training. 


