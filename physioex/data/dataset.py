import os
import pickle
from typing import Callable, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from physioex.utils import get_data_folder
from physioex.data.datareader import DataReader
from loguru import logger

class PhysioExDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling physiological data from multiple datasets.

    Attributes:
        datasets (List[str]): List of dataset names.
        L (int): Sequence length.
        channels_index (List[int]): Indices of selected channels.
        readers (List[DataReader]): List of DataReader objects for each dataset.
        tables (List[pd.DataFrame]): List of data tables for each dataset.
        dataset_idx (np.ndarray): Array indicating the dataset index for each sample.
        target_transform (Callable): Optional transform to be applied to the target.
        len (int): Total number of samples across all datasets.

    Methods:
        __len__(): Returns the total number of samples.
        split(fold: int = -1, dataset_idx: int = -1): Splits the data into train, validation, and test sets.
        get_num_folds(): Returns the minimum number of folds across all datasets.
        __getitem__(idx): Returns the input and target for a given index.
        get_sets(): Returns the indices for the train, validation, and test sets.
    """
    def __init__(
        self,
        datasets: List[str],
        data_folder: str,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        hpc : bool = False,
        indexed_channels : List[int] = ["EEG", "EOG", "EMG", "ECG"],
    ):
        """
        Initializes the PhysioExDataset.

        Args:
            datasets (List[str]): List of dataset names.
            data_folder (str): Path to the folder containing the data.
            preprocessing (str, optional): Type of preprocessing to apply. Defaults to "raw".
            selected_channels (List[int], optional): List of selected channels. Defaults to ["EEG"].
            sequence_length (int, optional): Length of the sequence. Defaults to 21.
            target_transform (Callable, optional): Optional transform to be applied to the target. Defaults to None.
            hpc (bool, optional): Flag indicating whether to use high-performance computing. Defaults to False.
            indexed_channels (List[int], optional): List of indexed channels. Defaults to ["EEG", "EOG", "EMG", "ECG"]. If you used a custom Preprocessor and you saved your signal channels in a different order, you should provide the correct order here. In any other case ignore this parameter.
        """
        self.datasets = datasets
        self.L = sequence_length
        self.channels_index = [ indexed_channels.index( ch ) for ch in selected_channels ]
        
        self.readers = []
        self.tables = []
        self.dataset_idx = []
        
        offset = 0
        for i, dataset in enumerate(datasets):
            reader = DataReader(
                data_folder = data_folder,
                dataset = dataset,
                preprocessing = preprocessing,
                sequence_length = sequence_length,
                channels_index = self.channels_index,
                offset = offset,
                hpc = hpc,
            )
            offset += len(reader)

            self.dataset_idx += list( np.ones( len(reader) ) * i ) 

            self.tables.append( reader.get_table() )
            self.readers += [reader]
            
        self.dataset_idx = np.array( self.dataset_idx, dtype=np.uint8 )
        # set the table fold to the 0 fold by default
        self.split()
        self.target_transform = target_transform

        self.len = offset

    def __len__(self):
        """
        Returns the total number of sequences of epochs across all the datasets.

        Returns:
            int: Total number of sequences.
        """
        return self.len

    def split(self, fold: int = -1, dataset_idx: int = -1):
        """
        Splits the data into train, validation, and test sets.
        if fold is -1, and dataset_idx is -1 : set the split to a random fold for each dataset 
        if fold is -1, and dataset_idx is not -1 : set the split to a random fold for the selected dataset
        if fold is not -1, and dataset_idx is -1 : set the split to the selected fold for each dataset
        if fold is not -1, and dataset_idx is not -1 : set the split to the selected fold for the selected dataset 
        Args:
            fold (int, optional): Fold number to use for splitting. Defaults to -1.
            dataset_idx (int, optional): Index of the dataset to split. Defaults to -1.
        """
        assert dataset_idx < len(self.tables), "ERR: dataset_idx out of range"

        # if fold is -1, set the split to a random fold for each dataset
        if fold == -1 and dataset_idx == -1:
            for i, table in enumerate(self.tables):
                num_folds = [col for col in table.columns if "fold_" in col]
                num_folds = len(num_folds)
                selcted_fold = np.random.randint(0, num_folds)

                self.tables[i]["split"] = table[f"fold_{selcted_fold}"].map(
                    {"train": 0, "valid": 1, "test": 2}
                )
        elif fold == -1 and dataset_idx != -1:
            num_folds = [
                col for col in self.tables[dataset_idx].columns if "fold_" in col
            ]
            num_folds = len(num_folds)
            selcted_fold = np.random.randint(0, num_folds)

            self.tables[dataset_idx]["split"] = table[f"fold_{selcted_fold}"].map(
                {"train": 0, "valid": 1, "test": 2}
            )
        elif fold != -1 and dataset_idx == -1:
            for i, table in enumerate(self.tables):
                self.tables[i]["split"] = table[f"fold_{fold}"].map(
                    {"train": 0, "valid": 1, "test": 2}
                )
        else:
            self.tables[dataset_idx]["split"] = table[f"fold_{fold}"].map(
                {"train": 0, "valid": 1, "test": 2}
            )

    def get_num_folds(self):
        """
        Returns the minimum number of folds across all datasets.

        Returns:
            int: Minimum number of folds.
        """
        # take the min number of folds for each dataset table
        num_folds = 100
        for table in self.tables:
            num_folds = min(
                num_folds, len([col for col in table.columns if "fold_" in col])
            )
        return num_folds

    def __getitem__(self, idx):
        """
        Returns the input and target sequence for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Input and target for the given index.
        """
        dataset_idx = int(self.dataset_idx[idx])
        
        X, y = self.readers[dataset_idx][idx]
        
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y

    def get_sets(self):
        """
        Returns the indices for the train, validation, and test sets.

        Returns:
            tuple: Indices for the train, validation, and test sets.
        """
        # return the indexes in the table of the train, valid and test subjects
        train_idx = []
        valid_idx = []
        test_idx = []

        start_index = 0

        for table in self.tables:
            for _, row in table.iterrows():

                num_windows = row["num_windows"] - self.L + 1

                indices = np.arange(
                    start=start_index, stop=start_index + num_windows
                ).astype(np.uint32)

                start_index += num_windows

                if row["split"] == 0:
                    train_idx.append(indices)
                elif row["split"] == 1:
                    valid_idx.append(indices)
                elif row["split"] == 2:
                    test_idx.append(indices)
                else:
                    error_string  = "ERR: split should be 0, 1 or 2. Not " + str(row["split"])
                    logger.error( error_string )
                    raise ValueError( "ERR: split should be 0, 1 or 2")

        train_idx = np.concatenate(train_idx)
        valid_idx = np.concatenate(valid_idx)
        test_idx = np.concatenate(test_idx)

        return train_idx, valid_idx, test_idx
