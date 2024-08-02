import os
import pickle
from typing import Callable, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from physioex.utils import get_data_folder, set_data_folder

DATASETS = {
    "mass": ["EEG", "EOG", "EMG"],
    "shhs": ["EEG", "EOG", "EMG"],
    #"hpap": ["EEG", "EOG", "EMG"], #
    "dcsm": ["EEG", "EOG", "EMG", "ECG"],
    "mesa": ["EEG", "EOG", "EMG"], #
    "mros": ["EEG", "EOG", "EMG"], #
}


class PhysioExDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[str],
        versions: List[str] = None,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        #task: str = "sleep",
        data_folder: str = None,
    ):
        print( datasets, versions )
        
        for dataset in datasets:
            assert (
                dataset in DATASETS.keys()
            ), f"ERR: dataset {dataset} not available"
            for channel in selected_channels:
                assert (
                    channel in DATASETS[dataset]
                ), f"ERR: channel {channel} not available in dataset {dataset}"

        if versions is not None:
            assert len(datasets) == len(
                versions
            ), "ERR: datasets and versions should have the same length"

        else:
            versions = [None] * len(datasets)

    
        assert preprocessing in [
            "raw",
            "xsleepnet",
        ], "ERR: preprocessing should be 'raw' or 'xsleepnet'"

        if data_folder is not None:
            data_folder = set_data_folder(data_folder)
        else:
            data_folder = get_data_folder()
        
        print( data_folder )
        
        self.dataset_folders = []
        self.tables = []
        self.scaling = []

        
        
        for dataset, version in zip(datasets, versions):
            self.dataset_folders += [os.path.join(data_folder, dataset)]
    
            if version is not None:
                self.dataset_folders[-1] = os.path.join(self.dataset_folders[-1], version)
    
            self.tables += [
                pd.read_csv(os.path.join(self.dataset_folders[-1], "table.csv"))
            ]

            scaling = np.load(
                os.path.join(self.dataset_folders[-1], preprocessing,  "scaling.npz")
            )
            mean, std = scaling["mean"], scaling["std"]
            
            # take the selected channels 
            mean = mean[[DATASETS[dataset].index(channel) for channel in selected_channels]]
            std = std[[DATASETS[dataset].index(channel) for channel in selected_channels]]
            
            self.scaling += [
                (
                    torch.tensor(mean).float(),
                    torch.tensor(std).float(),
                )
            ]

        # set the table fold to the 0 fold by default
        self.split()

        self.input_shape = [3000] if preprocessing == "raw" else [29, 129]
        self.preprocessing = preprocessing

        # add the channel dimension to the input shape
        self.input_shape = [len(selected_channels)] + self.input_shape

        self.channels_index = [
            DATASETS[datasets[0]].index(channel) for channel in selected_channels
        ]

        self.L = sequence_length

        self.target_transform = target_transform

        #self.task = task

        # create the index maps for the windows

        # window --> dataset table

        num_windows = self.__len__()
        start_index = 0

        self.dataset_idx = np.zeros(num_windows, dtype=np.uint8)
        for i, table in enumerate(self.tables):
            dataset_num_windows = int(
                np.sum(table["num_windows"].values - self.L + 1)
            )
            self.dataset_idx[start_index : start_index + dataset_num_windows] = (
                np.uint8(i)
            )
            start_index += dataset_num_windows

        # window --> subject
        # datasets can have up to 5000 subjects, so we need a uint16

        start_index = 0

        self.subject_idx = np.zeros(num_windows, dtype=np.uint16)
        for table in self.tables:
            for _, row in table.iterrows():
                subject_id = int(row["subject_id"])
                n_win = int(row["num_windows"]) - self.L + 1
                self.subject_idx[start_index : start_index + n_win] = np.uint16(
                    subject_id
                )
                start_index += n_win

        # window --> relative index
        # we need a uint32 for the relative index
        self.relative_idx = np.zeros(num_windows, dtype=np.uint16)
        start_index = 0
        for table in self.tables:
            for _, row in table.iterrows():
                n_win = int(row["num_windows"]) - self.L + 1
                self.relative_idx[start_index : start_index + n_win] = np.arange(
                    n_win, dtype=np.uint16
                )
                if n_win > np.iinfo(np.uint16).max:
                    raise ValueError(
                        f"Value {n_win} exceeds the maximum value for np.uint16"
                    )

                start_index += n_win

    def __len__(self):

        num_windows = 0
        for table in self.tables:
            num_windows += int(
                np.sum(table["num_windows"].values - self.L + 1)
            )

        return num_windows

    def split(self, fold: int = -1, dataset_idx: int = -1):
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
        # take the min number of folds for each dataset table
        num_folds = 100
        for table in self.tables:
            num_folds = min(num_folds, len([col for col in table.columns if "fold_" in col]))
        return num_folds
        
    def __getitem__(self, idx):

        table_idx = int(self.dataset_idx[idx])
        subject_id = int(self.subject_idx[idx])
        relative_id = int(self.relative_idx[idx])

        # get the path to the subject data from the table
        table = self.tables[table_idx]
        mean, std = self.scaling[table_idx]

        subject_info = table[table["subject_id"] == subject_id]

        labels_path = subject_info["labels"].values[0]
        data_path = subject_info[self.preprocessing].values[0]
        subject_windows = subject_info["num_windows"].values[0]

        # read the numpy memmap files
        y = np.load(labels_path)
        X = np.load(data_path)

        # select the windows

        X = X[relative_id : relative_id + self.L, self.channels_index]
        y = y[relative_id : relative_id + self.L]

        X = (torch.tensor(X).float() - mean) / std
        y = torch.tensor(y).long()


        if self.target_transform is not None:
            y = self.target_transform(y)

        
        return X, y

    def get_sets(self):
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
                ).astype(np.uint16)

                start_index += num_windows

                if row["split"] == 0:
                    train_idx.append(indices)
                elif row["split"] == 1:
                    valid_idx.append(indices)
                elif row["split"] == 2:
                    test_idx.append(indices)
                else:
                    raise ValueError("ERR: split should be 0, 1 or 2")

        train_idx = np.concatenate(train_idx)
        valid_idx = np.concatenate(valid_idx)
        test_idx = np.concatenate(test_idx)

        return train_idx, valid_idx, test_idx
