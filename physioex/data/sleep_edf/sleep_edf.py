from typing import List, Callable

from physioex.data.base import PhysioExDataset, transform_to_sequence
from physioex.data.utils import read_config

from pathlib import Path

import numpy as np

import h5py

from scipy.io import loadmat

import torch


class SleepEDF(PhysioExDataset):
    def __init__(
        self,
        version: str = "dodh",
        picks: List[str] = ["Fpz-Cz"],  # available [ "Fpz-Cz", "EOG", "EMG" ]
        preprocessing: str = "raw",  # available [ "raw", "xsleepnet" ]
        sequence_length: int = 21,
        target_transform: Callable = None,
    ):

        assert version in ["dodh", "dodo"], "version should be one of '2013'-'2018'"
        assert preprocessing in [
            "raw",
            "xsleepnet",
        ], "preprocessing should be one of 'raw'-'xsleepnet'"
        for pick in picks:
            assert pick in [
                "Fpz-Cz",
                "EOG",
                "EMG",
            ], "pick should be one of 'C3-M3', 'EOG', 'EMG'"

        super().__init__(
            version,
            picks,
            preprocessing,
            "config/sleep-edf.yaml",
            sequence_length,
            target_transform,
        )
        
        scaling_file = np.load( str(Path.home()) + self.config[ self.preprocessing + "_path"] + "scaling_" + self.version + ".npz" )
        
        EEG_mean, EOG_mean, EMG_mean = scaling_file["mean"]
        EEG_std, EOG_std, EMG_std = scaling_file["std"]
        
        self.mean = []
        self.std = []

        if "Fpz-Cz" in self.picks:
            self.mean.append(EEG_mean)
            self.std.append(EEG_std)
        if "EOG" in self.picks:
            self.mean.append(EOG_mean)
            self.std.append(EOG_std)
        if "EMG" in self.picks:
            self.mean.append(EMG_mean)
            self.std.append(EMG_std)
            
        self.mean = torch.tensor( np.array(self.mean) ).float()
        self.std = torch.tensor( np.array(self.std) ).float()

    def get_num_folds(self):
        split_matrix = loadmat(self.split_path)["test_sub"]

        return len(split_matrix)



    def split(self, fold: int = 0):

        split_matrix = loadmat(self.split_path)

        test_subjects = split_matrix["test_sub"][fold][0][0] 
        valid_subjects = split_matrix["eval_sub"][fold][0][0] 
        

        # add a column to the table with 0 if the subject is in train, 1 if in valid, 2 if in test

        split = np.zeros(len(self.table))
        split[self.table["subject_id"].isin(test_subjects)] = 2
        split[self.table["subject_id"].isin(valid_subjects)] = 1

        self.table["split"] = split

        
        
    def __getitem__(self, idx):
        x, y =  super().__getitem__(idx)

        x = ( x - self.mean ) / self.std
        
        if self.target_transform is not None:
            y = self.target_transform(y)
        
        return x, y

