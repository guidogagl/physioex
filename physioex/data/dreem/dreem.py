from typing import List, Callable

from physioex.data.base import PhysioExDataset, create_subject_index_map
from physioex.data.utils import read_config

from pathlib import Path

import numpy as np

import h5py

from scipy.io import loadmat

import torch
import pandas as pd

class Dreem(PhysioExDataset):
    def __init__(
        self,
        version: str = "dodh",
        picks: List[str] = ["C3-M2"],  # available [ "C3-M3", "EOG", "EMG" ]
        preprocessing: str = "raw",  # available [ "raw", "xsleepnet" ]
        sequence_length: int = 21,
        target_transform: Callable = None,
    ):

        assert version in ["dodo", "dodh"], "version should be one of 'dodo'-'dodh'"
        assert preprocessing in [
            "raw",
            "xsleepnet",
        ], "preprocessing should be one of 'raw'-'xsleepnet'"
        for pick in picks:
            assert pick in [
                "C3-M2",
                "EOG",
                "EMG",
            ], "pick should be one of 'C3-M2, 'EOG', 'EMG'"

        self.config = read_config( "config/dreem.yaml")

        self.table = pd.read_csv(str(Path.home()) + self.config["table_" + version])

        self.subjects = self.table['subject_id'].values.astype(int)
        
        self.subject_start_indices, self.subject_end_indices = create_subject_index_map(self.table, sequence_length)   
        
        self.split_path = str(Path.home()) + f"/dreem/{preprocessing}/{version}/scaling.npz"
        self.data_path =  str(Path.home()) + f"/dreem/{preprocessing}/{version}/"

        self.picks = picks
        self.version = version
        self.preprocessing = preprocessing

        self.mean = None
        self.std = None

        self.L = sequence_length
        self.target_transform = target_transform

        self.input_shape = self.config[ "shape_" + preprocessing ]
        
        scaling_file = np.load( self.split_path )
        
        EEG_mean, EOG_mean, EMG_mean = scaling_file["mean"]
        EEG_std, EOG_std, EMG_std = scaling_file["std"]
               
        self.mean = []
        self.std = []

        if "C3-M2" in self.picks:
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
        return 10

    def split(self, fold: int = 0):

        test_subjects = self.config[ self.version][f"fold_{fold}"][ "test"] 
        valid_subjects = self.config[ self.version][f"fold_{fold}"][ "valid"] 
        

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

