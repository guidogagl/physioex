from typing import Callable, List

import numpy as np
import torch
from loguru import logger

from physioex.data.datareader import DataReader

import torch

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

def merge_scaling(means, std_devs, sample_sizes):    
    # cast means and stds to float64

    means = [ mean.to( torch.float64) for mean in means ]
    std_devs = [ std.to( torch.float64) for std in std_devs ]

    total_samples = sum(sample_sizes)
    
    # Calcolo media ponderata
    total_mean = sum(M * N for M, N in zip(means, sample_sizes)) / total_samples
    
    # Calcolo varianza totale
    total_variance = (
        sum(
            N * (D**2 + (M - total_mean)**2)
            for M, D, N in zip(means, std_devs, sample_sizes)
        ) / total_samples
    )
    
    # Assicurati che la varianza non sia negativa (potrebbe accadere per errori di arrotondamento)
    total_variance = torch.clamp(total_variance, min=0.0) + 1e-10
    
    total_std_dev = torch.sqrt(total_variance)
    return total_mean.to( DTYPE ), total_std_dev.to( DTYPE )


class PhysioExDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[str],
        data_folder: str,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        hpc: bool = False,
        indexed_channels: List[int] = ["EEG", "EOG", "EMG", "ECG"],
        task: str = "sleep",
    ):
        self.datasets = datasets
        self.channels_index = [indexed_channels.index(ch) for ch in selected_channels]

        self.readers = []
        self.tables = []
        self.dataset_idx = []

        offset = 0

        means, stds, sizes = [], [], []

        for i, dataset in enumerate(datasets):
            reader = DataReader(
                data_folder=data_folder,
                dataset=dataset,
                preprocessing=preprocessing,
                sequence_length=sequence_length,
                channels_index=self.channels_index,
                offset=offset,
                hpc=hpc,
                task=task,
            )
            offset += len(reader)

            self.dataset_idx += list(np.ones(len(reader)) * i)

            self.tables.append(reader.get_table())
            self.readers += [reader]

            means.append( reader.reader.mean )
            stds.append( reader.reader.std )
            sizes.append( reader.reader.get_n_subjects() )

        if len( self.readers ) > 1:
            self.mean, self.std = merge_scaling( means, stds, sizes )
        else:
            self.mean, self.std = self.readers[0].reader.mean, self.readers[0].reader.std
        
        self.dataset_idx = np.array(self.dataset_idx, dtype=np.int8)
        # set the table fold to fold 0 by default
        self.split(0)
        self.target_transform = target_transform

        self.len = offset
        self.L = sequence_length if sequence_length != -1 else 30 * 2 * 60 * 24

    def __len__(self):
        return self.len

    def set_scaling(self, mean : torch.Tensor, std : torch.Tensor ):
        self.mean = mean
        self.std = std

        return
    
    def get_scaling(self):
        return self.mean, self.std

    def split(self, fold: int = 0, dataset_idx: int = -1):
        assert fold >= 0, "ERR: fold must be >= 0. fold=-1 (randomly selected fold) is deprecated."
        assert dataset_idx < len(self.tables), "ERR: dataset_idx out of range"

        if dataset_idx == -1:
            # Apply to all datasets
            for i, table in enumerate(self.tables):
                fold_columns = [col for col in table.columns if "fold_" in col]
                num_folds = len(fold_columns)
                if fold >= num_folds:
                    raise ValueError(f"ERR: fold {fold} is out of range for dataset {i}. Available folds: 0-{num_folds-1} (total: {num_folds} folds)")
                
                self.tables[i]["split"] = table[f"fold_{fold}"].map(
                    {"train": 0, "valid": 1, "test": 2}
                )
        else:
            # Apply to specific dataset
            table = self.tables[dataset_idx]
            fold_columns = [col for col in table.columns if "fold_" in col]
            num_folds = len(fold_columns)
            if fold >= num_folds:
                raise ValueError(f"ERR: fold {fold} is out of range for dataset {dataset_idx}. Available folds: 0-{num_folds-1} (total: {num_folds} folds)")
            
            self.tables[dataset_idx]["split"] = table[f"fold_{fold}"].map(
                {"train": 0, "valid": 1, "test": 2}
            )

    def get_num_folds(self):
        # take the min number of folds for each dataset table
        num_folds = 100
        for table in self.tables:
            num_folds = min(
                num_folds, len([col for col in table.columns if "fold_" in col])
            )
        return num_folds

    def __getitem__(self, idx):
        dataset_idx = int(self.dataset_idx[idx])

        X, y, subjects = self.readers[dataset_idx][idx]

        if self.target_transform is not None:
            y = self.target_transform(y)

        X = (X - self.mean) / self.std 
        
        return X, y, subjects, dataset_idx

    def get_sets(self):
        # return the indexes in the table of the train, valid and test subjects
        train_idx = []
        valid_idx = []
        test_idx = []

        start_index = 0

        for table in self.tables:
            for _, row in table.iterrows():
                num_windows = max(row["num_windows"] - self.L, 0) + 1

                indices = np.arange(
                    start=start_index, stop=start_index + num_windows
                ).astype(np.int32)

                start_index += num_windows

                if row["split"] == 0:
                    train_idx.append(indices)
                elif row["split"] == 1:
                    valid_idx.append(indices)
                elif row["split"] == 2:
                    test_idx.append(indices)
                else:
                    error_string = "ERR: split should be 0, 1 or 2. Not " + str(
                        row["split"]
                    )
                    logger.error(error_string)
                    raise ValueError("ERR: split should be 0, 1 or 2")

        train_idx = np.concatenate(train_idx) if train_idx else np.array([])
        valid_idx = np.concatenate(valid_idx) if valid_idx else np.array([])
        test_idx = np.concatenate(test_idx)

        return train_idx, valid_idx, test_idx
