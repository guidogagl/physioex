import os
import time
from abc import ABC, abstractmethod
from typing import List

import h5py as h5
import numpy as np
import pandas as pd
import torch
from loguru import logger


class Reader(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_table(self):
        pass


class MemmapReader(Reader):
    # this object abstracts the reading of subject data from memmap files for a specific dataset
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        sequence_length: int,
        channels_index: List[int],
        offset: int,
    ):

        self.preprocessing = preprocessing

        self.data_path = os.path.join(data_folder, dataset, preprocessing)
        self.labels_path = os.path.join(data_folder, dataset, "labels")

        self.L = sequence_length
        self.channels_index = channels_index
        self.offset = offset

        # get the scaling parameters
        scaling = np.load(os.path.join(self.data_path, "scaling.npz"))

        self.input_shape = list(scaling["mean"].shape)

        self.mean = torch.tensor(scaling["mean"][channels_index]).float()
        self.std = torch.tensor(scaling["std"][channels_index]).float()

        # read the table
        self.table = pd.read_csv(os.path.join(data_folder, dataset, "table.csv"))

        num_windows = self.table["num_windows"].values
        subjects_id = self.table["subject_id"].values

        self.len = int(np.sum(self.table["num_windows"].values - self.L + 1))
        self.subject_idx, self.relative_idx, self.windows_index = build_index(
            num_windows, subjects_id, self.L
        )

    def get_table(self):
        folds_colum = [col for col in self.table.columns if "fold_" in col]
        return self.table[["subject_id", "num_windows"] + folds_colum].copy()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = idx - self.offset

        relative_id = self.relative_idx[idx]
        subject_id = self.subject_idx[idx]
        num_windows = self.windows_index[subject_id]

        input_shape = tuple([num_windows] + self.input_shape)
        labels_shape = (num_windows,)

        data_path = os.path.join(self.data_path, str(subject_id) + ".npy")
        labels_path = os.path.join(self.labels_path, str(subject_id) + ".npy")

        X = np.memmap(data_path, dtype="float32", mode="r", shape=input_shape)
        y = np.memmap(labels_path, dtype="int16", mode="r", shape=labels_shape)

        X = X[relative_id : relative_id + self.L, self.channels_index]
        y = y[relative_id : relative_id + self.L]

        X = (torch.tensor(X).float() - self.mean) / self.std
        y = torch.tensor(y).long()

        return X, y


class H5Reader(Reader):
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        sequence_length: int,
        channels_index: List[int],
        offset: int,
    ):

        self.L = sequence_length
        self.channels_index = channels_index
        self.offset = offset
        self.preprocessing = preprocessing
        self.file_path = os.path.join(data_folder, dataset + ".h5")
        file = h5.File(self.file_path, "r")

        self.mean = torch.tensor(
            file[preprocessing]["mean"][channels_index][()]
        ).float()
        self.std = torch.tensor(file[preprocessing]["std"][channels_index][()]).float()

        num_windows = file["num_windows"][()]
        subjects_id = file["subject_id"][()]
        file.close()

        self.len = int(np.sum(num_windows - self.L + 1))

        self.subject_idx, self.relative_idx, _ = build_index(
            num_windows, subjects_id, self.L
        )

        self.input_shape = None
        self.windows_index = None

    def __len__(self):
        return self.len

    def get_table(self):

        table = pd.DataFrame([])

        with h5.File(self.file_path, "r") as file:
            table["subject_id"] = file["subject_id"][()]
            table["num_windows"] = file["num_windows"][()]

            keys = list(file.keys())
            for key in keys:
                if "fold_" in key:
                    table[key] = file[key][()].astype(int)
                    table[key] = table[key].map({0: "train", 1: "valid", 2: "test"})

        return table

    def __getitem__(self, idx):

        idx = idx - self.offset

        relative_id = self.relative_idx[idx]
        subject_id = self.subject_idx[idx]

        with h5.File(self.file_path, "r") as file:
            X = file[self.preprocessing][str(subject_id)][
                relative_id : relative_id + self.L, self.channels_index
            ][()]
            y = file["labels"][str(subject_id)][relative_id : relative_id + self.L][()]

        X = (torch.tensor(X).float() - self.mean) / self.std
        y = torch.tensor(y).long()

        return X, y


class DataReader(Reader):
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        sequence_length: int,
        channels_index: List[int],
        offset: int,
        hpc: bool,
    ):

        self.reader = MemmapReader(
            data_folder=data_folder,
            dataset=dataset,
            preprocessing=preprocessing,
            sequence_length=sequence_length,
            channels_index=channels_index,
            offset=offset,
        )

    def __len__(self):
        return self.reader.__len__()

    def __getitem__(self, idx):
        return self.reader[idx]

    def __del__(self):
        if hasattr(self, "reader"):
            del self.reader

    def get_table(self):
        return self.reader.get_table()


def build_index(nums_windows, subjects_ids, sequence_length):
    data_len = int(np.sum(nums_windows - sequence_length + 1))

    subject_idx = np.zeros(data_len, dtype=np.uint16)
    relative_idx = np.zeros(data_len, dtype=np.uint16)

    start_index = 0

    for i, (num_windows, subject_id) in enumerate(zip(nums_windows, subjects_ids)):

        # check if the subject_id or the relative_id can be stored in a uint16
        if subject_id > np.iinfo(np.uint16).max:
            raise ValueError(
                f"subject_id {subject_id} exceeds the maximum value for np.uint16"
            )

        # Check if the relative_id can be stored in a uint16
        if num_windows - sequence_length + 1 > np.iinfo(np.uint16).max:
            raise ValueError(
                f"Relative index {num_windows - sequence_length + 1} exceeds the maximum value for np.uint16"
            )

        subject_idx[start_index : start_index + num_windows - sequence_length + 1] = (
            subject_id
        )
        relative_idx[start_index : start_index + num_windows - sequence_length + 1] = (
            np.arange(num_windows - sequence_length + 1)
        )
        start_index += num_windows - sequence_length + 1

    windows_index = np.zeros(np.max(subjects_ids) + 1, dtype=np.uint16)
    for i, num_windows in enumerate(nums_windows):
        windows_index[subjects_ids[i]] = num_windows

    return subject_idx, relative_idx, windows_index
