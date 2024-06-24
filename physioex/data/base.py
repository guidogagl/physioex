import os
import pickle
from typing import Callable, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from physioex.data.constant import get_data_folder
from physioex.data.utils import read_config


def transform_to_sequence(x, y, sequence_length):

    # x shape num_samples x num_features convert it into num_samples x sequence_length x num_features
    # y shape num_samples convert it into num_samples x sequence_length

    x_seq = np.zeros((len(x) - sequence_length + 1, sequence_length, *x.shape[1:]))
    y_seq = np.zeros((len(y) - sequence_length + 1, sequence_length))

    for i in range(len(x) - sequence_length + 1):
        x_seq[i] = x[i : i + sequence_length]
        y_seq[i] = y[i : i + sequence_length]

    return x_seq, y_seq


def create_subject_index_map(df, sequence_length):

    max_windows = int(np.sum(df["num_samples"] - sequence_length + 1))
    window_to_subject = np.zeros(max_windows, dtype=np.int16)
    subject_to_start = np.zeros(df["subject_id"].max() + 1, dtype=np.int32)

    start_index = 0

    for _, row in df.iterrows():
        subject = int(row["subject_id"])
        num_windows = int(row["num_samples"] - sequence_length + 1)

        window_to_subject[start_index : start_index + num_windows] = subject
        subject_to_start[subject] = start_index

        start_index += num_windows

    return window_to_subject, subject_to_start


def find_subject_for_window(index, window_to_subject, subject_to_start):
    subject = window_to_subject[index]
    start_index = subject_to_start[subject]
    return subject, int(index - start_index)


class PhysioExDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_folder: str,
        preprocessing: str,  # available [ "raw", "xsleepnet" ]
        input_shape: List[int],
        sequence_length: int,
        selected_channels: List[int],
        target_transform: Callable = None,
        task: str = "sleep",
    ):
        table_path = os.path.join(dataset_folder, "table.csv")
        self.table = pd.read_csv(table_path)

        self.window_to_subject, self.subject_to_start = create_subject_index_map(
            self.table, sequence_length
        )

        data_path = os.path.join(dataset_folder, preprocessing + ".dat")
        labels_path = os.path.join(dataset_folder, "labels.dat")

        self.X = np.memmap(
            data_path,
            dtype="float32",
            mode="r",
            shape=(int(np.sum(self.table["num_samples"].values)), *input_shape),
        )

        self.y = np.memmap(
            labels_path,
            dtype="int16",
            mode="r",
            shape=(int(np.sum(self.table["num_samples"].values))),
        )

        # self.X = self.X[:]
        # self.y = self.y[:]

        split_path = os.path.join(dataset_folder, "splitting.pkl")

        with open(split_path, "rb") as f:
            self.splitting = pickle.load(f)

        scaling_path = os.path.join(dataset_folder, preprocessing + "_scaling.npz")
        scaling = np.load(scaling_path)

        self.mean, self.std = scaling["mean"], scaling["std"]
        self.mean, self.std = self.mean[selected_channels], self.std[selected_channels]

        self.mean, self.std = (
            torch.tensor(self.mean).float(),
            torch.tensor(self.std).float(),
        )

        self.L = sequence_length
        self.target_transform = target_transform
        self.selected_channels = selected_channels

        self.task = task

    def __len__(self):
        return int(np.sum(self.table["num_samples"] - self.L + 1))

    def __getitem__(self, idx):

        subject_id, relative_id = find_subject_for_window(
            idx, self.window_to_subject, self.subject_to_start
        )

        subject_start_indx = int(
            self.table[self.table["subject_id"] == subject_id]["start_index"].values[0]
        )

        absolute_id = subject_start_indx + relative_id

        X = self.X[absolute_id : absolute_id + self.L]

        if self.task == "sleep":
            y = self.y[absolute_id : absolute_id + self.L]
            y = torch.tensor(y).long()
        elif self.task == "age":
            y = (
                np.ones(self.L)
                * float(
                    self.table[self.table["subject_id"] == subject_id]["age"].values[0]
                )
                / 100
            )
            y = torch.tensor(y).float()
        elif self.task == "sex":
            y = np.ones(self.L) * int(
                self.table[self.table["subject_id"] == subject_id]["sex"].values[0]
            )
            y = torch.tensor(y).long()
        else:
            raise ValueError("the task is unrecognized")

        # L, n_channels, ... other dims

        X = X[:, self.selected_channels]

        return torch.tensor(X).float(), y.view(-1)

    def get_num_folds(self):
        return len(self.splitting["train"])

    def split(self, fold: int):

        valid_subjects = self.splitting["valid"][fold].astype(np.int16)
        test_subjects = self.splitting["test"][fold].astype(np.int16)

        # add a column to the table with 0 if the subject is in train, 1 if in valid, 2 if in test

        split = np.zeros(len(self.table)).astype(np.int8)
        split[self.table["subject_id"].isin(test_subjects)] = 2
        split[self.table["subject_id"].isin(valid_subjects)] = 1

        self.table["split"] = split

    def get_sets(self):
        # return the indexes in the table of the train, valid and test subjects
        train_idx = []
        valid_idx = []
        test_idx = []

        for _, row in self.table.iterrows():
            subject_id = int(row["subject_id"])

            start_index = self.subject_to_start[subject_id]
            num_windows = row["num_samples"] - self.L + 1
            indices = np.arange(
                start=start_index, stop=start_index + num_windows
            ).astype(np.int32)

            if row["split"] == 0:
                train_idx.append(indices)
            elif row["split"] == 1:
                valid_idx.append(indices)
            elif row["split"] == 2:
                test_idx.append(indices)

        train_idx = np.concatenate(train_idx)
        valid_idx = np.concatenate(valid_idx)
        test_idx = np.concatenate(test_idx)

        return train_idx, valid_idx, test_idx


class TimeDistributedModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: PhysioExDataset,
        batch_size: int = 32,
        fold: int = 0,
        # num_workers: int = 32,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        self.dataset.split(fold)

        self.train_idx, self.valid_idx, self.test_idx = self.dataset.get_sets()

        self.num_workers = os.cpu_count() // 4

    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.train_idx),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.valid_idx),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.test_idx),
            num_workers=self.num_workers,
        )


class CombinedTimeDistributedModule(TimeDistributedModule):
    def __init__(
        self,
        dataset: PhysioExDataset,
        batch_size: int = 32,
        num_workers: int = 32,
    ):
        super().__init__(dataset, batch_size, 0, num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
