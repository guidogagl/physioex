import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import torch
from ml_dtypes import bfloat16

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32


class DataReader:
    def __init__(
        self,
        data_folder: str,
        dataset: str,
        preprocessing: str,
        seqlen: int,
        channels_index: List[int],
        offset: int = 0,
    ):
        # Initialize parameters
        self.data_folder = data_folder
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.seqlen = seqlen
        self.channels_index = channels_index
        self.offset = offset

        # path to the data and labels
        self.data_path = os.path.join(data_folder, dataset, preprocessing)
        self.labels_path = os.path.join(data_folder, dataset, "labels")

        # get the scaling parameters
        scaling = np.load(os.path.join(self.data_path, "scaling.npz"))

        self.input_shape = list(scaling["mean"].shape)

        self.mean = torch.tensor(scaling["mean"][channels_index]).to(DTYPE)
        self.std = torch.tensor(scaling["std"][channels_index]).to(DTYPE)

        # load the table
        self.table = pd.read_csv(os.path.join(data_folder, dataset, "table.csv"))

        # subject indexing, the max-subjects is 10.000 --> unsigned 16 is enough

        self.subject_index = np.zeros(self.table["num_windows"].sum(), dtype=np.uint16)
        start = 0
        for i, row in self.table.iterrows():
            end = start + row["num_windows"]
            self.subject_index[start:end] = i
            start = end

    def __len__(self) -> int:
        return self.table["num_windows"].sum()

    def __getitem__(
        self,
        idx: int,
        seqlen: int = None,
        return_subject_id: bool = False,
        return_subject_age: bool = False,
    ):
        if idx < self.offset:
            raise IndexError(f"Index {idx} is less than offset {self.offset}.")

        idx = idx - self.offset

        subject_row = self.subject_index[idx]
        subject_id = self.table.iloc[subject_row]["subject_id"]

        idx = idx - self.table.iloc[subject_row]["start_index"]

        if return_subject_id and return_subject_age:
            signal, labels, subject_id, age = self.get_subject(
                idx=idx,
                subject_id=subject_id,
                seqlen=seqlen,
                return_subject_age=return_subject_age,
            )
            return signal, labels, subject_id, age

        if return_subject_id:
            signal, labels, subject_id = self.get_subject(
                idx=idx,
                subject_id=subject_id,
                seqlen=seqlen,
                return_subject_age=return_subject_age,
            )
            return signal, labels, subject_id

        if return_subject_age:
            signal, labels, age = self.get_subject(
                idx=idx,
                subject_id=subject_id,
                seqlen=seqlen,
                return_subject_age=return_subject_age,
            )
            return signal, labels, age

        signal, labels = self.get_subject(
            idx=idx,
            subject_id=subject_id,
            seqlen=seqlen,
            return_subject_age=return_subject_age,
        )

        return signal, labels

    def get_subject(
        self,
        idx: int,
        subject_id: int,
        seqlen: int = None,
        return_subject_age: bool = False,
    ):

        if seqlen is None:
            seqlen = self.seqlen

        if seqlen == -1:
            subject_row = self.table[self.table["subject_id"] == subject_id].index[0]
            seqlen = self.table.iloc[subject_row]["num_windows"]

        signal = self.get_signal(idx, subject_id, seqlen)
        labels = self.get_labels(idx, subject_id, seqlen)

        if return_subject_age:
            age = self.table.loc[
                self.table["subject_id"] == subject_id, "nsrr_age"
            ].values.astype(float)[0]
            return signal, labels, age

        return signal, labels

    def get_table(self) -> pd.DataFrame:
        return self.table.copy()

    def set_table(self, table: pd.DataFrame):
        old_table = self.table.copy()
        self.table = table.copy()
        return old_table

    def get_n_subjects(self) -> int:
        return len(self.table["subject_id"].unique())

    def get_signal(self, idx: int, subject_id: int, seqlen: int) -> torch.Tensor:
        mp, num_windows = self.get_signal_memmap(subject_id)

        if idx >= num_windows:
            raise IndexError(
                f"Index {idx} out of range for subject {subject_id} with {num_windows} windows."
            )

        if idx + seqlen <= num_windows:
            signal = mp[idx : idx + seqlen, self.channels_index]
        else:
            pad_length = idx + seqlen - num_windows
            signal = mp[idx:, self.channels_index]

            # signal shape is  seq x ...
            # pad along the time dimension seq + pad_length x ...
            pad_width = [(0, pad_length)] + [(0, 0)] * (signal.ndim - 1)
            signal = np.pad(
                signal,
                pad_width=pad_width,
                mode="constant",
                constant_values=0,
            )

        signal = signal.astype(np.float32)

        # return ( torch.from_numpy(signal).to(DTYPE) * self.std ) + self.mean
        return torch.from_numpy(signal).to(DTYPE)

    def get_labels(self, idx: int, subject_id: int, seqlen: int) -> torch.Tensor:
        mp, num_windows = self.get_labels_memmap(subject_id)
        if idx >= num_windows:
            raise IndexError(
                f"Index {idx} out of range for subject {subject_id} with {num_windows} windows."
            )

        if idx + seqlen <= num_windows:
            labels = mp[idx : idx + seqlen]
        else:
            pad_length = idx + seqlen - num_windows
            labels = mp[idx:]
            labels = np.pad(
                labels, pad_width=(0, pad_length), mode="constant", constant_values=-1
            )

        labels = labels.astype(np.int64)

        return torch.from_numpy(labels).long()

    def get_signal_memmap(
        self,
        subject_id: str,
    ) -> np.memmap:
        data_path = os.path.join(
            self.data_path,
            f"{subject_id}.npy",
        )

        num_windows = self.table.loc[
            self.table["subject_id"] == subject_id, "num_windows"
        ].values[0]
        input_shape = tuple([num_windows] + self.input_shape)

        return (
            np.memmap(data_path, dtype=bfloat16, mode="r", shape=input_shape),
            num_windows,
        )

    def get_labels_memmap(
        self,
        subject_id: str,
    ) -> np.memmap:
        labels_path = os.path.join(
            self.labels_path,
            f"{subject_id}.npy",
        )
        num_windows = self.table.loc[
            self.table["subject_id"] == subject_id, "num_windows"
        ].values[0]
        labels_shape = (num_windows,)
        return (
            np.memmap(labels_path, dtype="int16", mode="r", shape=labels_shape),
            num_windows,
        )

    def get_seqlen(self) -> int:
        return self.seqlen

    def set_seqlen(self, seqlen: int):
        old_seqlen = self.seqlen
        self.seqlen = seqlen
        return old_seqlen

    def get_scaling(self):
        return self.mean, self.std
