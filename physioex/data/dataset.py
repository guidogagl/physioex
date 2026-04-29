import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from typing import List, Callable, Tuple

from physioex.data.datareader import DataReader, DTYPE

import numpy as np
import pandas as pd


def merge_scaling(means, std_devs, sample_sizes):
    means = [mean.to(torch.float64) for mean in means]
    std_devs = [std.to(torch.float64) for std in std_devs]

    total_samples = sum(sample_sizes)

    total_mean = sum(M * N for M, N in zip(means, sample_sizes)) / total_samples

    total_variance = (
        sum(
            N * (D**2 + (M - total_mean) ** 2)
            for M, D, N in zip(means, std_devs, sample_sizes)
        )
        / total_samples
    )

    total_variance = torch.clamp(total_variance, min=0.0) + 1e-10
    total_std_dev = torch.sqrt(total_variance)
    return total_mean.to(DTYPE), total_std_dev.to(DTYPE)


class PhysioExDataset(Dataset):
    def __init__(
        self,
        datasets: List[str] = None,
        preprocessing: str = "raw",
        selected_channels: List[str] = ["EEG"],
        seqlen: int = 21,
        indexed_channels: List[str] = ["EEG", "EOG", "EMG", "ECG"],
        target_transform: Callable = None,
        data_folder: str = None,
    ):

        super().__init__()

        self.datasets = datasets
        self.preprocessing = preprocessing
        self.seqlen = seqlen

        self.selected_channels = selected_channels
        self.indexed_channels = indexed_channels
        self.channels_index = [indexed_channels.index(ch) for ch in selected_channels]

        self.target_transform = target_transform
        self.data_folder = data_folder

        if self.data_folder is None:
            self.data_folder = os.environ.get("PHYSIOEX_DATA_PATH", None)

        # read the PHYSIOEX_CONFIG.yaml file if it exists
        self.get_parameters_from_config()

        if self.data_folder is None:
            raise ValueError(
                "[Err.] PhysioExDataset : Data folder not specified. Please set the PHYSIOEX_DATA_PATH environment variable or provide the data_folder argument."
            )

        self.readers = []
        self.tables = []
        self.dataset_idx = []

        offset = 0

        means, stds, sizes = [], [], []
        for i, dataset in enumerate(self.datasets):
            reader = DataReader(
                data_folder=self.data_folder,
                dataset=dataset,
                preprocessing=self.preprocessing,
                seqlen=self.seqlen,
                channels_index=self.channels_index,
                offset=offset,
            )

            size = len(reader)

            offset += size

            mean, std = reader.get_scaling()

            means.append(mean)
            stds.append(std)
            sizes.append(size)

            self.dataset_idx += list(np.ones(size) * i)
            self.readers += [reader]

        self.dataset_idx = np.array(self.dataset_idx, dtype=np.uint16)

        self.mean, self.std = merge_scaling(means, stds, sizes)

    def __getitem__(self, idx: int):
        dataset_idx = self.dataset_idx[idx]
        signal, labels = self.readers[dataset_idx].__getitem__(idx)

        # scale the signal
        # signal = ( signal - self.mean ) / self.std

        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return signal, labels

    def split(self, fold: int = 0):
        train_indexes = []
        valid_subjects, test_subjects = [], []

        dataset_offset = 0
        for dataset_idx, reader in enumerate(self.readers):

            table = reader.get_table()
            table = table[["subject_id", "num_windows", f"fold_{fold}"]]
            for i, row in table.iterrows():
                num_windows = row["num_windows"]
                if row[f"fold_{fold}"] == "valid":
                    valid_subjects.append((dataset_idx, row["subject_id"]))
                elif row[f"fold_{fold}"] == "test":
                    test_subjects.append((dataset_idx, row["subject_id"]))
                else:
                    train_indexes += list(
                        range(dataset_offset, dataset_offset + num_windows)
                    )

                dataset_offset += num_windows

        train_indexes = np.array(train_indexes, dtype=np.int32)

        return train_indexes, valid_subjects, test_subjects

    def get_table(self, dataset_idx: int) -> pd.DataFrame:
        return self.readers[dataset_idx].get_table()

    def get_num_channels(self):
        return len(self.selected_channels)

    def set_table(self, dataset_idx: int, table: pd.DataFrame):
        old_table = self.readers[dataset_idx].set_table(table)
        return old_table

    def get_subject(
        self,
        dataset_idx: int,
        subject_id: int,
        seqlen: int = None,
        return_subject_age: bool = False,
    ):
        return self.readers[dataset_idx].get_subject(
            idx=0,
            subject_id=subject_id,
            seqlen=seqlen,
            return_subject_age=return_subject_age,
        )

    def get_n_subjects(self):
        n_subjects = 0
        for reader in self.readers:
            n_subjects += reader.get_n_subjects()
        return n_subjects

    def set_scaling(self, mean: torch.Tensor, std: torch.Tensor):
        old_mean = self.mean
        old_std = self.std

        self.mean = mean
        self.std = std

        return old_mean, old_std

    def get_scaling(self):
        return self.mean, self.std

    def get_parameters_from_config(self):
        config = {}
        try:
            with open("PHYSIOEX_CONFIG.yaml", "r") as f:
                yaml_content = yaml.safe_load(f) or {}
                config = yaml_content.get("PhysioExDataset", {})
                print(
                    "[Info - PhysioExDataset]: Loaded configuration from PHYSIOEX_CONFIG.yaml file."
                )
        except FileNotFoundError:
            print(
                "[Warning - PhysioExDataset]: PHYSIOEX_CONFIG.yaml file not found. Using default parameters or those provided in the function call."
            )
        except Exception as exc:
            print(
                f"[Warning - PhysioExDataset]: Failed to load PHYSIOEX_CONFIG.yaml due to {exc}. Using default parameters."
            )

        # override parameters from the config file if they are not None
        if "datasets" in config and config["datasets"] is not None:
            self.datasets = config["datasets"]
        if "preprocessing" in config and config["preprocessing"] is not None:
            self.preprocessing = config["preprocessing"]
        if "seqlen" in config and config["seqlen"] is not None:
            self.seqlen = config["seqlen"]
        if "indexed_channels" in config and config["indexed_channels"] is not None:
            self.indexed_channels = config["indexed_channels"]
        if "selected_channels" in config and config["selected_channels"] is not None:
            self.selected_channels = config["selected_channels"]
            self.channels_index = [
                self.indexed_channels.index(ch) for ch in self.selected_channels
            ]
        if "data_folder" in config and config["data_folder"] is not None:
            self.data_folder = config["data_folder"]


class _PhysioExTrainDataset(Dataset):
    def __init__(self, dataset: PhysioExDataset, train_indexes: List[int]):
        super().__init__()
        self.dataset = dataset
        self.train_indexes = train_indexes

    def __len__(self):
        return len(self.train_indexes)

    def __getitem__(self, idx: int):
        actual_idx = self.train_indexes[idx]
        return self.dataset[actual_idx]


class _PhysioExEvalDataset(Dataset):
    def __init__(self, dataset: PhysioExDataset, eval_subjects: List[tuple]):
        super().__init__()

        self.dataset = dataset
        self.eval_subjects = eval_subjects

        # get the num max_windows
        max_windows = 0
        for dataset_idx, subject_id in eval_subjects:
            table = self.dataset.get_table(dataset_idx)
            num_windows = table.loc[
                table["subject_id"] == subject_id, "num_windows"
            ].values[0]
            if num_windows > max_windows:
                max_windows = num_windows

        self.max_windows = max_windows

    def __len__(self):
        return len(self.eval_subjects)

    def __getitem__(self, idx: int):
        dataset_idx, subject_id = self.eval_subjects[idx]
        subject_signal, subject_labels = self.dataset.get_subject(
            dataset_idx=dataset_idx, subject_id=subject_id, seqlen=self.max_windows
        )
        return subject_signal, subject_labels


def get_dataloaders(
    dataset: PhysioExDataset,
    fold: int = 0,
    train_batch_size: int = 32,
    eval_batch_size: int = 1,
    distributed: bool = False,
    **dataloader_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_indexes, valid_subjects, test_subjects = dataset.split(fold=fold)
    train_dataset = _PhysioExTrainDataset(dataset, train_indexes)
    valid_dataset = _PhysioExEvalDataset(dataset, valid_subjects)
    test_dataset = _PhysioExEvalDataset(dataset, test_subjects)

    # create distalibuted samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=train_batch_size,
            **dataloader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=eval_batch_size,
            **dataloader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=eval_batch_size,
            **dataloader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=train_batch_size,
            **dataloader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            batch_size=eval_batch_size,
            **dataloader_kwargs,
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=eval_batch_size, **dataloader_kwargs
        )

    return train_loader, valid_loader, test_loader
