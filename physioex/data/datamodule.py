from typing import Callable, List, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from physioex.data.dataset import PhysioExDataset


class PreFetchTrainDataset(Dataset):
    def __init__(
        self,
        X: List[torch.Tensor],
        y: List[torch.Tensor],
        L: int,
    ):
        self.X = torch.cat(X, dim=0)
        self.y = torch.cat(y, dim=0)

        logger.info(f"Pre-Fetched X: {self.X.shape}, y: {self.y.shape}")

        self.L = L

    def __len__(self):
        return len(self.y) - self.L + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.L], self.y[idx : idx + self.L], 0, 0


class PreFetchTestDataset(Dataset):
    def __init__(
        self,
        X: List[torch.Tensor],
        y: List[torch.Tensor],
    ):
        self.X = X
        self.y = y

        logger.info(f"Pre-Fetched X: {len(self.X)}, y: {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], 0, 0


class PhysioExDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: Union[List[str], PhysioExDataset],
        eval_datasets :  Union[List[str], PhysioExDataset] = None,
        batch_size: int = 32,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        task: str = "sleep",
        folds: Union[int, List[int]] = 0,
        data_folder: str = None,
        num_workers: int = 0,
        data_prefetch: bool = True,
    ):
        super().__init__()

        self.datasets_id = datasets
        self.num_workers = num_workers

        if eval_datasets == None:
            eval_datasets = datasets

        # dataset configuration
        if isinstance(datasets, list):
            self.dataset = PhysioExDataset(
                datasets=datasets,
                preprocessing=preprocessing,
                selected_channels=selected_channels,
                sequence_length=sequence_length if not data_prefetch else -1,
                target_transform=target_transform,
                data_folder=data_folder,
                task=task,
            )
        elif isinstance(datasets, PhysioExDataset):
            self.dataset = datasets
        else:
            raise ValueError("ERR: datasets should be a list or a PhysioExDataset")

        if isinstance(eval_datasets, list):
            self.eval_dataset = PhysioExDataset(
                datasets=eval_datasets,
                preprocessing=preprocessing,
                selected_channels=selected_channels,
                sequence_length=-1,
                target_transform=target_transform,
                data_folder=data_folder,
                task=task,
            )
        elif isinstance(eval_datasets, PhysioExDataset):
            self.eval_dataset = eval_datasets

        self.batch_size = batch_size

        if isinstance(folds, int):
            self.dataset.split(folds)
            self.eval_dataset.split(folds)
        else:
            assert len(folds) == len(datasets), (
                "ERR: folds and datasets should have the same length"
            )
            for i, fold in enumerate(folds):
                self.dataset.split(fold, i)
                self.eval_dataset.split(fold, i)

        train_idx, _, _ = self.dataset.get_sets()
        _, valid_idx, test_idx = self.eval_dataset.get_sets()

        if not data_prefetch:
            self.train_dataset = Subset(self.dataset, train_idx)
            self.valid_dataset = Subset(self.eval_dataset, valid_idx)
            self.test_dataset = Subset(self.eval_dataset, test_idx)

            self.len = len(train_idx)
        else:
            X_train, y_train = [], []

            for idx in tqdm(train_idx, desc="Pre-fetching train data"):
                x, y, _, _ = self.dataset[idx]
                X_train.append(x)
                y_train.append(y)

            self.train_dataset = PreFetchTrainDataset(X_train, y_train, sequence_length)

            X_valid, y_valid = [], []
            for idx in tqdm(valid_idx, desc="Pre-fetching valid data"):
                x, y, _, _ = self.eval_dataset[idx]
                X_valid.append(x)
                y_valid.append(y)

            self.valid_dataset = PreFetchTestDataset(X_valid, y_valid)

            X_test, y_test = [], []
            for idx in tqdm(test_idx, desc="Pre-fetching test data"):
                x, y, _, _ = self.eval_dataset[idx]
                X_test.append(x)
                y_test.append(y)

            self.test_dataset = PreFetchTestDataset(X_test, y_test)

            self.len = len(self.train_dataset)

    def __len__(self):
        return self.len

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,  # self.batch_size if not self.eown else 1,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # self.batch_size if not self.eown else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            # persistent_workers=True
        )

    def all_dataloader(self, eval=False, shuffle=False):
        logger.warning('You are loading the full dataset, no split was made.')
        return DataLoader(
            self.dataset if not eval else self.eval_dataset,
            batch_size=self.batch_size if not eval else 1, # self.batch_size if not self.eown else 1,
            shuffle=shuffle,
            num_workers=self.num_workers,
            # persistent_workers=True
        )
