import os
from typing import Callable, List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, DistributedSampler, Subset, SubsetRandomSampler

from physioex.data.dataset import PhysioExDataset


class PhysioExDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: Union[List[str], PhysioExDataset],
        batch_size: int = 32,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        task: str = "sleep",
        folds: Union[int, List[int]] = -1,
        data_folder: str = None,
        num_nodes: int = 1,
        num_workers: int = os.cpu_count(),
    ):
        super().__init__()

        self.datasets_id = datasets
        self.num_workers = num_workers

        if isinstance(datasets, list):
            self.dataset = PhysioExDataset(
                datasets=datasets,
                preprocessing=preprocessing,
                selected_channels=selected_channels,
                sequence_length=sequence_length,
                target_transform=target_transform,
                data_folder=data_folder,
                task=task,
            )
        elif isinstance(datasets, PhysioExDataset):
            self.dataset = datasets
        else:
            raise ValueError("ERR: datasets should be a list or a PhysioExDataset")

        self.batch_size = batch_size

        if isinstance(folds, int):
            self.dataset.split(folds)
        else:
            assert len(folds) == len(
                datasets
            ), "ERR: folds and datasets should have the same length"
            for i, fold in enumerate(folds):
                self.dataset.split(fold, i)

        train_idx, valid_idx, test_idx = self.dataset.get_sets()

        self.train_dataset = Subset(self.dataset, train_idx)
        self.valid_dataset = Subset(self.dataset, valid_idx)
        self.test_dataset = Subset(self.dataset, test_idx)

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
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
