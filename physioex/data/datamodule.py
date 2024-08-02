import os
from typing import Callable, List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler

from physioex.data.dataset import PhysioExDataset


class PhysioExDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: List[str],
        versions: List[str] = None,
        folds: Union[int, List[int]] = -1,
        batch_size: int = 32,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        #task: str = "sleep",
        data_folder: str = None,
    ):
        super().__init__()

        self.dataset = PhysioExDataset(
            datasets=datasets,
            versions=versions,
            preprocessing=preprocessing,
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            target_transform=target_transform,
            #task=task,
            data_folder=data_folder,
        )

        self.batch_size = batch_size
         
        # if fold is an int
        if isinstance(folds, int):
            self.dataset.split(folds)
        else:
            assert len(folds) == len(
                datasets
            ), "ERR: folds and datasets should have the same length"
            for i, fold in enumerate(folds):
                self.dataset.split(fold, i)

        self.train_idx, self.valid_idx, self.test_idx = self.dataset.get_sets()

        self.num_workers = os.cpu_count() // 2

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
