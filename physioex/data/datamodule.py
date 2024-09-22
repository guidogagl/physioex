import os
from typing import Callable, List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler, DistributedSampler, Subset
from physioex.data.dataset import PhysioExDataset


class PhysioExDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: List[str],
        batch_size: int = 32,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        folds: Union[int, List[int]] = -1,
        data_folder: str = None,
        hpc: bool = False,
        num_nodes : int = 1,
        num_workers : int = os.cpu_count(),
    ):
        super().__init__()

        self.datasets_id = datasets
        self.num_workers = num_workers // 2 if (preprocessing == "xsleepnet") and (len(selected_channels) > 1 ) else num_workers
        self.dataset = PhysioExDataset(
            datasets=datasets,
            preprocessing=preprocessing,
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            target_transform=target_transform,
            data_folder=data_folder,
            hpc=False,
        )

        self.batch_size = batch_size
        self.hpc = ( hpc == True ) and ( num_nodes > 1 )
        self.num_nodes = num_nodes 
        # if fold is an int
        if isinstance(folds, int):
            self.dataset.split(folds)
        else:
            assert len(folds) == len(
                datasets
            ), "ERR: folds and datasets should have the same length"
            for i, fold in enumerate(folds):
                self.dataset.split(fold, i)

        train_idx, valid_idx, test_idx = self.dataset.get_sets()

        if not self.hpc:
            self.train_dataset = self.dataset
            self.valid_dataset = self.dataset
            self.test_dataset = self.dataset
            
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)
            self.test_sampler = SubsetRandomSampler(test_idx)
        else:
            self.train_dataset = Subset(self.dataset, train_idx)
            self.valid_dataset = Subset(self.dataset, valid_idx)
            self.test_dataset = Subset(self.dataset, test_idx)
            
            self.train_sampler = self.train_dataset
            self.valid_sampler = self.valid_dataset
            self.test_sampler = self.test_dataset
                    
    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return  DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.train_sampler) if self.hpc else self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.valid_sampler) if self.hpc else self.valid_sampler,
            num_workers=self.num_workers,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.test_sampler) if self.hpc else self.test_sampler,
            num_workers=self.num_workers,
        )
