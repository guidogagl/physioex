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
    ):
        super().__init__()

        self.dataset = PhysioExDataset(
            datasets=datasets,
            preprocessing=preprocessing,
            selected_channels=selected_channels,
            sequence_length=sequence_length,
            target_transform=target_transform,
            data_folder=data_folder,
            hpc=hpc,
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

        train_idx, valid_idx, test_idx = self.dataset.get_sets()

        if not hpc:
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.valid_sampler = SubsetRandomSampler(valid_idx)
            self.test_sampler = SubsetRandomSampler(test_idx)
        else:
            self.train_sampler = DistributedSampler(Subset(self.dataset, train_idx))
            self.valid_sampler = DistributedSampler(Subset(self.dataset, valid_idx))
            self.test_sampler = DistributedSampler(Subset(self.dataset, test_idx))

        self.train = DataLoader(
            self.dataset if not hpc else Subset(self.dataset, train_idx),
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=os.cpu_count(),
        )
        
        self.valid = DataLoader(
            self.dataset if not hpc else Subset(self.dataset, valid_idx),
            batch_size=self.batch_size,
            sampler=self.valid_sampler,
            num_workers=os.cpu_count(),
        )
        
        self.test = DataLoader(
            self.dataset if not hpc else Subset(self.dataset, test_idx),
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=os.cpu_count(),
        )        
                    
    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test
    
    def teardown(self, stage: str):
        del self.train, self.valid, self.test, self.dataset
        return