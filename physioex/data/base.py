from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from physioex.data.utils import StandardScaler


class PhysioExDataset(ABC):
    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def get_sets(self):
        pass


class TimeDistributedDataset(Dataset):
    def __init__(self, dataset, sequence_lenght, transform=None, target_transform=None):
        self.target_transform = target_transform
        self.input_transform = transform

        self.X = []
        self.y = []

        for i in range(len(dataset)):
            self.X.append(dataset[i][0])
            self.y.append(dataset[i][1])

        self.X = torch.tensor(np.array(self.X)).float()
        self.y = torch.tensor(np.array(self.y)).long()

        self.classes = torch.unique(self.y)
        self.n_classes = len(self.classes)

        self.scaler = None

        self.L = sequence_lenght

    def class_weights(self):

        class_weights = torch.zeros(len(self.classes))

        for i in range(len(self.classes)):
            class_weights[i] = torch.sum(self.y == self.classes[i])
            class_weights[i] = 1 / class_weights[i]

        class_weights = class_weights / class_weights.sum()

        return class_weights.float()

    def fit_scaler(self):

        if self.input_transform is not None:
            self.X = self.input_transform(self.X)
        
        self.scaler = StandardScaler().fit( self.X )
        self.X = self.scaler.transform(self.X)
        return self.scaler

    def set_scaler(self, scaler):
        self.scaler = scaler
        
        if self.input_transform is not None:
            self.X = self.input_transform(self.X)
            
        self.X = self.scaler.transform(self.X)
        return

    def __len__(self):
        return len(self.X) - self.L

    def __getitem__(self, idx):

        item = self.X[idx : idx + self.L].clone()

        label = self.y[idx : idx + self.L].clone()

        if self.target_transform:
            label = self.target_transform(label)

        return item, label


class TimeDistributedModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: PhysioExDataset,
        sequence_lenght: int,
        batch_size: int = 32,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.sequence_lenght = sequence_lenght
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size

        self.train, self.valid, self.test = self.dataset.get_sets()
        self.train = TimeDistributedDataset(
            self.train, self.sequence_lenght, self.transform, self.target_transform
        )
        self.valid = TimeDistributedDataset(
            self.valid, self.sequence_lenght, self.transform, self.target_transform
        )
        self.test = TimeDistributedDataset(
            self.test, self.sequence_lenght, self.transform, self.target_transform
        )

        scaler = self.train.fit_scaler()
        self.valid.set_scaler(scaler)
        self.test.set_scaler(scaler)

        self.classes = self.train.classes
        self.n_classes = self.train.n_classes

    def setup(self, stage: str):
        return

    def class_weights(self):
        return self.train.class_weights()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
