from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Dataset


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

        self.X = []
        self.y = []

        for i in range(len(dataset)):
            self.X.append(dataset[i][0])
            self.y.append(dataset[i][1])

        self.X = torch.tensor(np.array(self.X)).float()
        self.y = torch.tensor(np.array(self.y)).long()

        if transform is not None:
            self.X = transform(self.X)

        self.classes = torch.unique(self.y)
        self.n_classes = len(self.classes)

        self.L = sequence_lenght

    def class_weights(self):

        class_weights = torch.zeros(len(self.classes))

        for i in range(len(self.classes)):
            class_weights[i] = torch.sum(self.y == self.classes[i]) / len(self.y)

        return class_weights.float()

    def fit_scaler(self, scaler=StandardScaler()):
        shape = self.X.size()
        self.X = torch.tensor(
            scaler.fit_transform(self.X.reshape(shape[0], -1)).reshape(shape)
        ).float()

        return scaler

    def scale(self, scaler):
        shape = self.X.size()
        self.X = torch.tensor(
            scaler.transform(self.X.reshape(shape[0], -1)).reshape(shape)
        ).float()

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
        scaler=StandardScaler(),
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

        scaler = self.train.fit_scaler(scaler)
        self.valid.scale(scaler)
        self.test.scale(scaler)

        self.classes = self.train.classes
        self.n_classess = self.train.n_classes

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
