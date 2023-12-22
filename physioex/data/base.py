from abc import ABC, abstractmethod

import pytorch_lightning as pl 

from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np

from sklearn.utils import compute_class_weight

class PhysioExDataset(ABC):
    @abstractmethod
    def split(self):
        pass
    @abstractmethod
    def get_sets(self):
        pass


class TimeDistributedDataset(Dataset):
    def __init__(self, dataset, sequence_lenght, transform=None, target_transform=None):
        self.X = [] 
        self.y = []
        
        for i in range( len(dataset) ):
            self.X.append( dataset[i][0] )
            self.y.append( dataset[i][1] )
    
        self.X = torch.tensor( np.array(self.X) ).float()
        self.y = torch.tensor( np.array(self.y) ).long()

        self.classes = torch.unique( self.y )
        self.n_classes = len( self.classes )
        
        #self.weights = compute_class_weight( "balanced", self.classes.numpy(), self.y.numpy() )
        #self.weights = torch.tensor( self.weights )

        self.L = sequence_lenght 

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X) - self.L

    def __getitem__(self, idx):

        item = self.X[ idx:idx+self.L ].clone()
        label = self.y[ idx:idx+self.L ].clone()

        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            label = self.target_transform(label)

        return item, label


class TimeDistributedModule(pl.LightningDataModule):
    def __init__(self, dataset: PhysioExDataset, sequence_lenght : int, batch_size: int = 32, transform = None, target_transform = None):
        super().__init__()
        self.dataset = dataset
        self.sequence_lenght = sequence_lenght
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size

        self.train, self.valid, self.test = self.dataset.get_sets()
        self.train = TimeDistributedDataset(self.train, self.sequence_lenght, self.transform, self.target_transform)
        self.valid = TimeDistributedDataset(self.valid, self.sequence_lenght, self.transform, self.target_transform)
        self.test = TimeDistributedDataset(self.test, self.sequence_lenght, self.transform, self.target_transform)

        self.classes = self.train.classes
        self.n_classess = self.train.n_classes
        
    def setup(self, stage: str):
        return

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)
