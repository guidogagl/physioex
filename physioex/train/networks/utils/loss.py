from abc import ABC, abstractmethod
from typing import Dict

import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ClassWeightedReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from torch import nn


class PhysioExLoss(ABC):

    @abstractmethod
    def forward(self, emb, preds, targets):
        pass


class SimilarityCombinedLoss(nn.Module, PhysioExLoss):
    def __init__(self):
        super(SimilarityCombinedLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.contr_loss = losses.TripletMarginLoss(
            distance=CosineSimilarity(),
            embedding_regularizer=LpRegularizer(),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        loss = self.ce_loss(preds, targets)
        hard_pairs = self.miner(emb, targets)

        return loss + self.contr_loss(emb, targets, hard_pairs)


class CrossEntropyLoss(nn.Module, PhysioExLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        return self.ce_loss(preds, targets)


class HuberLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

        self.loss = nn.HuberLoss(delta=5)

    def forward(self, emb, preds, targets):
        return self.loss(preds, targets) / 112.5


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

        # mse
        self.loss = nn.MSELoss()

    def forward(self, emb, preds, targets):
        return self.loss(preds, targets)


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):

        # reshape embeddings
        emb = emb.reshape(-1, 5, 3)  # batch_size, nclass, nchan
        # consider only the target class
        emb = emb[:, targets, :]

        # we want to maximize the value of these gradients

        emb_loss = emb.sum(dim=-1).mean()

        return self.loss(preds, targets) - emb_loss


config = {"cel": CrossEntropyLoss, "scl": SimilarityCombinedLoss, "reg": RegressionLoss}
