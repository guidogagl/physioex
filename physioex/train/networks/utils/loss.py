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


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.mae_loss = nn.L1Loss()

    def forward(self, emb, preds, targets):
        mae = self.mae_loss(preds, targets)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        ss_res = torch.sum((targets - preds) ** 2)
        r2_score = 1 - ss_res / ss_tot

        combined_loss = 0.5 * mae + 0.5 * (1 - r2_score)
        return combined_loss


config = {"cel": CrossEntropyLoss, "scl": SimilarityCombinedLoss, "reg": RegressionLoss}
