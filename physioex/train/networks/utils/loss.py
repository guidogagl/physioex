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
    def __init__(self, params: Dict):
        super(SimilarityCombinedLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.contr_loss = losses.TripletMarginLoss(
            distance=CosineSimilarity(),
            reducer=ClassWeightedReducer(weights=params["class_weights"]),
            embedding_regularizer=LpRegularizer(),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, emb, preds, targets):
        loss = self.ce_loss(preds, targets)
        hard_pairs = self.miner(emb, targets)

        return loss + self.contr_loss(emb, targets, hard_pairs)


class CrossEntropyLoss(nn.Module, PhysioExLoss):
    def __init__(self, params: Dict = None):
        super(CrossEntropyLoss, self).__init__()

        # check if class weights are provided in params
        if params is not None:
            weights = params.get("class_weights", None)
        else:
            weights = None
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, emb, preds, targets):
        return self.ce_loss(preds, targets)


config = {"cel": CrossEntropyLoss, "scl": SimilarityCombinedLoss}
