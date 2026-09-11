"""End-to-end LRP for whole PhysioEx models (Phase 2c wiring).

:func:`prepare_model_for_lrp` swaps every fused / attention block in a model for
its relevance-carrying counterpart:

* ``nn.LSTM`` / ``nn.GRU``               → :class:`LRPLSTM` / :class:`LRPGRU`
* ``nn.TransformerEncoder(Layer)``       → the ``transformer`` LRP versions
* ``nn.MultiheadAttention``              → :class:`LRPMultiheadAttention`
* custom softmax pooling (matched by class name ``AttentionPooling`` /
  ``AttentionLayer``) → the ``pooling`` CP-LRP versions
* remaining leaf ``Linear`` / ``Conv`` / ``BatchNorm`` → LXT ``EpsilonRule``
* standalone ``LayerNorm``               → LXT ``IdentityRule``

:class:`ModelLRP` deep-copies the trained model, prepares it, then runs a forward
plus a target-seeded backward, returning relevance shaped like the input.

Requires the ``explain`` extra (``lxt``).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from physioex.explain.lrp.pooling import LRPAttentionLayer, LRPAttentionPooling
from physioex.explain.lrp.recurrent import LRPGRU, LRPLSTM
from physioex.explain.lrp.transformer import (
    LRPMultiheadAttentionModule,
    LRPTransformerEncoder,
    LRPTransformerEncoderLayer,
)

# custom pooling classes are matched by name to avoid importing the model modules
_POOLING_BY_NAME = {
    "AttentionPooling": LRPAttentionPooling,
    "AttentionLayer": LRPAttentionLayer,
}


def prepare_model_for_lrp(model: nn.Module, epsilon: float = 1e-6) -> nn.Module:
    """In-place: replace fused/attention blocks with their LRP counterparts and
    wrap remaining linear leaves with LXT rules.  Returns ``model``.

    Call on a copy (see :class:`ModelLRP`) — it mutates the module tree.
    """
    from lxt.explicit.rules import EpsilonRule, IdentityRule

    for name, child in list(model.named_children()):
        cname = type(child).__name__
        if isinstance(child, nn.LSTM):
            setattr(model, name, LRPLSTM.from_torch(child))
        elif isinstance(child, nn.GRU):
            setattr(model, name, LRPGRU.from_torch(child))
        elif isinstance(child, nn.TransformerEncoder):
            setattr(model, name, LRPTransformerEncoder.from_torch(child))
        elif isinstance(child, nn.TransformerEncoderLayer):
            setattr(model, name, LRPTransformerEncoderLayer.from_torch(child))
        elif isinstance(child, nn.MultiheadAttention):
            setattr(model, name, LRPMultiheadAttentionModule.from_torch(child))
        elif cname in _POOLING_BY_NAME:
            setattr(model, name, _POOLING_BY_NAME[cname](child, epsilon))
        elif isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            setattr(model, name, EpsilonRule(child, epsilon))
        elif isinstance(child, nn.LayerNorm):
            setattr(model, name, IdentityRule(child))
        elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # in eval mode BatchNorm is an affine-linear op → ε-LRP
            setattr(model, name, EpsilonRule(child, epsilon))
        else:
            # not a leaf/known block: recurse into it
            prepare_model_for_lrp(child, epsilon)
    return model


class ModelLRP(nn.Module):
    """LRP attribution for a whole PhysioEx model (recurrent / transformer /
    attention architectures).

    Args:
        model: trained model, ``(B, L, C, T[, F]) -> (B, L, n_classes)`` (or a
            dict output — set ``output_key``).
        in_index / out_index: sequence epoch and class to explain.
        output_key: for models returning a dict (e.g. CoReSleep → ``"combined"``).
        epsilon: ε for the ε-LRP rules.

    ``forward(x)`` returns relevance shaped like ``x``.  The trained model is
    deep-copied and left untouched.
    """

    def __init__(
        self,
        model: nn.Module,
        in_index: int = 0,
        out_index: int = 0,
        output_key: str | None = None,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.in_index = in_index
        self.out_index = out_index
        self.output_key = output_key
        prepared = copy.deepcopy(model).eval()
        self.model = prepare_model_for_lrp(prepared, epsilon)
        # LRP saves tensors for the custom backward, so in-place ops (e.g.
        # ReLU(inplace=True)) would corrupt them — disable them on the copy.
        for m in self.model.modules():
            if getattr(m, "inplace", False):
                m.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        out = self.model(x)
        if self.output_key is not None:
            out = out[self.output_key]
        seed = torch.zeros_like(out)
        seed[:, self.in_index, self.out_index] = out[
            :, self.in_index, self.out_index
        ].detach()
        out.backward(seed)
        return x.grad
