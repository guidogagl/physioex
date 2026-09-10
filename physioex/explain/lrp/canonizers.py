"""Canonizers for LRP over PhysioEx architectures.

Layer-wise Relevance Propagation (LRP) is *not* implementation-invariant: a
``BatchNorm -> Linear`` block is attributed differently from ``Linear ->
BatchNorm``.  Zennit **canonizers** temporarily rewrite the model into a
canonical form (merging BatchNorm into the adjacent linear / convolution) so
relevance is propagated consistently.

The canonizers are applied automatically by the composite while it is
registered; they are undone on exit.  Do **not** save ``model.state_dict()``
while a canonizer is active — it would persist the merged (modified) weights.

``zennit`` is an optional dependency (install ``physioex[explain]``); it is
imported lazily so ``import physioex.explain`` stays lightweight.
"""

from __future__ import annotations

from typing import List


def default_canonizers(model=None) -> List:
    """Return the default canonizer list for a PhysioEx model.

    Currently merges sequential ``BatchNorm`` layers into their neighbouring
    linear / convolution modules — needed by e.g. ``tinysleepnet``,
    ``sleepfm`` and ``tfc``.  Models without BatchNorm are unaffected (the
    canonizer simply finds nothing to merge).

    Args:
        model: unused for now; accepted so callers can pass the model and we
            can later dispatch model-specific canonizers (e.g. GroupNorm in
            ``neurolm`` or the ``protosleepnet`` prototype head).
    """
    from zennit.canonizers import SequentialMergeBatchNorm

    return [SequentialMergeBatchNorm()]
