"""LRP composites tuned for PhysioEx models.

A *composite* maps each module type to an LRP rule.  Following the
best-practice recipe (Montavon et al. 2019, *Layer-Wise Relevance Propagation:
An Overview*):

* dense / classifier ``Linear``     -> **Epsilon** (ε-rule), reduces noise
* convolutional feature layers      -> **Gamma** (γ-rule), stabilises evidence
* the **first** (input) layer       -> **WSquare** (w²-rule)

The w²-rule is the correct input-layer rule here because PhysioEx inputs are
**unbounded real signals** (z-scored EEG), *not* bounded pixels — so the
``ZBox`` (pixel-domain) rule does not apply.  Use ``first_rule="zbox"`` only
for models fed bounded inputs (e.g. BENDR's minmax-scaled signals), passing
``zbox_low``/``zbox_high``.

``zennit`` is an optional dependency (install ``physioex[explain]``); it is
imported lazily.
"""

from __future__ import annotations

from typing import List, Optional


def physioex_composite(
    first_rule: str = "w2",
    epsilon: float = 1e-6,
    gamma: float = 0.25,
    zbox_low: float = -3.0,
    zbox_high: float = 3.0,
    canonizers: Optional[List] = None,
):
    """Build the default PhysioEx LRP composite (ε dense / γ conv / w² first).

    Args:
        first_rule: input-layer rule — ``"w2"`` (default, unbounded signals),
            ``"zbox"`` (bounded inputs; needs ``zbox_low``/``zbox_high``) or
            ``"flat"`` (uniform baseline).
        epsilon: stabiliser for the ε-rule on dense layers.
        gamma: positive-weight amplification for the γ-rule on conv layers.
        zbox_low, zbox_high: bounds for the ``zbox`` first-layer rule.
        canonizers: list of Zennit canonizers to apply (e.g. from
            :func:`physioex.explain.lrp.canonizers.default_canonizers`).

    Returns:
        A ``zennit.composites.SpecialFirstLayerMapComposite``.
    """
    from torch.nn import Linear
    from zennit.composites import SpecialFirstLayerMapComposite
    from zennit.rules import Epsilon, Flat, Gamma, Norm, Pass, WSquare, ZBox
    from zennit.types import Activation, AvgPool, BatchNorm, Convolution

    # Rule per module type; the first matching entry wins.
    layer_map = [
        (Activation, Pass()),          # ReLU/GELU: relevance passes through
        (AvgPool, Norm()),             # average pooling: normalised redistribution
        (BatchNorm, Pass()),           # merged by the canonizer; guard otherwise
        (Convolution, Gamma(gamma=gamma)),
        (Linear, Epsilon(epsilon=epsilon)),
    ]

    if first_rule == "w2":
        first = WSquare()
    elif first_rule == "zbox":
        first = ZBox(low=zbox_low, high=zbox_high)
    elif first_rule == "flat":
        first = Flat()
    else:
        raise ValueError(
            f"first_rule must be one of 'w2', 'zbox', 'flat'; got {first_rule!r}"
        )

    # The first linear/convolution layer of the network gets the input rule.
    first_map = [
        (Convolution, first),
        (Linear, first),
    ]

    return SpecialFirstLayerMapComposite(
        layer_map=layer_map,
        first_map=first_map,
        canonizers=canonizers or [],
    )


def epsilon_composite(epsilon: float = 1e-6, canonizers: Optional[List] = None):
    """A pure ε-LRP composite (ε on every linear/conv layer).

    Unlike :func:`physioex_composite`, this uses no special first-layer rule,
    so it approximately **conserves relevance** (``Σ R ≈ f(x)``) — useful as
    the reference for the conservation sanity check in the test suite.
    """
    from torch.nn import Linear
    from zennit.composites import LayerMapComposite
    from zennit.rules import Epsilon, Norm, Pass
    from zennit.types import Activation, AvgPool, BatchNorm, Convolution

    layer_map = [
        (Activation, Pass()),
        (AvgPool, Norm()),
        (BatchNorm, Pass()),
        (Convolution, Epsilon(epsilon=epsilon)),
        (Linear, Epsilon(epsilon=epsilon)),
    ]
    return LayerMapComposite(layer_map=layer_map, canonizers=canonizers or [])
