"""ModelLRP on the real PhysioEx architectures (tiny random-init configs).

For each model: the prepared model reproduces the original forward, every
parametric leaf is covered by an LRP rule (no silent gradient fall-through),
and the relevance is finite with the input's shape.  Covers every family and
the tricky paths: bimodal CoReSleep (cross-attention + in-place residual write),
L-SeqSleepNet (fold/unfold BiLSTMs with residual + LayerNorm), ProtoSleepNet with
the channel mixer.  A model is skipped only if its module cannot be imported
(missing optional dependency); any other failure is a real failure.
"""

import copy
import importlib

import pytest
import torch

pytest.importorskip("zennit")

from physioex.explain.lrp import ModelLRP, prepare_model_for_lrp  # noqa: E402


def _cls(path):
    mod, name = path.split(":")
    try:
        module = importlib.import_module(mod)
    except ImportError as exc:  # missing optional dependency of the model module
        pytest.skip(f"{path} unavailable: {exc}")
    return getattr(module, name)


CASES = {
    "sleeptransformer": (
        "physioex.models.sleeptransformer:SleepTransformer",
        dict(
            n_classes=5,
            in_chan=1,
            d_model=32,
            n_heads=4,
            n_epoch_layers=1,
            n_seq_layers=1,
            d_ff=64,
            d_clf=64,
            attention_size=32,
        ),
        (2, 3, 1, 8, 32),
        None,
    ),
    "seqsleepnet": (
        "physioex.models.seqsleepnet:SeqSleepNet",
        dict(
            n_classes=5,
            in_chan=3,
            F=129,
            D=8,
            seqnhidden1=8,
            seqnlayer1=1,
            attentionsize=8,
            seqnhidden2=8,
            seqnlayer2=1,
        ),
        (2, 3, 3, 10, 129),
        None,
    ),
    "lseqsleepnet": (
        "physioex.models.lseqsleepnet:LSeqSleepNet",
        dict(
            n_classes=5,
            in_chan=1,
            F=129,
            D=8,
            epoch_hidden=8,
            epoch_attention=8,
            B=2,
            K=2,
            seq_hidden_ss=8,
            seq_hidden_ms=8,
            d_clf=16,
            dropout=0.0,
            # cuDNN nn.LSTM path: the LRP adapters key on nn.LSTM; the
            # paper-default BN-LSTM cell has no LRP surrogate yet.
            recurrent_bn=False,
        ),
        (2, 4, 1, 10, 129),
        None,
    ),
    "tinysleepnet": (
        "physioex.models.tinysleepnet:TinySleepNet",
        dict(n_classes=5, in_chan=3, n_rnn_units=16, n_rnn_layers=1),
        (2, 3, 3, 3000),
        None,
    ),
    "coresleep_unimodal": (
        "physioex.models.coresleep:CoReSleep",
        dict(
            n_classes=5,
            in_chan=1,
            F=129,
            d_model=32,
            n_heads=4,
            n_inner_layers=1,
            n_outer_layers=1,
            d_ff=64,
            dropout=0.0,
        ),
        (2, 3, 1, 10, 129),
        "combined",
    ),
    "coresleep_bimodal": (
        "physioex.models.coresleep:CoReSleep",
        dict(
            n_classes=5,
            in_chan=2,
            F=129,
            d_model=32,
            n_heads=4,
            n_inner_layers=1,
            n_outer_layers=1,
            d_ff=64,
            dropout=0.0,
        ),
        (2, 3, 2, 10, 129),
        "combined",
    ),
}

PROTO = {
    "transformer": (
        "from_sleep_transformer",
        dict(
            n_channels=1,
            n_classes=5,
            d_model=32,
            n_heads=4,
            n_epoch_layers=1,
            n_seq_layers=1,
            d_ff=64,
            d_clf=64,
            attention_size=32,
        ),
        (2, 3, 1, 8, 32),
    ),
    "seqsleepnet": (
        "from_seq_sleep_net",
        dict(
            n_channels=1,
            n_classes=5,
            F=129,
            D=8,
            seqnhidden1=8,
            seqnlayer1=1,
            attentionsize=8,
            seqnhidden2=8,
            seqnlayer2=1,
        ),
        (2, 3, 1, 10, 129),
    ),
    "seqsleepnet_channel_mixer": (
        "from_seq_sleep_net",
        dict(
            n_channels=3,
            use_channel_mixer=True,
            n_classes=5,
            F=129,
            D=8,
            seqnhidden1=8,
            seqnlayer1=1,
            attentionsize=8,
            seqnhidden2=8,
            seqnlayer2=1,
        ),
        (2, 3, 3, 10, 129),
    ),
}


def _check(model, x, key):
    with torch.no_grad():
        ref = model(x)
    prep = prepare_model_for_lrp(copy.deepcopy(model))
    with torch.no_grad():
        out = prep(x)
    if key is not None:
        ref, out = ref[key], out[key]
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
    explainer = ModelLRP(model, out_index=2, output_key=key)
    assert explainer.uncovered == [], explainer.uncovered
    rel, rep = explainer(x, return_report=True)
    assert rel.shape == x.shape and torch.isfinite(rel).all()
    assert torch.isfinite(rep.ratio).all()


@pytest.mark.parametrize("name", list(CASES))
def test_real_model(name):
    path, kwargs, shape, key = CASES[name]
    torch.manual_seed(0)
    _check(_cls(path)(**kwargs).eval(), torch.randn(*shape), key)


@pytest.mark.parametrize("which", list(PROTO))
def test_protosleepnet(which):
    factory, kwargs, shape = PROTO[which]
    P = _cls("physioex.models.protosleepnet:ProtoSleepNet")
    torch.manual_seed(0)
    _check(getattr(P, factory)(**kwargs).eval(), torch.randn(*shape), None)
