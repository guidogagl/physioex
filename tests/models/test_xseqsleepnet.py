"""Contract tests for XSeqSleepNet (pluggable encoders) and the paper-compliant
L-SeqSleepNet. Random init, tiny shapes, CPU. xLSTM variants are skipped when
the optional ``xlstm`` extra is not installed.
"""
import pytest
import torch

import physioex.models.xseqsleepnet  # noqa: F401  (sets the CUDA_HOME placeholder before `import xlstm`)

T, F = 29, 129
XLSTM_KW = dict(num_blocks=2, num_heads=2, context_length=64)


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


# ── XSeqSleepNet: parity with SeqSleepNet ────────────────────────────────


def test_xseqsleepnet_gru_matches_seqsleepnet_exactly():
    from physioex.models.seqsleepnet import SeqSleepNet
    from physioex.models.xseqsleepnet import XSeqSleepNet

    torch.manual_seed(0)
    ref = SeqSleepNet(n_classes=5, in_chan=2, D=8, seqnhidden1=8, seqnlayer1=2,
                      attentionsize=8, seqnhidden2=8, seqnlayer2=2).eval()
    new = XSeqSleepNet.from_seqsleepnet(ref).eval()
    assert _n_params(new) == _n_params(ref)

    x = torch.randn(2, 5, 2, T, F)
    with torch.no_grad():
        assert torch.allclose(new(x), ref(x), atol=1e-6)
        assert torch.allclose(new.encode(x), ref.encode(x), atol=1e-6)


@pytest.mark.parametrize("variant", ["gru", "gru_uni", "gru_wrapped"])
def test_xseqsleepnet_recurrent_variants_shape(variant):
    from physioex.models.xseqsleepnet import XSeqSleepNet

    m = XSeqSleepNet(n_classes=5, in_chan=1, D=8,
                     epoch_kwargs=dict(hidden=8, num_layers=1, attention_size=8),
                     sequence_encoder=variant, seq_kwargs=dict(hidden=8))
    out = m(torch.randn(2, 6, 1, T, F))
    assert out.shape == (2, 6, 5)
    assert m.encode(torch.randn(2, 6, 1, T, F)).shape[:2] == (2, 6)
    assert m.is_causal == (variant == "gru_uni")


def test_gru_uni_is_causal():
    from physioex.models.xseqsleepnet import XSeqSleepNet

    torch.manual_seed(0)
    m = XSeqSleepNet(in_chan=1, D=8, epoch_kwargs=dict(hidden=8, num_layers=1, attention_size=8),
                     sequence_encoder="gru_uni", seq_kwargs=dict(hidden=8)).eval()
    x = torch.randn(1, 8, 1, T, F)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])  # perturb the future only
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5)
    assert not torch.allclose(a[:, 5:], b[:, 5:])


def test_gru_matched_has_at_least_target_params():
    from physioex.models.xseqsleepnet import WrappedGRUSequenceEncoder

    enc = WrappedGRUSequenceEncoder.matched_to(50_000, d_model=32, num_blocks=2)
    assert _n_params(enc) >= 50_000
    smaller = WrappedGRUSequenceEncoder(32, enc.hidden - 1, num_blocks=2)
    assert _n_params(smaller) < 50_000


# ── xLSTM variants (optional extra) ──────────────────────────────────────


@pytest.mark.parametrize("variant", ["xlstm_bi", "xlstm_causal", "xlstm_alt"])
def test_xseqsleepnet_xlstm_variants_shape(variant):
    pytest.importorskip("xlstm")
    from physioex.models.xseqsleepnet import XSeqSleepNet

    m = XSeqSleepNet(n_classes=5, in_chan=1, D=8,
                     epoch_kwargs=dict(hidden=8, num_layers=1, attention_size=8),
                     sequence_encoder=variant, seq_kwargs=XLSTM_KW)
    out = m(torch.randn(2, 6, 1, T, F))
    assert out.shape == (2, 6, 5)
    assert m.is_causal == (variant == "xlstm_causal")


def test_xlstm_causal_is_causal_and_runs_longer_than_train_length():
    pytest.importorskip("xlstm")
    from physioex.models.xseqsleepnet import XSeqSleepNet

    torch.manual_seed(0)
    m = XSeqSleepNet(in_chan=1, D=8, epoch_kwargs=dict(hidden=8, num_layers=1, attention_size=8),
                     sequence_encoder="xlstm_causal", seq_kwargs=XLSTM_KW).eval()
    x = torch.randn(1, 12, 1, T, F)
    x2 = x.clone()
    x2[:, 7:] = torch.randn_like(x2[:, 7:])
    with torch.no_grad():
        a, b = m(x), m(x2)
        long = m(torch.randn(1, 40, 1, T, F))  # "whole night" longer than any window
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-4)
    assert not torch.allclose(a[:, 7:], b[:, 7:])
    assert long.shape == (1, 40, 5)


def test_xlstm_context_length_is_enforced():
    pytest.importorskip("xlstm")
    from physioex.models.xseqsleepnet import XLSTMSequenceEncoder

    enc = XLSTMSequenceEncoder(16, num_blocks=1, num_heads=2, direction="causal", context_length=8)
    with pytest.raises(ValueError, match="context_length"):
        enc(torch.randn(1, 9, 16))


def test_xlstm_epoch_encoder_shape():
    pytest.importorskip("xlstm")
    from physioex.models.xseqsleepnet import XSeqSleepNet

    m = XSeqSleepNet(in_chan=1, D=8, epoch_encoder="xlstm",
                     epoch_kwargs=dict(d_model=16, attention_size=8, num_blocks=1, num_heads=2),
                     sequence_encoder="gru", seq_kwargs=dict(hidden=8, num_layers=1))
    assert m(torch.randn(2, 3, 1, T, F)).shape == (2, 3, 5)


def test_gru_matched_targets_xlstm_bi_params():
    pytest.importorskip("xlstm")
    from physioex.models.xseqsleepnet import XLSTMSequenceEncoder, make_sequence_encoder

    d = 32
    target = _n_params(XLSTMSequenceEncoder(d, direction="bi", **XLSTM_KW))
    enc = make_sequence_encoder("gru_matched", d, **XLSTM_KW)
    assert _n_params(enc) >= target


# ── L-SeqSleepNet (paper-compliant) ──────────────────────────────────────


@pytest.mark.parametrize("recurrent_bn", [True, False])
def test_lseqsleepnet_paper_shape_and_gradients(recurrent_bn):
    from physioex.models.lseqsleepnet import LSeqSleepNet

    m = LSeqSleepNet(n_classes=5, in_chan=1, D=8, epoch_hidden=8, epoch_attention=8,
                     B=2, K=3, seq_hidden_ss=8, seq_hidden_ms=8, d_clf=16, dropout=0.1,
                     recurrent_bn=recurrent_bn)
    x = torch.randn(2, 6, 1, T, F)
    out = m(x)
    assert out.shape == (2, 6, 5)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    assert m.encode(x).shape == (2, 6, 16)


def test_lseqsleepnet_rejects_longer_than_BK_and_pads_shorter():
    from physioex.models.lseqsleepnet import LSeqSleepNet

    m = LSeqSleepNet(in_chan=1, D=8, epoch_hidden=8, epoch_attention=8, B=2, K=3,
                     seq_hidden_ss=8, seq_hidden_ms=8, d_clf=16, recurrent_bn=False).eval()
    with pytest.raises(ValueError, match="exactly B\\*K"):
        m(torch.randn(1, 7, 1, T, F))
    with pytest.warns(UserWarning, match="padding"):
        assert m(torch.randn(1, 4, 1, T, F)).shape == (1, 4, 5)


def test_lseqsleepnet_hidden_must_match_d_model():
    from physioex.models.lseqsleepnet import LSeqSleepNet

    with pytest.raises(ValueError, match="2\\*hidden_ss"):
        LSeqSleepNet(epoch_hidden=8, seq_hidden_ss=4, seq_hidden_ms=8, recurrent_bn=False)


def test_residual_ln_block_follows_eq9():
    """o_bar = o~ + LN(W o~ + b): residual from the block input, LN only on the fc."""
    from physioex.models.lseqsleepnet import _ResidualLNBlock

    torch.manual_seed(0)
    blk = _ResidualLNBlock(8, dropout=0.0).eval()
    o = torch.randn(3, 4, 8)
    expected = o + blk.ln(blk.fc(o))
    assert torch.allclose(blk(o), expected, atol=1e-6)
    # post-LN (the draft's formulation) would differ
    assert not torch.allclose(blk(o), blk.ln(blk.fc(o) + o))


def test_bnlstm_cell_shapes_and_eval_mode():
    from physioex.models.lseqsleepnet import BNLSTMCell, BiLSTM

    cell = BNLSTMCell(6, 5)
    out, (h, c) = cell(torch.randn(4, 7, 6))
    assert out.shape == (4, 7, 5) and h.shape == (4, 5) and c.shape == (4, 5)
    # forget-gate bias initialised to 1, gammas to 0.1
    assert torch.allclose(cell.bias[5:10], torch.ones(5))
    assert torch.allclose(cell.bn_hh.weight, torch.full((20,), 0.1))

    bi = BiLSTM(6, 5, recurrent_bn=True).eval()
    with torch.no_grad():
        y = bi(torch.randn(1, 7, 6))  # batch of 1 must work in eval (running stats)
    assert y.shape == (1, 7, 10)


def test_lseqsleepnet_draft_still_loads_and_is_deprecated():
    from physioex.models.lseqsleepnet import LSeqSleepNet, LSeqSleepNetDraft

    with pytest.warns(DeprecationWarning):
        draft = LSeqSleepNetDraft(in_chan=1, D=8, epoch_hidden=8, epoch_attention=8, B=2, K=2,
                                  seq_hidden_ss=8, seq_hidden_ms=8, d_clf=16)
    assert draft(torch.randn(1, 4, 1, T, F)).shape == (1, 4, 5)

    new = LSeqSleepNet(in_chan=1, D=8, epoch_hidden=8, epoch_attention=8, B=2, K=2,
                       seq_hidden_ss=8, seq_hidden_ms=8, d_clf=16, recurrent_bn=False)
    with pytest.raises(RuntimeError, match="LSeqSleepNetDraft"):
        new.load_state_dict(draft.state_dict())
