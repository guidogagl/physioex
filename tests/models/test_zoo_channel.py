"""Model zoo of the multimodal framework (xmai-lab/sleep-mechinterp#39): block API, lockstep fusion, round trip and
the exactness of the parallel forms (S4D kernel, mLSTM) against their recurrences."""

import json

import pytest
import torch

from physioex.models.channelnet_mm import MultiChannelNet
from physioex.models.seqsleepnet_clean import ChannelNet
from physioex.models.zoo_blocks import (RAW_K, RAW_ZMAX, MambaMixer, S4DKernel, mlstm_parallel, mlstm_recurrent,
                                        raw_scale)
from physioex.models.zoo_channel import EPOCH_ENCODERS, SEQ_ENCODERS, ZooChannel

B, L = 2, 3
PAIRS = [(e, "gru") for e in EPOCH_ENCODERS] + [("lstm", s) for s in SEQ_ENCODERS if s != "gru"] + [
    ("tf", "tf"), ("s4d", "s4d"), ("mamba", "mamba"), ("mlstm", "mlstm"), ("rawcnn", "tcn")]


def _x(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, 3000, generator=g), torch.tensor([0.0, 3.0])


@pytest.mark.parametrize("epoch,seq", PAIRS)
def test_shapes_and_layernorm_outputs(epoch, seq):
    torch.manual_seed(0)
    m = ZooChannel("EMG", epoch, seq).eval()
    x, off = _x()
    logits, out = m(x, off)
    assert logits.shape == (B, L, 5) and torch.isfinite(logits).all()
    assert m.dim == 128 and m.n_epoch_blocks == 4 and m.n_seq_blocks == 4
    for i in range(1, 5):
        h, q = out[f"L{i}"], out[f"G{i}"]
        assert h.shape == (B * L, 29, 128) and q.shape == (B, L, 128)
        for t in (h, q):                                   # LayerNorm-ended: zero mean, unit var per token
            assert t.mean(-1).abs().max() < 1e-4
    assert out["e"].shape == (B, L, 128) and out["alpha"].shape == (B, L, 29)
    logits.sum().backward()


def test_z0_has_channelnet_structure():
    z, c = ZooChannel("EEG", "lstm", "gru"), ChannelNet("EEG")
    assert sum(p.numel() for p in z.parameters()) == sum(p.numel() for p in c.parameters())
    # copy ChannelNet weights into Z0 and check the forward is the same
    zs = z.state_dict()
    new = {}
    for k in zs:
        if k.startswith("epoch_blocks."):
            i, rest = k.split(".")[1], ".".join(k.split(".")[2:])
            new[k] = c.state_dict()[f"epoch_rnn.{'rnns' if rest.startswith('rnn') else 'norms'}.{i}." +
                                    rest.split(".", 1)[1]]
        elif k.startswith("seq_blocks."):
            i, rest = k.split(".")[1], ".".join(k.split(".")[2:])
            new[k] = c.state_dict()[f"seq_rnn.{'rnns' if rest.startswith('rnn') else 'norms'}.{i}." +
                                    rest.split(".", 1)[1]]
        else:
            new[k] = c.state_dict()[k]
    z.load_state_dict(new)
    x, off = _x()
    torch.testing.assert_close(z.eval()(x, off)[0], c.eval()(x, off)[0])


@pytest.mark.parametrize("epoch,seq", [("lstm", "gru"), ("rawcnn", "tcn"), ("mamba", "mlstm"), ("patch", "slstm"),
                                       ("s4d", "tf")])
def test_multichannel_gates_zero_equal_branches(epoch, seq):
    torch.manual_seed(0)
    br = {m: ZooChannel(m, epoch, seq).eval() for m in ("EEG", "EOG", "EMG")}
    mm = MultiChannelNet(br, ("F", "E", "Q")).eval()
    x = {m: _x(i)[0] for i, m in enumerate(br)}
    off = {m: torch.zeros(B) for m in br}
    with torch.no_grad():
        _, aux, _ = mm(x, off)
        for m, net in br.items():
            ref, _ = net(x[m], off[m])
            torch.testing.assert_close(aux[m], ref, atol=1e-5, rtol=1e-5)


def test_run_json_round_trip(tmp_path):
    from physioex.train.bin.train_channelnet_mm import load_branches
    for m in ("EEG", "EOG", "EMG"):
        torch.manual_seed(1)
        net = ZooChannel(m, "mamba", "tcn")
        d = tmp_path / m
        d.mkdir()
        torch.save(net.state_dict(), d / "best.pt")
        (d / "run.json").write_text(json.dumps({"config": {"arch": "zoo", "epoch_enc": "mamba", "seq_enc": "tcn",
                                                           "resolved": {"dropout": 0.0}},
                                                "final": {"test": {"kappa": 0.5}}}))
    br, kappa = load_branches(tmp_path, ["EEG", "EOG", "EMG"])
    assert isinstance(br["EMG"], ZooChannel) and br["EMG"].epoch_kind == "mamba" and kappa["EEG"] == 0.5


def test_s4d_kernel_equals_recurrence():
    torch.manual_seed(0)
    k = S4DKernel(4, 8)
    T = 12
    K = k(T)                                               # (d, T)
    dtA, C = k.discrete()
    # recurrence h_t = exp(dtA) h_{t-1} + u_t (B_bar folded into C), y_t = 2 Re(C h_t): impulse response = K
    h = torch.zeros(4, 4, dtype=torch.cfloat)
    ys = []
    for t in range(T):
        h = dtA.exp() * h + (1.0 if t == 0 else 0.0)
        ys.append(2 * (C * h).sum(-1).real)
    torch.testing.assert_close(torch.stack(ys, -1), K, atol=1e-5, rtol=1e-4)


def test_mlstm_parallel_equals_recurrent():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 9, 8) for _ in range(3))
    ip, fp = torch.randn(2, 3, 9, 1), torch.randn(2, 3, 9, 1) + 2
    torch.testing.assert_close(mlstm_parallel(q, k, v, ip, fp), mlstm_recurrent(q, k, v, ip, fp), atol=1e-5,
                               rtol=1e-4)


def test_mamba_scan_is_causal():
    torch.manual_seed(0)
    mx = MambaMixer(16).eval()
    x = torch.randn(1, 10, 16)
    y = mx(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 4, 16)
    torch.testing.assert_close(mx(x2)[:, :6], y[:, :6])


def test_raw_scale_fixed_monotone_bounded():
    x = torch.linspace(-100, 100, 2001).view(1, 1, -1)
    y = raw_scale(x)
    assert (y.diff(dim=-1) >= 0).all() and y.abs().max() <= 1
    torch.testing.assert_close(raw_scale(torch.full((1, 1, 3), RAW_ZMAX)), torch.ones(1, 1, 3))
    # no dataset statistics: the same input gives the same output whatever the batch contains
    a = torch.randn(1, 1, 3000)
    torch.testing.assert_close(raw_scale(torch.cat([a, 50 * torch.randn(1, 1, 3000)]))[:1], raw_scale(a))
    # EMG gain: offset_db = 20 dB is a factor 10 on the input
    torch.testing.assert_close(raw_scale(a, torch.tensor([20.0])), raw_scale(10 * a))
    assert RAW_K > 0 and RAW_ZMAX > RAW_K


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuDNN RNN backward check needs a GPU")
def test_frozen_rnn_branches_backward_on_cuda():
    br = {m: ZooChannel(m, "lstm", "gru").cuda() for m in ("EEG", "EOG", "EMG")}
    mm = MultiChannelNet(br, ("F",)).cuda()
    mm.frozen_branches = True
    mm.train()
    for p in mm.branches.parameters():
        p.requires_grad_(False)
    x = {m: _x(i)[0].cuda() for i, m in enumerate(br)}
    logits, _, _ = mm(x)
    logits.sum().backward()
    assert all(p.grad is not None for n, p in mm.named_parameters() if n.startswith("conn."))


def test_band_limited_raw_input():
    m = ZooChannel("EEG", "rawcnnbp", "gru").eval()
    t = torch.arange(3000) / 100.0
    x = (torch.sin(2 * torch.pi * 10 * t) + 3 + torch.sin(2 * torch.pi * 40 * t) + torch.sin(2 * torch.pi * 0.1 * t))
    xb = torch.fft.irfft(torch.fft.rfft(x.view(1, 1, -1), dim=-1) * m.band_mask, n=3000, dim=-1).flatten()
    torch.testing.assert_close(xb, torch.sin(2 * torch.pi * 10 * t), atol=1e-3, rtol=0)   # DC, 0.1 Hz, 40 Hz removed (fp32 FFT)
