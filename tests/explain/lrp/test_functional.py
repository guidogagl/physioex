"""Unit tests for the shared LRP primitives (physioex.explain.lrp._functional)."""

import pytest
import torch


from physioex.explain.lrp._functional import (  # noqa: E402
    add_eps,
    linear_eps,
    mul_signal_take,
    st_identity,
    stabilize,
    target_seed,
)


class TestStabilize:
    def test_signed_and_zero_positive(self):
        z = torch.tensor([2.0, -2.0, 0.0])
        assert torch.allclose(stabilize(z, 0.5), torch.tensor([2.5, -2.5, 0.5]))

    def test_keeps_half_precision_dtype(self):
        z = torch.randn(4, dtype=torch.half)
        assert stabilize(z, 1e-3).dtype == torch.half


class TestLinearEps:
    def test_matches_arras_epsilon_rule_and_conserves(self):
        torch.manual_seed(0)
        W = torch.randn(3, 4)
        x = torch.randn(2, 4, requires_grad=True)
        y = linear_eps(x, W, None, 1e-6)
        R = torch.randn_like(y)
        y.backward(R)
        z = (x.detach() @ W.T)
        expected = x.detach() * ((R / (z + 1e-6 * torch.sign(z))) @ W)
        assert torch.allclose(x.grad, expected, atol=1e-6)
        assert torch.allclose(x.grad.sum(), R.sum(), rtol=1e-4)

    def test_forward_equals_linear(self):
        W, b, x = torch.randn(3, 4), torch.randn(3), torch.randn(5, 4)
        assert torch.allclose(linear_eps(x, W, b), torch.nn.functional.linear(x, W, b))


class TestAddEps:
    def test_proportional_split(self):
        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([5.0, 7.0], requires_grad=True)
        y = add_eps(a, b, 1e-6)
        y.backward(y.detach())
        assert torch.allclose(a.grad, a.detach(), rtol=1e-4)
        assert torch.allclose(b.grad, b.detach(), rtol=1e-4)

    def test_near_cancelling_negative_sum_is_bounded(self):
        # LXT's sign-blind add2 gives ±1e8 here; the signed stabiliser stays ≤ 1/ε
        eps = 1e-6
        a = torch.tensor([1.0], requires_grad=True)
        b = torch.tensor([-1.0 + 1e-9], requires_grad=True)
        add_eps(a, b, eps).backward(torch.ones(1))
        assert a.grad.abs().item() <= 1.0 / eps + 1e-3

    def test_broadcast_operand(self):
        a = torch.randn(2, 3, 4, requires_grad=True)
        b = torch.randn(1, 3, 4, requires_grad=True)
        y = add_eps(a, b)
        y.backward(torch.ones_like(y))
        assert b.grad.shape == b.shape


class TestGateAndIdentity:
    def test_signal_take_routes_all_to_source(self):
        g = torch.randn(3, requires_grad=True)
        s = torch.randn(3, requires_grad=True)
        mul_signal_take(g, s).backward(torch.ones(3))
        assert torch.all(g.grad == 0) and torch.all(s.grad == 1)

    def test_st_identity_is_bit_exact(self):
        c = torch.tensor([30.0, 100.0, 1e8])
        assert torch.equal(st_identity(c, torch.tanh(c)), torch.tanh(c))

    def test_st_identity_backward(self):
        c = torch.randn(4, requires_grad=True)
        st_identity(c, torch.tanh(c)).backward(torch.ones(4))
        assert torch.all(c.grad == 1)


class TestTargetSeed:
    def test_rank3_and_rank2(self):
        out3 = torch.randn(2, 3, 5)
        seed, tgt = target_seed(out3, 1, 4)
        assert torch.equal(tgt, out3[:, 1, 4]) and seed.abs().sum() == tgt.abs().sum()
        out2 = torch.randn(2, 5)
        seed2, tgt2 = target_seed(out2, 0, 3)
        assert torch.equal(tgt2, out2[:, 3]) and torch.equal(seed2[:, 3], tgt2)

    def test_bad_rank_raises(self):
        with pytest.raises(ValueError):
            target_seed(torch.randn(5), 0, 0)
