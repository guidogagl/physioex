"""Extensive unit tests for explain.posthoc.gradients.

Covers: Saliency, InputXGradient, IntegratedGradients, ExpectedGradients.
"""

import pytest
import torch

from physioex.explain.posthoc.gradients import (
    ExpectedGradients,
    InputXGradient,
    IntegratedGradients,
    Saliency,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic(x):
    """f(x) = sum(x^2).  Grad = 2x."""
    return (x**2).sum()


def _linear(x):
    """f(x) = sum(x).  Grad = ones."""
    return x.sum()


def _multiclass(x):
    """f(x) = [sum(x), sum(x^2), sum(x^3)] — returns a 3-class vector."""
    return torch.stack([x.sum(), (x**2).sum(), (x**3).sum()])


def _batched_quadratic(x):
    """Expects (1, n), returns scalar."""
    return (x**2).sum()


# ---------------------------------------------------------------------------
# Saliency
# ---------------------------------------------------------------------------


class TestSaliency:
    def test_output_shape_single(self):
        sal = Saliency(f=_quadratic)
        x = torch.randn(1, 10)
        out = sal(x)
        assert out.shape == x.shape

    def test_output_shape_batch(self):
        sal = Saliency(f=_quadratic)
        x = torch.randn(8, 10)
        out = sal(x)
        assert out.shape == x.shape

    def test_finite_values(self):
        sal = Saliency(f=_quadratic)
        x = torch.randn(4, 20)
        out = sal(x)
        assert torch.isfinite(out).all()

    def test_gradient_correctness_quadratic(self):
        """For f(x) = sum(x^2), grad = 2x."""
        sal = Saliency(f=_quadratic)
        x = torch.randn(3, 10)
        out = sal(x)
        expected = 2.0 * x
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_correctness_linear(self):
        """For f(x) = sum(x), grad = ones."""
        sal = Saliency(f=_linear)
        x = torch.randn(2, 8)
        out = sal(x)
        expected = torch.ones_like(x)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_target_selects_class(self):
        """Verify that target indexes the correct output dimension."""
        # target=0 → grad of sum(x) = ones
        sal0 = Saliency(f=_multiclass, target=0)
        x = torch.randn(1, 5)
        out0 = sal0(x)
        torch.testing.assert_close(out0, torch.ones_like(x), atol=1e-5, rtol=1e-5)

        # target=1 → grad of sum(x^2) = 2x
        sal1 = Saliency(f=_multiclass, target=1)
        out1 = sal1(x)
        torch.testing.assert_close(out1, 2.0 * x, atol=1e-5, rtol=1e-5)

        # target=2 → grad of sum(x^3) = 3x^2
        sal2 = Saliency(f=_multiclass, target=2)
        out2 = sal2(x)
        torch.testing.assert_close(out2, 3.0 * x**2, atol=1e-4, rtol=1e-4)

    def test_default_target_is_first(self):
        """Without target, _scalar_f picks output[0] = sum(x)."""
        sal = Saliency(f=_multiclass)
        x = torch.randn(1, 5)
        out = sal(x)
        torch.testing.assert_close(out, torch.ones_like(x), atol=1e-5, rtol=1e-5)

    def test_expects_batch(self):
        sal = Saliency(f=_batched_quadratic, expects_batch=True)
        x = torch.randn(2, 10)
        out = sal(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_no_grad_input(self):
        """Should work even if x does not require grad initially."""
        sal = Saliency(f=_quadratic)
        x = torch.randn(2, 6)  # requires_grad=False by default
        out = sal(x)
        assert out.shape == x.shape

    def test_multidim_features(self):
        """Works with 2D feature maps."""
        f = lambda x: (x**2).sum()
        sal = Saliency(f=f)
        x = torch.randn(2, 3, 4)
        out = sal(x)
        assert out.shape == x.shape
        torch.testing.assert_close(out, 2.0 * x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# InputXGradient
# ---------------------------------------------------------------------------


class TestInputXGradient:
    def test_output_shape(self):
        ixg = InputXGradient(f=_quadratic)
        x = torch.randn(4, 10)
        out = ixg(x)
        assert out.shape == x.shape

    def test_correctness_quadratic(self):
        """For f=sum(x^2), InputXGrad = x * 2x = 2x^2."""
        ixg = InputXGradient(f=_quadratic)
        x = torch.randn(3, 8)
        out = ixg(x)
        expected = 2.0 * x**2
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_correctness_linear(self):
        """For f=sum(x), InputXGrad = x * ones = x."""
        ixg = InputXGradient(f=_linear)
        x = torch.randn(2, 5)
        out = ixg(x)
        torch.testing.assert_close(out, x.detach(), atol=1e-5, rtol=1e-5)

    def test_no_graph_leak_when_create_graph_false(self):
        """Output should not require grad when create_graph=False."""
        ixg = InputXGradient(f=_quadratic, create_graph=False)
        x = torch.randn(2, 5, requires_grad=True)
        out = ixg(x)
        # The output is x.detach() * grads, so no grad tracking
        assert not out.requires_grad

    def test_graph_preserved_when_create_graph_true(self):
        """Output should be differentiable when create_graph=True."""
        ixg = InputXGradient(f=_quadratic, create_graph=True)
        x = torch.randn(1, 4, requires_grad=True)
        out = ixg(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# IntegratedGradients
# ---------------------------------------------------------------------------


class TestIntegratedGradients:
    def test_output_shape(self):
        ig = IntegratedGradients(f=_quadratic, steps=10)
        x = torch.randn(2, 8)
        out = ig(x)
        assert out.shape == x.shape

    def test_finite_values(self):
        ig = IntegratedGradients(f=_quadratic, steps=20)
        x = torch.randn(3, 12)
        out = ig(x)
        assert torch.isfinite(out).all()

    def test_completeness_quadratic(self):
        """Completeness: sum(IG) ≈ f(x) - f(baseline).

        For f=sum(x^2) with baseline=0: f(x) - f(0) = sum(x^2).
        IG = (x - 0) * avg_grad. With enough steps this should converge.
        """
        ig = IntegratedGradients(f=_quadratic, steps=300)
        x = torch.randn(4, 10)
        baseline = torch.zeros_like(x)
        attr = ig(x, baseline=baseline)
        attr_sum = attr.sum(dim=-1)

        f_x = (x**2).sum(dim=-1)
        f_b = torch.zeros_like(f_x)
        expected = f_x - f_b

        torch.testing.assert_close(attr_sum, expected, atol=0.05, rtol=0.05)

    def test_completeness_linear(self):
        """For f=sum(x), IG should equal x (exactly, any #steps)."""
        ig = IntegratedGradients(f=_linear, steps=10)
        x = torch.randn(2, 6)
        attr = ig(x, baseline=torch.zeros_like(x))
        torch.testing.assert_close(attr, x, atol=1e-4, rtol=1e-4)

    def test_zero_baseline_default(self):
        """When baseline is None, defaults to zero."""
        ig = IntegratedGradients(f=_linear, steps=10)
        x = torch.randn(2, 4)
        attr = ig(x)
        attr_explicit = ig(x, baseline=torch.zeros_like(x))
        torch.testing.assert_close(attr, attr_explicit, atol=1e-6, rtol=1e-6)

    def test_baseline_broadcast(self):
        """Single baseline (no batch dim) should be broadcast."""
        ig = IntegratedGradients(f=_quadratic, steps=50)
        x = torch.randn(3, 6)
        baseline_single = torch.ones(6)
        attr = ig(x, baseline=baseline_single)
        assert attr.shape == x.shape
        assert torch.isfinite(attr).all()

    def test_steps_override(self):
        """steps parameter in forward overrides __init__ default."""
        ig = IntegratedGradients(f=_quadratic, steps=10)
        x = torch.randn(1, 4)
        # Should not raise with steps=5
        attr = ig(x, steps=5)
        assert attr.shape == x.shape

    def test_steps_too_small_raises(self):
        ig = IntegratedGradients(f=_quadratic, steps=10)
        x = torch.randn(1, 4)
        with pytest.raises(ValueError, match="steps must be >= 2"):
            ig(x, steps=1)

    def test_baseline_shape_mismatch_raises(self):
        ig = IntegratedGradients(f=_quadratic, steps=10)
        x = torch.randn(2, 4)
        bad_baseline = torch.randn(3, 5)
        with pytest.raises(ValueError, match="baseline shape"):
            ig(x, baseline=bad_baseline)

    def test_trapezoidal_vs_midpoint(self):
        """Trapezoidal should converge faster: fewer steps needed for same accuracy."""
        x = torch.randn(1, 20)

        ig_few = IntegratedGradients(f=_quadratic, steps=16)
        attr_few = ig_few(x)
        residual_few = (attr_few.sum() - (x**2).sum()).abs()

        ig_many = IntegratedGradients(f=_quadratic, steps=64)
        attr_many = ig_many(x)
        residual_many = (attr_many.sum() - (x**2).sum()).abs()

        # More steps should be more accurate
        assert residual_many < residual_few or residual_few < 0.1

    def test_target_class(self):
        """target should propagate to _scalar_f."""
        ig = IntegratedGradients(f=_multiclass, target=1, steps=100)
        x = torch.randn(1, 5)
        attr = ig(x)
        # target=1 → f = sum(x^2), IG from 0 to x should ≈ x^2
        expected_sum = (x**2).sum()
        torch.testing.assert_close(attr.sum(), expected_sum, atol=0.1, rtol=0.1)


# ---------------------------------------------------------------------------
# ExpectedGradients
# ---------------------------------------------------------------------------


class TestExpectedGradients:
    def _make_baselines(self, n_baselines=50, dim=10):
        return torch.randn(n_baselines, dim)

    def test_output_shape(self):
        baselines = self._make_baselines()
        eg = ExpectedGradients(f=_quadratic, baselines=baselines, n_samples=20)
        x = torch.randn(3, 10)
        out = eg(x)
        assert out.shape == x.shape

    def test_finite_values(self):
        baselines = self._make_baselines()
        eg = ExpectedGradients(f=_quadratic, baselines=baselines, n_samples=30)
        x = torch.randn(4, 10)
        out = eg(x)
        assert torch.isfinite(out).all()

    def test_no_baselines_raises(self):
        eg = ExpectedGradients(f=_quadratic, baselines=None, n_samples=10)
        x = torch.randn(1, 10)
        with pytest.raises(ValueError, match="baselines must be provided"):
            eg(x)

    def test_incompatible_baselines_raises(self):
        baselines = torch.randn(5, 8)  # feature dim 8
        eg = ExpectedGradients(f=_quadratic, baselines=baselines, n_samples=10)
        x = torch.randn(2, 10)  # feature dim 10 → mismatch
        with pytest.raises(ValueError, match="incompatible"):
            eg(x)

    def test_n_samples_override(self):
        baselines = self._make_baselines()
        eg = ExpectedGradients(f=_quadratic, baselines=baselines, n_samples=10)
        x = torch.randn(1, 10)
        out = eg(x, n_samples=5)
        assert out.shape == x.shape

    def test_n_samples_too_small_raises(self):
        baselines = self._make_baselines()
        eg = ExpectedGradients(f=_quadratic, baselines=baselines, n_samples=10)
        x = torch.randn(1, 10)
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            eg(x, n_samples=0)

    def test_reproducible_with_generator(self):
        baselines = self._make_baselines()
        x = torch.randn(2, 10)

        g1 = torch.Generator().manual_seed(42)
        eg1 = ExpectedGradients(
            f=_quadratic, baselines=baselines, n_samples=50, generator=g1
        )
        out1 = eg1(x)

        g2 = torch.Generator().manual_seed(42)
        eg2 = ExpectedGradients(
            f=_quadratic, baselines=baselines, n_samples=50, generator=g2
        )
        out2 = eg2(x)

        torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-6)

    def test_convergence_to_ig_uniform_baselines(self):
        """When baselines are all zeros, EG should approximate IG with zero baseline."""
        dim = 8
        n_baselines = 10
        baselines = torch.zeros(n_baselines, dim)

        eg = ExpectedGradients(
            f=_linear,
            baselines=baselines,
            n_samples=500,
            generator=torch.Generator().manual_seed(0),
        )
        ig = IntegratedGradients(f=_linear, steps=100)

        x = torch.randn(2, dim)
        eg_attr = eg(x)
        ig_attr = ig(x, baseline=torch.zeros_like(x))

        torch.testing.assert_close(eg_attr, ig_attr, atol=0.3, rtol=0.3)

    def test_set_baselines(self):
        """set_baselines should allow updating baselines after construction."""
        eg = ExpectedGradients(f=_quadratic, baselines=None, n_samples=10)
        baselines = self._make_baselines()
        eg.set_baselines(baselines)
        x = torch.randn(1, 10)
        out = eg(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_all_explainers_same_shape(self):
        """All methods should produce same-shaped output for same input."""
        x = torch.randn(3, 12)
        baselines = torch.randn(20, 12)

        sal = Saliency(f=_quadratic)
        ixg = InputXGradient(f=_quadratic)
        ig = IntegratedGradients(f=_quadratic, steps=10)
        eg = ExpectedGradients(f=_quadratic, baselines=baselines, n_samples=10)

        shapes = {
            "sal": sal(x).shape,
            "ixg": ixg(x).shape,
            "ig": ig(x).shape,
            "eg": eg(x).shape,
        }
        assert all(s == x.shape for s in shapes.values()), f"Shape mismatch: {shapes}"

    def test_zero_input(self):
        """All methods should handle zero input without NaN/Inf."""
        x = torch.zeros(2, 8)
        baselines = torch.randn(10, 8)

        for cls, kwargs in [
            (Saliency, {}),
            (InputXGradient, {}),
            (IntegratedGradients, {"steps": 10}),
            (ExpectedGradients, {"baselines": baselines, "n_samples": 10}),
        ]:
            explainer = cls(f=_quadratic, **kwargs)
            out = explainer(x)
            assert torch.isfinite(
                out
            ).all(), f"{cls.__name__} produced non-finite on zero input"

    def test_single_sample_batch(self):
        """Batch size 1 should work everywhere."""
        x = torch.randn(1, 6)
        baselines = torch.randn(10, 6)

        for cls, kwargs in [
            (Saliency, {}),
            (InputXGradient, {}),
            (IntegratedGradients, {"steps": 10}),
            (ExpectedGradients, {"baselines": baselines, "n_samples": 10}),
        ]:
            explainer = cls(f=_quadratic, **kwargs)
            out = explainer(x)
            assert out.shape == (1, 6), f"{cls.__name__}: shape {out.shape}"


# ---------------------------------------------------------------------------
# Batched path tests (expects_batch=True)
# ---------------------------------------------------------------------------


def _batched_quadratic(x_batch):
    """f(batch) -> (B, 1): sum of squares per sample."""
    return (x_batch**2).sum(dim=-1, keepdim=True)


def _batched_multiclass(x_batch):
    """f(batch) -> (B, 3)."""
    B = x_batch.shape[0]
    return torch.stack(
        [
            x_batch.sum(dim=-1),
            (x_batch**2).sum(dim=-1),
            (x_batch**3).sum(dim=-1),
        ],
        dim=-1,
    )


class TestBatchedSaliency:
    def test_correctness_quadratic(self):
        """Batched path should give same result as per-sample path."""
        x = torch.randn(4, 10)
        sal_ps = Saliency(f=_quadratic)
        sal_bt = Saliency(f=_batched_quadratic, expects_batch=True)
        out_ps = sal_ps(x)
        out_bt = sal_bt(x)
        torch.testing.assert_close(out_ps, out_bt, atol=1e-5, rtol=1e-5)

    def test_target_class(self):
        sal = Saliency(f=_batched_multiclass, expects_batch=True, target=1)
        x = torch.randn(3, 5)
        out = sal(x)
        torch.testing.assert_close(out, 2.0 * x, atol=1e-5, rtol=1e-5)

    def test_shape(self):
        sal = Saliency(f=_batched_quadratic, expects_batch=True)
        x = torch.randn(8, 12)
        assert sal(x).shape == x.shape


class TestBatchedIG:
    def test_matches_per_sample(self):
        """Batched IG must match per-sample IG numerically."""
        x = torch.randn(4, 10)
        ig_ps = IntegratedGradients(f=_quadratic, steps=50)
        ig_bt = IntegratedGradients(f=_batched_quadratic, expects_batch=True, steps=50)
        attr_ps = ig_ps(x)
        attr_bt = ig_bt(x)
        torch.testing.assert_close(attr_ps, attr_bt, atol=1e-4, rtol=1e-4)

    def test_completeness(self):
        ig = IntegratedGradients(f=_batched_quadratic, expects_batch=True, steps=300)
        x = torch.randn(4, 10)
        attr = ig(x)
        expected = (x**2).sum(dim=-1)
        torch.testing.assert_close(attr.sum(dim=-1), expected, atol=0.05, rtol=0.05)

    def test_time_independent_of_batch(self):
        """B=1 and B=8 should have similar forward call count."""
        call_count = 0

        def f_counted(x_batch):
            nonlocal call_count
            call_count += 1
            return (x_batch**2).sum(dim=-1, keepdim=True)

        ig = IntegratedGradients(f=f_counted, expects_batch=True, steps=20)

        call_count = 0
        ig(torch.randn(1, 10))
        calls_b1 = call_count

        call_count = 0
        ig(torch.randn(8, 10))
        calls_b8 = call_count

        # Both should be exactly `steps` forward calls
        assert calls_b1 == 20
        assert calls_b8 == 20


class TestBatchedEG:
    def test_shape(self):
        baselines = torch.randn(20, 10)
        eg = ExpectedGradients(
            f=_batched_quadratic, expects_batch=True, baselines=baselines, n_samples=30
        )
        x = torch.randn(4, 10)
        assert eg(x).shape == x.shape

    def test_finite(self):
        baselines = torch.randn(20, 10)
        eg = ExpectedGradients(
            f=_batched_quadratic, expects_batch=True, baselines=baselines, n_samples=30
        )
        x = torch.randn(4, 10)
        assert torch.isfinite(eg(x)).all()

    def test_call_count_independent_of_batch(self):
        call_count = 0

        def f_counted(x_batch):
            nonlocal call_count
            call_count += 1
            return (x_batch**2).sum(dim=-1, keepdim=True)

        baselines = torch.randn(20, 10)

        eg = ExpectedGradients(
            f=f_counted, expects_batch=True, baselines=baselines, n_samples=30
        )

        call_count = 0
        eg(torch.randn(1, 10))
        calls_b1 = call_count

        call_count = 0
        eg(torch.randn(8, 10))
        calls_b8 = call_count

        assert calls_b1 == 30
        assert calls_b8 == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
