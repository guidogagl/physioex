"""Unit tests for the unified metrics module (complexity, localization, infidelity, tfle)."""

import math

import pytest
import torch

from physioex.explain.posthoc.metrics import (
    complexity,
    localization,
    infidelity,
    tfle,
    resolution_product,
)


# =====================================================================
# Complexity
# =====================================================================


class TestComplexity:
    def test_uniform_max_entropy(self):
        """Uniform attribution → normalised entropy ≈ 1.0."""
        attr = torch.ones(1, 100)
        c = complexity(attr)
        assert c.item() == pytest.approx(1.0, rel=0.01)

    def test_concentrated_low_entropy(self):
        """Attribution concentrated on 1 feature → near-zero entropy."""
        attr = torch.zeros(1, 100)
        attr[0, 0] = 1.0
        c = complexity(attr)
        assert c.item() < 1.0

    def test_more_spread_higher_entropy(self):
        """Spreading attribution increases entropy."""
        attr_narrow = torch.zeros(1, 50)
        attr_narrow[0, :5] = 1.0

        attr_wide = torch.zeros(1, 50)
        attr_wide[0, :25] = 1.0

        assert complexity(attr_narrow).item() < complexity(attr_wide).item()

    def test_batched_shape(self):
        attr = torch.randn(4, 20)
        c = complexity(attr)
        assert c.shape == (4,)

    def test_unbatched(self):
        attr = torch.randn(20)
        c = complexity(attr, batched=False)
        assert c.dim() == 0

    def test_sign_invariant(self):
        """Complexity depends on magnitude, not sign."""
        attr = torch.randn(2, 30)
        torch.testing.assert_close(complexity(attr), complexity(-attr))

    def test_nonnegative(self):
        """Entropy is always >= 0."""
        attr = torch.randn(5, 40)
        assert (complexity(attr) >= 0).all()

    def test_multidim_flattened(self):
        """Extra dims are flattened."""
        attr = torch.randn(2, 3, 4)
        c = complexity(attr)
        assert c.shape == (2,)


# =====================================================================
# Localization
# =====================================================================


class TestLocalization:
    def test_perfect_overlap(self):
        """Attribution fully inside mask → high score."""
        attr = torch.zeros(1, 100)
        attr[0, 10:20] = 1.0
        mask = torch.zeros(1, 100)
        mask[0, 10:20] = 1.0
        loc = localization(attr=attr, mask=mask)
        # mu = 1.0, S_tot/S_in = 100/10 = 10
        assert loc.item() == pytest.approx(10.0, rel=0.01)

    def test_no_overlap(self):
        """Attribution fully outside mask → 0."""
        attr = torch.zeros(1, 100)
        attr[0, 50:60] = 1.0
        mask = torch.zeros(1, 100)
        mask[0, 10:20] = 1.0
        loc = localization(attr=attr, mask=mask)
        assert loc.item() == pytest.approx(0.0, abs=1e-6)

    def test_uniform_attribution(self):
        """Uniform attribution → score ≈ 1 (random baseline)."""
        attr = torch.ones(1, 100)
        mask = torch.zeros(1, 100)
        mask[0, :25] = 1.0
        loc = localization(attr=attr, mask=mask)
        assert loc.item() == pytest.approx(1.0, rel=0.01)

    def test_sign_weight(self):
        """sign_weight flips attribution signs before ReLU."""
        attr = torch.ones(1, 10)  # all positive
        mask = torch.ones(1, 10)

        # With sign_weight all positive → no change
        pos_w = torch.ones(1, 10)
        loc_pos = localization(attr=attr, mask=mask, sign_weight=pos_w)

        # With sign_weight all negative → attr flipped → ReLU kills all → 0
        neg_w = -torch.ones(1, 10)
        loc_neg = localization(attr=attr, mask=mask, sign_weight=neg_w)
        assert loc_neg.item() == pytest.approx(0.0, abs=1e-6)
        assert loc_pos.item() > 0

    def test_batched_shape(self):
        attr = torch.randn(4, 20).abs()
        mask = (torch.randn(4, 20) > 0).float()
        loc = localization(attr=attr, mask=mask)
        assert loc.shape == (4,)

    def test_unbatched(self):
        attr = torch.randn(20).abs()
        mask = (torch.randn(20) > 0).float()
        loc = localization(attr=attr, mask=mask, batched=False)
        assert loc.dim() == 0

    def test_zero_attribution(self):
        """All-zero attribution → no crash (inf guard)."""
        attr = torch.zeros(1, 50)
        mask = torch.ones(1, 50)
        loc = localization(attr=attr, mask=mask)
        assert torch.isfinite(loc).all()

    def test_empty_mask(self):
        """All-zero mask → clamp to 1 avoids division by zero."""
        attr = torch.randn(1, 30).abs()
        mask = torch.zeros(1, 30)
        loc = localization(attr=attr, mask=mask)
        assert torch.isfinite(loc).all()


# =====================================================================
# Infidelity — time domain
# =====================================================================


class TestInfidelityTime:
    @staticmethod
    def _f(x):
        """Simple energy function."""
        return (x**2).sum(dim=-1) if x.dim() > 1 else (x**2).sum()

    def test_oracle_lower_than_random(self):
        """Oracle attr (|x|) removes high-energy first → steeper drop → LOWER score."""
        torch.manual_seed(0)
        x = torch.randn(1, 50)
        attr_good = x.abs()
        attr_rand = torch.rand(1, 50)

        inf_good = infidelity(self._f, x, attr_good, domain="time", patch_size=5)
        inf_rand = infidelity(self._f, x, attr_rand, domain="time", patch_size=5)
        assert inf_good.item() <= inf_rand.item()

    def test_output_range(self):
        """Score should be in [0, 1]."""
        torch.manual_seed(42)
        x = torch.randn(3, 40)
        attr = torch.randn(3, 40).abs()
        inf = infidelity(self._f, x, attr, domain="time", patch_size=5)
        assert (inf >= -0.01).all() and (inf <= 1.01).all()

    def test_batched_shape(self):
        x = torch.randn(4, 30)
        attr = torch.randn(4, 30).abs()
        inf = infidelity(self._f, x, attr, domain="time", patch_size=5)
        assert inf.shape == (4,)

    def test_unbatched(self):
        x = torch.randn(30)
        attr = torch.randn(30).abs()
        inf = infidelity(self._f, x, attr, domain="time", patch_size=5, batched=False)
        assert inf.dim() == 0

    def test_patch_size_validation(self):
        x = torch.randn(1, 10)
        attr = torch.randn(1, 10)
        with pytest.raises(ValueError, match="patch_size"):
            infidelity(self._f, x, attr, domain="time", patch_size=0)


# =====================================================================
# Infidelity — frequency domain
# =====================================================================


class TestInfidelityFreq:
    @staticmethod
    def _f(x):
        return (x**2).sum(dim=-1) if x.dim() > 1 else (x**2).sum()

    def test_basic_finite(self):
        torch.manual_seed(0)
        x = torch.randn(1, 100)
        attr = torch.randn(1, 100).abs()
        inf = infidelity(self._f, x, attr, domain="frequency", fs=100.0, patch_size=5)
        assert torch.isfinite(inf).all()

    def test_requires_fs(self):
        x = torch.randn(1, 50)
        attr = torch.randn(1, 50)
        with pytest.raises(ValueError, match="fs is required"):
            infidelity(self._f, x, attr, domain="frequency", patch_size=5)

    def test_invalid_domain(self):
        x = torch.randn(1, 50)
        attr = torch.randn(1, 50)
        with pytest.raises(ValueError, match="domain must be"):
            infidelity(self._f, x, attr, domain="spatial", patch_size=5)

    def test_oracle_lower_than_random(self):
        """DFT-magnitude attribution → steepest freq drop → LOWER score."""
        torch.manual_seed(42)
        x = torch.randn(1, 100)
        X = torch.fft.rfft(x, dim=-1)
        attr_good = X.abs().squeeze()
        attr_rand = torch.rand(X.shape[-1])

        inf_good = infidelity(
            self._f,
            x,
            attr_good.unsqueeze(0),
            domain="frequency",
            fs=100.0,
            patch_size=3,
        )
        inf_rand = infidelity(
            self._f,
            x,
            attr_rand.unsqueeze(0),
            domain="frequency",
            fs=100.0,
            patch_size=3,
        )
        assert inf_good.item() <= inf_rand.item() + 0.05


# =====================================================================
# Cross-metric consistency
# =====================================================================


class TestCrossMetric:
    def test_all_metrics_accept_same_shapes(self):
        """All three metrics work on (B, N) tensors."""
        torch.manual_seed(0)
        x = torch.randn(3, 40)
        attr = torch.randn(3, 40).abs()
        mask = (torch.randn(3, 40) > 0).float()
        f = lambda x: (x**2).sum(dim=-1) if x.dim() > 1 else (x**2).sum()

        c = complexity(attr)
        l = localization(attr=attr, mask=mask)
        i = infidelity(f, x, attr, domain="time", patch_size=5)

        assert c.shape == (3,)
        assert l.shape == (3,)
        assert i.shape == (3,)

    def test_zero_attr_no_crash(self):
        x = torch.randn(1, 20)
        attr = torch.zeros(1, 20)
        mask = torch.ones(1, 20)
        f = lambda x: (x**2).sum(dim=-1) if x.dim() > 1 else (x**2).sum()

        assert torch.isfinite(complexity(attr)).all()
        assert torch.isfinite(localization(attr=attr, mask=mask)).all()
        assert torch.isfinite(infidelity(f, x, attr, domain="time", patch_size=5)).all()


# =====================================================================
# Multi-dim features (B, *feat)
# =====================================================================


class TestMultiDim:
    def test_complexity_2d_feat(self):
        attr = torch.randn(3, 4, 5)
        c = complexity(attr)
        assert c.shape == (3,)

    def test_complexity_3d_feat(self):
        attr = torch.randn(2, 3, 4, 5)
        c = complexity(attr)
        assert c.shape == (2,)

    def test_localization_2d_feat(self):
        attr = torch.randn(3, 4, 5).abs()
        mask = (torch.randn(3, 4, 5) > 0).float()
        l = localization(attr=attr, mask=mask)
        assert l.shape == (3,)

    def test_infidelity_2d_feat(self):
        """Infidelity flattens (B, *feat) → (B, N) internally."""
        f = lambda x: (x**2).sum(dim=-1)
        x = torch.randn(2, 30)
        attr = torch.randn(2, 30).abs()
        inf = infidelity(f, x, attr, domain="time", patch_size=5)
        assert inf.shape == (2,)


# =====================================================================
# Infidelity: vectorised scatter + batch_size
# =====================================================================


class TestInfidelityParallel:
    @staticmethod
    def _f(x):
        return (x**2).sum(dim=-1)

    def test_no_python_loop_over_batch(self):
        """Infidelity should work without per-sample Python loops."""
        x = torch.randn(8, 100)
        attr = torch.randn(8, 100).abs()
        inf = infidelity(self._f, x, attr, domain="time", patch_size=10)
        assert inf.shape == (8,)
        assert torch.isfinite(inf).all()

    def test_batch_size_same_result(self):
        """batch_size only controls chunking, not results."""
        torch.manual_seed(0)
        x = torch.randn(4, 60)
        attr = torch.randn(4, 60).abs()

        inf_all = infidelity(
            self._f, x, attr, domain="time", patch_size=10, batch_size=0
        )
        inf_8 = infidelity(self._f, x, attr, domain="time", patch_size=10, batch_size=8)
        inf_1 = infidelity(self._f, x, attr, domain="time", patch_size=10, batch_size=1)

        torch.testing.assert_close(inf_all, inf_8, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(inf_all, inf_1, atol=1e-5, rtol=1e-5)

    def test_batch_size_freq_same_result(self):
        torch.manual_seed(0)
        x = torch.randn(3, 80)
        attr = torch.randn(3, 80).abs()

        inf_all = infidelity(
            self._f, x, attr, domain="frequency", fs=100.0, patch_size=5, batch_size=0
        )
        inf_4 = infidelity(
            self._f, x, attr, domain="frequency", fs=100.0, patch_size=5, batch_size=4
        )

        torch.testing.assert_close(inf_all, inf_4, atol=1e-5, rtol=1e-5)

    def test_f_call_count(self):
        """With batch_size=0, f should be called ONCE for all steps."""
        count = 0

        def f_counted(x):
            nonlocal count
            count += 1
            return (x**2).sum(dim=-1)

        x = torch.randn(4, 100)
        attr = torch.randn(4, 100).abs()

        count = 0
        infidelity(f_counted, x, attr, domain="time", patch_size=10, batch_size=0)
        assert count == 1, f"expected 1 f-call, got {count}"

    def test_f_call_count_chunked(self):
        """With small batch_size, f is called multiple times."""
        count = 0

        def f_counted(x):
            nonlocal count
            count += 1
            return (x**2).sum(dim=-1)

        x = torch.randn(2, 50)
        attr = torch.randn(2, 50).abs()

        count = 0
        infidelity(f_counted, x, attr, domain="time", patch_size=10, batch_size=4)
        assert count > 1  # multiple chunks

    def test_large_batch(self):
        """Handles large batch without issues."""
        x = torch.randn(32, 200)
        attr = torch.randn(32, 200).abs()
        inf = infidelity(self._f, x, attr, domain="time", patch_size=20)
        assert inf.shape == (32,)
        assert torch.isfinite(inf).all()


# =====================================================================
# TFLE (Time-Frequency Localization Error)
# =====================================================================


class TestTFLE:
    @staticmethod
    def _make_event_map(n_freq, n_time, freq_idx, time_idx, spread=3):
        """Synthetic attribution map with a Gaussian event."""
        attr = torch.zeros(n_freq, n_time)
        for df in range(-spread, spread + 1):
            for dt in range(-spread * 5, spread * 5 + 1):
                fi = freq_idx + df
                ti = time_idx + dt
                if 0 <= fi < n_freq and 0 <= ti < n_time:
                    r2 = (df / max(spread, 1)) ** 2 + (dt / max(spread * 5, 1)) ** 2
                    attr[fi, ti] = math.exp(-r2 * 2)
        return attr

    def test_perfect_localization(self):
        """Delta-peak at the event location → zero error."""
        n_freq, n_time, fs = 26, 1000, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)

        attr = torch.zeros(n_freq, n_time)
        # Event at 10Hz (band ~5), t=3.0s (sample 300)
        attr[5, 300] = 1.0

        result = tfle(
            attr,
            event_freq=10.0,
            event_time=3.0,
            fs=fs,
            band_frequencies=band_freqs,
        )
        assert result["time_error"].item() == pytest.approx(0.0, abs=0.02)
        assert result["freq_error"].item() < 3.0  # within one band

    def test_shifted_peak_gives_nonzero_error(self):
        """Peak offset from GT → nonzero error."""
        n_freq, n_time, fs = 26, 1000, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)

        attr = torch.zeros(n_freq, n_time)
        # Event at 10Hz, t=3.0s, but peak at t=4.0s
        attr[5, 400] = 1.0

        result = tfle(
            attr,
            event_freq=10.0,
            event_time=3.0,
            fs=fs,
            band_frequencies=band_freqs,
        )
        assert result["time_error"].item() == pytest.approx(1.0, abs=0.02)

    def test_batched_shape(self):
        n_freq, n_time, fs = 10, 200, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)
        attr = torch.randn(4, n_freq, n_time).abs()

        result = tfle(
            attr,
            event_freq=20.0,
            event_time=1.0,
            fs=fs,
            band_frequencies=band_freqs,
        )
        assert result["time_error"].shape == (4,)
        assert result["freq_error"].shape == (4,)
        assert result["tf_spread"].shape == (4,)

    def test_unbatched_scalar(self):
        n_freq, n_time, fs = 10, 200, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)
        attr = torch.randn(n_freq, n_time).abs()

        result = tfle(
            attr,
            event_freq=20.0,
            event_time=1.0,
            fs=fs,
            band_frequencies=band_freqs,
        )
        assert result["time_error"].dim() == 0
        assert result["tf_spread"].dim() == 0

    def test_spread_decreases_with_concentration(self):
        """More concentrated map → lower spread."""
        n_freq, n_time, fs = 26, 1000, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)

        # Narrow event
        narrow = self._make_event_map(n_freq, n_time, 5, 300, spread=1)
        # Wide event
        wide = self._make_event_map(n_freq, n_time, 5, 300, spread=5)

        r_narrow = tfle(narrow, 10.0, 3.0, fs, band_freqs)
        r_wide = tfle(wide, 10.0, 3.0, fs, band_freqs)

        assert r_narrow["tf_spread"].item() < r_wide["tf_spread"].item()

    def test_all_values_finite(self):
        n_freq, n_time, fs = 10, 100, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)
        attr = torch.randn(3, n_freq, n_time).abs()

        result = tfle(attr, 25.0, 0.5, fs, band_freqs)
        for k, v in result.items():
            assert torch.isfinite(v).all(), f"{k} has non-finite values"

    def test_all_values_nonnegative(self):
        n_freq, n_time, fs = 10, 100, 100.0
        band_freqs = torch.linspace(0, 50, n_freq)
        attr = torch.randn(2, n_freq, n_time).abs()

        result = tfle(attr, 25.0, 0.5, fs, band_freqs)
        for k, v in result.items():
            assert (v >= 0).all(), f"{k} has negative values"

    def test_sg_vs_stft_resolution(self):
        """SG with fine resolution should have lower spread than coarse STFT."""
        n_time, fs = 1000, 100.0

        # SG-like map: 26 bands × 1000 time (full resolution)
        n_freq_sg = 26
        band_freqs_sg = torch.linspace(0, 50, n_freq_sg)
        attr_sg = TestTFLE._make_event_map(n_freq_sg, n_time, 5, 300, spread=2)

        # STFT-like map: 33 bands × 63 frames (Gabor-limited)
        n_freq_stft, n_frames = 33, 63
        band_freqs_stft = torch.linspace(0, 50, n_freq_stft)
        attr_stft = TestTFLE._make_event_map(n_freq_stft, n_frames, 7, 19, spread=2)

        r_sg = tfle(attr_sg, 10.0, 3.0, fs, band_freqs_sg)
        # STFT has different time mapping: frame 19 at hop=16 → t≈3.04s
        fs_stft = n_frames / (n_time / fs)  # effective frame rate
        r_stft = tfle(attr_stft, 10.0, 3.0, fs_stft, band_freqs_stft)

        # SG time spread should be smaller (finer time resolution)
        assert r_sg["time_spread"].item() < r_stft["time_spread"].item()


# =====================================================================
# Resolution Product (theoretical bound)
# =====================================================================


class TestResolutionProduct:
    def test_stft_gabor_bound(self):
        """STFT resolution product is always 1.0 (Gabor bound)."""
        for n_fft in [32, 64, 128, 256]:
            hop = n_fft // 4
            rp = resolution_product(
                "stft", fs=100.0, signal_length=3000, n_fft=n_fft, hop_length=hop
            )
            assert rp["dt_x_df"] == pytest.approx(1.0, rel=1e-6)

    def test_sg_below_gabor(self):
        """SG resolution product is below Gabor bound."""
        rp = resolution_product("sg", fs=100.0, signal_length=3000, freq_step=2.0)
        assert rp["dt_x_df"] < 1.0
        assert rp["dt_x_df"] == pytest.approx(2.0 / 100.0, rel=1e-6)

    def test_sg_more_cells_than_stft(self):
        """SG has more independent TF cells than STFT."""
        rp_sg = resolution_product("sg", fs=100.0, signal_length=3000, freq_step=2.0)
        rp_stft = resolution_product(
            "stft", fs=100.0, signal_length=3000, n_fft=64, hop_length=16
        )
        assert rp_sg["n_cells"] > rp_stft["n_cells"]

    def test_sg_cells_scale_with_n(self):
        """SG cells grow as O(N × n_bands), STFT as O(N)."""
        for N in [1000, 2000, 4000]:
            rp_sg = resolution_product("sg", fs=100.0, signal_length=N, freq_step=2.0)
            rp_stft = resolution_product(
                "stft", fs=100.0, signal_length=N, n_fft=64, hop_length=16
            )
            ratio = rp_sg["n_cells"] / rp_stft["n_cells"]
            assert ratio > 5  # SG has at least 5x more cells

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            resolution_product("wavelet", fs=100.0, signal_length=1000)

    def test_stft_requires_params(self):
        with pytest.raises(ValueError, match="n_fft and hop_length"):
            resolution_product("stft", fs=100.0, signal_length=1000)

    def test_sg_requires_freq_step(self):
        with pytest.raises(ValueError, match="freq_step"):
            resolution_product("sg", fs=100.0, signal_length=1000)

    def test_return_keys(self):
        rp = resolution_product("sg", fs=100.0, signal_length=1000, freq_step=2.0)
        expected_keys = {"dt", "df", "dt_x_df", "n_freq", "n_time", "n_cells"}
        assert set(rp.keys()) == expected_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
