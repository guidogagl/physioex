"""
Unit tests for the 4 explain module bug fixes:
  - Bug 1: einsum dimension mismatch in metrics/time.py complexity()
  - Bug 2: shape construction error in functionizer.py Funct.forward()
  - Bug 3: DFTExpectedGradients passes invalid baselines kwarg (vidft.py)
  - Bug 4: STFTExpectedGradients passes invalid baselines kwarg (vistdft.py)

Run:  cd /mnt/nfs/guido/home/dev/physioex && python test/tests/test_explain_fixes.py
"""

import sys

import torch

passed, failed = 0, 0


def report(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"[{tag}] {name}{suffix}")


# ---------------------------------------------------------------------------
# Test 1: Bug 1 -- einsum dimension mismatch in complexity()
# ---------------------------------------------------------------------------
def test_complexity_einsum():
    try:
        from physioex.explain.posthoc.metrics import complexity

        C_i = torch.rand(2, 4)  # batch=2, features=4
        result = complexity(C_i, batched=True)
        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        assert torch.all(torch.isfinite(result)), "Result contains non-finite values"
        report("Bug 1: complexity() einsum dimension fix", True)
    except Exception as exc:
        report("Bug 1: complexity() einsum dimension fix", False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: Bug 2 -- shape construction error in Funct.forward()
# ---------------------------------------------------------------------------
def test_functionizer_shape():
    try:
        from physioex.explain.posthoc.functionizer import Funct

        # Dummy model: expects (batch, L, n) -> (batch, L, n_classes)
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # x: (batch, L, n) -> return (batch, L, 3) logits
                b, seq, n = x.shape
                return torch.randn(b, seq, 3)

        model = DummyModel()
        funct = Funct(model, out_index=0, softmax=False)

        # Test with 1D input (the failing case before the fix)
        x_1d = torch.randn(64)
        result_1d = funct(x_1d)
        assert result_1d.ndim == 1, f"Expected 1D output, got {result_1d.ndim}D"
        report("Bug 2: Funct.forward() 1D input shape fix", True)
    except Exception as exc:
        report("Bug 2: Funct.forward() 1D input shape fix", False, str(exc))


def test_functionizer_shape_2d_regression():
    try:
        from physioex.explain.posthoc.functionizer import Funct

        # With 2D input (4, 64), Funct prepends [1, 1] -> (1, 1, 4, 64)
        # The model needs to handle that 4D shape.
        class DummyModel2D(torch.nn.Module):
            def forward(self, x):
                # x could be (batch, L, ...) with arbitrary trailing dims
                b = x.shape[0]
                L = x.shape[1]
                return torch.randn(b, L, 3)

        model = DummyModel2D()
        funct = Funct(model, out_index=0, softmax=False)

        # Test with 2D input (batch, features) -- regression protection
        x_2d = torch.randn(4, 64)
        result_2d = funct(x_2d)
        assert result_2d.ndim == 1, f"Expected 1D output, got {result_2d.ndim}D"
        report("Bug 2: Funct.forward() 2D input regression protection", True)
    except Exception as exc:
        report("Bug 2: Funct.forward() 2D input regression protection", False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: Bug 3 -- DFTExpectedGradients baselines kwarg fix
# ---------------------------------------------------------------------------
def test_dft_expected_gradients_baselines():
    try:
        from physioex.explain.posthoc.vidft import DFTExpectedGradients

        N = 32  # signal length

        # Simple dummy function: sum of absolute values
        def dummy_f(x):
            return torch.sum(torch.abs(x))

        # Create initial baselines in DFT domain (rfft of length 32 -> 17 complex bins)
        init_baselines_time = torch.randn(3, N)  # K=3 baseline samples
        # We need DFT-domain baselines for the constructor since ExpectedGradients
        # stores them as a buffer. The DFTExpectedGradients constructor passes
        # baselines through **kwargs to ExpectedGradients which registers them.
        # But baselines in __init__ are expected to already be in DFT domain
        # because forward() will DFT-transform the time-domain baselines passed at call time.
        # Actually, let's just pass time-domain baselines at forward() and
        # provide DFT-domain initial baselines to __init__.
        init_baselines_dft = torch.fft.rfft(init_baselines_time)

        explainer = DFTExpectedGradients(
            f=dummy_f,
            n=N,
            baselines=init_baselines_dft,
            steps=1,
            n_samples=2,
        )

        # Time-domain input and new baselines
        x = torch.randn(1, N)
        new_baselines = torch.randn(3, N)

        # This should NOT raise TypeError about unexpected 'baselines' kwarg
        result = explainer.forward(x, baselines=new_baselines, steps=1)
        assert result.shape[0] == 1, f"Expected batch dim 1, got {result.shape[0]}"
        report("Bug 3: DFTExpectedGradients baselines kwarg fix", True)
    except Exception as exc:
        report("Bug 3: DFTExpectedGradients baselines kwarg fix", False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: Bug 4 -- STFTExpectedGradients baselines kwarg fix
# ---------------------------------------------------------------------------
def test_stft_expected_gradients_baselines():
    try:
        from physioex.explain.posthoc.vistdft import STFTExpectedGradients

        N = 64  # signal length
        N_FFT = 16
        HOP = 8

        def dummy_f(x):
            return torch.sum(torch.abs(x))

        # Create initial baselines in STFT domain for __init__
        init_baselines_time = torch.randn(3, N)
        # Compute STFT to get the right shape for initial baselines
        window = torch.hann_window(N_FFT)
        init_stft = torch.stft(
            init_baselines_time,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=N_FFT,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        explainer = STFTExpectedGradients(
            f=dummy_f,
            n_fft=N_FFT,
            length=N,
            hop_length=HOP,
            baselines=init_stft,
            steps=1,
            n_samples=2,
        )

        # Time-domain input and new baselines
        x = torch.randn(1, N)
        new_baselines = torch.randn(3, N)

        # This should NOT raise TypeError about unexpected 'baselines' kwarg
        result = explainer.forward(x, baselines=new_baselines, steps=1)
        assert result.shape[0] == 1, f"Expected batch dim 1, got {result.shape[0]}"
        report("Bug 4: STFTExpectedGradients baselines kwarg fix", True)
    except Exception as exc:
        report("Bug 4: STFTExpectedGradients baselines kwarg fix", False, str(exc))


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Running explain module fix tests")
    print("=" * 60)

    test_complexity_einsum()
    test_functionizer_shape()
    test_functionizer_shape_2d_regression()
    test_dft_expected_gradients_baselines()
    test_stft_expected_gradients_baselines()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
