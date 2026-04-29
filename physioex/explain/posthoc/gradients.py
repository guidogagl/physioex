import torch
import torch.nn as nn


class Saliency(torch.nn.Module):
    """Vanilla gradient (saliency map) attribution.

    Computes df/dx for each sample independently.

    Args:
        f: Scalar-valued function (or model wrapper).
        target: If not None, index into f's output vector to select class.
        create_graph: Keep computational graph for higher-order gradients.
        expects_batch: If True, ``f`` accepts ``(B, *features)`` input and
            returns ``(B, ...)`` output.  Enables **batched evaluation**:
            all B samples are processed in a single GPU kernel, making
            computation time independent of B.  When False, ``f`` is called
            once per sample in a Python loop.
        grad_batch_size: Maximum number of computation graphs alive at once
            during the backward pass (per-sample path only).
            ``0`` (default) = process everything in one backward call.
    """

    def __init__(
        self,
        f: callable,
        target: int = None,
        create_graph: bool = False,
        expects_batch: bool = False,
        grad_batch_size: int = 0,
        **kwargs,
    ):
        super(Saliency, self).__init__()
        self.f = f
        self.target = target
        self.create_graph = create_graph
        self.expects_batch = expects_batch
        self.grad_batch_size = grad_batch_size

    # ------------------------------------------------------------------
    # Single-sample evaluation (used when expects_batch=False)
    # ------------------------------------------------------------------

    def _scalar_f(self, x):
        """Evaluate *f* on a single (unbatched) sample and return a scalar."""
        if self.expects_batch:
            out = self.f(x.unsqueeze(0))
        else:
            out = self.f(x)

        flat = out.view(-1)
        if self.target is not None:
            return flat[self.target]
        return flat[0]

    # ------------------------------------------------------------------
    # Batched evaluation (used when expects_batch=True)
    # ------------------------------------------------------------------

    def _batched_scores(self, x_batch):
        """Evaluate *f* on ``(B, *feat)`` and return ``(B,)`` per-sample scores.

        Only valid when ``expects_batch=True``.
        """
        out = self.f(x_batch)  # (B, C) or (B,)
        B = x_batch.shape[0]
        flat = out.view(B, -1)
        if self.target is not None:
            return flat[:, self.target]
        return flat[:, 0]

    def _batched_grad(self, x_batch):
        """1 forward + 1 backward for an entire batch.

        Uses the sum trick: ``grad(scores.sum(), x_batch)`` gives correct
        per-sample gradients because ``scores[i]`` depends only on
        ``x_batch[i]``.  The GPU processes all B samples in parallel.
        """
        x_leaf = x_batch.detach().requires_grad_(True)
        scores = self._batched_scores(x_leaf)  # (B,)
        grads = torch.autograd.grad(
            scores.sum(),
            x_leaf,
            retain_graph=self.create_graph,
            create_graph=self.create_graph,
        )[0]
        return grads

    # ------------------------------------------------------------------
    # Per-sample helpers (used when expects_batch=False)
    # ------------------------------------------------------------------

    def _collect_grads(self, leaves, outputs):
        """Single backward over all independent (leaf, output) pairs."""
        total = torch.stack(outputs).sum()
        return torch.autograd.grad(
            total,
            leaves,
            retain_graph=self.create_graph,
            create_graph=self.create_graph,
        )

    def _chunked_grads(self, make_leaf_output, N):
        """Compute N per-sample gradients in memory-bounded chunks."""
        chunk = self.grad_batch_size if self.grad_batch_size > 0 else N
        all_grads = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            leaves, outputs = [], []
            for i in range(start, end):
                leaf, out = make_leaf_output(i)
                leaves.append(leaf)
                outputs.append(out)
            all_grads.extend(self._collect_grads(leaves, outputs))
        return all_grads

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x):
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        if self.expects_batch:
            # Batched: 1 forward + 1 backward regardless of B
            return self._batched_grad(x)

        # Per-sample fallback
        B = x.shape[0]

        def make(i):
            xi = x[i].detach().requires_grad_(True)
            return xi, self._scalar_f(xi)

        return torch.stack(self._chunked_grads(make, B), dim=0)


class InputXGradient(Saliency):
    """Input times gradient attribution: ``x * df/dx``."""

    def __init__(
        self,
        f: callable,
        create_graph: bool = False,
        expects_batch: bool = False,
        **kwargs,
    ):
        super(InputXGradient, self).__init__(
            f, create_graph=create_graph, expects_batch=expects_batch, **kwargs
        )

    def forward(self, x):
        grads = super().forward(x)
        x_val = x if self.create_graph else x.detach()
        return x_val * grads


class IntegratedGradients(Saliency):
    """Integrated Gradients (Sundararajan et al. 2017).

    Approximates the path integral from *baseline* to *x* using the
    trapezoidal rule.

    When ``expects_batch=True``, each interpolation step processes all B
    samples in a single GPU call, so total cost is ``2 * steps`` kernel
    launches regardless of B.
    """

    def __init__(
        self,
        f: callable,
        steps: int = 64,
        create_graph: bool = False,
        expects_batch: bool = False,
        **kwargs,
    ):
        super(IntegratedGradients, self).__init__(
            f, create_graph=create_graph, expects_batch=expects_batch, **kwargs
        )
        self.steps = steps

    def forward(self, x, baseline=None, steps=None):
        if steps is None:
            steps = self.steps
        if steps < 2:
            raise ValueError("steps must be >= 2")

        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        if baseline is None:
            baseline = torch.zeros_like(x)
        elif baseline.shape == x.shape[1:]:
            baseline = baseline.unsqueeze(0).expand_as(x)
        elif baseline.shape != x.shape:
            raise ValueError(
                f"baseline shape {baseline.shape} must match x {x.shape} or x[1:] {x.shape[1:]}"
            )

        B = x.shape[0]
        alphas = torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype)
        delta = x - baseline
        baseline_d = baseline.detach()
        delta_d = delta.detach()

        if self.expects_batch:
            # ---- Batched path: steps × (1 fwd + 1 bwd) ----
            # Each step evaluates all B samples in one GPU kernel.
            step_grads = []
            for s in range(steps):
                x_s = (baseline_d + alphas[s] * delta_d).requires_grad_(True)
                scores = self._batched_scores(x_s)  # (B,)
                g = torch.autograd.grad(
                    scores.sum(),
                    x_s,
                    retain_graph=self.create_graph,
                    create_graph=self.create_graph,
                )[0]
                step_grads.append(g)
            grads = torch.stack(step_grads)  # (steps, B, *feat)
        else:
            # ---- Per-sample path: steps*B forward, chunked backward ----
            N = steps * B

            def make(idx):
                s, b = divmod(idx, B)
                xi = (baseline_d[b] + alphas[s] * delta_d[b]).requires_grad_(True)
                return xi, self._scalar_f(xi)

            grads = torch.stack(self._chunked_grads(make, N)).reshape(
                steps, B, *x.shape[1:]
            )

        avg_grads = torch.trapezoid(grads, dx=1.0 / (steps - 1), dim=0)
        return delta * avg_grads


class ExpectedGradients(Saliency):
    """Expected Gradients (Erion et al. 2021).

    When ``expects_batch=True``, each MC sample processes all B input
    samples in one GPU call, so cost is ``2 * n_samples`` kernel
    launches regardless of B.
    """

    def __init__(
        self,
        f: callable,
        baselines: torch.Tensor = None,
        n_samples: int = 200,
        create_graph: bool = False,
        expects_batch: bool = False,
        generator: torch.Generator = None,
        **kwargs,
    ):
        super(ExpectedGradients, self).__init__(
            f, create_graph=create_graph, expects_batch=expects_batch, **kwargs
        )
        self.n_samples = n_samples
        if baselines is not None:
            self.register_buffer("baselines", baselines)
        else:
            self.baselines = None
        self._generator = generator

    def forward(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.baselines is None:
            raise ValueError(
                "baselines must be provided either at __init__ or via set_baselines()"
            )

        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        baselines = self.baselines.to(device=x.device, dtype=x.dtype)
        if baselines.dim() != x.dim() or baselines.shape[1:] != x.shape[1:]:
            raise ValueError(
                f"baselines shape {baselines.shape} incompatible with x shape {x.shape}: "
                f"expected (K, {', '.join(str(s) for s in x.shape[1:])})"
            )

        K = baselines.shape[0]
        B = x.shape[0]

        idx = torch.randint(
            0, K, (n_samples, B), device=x.device, generator=self._generator
        )
        alphas = torch.rand(
            (n_samples, B), device=x.device, dtype=x.dtype, generator=self._generator
        )

        x_d = x.detach()
        baselines_d = baselines.detach()

        if self.expects_batch:
            # ---- Batched path: n_samples × (1 fwd + 1 bwd) ----
            result = torch.zeros_like(x_d)
            for s in range(n_samples):
                base_s = baselines_d[idx[s]]  # (B, *feat)
                alpha_s = alphas[s].view(B, *([1] * (x.dim() - 1)))
                x_s = (base_s + alpha_s * (x_d - base_s)).requires_grad_(True)
                scores = self._batched_scores(x_s)  # (B,)
                g = torch.autograd.grad(
                    scores.sum(),
                    x_s,
                    retain_graph=self.create_graph,
                    create_graph=self.create_graph,
                )[0]
                result += (x_d - base_s) * g
            return result / n_samples
        else:
            # ---- Per-sample chunked path ----
            N = n_samples * B
            chunk = self.grad_batch_size if self.grad_batch_size > 0 else N
            result = torch.zeros_like(x_d)

            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                leaves, outputs, chunk_deltas = [], [], []
                for flat_idx in range(start, end):
                    s, b = divmod(flat_idx, B)
                    base_sb = baselines_d[idx[s, b]]
                    alpha_sb = alphas[s, b]
                    xi = (base_sb + alpha_sb * (x_d[b] - base_sb)).requires_grad_(True)
                    leaves.append(xi)
                    outputs.append(self._scalar_f(xi))
                    chunk_deltas.append(x_d[b] - base_sb)

                grads_tuple = self._collect_grads(leaves, outputs)
                chunk_grads = torch.stack(list(grads_tuple))
                chunk_delta = torch.stack(chunk_deltas)
                weighted = chunk_delta * chunk_grads
                for j, flat_idx in enumerate(range(start, end)):
                    _, b = divmod(flat_idx, B)
                    result[b] += weighted[j]

            return result / n_samples

    def set_baselines(self, baselines: torch.Tensor):
        """Replace baselines after construction."""
        if "baselines" in self.__dict__:
            del self.__dict__["baselines"]
        if "baselines" in self._buffers:
            del self._buffers["baselines"]
        self.register_buffer("baselines", baselines)
