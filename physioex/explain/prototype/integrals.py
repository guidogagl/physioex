import torch
import torch.nn as nn

from typing import Union, Callable


def expected_line_integral(fn: callable, P: torch.Tensor, baselines: torch.Tensor):

    SG = 0

    x = baselines[0]
    baselines = baselines[1:]
    n_points = len(baselines)

    t = torch.linspace(0, 1, n_points, device=x.device, dtype=x.dtype)

    for i in range(1, n_points):

        def path(alpha):
            return x * alpha - (1 - alpha) * baselines[i].to(x.device)

        def jac_path(alpha):
            return x - baselines[i].to(x.device)

        SG += step_integral(fn, P, t, i, path, jac_path)

    return SG


def step_integral(
    fn: callable,
    P: torch.Tensor,
    t: torch.Tensor,
    i: int,
    path: callable,
    jac_path: callable = None,
):

    def distance(x: torch.Tensor):
        return -torch.cdist(fn(x), P, p=2).view(-1)

    G0 = torch.autograd.grad(outputs=distance(path(t[i - 1])), inputs=path(t[i - 1]))[0]
    G1 = torch.autograd.grad(outputs=distance(path(t[i])), inputs=path(t[i]))[0]

    Jp0 = jac_path(t[i - 1])  #
    Jp1 = jac_path(t[i])

    G0 = torch.matmul(G0, Jp0)
    G1 = torch.matmul(G1, Jp1)

    SG = (t[i] - t[i - 1]) * (G0 + G1) / 2

    return SG


class ExpectedGradients(nn.Module):
    def __init__(
        self,
        f: nn.Module,
        baselines: torch.Tensor,
    ):
        super().__init__()

        self.baselines = baselines
        self.n_points = len(baselines)
        self.fn = f

    def forward(self, P: torch.Tensor, channel_index: int = 0):

        def fn(x: torch.Tensor):
            return self.fn(x)[0, channel_index]

        emb = self.fn(self.baselines)[:, channel_index]
        # order the baselines based on the distance to the prototype P
        dist = torch.cdist(emb.unsqueeze(1), P.unsqueeze(0), p=2).squeeze()
        sorted_indices = torch.argsort(dist)
        self.baselines = self.baselines[sorted_indices]

        return expected_line_integral(fn, P, self.baselines)
