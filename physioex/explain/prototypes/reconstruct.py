import torch
import torch.nn as nn
from tqdm import tqdm

from physioex.explain.posthoc.gradients import IntegratedGradients
from physioex.explain.prototypes.local import (
    proj_fn,
    get_prototypes,
    PrototypeRelevance,
)

from physioex.models.prosleepnet import ProtoSleepTransformer


def data_driven_reconstruction(
    loader: torch.utils.data.DataLoader,
    model: ProtoSleepTransformer,
    index: int = 0,
    device: torch.device = torch.device("cpu"),
    n: int = 1,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    model.to(device)

    prototypes = get_prototypes(model).to(device)
    prototype = prototypes[index].unsqueeze(0)

    best = []

    for batch in tqdm(loader, desc="Scanning", leave=False):
        x, y = batch

        # first remove -1 labels
        # x shape : batch x L x in_chans x T x F
        # y shape : batch x L
        batch_size, L, in_chans, T, F = x.shape

        y = y.reshape(batch_size * L)
        x = x.reshape(batch_size * L, in_chans, T, F)

        mask = y != -1

        y = y[mask]
        x = x[mask].to(device).float()

        with torch.no_grad():
            embeddings = proj_fn(model, x)
            all_sim = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1),
                prototypes.unsqueeze(0),
                dim=-1,
            )
            assigned_mask = all_sim.argmax(dim=1) == int(index)

            if not torch.any(assigned_mask):
                continue

            filtered_x = x[assigned_mask]
            batch_sim = torch.nn.functional.cosine_similarity(
                embeddings[assigned_mask],
                prototype,
            )

            topk = min(int(n), int(batch_sim.numel()))
            if topk == 0:
                continue
            batch_vals, batch_idx = torch.topk(batch_sim, k=topk, largest=True)
            for val, idx in zip(batch_vals, batch_idx):
                best.append((float(val.item()), filtered_x[idx].detach().cpu()))

        if best:
            best.sort(key=lambda item: item[0], reverse=True)
            if len(best) > n:
                best = best[:n]

    if not best:
        return None

    if n == 1:
        return best[0][1]

    return torch.stack([item[1] for item in best], dim=0)


def model_driven_reconstructions(
    model: ProtoSleepTransformer,
    index: int = 0,
    device: torch.device = torch.device("cpu"),
    n: int = 1,
    init: torch.Tensor = None,
    steps: int = 200,
    lr: float = 1e-2,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    model.to(device)

    n = int(n)
    if n <= 0:
        return None

    def infer_input_shape():
        in_chans = getattr(model, "in_chan", None)
        T = getattr(model, "T", None)
        F = getattr(model, "F", None)

        if in_chans is None or T is None or F is None:
            raise ValueError(
                "Cannot infer input shape; provide init with shape (n, in_chans, T, F)."
            )

        return int(in_chans), int(T), int(F)

    if init is None:
        in_chans, T, F = infer_input_shape()
        init = torch.randn(n, in_chans, T, F, device=device)
    else:
        init = init.to(device)
        if init.dim() == 3:
            init = init.unsqueeze(0)

        if init.size(0) < n:
            in_chans, T, F = init.shape[1:]
            extra = torch.randn(n - init.size(0), in_chans, T, F, device=device)
            init = torch.cat([init, extra], dim=0)
        elif init.size(0) > n:
            init = init[:n]

    params = init.clone().detach().requires_grad_(True)

    prototypes = get_prototypes(model).to(device)
    prototype = prototypes[index].unsqueeze(0)

    param_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam([params], lr=lr)
    progress = tqdm(range(int(steps)), desc="Reconstructing", leave=False)

    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        sims = torch.nn.functional.cosine_similarity(
            proj_fn(model, params),
            prototype,
        )
        mean_sim = sims.mean()
        loss = -mean_sim
        loss.backward()
        optimizer.step()
        progress.set_postfix({"mean_sim": f"{mean_sim.item():.4f}"})

    recon = params.detach().cpu()

    return recon
