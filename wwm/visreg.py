"""VISReg regularization for robust JEPA collapse prevention.

The loss follows Wu et al. (2026): center, scale, and distributional shape
are optimized separately. Shape normalization detaches the feature standard
deviation so its gradients cannot fight the scale objective.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def visreg_loss(
    z: torch.Tensor,
    num_slices: int = 128,
    eps: float = 5e-2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return unweighted VISReg total and its center/scale/shape components."""
    z = z.reshape(-1, z.shape[-1]).float()
    n, dim = z.shape
    if n < 2:
        zero = z.new_zeros(())
        return zero, {"center": zero, "scale": zero, "shape": zero, "mean_std": zero, "min_std": zero}

    mean = z.mean(dim=0)
    centered = z - mean
    std = centered.std(dim=0, unbiased=False)

    center = mean.square().mean()
    scale = (1.0 - std).square().mean()

    # Shape gradients otherwise scale as 1/std and become extreme while a
    # collapsed projector is still near zero variance. A smooth detached floor
    # keeps the shape objective useful without allowing it to destabilize the
    # scale objective during early recovery.
    shape_scale = torch.sqrt(std.detach().square() + float(eps) ** 2)
    normalized = centered / shape_scale
    directions = torch.randn(dim, int(num_slices), device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(float(eps))
    projected = torch.sort(normalized @ directions, dim=0).values

    quantiles = torch.arange(1, n + 1, device=z.device, dtype=z.dtype) / float(n + 1)
    normal = torch.distributions.Normal(z.new_tensor(0.0), z.new_tensor(1.0))
    target = normal.icdf(quantiles).unsqueeze(1)
    shape = (projected - target).square().mean()

    total = center + scale + shape
    return total, {
        "center": center,
        "scale": scale,
        "shape": shape,
        "mean_std": std.mean().detach(),
        "min_std": std.min().detach(),
    }
