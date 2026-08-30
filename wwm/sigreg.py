"""Per-modality SIGReg regularizer (LeJEPA, Balestriero & LeCun, arXiv:2511.08544).

Pushes an embedding set toward the standard isotropic Gaussian by testing many
random 1-D projections for standard-normality (Cramér–Wold: if every 1-D
projection is N(0,1), the joint law is the standard isotropic Gaussian).

VERIFIED against the official implementation (galilai-group/lejepa, MINIMAL.md +
lejepa/univariate/epps_pulley.py). The 1-D goodness-of-fit statistic is the
Epps–Pulley test: T = N * ∫ |phi_hat(t) - phi_N(t)|^2 w(t) dt, evaluated by
trapezoidal quadrature over t in [0, t_max] (symmetry of the CF halves the
domain). phi_N(t)=exp(-t^2/2) is the N(0,1) CF and doubles as the integration
window; the quadrature weights carry it. Random directions are columns of
randn(D, P) normalised to unit L2 (≈ uniform on the sphere).
"""
from __future__ import annotations

import torch
from torch import distributed as dist


def _all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn import all_reduce as functional_all_reduce
        from torch.distributed.nn import ReduceOp

        return functional_all_reduce(x, ReduceOp.AVG)
    return x


def _quadrature(t_max: float, n_points: int, device, dtype) -> tuple:
    """Trapezoidal knots/weights on [0, t_max] with the exp(-t^2/2) window folded in."""
    assert n_points % 2 == 1, "Epps–Pulley n_points must be odd"
    t = torch.linspace(0.0, t_max, n_points, device=device, dtype=dtype)
    dt = t_max / (n_points - 1)
    w = torch.full((n_points,), 2.0 * dt, device=device, dtype=dtype)
    w[0] = dt
    w[-1] = dt  # half-weight at the endpoints (trapezoid rule)
    phi = torch.exp(-0.5 * t.square())  # N(0,1) characteristic function (real)
    return t, phi, w * phi  # weights carry the window == integration measure


def sigreg_loss(z: torch.Tensor, num_projections: int = 256, t_max: float = 3.0, n_points: int = 17) -> torch.Tensor:
    """z: [..., D] embeddings. Returns the mean per-projection Epps–Pulley statistic (>=0)."""
    z = z.reshape(-1, z.shape[-1]).float()
    n, d = z.shape
    if n < 2:
        return z.new_zeros(())
    # Random unit-L2-column directions (≈ uniform on S^{d-1}); real RNG here.
    A = torch.randn(d, num_projections, device=z.device, dtype=z.dtype)
    A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
    proj = z @ A  # [n, P] — 1-D marginals to test against N(0,1)
    t, phi, weights = _quadrature(t_max, n_points, z.device, z.dtype)
    x_t = proj.unsqueeze(-1) * t  # [n, P, K]
    cos_mean = _all_reduce_mean(torch.cos(x_t).mean(dim=0))  # [P, K] Re(phi_hat)
    sin_mean = _all_reduce_mean(torch.sin(x_t).mean(dim=0))  # [P, K] Im(phi_hat)
    err = (cos_mean - phi).square() + sin_mean.square()  # |phi_hat - phi_N|^2
    statistic = (err @ weights) * float(n)  # Epps–Pulley N-scaling, per projection
    return statistic.mean()
