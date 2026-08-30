"""EM physical constraints: Jakes temporal kernel + truncated-KL whitening.

Kernel derivation and the patch_t=2 deviation are documented in
预训练指挥文档_电磁物理约束v17.md §1.1 / §3. The single free parameter is
speed_scale (eta=0.97, dimensionless, fitted offline on 5/30/60 km/h); do not
re-fit it here.

Everything in this module runs in fp32 regardless of the autocast dtype: eigh on
a bf16 kernel loses the small eigenvalues the truncation rank depends on.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

LIGHT_SPEED = 299792458.0


def batch_speed_from_traj(traj: torch.Tensor, pos_scale: float, sample_period_s: float) -> torch.Tensor:
    """Metric UE speed [B] (m/s) from the trajectory tokens.

    traj[..., :3] is (pos - bs_position) / pos_scale (data.trajectory_features,
    mode "pos"), so multiplying the frame-to-frame difference by pos_scale
    restores metres. The engine never passes raw `pos`, so this is the only route
    to the physical speed. Verified against 训练集/train/pos: recovers
    5.000 / 29.998 / 59.994 km/h on the three labelled speed bins.
    """
    p = traj[..., :3].float() * float(pos_scale)          # [B,T,3] metres
    step = (p[:, 1:] - p[:, :-1]).norm(dim=-1)            # [B,T-1]
    return step.mean(dim=-1) / float(sample_period_s)     # [B]


def build_em_kernel(
    traj: torch.Tensor,
    *,
    n_time: int,
    patch_t: int,
    pos_scale: float,
    carrier_frequency_hz: float,
    sample_period_s: float,
    speed_scale: float = 0.97,
    jitter: float = 1e-4,
    time_basis: str = "center",
    kind: str = "doppler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jakes/Clarke temporal correlation kernel [B, n_time, n_time] and speed [B].

    time_basis="center": tau = patch_t*Δt*|i-j| (kernel sampled at tubelet
    centres). time_basis="blockavg": A K_full A^T, the linear push-forward of mean
    pooling. See §3.2 — "center" keeps the 30 km/h spectrum alive (spectral relstd
    0.299 vs 0.046) and is the default; "blockavg" exists only for the ablation.
    """
    speed_raw = batch_speed_from_traj(traj, pos_scale, sample_period_s)
    speed = speed_raw * float(speed_scale)
    batch = speed.shape[0]
    device = traj.device

    if kind == "identity":
        kernel = torch.eye(n_time, device=device).expand(batch, n_time, n_time).contiguous()
        return kernel, speed_raw

    lam = LIGHT_SPEED / float(carrier_frequency_hz)
    coeff = (2.0 * math.pi * speed / lam).view(-1, 1, 1)          # [B,1,1]

    if time_basis == "center":
        idx = torch.arange(n_time, device=device, dtype=torch.float32)
        tau = (idx[:, None] - idx[None, :]).abs() * (float(patch_t) * float(sample_period_s))
        kernel = torch.special.bessel_j0(coeff * tau)
    elif time_basis == "blockavg":
        full = n_time * int(patch_t)
        idx = torch.arange(full, device=device, dtype=torch.float32)
        tau = (idx[:, None] - idx[None, :]).abs() * float(sample_period_s)
        kernel_full = torch.special.bessel_j0(coeff * tau)        # [B,full,full]
        pool = torch.zeros(n_time, full, device=device, dtype=torch.float32)
        for group in range(n_time):
            pool[group, group * patch_t : (group + 1) * patch_t] = 1.0 / float(patch_t)
        kernel = torch.einsum("gf,bfh,kh->bgk", pool, kernel_full, pool)
    else:
        raise ValueError("Unknown time_basis: %s" % time_basis)

    kernel = 0.5 * (kernel + kernel.transpose(-1, -2))
    scale = kernel.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).rsqrt()
    kernel = kernel * scale.unsqueeze(-1) * scale.unsqueeze(-2)   # correlation, diag == 1
    eye = torch.eye(n_time, device=device, dtype=kernel.dtype).expand_as(kernel)
    kernel = kernel * (1.0 - float(jitter)) + float(jitter) * eye
    return kernel, speed_raw


def shuffle_kernel_across_speed(kernel: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    """Cross-speed-bin negative control (§10).

    An in-bin shuffle is NOT a valid control: within one speed bin the kernels are
    nearly identical, so shuffling changes nothing (source doc measured jakes vs
    in-bin shuffle = 0.8642 vs 0.8641). Sorting by speed and reversing guarantees
    the slowest sample receives the fastest sample's kernel.
    """
    order = torch.argsort(speed)
    return kernel[order.flip(0)][torch.argsort(order)]


def truncated_whitener(
    kernel: torch.Tensor, energy: float = 0.999
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, int]], torch.Tensor]:
    """Rank-truncated KL whitener. Returns (buckets, ranks).

    Each bucket is (L, sample_index, rank) with L of shape [n_sel, rank, T] so
    that L @ Z gives the whitened [n_sel, rank, D] latent.

    MUST be truncation, not clamp_min: at 5 km/h the 10x10 kernel keeps only 6
    credible modes (cond ~2.5e2 here, ~1e12 at 20 frames). clamp_min preserves the
    numerical null space and amplifies it ~1e3x — the source doc measured the
    Epps-Pulley statistic going to 201.0 / 6466.8 (reference 0.89) once noise
    exceeded 1e-2. Real encoder output always contains non-physical components.

    Ranks differ per sample (6 at 5 km/h, 10 at 30/60), so samples are bucketed by
    rank; each bucket is whitened as one batched tensor.
    """
    kernel = kernel.float()
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)          # ascending
    eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(-1)
    cumulative = eigenvalues.cumsum(-1) / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-12)
    ranks = (cumulative < float(energy)).sum(-1) + 1               # [B]
    buckets: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for rank in torch.unique(ranks).tolist():
        rank = int(rank)
        selected = (ranks == rank).nonzero(as_tuple=True)[0].contiguous()
        values = eigenvalues[selected, :rank].clamp_min(1e-8)
        vectors = eigenvectors[selected][:, :, :rank]
        # .contiguous() is REQUIRED, not defensive. flip() + advanced indexing +
        # narrowing + transpose leaves a view with a non-zero storage offset and
        # permuted strides; feeding that straight into the batched matmul in
        # whiten -> latent violates cuBLAS's alignment expectation and crashes
        # asynchronously in a later backward() with "misaligned address" /
        # "illegal memory access". The same failure mode is documented for
        # _angular_power_spectrum in model.py.
        whitener = (values.rsqrt().unsqueeze(-1) * vectors.transpose(-1, -2)).contiguous()
        buckets.append((whitener, selected, rank))
    return buckets, ranks


def temporal_correlation_gram(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-normalised Gram of [B, T, D] -> [B, T, T] with unit diagonal.

    Row normalisation (not the global-RMS form in the source implementation) is
    required: the kernel has diag == 1, so a Gram whose diagonal is not 1 makes the
    loss fit cross-time POWER dynamics instead of pure correlation. The source doc
    flags this as the first thing to check if the relation loss misbehaves.
    """
    z = z.float()
    gram = z @ z.transpose(-1, -2)
    scale = gram.diagonal(dim1=-2, dim2=-1).clamp_min(eps).rsqrt()
    return gram * scale.unsqueeze(-1) * scale.unsqueeze(-2)


def center_time(z: torch.Tensor) -> torch.Tensor:
    """Remove the time-axis mean of [B, T, D] (the C_T = I - 11^T/T operator)."""
    return z - z.mean(dim=1, keepdim=True)


@torch.no_grad()
def empirical_deltar_target(
    h_seq: torch.Tensor,
    speed_mps: torch.Tensor,
    *,
    lags: Sequence[int],
    carrier_frequency_hz: float,
    sample_period_s: float,
    speed_scale: float = 0.97,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample complex temporal-correlation residual against the Jakes prior.

    ``h_seq`` is physical (inverse-transformed) CSI with shape [B,T,2,H,W].
    The output concatenates Re(DeltaR) and Im(DeltaR), giving [B,2*len(lags)].
    """
    if h_seq.ndim != 5 or h_seq.shape[2] != 2:
        raise ValueError("Expected physical CSI [B,T,2,H,W], got %s" % (tuple(h_seq.shape),))
    lags = tuple(int(lag) for lag in lags)
    if not lags or min(lags) <= 0 or max(lags) >= h_seq.shape[1]:
        raise ValueError("DeltaR lags must be in [1,T-1], got %s for T=%d" % (lags, h_seq.shape[1]))

    h = h_seq.float()
    real, imag = h[:, :, 0], h[:, :, 1]
    empirical_real, empirical_imag = [], []
    for lag in lags:
        real_later, real_earlier = real[:, lag:], real[:, :-lag]
        imag_later, imag_earlier = imag[:, lag:], imag[:, :-lag]
        cross_real = real_later * real_earlier + imag_later * imag_earlier
        cross_imag = imag_later * real_earlier - real_later * imag_earlier
        power = 0.5 * (
            real_later.square() + imag_later.square()
            + real_earlier.square() + imag_earlier.square()
        )
        reduce_dims = tuple(range(1, cross_real.ndim))
        denominator = power.mean(dim=reduce_dims).clamp_min(float(eps))
        empirical_real.append(cross_real.mean(dim=reduce_dims) / denominator)
        empirical_imag.append(cross_imag.mean(dim=reduce_dims) / denominator)

    lag_tensor = torch.tensor(lags, device=h.device, dtype=torch.float32)
    wavelength = LIGHT_SPEED / float(carrier_frequency_hz)
    argument = (
        2.0 * math.pi * float(speed_scale) * speed_mps.float().unsqueeze(1)
        * float(sample_period_s) * lag_tensor.unsqueeze(0) / wavelength
    )
    jakes = torch.special.bessel_j0(argument)
    delta_real = torch.stack(empirical_real, dim=1) - jakes
    delta_imag = torch.stack(empirical_imag, dim=1)
    return torch.cat([delta_real, delta_imag], dim=1).contiguous()


def shuffle_target_within_speed_bins(
    target: torch.Tensor, speed_mps: torch.Tensor
) -> torch.Tensor:
    """Matched negative control: permute a [B, F] target WITHIN each speed bin.

    The per-speed label distribution is unchanged (unlike a cross-speed shuffle),
    so the Huber loss keeps its scale, but no sample keeps its own physics. Any
    skill the head acquires then comes from the population structure per bin,
    not from the per-sample correspondence. Speed bins are 1 km/h integers
    (speed * 3.6 rounded), which matches the grouping used by the conditional
    K-SIGReg loop.

    Implementation note: the permutation is computed in numpy on the CPU. The
    target is detached (a no-grad constant), so the round-trip costs nothing
    autograd-wise, and it keeps the CUDA stream free of the small index/randperm
    kernels that empirically corrupt the CUDA context on Windows (arm D
    reproduced a cublasGemmEx execution failure at the first backward when the
    shuffle ran as CUDA index_copy_/randperm ops inside the bf16 graph).
    """
    import numpy as np

    values = target.detach().cpu().numpy()
    groups = np.round(speed_mps.detach().float().cpu().numpy() * 3.6).astype(np.int64)
    permutation = np.arange(values.shape[0], dtype=np.int64)
    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        permutation[indices] = np.random.permutation(indices)
    return torch.from_numpy(values[permutation]).to(
        device=target.device, dtype=target.dtype
    )


def physical_relation_loss(
    z: torch.Tensor, kernel: torch.Tensor, centered: bool = False
) -> torch.Tensor:
    """||G_Z - K_c||^2 elementwise-mean. Both sides have unit diagonal.

    centered=False follows 预训练指挥文档_电磁物理约束v17.md §4: the raw
    row-normalised Gram against the raw kernel.

    centered=True follows WWM_EM_METHOD §5.2, which defines the target as
    K_bar = C_T K_c C_T with a time-centred representation. The two differ by more
    than a constant: a representation carrying a large DC component shared across
    all time groups (scenario identity, city, absolute power — information the
    localization and compression heads consume) has a raw Gram near the all-ones
    matrix regardless of its temporal structure. Uncentred, the loss therefore
    demands that component be deleted; centred, it constrains only the temporal
    VARIATION, which is what the Bessel correlation actually describes. Both sides
    are re-normalised to unit diagonal after centring so the comparison stays
    correlation-vs-correlation.
    """
    kernel = kernel.float()
    if not centered:
        return (temporal_correlation_gram(z) - kernel).pow(2).mean()
    n_time = kernel.shape[-1]
    centering = (
        torch.eye(n_time, device=kernel.device, dtype=kernel.dtype)
        - 1.0 / float(n_time)
    )
    centered_kernel = centering @ kernel @ centering
    scale = centered_kernel.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6).rsqrt()
    centered_kernel = centered_kernel * scale.unsqueeze(-1) * scale.unsqueeze(-2)
    return (temporal_correlation_gram(center_time(z)) - centered_kernel).pow(2).mean()


def context_alpha(kernel: torch.Tensor, masked_time: torch.Tensor) -> Optional[torch.Tensor]:
    """Single-block explanation ratio alpha_i [B, T] (§1.2a).

    alpha_i = mean_{m in M} K[m,i]^2, the fraction of the masked covariance trace a
    LMMSE estimate from time block i alone removes. Schur-complement positivity
    gives 0 <= alpha_i <= 1 for a correlation kernel. Returns None when some sample
    has no masked time block (the caller then falls back to grid weights).

    masked_time: [B, T] bool, True where the time block contains a masked token.
    """
    kernel = kernel.float().contiguous()
    masked = masked_time.to(kernel.dtype).contiguous()              # [B,T]
    counts = masked.sum(dim=-1)                                     # [B]
    if bool((counts <= 0).any()):
        return None
    squared = kernel.pow(2)                                         # [B,T,T]
    # sum over masked rows m of K[m,i]^2, normalised by |M|. Written as a bmm on
    # contiguous operands rather than einsum: this is a batched GEMM on the CUDA
    # stream and an unaligned operand here faults asynchronously later.
    weighted = torch.bmm(masked.unsqueeze(1), squared).squeeze(1)    # [B,T]
    return (weighted / counts.unsqueeze(-1)).contiguous()


class EMKernelCache:
    """Speed-quantised cache of kernels and whiteners, resident on the GPU.

    The kernel is a function of the single scalar theta = eta*v*fc*dt/c, so the whole
    dataset needs very few distinct kernels: the three labelled speed bins here
    (5.000 / 29.998 / 59.994 km/h) collapse to three entries. Caching them removes
    per-batch bessel_j0 / eigh / rank-bucketing from the training step entirely.

    That is not just an optimisation, it fixes two real failures observed on Windows:

      * Building the kernel on the GPU each step interleaved small batched-linalg
        kernels with the bf16 transformer graph and corrupted the CUDA context,
        surfacing asynchronously as "illegal memory access" / "misaligned address"
        inside a later backward().
      * Building it on the CPU each step instead put a host-to-device copy inside
        graph construction; the autograd engine then deadlocked with a worker thread
        wedged in cuMemcpyHtoDAsync_v2 (main thread waiting on a condition variable).

    With the cache, entries are built on the CPU and transferred once, under no_grad,
    outside any autograd graph. The steady-state training step does a single
    index_select on tensors that already live on the device.
    """

    def __init__(
        self,
        *,
        n_time: int,
        patch_t: int,
        carrier_frequency_hz: float,
        sample_period_s: float,
        speed_scale: float,
        jitter: float,
        energy: float,
        time_basis: str,
        kind: str,
        speed_quantum_mps: float = 1e-3,
    ) -> None:
        self.n_time = int(n_time)
        self.patch_t = int(patch_t)
        self.carrier_frequency_hz = float(carrier_frequency_hz)
        self.sample_period_s = float(sample_period_s)
        self.speed_scale = float(speed_scale)
        self.jitter = float(jitter)
        self.energy = float(energy)
        self.time_basis = str(time_basis)
        self.kind = str(kind)
        self.speed_quantum_mps = float(speed_quantum_mps)
        self._keys: Dict[int, int] = {}          # quantised speed -> row in the stacks
        self._kernels: Optional[torch.Tensor] = None    # [n_entries, T, T] on device
        self._whiteners: Optional[torch.Tensor] = None  # [n_entries, T, T] padded
        self._ranks: Optional[torch.Tensor] = None      # [n_entries] on device
        self._table: Optional[torch.Tensor] = None      # quantised speed -> row, on device

    def _build_entry(self, quantised: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """One kernel + padded whitener, built on the CPU."""
        speed = torch.tensor([quantised * self.speed_quantum_mps], dtype=torch.float32)
        # build_em_kernel wants a trajectory; synthesise a straight line at this speed.
        frames = self.n_time * self.patch_t
        step = speed.item() * self.sample_period_s
        positions = torch.zeros(1, frames, 3, dtype=torch.float32)
        positions[0, :, 0] = step * torch.arange(frames, dtype=torch.float32)
        kernel, _ = build_em_kernel(
            positions,
            n_time=self.n_time,
            patch_t=self.patch_t,
            pos_scale=1.0,                     # positions are already metric
            carrier_frequency_hz=self.carrier_frequency_hz,
            sample_period_s=self.sample_period_s,
            speed_scale=self.speed_scale,
            jitter=self.jitter,
            time_basis=self.time_basis,
            kind=self.kind,
        )
        buckets, ranks = truncated_whitener(kernel, self.energy)
        whitener, _selected, rank = buckets[0]
        padded = torch.zeros(self.n_time, self.n_time, dtype=torch.float32)
        padded[:rank] = whitener[0]
        return kernel[0].contiguous(), padded.contiguous(), int(rank)

    @torch.no_grad()
    def _ensure(self, quantised_values: Sequence[int], device: torch.device) -> None:
        missing = [int(v) for v in quantised_values if int(v) not in self._keys]
        if not missing and self._kernels is not None and self._kernels.device == device:
            return
        kernels = [] if self._kernels is None else [self._kernels.cpu()]
        whiteners = [] if self._whiteners is None else [self._whiteners.cpu()]
        ranks = [] if self._ranks is None else [self._ranks.cpu()]
        for quantised in sorted(set(missing)):
            kernel, whitener, rank = self._build_entry(quantised)
            self._keys[quantised] = len(self._keys)
            kernels.append(kernel.unsqueeze(0))
            whiteners.append(whitener.unsqueeze(0))
            ranks.append(torch.tensor([rank], dtype=torch.long))
        # One transfer per new entry, under no_grad and outside any graph. The lookup
        # table is built here too (not per call) so the steady-state path performs no
        # host-to-device writes at all.
        self._kernels = torch.cat(kernels, dim=0).to(device).contiguous()
        self._whiteners = torch.cat(whiteners, dim=0).to(device).contiguous()
        self._ranks = torch.cat(ranks, dim=0).to(device).contiguous()
        table = torch.full((max(self._keys) + 1,), 0, dtype=torch.long)
        for quantised_value, row in self._keys.items():
            table[quantised_value] = row
        self._table = table.to(device).contiguous()

    @torch.no_grad()
    def lookup(self, speed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """speed [B] (m/s, unscaled) -> (kernel [B,T,T], whitener [B,T,T], rank [B])."""
        device = speed.device
        quantised = torch.round(speed.detach().float() / self.speed_quantum_mps).long()
        # The only host round-trip: a handful of ints to check cache membership. This
        # is a device-to-host read outside the graph, not a host-to-device copy inside
        # it, so it cannot deadlock the autograd engine.
        self._ensure(torch.unique(quantised).tolist(), device)
        rows = self._table.index_select(
            0, quantised.clamp_(0, self._table.numel() - 1)
        )
        return (
            self._kernels.index_select(0, rows).contiguous(),
            self._whiteners.index_select(0, rows).contiguous(),
            self._ranks.index_select(0, rows),
        )


def whitener_buckets_from_cache(
    whiteners: torch.Tensor, ranks: torch.Tensor
) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Regroup per-sample padded whiteners into (whitener, index, rank) buckets."""
    buckets: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for rank in torch.unique(ranks).tolist():
        rank = int(rank)
        selected = (ranks == rank).nonzero(as_tuple=True)[0]
        buckets.append(
            (whiteners.index_select(0, selected)[:, :rank].contiguous(), selected, rank)
        )
    return buckets


def kernel_diagnostics(kernel: torch.Tensor, energy: float = 0.999) -> Dict[str, torch.Tensor]:
    """|offdiag|, retained rank, condition number, spectral relstd — the §3.3 audit."""
    k = kernel.float()
    n = k.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=k.device)
    eigenvalues = torch.linalg.eigvalsh(k).flip(-1).clamp_min(0.0)
    cumulative = eigenvalues.cumsum(-1) / eigenvalues.sum(-1, keepdim=True).clamp_min(1e-12)
    ranks = (cumulative < float(energy)).sum(-1) + 1
    stats: Dict[str, torch.Tensor] = {
        "offdiag": k[:, off].abs().mean(),
        "rank": ranks.float().mean(),
    }
    top = int(ranks.max())
    retained = eigenvalues[:, :top].clamp_min(1e-12)
    stats["cond"] = (retained[:, 0] / retained[:, -1]).mean()
    # Population std (unbiased=False) so the numbers match the §3.2 reference table
    # (0.299 at 30 km/h, 0.731 at 5 km/h) exactly; the unbiased estimator would
    # inflate them by sqrt(n/(n-1)) and make the gate ambiguous.
    stats["relstd"] = (
        retained.std(-1, unbiased=False) / retained.mean(-1).clamp_min(1e-12)
    ).mean()
    return stats
