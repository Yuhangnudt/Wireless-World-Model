"""3GPP TS 38.214 Type-I Single-Panel rank-1 codebook.

Why this exists: the four-task eval labels beams with a 32-point DFT proxy over the
antenna axis. The WWM paper instead reports Type-I codebook beam prediction, so the
proxy's Top-1 is not comparable to the paper's -- a 32-way problem versus the
paper's codebook is a different classification task, and Top-1 scales with codebook
size. This module builds the real Type-I SP codebook so beam accuracy can be
reported under the paper's label definition.

Array assumption (MUST be stated whenever these numbers are quoted): the datasets
only record 32 antenna ports and no geometry. The legacy generator manifest
(archive_dataset_non_current/.../paper_scenario_manifest.json) records
tx_rows=4, tx_cols=4, polarization=VH, i.e. a 4x4 dual-polarised panel ->
2*N1*N2 = 32 ports with (N1, N2) = (4, 4). Oversampling (O1, O2) = (4, 4) per
TS 38.214 Table 5.2.2.2.1-2.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def type1_sp_codebook(
    n1: int = 4,
    n2: int = 4,
    o1: int = 4,
    o2: int = 4,
    device: torch.device | str = "cpu",
    port_order: str = "pol_n2_n1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W, index) for Type-I Single-Panel rank 1.

    W: complex64 [K, 2*n1*n2] unit-norm codewords, K = o1*n1 * o2*n2 * 4.
    index: int64 [K, 3] giving (l, m, n) = (beam1, beam2, co-phase) per codeword.

    port_order says how the 32-long port axis of H factorises:
      "pol_n1_n2" -- port = pol*n1*n2 + i1*n2 + i2
      "pol_n2_n1" -- port = pol*n1*n2 + i2*n1 + i1   (default)

    The default is n2-major because the datasets never document the port layout and an
    audit had to infer it (tools/wwm_beam_port_order_audit.py): scoring every candidate
    permutation by how well the first beam index tracks the azimuth angle-of-departure
    and the second tracks elevation, "pol_n2_n1" won (0.261 / 0.257) over the previously
    assumed "pol_n1_n2" (0.199 / 0.212). Correlations are weak for every permutation
    because the channel is dense urban multipath, not a single plane wave, so this is a
    ranking between candidates rather than a confirmation of one.
    """
    if port_order not in ("pol_n1_n2", "pol_n2_n1"):
        raise ValueError("unknown port_order: %s" % port_order)
    ports = n1 * n2
    l = torch.arange(o1 * n1, device=device, dtype=torch.float32)
    m = torch.arange(o2 * n2, device=device, dtype=torch.float32)
    a1 = torch.arange(n1, device=device, dtype=torch.float32)
    a2 = torch.arange(n2, device=device, dtype=torch.float32)

    # v_{l,m}[i,j] = exp(2pi j l i /(o1 n1)) * exp(2pi j m j /(o2 n2))
    ph1 = 2.0 * math.pi * l[:, None] * a1[None, :] / float(o1 * n1)  # [L, n1]
    ph2 = 2.0 * math.pi * m[:, None] * a2[None, :] / float(o2 * n2)  # [M, n2]
    e1 = torch.polar(torch.ones_like(ph1), ph1)
    e2 = torch.polar(torch.ones_like(ph2), ph2)
    v = e1[:, None, :, None] * e2[None, :, None, :]  # [L, M, n1, n2]
    if port_order == "pol_n2_n1":
        # flatten n2-major so the codeword matches port = pol*n1*n2 + i2*n1 + i1
        v = v.permute(0, 1, 3, 2)
    v = v.reshape(-1, ports)  # [L*M, ports]

    # co-phasing phi_n = exp(i pi n / 2), n in {0,1,2,3}
    n = torch.arange(4, device=device, dtype=torch.float32)
    phi = torch.polar(torch.ones_like(n), 0.5 * math.pi * n)  # [4]

    w = torch.cat(
        [
            v[:, None, :].expand(-1, 4, -1),
            (v[:, None, :] * phi[None, :, None]),
        ],
        dim=-1,
    )  # [L*M, 4, 2*ports]
    w = w.reshape(-1, 2 * ports) / math.sqrt(2.0 * ports)

    li, mi = torch.meshgrid(l.long(), m.long(), indexing="ij")
    idx = torch.stack(
        [
            li.reshape(-1)[:, None].expand(-1, 4).reshape(-1),
            mi.reshape(-1)[:, None].expand(-1, 4).reshape(-1),
            n.long()[None, :].expand(l.numel() * m.numel(), -1).reshape(-1),
        ],
        dim=-1,
    )
    return w.to(torch.complex64), idx


def normalized_buckets(
    pred_idx,
    true_idx,
    index_table,
    o1: int,
    o2: int,
    n1: int = 4,
    n2: int = 4,
):
    """Classify beam errors in units of ORTHOGONAL beamwidths, not raw grid steps.

    A raw index distance is not comparable across codebooks: with (O1, O2) = (4, 4) a
    step of 1 in l is a quarter of a beamwidth and costs almost nothing, while with
    O2 = 1 a step of 1 in m is a fully orthogonal beam and costs nearly everything.
    Bucketing on dl + dm therefore mixed the two and produced the contradiction that
    motivated this function: "near beam" errors appeared to retain only 0.31 of the
    optimal gain even though an audit measured 0.89-0.97 for a genuine one-grid-step
    move. Normalising by the oversampling factor removes the artefact.

    Returns (tag, ul, um) where ul = dl / o1 and um = dm / o2 are displacements in
    orthogonal beamwidths.
    """
    import numpy as _np

    pl, pm, pn = _np.asarray(index_table)[_np.asarray(pred_idx)].T
    tl, tm, tn = _np.asarray(index_table)[_np.asarray(true_idx)].T
    l_span, m_span = o1 * n1, o2 * n2
    dl = _np.abs(pl - tl)
    dl = _np.minimum(dl, l_span - dl)
    dm = _np.abs(pm - tm)
    dm = _np.minimum(dm, m_span - dm)
    ul = dl / float(o1)
    um = dm / float(o2)
    rad = _np.maximum(ul, um)
    same_beam = (dl == 0) & (dm == 0)
    tag = _np.full(len(pl), "far", dtype=object)
    tag[rad <= 2.0] = "two_beam"
    tag[rad <= 1.0] = "one_beam"
    tag[rad < 1.0] = "sub_beam"          # inside one orthogonal beamwidth
    tag[same_beam & (pn != tn)] = "cophase_only"
    tag[same_beam & (pn == tn)] = "exact"
    return tag, ul, um


def type1_beam_power(
    h_raw_t16: torch.Tensor,
    n1: int = 4,
    n2: int = 4,
    o1: int = 4,
    o2: int = 4,
    codebook: torch.Tensor | None = None,
    port_order: str = "pol_n2_n1",
) -> torch.Tensor:
    """|h^H w|^2 summed over subcarriers, for every Type-I codeword.

    h_raw_t16: [B, 2, 32, S] real/imag stacked raw CSI at one timestep.
    Returns [B, K] float32.
    """
    hr = h_raw_t16[:, 0].float().contiguous()
    hi = h_raw_t16[:, 1].float().contiguous()
    if codebook is None:
        codebook, _ = type1_sp_codebook(n1, n2, o1, o2, device=h_raw_t16.device,
                                        port_order=port_order)
    cb = codebook.to(h_raw_t16.device)
    # real-only matmuls (a complex einsum here previously corrupted the CUDA context)
    cr = cb.real.contiguous()
    ci = cb.imag.contiguous()
    # conjugate transpose: h^H w -> sum_a conj(h_a) w_a
    real = torch.einsum("bas,ka->bks", hr, cr) + torch.einsum("bas,ka->bks", hi, ci)
    imag = torch.einsum("bas,ka->bks", hr, ci) - torch.einsum("bas,ka->bks", hi, cr)
    return (real.square() + imag.square()).sum(dim=-1).float()
