"""A beam readout that can express coherent combining.

Why the paper's head cannot. Beam power is
    P[k] = sum_s | sum_a conj(w_k[a]) H[a, s] |^2
-- a COHERENT (complex, phase-aligned) sum over antennas inside the magnitude, then an
incoherent sum over subcarriers. The paper's "1-layer attentive classifier" pools tokens
with learned real scalars and only then applies a linear map, so the nonlinearity sits
AFTER the sum. A scalar-weighted mean followed by a linear layer cannot represent
|complex sum|^2, which is why a head fed the RAW patches -- which contain the complete CSI
-- reaches only 0.7074 amplitude gain while an analytic DFT on the same CSI reaches 0.9766.

This head restores the correct order of operations:
    per-antenna-group complex projection -> coherent sum over antenna groups
    -> |.|^2 -> incoherent sum over subcarrier and time groups.

Token layout it relies on (verified against wwm.common.patchify_csi_tokens): CSI tokens are
ordered (t_grid, h_grid, w_grid); h_seq is [B, T, 2, ant, sc] so patch_h splits the ANTENNA
axis and patch_w the SUBCARRIER axis. Hence 512 visible beam tokens reshape to
(8 time, 8 antenna, 8 subcarrier) and the coherent sum must run over the antenna axis.

The projection is NOT shared across antenna groups on purpose: with a shared W the sum over
groups collapses to "pool then project", which is exactly the degenerate case being fixed.
Each group gets its own weights so the head can learn per-sub-aperture phase alignment,
the learned analogue of a DFT steering vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoherentBeamHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_beams: int,
        t_groups: int = 8,
        ant_groups: int = 8,
        sc_groups: int = 8,
        share_time: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_beams = int(num_beams)
        self.t_groups = int(t_groups)
        self.ant_groups = int(ant_groups)
        self.sc_groups = int(sc_groups)
        self.share_time = bool(share_time)
        # NO per-token LayerNorm. A sanity test that planted exact DFT weights showed the
        # head then matched an analytic beamformer with correlation 0.85 instead of 1.0:
        # LayerNorm rescales every (t, antenna, subcarrier) token independently, which
        # destroys the cross-antenna relative amplitude that coherent combining depends on
        # -- the same defect this head exists to fix. A single global affine keeps the
        # logits in range without touching relative amplitude.
        self.in_scale = nn.Parameter(torch.ones(1))
        self.in_shift = nn.Parameter(torch.zeros(1))
        # real and imaginary weights per antenna group: [A, D, K] each
        self.wr = nn.Parameter(torch.randn(ant_groups, latent_dim, num_beams) * latent_dim ** -0.5)
        self.wi = nn.Parameter(torch.randn(ant_groups, latent_dim, num_beams) * latent_dim ** -0.5)
        # a learned log-scale keeps the logits in a sane range for cross-entropy
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(num_beams))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = tokens.shape
        expect = self.t_groups * self.ant_groups * self.sc_groups
        if n != expect:
            raise ValueError("expected %d CSI tokens, got %d" % (expect, n))
        x = (tokens * self.in_scale + self.in_shift).reshape(
            b, self.t_groups, self.ant_groups, self.sc_groups, d)
        # complex projection, per antenna group
        re = torch.einsum("btasd,adk->btask", x, self.wr)
        im = torch.einsum("btasd,adk->btask", x, self.wi)
        # coherent sum over antenna groups, then power
        power = re.sum(2).square() + im.sum(2).square()          # [B, T, S, K]
        # incoherent sum over subcarrier groups and time groups
        logits = power.sum(dim=(1, 2)) * self.log_scale.exp() + self.bias
        return torch.log1p(logits.clamp_min(0.0))


class CoherentPatchBeamHead(nn.Module):
    """Same operation directly on raw patch vectors (2*pt*ph*pw wide, no learned embedding).

    Included as the strongest test of the readout hypothesis: raw patches provably contain
    the full CSI, so if this reaches the analytic 0.9766 the bottleneck was never the
    backbone or the tokenizer.
    """

    def __init__(self, patch_dim: int, num_beams: int, t_groups: int = 8,
                 ant_groups: int = 8, sc_groups: int = 8) -> None:
        super().__init__()
        self.core = CoherentBeamHead(patch_dim, num_beams, t_groups, ant_groups, sc_groups)

    def forward(self, patches: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.core(patches)
