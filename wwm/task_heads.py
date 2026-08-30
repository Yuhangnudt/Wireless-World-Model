from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import init_weights, unpatchify_csi_tokens


class AttentivePool(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1),
        )
        self.apply(init_weights)

    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.score(tokens).squeeze(-1)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class BeamPredictionHead(nn.Module):
    """Beam classifier over attentive-pooled backbone tokens.

    Two readout variants, selected by `variant`:

    - "paper" (default): AttentivePool -> single Linear. This matches the WWM paper,
      Supplementary Note 3: "a task-specific 1-layer attentive classifier ... only the
      attentive classifier parameters are updated, while the WWM backbone remains
      fixed". Use this whenever a number is to be compared against the paper.
    - "mlp": AttentivePool -> LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear.
      Beam selection is a quadratic function of CSI (|DFT(h)|^2), so one linear layer
      underfits: on identical frozen tokens the linear readout reached top1 ~0.57 while
      the MLP reached ~0.67 (raw-frame ceiling ~0.91). This is OUR improvement, and it
      deviates from the paper's protocol -- report it separately, never as a
      reproduction.
    """

    def __init__(self, latent_dim: int, num_beams: int, variant: str = "paper") -> None:
        super().__init__()
        if variant not in ("paper", "mlp"):
            raise ValueError("unknown beam head variant: %s" % variant)
        self.variant = variant
        self.pool = AttentivePool(latent_dim)
        if variant == "paper":
            self.classifier = nn.Sequential(nn.Linear(latent_dim, int(num_beams)))
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, int(num_beams)),
            )
        self.classifier.apply(init_weights)

    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.classifier(self.pool(tokens, padding_mask))


class LocalizationHead(nn.Module):
    """Attentive 2D localization head with optional non-coordinate conditioning."""

    def __init__(self, latent_dim: int, out_dim: int = 2, conditioning_dim: int = 0) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.pool = AttentivePool(latent_dim)
        self.regressor = nn.Sequential(
            nn.LayerNorm(latent_dim + self.conditioning_dim),
            nn.Linear(latent_dim + self.conditioning_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.out_dim),
        )
        self.regressor.apply(init_weights)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooled = self.pool(tokens, padding_mask)
        if self.conditioning_dim:
            if conditioning is None:
                raise ValueError("Localization conditioning is required")
            if conditioning.ndim == 1:
                conditioning = conditioning.unsqueeze(1)
            expected = (tokens.shape[0], self.conditioning_dim)
            if tuple(conditioning.shape) != expected:
                raise ValueError("Expected localization conditioning %s, got %s" % (expected, tuple(conditioning.shape)))
            pooled = torch.cat([pooled, conditioning.to(dtype=pooled.dtype)], dim=1)
        elif conditioning is not None:
            raise ValueError("This LocalizationHead was created without conditioning")
        return self.regressor(pooled)


class MuLawQuantizer(nn.Module):
    def __init__(self, bits: int = 4, mu: float = 255.0) -> None:
        super().__init__()
        self.bits = int(bits)
        self.mu = float(mu)
        self.levels = 2 ** self.bits

    def forward(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        bounded = torch.tanh(values)
        companded = torch.sign(bounded) * torch.log1p(self.mu * torch.abs(bounded)) / math.log1p(self.mu)
        codes = torch.round((companded + 1.0) * 0.5 * (self.levels - 1)).clamp(0, self.levels - 1)
        quantized = codes / float(self.levels - 1) * 2.0 - 1.0
        straight_through = companded + (quantized - companded).detach()
        expanded = torch.sign(straight_through) * torch.expm1(torch.abs(straight_through) * math.log1p(self.mu)) / self.mu
        return {"values": expanded, "codes": codes.to(torch.int64)}


class CSICompressionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        bottleneck_dim: int,
        total_steps: int,
        patch_t: int,
        patch_h: int,
        patch_w: int,
        quantization_bits: int = 4,
        input_tokens: Optional[int] = None,
        compressed_tokens: Optional[int] = None,
        decoder_layers: int = 2,
        decoder_heads: int = 6,
        decoder_ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.total_steps = int(total_steps)
        self.patch_t = int(patch_t)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.input_tokens = int(
            input_tokens
            if input_tokens is not None
            else (self.total_steps // self.patch_t) * (32 // self.patch_h) * (32 // self.patch_w)
        )
        self.compressed_tokens = int(compressed_tokens if compressed_tokens is not None else self.input_tokens)
        self.bottleneck_dim = int(bottleneck_dim)
        self.quantization_bits = int(quantization_bits)
        self.reduce = nn.Linear(latent_dim, int(bottleneck_dim))
        self.token_mix = nn.Linear(self.input_tokens, self.compressed_tokens)
        self.quantizer = MuLawQuantizer(bits=quantization_bits)
        self.token_unmix = nn.Linear(self.compressed_tokens, self.input_tokens)
        self.expand = nn.Linear(int(bottleneck_dim), latent_dim)
        self.position = nn.Parameter(torch.zeros(1, self.input_tokens, latent_dim))
        self.decoder_scale = nn.Parameter(torch.tensor(0.1))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=int(decoder_heads),
            dim_feedforward=latent_dim * int(decoder_ffn_mult),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=int(decoder_layers))
        self.norm = nn.LayerNorm(latent_dim)
        self.to_patch = nn.Linear(latent_dim, 2 * self.patch_t * self.patch_h * self.patch_w)
        self.apply(init_weights)
        nn.init.zeros_(self.position)
        self._init_token_mixing()

    def _init_token_mixing(self) -> None:
        with torch.no_grad():
            self.token_mix.weight.zero_()
            self.token_unmix.weight.zero_()
            self.token_mix.bias.zero_()
            self.token_unmix.bias.zero_()
            if self.compressed_tokens == self.input_tokens:
                self.token_mix.weight.copy_(torch.eye(self.input_tokens))
                self.token_unmix.weight.copy_(torch.eye(self.input_tokens))
                return
            if self.input_tokens % self.compressed_tokens != 0:
                raise ValueError("input_tokens must be divisible by compressed_tokens")
            group = self.input_tokens // self.compressed_tokens
            for output_index in range(self.compressed_tokens):
                start = output_index * group
                self.token_mix.weight[output_index, start : start + group] = 1.0 / float(group)
                self.token_unmix.weight[start : start + group, output_index] = 1.0

    @property
    def payload_bits(self) -> int:
        return self.compressed_tokens * self.bottleneck_dim * self.quantization_bits

    @property
    def compression_ratio(self) -> float:
        source_bits = self.total_steps * 2 * 32 * 32 * 32
        return float(source_bits) / float(self.payload_bits)

    def forward(self, csi_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if csi_tokens.ndim != 3 or csi_tokens.shape[1] != self.input_tokens:
            raise ValueError(
                "Expected CSI tokens [B, %d, D], got %s"
                % (self.input_tokens, tuple(csi_tokens.shape))
            )
        compressed = self.reduce(csi_tokens)
        mixed = self.token_mix(compressed.transpose(1, 2)).transpose(1, 2).contiguous()
        quantized = self.quantizer(mixed)
        unmixed = self.token_unmix(quantized["values"].transpose(1, 2)).transpose(1, 2).contiguous()
        restored_tokens = self.expand(unmixed)
        decoder_input = restored_tokens + self.position.to(restored_tokens.dtype)
        transformed_tokens = self.decoder(decoder_input)
        scale = self.decoder_scale.to(dtype=restored_tokens.dtype)
        decoded_tokens = decoder_input + scale * (transformed_tokens - decoder_input)
        patch = self.to_patch(self.norm(decoded_tokens))
        reconstructed_h = unpatchify_csi_tokens(
            patch,
            self.total_steps,
            self.patch_t,
            self.patch_h,
            self.patch_w,
        )
        return {
            "pred_h": reconstructed_h,
            "codes": quantized["codes"],
            "compressed": compressed,
            "mixed": mixed,
            "restored_tokens": restored_tokens,
            "payload_bits": torch.tensor(self.payload_bits, device=csi_tokens.device),
        }
