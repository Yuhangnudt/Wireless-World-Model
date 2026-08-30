from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

from .common import (init_weights, axis_sincos, sincos_1d, sincos_3d,
    patchify_csi_tokens, unpatchify_csi_tokens)
from .em_physics import (EMKernelCache, batch_speed_from_traj, build_em_kernel,
    center_time, context_alpha, empirical_deltar_target, kernel_diagnostics, physical_relation_loss,
    shuffle_kernel_across_speed, shuffle_target_within_speed_bins, truncated_whitener,
    whitener_buckets_from_cache)
from .metrics import sgcs_metric_at_step
from .pointbert import (farthest_point_sample, PointBERTDiscreteVAE)
from .sigreg import sigreg_loss
from .visreg import visreg_loss


class CSITubeletEmbed(nn.Module):
    def __init__(self, latent_dim: int, patch_t: int, patch_h: int, patch_w: int, total_steps: int) -> None:
        super().__init__()
        if total_steps % patch_t != 0 or 32 % patch_h != 0 or 32 % patch_w != 0:
            raise ValueError("Patch sizes must exactly divide CSI shape.")
        self.latent_dim = int(latent_dim)
        self.patch_t = int(patch_t)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.grid = (total_steps // patch_t, 32 // patch_h, 32 // patch_w)
        self.proj = nn.Conv3d(
            2,
            latent_dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.norm = nn.LayerNorm(latent_dim)

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2]

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        x = h_seq.permute(0, 2, 1, 3, 4).contiguous()
        y = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        pos = sincos_3d(self.grid, self.latent_dim, y.device).to(y.dtype)
        return self.norm(y + pos.unsqueeze(0))


class PointPatchEmbed(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        point_tokens: int,
        group_size: int,
        center_sampling: str,
        tokenizer: str,
        dvae_encoder_dims: int,
        dvae_codebook_size: int,
        dvae_codebook_dim: int,
        dvae_decoder_dims: int,
        dvae_temperature: float,
        dvae_hard: bool,
        dvae_token_source: str,
        freeze_dvae: bool,
        encode_centers: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.point_tokens = int(point_tokens)
        self.group_size = int(group_size)
        self.center_sampling = str(center_sampling)
        self.tokenizer = str(tokenizer)
        self.dvae_temperature = float(dvae_temperature)
        self.dvae_hard = bool(dvae_hard)
        self.dvae_feature_source = str(dvae_token_source)
        self.encode_centers = bool(encode_centers)
        self.center_proj = nn.Sequential(
            nn.Linear(3, int(latent_dim)),
            nn.SiLU(),
            nn.Linear(int(latent_dim), int(latent_dim)),
        )
        self.dvae: Optional[PointBERTDiscreteVAE] = None
        if self.tokenizer == "pointbert_dvae":
            self.dvae = PointBERTDiscreteVAE(
                point_tokens=point_tokens,
                group_size=group_size,
                center_sampling=center_sampling,
                encoder_dims=dvae_encoder_dims,
                codebook_size=dvae_codebook_size,
                codebook_dim=dvae_codebook_dim,
                decoder_dims=dvae_decoder_dims,
            )
            if self.dvae_feature_source == "encoder":
                dvae_feature_dim = int(dvae_encoder_dims)
            elif self.dvae_feature_source == "sampled":
                dvae_feature_dim = int(dvae_codebook_dim)
            elif self.dvae_feature_source == "refined":
                dvae_feature_dim = int(dvae_decoder_dims)
            else:
                raise ValueError("Unknown point dVAE token source: %s" % self.dvae_feature_source)
            self.token_proj = nn.Linear(dvae_feature_dim, int(latent_dim))
            self.token_norm = nn.LayerNorm(int(latent_dim))
            self.set_dvae_frozen(freeze_dvae)
            return
        if self.tokenizer != "pointnet":
            raise ValueError("Unknown point tokenizer: %s" % self.tokenizer)
        hidden = max(64, latent_dim // 2)
        self.local_mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.out = nn.LayerNorm(latent_dim)

    def set_dvae_frozen(self, frozen: bool) -> None:
        if self.dvae is None:
            return
        self.dvae.train(not frozen)
        for p in self.dvae.parameters():
            p.requires_grad = not frozen

    def forward(self, points: Any) -> torch.Tensor:
        point_input, point_origin, point_scale = self._unpack_point_input(points)
        if self.tokenizer == "pointbert_dvae":
            if self.dvae is None:
                raise RuntimeError("Point-BERT dVAE tokenizer is not initialized.")
            if not any(p.requires_grad for p in self.dvae.parameters()):
                self.dvae.eval()
                with torch.no_grad():
                    _, aux = self.dvae.encode_tokens(
                        point_input,
                        temperature=self.dvae_temperature,
                        hard=self.dvae_hard,
                        deterministic=True,
                    )
                    features = self._select_dvae_features(aux)
                return self._project_tokens(features, aux["center"], point_origin, point_scale)
            _, aux = self.dvae.encode_tokens(point_input, temperature=self.dvae_temperature, hard=self.dvae_hard)
            features = self._select_dvae_features(aux)
            return self._project_tokens(features, aux["center"], point_origin, point_scale)
        if not torch.is_tensor(point_input):
            raise TypeError("PointNet tokenizer requires a point tensor.")
        b, n, _ = point_input.shape
        if n < self.point_tokens:
            raise ValueError("point_count must be >= point_tokens")
        if self.center_sampling == "linspace":
            center_idx = torch.linspace(0, n - 1, self.point_tokens, device=points.device).round().long()
            center_idx = center_idx.unsqueeze(0).expand(b, -1)
        elif self.center_sampling == "fps":
            center_idx = farthest_point_sample(point_input.float(), self.point_tokens)
        else:
            raise ValueError("Unknown point center sampling: %s" % self.center_sampling)
        center_gather = center_idx.unsqueeze(-1).expand(b, self.point_tokens, 3)
        centers = torch.gather(point_input, 1, center_gather)
        # cdist is run in fp32 for stable nearest-neighbor grouping under AMP.
        dist = torch.cdist(centers.float(), point_input.float())
        k = min(self.group_size, n)
        knn = torch.topk(dist, k=k, dim=-1, largest=False).indices
        gather_points = point_input[:, None, :, :].expand(b, self.point_tokens, n, 3)
        gather_idx = knn.unsqueeze(-1).expand(b, self.point_tokens, k, 3)
        neigh = torch.gather(gather_points, 2, gather_idx)
        rel = neigh - centers[:, :, None, :]
        feat = torch.cat([rel, centers[:, :, None, :].expand_as(rel)], dim=-1)
        x = self.local_mlp(feat)
        features = self.out(x.max(dim=2).values)
        if not self.encode_centers:
            return features
        common_centers = self._common_centers(centers, point_origin, point_scale)
        return self.out(features + self.center_proj(common_centers.to(dtype=features.dtype)))

    @staticmethod
    def _unpack_point_input(points: Any) -> Tuple[Any, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(points, dict):
            return points, None, None
        origin = points.get("point_origin")
        scale = points.get("point_scale")
        if "points" in points:
            return points["points"], origin, scale
        if "point_group" in points:
            return points["point_group"], origin, scale
        if "neighborhood" in points and "center" in points:
            return {"neighborhood": points["neighborhood"], "center": points["center"]}, origin, scale
        raise KeyError("Point input dict needs points, point_group, or neighborhood/center.")

    @staticmethod
    def _common_centers(
        centers: torch.Tensor,
        origin: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if origin is None or scale is None:
            return centers
        return origin.to(dtype=centers.dtype).unsqueeze(1) + centers * scale.to(dtype=centers.dtype).reshape(-1, 1, 1)

    def _project_tokens(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
        origin: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tokens = self.token_proj(features)
        if self.encode_centers:
            common_centers = self._common_centers(centers, origin, scale)
            tokens = tokens + self.center_proj(common_centers.to(dtype=tokens.dtype))
        return self.token_norm(tokens)

    def _select_dvae_features(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.dvae_feature_source == "encoder":
            return aux["encoder_tokens"]
        if self.dvae_feature_source == "sampled":
            return aux["sampled"]
        if self.dvae_feature_source == "refined":
            return aux["refined"]
        raise ValueError("Unknown point dVAE token source: %s" % self.dvae_feature_source)


class TrajectoryEmbed(nn.Module):
    def __init__(self, latent_dim: int, total_steps: int, input_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.total_steps = int(total_steps)
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        y = self.net(traj)
        pos = sincos_1d(self.total_steps, self.latent_dim, y.device).to(y.dtype)
        return y + pos.unsqueeze(0)


class TokenStem(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        total_steps: int,
        patch_t: int,
        patch_h: int,
        patch_w: int,
        point_tokens: int,
        point_group_size: int,
        point_center_sampling: str,
        point_tokenizer: str,
        point_dvae_encoder_dims: int,
        point_dvae_codebook_size: int,
        point_dvae_codebook_dim: int,
        point_dvae_decoder_dims: int,
        point_dvae_temperature: float,
        point_dvae_hard: bool,
        point_dvae_token_source: str,
        freeze_point_dvae: bool,
        point_center_encoding: bool,
        trajectory_features_mode: str,
        context_rms_feature: bool = False,
    ) -> None:
        super().__init__()
        self.csi = CSITubeletEmbed(latent_dim, patch_t, patch_h, patch_w, total_steps)
        self.point = PointPatchEmbed(
            latent_dim=latent_dim,
            point_tokens=point_tokens,
            group_size=point_group_size,
            center_sampling=point_center_sampling,
            tokenizer=point_tokenizer,
            dvae_encoder_dims=point_dvae_encoder_dims,
            dvae_codebook_size=point_dvae_codebook_size,
            dvae_codebook_dim=point_dvae_codebook_dim,
            dvae_decoder_dims=point_dvae_decoder_dims,
            dvae_temperature=point_dvae_temperature,
            dvae_hard=point_dvae_hard,
            dvae_token_source=point_dvae_token_source,
            freeze_dvae=freeze_point_dvae,
            encode_centers=point_center_encoding,
        )
        traj_input_dim = 3 if trajectory_features_mode == "pos" else 6
        if context_rms_feature:
            traj_input_dim += 1  # standardized log10(context_rms) appended in data.py
        self.traj = TrajectoryEmbed(latent_dim, total_steps, traj_input_dim)
        self.csi_modality = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.point_modality = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.traj_modality = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.total_steps = int(total_steps)
        self.latent_dim = int(latent_dim)
        self.apply(init_weights)
        for p in (self.csi_modality, self.point_modality, self.traj_modality):
            nn.init.trunc_normal_(p, std=0.02)

    def lengths(self) -> Tuple[int, int, int]:
        return (self.csi.num_tokens, self.point.point_tokens, self.total_steps)

    def forward(self, h_seq: torch.Tensor, point_cloud: torch.Tensor, traj: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        csi = self.csi(h_seq) + self.csi_modality
        point = self.point(point_cloud) + self.point_modality
        traj_tokens = self.traj(traj) + self.traj_modality
        return torch.cat([csi, point, traj_tokens], dim=1), self.lengths()

    def mask_template(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        nc, np_, nt = self.lengths()
        csi_pos = sincos_3d(self.csi.grid, self.latent_dim, device).to(dtype) + self.csi_modality.to(dtype).squeeze(0)
        # Full point masking needs distinct queries for the 256 deterministic FPS
        # slots. Identical point mask tokens are permutation-equivariant and would
        # force every predicted point latent to be identical.
        pc = sincos_1d(np_, self.latent_dim, device).to(dtype) + self.point_modality.to(dtype).squeeze(0)
        traj_pos = sincos_1d(nt, self.latent_dim, device).to(dtype) + self.traj_modality.to(dtype).squeeze(0)
        template = torch.cat([csi_pos, pc, traj_pos], dim=0)
        return template.unsqueeze(0).expand(batch_size, -1, -1)


class FeedForward(nn.Module):
    def __init__(self, latent_dim: int, mult: int) -> None:
        super().__init__()
        hidden = int(latent_dim) * int(mult)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityMoEBlock(nn.Module):
    """MoE transformer block with a SHARED self-attention and per-modality FFNs.

    Change A (modality_layernorm="per_modality", default): the pre-attention and
    pre-FFN LayerNorms are split per modality (CSI / point / trajectory), matching
    the Multiway Transformer design (VLMo / BEiT-3). Each modality is normalized by
    its own activation statistics before projecting into the shared Q/K/V space, so
    cross-modal fusion (the shared attention) is preserved — only the normalization
    is modality-specific. Set "shared" to recover the single-LayerNorm baseline.
    """

    def __init__(self, latent_dim: int, heads: int, ffn_mult: int, modality_layernorm: str = "per_modality") -> None:
        super().__init__()
        self.modality_layernorm = str(modality_layernorm)
        if self.modality_layernorm == "per_modality":
            self.norm_attn_csi = nn.LayerNorm(latent_dim)
            self.norm_attn_point = nn.LayerNorm(latent_dim)
            self.norm_attn_traj = nn.LayerNorm(latent_dim)
            self.norm_ffn_csi = nn.LayerNorm(latent_dim)
            self.norm_ffn_point = nn.LayerNorm(latent_dim)
            self.norm_ffn_traj = nn.LayerNorm(latent_dim)
        elif self.modality_layernorm == "shared":
            self.norm_attn = nn.LayerNorm(latent_dim)
            self.norm_ffn = nn.LayerNorm(latent_dim)
        else:
            raise ValueError("Unknown modality_layernorm: %s" % self.modality_layernorm)
        self.attn = nn.MultiheadAttention(latent_dim, heads, dropout=0.0, batch_first=True)
        self.csi_ffn = FeedForward(latent_dim, ffn_mult)
        self.point_ffn = FeedForward(latent_dim, ffn_mult)
        self.traj_ffn = FeedForward(latent_dim, ffn_mult)

    def _norm(self, x: torch.Tensor, lengths: Tuple[int, int, int], which: str) -> torch.Tensor:
        if self.modality_layernorm == "shared":
            return (self.norm_attn if which == "attn" else self.norm_ffn)(x)
        nc, np_, nt = lengths
        csi, point, traj = torch.split(x, [nc, np_, nt], dim=1)
        if which == "attn":
            return torch.cat([self.norm_attn_csi(csi), self.norm_attn_point(point), self.norm_attn_traj(traj)], dim=1)
        return torch.cat([self.norm_ffn_csi(csi), self.norm_ffn_point(point), self.norm_ffn_traj(traj)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Tuple[int, int, int],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self._norm(x, lengths, "attn")
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        h = self._norm(x, lengths, "ffn")
        nc, np_, nt = lengths
        csi, point, traj = torch.split(h, [nc, np_, nt], dim=1)
        routed = torch.cat([self.csi_ffn(csi), self.point_ffn(point), self.traj_ffn(traj)], dim=1)
        return x + routed


class MMoETransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        layers: int,
        heads: int,
        ffn_mult: int,
        checkpoint_activations: bool,
        modality_layernorm: str = "per_modality",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ModalityMoEBlock(latent_dim, heads, ffn_mult, modality_layernorm) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.checkpoint_activations = bool(checkpoint_activations)
        self.apply(init_weights)

    @staticmethod
    def intermediate_indices(num_blocks: int, num_levels: int, selection: str = "last") -> List[int]:
        """Return deterministic block indices for deep self-supervision.

        ``uniform`` mirrors V-JEPA 2.1's equally spaced encoder taps.  For a
        12-block encoder with four levels it returns [2, 5, 8, 11], i.e. blocks
        3/6/9/12 in one-based notation.  ``last`` preserves the historical WWM
        behaviour and returns the final consecutive blocks.
        """
        count = min(max(int(num_levels), 0), int(num_blocks))
        if count <= 0:
            return []
        if selection == "last":
            return list(range(int(num_blocks) - count, int(num_blocks)))
        if selection != "uniform":
            raise ValueError("Unknown intermediate layer selection: %s" % selection)
        return [
            int(math.ceil(float(level + 1) * float(num_blocks) / float(count))) - 1
            for level in range(count)
        ]

    def forward(
        self,
        x: torch.Tensor,
        lengths: Tuple[int, int, int],
        key_padding_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        num_intermediate_layers: int = 4,
        intermediate_selection: str = "last",
    ) -> Any:
        selected: List[torch.Tensor] = []
        selected_indices = set(self.intermediate_indices(
            len(self.blocks), num_intermediate_layers, intermediate_selection
        ))
        for block_index, block in enumerate(self.blocks):
            if self.training and self.checkpoint_activations:
                x = checkpoint(lambda y, b=block: b(y, lengths, key_padding_mask), x, use_reentrant=False)
            else:
                x = block(x, lengths, key_padding_mask=key_padding_mask)
            if return_intermediates and block_index in selected_indices:
                selected.append(self.norm(x))
        output = self.norm(x)
        if not return_intermediates:
            return output
        if selected:
            selected[-1] = output
        return output, selected


class ResidualMultiLevelFusionMLP(nn.Module):
    """V-JEPA 2.1-style multi-level fusion with a stable warm-start path.

    The learned branch sees the channel-wise concatenation of uniformly sampled
    encoder levels.  A residual connection preserves the final encoder level,
    while a small learnable scale prevents a freshly initialized fusion MLP from
    abruptly changing the predictor input distribution of a resumed checkpoint.
    """

    def __init__(self, latent_dim: int, num_levels: int, hidden_mult: float, residual_scale: float) -> None:
        super().__init__()
        input_dim = int(latent_dim) * int(num_levels)
        hidden_dim = max(int(round(float(latent_dim) * float(hidden_mult))), int(latent_dim))
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, int(latent_dim))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.apply(init_weights)

    def forward(
        self,
        concatenated: torch.Tensor,
        final_level: torch.Tensor,
        residual_multiplier: float = 1.0,
    ) -> torch.Tensor:
        update = self.fc2(self.act(self.fc1(self.norm(concatenated))))
        scale = self.residual_scale.to(dtype=update.dtype) * float(residual_multiplier)
        return final_level + scale * update


class CSITubeletDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        future_steps: int,
        patch_t: int,
        patch_h: int,
        patch_w: int,
        layers: int,
        heads: int,
        ffn_mult: int,
    ) -> None:
        super().__init__()
        if future_steps % patch_t != 0:
            raise ValueError("future_steps must be divisible by patch_t.")
        self.future_steps = int(future_steps)
        self.patch_t = int(patch_t)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.future_t_grid = future_steps // patch_t
        self.h_grid = 32 // patch_h
        self.w_grid = 32 // patch_w
        self.num_tokens = self.future_t_grid * self.h_grid * self.w_grid
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=latent_dim * ffn_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(latent_dim)
        self.to_patch = nn.Linear(latent_dim, 2 * patch_t * patch_h * patch_w)
        self.apply(init_weights)

    def forward(self, csi_tokens: torch.Tensor, future_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if future_token_mask is None:
            if csi_tokens.shape[1] != self.num_tokens:
                raise ValueError("Expected %d future CSI tokens, got %d" % (self.num_tokens, csi_tokens.shape[1]))
            x = self.blocks(csi_tokens)
        else:
            if future_token_mask.shape != csi_tokens.shape[:2]:
                raise ValueError(
                    "future_token_mask shape %s does not match CSI tokens %s"
                    % (tuple(future_token_mask.shape), tuple(csi_tokens.shape[:2]))
                )
            x_all = self.blocks(csi_tokens)
            future_counts = future_token_mask.sum(dim=1)
            if not bool(torch.all(future_counts == self.num_tokens).item()):
                raise ValueError("Expected %d future CSI tokens per sample, got %s" % (self.num_tokens, future_counts.tolist()))
            x = x_all[future_token_mask].reshape(csi_tokens.shape[0], self.num_tokens, csi_tokens.shape[-1])
        patch = self.to_patch(self.norm(x))
        b = patch.shape[0]
        patch = patch.reshape(
            b,
            self.future_t_grid,
            self.h_grid,
            self.w_grid,
            2,
            self.patch_t,
            self.patch_h,
            self.patch_w,
        )
        patch = patch.permute(0, 1, 5, 4, 2, 6, 3, 7).contiguous()
        return patch.reshape(b, self.future_steps, 2, 32, 32)


class PaperWWM(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        total_steps = int(args.context_steps + args.future_steps)
        self.latent_dim = int(args.latent_dim)
        self.context_steps = int(args.context_steps)
        self.future_steps = int(args.future_steps)
        self.patch_t = int(args.patch_t)
        self.patch_h = int(args.patch_h)
        self.patch_w = int(args.patch_w)
        self.temporal_anchor = str(args.temporal_anchor)
        self.anchor_delta_scale = float(args.anchor_delta_scale)
        self.csi_transform = str(args.csi_transform)
        self.signed_log_eps = float(args.signed_log_eps)
        self.inverse_signed_log_clip = float(getattr(args, "inverse_signed_log_clip", 12.0))
        self.csi_mean = float(args.csi_mean)
        self.csi_std = float(args.csi_std)
        self.encoder_visible_mode = str(args.encoder_visible_mode)
        self.decoder_token_input = str(args.decoder_token_input)
        self.online_stem = TokenStem(
            latent_dim=args.latent_dim,
            total_steps=total_steps,
            patch_t=args.patch_t,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            point_tokens=args.point_tokens,
            point_group_size=args.point_group_size,
            point_center_sampling=args.point_center_sampling,
            point_tokenizer=args.point_tokenizer,
            point_dvae_encoder_dims=args.point_dvae_encoder_dims,
            point_dvae_codebook_size=args.point_dvae_codebook_size,
            point_dvae_codebook_dim=args.point_dvae_codebook_dim,
            point_dvae_decoder_dims=args.point_dvae_decoder_dims,
            point_dvae_temperature=args.point_dvae_temperature,
            point_dvae_hard=args.point_dvae_hard,
            point_dvae_token_source=args.point_dvae_token_source,
            freeze_point_dvae=args.freeze_point_dvae,
            point_center_encoding=bool(getattr(args, "point_center_encoding", True)),
            trajectory_features_mode=args.trajectory_features,
            context_rms_feature=bool(getattr(args, "context_rms_feature", False)),
        )
        modality_layernorm = str(getattr(args, "modality_layernorm", "per_modality"))
        self.online_encoder = MMoETransformer(
            latent_dim=args.latent_dim,
            layers=args.mmoe_layers,
            heads=args.mmoe_heads,
            ffn_mult=args.ffn_mult,
            checkpoint_activations=args.checkpoint_activations,
            modality_layernorm=modality_layernorm,
        )
        self.predictor = MMoETransformer(
            latent_dim=args.latent_dim,
            layers=args.predictor_layers,
            heads=args.mmoe_heads,
            ffn_mult=args.ffn_mult,
            checkpoint_activations=args.checkpoint_activations,
            modality_layernorm=modality_layernorm,
        )
        self.decoder = CSITubeletDecoder(
            latent_dim=args.latent_dim,
            future_steps=args.future_steps,
            patch_t=args.patch_t,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            layers=args.decoder_layers,
            heads=args.decoder_heads,
            ffn_mult=args.decoder_ffn_mult,
        )
        if self.temporal_anchor != "none" and bool(getattr(args, "zero_init_residual_decoder", True)):
            nn.init.zeros_(self.decoder.to_patch.weight)
            nn.init.zeros_(self.decoder.to_patch.bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.latent_dim))
        self.anchor_residual_scale = nn.Parameter(torch.tensor(float(args.anchor_residual_scale)))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Hybrid-JEPA pre-training: a per-token CSI reconstruction head decodes the
        # predictor's masked CSI tokens straight back to CSI patches. The latent
        # prediction (L1) loss alone does not guarantee the masked-position latents
        # are CSI-decodable (verified: predictor latents gave SGCS ~0.15 despite low
        # latent L1) and, with "easy" temporal masking, encourages representation
        # collapse. Adding a reconstruction loss forces each predicted masked token
        # to carry its true CSI content, which both fixes decodability and prevents
        # collapse (a collapsed latent cannot reconstruct varying CSI).
        self.pretrain_recon_weight = float(getattr(args, "pretrain_recon_weight", 0.0))
        geo_recon_weight = getattr(args, "pretrain_geo_recon_weight", None)
        self.pretrain_geo_recon_weight = (
            self.pretrain_recon_weight if geo_recon_weight is None else float(geo_recon_weight)
        )
        self.pretrain_visible_recon_weight = float(getattr(args, "pretrain_visible_recon_weight", 0.0))
        self.pretrain_recon_mag_weight = float(getattr(args, "pretrain_recon_mag_weight", 0.5))
        self.pretrain_recon_phase_weight = float(getattr(args, "pretrain_recon_phase_weight", 0.5))
        self.pretrain_phase_eps_ratio = float(getattr(args, "pretrain_phase_eps_ratio", 0.05))
        self.pretrain_sgcs_weight = float(getattr(args, "pretrain_sgcs_weight", 0.0))
        self.pretrain_residual_head = bool(getattr(args, "pretrain_residual_head", False))
        self.pretrain_csi_head = nn.Sequential(
            nn.Linear(args.latent_dim, 2 * self.patch_t * self.patch_h * self.patch_w),
        )
        self.pretrain_context_csi_head = nn.Sequential(
            nn.Linear(args.latent_dim, 2 * self.patch_t * self.patch_h * self.patch_w),
        )
        if self.pretrain_residual_head:
            # Zero-init last layer so the head starts predicting delta=0, i.e.
            # pred = anchor = copy-baseline (~0.91 SGCS). The model then only has
            # to learn the correction on top, sidestepping the encoder ceiling.
            nn.init.zeros_(self.pretrain_csi_head[-1].weight)
            nn.init.zeros_(self.pretrain_csi_head[-1].bias)

        # Angular power spectrum auxiliary head. Predicts the 32-beam DFT angular
        # power spectrum (the exact quantity the beam label argmaxes) for each
        # fully-visible tubelet time group, from mean-pooled encoder context CSI
        # tokens. A time group spans patch_t frames and (32/patch_h)*(32/patch_w)
        # tokens, so pooling recovers the full 32-antenna aperture that a single
        # token (patch_h antennas) cannot represent.
        self.pretrain_angular_weight = float(getattr(args, "pretrain_angular_weight", 0.0))
        self.pretrain_angular_warmup_steps = int(getattr(args, "pretrain_angular_warmup_steps", 0))
        self.pretrain_angular_beams = int(getattr(args, "pretrain_angular_beams", 32))
        self.pretrain_angular_head = nn.Sequential(
            nn.LayerNorm(args.latent_dim),
            nn.Linear(args.latent_dim, args.latent_dim),
            nn.GELU(),
            nn.Linear(args.latent_dim, self.pretrain_angular_beams),
        )

        # UE-position auxiliary head. Applied ONLY to samples whose point_origin was
        # dropped, so the absolute-position shortcut is unavailable and the position
        # must come from CSI + UE-relative geometry. Mirrors the angular aux head that
        # fixed beam: latent-only supervision (the `traj` mask task) never forced
        # explicit coordinate decodability, which is why the frozen backbone scored
        # 40.7 m even on training samples.
        self.pretrain_position_weight = float(getattr(args, "pretrain_position_weight", 0.0))
        self.pretrain_position_warmup_steps = int(getattr(args, "pretrain_position_warmup_steps", 0))
        self.pretrain_position_head = nn.Sequential(
            nn.LayerNorm(args.latent_dim),
            nn.Linear(args.latent_dim, args.latent_dim),
            nn.GELU(),
            nn.Linear(args.latent_dim, 3),
        )
        # Inline attentive pooling (task_heads.AttentivePool is not imported here to
        # avoid a circular import: task_heads imports from this module's package).
        self.pretrain_position_score = nn.Linear(args.latent_dim, 1)

        # Point-cloud height auxiliary head (v19). Read out of the PREDICTOR tokens at
        # the masked point positions, i.e. exactly the `point` mask task: all point
        # tokens are hidden and the predictor must recover them from CSI + trajectory.
        #
        # Why height and why here. v18 added the `point` task at sampling weight 3.0
        # but supervised it with the latent L1 objective only; on the independent
        # Frankfurt protocol it bought nothing (v18 vs v17 sparse-map CD -0.0004 m
        # [-0.0017, 0.0009], and it lost to a random frozen encoder). Probing the
        # frozen backbone showed why a *decodable* target was missing rather than the
        # information: ridge R^2 from pooled context to
        #   height-layer occupancy (7 layers)  +0.528
        #   [mean z, std z, max z, p90 z]      +0.547
        #   BEV planar occupancy               +0.139
        #   FPS centers                        +0.003
        # so vertical structure is the most linearly available geometry by ~4x, yet
        # the decoded clouds are flat sheets at z~0 because 73.8% of the ground-truth
        # points lie below z=0 and Chamfer is minimized by matching that mass. The
        # aux loss supervises the vertical mass distribution directly, which Chamfer
        # never penalizes. Layer edges are in metres relative to the UE.
        self.pretrain_height_weight = float(getattr(args, "pretrain_height_weight", 0.0))
        self.pretrain_height_warmup_steps = int(getattr(args, "pretrain_height_warmup_steps", 0))
        self.pretrain_height_stat_weight = float(getattr(args, "pretrain_height_stat_weight", 1.0))
        self.pretrain_height_scale_m = float(getattr(args, "pretrain_height_scale_m", 20.0))
        edges = [
            float(x)
            for x in str(getattr(args, "pretrain_height_edges", "0,3,6,10,15,25")).split(",")
            if str(x).strip()
        ]
        self.register_buffer(
            "pretrain_height_edges", torch.tensor(edges, dtype=torch.float32), persistent=False
        )
        self.pretrain_height_layers = len(edges) + 1
        # Built only when the term is active. Every downstream/eval script rebuilds
        # PaperWWM from the checkpoint's own args and then load_state_dict(strict=True),
        # so unconditionally adding parameters would break loading every pre-v19
        # checkpoint.
        if self.pretrain_height_weight > 0:
            self.pretrain_height_head = nn.Sequential(
                nn.LayerNorm(args.latent_dim),
                nn.Linear(args.latent_dim, args.latent_dim),
                nn.GELU(),
                nn.Linear(args.latent_dim, self.pretrain_height_layers + 3),
            )
            self.pretrain_height_score = nn.Linear(args.latent_dim, 1)

        # Change B — per-modality SIGReg (LeJEPA). Isotropic-Gaussian regularizer
        # applied per modality to the encoder (or predictor) embeddings during
        # pre-training, as an explicit anti-collapse term. EMA target is RETAINED;
        # SIGReg is additive, not a replacement. Weights are per modality so the
        # geometry of each stream (CSI / point / trajectory) is shaped independently.
        self.sigreg_enable = bool(getattr(args, "sigreg_enable", False))
        self.sigreg_apply_on = str(getattr(args, "sigreg_apply_on", "encoder_out"))
        self.sigreg_num_projections = int(getattr(args, "sigreg_num_projections", 512))
        self.sigreg_weight_csi = float(getattr(args, "sigreg_weight_csi", 0.0))
        self.sigreg_weight_point = float(getattr(args, "sigreg_weight_point", 0.0))
        self.sigreg_weight_traj = float(getattr(args, "sigreg_weight_traj", 0.0))

        # VISReg is applied to one pooled vector per sample and modality through a
        # disposable projection head. This measures cross-sample diversity instead
        # of letting fixed token positions make a collapsed representation appear
        # diverse, which can happen when all tokens are flattened for SIGReg.
        self.visreg_enable = bool(getattr(args, "visreg_enable", False))
        self.visreg_num_slices = int(getattr(args, "visreg_num_slices", 128))
        self.visreg_scale_weight = float(getattr(args, "visreg_scale_weight", 1.0))
        self.visreg_shape_weight = float(getattr(args, "visreg_shape_weight", 1.0))
        self.visreg_center_weight = float(getattr(args, "visreg_center_weight", 1.0))
        self.visreg_weight_csi = float(getattr(args, "visreg_weight_csi", 0.0))
        self.point_spread_weight = float(getattr(args, "point_spread_weight", 0.0))
        self.point_spread_surplus_weight = float(getattr(args, "point_spread_surplus_weight", 0.25))
        self.point_spread_predictor_only = bool(getattr(args, "point_spread_predictor_only", True))
        self.visreg_weight_point = float(getattr(args, "visreg_weight_point", 0.0))
        self.visreg_weight_traj = float(getattr(args, "visreg_weight_traj", 0.0))
        visreg_dim = int(getattr(args, "visreg_projection_dim", 128))
        self.visreg_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(args.latent_dim),
                    nn.Linear(args.latent_dim, args.latent_dim),
                    nn.GELU(),
                    nn.Linear(args.latent_dim, visreg_dim),
                )
                for name in ("csi", "point", "traj")
            }
        )

        # Anti-collapse mode. "ema" keeps the teacher-student EMA target (baseline
        # WWM). "sigreg" removes the EMA target entirely: the target is a stop-grad
        # pass of the ONLINE encoder over the full input (SimSiam-style), and SIGReg
        # is the sole collapse guard, so it must be active with a positive weight.
        # Guard-ablation escape hatch: K-SIGReg (the EM kernel-whitened distribution
        # regularizer) may serve as the sole guard when --em-kvisreg-enable is given,
        # so the anti-collapse capability of the physics term can be measured directly
        # (wwm_em_guard_ablation.py G2/G3). It is deliberately NOT counted by default
        # to keep the historical semantics unchanged.
        self.jepa_mode = str(getattr(args, "jepa_mode", "ema"))
        self.jepa_target_detach = not bool(getattr(args, "jepa_no_target_detach", False))
        if self.jepa_mode == "sigreg":
            self.sigreg_enable = self.sigreg_enable or not self.visreg_enable
        self.sigreg_weight_total = self.sigreg_weight_csi + self.sigreg_weight_point + self.sigreg_weight_traj
        self.visreg_weight_total = self.visreg_weight_csi + self.visreg_weight_point + self.visreg_weight_traj
        self._em_guard_ok = bool(getattr(args, "em_physics_enable", False)) and bool(
            getattr(args, "em_kvisreg_enable", False)
        )
        if self.jepa_mode == "sigreg" and self.sigreg_weight_total + self.visreg_weight_total <= 0.0 \
                and not self._em_guard_ok:
            raise ValueError(
                "--jepa-mode sigreg removes the EMA target, so SIGReg or VISReg must guard against collapse. "
                "Set at least one modality regularization weight > 0, or enable K-SIGReg as the sole guard."
            )

        # V-JEPA 2.1 context loss (facebookresearch/vjepa2, app/vjepa_2_1/train.py).
        # Besides the standard masked-token objective, align predictor outputs at
        # nearby visible tokens to the target with inverse-sqrt distance weights.
        self.context_loss_weight = float(getattr(args, "context_loss_weight", 0.0))
        self.context_loss_exp = float(getattr(args, "context_loss_exp", 1.0))
        self.context_loss_warmup_steps = int(getattr(args, "context_loss_warmup_steps", 0))
        self.context_loss_source = str(getattr(args, "context_loss_source", "encoder"))
        self.pretrain_sgcs_warmup_steps = int(getattr(args, "pretrain_sgcs_warmup_steps", 0))
        self._pretrain_step = 0

        # --- EM physical constraints (v17). See 预训练指挥文档_电磁物理约束v17.md.
        # The Jakes kernel is derived analytically from the trajectory, so nothing
        # here is learned except the K-SIGReg feature-axis projector. pos_scale is
        # stored because the engine only forwards h/point_cloud/traj — the metric
        # speed has to be recovered from the trajectory tokens (§3.5).
        self.em_physics_enable = bool(getattr(args, "em_physics_enable", False))
        self.em_kernel_kind = str(getattr(args, "em_kernel", "doppler"))
        self.em_kernel_time_basis = str(getattr(args, "em_kernel_time_basis", "center"))
        self.em_carrier_frequency_hz = float(getattr(args, "em_carrier_frequency_hz", 3.5e9))
        self.em_sample_period_s = float(getattr(args, "em_sample_period_s", 0.005))
        self.em_speed_scale = float(getattr(args, "em_speed_scale", 0.97))
        self.em_kernel_energy = float(getattr(args, "em_kernel_energy", 0.999))
        self.em_kernel_jitter = float(getattr(args, "em_kernel_jitter", 1e-4))
        self.em_physics_warmup_steps = int(getattr(args, "em_physics_warmup_steps", 500))
        self.em_apply_on = str(getattr(args, "em_apply_on", "predictor"))
        self.em_context_weight_enable = bool(getattr(args, "em_context_weight_enable", False))
        self.em_relation_weight = float(getattr(args, "em_relation_weight", 0.0))
        self.em_kvisreg_enable = bool(getattr(args, "em_kvisreg_enable", False))
        self.em_kvisreg_weight = float(getattr(args, "em_kvisreg_weight", 0.0))
        self.em_kvisreg_projections = int(getattr(args, "em_kvisreg_projections", 256))
        self.em_kvisreg_time_weight = float(getattr(args, "em_kvisreg_time_weight", 1.0))
        self.em_kvisreg_time_axis = bool(getattr(args, "em_kvisreg_time_axis", True))
        self.em_kvisreg_balance = bool(getattr(args, "em_kvisreg_balance", True))
        self.em_relation_centered = bool(getattr(args, "em_relation_centered", True))
        self.pos_scale = float(getattr(args, "pos_scale", 1000.0))
        self.em_tangent_enable = bool(getattr(args, "em_tangent_enable", False))
        self.em_tangent_weight = float(getattr(args, "em_tangent_weight", 0.0))
        self.em_tangent_vector_weight = float(
            getattr(args, "em_tangent_vector_weight", 0.05)
        )
        self.em_tangent_dim = int(getattr(args, "em_tangent_dim", 32))
        self.em_tangent_target_scale = float(
            getattr(args, "em_tangent_target_scale", 0.05)
        )
        self.em_tangent_beta_r = float(getattr(args, "em_tangent_beta_r", 1.0))
        self.em_tangent_beta_v = float(getattr(args, "em_tangent_beta_v", 0.25))
        self.em_tangent_beta_f = float(getattr(args, "em_tangent_beta_f", 1.0))
        self.em_tangent_position_scale = float(
            getattr(args, "em_tangent_position_scale", 1.0)
        )
        self.em_tangent_speed_scale = float(
            getattr(args, "em_tangent_speed_scale", 20.0)
        )
        self.em_tangent_phase_scale = float(
            getattr(args, "em_tangent_phase_scale", 1.0)
        )
        self.em_tangent_physics_mode = str(getattr(args, "em_tangent_physics_mode", "radial"))
        self.em_tangent_spread_scale = float(getattr(args, "em_tangent_spread_scale", 200.0))
        self.em_tangent_eta = float(getattr(args, "em_tangent_eta", 1.0))
        self.em_tangent_delta_ksigreg_enable = bool(
            getattr(args, "em_tangent_delta_ksigreg_enable", False)
        )
        self.em_tangent_delta_ksigreg_weight = float(
            getattr(args, "em_tangent_delta_ksigreg_weight", 0.0)
        )
        self.em_tangent_delta_ksigreg_projections = int(
            getattr(args, "em_tangent_delta_ksigreg_projections", 128)
        )
        self.em_tangent_delta_whiten = bool(getattr(args, "em_tangent_delta_whiten", False))
        self.em_direct_apply_on = str(getattr(args, "em_direct_apply_on", "predictor"))
        self.em_direct_multilevel = bool(getattr(args, "em_direct_multilevel", False))
        self.em_direct_level_weight = float(getattr(args, "em_direct_level_weight", 0.5))
        self.em_modal_enable = bool(getattr(args, "em_modal_enable", False))
        self.em_modal_weight = float(getattr(args, "em_modal_weight", 0.0))
        self.em_modal_dim = int(getattr(args, "em_modal_dim", 32))
        self.em_modal_temperature = float(
            getattr(args, "em_modal_temperature", 1.0)
        )
        self.em_modal_domain = str(getattr(args, "em_modal_domain", "fft"))
        self.em_modal_lags = tuple(
            int(value.strip()) for value in str(getattr(args, "em_modal_lags", "1,2,3,4")).split(",")
            if value.strip()
        )
        self.em_modal_smoothing = max(1, int(getattr(args, "em_modal_smoothing", 1)))
        self.em_modal_floor = max(float(getattr(args, "em_modal_floor", 1e-4)), 1e-8)
        self.em_direct_shuffle = bool(getattr(args, "em_direct_shuffle", False))
        if self.em_tangent_dim <= 0 or self.em_modal_dim <= 0:
            raise ValueError("EM tangent/modal dimensions must be positive")
        tangent_active = self.em_tangent_enable and (
            self.em_tangent_weight > 0 or self.em_tangent_vector_weight > 0
        )
        # Fixed projections provide a nonzero escape gradient at exact Delta-z == 0.
        # They are buffers, not trainable shortcuts; the network must encode the
        # physical change in its temporal representation.
        if tangent_active:
            self.register_buffer(
                "em_tangent_latent_proj",
                torch.randn(args.latent_dim, self.em_tangent_dim)
                / math.sqrt(float(args.latent_dim)),
                persistent=True,
            )
            self.register_buffer(
                "em_tangent_feature_proj",
                torch.randn(8, self.em_tangent_dim)
                / math.sqrt(8.0),
                persistent=True,
            )
        else:
            self.register_buffer(
                "em_tangent_latent_proj",
                torch.empty(args.latent_dim, self.em_tangent_dim),
                persistent=True,
            )
            self.register_buffer(
                "em_tangent_feature_proj",
                torch.empty(8, self.em_tangent_dim),
                persistent=True,
            )
        if self.em_modal_enable:
            self.em_modal_projector = nn.Sequential(
                nn.LayerNorm(args.latent_dim),
                nn.Linear(args.latent_dim, self.em_modal_dim, bias=False),
            )
        else:
            self.em_modal_projector = None
        em_projection_dim = int(getattr(args, "em_kvisreg_projection_dim", 128))
        if self.em_physics_enable and self.em_kvisreg_enable:
            # LayerNorm, NOT BatchNorm: each sample carries its own kernel (different
            # speed/geometry) and batch statistics would average away exactly the
            # physical differences this term is meant to preserve. Kept independent
            # from visreg_projectors["csi"], whose target is N(0,I) on UNWHITENED
            # tokens — the two objectives are incompatible.
            self.em_physics_projector = nn.Sequential(
                nn.LayerNorm(args.latent_dim),
                nn.Linear(args.latent_dim, args.latent_dim),
                nn.GELU(),
                nn.Linear(args.latent_dim, em_projection_dim),
            )
        else:
            self.em_physics_projector = None
        # Matched low-dimensional EM branch. Instantiate it for both A and C so the
        # control has identical parameter count and initialization; A sets all
        # three objective weights to zero.
        self.em_conditional_enable = bool(getattr(args, "em_conditional_enable", False))
        self.em_conditional_dim = int(getattr(args, "em_conditional_dim", 32))
        self.em_deltar_weight = float(getattr(args, "em_deltar_weight", 0.0))
        self.em_deltar_lags = tuple(
            int(value.strip())
            for value in str(getattr(args, "em_deltar_lags", "1,2,3,4,5,6,7,8")).split(",")
            if value.strip()
        )
        self.em_deltar_huber_delta = float(getattr(args, "em_deltar_huber_delta", 1.0))
        self.em_deltar_shuffle = bool(getattr(args, "em_deltar_shuffle", False))
        self.em_conditional_sigreg_weight = float(
            getattr(args, "em_conditional_sigreg_weight", 0.0)
        )
        self.em_conditional_scale_weight = float(
            getattr(args, "em_conditional_scale_weight", 0.0)
        )
        self.em_conditional_covariance_weight = float(
            getattr(args, "em_conditional_covariance_weight", 0.0)
        )
        self.em_conditional_rank_weight = float(
            getattr(args, "em_conditional_rank_weight", 0.0)
        )
        self.em_conditional_projections = int(
            getattr(args, "em_conditional_projections", 256)
        )
        self.em_rt_path_weight = float(getattr(args, "em_rt_path_weight", 0.0))
        self.em_rt_path_dim = int(getattr(args, "em_rt_path_dim", 0))
        if self.em_rt_path_weight > 0 and not self.em_conditional_enable:
            raise ValueError("RT path distillation requires --em-conditional-enable")
        if self.em_rt_path_weight > 0 and self.em_rt_path_dim <= 0:
            raise ValueError("em_rt_path_dim must be positive when em_rt_path_weight > 0")
        if self.em_conditional_enable:
            t_grid = (args.context_steps + args.future_steps) // int(args.patch_t)
            self.em_conditional_projector = nn.Sequential(
                nn.LayerNorm(args.latent_dim),
                nn.Linear(args.latent_dim, self.em_conditional_dim),
                nn.GELU(),
                nn.Linear(self.em_conditional_dim, self.em_conditional_dim),
            )
            flattened = t_grid * self.em_conditional_dim
            hidden = max(64, 2 * self.em_conditional_dim)
            self.em_deltar_head = nn.Sequential(
                nn.LayerNorm(flattened),
                nn.Linear(flattened, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2 * len(self.em_deltar_lags)),
            )
            # This head is deliberately separate from the CSI-derived Delta-R
            # head.  It consumes only the privileged RT sidecar residual and is
            # removed together with the sidecar at deployment time.
            self.em_rt_path_head = None
            # Instantiate this head in matched controls whenever a dimension is
            # specified. Its loss remains disabled unless em_rt_path_weight > 0.
            if self.em_rt_path_dim > 0:
                t_grid = (args.context_steps + args.future_steps) // int(args.patch_t)
                hidden = max(64, 2 * self.em_conditional_dim)
                self.em_rt_path_head = nn.Sequential(
                    nn.LayerNorm(t_grid * self.em_conditional_dim),
                    nn.Linear(t_grid * self.em_conditional_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, t_grid * self.em_rt_path_dim),
                )
        else:
            self.em_conditional_projector = None
            self.em_deltar_head = None
            self.em_rt_path_head = None

        # Not an nn.Module: holds plain buffers built lazily on first use, and must not
        # appear in state_dict (it is fully derived from the --em-* settings).
        self._em_kernel_cache = EMKernelCache(
            n_time=(args.context_steps + args.future_steps) // int(args.patch_t),
            patch_t=int(args.patch_t),
            carrier_frequency_hz=self.em_carrier_frequency_hz,
            sample_period_s=self.em_sample_period_s,
            speed_scale=self.em_speed_scale,
            jitter=self.em_kernel_jitter,
            energy=self.em_kernel_energy,
            time_basis=self.em_kernel_time_basis,
            kind="identity" if self.em_kernel_kind == "identity" else "doppler",
        )

        # V-JEPA 2.1 supervises the final four encoder/predictor levels. The
        # context fuser starts as an exact identity on the final level, preserving
        # the behavior of a warm-started V6 checkpoint at initialization.
        self.deep_supervision_layers = min(
            int(getattr(args, "deep_supervision_layers", 4)),
            int(args.mmoe_layers),
            int(args.predictor_layers),
        )
        self.deep_supervision_weight = float(getattr(args, "deep_supervision_weight", 0.0))
        self.deep_layer_selection = str(getattr(args, "deep_layer_selection", "last"))
        self.deep_context_fusion = str(getattr(args, "deep_context_fusion", "linear"))
        if self.deep_supervision_layers > 0:
            if self.deep_context_fusion == "linear":
                self.deep_context_fuser = nn.Linear(
                    self.deep_supervision_layers * args.latent_dim, args.latent_dim
                )
                nn.init.zeros_(self.deep_context_fuser.weight)
                nn.init.zeros_(self.deep_context_fuser.bias)
                start = (self.deep_supervision_layers - 1) * args.latent_dim
                with torch.no_grad():
                    self.deep_context_fuser.weight[:, start : start + args.latent_dim].copy_(
                        torch.eye(args.latent_dim)
                    )
            elif self.deep_context_fusion == "mlp":
                self.deep_context_fuser = ResidualMultiLevelFusionMLP(
                    latent_dim=args.latent_dim,
                    num_levels=self.deep_supervision_layers,
                    hidden_mult=float(getattr(args, "deep_fusion_hidden_mult", 1.0)),
                    residual_scale=float(getattr(args, "deep_fusion_residual_scale", 0.1)),
                )
            else:
                raise ValueError("Unknown deep context fusion: %s" % self.deep_context_fusion)
            self.deep_predictor_heads = nn.ModuleList(
                [nn.Linear(args.latent_dim, args.latent_dim) for _ in range(self.deep_supervision_layers)]
            )
            for head in self.deep_predictor_heads:
                nn.init.eye_(head.weight)
                nn.init.zeros_(head.bias)

        # In sigreg mode there is no separate teacher: target = stop-grad online
        # encoder, so we do not allocate the target branch at all (saves ~half the
        # pre-training parameters and all EMA bookkeeping).
        self.target_stem: Optional[TokenStem] = None
        self.target_encoder: Optional[MMoETransformer] = None
        self.target_predictor: Optional[MMoETransformer] = None
        if self.jepa_mode == "ema":
            self.target_stem = copy.deepcopy(self.online_stem)
            self.target_encoder = copy.deepcopy(self.online_encoder)
            self.target_predictor = copy.deepcopy(self.predictor) if args.ema_predictor_branch else None
            self.freeze_target()

    @property
    def has_ema_target(self) -> bool:
        return self.target_encoder is not None

    def point_dvae_modules(self) -> List[PointBERTDiscreteVAE]:
        modules: List[PointBERTDiscreteVAE] = []
        if self.online_stem.point.dvae is not None:
            modules.append(self.online_stem.point.dvae)
        if self.target_stem is not None and self.target_stem.point.dvae is not None:
            modules.append(self.target_stem.point.dvae)
        return modules

    def set_point_dvae_frozen(self, frozen: bool) -> None:
        self.online_stem.point.set_dvae_frozen(frozen)
        if self.target_stem is not None:
            self.target_stem.point.set_dvae_frozen(True)

    @torch.no_grad()
    def freeze_target(self) -> None:
        if self.target_encoder is None:
            return
        modules: List[nn.Module] = [self.target_stem, self.target_encoder]
        if self.target_predictor is not None:
            modules.append(self.target_predictor)
        for module in modules:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def sync_target_from_online(self) -> None:
        """Copy the online branch into the EMA target (frozen after the copy).

        Used when warm-starting an EMA-mode run from a sigreg-mode checkpoint:
        the checkpoint has no target-branch keys, so the deep-copied teacher is
        random and the first epoch would chase it. After this call the teacher
        equals the loaded online weights exactly.
        """
        if self.target_encoder is None:
            return
        self.target_stem.load_state_dict(self.online_stem.state_dict())
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        if self.target_predictor is not None:
            self.target_predictor.load_state_dict(self.predictor.state_dict())
        self.freeze_target()

    def train(self, mode: bool = True) -> "PaperWWM":
        super().train(mode)
        self.freeze_target()
        return self

    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        if self.target_encoder is None:  # sigreg mode: no EMA target to update.
            return
        pairs: List[Tuple[nn.Module, nn.Module]] = [(self.online_stem, self.target_stem), (self.online_encoder, self.target_encoder)]
        if self.target_predictor is not None:
            pairs.append((self.predictor, self.target_predictor))
        for online, target in pairs:
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.data.mul_(decay).add_(op.data, alpha=1.0 - decay)
            for ob, tb in zip(online.buffers(), target.buffers()):
                tb.copy_(ob)

    def set_pretrain_step(self, step: int) -> None:
        self._pretrain_step = max(int(step), 0)

    @staticmethod
    def _warmup_factor(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(max(float(step) / float(warmup_steps), 0.0), 1.0)

    def encode_predict(
        self,
        h_seq: torch.Tensor,
        point_cloud: torch.Tensor,
        traj: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int], torch.Tensor, Dict[str, Any]]:
        tokens, lengths = self.online_stem(h_seq, point_cloud, traj)
        use_deep = self.deep_supervision_weight > 0 and self.deep_supervision_layers > 0
        if self.encoder_visible_mode == "compact":
            ctx, context_levels = self.encode_visible_tokens(tokens, lengths, mask, return_intermediates=use_deep)
        elif self.encoder_visible_mode == "padded":
            masked_online = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
            if use_deep:
                ctx, context_levels = self.online_encoder(
                    masked_online,
                    lengths,
                    key_padding_mask=mask,
                    return_intermediates=True,
                    num_intermediate_layers=self.deep_supervision_layers,
                    intermediate_selection=self.deep_layer_selection,
                )
            else:
                ctx = self.online_encoder(masked_online, lengths, key_padding_mask=mask)
                context_levels = []
        else:
            raise ValueError("Unknown encoder_visible_mode: %s" % self.encoder_visible_mode)
        if use_deep:
            ctx = self.fuse_multilevel_context(ctx, context_levels)
        template = self.online_stem.mask_template(tokens.shape[0], tokens.device, tokens.dtype)
        mask_tokens = self.mask_token.to(tokens.dtype) + template
        pred_in = torch.where(mask.unsqueeze(-1), mask_tokens, ctx)
        if use_deep:
            pred, predictor_levels = self.predictor(
                pred_in,
                lengths,
                return_intermediates=True,
                num_intermediate_layers=self.deep_supervision_layers,
                intermediate_selection=self.deep_layer_selection,
            )
        else:
            pred = self.predictor(pred_in, lengths)
            predictor_levels = []
        if self.target_encoder is not None:
            # EMA teacher-student: target from the frozen EMA branch.
            with torch.no_grad():
                target_tokens, _ = self.target_stem(h_seq, point_cloud, traj)
                if use_deep:
                    target, target_levels = self.target_encoder(
                        target_tokens,
                        lengths,
                        return_intermediates=True,
                        num_intermediate_layers=self.deep_supervision_layers,
                        intermediate_selection=self.deep_layer_selection,
                    )
                else:
                    target = self.target_encoder(target_tokens, lengths)
                    target_levels = []
        else:
            # SIGReg mode: target = the ONLINE encoder over the full (unmasked)
            # input. Collapse is prevented by SIGReg, not by an asymmetric teacher.
            # By default we stop-grad the target (SimSiam); --jepa-no-target-detach
            # lets gradients flow through it (pure LeJEPA symmetric objective).
            if self.jepa_target_detach:
                with torch.no_grad():
                    if use_deep:
                        target, target_levels = self.online_encoder(
                            tokens,
                            lengths,
                            return_intermediates=True,
                            num_intermediate_layers=self.deep_supervision_layers,
                            intermediate_selection=self.deep_layer_selection,
                        )
                    else:
                        target = self.online_encoder(tokens, lengths)
                        target_levels = []
            else:
                if use_deep:
                    target, target_levels = self.online_encoder(
                        tokens,
                        lengths,
                        return_intermediates=True,
                        num_intermediate_layers=self.deep_supervision_layers,
                        intermediate_selection=self.deep_layer_selection,
                    )
                else:
                    target = self.online_encoder(tokens, lengths)
                    target_levels = []
        deep_state: Dict[str, Any] = {
            "pred_levels": [],
            "context_levels": context_levels,
            "target_levels": target_levels,
        }
        if use_deep:
            deep_state["pred_levels"] = [
                head(level) for head, level in zip(self.deep_predictor_heads, predictor_levels)
            ]
        return pred, target, lengths, ctx, deep_state

    def encode_visible_tokens(
        self,
        tokens: torch.Tensor,
        lengths: Tuple[int, int, int],
        mask: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bsz, total_tokens, dim = tokens.shape
        nc, np_, nt = lengths
        visible_counts = (~mask).sum(dim=1)
        ctx = tokens.new_zeros(bsz, total_tokens, dim)
        levels = [tokens.new_zeros(bsz, total_tokens, dim) for _ in range(self.deep_supervision_layers)] if return_intermediates else []
        if bool(torch.all(visible_counts == visible_counts[0]).item()):
            visible_tokens = tokens[~mask].reshape(bsz, int(visible_counts[0].item()), dim)
            visible_lengths = (
                int((~mask[:, :nc]).sum(dim=1)[0].item()),
                int((~mask[:, nc : nc + np_]).sum(dim=1)[0].item()),
                int((~mask[:, nc + np_ :]).sum(dim=1)[0].item()),
            )
            if return_intermediates:
                encoded_visible, encoded_levels = self.online_encoder(
                    visible_tokens,
                    visible_lengths,
                    return_intermediates=True,
                    num_intermediate_layers=self.deep_supervision_layers,
                    intermediate_selection=self.deep_layer_selection,
                )
            else:
                encoded_visible = self.online_encoder(visible_tokens, visible_lengths)
                encoded_levels = []
            ctx[~mask] = encoded_visible.reshape(-1, dim)
            for output_level, encoded_level in zip(levels, encoded_levels):
                output_level[~mask] = encoded_level.reshape(-1, dim)
            return ctx, levels

        for idx in range(bsz):
            sample_visible = ~mask[idx]
            visible_tokens = tokens[idx : idx + 1, sample_visible]
            visible_lengths = (
                int(sample_visible[:nc].sum().item()),
                int(sample_visible[nc : nc + np_].sum().item()),
                int(sample_visible[nc + np_ :].sum().item()),
            )
            if return_intermediates:
                encoded_visible, encoded_levels = self.online_encoder(
                    visible_tokens,
                    visible_lengths,
                    return_intermediates=True,
                    num_intermediate_layers=self.deep_supervision_layers,
                    intermediate_selection=self.deep_layer_selection,
                )
            else:
                encoded_visible = self.online_encoder(visible_tokens, visible_lengths)
                encoded_levels = []
            ctx[idx, sample_visible] = encoded_visible.squeeze(0)
            for output_level, encoded_level in zip(levels, encoded_levels):
                output_level[idx, sample_visible] = encoded_level.squeeze(0)
        return ctx, levels

    def fuse_multilevel_context(
        self,
        context: torch.Tensor,
        context_levels: List[torch.Tensor],
        residual_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Expose the same multi-level fusion used during pretraining.

        Downstream evaluators often call ``encode_visible_tokens`` directly;
        applying this helper there keeps frozen-feature evaluation consistent
        with the predictor path optimized during pretraining.
        """
        if (
            self.deep_supervision_weight <= 0
            or self.deep_supervision_layers <= 0
            or len(context_levels) != self.deep_supervision_layers
        ):
            return context
        concatenated_context = torch.cat(context_levels, dim=-1)
        if self.deep_context_fusion == "mlp":
            return self.deep_context_fuser(
                concatenated_context,
                context_levels[-1],
                residual_multiplier=residual_multiplier,
            )
        return self.deep_context_fuser(concatenated_context)

    def predict_point_latents(self, h_seq: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        """Predict every point-token latent from CSI and trajectory only."""
        lengths = self.online_stem.lengths()
        nc, np_, nt = lengths
        if traj.shape[1] != nt:
            raise ValueError("Trajectory token count expected=%d actual=%d" % (nt, traj.shape[1]))

        csi_tokens = self.online_stem.csi(h_seq) + self.online_stem.csi_modality
        traj_tokens = self.online_stem.traj(traj) + self.online_stem.traj_modality
        template = self.online_stem.mask_template(
            h_seq.shape[0], csi_tokens.device, csi_tokens.dtype
        )
        tokens = template.clone()
        tokens[:, :nc] = csi_tokens
        tokens[:, nc + np_ :] = traj_tokens
        mask = torch.zeros(
            h_seq.shape[0], nc + np_ + nt, dtype=torch.bool, device=h_seq.device
        )
        mask[:, nc : nc + np_] = True

        use_deep = self.deep_supervision_weight > 0 and self.deep_supervision_layers > 0
        ctx, context_levels = self.encode_visible_tokens(
            tokens, lengths, mask, return_intermediates=use_deep
        )
        if use_deep:
            ctx = self.fuse_multilevel_context(ctx, context_levels)
        mask_tokens = self.mask_token.to(tokens.dtype) + template
        pred_in = torch.where(mask.unsqueeze(-1), mask_tokens, ctx)
        pred = self.predictor(pred_in, lengths)
        return pred[:, nc : nc + np_]

    def future_csi_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        lengths = self.online_stem.lengths()
        nc, np_, nt = lengths
        tg, hg, wg = self.online_stem.csi.grid
        if self.context_steps % self.patch_t != 0:
            raise ValueError("context_steps must be divisible by patch_t for future masking.")
        future_t0 = self.context_steps // self.patch_t
        csi_mask = torch.zeros(nc, dtype=torch.bool, device=device)
        csi_grid = csi_mask.reshape(tg, hg, wg)
        csi_grid[future_t0:, :, :] = True
        full = torch.zeros(nc + np_ + nt, dtype=torch.bool, device=device)
        full[:nc] = csi_mask
        return full.unsqueeze(0).expand(batch_size, -1).clone()

    def forward(
        self,
        h_seq: torch.Tensor,
        point_cloud: torch.Tensor,
        traj: torch.Tensor,
        pretrain_mask: Optional[torch.Tensor] = None,
        pretrain_task: Optional[str] = None,
        em_path: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ) -> Dict[str, torch.Tensor]:
        if mode == "pretrain":
            if pretrain_mask is None:
                raise ValueError("pretrain_mask is required when mode='pretrain'.")
            return {
                "pretrain_loss": self.pretrain_loss(
                    h_seq, point_cloud, traj, pretrain_mask, pretrain_task=pretrain_task,
                    em_path=em_path
                )
            }
        if mode != "forecast":
            raise ValueError("Unknown PaperWWM forward mode: %s" % mode)
        mask = self.future_csi_mask(h_seq.shape[0], h_seq.device)
        pred, target, lengths, _ctx, _deep_state = self.encode_predict(h_seq, point_cloud, traj, mask)
        nc = lengths[0]
        future_token_mask = mask[:, :nc]
        if self.decoder_token_input == "future":
            future_tokens = pred[:, :nc][future_token_mask].reshape(h_seq.shape[0], -1, self.latent_dim)
            residual_h = self.decoder(future_tokens)
        elif self.decoder_token_input == "all_csi":
            residual_h = self.decoder(pred[:, :nc], future_token_mask=future_token_mask)
        else:
            raise ValueError("Unknown decoder_token_input: %s" % self.decoder_token_input)
        if self.temporal_anchor == "none":
            anchor = None
            pred_h = residual_h
        else:
            anchor = self.build_temporal_anchor(h_seq).to(dtype=residual_h.dtype)
            pred_h = anchor + self.anchor_residual_scale.to(dtype=residual_h.dtype) * residual_h
        latent_loss = F.l1_loss(pred[mask], target[mask])
        return {
            "pred_h": pred_h,
            "pred_tokens": pred,
            "target_tokens": target,
            "future_mask": mask,
            "latent_loss": latent_loss,
            "residual_h": residual_h,
            "anchor_h": anchor,
        }

    def build_temporal_anchor(self, h_seq: torch.Tensor) -> torch.Tensor:
        source = self._to_raw_csi(h_seq)
        last = source[:, self.context_steps - 1 : self.context_steps]
        if self.temporal_anchor == "copy":
            return self._from_raw_csi(last.expand(-1, self.future_steps, -1, -1, -1).contiguous())
        if self.temporal_anchor == "phasor":
            # EM inductive bias: over a 100 ms window the channel is nearly a
            # per-sample single complex phase rotation (measured |rho| 0.9994+).
            # Estimate the per-sample phase rate omega from the CONTEXT only:
            #   omega = angle( sum_{t=1..C-1} sum_space h_t * conj(h_{t-1}) )
            # and extrapolate anchor_k = h_{C-1} * exp(j*omega*k). No trainable
            # parameters; the decoder keeps learning the residual.
            phasor = self.phasor_anchor_frames(h_seq)
            return self._from_raw_csi(phasor)
        if self.temporal_anchor == "coherent":
            # WRONG EM assumption (negative-control anchor): static channel,
            # i.e. zero phase drift across the 100 ms window. The anchor is the
            # coherent mean of the context frames; the true per-sample phase
            # rotation smears its shape, so SGCS drops clearly below copy.
            source_c = self._to_raw_csi(h_seq)
            mean_c = source_c[:, : self.context_steps].mean(dim=1, keepdim=True)
            return self._from_raw_csi(mean_c.expand(-1, self.future_steps, -1, -1, -1).contiguous())
        if self.temporal_anchor != "linear":
            raise ValueError("Unknown temporal_anchor: %s" % self.temporal_anchor)
        prev = source[:, self.context_steps - 2 : self.context_steps - 1]
        delta = last - prev
        frames = []
        cur = last
        for _ in range(self.future_steps):
            cur = cur + self.anchor_delta_scale * delta
            frames.append(cur)
        return self._from_raw_csi(torch.cat(frames, dim=1))

    def phasor_anchor_frames(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Raw-domain phasor extrapolation [B, F, 2, 32, 32] from the context only.

        omega = angle( sum_{t=1..C-1} sum_space h_t * conj(h_{t-1}) ) per sample;
        anchor_k = h_{C-1} * exp(j*omega*k), k = 1..F. Pure EM inductive bias,
        zero trainable parameters, no trajectory input (the phase rate is a
        genuine channel quantity: measured R^2 vs trajectory speed ~ 0.03).
        """
        source = self._to_raw_csi(h_seq)
        hc = torch.complex(source[:, :, 0], source[:, :, 1])
        context = hc[:, : self.context_steps]
        acc = (context[:, 1:] * torch.conj(context[:, :-1])).sum(dim=(-2, -1)).sum(dim=1)
        omega = torch.angle(acc)  # [B]
        anchor_c = hc[:, self.context_steps - 1]  # [B, 32, 32]
        k = torch.arange(1, self.future_steps + 1, device=hc.device, dtype=hc.real.dtype)
        phasor = anchor_c.unsqueeze(1) * torch.exp(
            1j * omega[:, None, None, None] * k[None, :, None, None]
        )  # [B, F, 32, 32]
        return torch.stack([phasor.real, phasor.imag], dim=2).to(h_seq.dtype)

    def phasor_anchor_patch(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Patch-space phasor anchor rows for the FUTURE tubelets: [B, 128, D]."""
        stored = self._from_raw_csi(self.phasor_anchor_frames(h_seq))
        ph_patch = patchify_csi_tokens(stored, self.patch_t, self.patch_h, self.patch_w)
        return ph_patch.reshape(h_seq.shape[0], -1, ph_patch.shape[-1])

    def _to_raw_csi(self, h: torch.Tensor) -> torch.Tensor:
        if self.csi_transform != "signed_log":
            return h.float()
        y = h.float() * self.csi_std + self.csi_mean
        if self.inverse_signed_log_clip > 0:
            clip = float(self.inverse_signed_log_clip)
            y = clip * torch.tanh(y / clip)
        return torch.sign(y) * torch.expm1(torch.abs(y)) * self.signed_log_eps

    def _from_raw_csi(self, h: torch.Tensor) -> torch.Tensor:
        if self.csi_transform != "signed_log":
            return h
        y = torch.sign(h.float()) * torch.log1p(torch.abs(h.float()) / self.signed_log_eps)
        y = (y - self.csi_mean) / max(self.csi_std, 1e-6)
        return y.to(dtype=h.dtype)

    def _sigreg_pretrain(
        self,
        ctx: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
    ) -> Optional[torch.Tensor]:
        """Per-modality SIGReg over encoder (ctx) or predictor (pred) embeddings.

        Returns the weighted sum (>=0) or None when disabled. For encoder_out the
        regularizer sees only VISIBLE tokens per modality (masked slots in compact
        mode are zero-filled placeholders and would bias the marginal toward 0);
        for predictor_out every token is dense and used.
        """
        if not self.sigreg_enable or self.sigreg_weight_total <= 0:
            return None
        src = ctx if self.sigreg_apply_on == "encoder_out" else pred
        nc, np_, nt = lengths
        vis = ~mask  # [B, total] — True where a token was actually observed
        spans = (
            (0, nc, self.sigreg_weight_csi),
            (nc, nc + np_, self.sigreg_weight_point),
            (nc + np_, nc + np_ + nt, self.sigreg_weight_traj),
        )
        total = src.new_zeros(())
        raw = {}  # per-modality RAW (unweighted) SIGReg — for λ calibration/logging
        for name, (a, b, w) in zip(("csi", "point", "traj"), spans):
            if w <= 0 or b <= a:
                continue
            seg = src[:, a:b, :]
            if self.sigreg_apply_on == "encoder_out":
                emb = seg[vis[:, a:b]]
            else:
                emb = seg.reshape(-1, seg.shape[-1])
            if emb.shape[0] < 2:
                continue
            r = sigreg_loss(emb, self.sigreg_num_projections)
            raw[name] = float(r.detach())
            total = total + w * r
        self._sigreg_raw = raw
        return total

    def _visreg_pretrain(
        self,
        ctx: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
    ) -> Optional[torch.Tensor]:
        if not self.visreg_enable:
            return None
        nc, np_, nt = lengths
        spans = (
            ("csi", 0, nc, self.visreg_weight_csi),
            ("point", nc, nc + np_, self.visreg_weight_point),
            ("traj", nc + np_, nc + np_ + nt, self.visreg_weight_traj),
        )
        total = ctx.new_zeros(())
        raw: Dict[str, float] = {}
        active = False
        visible = ~mask
        for name, start, end, modality_weight in spans:
            if modality_weight <= 0 or end <= start:
                continue
            segment = ctx[:, start:end]
            segment_visible = visible[:, start:end]
            counts = segment_visible.sum(dim=1)
            valid = counts > 0
            if int(valid.sum().item()) < 2:
                continue
            pooled = (segment * segment_visible.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / counts.clamp_min(1).unsqueeze(-1)
            projected = self.visreg_projectors[name](pooled[valid])
            _, components = visreg_loss(projected, num_slices=self.visreg_num_slices)
            weighted_components = (
                self.visreg_scale_weight * components["scale"]
                + self.visreg_shape_weight * components["shape"]
                + self.visreg_center_weight * components["center"]
            )
            total = total + modality_weight * weighted_components
            active = True
            for component_name, value in components.items():
                raw["%s_%s" % (name, component_name)] = float(value.detach())
        self._visreg_raw = raw
        return total if active else None

    def _point_spread_pretrain(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
        context: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Match the *within-sample* spread of predicted point tokens to the target.

        Diagnosis (2026-08-18).  On v18/v19/v21 the predictor emits point tokens
        that are nearly identical to each other inside one sample while varying
        strongly across samples -- the opposite of the target's structure:

            target encoder(point tokens):  within 0.610  between 0.207  ratio 2.95
            predictor output:              within 0.040  between 0.331  ratio 0.12

        So the point branch encodes "which sample is this" rather than "which
        local patch is this token".  Retrieval downstreams reward the former and
        are unaffected; generative decoding needs the latter and fails, because
        the frozen dVAE folding decoder must be handed 256 *different* tokens to
        render 256 different patches.  Raising ``mask_point_weight`` does not fix
        it (already 3.0 in v18/v19/v21): sampling the task more often does not
        change which direction the L1 gradient favours, and the masked-token L1
        is in fact *worse* than emitting the per-token mean (0.698 vs 0.529),
        i.e. this direction is simply unfit rather than degenerately fit.

        The term below is a spread ratio on the masked point tokens only.  It
        leaves the mean token untouched -- the JEPA L1 keeps owning that -- and
        penalises log-ratio asymmetrically so a *deficit* of within-sample spread
        (the observed failure) costs more than a surplus.  Working on the ratio
        rather than the absolute value keeps it scale-free, so it does not fight
        the L1 term for control of the representation's overall magnitude.
        """
        if self.point_spread_weight <= 0.0:
            return None
        nc, np_, _ = lengths
        if np_ <= 1:
            return None
        point_mask = mask[:, nc : nc + np_]
        rows = point_mask.any(dim=1)
        if int(rows.sum().item()) < 2:
            return None
        if self.point_spread_predictor_only:
            if context is None:
                return None
            # The first v22 attempt let this term reach the shared encoder and
            # collapsed the CSI branch: repr_cos_csi went 0.135 -> 0.568 in 80
            # steps while repr_std_csi fell 0.399 -> 0.335, against a flat v21
            # baseline (cos ~0.38, std ~0.458).  The point and CSI branches share
            # the encoder, so the cheapest way for the predictor to widen
            # point-token spread was to flatten the encoder's CSI context.
            # Re-centring the output is not enough -- the deviations still carry
            # gradient back through the predictor into the encoder (measured
            # encoder gradient ratio stayed at 6.0).  So the term is applied to a
            # second predictor pass over a *detached* context: the predictor still
            # learns to spread its point tokens, and the encoder receives nothing
            # from this term at all.
            detached_input = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.to(context.dtype)
                + self.online_stem.mask_template(context.shape[0], context.device, context.dtype),
                context.detach(),
            )
            pred = self.predictor(detached_input, lengths)
        predicted = pred[:, nc : nc + np_][rows].float()
        expected = target[:, nc : nc + np_][rows].float().detach()
        selected = point_mask[rows].unsqueeze(-1)
        counts = selected.sum(dim=1).clamp_min(1)
        if int(counts.min().item()) < 2:
            return None

        def within_spread(value: torch.Tensor) -> torch.Tensor:
            centre = (value * selected).sum(dim=1, keepdim=True) / counts.unsqueeze(1)
            deviation = ((value - centre) * selected).square().sum(dim=1) / counts
            return deviation.mean(dim=-1).clamp_min(1e-8).sqrt()

        predicted_spread = within_spread(predicted)
        expected_spread = within_spread(expected)
        ratio = torch.log(predicted_spread / expected_spread.clamp_min(1e-8))
        # Asymmetric: a deficit (ratio < 0) is the failure mode being corrected.
        deficit = F.relu(-ratio)
        surplus = F.relu(ratio)
        loss = (deficit + self.point_spread_surplus_weight * surplus).mean()
        self._point_spread_raw = {
            "pred_within": float(predicted_spread.mean().detach()),
            "target_within": float(expected_spread.mean().detach()),
            "log_ratio": float(ratio.mean().detach()),
        }
        return self.point_spread_weight * loss

    # ------------------------------------------------------------------
    # EM physical constraints (v17). 预训练指挥文档_电磁物理约束v17.md §3–§6.
    # ------------------------------------------------------------------
    def em_kernel_for_batch(self, traj: torch.Tensor, n_time: int) -> Optional[torch.Tensor]:
        """Jakes temporal kernel [B, n_time, n_time] for this batch, or None.

        Built once per batch and shared by all three constraints. Runs with
        autocast disabled: eigh/bessel on bf16 loses the small eigenvalues the
        truncation rank depends on.
        """
        if not self.em_physics_enable:
            return None
        # Served from a speed-quantised cache that lives on the GPU (EMKernelCache).
        # Neither building the kernel on the device each step nor building it on the
        # host each step is safe on Windows: the former corrupts the CUDA context
        # (async "illegal memory access" in a later backward), the latter puts a
        # host-to-device copy inside graph construction and deadlocks the autograd
        # engine in cuMemcpyHtoDAsync_v2. The cache does the linalg once per distinct
        # speed, under no_grad, outside any graph.
        with torch.no_grad():
            speed = batch_speed_from_traj(traj, self.pos_scale, self.em_sample_period_s)
            kernel, whitener, ranks = self._em_kernel_cache.lookup(speed)
            if self.em_kernel_kind == "shuffle":
                # Cross-speed-bin control: the slowest sample gets the fastest
                # sample's kernel. Keeps the regularizer's scale, destroys the
                # sample<->physics correspondence.
                order = torch.argsort(speed)
                inverse = torch.argsort(order)
                permutation = order.flip(0)[inverse]
                kernel = kernel.index_select(0, permutation).contiguous()
                whitener = whitener.index_select(0, permutation).contiguous()
                ranks = ranks.index_select(0, permutation)
            self._em_speed = speed
            self._em_traj = traj.detach()
            self._em_whitener = whitener
            self._em_ranks = ranks
        return kernel

    def _em_temporal_latent(
        self,
        tokens: torch.Tensor,
        lengths: Tuple[int, int, int],
        visible: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """CSI tokens -> [B, t_grid, D] by averaging over the spatial axis.

        The kernel has time semantics only (t_grid x t_grid) while the CSI tokens
        interleave time and space, so the spatial pooling is mandatory. It is a
        FIXED linear operator, hence covariances push forward as A K A^T. The cost
        is that spatial structure is invisible to the EM terms; plain VISReg and
        the latent term cover it.

        When `visible` is given (encoder mode) the mean uses visible tokens only and
        the method returns None if any sample has a fully-masked time group.
        """
        nc = lengths[0]
        t_grid = self.online_stem.csi.grid[0]
        spatial = nc // t_grid
        csi = tokens[:, :nc].float().reshape(tokens.shape[0], t_grid, spatial, -1)
        if visible is None:
            return csi.mean(dim=2)
        weights = visible[:, :nc].reshape(tokens.shape[0], t_grid, spatial, 1).float()
        counts = weights.sum(dim=2)
        if bool((counts <= 0).any()):
            return None
        return (csi * weights).sum(dim=2) / counts

    def _em_context_alpha(
        self,
        kernel: Optional[torch.Tensor],
        csi_mask: torch.Tensor,
        nc: int,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """Physical context weights alpha_i [B, t_grid] and the whole-row verdict.

        alpha_i is defined on the time axis, so broadcasting it over the spatial
        axis gives every token in a time group the same weight and throws away the
        grid weight's spatial resolution. That resolution is exactly 1.000 for
        whole-time-row masking (temporal task) and 1.778 for space-time block
        masking (fine/coarse), so the physical weight is only used in the first
        case. The verdict comes from the mask tensor, not from the task label, so
        the downstream/prediction paths behave identically.
        """
        if kernel is None or not self.em_context_weight_enable:
            return None, False
        t_grid = self.online_stem.csi.grid[0]
        spatial = nc // t_grid
        blocks = csi_mask.reshape(csi_mask.shape[0], t_grid, spatial)
        masked_time = blocks.any(dim=2)
        whole_row = bool(torch.all(blocks.all(dim=2) == masked_time))
        if not whole_row:
            return None, False
        # kernel is already on the mask's device (cache), and alpha is only pow + bmm.
        alpha = context_alpha(kernel, masked_time)
        if alpha is None:
            return None, False
        return alpha.contiguous(), True

    def _em_physics_pretrain(
        self,
        ctx: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
        kernel: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Relation matching + K-SIGReg on the time-pooled CSI representation."""
        stats: Dict[str, float] = {}
        if kernel is None:
            return None, stats
        use_relation = self.em_relation_weight > 0
        use_kvisreg = self.em_kvisreg_enable and self.em_kvisreg_weight > 0
        if not use_relation and not use_kvisreg:
            return None, stats
        source = pred if self.em_apply_on == "predictor" else ctx
        visible = None if self.em_apply_on == "predictor" else ~mask
        latent = self._em_temporal_latent(source, lengths, visible=visible)
        if latent is None:
            return None, stats
        # kernel and whitener are already on the device, straight from the cache: no
        # transfer and no linalg inside the graph.
        device_type = latent.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            total = latent.new_zeros(())
            if use_relation:
                relation = physical_relation_loss(
                    latent, kernel, centered=self.em_relation_centered
                )
                total = total + self.em_relation_weight * relation
                stats["em_relation_raw"] = float(relation.detach())
            if use_kvisreg:
                # The whitened coordinates inherit the representation's overall scale,
                # and the Epps-Pulley statistic is scale-sensitive (it compares against
                # N(0,1)). Measured on v16 the time-pooled latent carries a 90% DC
                # component, which alone drives the statistic to ~590 (raw CSI: 770).
                # Centring the time axis and normalising to unit variance removes the
                # part of the mismatch that is a scale/offset artefact, leaving the
                # covariance SHAPE — which is what Cov = K is actually about.
                if self.em_relation_centered:
                    latent = center_time(latent)
                    latent = latent / latent.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                ranks = self._em_ranks
                buckets = whitener_buckets_from_cache(self._em_whitener, ranks)
                stats["em_kvisreg_rank"] = float(ranks.float().mean())
                batch = latent.shape[0]
                feature_terms = latent.new_zeros(())
                time_terms = latent.new_zeros(())
                balanced_time_terms = latent.new_zeros(())
                for whitener, selected, rank in buckets:
                    # [n_sel, r, T] @ [n_sel, T, D] -> [n_sel, r, D]. Both operands
                    # are forced contiguous: latent[selected] is a gather whose
                    # backward scatters, and an unaligned right-hand operand here
                    # crashes cuBLAS asynchronously (see truncated_whitener).
                    whitened = whitener @ latent[selected].contiguous()
                    share = float(selected.numel()) / float(batch)
                    if self.em_physics_projector is not None:
                        projected = self.em_physics_projector(
                            whitened.reshape(-1, whitened.shape[-1])
                        )
                        _, components = visreg_loss(projected, num_slices=self.visreg_num_slices)
                        feature_terms = feature_terms + share * (
                            self.visreg_scale_weight * components["scale"]
                            + self.visreg_shape_weight * components["shape"]
                            + self.visreg_center_weight * components["center"]
                        )
                    if self.em_kvisreg_time_axis and rank >= 2:
                        # The time axis is the branch that actually enforces Cov = K:
                        # SIGReg/VISReg only ever look at the LAST tensor dimension, so
                        # without this reshape the whitening is invisible to the
                        # objective. No projector here — the r coordinates ARE the
                        # truncated KL physical modes and an MLP would destroy the
                        # one-dimension-per-EM-mode correspondence.
                        # .contiguous() before the reshape, for the same reason as the
                        # matmul above: transpose+reshape on this gather-derived tensor
                        # yields a view with a storage offset, and sigreg_loss feeds it
                        # straight into cuBLAS matmuls, which can fail asynchronously
                        # ("misaligned address" / "invalid argument" / CUBLAS_STATUS_
                        # EXECUTION_FAILED surfacing in a later backward or synchronize).
                        #
                        # This hardening is correct on its own merits, but it did NOT
                        # fully fix the v18 (mask_point_weight>0) crashes: v18 died again
                        # at step ~83 in a bf16 GEMM inside backward. Do not treat this
                        # line as the known root cause. An earlier hypothesis -- that the
                        # `point` task is special because it leaves all CSI visible and so
                        # drives this path at full width -- is REFUTED: the `traj` task
                        # also yields 640/640 visible CSI and ran at weight 1.5 for v17's
                        # full 4520 steps with zero crashes. See
                        # docs/CODEX_v18_point_grounding_交接.md for the open diagnosis.
                        time_axis = whitened.transpose(1, 2).contiguous().reshape(-1, rank)
                        time_term = sigreg_loss(time_axis, self.em_kvisreg_projections)
                        time_terms = time_terms + share * time_term
                        time_scale = (
                            float(rank) / float(whitened.shape[-1])
                            if self.em_kvisreg_balance
                            else 1.0
                        )
                        balanced_time_terms = balanced_time_terms + share * time_scale * time_term
                stats["em_kvisreg_feature_raw"] = float(feature_terms.detach())
                stats["em_kvisreg_time_raw"] = float(time_terms.detach())
                stats["em_kvisreg_time_balanced"] = float(balanced_time_terms.detach())
                total = total + self.em_kvisreg_weight * (
                    feature_terms + self.em_kvisreg_time_weight * balanced_time_terms
                )
        return total, stats

    def _em_direct_anticollapse_pretrain(
        self,
        h_seq: torch.Tensor,
        traj: torch.Tensor,
        pred: torch.Tensor,
        lengths: Tuple[int, int, int],
        source: Optional[torch.Tensor] = None,
        visible: Optional[torch.Tensor] = None,
        stats_prefix: str = "",
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Direct EM tangent and Doppler-modal constraints on predictor CSI latents.

        The tangent norm term implements the requested log-distance matching. A
        small fixed-direction term is essential: a squared norm has zero gradient
        at exact Delta-z == 0 even though its loss is positive. The fixed projection
        supplies an escape gradient without adding a trainable physics shortcut.

        The modal term matches the non-DC temporal spectrum of the predictor latent
        to the corresponding CSI spectrum. It complements the tangent guard: modal
        energy alone is also quadratic and therefore is not an independent escape
        mechanism at the exact collapsed point.
        """
        use_tangent = self.em_tangent_enable and self.em_tangent_weight > 0
        use_modal = (
            self.em_modal_enable
            and self.em_modal_weight > 0
            and self.em_modal_projector is not None
        )
        if not use_tangent and not use_modal:
            return None, {}
        latent = self._em_temporal_latent(pred if source is None else source, lengths, visible=visible)
        if latent is None or latent.shape[1] < 3:
            return None, {}
        stats: Dict[str, float] = {}
        with torch.amp.autocast(device_type=latent.device.type, enabled=False):
            latent = latent.float()
            total = latent.new_zeros(())
            speed = batch_speed_from_traj(
                traj, self.pos_scale, self.em_sample_period_s
            ).float()
            permutation = torch.arange(latent.shape[0], device=latent.device)
            if self.em_direct_shuffle:
                speed_groups = torch.round(speed * 3.6).long()
                for group in torch.unique(speed_groups).tolist():
                    indices = torch.where(speed_groups == int(group))[0]
                    if indices.numel() > 1:
                        permutation[indices] = indices.roll(1)

            if use_tangent:
                total_steps = traj.shape[1]
                patch_t = int(self.patch_t)
                if total_steps % patch_t != 0 or total_steps // patch_t != latent.shape[1]:
                    raise ValueError(
                        "EM tangent requires trajectory time grid to match CSI tubelets"
                    )
                position = traj[..., :3].float() * float(self.pos_scale)
                position = position.reshape(
                    position.shape[0], latent.shape[1], patch_t, 3
                ).mean(dim=2)
                displacement = position[:, 1:] - position[:, :-1]
                delta_t = float(patch_t) * float(self.em_sample_period_s)
                velocity = displacement / max(delta_t, 1e-8)
                midpoint = 0.5 * (position[:, 1:] + position[:, :-1])
                los = F.normalize(midpoint, dim=-1, eps=1e-6)
                radial_velocity = (velocity * los).sum(dim=-1)
                doppler_hz = (
                    float(self.em_carrier_frequency_hz) / 299792458.0
                ) * radial_velocity
                phase_advance = 2.0 * math.pi * doppler_hz * delta_t
                features = torch.cat(
                    [
                        displacement / max(self.em_tangent_position_scale, 1e-6),
                        velocity / max(self.em_tangent_speed_scale, 1e-6),
                        (radial_velocity / max(self.em_tangent_speed_scale, 1e-6)).unsqueeze(-1),
                        (phase_advance / max(self.em_tangent_phase_scale, 1e-6)).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                target_features = features.index_select(0, permutation).detach()
                speed_mag = velocity.norm(dim=-1)
                spread_hz = (
                    float(self.em_carrier_frequency_hz) / 299792458.0
                ) * speed_mag * float(self.em_tangent_eta)
                mode = self.em_tangent_physics_mode
                if mode == "radial":
                    distance_sq = (
                        self.em_tangent_beta_r * target_features[..., :3].square().sum(dim=-1)
                        + self.em_tangent_beta_v * target_features[..., 3:6].square().sum(dim=-1)
                        + self.em_tangent_beta_f * target_features[..., 7].square()
                    )
                elif mode == "spread":
                    distance_sq = (
                        self.em_tangent_beta_r * target_features[..., :3].square().sum(dim=-1)
                        + self.em_tangent_beta_f * (spread_hz / max(self.em_tangent_spread_scale, 1e-6)).square()
                    )
                else:  # speed_spread: radial Doppler is retained only as a diagnostic.
                    distance_sq = (
                        self.em_tangent_beta_r * target_features[..., :3].square().sum(dim=-1)
                        + self.em_tangent_beta_v * (speed_mag / max(self.em_tangent_speed_scale, 1e-6)).square()
                        + self.em_tangent_beta_f * (spread_hz / max(self.em_tangent_spread_scale, 1e-6)).square()
                    )
                distance_sq = distance_sq.clamp_min(1e-8)
                delta_latent = latent[:, 1:] - latent[:, :-1]
                delta_projected = delta_latent @ self.em_tangent_latent_proj.float()
                target_energy = (
                    self.em_tangent_target_scale ** 2 * distance_sq
                ).clamp_min(1e-8)
                predicted_energy = delta_projected.square().mean(dim=-1)
                norm_term = (
                    torch.log(predicted_energy + 1e-8)
                    - torch.log(target_energy + 1e-8)
                ).square().mean()
                direction = F.normalize(
                    target_features @ self.em_tangent_feature_proj.float(),
                    dim=-1,
                    eps=1e-6,
                )
                target_vector = direction * torch.sqrt(
                    float(self.em_tangent_dim) * target_energy
                ).unsqueeze(-1)
                vector_term = F.smooth_l1_loss(delta_projected, target_vector)
                tangent_raw = norm_term + self.em_tangent_vector_weight * vector_term
                delta_ksigreg = latent.new_zeros(())
                delta_rank = 0.0
                if (
                    self.em_tangent_delta_ksigreg_enable
                    and self.em_tangent_delta_ksigreg_weight > 0
                    and delta_latent.shape[0] * delta_latent.shape[1] >= 2
                ):
                    delta_for_reg = delta_latent.reshape(-1, delta_latent.shape[-1])
                    delta_for_reg = delta_for_reg - delta_for_reg.mean(dim=0, keepdim=True)
                    if self.em_tangent_delta_whiten:
                        delta_for_reg = delta_for_reg / delta_for_reg.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
                    delta_rank = float(torch.linalg.matrix_rank(delta_for_reg.detach()).float().detach())
                    delta_ksigreg = sigreg_loss(
                        delta_for_reg,
                        self.em_tangent_delta_ksigreg_projections,
                    )
                    tangent_raw = tangent_raw + self.em_tangent_delta_ksigreg_weight * delta_ksigreg
                total = total + self.em_tangent_weight * tangent_raw
                stats.update({
                    stats_prefix + "em_tangent_raw": float(tangent_raw.detach()),
                    stats_prefix + "em_tangent_norm_raw": float(norm_term.detach()),
                    stats_prefix + "em_tangent_vector_raw": float(vector_term.detach()),
                    stats_prefix + "em_tangent_delta_ksigreg": float(delta_ksigreg.detach()),
                    stats_prefix + "em_tangent_delta_rank": delta_rank,
                    stats_prefix + "em_tangent_pred_energy": float(predicted_energy.mean().detach()),
                    stats_prefix + "em_tangent_target_energy": float(target_energy.mean().detach()),
                    stats_prefix + "em_tangent_doppler_hz": float(doppler_hz.abs().mean().detach()),
                    stats_prefix + "em_tangent_spread_hz": float(spread_hz.mean().detach()),
                })

            if use_modal:
                raw = self._to_raw_csi(h_seq).float()
                real = raw[:, :, 0]
                imag = raw[:, :, 1]
                csi = torch.complex(real, imag)
                csi = csi.reshape(
                    csi.shape[0], latent.shape[1], int(self.patch_t), 32, 32
                ).mean(dim=2)
                projected = self.em_modal_projector(latent)
                projected = projected - projected.mean(dim=1, keepdim=True)
                if self.em_modal_domain == "lag":
                    lags = tuple(sorted({x for x in self.em_modal_lags if 0 < x < latent.shape[1]}))
                    if not lags:
                        return None, stats
                    pred_profiles = []
                    target_profiles = []
                    for lag in lags:
                        pred_profiles.append(
                            (projected[:, lag:] * projected[:, :-lag]).mean(dim=(1, 2)).abs()
                        )
                        csi_lag = csi[:, lag:] * csi[:, :-lag].conj()
                        # mean over (time, antenna, subband) -> [B] per lag, so the
                        # stacked target profile is [B, n_lags] and matches the latent
                        # profile [B, n_lags] (the (1,2)-mean above left the subband
                        # axis, causing a [B,32] vs [B] broadcast failure).
                        target_profiles.append(csi_lag.real.mean(dim=(1, 2, 3)).abs())
                    latent_energy = torch.stack(pred_profiles, dim=-1)
                    target_energy = torch.stack(target_profiles, dim=-1)
                    if self.em_modal_smoothing > 1 and latent_energy.shape[-1] > 1:
                        width = min(self.em_modal_smoothing, latent_energy.shape[-1])
                        # Average pooling with odd width is deterministic and avoids
                        # non-constant padding kernels that are not available on all
                        # supported Torch CPU builds.
                        if width % 2 == 0:
                            width -= 1
                        if width > 1:
                            latent_energy = F.avg_pool1d(
                                latent_energy.unsqueeze(1), width, stride=1,
                                padding=width // 2, count_include_pad=False,
                            ).squeeze(1)
                            target_energy = F.avg_pool1d(
                                target_energy.unsqueeze(1), width, stride=1,
                                padding=width // 2, count_include_pad=False,
                            ).squeeze(1)
                    n_modes = len(lags)
                else:
                    latent_fft = torch.fft.fft(projected, dim=1)
                    n_modes = latent.shape[1] // 2
                    latent_energy = latent_fft[:, 1 : n_modes + 1].abs().square().mean(dim=-1)
                    csi = csi - csi.mean(dim=1, keepdim=True)
                    csi_fft = torch.fft.fft(csi, dim=1)
                    target_energy = csi_fft[:, 1 : n_modes + 1].abs().square().mean(dim=(-1, -2))
                latent_energy = latent_energy + self.em_modal_floor
                target_energy = target_energy + self.em_modal_floor
                target_prob = target_energy / target_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                target_prob = target_prob.index_select(0, permutation).detach()
                temperature = max(self.em_modal_temperature, 1e-3)
                latent_energy = (latent_energy + 1e-8).pow(1.0 / temperature)
                predicted_prob = latent_energy / latent_energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                modal_raw = (
                    target_prob
                    * (
                        torch.log(target_prob.clamp_min(1e-8))
                        - torch.log(predicted_prob.clamp_min(1e-8))
                    )
                ).sum(dim=-1).mean()
                total = total + self.em_modal_weight * modal_raw
                modal_cos = F.cosine_similarity(target_prob, predicted_prob, dim=-1).mean()
                target_entropy = -(
                    target_prob * torch.log(target_prob.clamp_min(1e-8))
                ).sum(dim=-1).mean()
                stats.update({
                    stats_prefix + "em_modal_raw": float(modal_raw.detach()),
                    stats_prefix + "em_modal_cosine": float(modal_cos.detach()),
                    stats_prefix + "em_modal_target_entropy": float(target_entropy.detach()),
                    stats_prefix + "em_modal_pred_entropy": float((-(predicted_prob * torch.log(predicted_prob.clamp_min(1e-8))).sum(dim=-1).mean()).detach()),
                    stats_prefix + "em_modal_modes": float(n_modes),
                })
        return total, stats

    def _em_conditional_pretrain(
        self,
        h_seq: torch.Tensor,
        ctx: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
        em_path: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Sample-level DeltaR distillation plus speed-conditioned K-SIGReg."""
        stats: Dict[str, float] = {}
        if not self.em_conditional_enable or self.em_conditional_projector is None:
            return None, stats
        active = (
            self.em_deltar_weight > 0
            or self.em_conditional_sigreg_weight > 0
            or self.em_conditional_scale_weight > 0
            or self.em_conditional_covariance_weight > 0
            or self.em_conditional_rank_weight > 0
            or self.em_rt_path_weight > 0
        )
        if not active:
            return None, stats
        source = pred if self.em_apply_on == "predictor" else ctx
        visible = None if self.em_apply_on == "predictor" else ~mask
        latent = self._em_temporal_latent(source, lengths, visible=visible)
        if latent is None:
            return None, stats

        with torch.amp.autocast(device_type=latent.device.type, enabled=False):
            z_em = self.em_conditional_projector(latent.float())
            total = z_em.new_zeros(())
            if self.em_deltar_weight > 0 and self.em_deltar_head is not None:
                target = empirical_deltar_target(
                    self._to_raw_csi(h_seq).detach(),
                    self._em_speed,
                    lags=self.em_deltar_lags,
                    carrier_frequency_hz=self.em_carrier_frequency_hz,
                    sample_period_s=self.em_sample_period_s,
                    speed_scale=self.em_speed_scale,
                )
                if self.em_deltar_shuffle:
                    # Matched negative control: permute the target across samples
                    # WITHIN each speed bin. The per-speed label distribution is
                    # unchanged (unlike a cross-speed shuffle), but no sample keeps
                    # its own physics; anything the Huber head learns then comes from
                    # the population structure, not the per-sample correspondence.
                    target = shuffle_target_within_speed_bins(target, self._em_speed)
                estimate = self.em_deltar_head(z_em.reshape(z_em.shape[0], -1))
                deltar = F.huber_loss(
                    estimate, target, delta=self.em_deltar_huber_delta, reduction="mean"
                )
                total = total + self.em_deltar_weight * deltar
                mae = (estimate - target).abs().mean()
                null_mae = target.abs().mean().clamp_min(1e-8)
                stats["em_deltar_raw"] = float(deltar.detach())
                stats["em_deltar_mae"] = float(mae.detach())
                stats["em_deltar_null_mae"] = float(null_mae.detach())
                stats["em_deltar_skill"] = float((1.0 - mae / null_mae).detach())
                stats["em_deltar_target_rms"] = float(target.square().mean().sqrt())

            if self.em_rt_path_weight > 0:
                if em_path is None or self.em_rt_path_head is None:
                    raise ValueError("RT path distillation is enabled but batch has no em_path sidecar")
                if em_path.ndim != 3 or int(em_path.shape[-1]) != self.em_rt_path_dim:
                    raise ValueError(
                        "em_path must be [B,T,%d], got %s" % (self.em_rt_path_dim, tuple(em_path.shape))
                    )
                # Geometry nuisance removal: regress each path feature on a
                # detached basis made from position, velocity and BS-relative
                # distance, then distil only the batch-orthogonal residual.
                path_steps = int(em_path.shape[1])
                if self._em_traj.shape[1] != path_steps:
                    raise ValueError(
                        "RT sidecar time length %d must match trajectory length %d"
                        % (path_steps, self._em_traj.shape[1])
                    )
                geom_continuous = torch.cat([
                    self._em_traj.float(),
                    self._em_speed.float().view(-1, 1, 1).expand(-1, path_steps, -1),
                ], dim=-1).reshape(-1, self._em_traj.shape[-1] + 1)
                raw_path = em_path.float().reshape(-1, self.em_rt_path_dim)
                geom_continuous = geom_continuous - geom_continuous.mean(dim=0, keepdim=True)
                geom = torch.cat([raw_path.new_ones((raw_path.shape[0], 1)), geom_continuous], dim=-1)
                gram = geom.transpose(0, 1) @ geom
                gram = gram + 1e-4 * torch.eye(gram.shape[0], device=geom.device, dtype=geom.dtype)
                beta = torch.linalg.solve(gram, geom.transpose(0, 1) @ raw_path)
                nuisance = geom @ beta
                residual = (raw_path - nuisance).reshape(ctx.shape[0], path_steps, self.em_rt_path_dim)
                t_grid = int(z_em.shape[1])
                if path_steps % self.patch_t != 0 or path_steps // self.patch_t != t_grid:
                    raise ValueError("RT sidecar time length must equal model context+future and patch grid")
                target_path = residual.reshape(ctx.shape[0], t_grid, self.patch_t, self.em_rt_path_dim).mean(dim=2)
                estimate_path = self.em_rt_path_head(z_em.reshape(z_em.shape[0], -1)).reshape_as(target_path)
                path_loss = F.huber_loss(estimate_path, target_path.detach(), delta=1.0, reduction="mean")
                total = total + self.em_rt_path_weight * path_loss
                path_mae = (estimate_path - target_path).abs().mean()
                path_null_mae = target_path.abs().mean().clamp_min(1e-8)
                stats["em_rt_path_mae"] = float(path_mae.detach())
                stats["em_rt_path_null_mae"] = float(path_null_mae.detach())
                stats["em_rt_path_skill"] = float((1.0 - path_mae / path_null_mae).detach())
                stats["em_rt_path_target_rms"] = float(target_path.square().mean().sqrt().detach())
                stats["em_rt_path_residual_dim"] = float(self.em_rt_path_dim)

            if (
                self.em_conditional_sigreg_weight > 0
                or self.em_conditional_scale_weight > 0
                or self.em_conditional_covariance_weight > 0
                or self.em_conditional_rank_weight > 0
            ):
                ranks = self._em_ranks
                speed_groups = torch.round(self._em_speed.float() * 3.6).long()
                sigreg_total = z_em.new_zeros(())
                scale_total = z_em.new_zeros(())
                covariance_total = z_em.new_zeros(())
                rank_total = z_em.new_zeros(())
                batch = z_em.shape[0]
                min_rows = 0
                group_count = 0
                effective_rank_total = z_em.new_zeros(())
                effective_rank_share = 0.0
                for speed_group in torch.unique(speed_groups).tolist():
                    group_mask = speed_groups == int(speed_group)
                    for rank in torch.unique(ranks[group_mask]).tolist():
                        selected = (group_mask & (ranks == int(rank))).nonzero(as_tuple=True)[0]
                        if selected.numel() == 0 or int(rank) < 2:
                            continue
                        whitener = self._em_whitener.index_select(0, selected)[:, : int(rank)]
                        whitened = whitener @ z_em.index_select(0, selected).contiguous()
                        time_axis = (
                            whitened.transpose(1, 2).contiguous().reshape(-1, int(rank))
                        )
                        share = float(selected.numel()) / float(batch)
                        if self.em_conditional_sigreg_weight > 0:
                            sigreg_total = sigreg_total + share * sigreg_loss(
                                time_axis, self.em_conditional_projections
                            )
                        if self.em_conditional_scale_weight > 0:
                            mode_std = time_axis.std(dim=0, unbiased=False)
                            scale_total = scale_total + share * (1.0 - mode_std).square().mean()
                        if self.em_conditional_covariance_weight > 0 or self.em_conditional_rank_weight > 0:
                            centered = time_axis - time_axis.mean(dim=0, keepdim=True)
                            cov = centered.transpose(0, 1) @ centered / max(int(centered.shape[0] - 1), 1)
                            eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
                            cov_error = (cov - eye).square().mean()
                            if self.em_conditional_covariance_weight > 0:
                                covariance_total = covariance_total + share * cov_error
                            if self.em_conditional_rank_weight > 0:
                                eig = torch.linalg.eigvalsh(cov.float()).clamp_min(1e-8)
                                p = eig / eig.sum().clamp_min(1e-8)
                                erank = torch.exp(-(p * p.log()).sum())
                                rank_floor = 0.6 * float(int(rank))
                                rank_total = rank_total + share * F.relu(
                                    (rank_floor - erank) / max(rank_floor, 1.0)
                                )
                                effective_rank_total = effective_rank_total + share * erank
                                effective_rank_share += share
                        rows = int(time_axis.shape[0])
                        min_rows = rows if min_rows == 0 else min(min_rows, rows)
                        group_count += 1
                total = total + self.em_conditional_sigreg_weight * sigreg_total
                total = total + self.em_conditional_scale_weight * scale_total
                total = total + self.em_conditional_covariance_weight * covariance_total
                total = total + self.em_conditional_rank_weight * rank_total
                stats["em_cond_sigreg_raw"] = float(sigreg_total.detach())
                stats["em_cond_scale_raw"] = float(scale_total.detach())
                stats["em_cond_covariance_raw"] = float(covariance_total.detach())
                stats["em_cond_rank_raw"] = float(rank_total.detach())
                stats["em_cond_groups"] = float(group_count)
                stats["em_cond_min_rows"] = float(min_rows)
                stats["em_cond_z_std"] = float(z_em.std(unbiased=False).detach())
                if effective_rank_share > 0:
                    stats["em_cond_effective_rank"] = float(
                        (effective_rank_total / effective_rank_share).detach()
                    )
        return total, stats

    def _context_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
        kernel: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Distance-weighted V-JEPA 2.1 loss on visible CSI tokens."""
        if self.context_loss_weight <= 0:
            return None
        nc = lengths[0]
        csi_mask = mask[:, :nc]
        csi_visible = ~csi_mask
        if not bool(csi_mask.any()) or not bool(csi_visible.any()):
            return None
        coords = torch.cartesian_prod(
            torch.arange(self.online_stem.csi.grid[0], device=mask.device),
            torch.arange(self.online_stem.csi.grid[1], device=mask.device),
            torch.arange(self.online_stem.csi.grid[2], device=mask.device),
        ).float()
        token_errors = (predicted[:, :nc] - target[:, :nc]).abs().mean(dim=-1)
        # Physical alpha_i replaces the grid distance weights only for whole-time-row
        # masking (see _em_context_alpha). Per-sample rescaling to the grid weights'
        # own mean keeps the term's magnitude identical to the geometric branch, so
        # the single variable under test is the ALLOCATION, not the coefficient.
        alpha, use_physical = self._em_context_alpha(kernel, csi_mask, nc)
        t_grid = self.online_stem.csi.grid[0]
        spatial = nc // t_grid
        weighted_sum = token_errors.new_zeros(())
        visible_count = 0
        alpha_max = 0.0
        alpha_min_visible = 0.0
        for batch_index in range(mask.shape[0]):
            vis_index = csi_visible[batch_index]
            masked_index = csi_mask[batch_index]
            if not bool(vis_index.any()) or not bool(masked_index.any()):
                continue
            distance = torch.cdist(coords[vis_index], coords[masked_index]).amin(dim=1).clamp_min(1.0)
            weights = distance.rsqrt().to(dtype=token_errors.dtype)
            if use_physical and alpha is not None:
                # repeat_interleave, not unsqueeze+expand+reshape: the latter forces a
                # copy of a zero-stride view whose result carried a storage offset that
                # broke downstream bf16 cuBLAS alignment.
                token_alpha = alpha[batch_index].repeat_interleave(spatial).contiguous()
                physical = token_alpha[vis_index].to(dtype=token_errors.dtype).contiguous()
                normalizer = physical.mean().clamp_min(1e-8)
                physical = physical * (weights.mean() / normalizer)
                alpha_max = max(alpha_max, float(alpha[batch_index].max()))
                alpha_min_visible = min(
                    alpha_min_visible if visible_count else float("inf"),
                    float(token_alpha[vis_index].min()),
                )
                weights = physical
            errors = token_errors[batch_index, vis_index]
            if self.context_loss_exp != 1.0:
                errors = errors.pow(self.context_loss_exp) / self.context_loss_exp
            weighted_sum = weighted_sum + (errors * weights).sum()
            visible_count += int(errors.numel())
        if visible_count <= 0:
            return None
        self._context_weight_physical = 1.0 if (use_physical and alpha is not None) else 0.0
        self._context_alpha_stats = (
            {"context_alpha_max": alpha_max, "context_alpha_min_visible": alpha_min_visible}
            if use_physical and alpha is not None
            else {}
        )
        return weighted_sum / float(visible_count)

    def _angular_power_spectrum(self, h_raw: torch.Tensor) -> torch.Tensor:
        """32-beam DFT angular power spectrum per timestep, matching the beam
        downstream label exactly (see dft_beam_power in the eval): raw complex CSI
        [B, T, 2, 32, 32] -> antenna-axis DFT -> power summed over subcarriers.

        Returns [B, T, num_beams]. The antenna axis is dim -2 (the eval reshapes
        [B, 2, 32, 32] to [B, 32(ant), 4, 8] and contracts the leading 32).

        Implemented with REAL-valued matmuls only. A complex64 version
        (torch.complex + complex einsum) corrupted the CUDA context and crashed
        asynchronously in a later backward() with "CUDA error: invalid argument"
        / "misaligned address": the complex tensor was built from non-contiguous
        slices, so the resulting layout violated cuBLAS complex-gemm alignment.
        A control run with --pretrain-angular-weight 0.0 and this code otherwise
        unchanged passed step 80 cleanly, isolating the fault to this function.
        Expanding into real/imag parts is algebraically identical:
            |sum_n (hr+i*hi)(cr+i*ci)|^2 = (hr@cr - hi@ci)^2 + (hr@ci + hi@cr)^2
        """
        num_beams = self.pretrain_angular_beams
        # .contiguous() matters: h_raw[:, :, 0] is a strided view.
        hr = h_raw[:, :, 0].float().contiguous()  # [B,T,32(ant),32(sc)]
        hi = h_raw[:, :, 1].float().contiguous()
        antenna = torch.arange(32, device=h_raw.device, dtype=torch.float32)
        beam = torch.arange(num_beams, device=h_raw.device, dtype=torch.float32)
        phase = 2.0 * math.pi * beam[:, None] * antenna[None, :] / float(num_beams)
        scale = 1.0 / math.sqrt(32.0)
        cr = (torch.cos(phase) * scale).contiguous()  # [K,32]
        ci = (torch.sin(phase) * scale).contiguous()
        # Contract the antenna axis: [B,T,K,sc] = [K,ant] x [B,T,ant,sc]
        real = torch.einsum("btns,kn->btks", hr, cr) - torch.einsum("btns,kn->btks", hi, ci)
        imag = torch.einsum("btns,kn->btks", hr, ci) + torch.einsum("btns,kn->btks", hi, cr)
        power = real.square() + imag.square()
        return power.sum(dim=-1).float()  # [B,T,num_beams]

    def _position_aux_loss(
        self,
        ctx: torch.Tensor,
        traj: torch.Tensor,
        lengths: Tuple[int, int, int],
        dropped: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Regress the UE position at the last context step from CSI + point tokens.

        Scored only on samples whose point_origin was zeroed (`dropped`), because on
        the remaining samples the position is still available for free and the loss
        would be trivially satisfied without teaching anything.

        Trajectory tokens are excluded from the pooled input so the target cannot be
        read straight out of the input, matching the localization downstream, which
        masks all trajectory tokens.
        """
        nc, np_, _ = lengths
        if not bool(dropped.any()):
            return ctx.new_zeros((), dtype=torch.float32), {}
        with torch.autocast(device_type=ctx.device.type, enabled=False):
            feats = ctx[:, : nc + np_].float()
            score = self.pretrain_position_score(feats)
            weights = torch.softmax(score, dim=1)
            pooled = (weights * feats).sum(dim=1)
            pred = self.pretrain_position_head(pooled)
            target = traj[:, self.context_steps - 1, :3].float()
            sel = dropped.to(ctx.device)
            per = F.mse_loss(pred[sel], target[sel], reduction="none").sum(dim=-1)
            loss = per.mean()
            with torch.no_grad():
                # pos_scale-normalised units; multiply by pos_scale for metres.
                err = torch.linalg.vector_norm(pred[sel][:, :2] - target[sel][:, :2], dim=1)
                mean_err = err.mean()
                frac = sel.float().mean()
        return loss, {"position_err_norm": mean_err, "position_dropped_frac": frac}

    @staticmethod
    def _point_cloud_xyz(point_cloud: Any) -> Optional[torch.Tensor]:
        """Return the local cloud as [B, M, 3] in the UE-centred normalized frame.

        Accepts every shape the point pipeline emits: a raw tensor, {"points": ...},
        or the cached {"neighborhood", "center"} grouping (in which case the cloud is
        rebuilt as neighborhood + center, the exact inverse of PointBERTGroup, giving
        num_group*group_size points with the kNN overlap the tokenizer itself sees).
        """
        if torch.is_tensor(point_cloud):
            return point_cloud
        if not isinstance(point_cloud, dict):
            return None
        if torch.is_tensor(point_cloud.get("points")):
            return point_cloud["points"]
        group = point_cloud.get("point_group") if isinstance(point_cloud.get("point_group"), dict) else point_cloud
        neighborhood = group.get("neighborhood") if isinstance(group, dict) else None
        center = group.get("center") if isinstance(group, dict) else None
        if torch.is_tensor(neighborhood) and torch.is_tensor(center):
            full = neighborhood + center.unsqueeze(2)
            return full.reshape(full.shape[0], -1, 3)
        return None

    def _point_height_aux_loss(
        self,
        pred: torch.Tensor,
        point_cloud: Any,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Recover the vertical mass distribution of the hidden point cloud.

        Scored only on samples whose point tokens are ALL masked (the `point` task).
        Any visible point token would carry the answer through the dVAE features, so
        partially-masked samples are dropped rather than supervised -- the same
        leakage guard the position aux applies with `origin_dropped`.

        Two terms, both computed from the PREDICTOR output at the masked point slots:
          * KL over an L-layer height histogram (the layer occupancy probe target)
          * smooth-L1 on [mean z, std z, p90 z], normalized by pretrain_height_scale_m

        `height_const_kl` logs the KL of the batch-mean histogram, i.e. what a head
        that ignores its input entirely would score. The gap between it and
        `height_kl` is the only evidence that the term learns anything sample
        specific; without it a falling loss proves nothing but prior fitting.
        """
        nc, np_, _ = lengths
        zero = pred.new_zeros((), dtype=torch.float32)
        if np_ <= 0:
            return zero, {}
        point_mask = mask[:, nc : nc + np_]
        full_masked = point_mask.all(dim=1)
        if not bool(full_masked.any()):
            return zero, {}
        xyz = self._point_cloud_xyz(point_cloud)
        if xyz is None:
            return zero, {}
        scale = point_cloud.get("point_scale") if isinstance(point_cloud, dict) else None
        if not torch.is_tensor(scale):
            return zero, {}
        sel = full_masked
        # Targets stay in fp32 with autocast off: bucketize/quantile on bf16 heights
        # silently collapses adjacent layers once |z| exceeds a few metres.
        with torch.no_grad(), torch.autocast(device_type=pred.device.type, enabled=False):
            # point_scale is the unit-sphere radius divided by pos_scale, so
            # z_norm * point_scale * pos_scale restores metres relative to the UE.
            z = (
                xyz[sel][..., 2].float()
                * scale[sel].float().reshape(-1, 1)
                * float(self.pos_scale)
            )
            edges = self.pretrain_height_edges.to(z.device, torch.float32)
            layer = torch.bucketize(z, edges)  # [n, M] in [0, L-1]
            hist = torch.zeros(z.shape[0], self.pretrain_height_layers, device=z.device, dtype=torch.float32)
            hist.scatter_add_(1, layer, torch.ones_like(z))
            target_hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
            hs = max(self.pretrain_height_scale_m, 1e-6)
            target_stats = torch.stack(
                [
                    z.mean(dim=1) / hs,
                    z.std(dim=1, unbiased=False) / hs,
                    torch.quantile(z, 0.9, dim=1) / hs,
                ],
                dim=-1,
            )
        feats = pred[:, nc : nc + np_][sel].contiguous()
        weights = torch.softmax(self.pretrain_height_score(feats).float(), dim=1)
        pooled = (weights * feats.float()).sum(dim=1).to(feats.dtype)
        out = self.pretrain_height_head(pooled).float()
        log_pred = F.log_softmax(out[:, : self.pretrain_height_layers], dim=-1)
        kl = F.kl_div(log_pred, target_hist, reduction="none").sum(dim=-1).mean()
        stat = F.smooth_l1_loss(out[:, self.pretrain_height_layers :], target_stats)
        loss = kl + self.pretrain_height_stat_weight * stat
        with torch.no_grad():
            const = target_hist.mean(dim=0, keepdim=True).clamp_min(1e-8).expand_as(target_hist)
            const_kl = F.kl_div(const.log(), target_hist, reduction="none").sum(dim=-1).mean()
            stats = {
                "height_kl": kl,
                "height_const_kl": const_kl,
                "height_stat": stat,
                "height_samples": full_masked.float().sum(),
                "height_ground_frac": target_hist[:, 0].mean(),
            }
        return loss, stats

    def _angular_aux_loss(
        self,
        ctx: torch.Tensor,
        h_seq: torch.Tensor,
        mask: torch.Tensor,
        lengths: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """KL divergence between the true and predicted normalized angular power
        spectrum, over tubelet time groups whose CSI tokens are all visible.

        Only fully-visible groups are supervised because the beam downstream feeds
        the encoder output of visible tokens; masked groups are the JEPA/recon
        objective's business. The spectrum is normalized to a distribution, which
        makes the target invariant to the per-sample context-RMS scale that the
        data pipeline divides out.
        """
        nc = lengths[0]
        total_steps = self.context_steps + self.future_steps
        tg = total_steps // self.patch_t
        spatial = nc // max(tg, 1)
        if spatial <= 0 or nc <= 0:
            zero = h_seq.new_zeros((), dtype=torch.float32)
            return zero, {}
        csi_mask = mask[:, :nc]
        # A group is supervised only when none of its CSI tokens are masked.
        group_visible = ~csi_mask.reshape(csi_mask.shape[0], tg, spatial).any(dim=-1)  # [B,tg]
        if not bool(group_visible.any()):
            zero = h_seq.new_zeros((), dtype=torch.float32)
            return zero, {}
        # Target construction stays in fp32 because signed-log inversion uses
        # expm1 and the DFT spectrum is numerically sensitive. The trainable head
        # deliberately remains in the caller's bf16 autocast region. Forcing its
        # backward path to fp32 while it reconnects to the bf16 encoder scatter
        # intermittently corrupts the Windows CUDA context after tens of steps.
        with torch.no_grad(), torch.autocast(device_type=h_seq.device.type, enabled=False):
            raw = self._to_raw_csi(h_seq.float())  # [B,T,2,32,32], relative scale
            spec = self._angular_power_spectrum(raw)  # [B,T,K]
            # Average power over the patch_t frames a group covers, then normalize.
            spec = spec.reshape(spec.shape[0], tg, self.patch_t, -1).mean(dim=2)
            target = spec / spec.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = ctx[:, :nc].reshape(ctx.shape[0], tg, spatial, ctx.shape[-1]).mean(dim=2)
        logits = self.pretrain_angular_head(pooled)
        # Keep the probability arithmetic in fp32 while preserving the bf16 head.
        log_pred = F.log_softmax(logits.float(), dim=-1)
        per_group = F.kl_div(log_pred, target, reduction="none").sum(dim=-1)  # [B,tg]
        weight = group_visible.to(per_group.dtype)
        denom = float(weight.sum().item())
        if denom <= 0.0:
            return h_seq.new_zeros((), dtype=torch.float32), {}
        loss = (per_group * weight).sum() / denom
        with torch.no_grad():
            hit = (log_pred.argmax(-1) == target.argmax(-1)).to(per_group.dtype)
            top1 = (hit * weight).sum() / denom
        return loss, {"angular_top1": top1, "angular_groups": weight.sum()}

    def _patch_reconstruction_loss(
        self,
        pred_patch: torch.Tensor,
        true_patch: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        predicted = pred_patch[token_mask]
        target = true_patch[token_mask]
        raw_mse = F.mse_loss(predicted, target)
        recon = raw_mse
        zero = raw_mse.new_zeros(())
        components = {"mse": raw_mse, "magnitude": zero, "phase": zero}
        if self.pretrain_recon_mag_weight <= 0 and self.pretrain_recon_phase_weight <= 0:
            return recon, components
        predicted_complex = predicted.float().view(-1, 2, self.patch_t, self.patch_h, self.patch_w)
        target_complex = target.float().view(-1, 2, self.patch_t, self.patch_h, self.patch_w)
        if self.pretrain_recon_mag_weight > 0:
            predicted_mag = torch.sqrt(predicted_complex[:, 0] ** 2 + predicted_complex[:, 1] ** 2 + 1e-8)
            target_mag = torch.sqrt(target_complex[:, 0] ** 2 + target_complex[:, 1] ** 2 + 1e-8)
            magnitude_loss = F.mse_loss(predicted_mag, target_mag)
            recon = recon + self.pretrain_recon_mag_weight * magnitude_loss
            components["magnitude"] = magnitude_loss
        if self.pretrain_recon_phase_weight > 0:
            # signed-log is applied independently to real and imaginary parts, so
            # atan2 in transformed space is not the physical channel phase. Restore
            # physical CSI first, then use a smoothed unit-vector cosine whose
            # derivative is bounded even when the prediction is near zero.
            predicted_physical = self._to_raw_csi(predicted_complex)
            target_physical = self._to_raw_csi(target_complex)
            predicted_real, predicted_imag = predicted_physical[:, 0], predicted_physical[:, 1]
            target_real, target_imag = target_physical[:, 0], target_physical[:, 1]
            target_amplitude = torch.sqrt(
                target_real.square() + target_imag.square()
            ).detach()
            amplitude_rms = target_amplitude.square().mean().sqrt().detach().clamp_min(1e-12)
            phase_eps = amplitude_rms * max(self.pretrain_phase_eps_ratio, 1e-4)
            predicted_norm = torch.sqrt(predicted_real.square() + predicted_imag.square() + phase_eps.square())
            target_norm = torch.sqrt(target_real.square() + target_imag.square() + phase_eps.square())
            phase_cosine = (
                (predicted_real * target_real + predicted_imag * target_imag)
                / (predicted_norm * target_norm).clamp_min(1e-12)
            ).clamp(min=-1.0, max=1.0)
            phase_weights = target_amplitude / target_amplitude.mean().clamp_min(1e-6)
            phase_weights = phase_weights.clamp(max=10.0)
            phase_error = 1.0 - phase_cosine
            phase_loss = (phase_error * phase_weights).sum() / phase_weights.sum().clamp_min(1.0)
            recon = recon + self.pretrain_recon_phase_weight * phase_loss
            components["phase"] = phase_loss
        return recon, components

    def pretrain_loss(
        self,
        h_seq: torch.Tensor,
        point_cloud: torch.Tensor,
        traj: torch.Tensor,
        mask: torch.Tensor,
        pretrain_task: Optional[str] = None,
        em_path: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred, target, lengths, ctx, deep_state = self.encode_predict(h_seq, point_cloud, traj, mask)
        latent_loss = F.l1_loss(pred[mask], target[mask])
        # Store components for logging
        self._loss_components = {
            "latent": float(latent_loss.detach()), "recon": 0.0, "sgcs1m": 0.0,
            "sigreg": 0.0, "visreg": 0.0, "context": 0.0, "em": 0.0,
            "deep": 0.0, "visible_recon": 0.0,
            "recon_mse": 0.0, "recon_magnitude": 0.0, "recon_phase": 0.0,
            "visible_recon_mse": 0.0, "visible_recon_magnitude": 0.0, "visible_recon_phase": 0.0,
        }
        nc, np_, nt = lengths
        with torch.no_grad():
            for name, segment in zip(
                ("csi", "point", "traj"),
                torch.split(target.detach().float(), [nc, np_, nt], dim=1),
            ):
                pooled = segment.mean(dim=1)
                feature_std = pooled.std(dim=0, unbiased=False).mean()
                paired_cos = F.cosine_similarity(pooled, torch.roll(pooled, shifts=1, dims=0), dim=-1).mean()
                self._loss_components["repr_std_%s" % name] = float(feature_std)
                self._loss_components["repr_cos_%s" % name] = float(paired_cos)
        base = latent_loss
        # Change B — per-modality SIGReg (anti-collapse; additive to the JEPA loss).
        sigreg_term = self._sigreg_pretrain(ctx, pred, mask, lengths)
        if sigreg_term is not None:
            base = base + sigreg_term
            self._loss_components["sigreg"] = float(sigreg_term.detach())
            for _k, _v in getattr(self, "_sigreg_raw", {}).items():
                self._loss_components["sigreg_raw_%s" % _k] = _v
        visreg_term = self._visreg_pretrain(ctx, pred, mask, lengths)
        if visreg_term is not None:
            base = base + visreg_term
            self._loss_components["visreg"] = float(visreg_term.detach())
            for key, value in getattr(self, "_visreg_raw", {}).items():
                self._loss_components["visreg_raw_%s" % key] = value

        # Within-sample point-token spread (v22). See _point_spread_pretrain.
        self._loss_components["pspread"] = 0.0
        spread_term = self._point_spread_pretrain(pred, target, mask, lengths, context=ctx)
        if spread_term is not None and torch.isfinite(spread_term):
            base = base + spread_term
            self._loss_components["pspread"] = float(spread_term.detach())
            for key, value in getattr(self, "_point_spread_raw", {}).items():
                self._loss_components["pspread_%s" % key] = value

        # EM physical constraints (v17). One kernel per batch, shared by the physical
        # context weights, relation matching and K-SIGReg.
        em_kernel = self.em_kernel_for_batch(traj, self.online_stem.csi.grid[0])
        self._context_weight_physical = 0.0
        self._context_alpha_stats = {}
        if em_kernel is not None:
            em_term, em_stats = self._em_physics_pretrain(ctx, pred, mask, lengths, em_kernel)
            direct_term = None
            direct_stats: Dict[str, float] = {}
            if self.em_direct_apply_on in ("predictor", "both"):
                direct_term, direct_stats = self._em_direct_anticollapse_pretrain(
                    h_seq, traj, pred, lengths, stats_prefix="em_pred_"
                )
            if self.em_direct_apply_on in ("context", "both"):
                context_direct, context_stats = self._em_direct_anticollapse_pretrain(
                    h_seq, traj, ctx, lengths, visible=~mask, stats_prefix="em_ctx_"
                )
                if context_direct is not None:
                    direct_term = context_direct if direct_term is None else direct_term + context_direct
                direct_stats.update(context_stats)
                if self.em_direct_multilevel:
                    context_levels = deep_state.get("context_levels", [])
                    for level_index, level in enumerate(context_levels[:-1]):
                        level_term, level_stats = self._em_direct_anticollapse_pretrain(
                            h_seq, traj, level, lengths, visible=~mask,
                            stats_prefix="em_ctx_l%d_" % level_index,
                        )
                        if level_term is not None:
                            scaled = self.em_direct_level_weight * level_term
                            direct_term = scaled if direct_term is None else direct_term + scaled
                        direct_stats.update(level_stats)
            if direct_term is not None:
                em_term = direct_term if em_term is None else em_term + direct_term
            em_stats.update(direct_stats)
            conditional_term, conditional_stats = self._em_conditional_pretrain(
                h_seq, ctx, pred, mask, lengths, em_path=em_path
            )
            if conditional_term is not None:
                em_term = conditional_term if em_term is None else em_term + conditional_term
            em_stats.update(conditional_stats)
            em_factor = self._warmup_factor(self._pretrain_step, self.em_physics_warmup_steps)
            self._loss_components["em_warmup"] = float(em_factor)
            with torch.no_grad():
                # Diagnostics only: the eigendecomposition here is logging, so run it
                # on the CPU copy to keep batched-linalg off the training stream.
                diagnostics = kernel_diagnostics(em_kernel.detach().cpu(), self.em_kernel_energy)
            for name, value in diagnostics.items():
                self._loss_components["em_kernel_%s" % name] = float(value)
            self._loss_components["em_speed_mean"] = float(getattr(self, "_em_speed", em_kernel.new_zeros(1)).mean())
            for name, value in em_stats.items():
                self._loss_components[name] = value
            if em_term is not None and torch.isfinite(em_term):
                weighted_em = em_factor * em_term.to(base.dtype)
                base = base + weighted_em
                self._loss_components["em"] = float(weighted_em.detach())

        pred_levels = deep_state.get("pred_levels", [])
        target_levels = deep_state.get("target_levels", [])
        if self.deep_supervision_weight > 0 and len(pred_levels) > 1:
            deep_terms = [
                F.l1_loss(predicted_level[mask], target_level[mask])
                for predicted_level, target_level in zip(pred_levels[:-1], target_levels[:-1])
            ]
            deep_term = self.deep_supervision_weight * torch.stack(deep_terms).mean()
            base = base + deep_term
            self._loss_components["deep"] = float(deep_term.detach())

        context_source = ctx if self.context_loss_source == "encoder" else pred
        context_term = self._context_loss(context_source, target, mask, lengths, kernel=em_kernel)
        if context_term is not None:
            context_levels = deep_state.get("context_levels", [])
            source_levels = context_levels if self.context_loss_source == "encoder" else pred_levels
            auxiliary_context = [
                self._context_loss(source_level, target_level, mask, lengths, kernel=em_kernel)
                for source_level, target_level in zip(source_levels[:-1], target_levels[:-1])
            ]
            auxiliary_context = [term for term in auxiliary_context if term is not None]
            if auxiliary_context:
                context_term = torch.stack([context_term] + auxiliary_context).mean()
            context_factor = self._warmup_factor(self._pretrain_step, self.context_loss_warmup_steps)
            context_weight = self.context_loss_weight * context_factor
            weighted_context = context_weight * context_term
            base = base + weighted_context
            self._loss_components["context"] = float(weighted_context.detach())
            self._loss_components["context_weight"] = float(context_weight)
            # 1.0 = physical alpha_i in use, 0.0 = grid weights, 0.5 = mixed across
            # the gradient-accumulation window (temporal + fine/coarse steps).
            self._loss_components["context_weight_physical"] = float(
                getattr(self, "_context_weight_physical", 0.0)
            )
            for stat_name, stat_value in getattr(self, "_context_alpha_stats", {}).items():
                self._loss_components[stat_name] = float(stat_value)
        # Angular auxiliary supervision. Added BEFORE the recon early-return so it
        # also applies on geo steps (which run with recon weight 0 in the v10 recipe).
        if self.pretrain_angular_weight > 0:
            angular_factor = self._warmup_factor(self._pretrain_step, self.pretrain_angular_warmup_steps)
            angular_weight = self.pretrain_angular_weight * angular_factor
            angular_term, angular_stats = self._angular_aux_loss(ctx, h_seq, mask, lengths)
            weighted_angular = angular_weight * angular_term
            base = base + weighted_angular
            self._loss_components["angular"] = float(weighted_angular.detach())
            self._loss_components["angular_weight"] = float(angular_weight)
            for stat_name, stat_value in angular_stats.items():
                self._loss_components[stat_name] = float(stat_value.detach())
        # Position auxiliary supervision, also before the recon early-return so geo
        # steps (recon weight 0 in the v10/v15 recipe) still contribute.
        if self.pretrain_position_weight > 0 and isinstance(point_cloud, dict):
            dropped = point_cloud.get("origin_dropped")
            if dropped is not None:
                pos_factor = self._warmup_factor(self._pretrain_step, self.pretrain_position_warmup_steps)
                pos_weight = self.pretrain_position_weight * pos_factor
                pos_term, pos_stats = self._position_aux_loss(ctx, traj, lengths, dropped)
                weighted_pos = pos_weight * pos_term
                base = base + weighted_pos
                self._loss_components["position"] = float(weighted_pos.detach())
                self._loss_components["position_weight"] = float(pos_weight)
                for stat_name, stat_value in pos_stats.items():
                    self._loss_components[stat_name] = float(stat_value.detach())
        # Point-cloud height auxiliary supervision (v19). Fires only on steps whose
        # point tokens are fully masked, so it costs nothing on the other five tasks.
        if self.pretrain_height_weight > 0:
            height_factor = self._warmup_factor(self._pretrain_step, self.pretrain_height_warmup_steps)
            height_weight = self.pretrain_height_weight * height_factor
            height_term, height_stats = self._point_height_aux_loss(pred, point_cloud, mask, lengths)
            weighted_height = height_weight * height_term
            base = base + weighted_height
            self._loss_components["height"] = float(weighted_height.detach())
            self._loss_components["height_weight"] = float(height_weight)
            for stat_name, stat_value in height_stats.items():
                self._loss_components[stat_name] = float(stat_value.detach())
        active_recon_weight = (
            self.pretrain_geo_recon_weight if pretrain_task == "geo" else self.pretrain_recon_weight
        )
        self._loss_components["recon_weight"] = float(active_recon_weight)
        if active_recon_weight <= 0:
            return base
        nc = lengths[0]
        csi_mask = mask[:, :nc]
        if not bool(csi_mask.any()):
            return base
        pred_head = self.pretrain_csi_head(pred[:, :nc])
        true_patch = patchify_csi_tokens(h_seq, self.patch_t, self.patch_h, self.patch_w).to(pred_head.dtype)
        if self.pretrain_residual_head:
            # Residual/delta prediction: anchor = the last VISIBLE context frame's
            # CSI patch (a real, non-zero signal), tiled across every t_grid row and
            # spatial position. pred = anchor + head(delta). This exploits the
            # copy-anchor ~0.91 SGCS as a free starting point (head zero-init => at
            # step 0 pred == anchor == copy baseline) and, unlike a previous-row
            # anchor, never yields a near-zero anchor (which made the atan2 phase
            # gradient explode to NaN). Zero leakage: anchor is a context frame only.
            total_steps = self.context_steps + self.future_steps
            tg = total_steps // self.patch_t
            last_ctx = h_seq[:, self.context_steps - self.patch_t:self.context_steps]
            anc_row = patchify_csi_tokens(last_ctx, self.patch_t, self.patch_h, self.patch_w).to(pred_head.dtype)  # [B, hgwg, D]
            anchor = anc_row.repeat(1, tg, 1)[:, :nc, :]
            if self.temporal_anchor == "phasor":
                # Replace the future-tubelet rows with the phasor extrapolation
                # (EM inductive bias); context rows keep the copy anchor.
                ph = self.phasor_anchor_patch(h_seq).to(pred_head.dtype)
                anchor[:, -ph.shape[1]:] = ph
            elif self.temporal_anchor == "coherent":
                # WRONG EM assumption: static channel. Future rows carry the
                # coherent mean of the context frames (phase-walk smearing).
                mean_c = self._to_raw_csi(h_seq)[:, : self.context_steps].mean(dim=1, keepdim=True)
                stored = self._from_raw_csi(mean_c.expand(-1, self.future_steps, -1, -1, -1).contiguous())
                co_patch = patchify_csi_tokens(stored, self.patch_t, self.patch_h, self.patch_w)
                co_rows = co_patch.reshape(h_seq.shape[0], -1, co_patch.shape[-1]).to(pred_head.dtype)
                anchor[:, -co_rows.shape[1]:] = co_rows
            # Clamp the residual delta: CSI in signed_log space is ~±4 (csi_std~4.2),
            # so a runaway head output makes anchor+delta overflow the SGCS/phase
            # power-iteration (expm1) -> NaN after a few hundred steps. Bounding the
            # delta to ±8 keeps the objective stable without limiting real corrections.
            pred_head = torch.clamp(pred_head, min=-8.0, max=8.0)
            pred_patch = anchor + pred_head
        else:
            pred_patch = pred_head
        recon, recon_components = self._patch_reconstruction_loss(pred_patch, true_patch, csi_mask)
        recon_weighted = active_recon_weight * recon
        total = base + recon_weighted  # `base` already folds in the SIGReg term (change B)
        self._loss_components["recon"] = float(recon_weighted.detach())
        for component_name, component_value in recon_components.items():
            self._loss_components["recon_%s" % component_name] = float(component_value.detach())
        visible_csi = ~csi_mask
        if self.pretrain_visible_recon_weight > 0 and bool(visible_csi.any()):
            context_patch = self.pretrain_context_csi_head(ctx[:, :nc])
            visible_recon, visible_components = self._patch_reconstruction_loss(context_patch, true_patch, visible_csi)
            visible_weighted = self.pretrain_visible_recon_weight * visible_recon
            total = total + visible_weighted
            self._loss_components["visible_recon"] = float(visible_weighted.detach())
            for component_name, component_value in visible_components.items():
                self._loss_components["visible_recon_%s" % component_name] = float(component_value.detach())
        # Explicit SGCS loss: directly optimize the paper's target metric on the
        # head-reconstructed CSI. MSE+mag+phase plateaued head-SGCS at ~0.70; this
        # pushes the dominant right-singular-vector alignment that SGCS measures.
        if self.pretrain_sgcs_weight > 0:
            total_steps = self.context_steps + self.future_steps
            pred_transformed = unpatchify_csi_tokens(pred_patch.float(), total_steps, self.patch_t, self.patch_h, self.patch_w)
            true_transformed = unpatchify_csi_tokens(true_patch.float(), total_steps, self.patch_t, self.patch_h, self.patch_w)
            pred_full = self._to_raw_csi(pred_transformed)
            true_full = self._to_raw_csi(true_transformed)
            # per-sample masked-timestep weight: a timestep contributes where its
            # CSI tokens were masked (temporal task masks whole timesteps).
            tg = total_steps // self.patch_t
            step_masked = csi_mask.reshape(csi_mask.shape[0], tg, -1).any(dim=-1)  # [B, tg]
            sgcs_terms = []
            for tgi in range(tg):
                if not bool(step_masked[:, tgi].any()):
                    continue
                active_samples = step_masked[:, tgi]
                for sub in range(self.patch_t):
                    si = tgi * self.patch_t + sub
                    sgcs_terms.append(
                        sgcs_metric_at_step(pred_full[active_samples], true_full[active_samples], si, mode="power")
                    )
            if sgcs_terms:
                sgcs_val = torch.stack(sgcs_terms).mean()
                sgcs_factor = self._warmup_factor(self._pretrain_step, self.pretrain_sgcs_warmup_steps)
                sgcs_weight = self.pretrain_sgcs_weight * sgcs_factor
                sgcs_term = sgcs_weight * (1.0 - sgcs_val)
                # Drop the SGCS term if the power-iteration produced a non-finite
                # value (degenerate complex matrix); keep training on latent+recon.
                if torch.isfinite(sgcs_term):
                    total = total + sgcs_term
                    self._loss_components["sgcs1m"] = float(sgcs_term.detach())
                    self._loss_components["sgcs_weight"] = float(sgcs_weight)
        return total
