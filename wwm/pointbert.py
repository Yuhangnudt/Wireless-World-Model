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



def farthest_point_sample(points: torch.Tensor, count: int) -> torch.Tensor:
    b, n, _ = points.shape
    count = min(int(count), int(n))
    centroids = torch.zeros(b, count, dtype=torch.long, device=points.device)
    distance = torch.full((b, n), float("inf"), dtype=torch.float32, device=points.device)
    farthest = torch.zeros(b, dtype=torch.long, device=points.device)
    batch_indices = torch.arange(b, dtype=torch.long, device=points.device)
    for idx in range(count):
        centroids[:, idx] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1).indices
    return centroids


class PointBERTGroup(nn.Module):
    def __init__(self, num_group: int, group_size: int, center_sampling: str) -> None:
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)
        self.center_sampling = str(center_sampling)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        if num_points < self.num_group:
            raise ValueError("point_count must be >= point_tokens")
        if self.center_sampling == "linspace":
            center_idx = torch.linspace(0, num_points - 1, self.num_group, device=xyz.device).round().long()
            center_idx = center_idx.unsqueeze(0).expand(batch_size, -1)
        elif self.center_sampling == "fps":
            center_idx = farthest_point_sample(xyz.float(), self.num_group)
        else:
            raise ValueError("Unknown point center sampling: %s" % self.center_sampling)
        center_gather = center_idx.unsqueeze(-1).expand(batch_size, self.num_group, 3)
        center = torch.gather(xyz, 1, center_gather)
        group_size = min(self.group_size, num_points)
        group_idx = knn_point(group_size, xyz, center)
        gather_points = xyz[:, None, :, :].expand(batch_size, self.num_group, num_points, 3)
        gather_idx = group_idx.unsqueeze(-1).expand(batch_size, self.num_group, group_size, 3)
        neighborhood = torch.gather(gather_points, 2, gather_idx).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class PointBERTEncoder(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        batch_size, num_group, group_size, _ = point_groups.shape
        point_groups = point_groups.reshape(batch_size * num_group, group_size, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True).values
        feature = torch.cat([feature_global.expand(-1, -1, group_size), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False).values
        return feature_global.reshape(batch_size, num_group, self.encoder_channel)


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(new_xyz.float(), xyz.float())
    return torch.topk(dist, k=min(int(nsample), xyz.shape[1]), dim=-1, largest=False, sorted=False).indices


class PointBERTDGCNN(nn.Module):
    def __init__(self, encoder_channel: int, output_channel: int) -> None:
        super().__init__()
        self.input_trans = nn.Conv1d(int(encoder_channel), 128, 1)
        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.GroupNorm(4, 256), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False), nn.GroupNorm(4, 512), nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, 1, bias=False), nn.GroupNorm(4, 512), nn.LeakyReLU(0.2))
        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False), nn.GroupNorm(4, 1024), nn.LeakyReLU(0.2))
        self.layer5 = nn.Sequential(
            nn.Conv1d(2304, int(output_channel), 1, bias=False),
            nn.GroupNorm(4, int(output_channel)),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def get_knn_idx(coor_q: torch.Tensor, coor_k: torch.Tensor) -> torch.Tensor:
        k = min(4, int(coor_k.shape[2]))
        with torch.no_grad():
            dist = torch.cdist(coor_q.transpose(1, 2).float(), coor_k.transpose(1, 2).float())
            return torch.topk(dist, k=k, dim=-1, largest=False, sorted=False).indices

    @staticmethod
    def get_graph_feature(
        coor_q: torch.Tensor,
        x_q: torch.Tensor,
        coor_k: torch.Tensor,
        x_k: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if idx is None:
            idx = PointBERTDGCNN.get_knn_idx(coor_q, coor_k)
        k = int(idx.shape[-1])
        batch_size, channels, num_points_q = x_q.shape
        x_k_t = x_k.transpose(1, 2).contiguous()
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, channels)
        feature = torch.gather(x_k_t.unsqueeze(1).expand(-1, num_points_q, -1, -1), 2, gather_idx)
        feature = feature.permute(0, 3, 1, 2).contiguous()
        x_q = x_q.view(batch_size, channels, num_points_q, 1).expand(-1, -1, -1, k)
        return torch.cat((feature - x_q, x_q), dim=1)

    def forward(self, f: torch.Tensor, coor: torch.Tensor) -> torch.Tensor:
        feature_list = []
        coor = coor.transpose(1, 2).contiguous()
        f = f.transpose(1, 2).contiguous()
        f = self.input_trans(f)
        knn_idx = self.get_knn_idx(coor, coor)

        f = self.get_graph_feature(coor, f, coor, f, knn_idx)
        f = self.layer1(f).max(dim=-1, keepdim=False).values
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f, knn_idx)
        f = self.layer2(f).max(dim=-1, keepdim=False).values
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f, knn_idx)
        f = self.layer3(f).max(dim=-1, keepdim=False).values
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f, knn_idx)
        f = self.layer4(f).max(dim=-1, keepdim=False).values
        feature_list.append(f)

        f = torch.cat(feature_list, dim=1)
        return self.layer5(f).transpose(-1, -2).contiguous()


class PointBERTFoldingDecoder(nn.Module):
    def __init__(self, encoder_channel: int, num_fine: int) -> None:
        super().__init__()
        self.num_fine = int(num_fine)
        self.grid_size = 2
        if self.num_fine % (self.grid_size**2) != 0:
            raise ValueError("Point-BERT FoldingNet decoder requires point_group_size divisible by 4.")
        self.num_coarse = self.num_fine // (self.grid_size**2)
        self.mlp = nn.Sequential(
            nn.Linear(int(encoder_channel), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse),
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(int(encoder_channel) + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float32)
        a = a.view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float32)
        b = b.view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.register_buffer("folding_seed", torch.cat([a, b], dim=0).view(1, 2, self.grid_size**2), persistent=False)

    def forward(self, feature_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_group, channels = feature_global.shape
        feature_global = feature_global.reshape(batch_size * num_group, channels)
        coarse = self.mlp(feature_global).reshape(batch_size * num_group, self.num_coarse, 3)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        point_feat = point_feat.reshape(batch_size * num_group, self.num_fine, 3).transpose(2, 1)
        seed = self.folding_seed.unsqueeze(2).expand(batch_size * num_group, -1, self.num_coarse, -1)
        seed = seed.reshape(batch_size * num_group, -1, self.num_fine).to(feature_global.device)
        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        center = center.reshape(batch_size * num_group, self.num_fine, 3).transpose(2, 1)
        fine = self.final_conv(feat) + center
        fine = fine.reshape(batch_size, num_group, 3, self.num_fine).transpose(-1, -2).contiguous()
        coarse = coarse.reshape(batch_size, num_group, self.num_coarse, 3).contiguous()
        return coarse, fine


def chamfer_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred.float(), target.float(), p=2)
    loss_a = dist.min(dim=2).values.mean(dim=1)
    loss_b = dist.min(dim=1).values.mean(dim=1)
    return 0.5 * (loss_a + loss_b).mean()


class PointBERTDiscreteVAE(nn.Module):
    def __init__(
        self,
        point_tokens: int,
        group_size: int,
        center_sampling: str,
        encoder_dims: int,
        codebook_size: int,
        codebook_dim: int,
        decoder_dims: int,
    ) -> None:
        super().__init__()
        self.group_size = int(group_size)
        self.num_group = int(point_tokens)
        self.encoder_dims = int(encoder_dims)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.decoder_dims = int(decoder_dims)
        self.group_divider = PointBERTGroup(self.num_group, self.group_size, center_sampling)
        self.encoder = PointBERTEncoder(self.encoder_dims)
        self.dgcnn_1 = PointBERTDGCNN(self.encoder_dims, self.codebook_size)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, self.codebook_dim))
        self.dgcnn_2 = PointBERTDGCNN(self.codebook_dim, self.decoder_dims)
        self.decoder = PointBERTFoldingDecoder(self.decoder_dims, self.group_size)

    def encode_tokens(
        self,
        points: Any,
        temperature: float = 1.0,
        hard: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(points, dict) and "neighborhood" in points and "center" in points:
            neighborhood = points["neighborhood"]
            center = points["center"]
        elif isinstance(points, dict) and "point_group" in points:
            neighborhood = points["point_group"]["neighborhood"]
            center = points["point_group"]["center"]
        else:
            neighborhood, center = self.group_divider(points)
        encoder_tokens = self.encoder(neighborhood)
        logits = self.dgcnn_1(encoder_tokens, center)
        logit_clip = float(getattr(self, "logit_clip", 0.0))
        if logit_clip > 0:
            logits = logits.clamp(min=-logit_clip, max=logit_clip)
        if deterministic:
            probabilities = F.softmax(logits.float() / max(float(temperature), 1e-6), dim=2)
            if hard:
                indices = probabilities.argmax(dim=2)
                probabilities = F.one_hot(indices, num_classes=self.codebook_size).to(probabilities.dtype)
            soft_one_hot = probabilities.to(logits.dtype)
        else:
            soft_one_hot = F.gumbel_softmax(logits.float(), tau=float(temperature), dim=2, hard=hard).to(logits.dtype)
        sampled = torch.einsum("bgn,nc->bgc", soft_one_hot, self.codebook.to(dtype=soft_one_hot.dtype))
        refined = self.dgcnn_2(sampled, center)
        aux = {
            "neighborhood": neighborhood,
            "center": center,
            "encoder_tokens": encoder_tokens,
            "logits": logits,
            "soft_one_hot": soft_one_hot,
            "sampled": sampled,
            "refined": refined,
        }
        return refined, aux

    def forward(self, points: Any, temperature: float = 1.0, hard: bool = False) -> Dict[str, torch.Tensor]:
        tokens, aux = self.encode_tokens(points, temperature=temperature, hard=hard)
        coarse, fine = self.decoder(aux["refined"])
        center = aux["center"]
        batch_size = int(center.shape[0])
        return {
            "tokens": tokens,
            "whole_coarse": (coarse + center.unsqueeze(2)).reshape(batch_size, -1, 3),
            "whole_fine": (fine + center.unsqueeze(2)).reshape(batch_size, -1, 3),
            "coarse": coarse,
            "fine": fine,
            "neighborhood": aux["neighborhood"],
            "center": center,
            "logits": aux["logits"],
        }

    def get_loss(self, out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_group, _, _ = out["fine"].shape
        coarse = out["coarse"].reshape(batch_size * num_group, -1, 3).contiguous()
        fine = out["fine"].reshape(batch_size * num_group, -1, 3).contiguous()
        target = out["neighborhood"].reshape(batch_size * num_group, -1, 3).contiguous()
        loss_recon = chamfer_l1(coarse, target) + chamfer_l1(fine, target)
        logits = out["logits"]
        logit_clip = float(getattr(self, "logit_clip", 0.0))
        if logit_clip > 0:
            logits = logits.clamp(min=-logit_clip, max=logit_clip)
        softmax = F.softmax(logits.float(), dim=-1)
        mean_softmax = softmax.mean(dim=1).clamp_min(1e-8)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.full_like(log_qy, -math.log(float(self.codebook_size)))
        loss_kl = F.kl_div(log_qy, log_uniform, reduction="batchmean", log_target=True)
        return loss_recon, loss_kl
