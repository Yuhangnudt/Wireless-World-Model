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

from .common import csi_complex_view


def principal_right_vector_power(mat: torch.Tensor, num_iters: int = 4) -> torch.Tensor:
    mat = mat.to(torch.complex64)
    # SGCS is invariant to channel scale. Normalizing before iteration makes the
    # absolute norm floors meaningful for tiny physical CSI and avoids a 1e6
    # derivative when the channel is below the old fixed normalization floor.
    mat_scale = torch.linalg.vector_norm(mat, dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    mat = mat / mat_scale
    width = int(mat.shape[-1])
    phase = torch.arange(width, device=mat.device, dtype=mat.real.dtype) * (2.0 * math.pi / max(width, 1))
    seed = torch.polar(torch.ones_like(phase), phase).to(mat.dtype) / math.sqrt(max(width, 1))
    v = seed.unsqueeze(0).expand(mat.shape[0], -1)
    for _ in range(num_iters):
        u = torch.matmul(mat, v.unsqueeze(-1)).squeeze(-1)
        u = u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-4)
        v = torch.matmul(mat.conj().transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-4)
    return v


def sgcs_metric_at_step(pred_h: torch.Tensor, target_h: torch.Tensor, step_index: int, mode: str = "svd") -> torch.Tensor:
    pred_c = csi_complex_view(pred_h)
    target_c = csi_complex_view(target_h)
    pred_last = pred_c[:, step_index]
    target_last = target_c[:, step_index]
    b, nbs, w = pred_last.shape
    if w != 32:
        return pred_h.new_zeros(())
    pred_mat = pred_last.reshape(b, nbs, 4, 8).permute(0, 3, 2, 1).reshape(b * 8, 4, nbs)
    target_mat = target_last.reshape(b, nbs, 4, 8).permute(0, 3, 2, 1).reshape(b * 8, 4, nbs)
    if mode == "svd":
        _, _, pred_vh = torch.linalg.svd(pred_mat, full_matrices=False)
        _, _, target_vh = torch.linalg.svd(target_mat, full_matrices=False)
        pred_v = pred_vh[:, 0, :]
        target_v = target_vh[:, 0, :]
    elif mode == "power":
        pred_v = principal_right_vector_power(pred_mat)
        target_v = principal_right_vector_power(target_mat)
    else:
        raise ValueError("Unknown sgcs mode: %s" % mode)
    numerator = torch.abs(torch.sum(torch.conj(pred_v) * target_v, dim=-1)) + 1e-12
    denom = (torch.linalg.norm(pred_v, dim=-1) * torch.linalg.norm(target_v, dim=-1)).clamp_min(1e-6)
    return (numerator / denom).real.mean()


def sgcs_metric(pred_h: torch.Tensor, target_h: torch.Tensor, mode: str = "svd") -> torch.Tensor:
    return sgcs_metric_at_step(pred_h, target_h, pred_h.shape[1] - 1, mode=mode)


def sgcs_step_terms(pred_h: torch.Tensor, target_h: torch.Tensor, mode: str = "svd") -> Dict[str, torch.Tensor]:
    values = [sgcs_metric_at_step(pred_h, target_h, idx, mode=mode) for idx in range(pred_h.shape[1])]
    terms: Dict[str, torch.Tensor] = {"sgcs_h%d" % (idx + 1): value for idx, value in enumerate(values)}
    # Backward-compatible aliases for historical 14->2 logs. New 16->4 runs use
    # sgcs_h1..sgcs_h4 so the names describe forecast horizon, not absolute frame.
    if len(values) >= 1:
        terms["sgcs_t15"] = values[0]
    if len(values) >= 2:
        terms["sgcs_t16"] = values[1]
    if values:
        terms["sgcs_final"] = values[-1]
    terms["sgcs_avg"] = torch.stack(values).mean() if values else pred_h.new_zeros(())
    return terms


def select_sgcs(terms: Dict[str, torch.Tensor], selector: str) -> torch.Tensor:
    if selector == "avg":
        return terms["sgcs_avg"]
    if selector == "final":
        return terms["sgcs_final"]
    if selector.startswith("h") and selector[1:].isdigit():
        return terms["sgcs_h%d" % int(selector[1:])]
    if selector == "t15":
        return terms["sgcs_t15"]
    if selector == "t16":
        return terms["sgcs_t16"]
    raise ValueError("Unknown SGCS selector: %s" % selector)


def nmse_db_metric(pred_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
    nmse = torch.sum((pred_h - target_h).pow(2)) / (torch.sum(target_h.pow(2)) + 1e-8)
    return 10.0 * torch.log10(nmse + 1e-12)
