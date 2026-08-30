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

from .model import PaperWWM


def make_csi_block_mask(
    batch_size: int,
    grid: Tuple[int, int, int],
    num_blocks: int,
    block_shape: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    # Built on CPU with python RNG to avoid hundreds of GPU<->CPU syncs per
    # step (one .item() per randint would stall the GPU); moved to device once.
    tg, hg, wg = grid
    bt, bh, bw = block_shape
    mask = torch.zeros(batch_size, tg, hg, wg, dtype=torch.bool)
    for b in range(batch_size):
        for _ in range(num_blocks):
            t0 = random.randint(0, max(tg - bt, 0))
            h0 = random.randint(0, max(hg - bh, 0))
            w0 = random.randint(0, max(wg - bw, 0))
            mask[b, t0 : min(t0 + bt, tg), h0 : min(h0 + bh, hg), w0 : min(w0 + bw, wg)] = True
        if not bool(mask[b].any()):
            mask[b].view(-1)[random.randint(0, tg * hg * wg - 1)] = True
    return mask.flatten(1).to(device)


def block_shape(grid: Tuple[int, int, int], time_fraction: float, spatial_fraction: float) -> Tuple[int, int, int]:
    tg, hg, wg = grid
    return (
        max(1, int(math.ceil(tg * float(time_fraction)))),
        max(1, int(math.ceil(hg * float(spatial_fraction)))),
        max(1, int(math.ceil(wg * float(spatial_fraction)))),
    )


def make_temporal_row_mask(
    batch_size: int,
    grid: Tuple[int, int, int],
    min_rows: int,
    max_rows: int,
    future_bias: float,
    device: torch.device,
) -> torch.Tensor:
    """Mask ENTIRE t_grid rows (all freq/antenna CSI tokens at those timesteps).

    With probability ``future_bias`` the masked rows are the last ``r`` rows
    (future extrapolation, matching the downstream forecast); otherwise a random
    contiguous block of whole rows is masked (cross-time in-fill). At least one
    time row is always left visible, and point-cloud / trajectory tokens stay
    visible so the model must infer the unobserved timesteps from geometry,
    motion and the other timesteps. This teaches the predictor the temporal
    inference that fine/coarse spatial-block masking never exercises.
    """
    tg, hg, wg = grid
    max_rows = max(1, min(int(max_rows), tg - 1))
    min_rows = max(1, min(int(min_rows), max_rows))
    mask = torch.zeros(batch_size, tg, hg, wg, dtype=torch.bool)
    for b in range(batch_size):
        r = random.randint(min_rows, max_rows)
        if random.random() < future_bias:
            mask[b, tg - r :, :, :] = True
        else:
            t0 = random.randint(0, tg - r)
            mask[b, t0 : t0 + r, :, :] = True
    return mask.flatten(1).to(device)


def make_pretrain_mask(model: PaperWWM, args: argparse.Namespace, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, str]:
    nc, np_, nt = model.online_stem.lengths()
    full = torch.zeros(batch_size, nc + np_ + nt, dtype=torch.bool, device=device)
    grid = model.online_stem.csi.grid
    tasks = ["fine", "coarse", "traj", "point", "temporal", "geo"]
    weights = [
        float(args.mask_fine_weight),
        float(args.mask_coarse_weight),
        float(args.mask_traj_weight),
        float(getattr(args, "mask_point_weight", 0.0)),
        float(args.mask_temporal_weight),
        float(getattr(args, "mask_geo_weight", 0.0)),
    ]
    if sum(weights) <= 0:
        weights = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    task = random.choices(tasks, weights=weights, k=1)[0]
    if task == "fine":
        full[:, :nc] = make_csi_block_mask(
            batch_size,
            grid,
            args.fine_mask_blocks,
            block_shape(grid, args.fine_mask_time_fraction, args.fine_mask_spatial_fraction),
            device,
        )
    elif task == "coarse":
        full[:, :nc] = make_csi_block_mask(
            batch_size,
            grid,
            args.coarse_mask_blocks,
            block_shape(grid, args.coarse_mask_time_fraction, args.coarse_mask_spatial_fraction),
            device,
        )
    elif task == "temporal":
        full[:, :nc] = make_temporal_row_mask(
            batch_size,
            grid,
            int(getattr(args, "temporal_mask_min_rows", 1)),
            int(args.temporal_mask_max_rows),
            float(args.temporal_future_bias),
            device,
        )
    elif task == "geo":
        # Geometry->CSI grounding: mask ALL CSI tokens, keep point + trajectory
        # visible. Forces the encoder/predictor to infer the CSI latent purely
        # from geometry+motion, fusing absolute geometry into the shared
        # representation the localization/compression heads consume. (Doc 3.2.)
        full[:, :nc] = True
    elif task == "point":
        # CSI+trajectory -> point-cloud grounding. The point modality is entirely
        # unavailable to the online encoder; the predictor must reconstruct its
        # target latents from channel observations and motion alone.
        full[:, nc : nc + np_] = True
    else:
        full[:, nc + np_ :] = True
    return full, task
