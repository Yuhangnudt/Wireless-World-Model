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
import shutil
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

@dataclass
class ScenarioFile:
    base: str
    h_path: str
    pos_path: str
    meta_path: str
    city_key: str
    samples: int
    bs_position: Tuple[float, float, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def optional_limit(value: Optional[int]) -> Optional[int]:
    if value is None or value <= 0:
        return None
    return int(value)


def pointbert_worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class CheckpointSaveError(RuntimeError):
    pass


def _checkpoint_size_hint(path: Path) -> int:
    candidates = [path]
    if path.name.startswith("point_dvae_step_") or path.name == "point_dvae.pt":
        candidates.append(path.parent / "point_dvae_latest.pt")
    elif path.name.startswith("checkpoint_step_") or path.name == "checkpoint.pt":
        candidates.append(path.parent / "checkpoint_latest.pt")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return int(candidate.stat().st_size)
        except OSError:
            continue
    return 0


def checkpoint_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path.parent).free)


def prune_old_checkpoints(directory: Path, pattern: str, keep_last: int) -> List[Path]:
    keep_last = max(int(keep_last), 0)
    checkpoints = sorted(directory.glob(pattern), key=lambda item: item.name)
    stale = checkpoints[:-keep_last] if keep_last else checkpoints
    removed: List[Path] = []
    for checkpoint_path in stale:
        try:
            checkpoint_path.unlink()
            removed.append(checkpoint_path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            print("checkpoint_prune_warning path=%s reason=%s" % (checkpoint_path, exc))
    return removed


def atomic_torch_save(payload: Any, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        reserve_gb = max(float(os.environ.get("WWM_CHECKPOINT_RESERVE_GB", "2.0")), 0.0)
        reserve_bytes = int(reserve_gb * (1024 ** 3))
        size_hint = _checkpoint_size_hint(path)
        free_bytes = checkpoint_free_bytes(path)
        required_bytes = reserve_bytes + size_hint
        if free_bytes < required_bytes:
            raise OSError(
                28,
                "checkpoint requires about %.2f GB free (%.2f GB payload + %.2f GB reserve), only %.2f GB available"
                % (
                    required_bytes / (1024 ** 3),
                    size_hint / (1024 ** 3),
                    reserve_bytes / (1024 ** 3),
                    free_bytes / (1024 ** 3),
                ),
            )
        torch.save(payload, tmp_path)
        last_error: Optional[BaseException] = None
        for attempt in range(20):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(min(0.25 * (attempt + 1), 3.0))
        raise last_error if last_error is not None else PermissionError("Failed to replace checkpoint: %s" % path)
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointSaveError("Failed to save checkpoint %s: %s" % (path, exc)) from exc


def atomic_json_dump(payload: Any, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        for attempt in range(20):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                time.sleep(min(0.1 * (attempt + 1), 2.0))
    except OSError:
        pass
    finally:
        # Live monitoring is non-critical and must never stop model training.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_city(base: str, meta: Dict[str, Any]) -> str:
    scenario = meta.get("scenario", {})
    if scenario.get("city_key"):
        return str(scenario["city_key"])
    for city in ("beijing_cbd", "forbidden_city", "wall_street", "munich", "etoile"):
        if city in base:
            return city
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    raise ValueError("Cannot infer city_key from %s" % base)


def infer_bs_position(meta: Dict[str, Any]) -> Tuple[float, float, float]:
    value = meta.get("bs_position_xyz_m", [0.0, 0.0, 0.0])
    if not isinstance(value, list) or len(value) != 3:
        return (0.0, 0.0, 0.0)
    return (float(value[0]), float(value[1]), float(value[2]))


def axis_sincos(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("Sinusoidal dim must be even, got %d" % dim)
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    emb = torch.zeros(length, dim, dtype=torch.float32, device=device)
    emb[:, 0::2] = torch.sin(pos * div)
    emb[:, 1::2] = torch.cos(pos * div)
    return emb


def sincos_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    return axis_sincos(length, dim, device)


def sincos_3d(grid: Tuple[int, int, int], dim: int, device: torch.device) -> torch.Tensor:
    if dim % 6 != 0:
        raise ValueError("3D sincos requires dim divisible by 6, got %d" % dim)
    t, h, w = grid
    part = dim // 3
    et = axis_sincos(t, part, device)
    eh = axis_sincos(h, part, device)
    ew = axis_sincos(w, part, device)
    out = []
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):
                out.append(torch.cat([et[ti], eh[hi], ew[wi]], dim=0))
    return torch.stack(out, dim=0)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def csi_complex_view(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def patchify_csi_tokens(h_seq: torch.Tensor, patch_t: int, patch_h: int, patch_w: int) -> torch.Tensor:
    """Tokenize CSI [B, T, 2, 32, 32] into per-token patches [B, n_tok, 2*pt*ph*pw].

    Token order (t_grid, h_grid, w_grid) and per-patch layout (2, pt, ph, pw) match
    CSITubeletEmbed's Conv3d flatten and CSITubeletDecoder's reshape, so head outputs
    align element-for-element with the CSI tokens the predictor produces.
    """
    b, t = h_seq.shape[0], h_seq.shape[1]
    tg, hg, wg = t // patch_t, 32 // patch_h, 32 // patch_w
    g = h_seq.reshape(b, tg, patch_t, 2, hg, patch_h, wg, patch_w)
    g = g.permute(0, 1, 4, 6, 3, 2, 5, 7).contiguous()
    return g.reshape(b, tg * hg * wg, 2 * patch_t * patch_h * patch_w)


def unpatchify_csi_tokens(patch: torch.Tensor, total_steps: int, patch_t: int, patch_h: int, patch_w: int) -> torch.Tensor:
    """Inverse of patchify_csi_tokens: [B, n_tok, 2*pt*ph*pw] -> CSI [B, T, 2, 32, 32]."""
    b = patch.shape[0]
    tg, hg, wg = total_steps // patch_t, 32 // patch_h, 32 // patch_w
    g = patch.reshape(b, tg, hg, wg, 2, patch_t, patch_h, patch_w)
    g = g.permute(0, 1, 5, 4, 2, 6, 3, 7).contiguous()
    return g.reshape(b, total_steps, 2, 32, 32)


def inverse_csi_transform(h: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.csi_transform == "signed_log":
        y = h.float() * float(args.csi_std) + float(args.csi_mean)
        clip = float(args.inverse_signed_log_clip)
        if clip > 0:
            y = clip * torch.tanh(y / clip)
        return torch.sign(y) * torch.expm1(torch.abs(y)) * float(args.signed_log_eps)
    return h.float()
