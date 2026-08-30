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
import re
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

from .common import (ScenarioFile, stable_int, read_meta, infer_city,
    infer_bs_position, optional_limit)


def discover_scenarios(
    dataset_root: Path,
    split: str,
    max_files: Optional[int],
    max_samples_per_file: Optional[int],
) -> List[ScenarioFile]:
    split_root = dataset_root / split
    h_dir = split_root / "H"
    pos_dir = split_root / "pos"
    meta_dir = split_root / "meta"
    if not h_dir.exists():
        raise FileNotFoundError("Missing H directory: %s" % h_dir)

    h_files = sorted(h_dir.glob("*_H.npy"))
    if max_files is not None:
        h_files = h_files[:max_files]

    scenarios: List[ScenarioFile] = []
    for h_path in h_files:
        base = h_path.name[: -len("_H.npy")]
        pos_path = pos_dir / ("%s_pos.npy" % base)
        meta_path = meta_dir / ("%s_meta.json" % base)
        if not pos_path.exists():
            raise FileNotFoundError("Missing position file for %s: %s" % (base, pos_path))
        meta = read_meta(meta_path)
        h_shape = np.load(h_path, mmap_mode="r").shape
        samples = int(h_shape[0])
        if max_samples_per_file is not None:
            samples = min(samples, int(max_samples_per_file))
        if samples <= 0:
            continue
        scenarios.append(
            ScenarioFile(
                base=base,
                h_path=str(h_path),
                pos_path=str(pos_path),
                meta_path=str(meta_path),
                city_key=infer_city(base, meta),
                samples=samples,
                bs_position=infer_bs_position(meta),
            )
        )
    if not scenarios:
        raise RuntimeError("No scenarios found under %s" % h_dir)
    return scenarios


def scenario_speed_kmh(scenario: ScenarioFile) -> float:
    meta = read_meta(Path(scenario.meta_path))
    value = meta.get("scenario", {}).get("speed_kmh")
    if value is not None:
        return float(value)
    match = re.search(r"_(\d+(?:\.\d+)?)kmh_", scenario.base)
    if match:
        return float(match.group(1))
    raise ValueError("Cannot infer speed_kmh for %s" % scenario.base)


def filter_scenarios_by_speeds(
    scenarios: List[ScenarioFile],
    allowed_speeds: Iterable[float],
) -> List[ScenarioFile]:
    allowed = {round(float(value), 6) for value in allowed_speeds}
    if not allowed:
        return scenarios
    filtered = [scenario for scenario in scenarios if round(scenario_speed_kmh(scenario), 6) in allowed]
    if not filtered:
        raise RuntimeError("Speed filter %s removed every scenario" % sorted(allowed))
    return filtered


def load_quality_index(path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Map (scenario_base, sample_idx) -> {accepted, weight, context_rms} from quality_index.npz."""
    data = np.load(str(path), allow_pickle=True)
    bases = data["scenario_bases"]
    scenario_id = data["scenario_id"]
    sample_idx = data["sample_idx"]
    accepted = data["accepted"]
    weight = data["sampler_weight"]
    context_rms = data["context_rms"]
    lookup: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in range(scenario_id.shape[0]):
        base = str(bases[int(scenario_id[row])])
        lookup[(base, int(sample_idx[row]))] = {
            "accepted": bool(accepted[row]),
            "weight": float(weight[row]),
            "context_rms": float(context_rms[row]),
        }
    return lookup


def load_per_city_csi_stats(path: Path) -> Dict[str, Any]:
    """Load csi_stats_per_city.json; return dict with global + per-city relative_signed_log
    and log_context_rms mean/std, plus recorded signed_log_eps for validation."""
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    norm = stats.get("normalization", {})
    out: Dict[str, Any] = {
        "signed_log_eps": float(norm.get("signed_log_eps", 1.0)),
        "scope": str(norm.get("scope", "per_city")),
        "context_rms": bool(norm.get("context_rms", False)),
        "global_mean": float(stats["global"]["relative_signed_log"]["mean"]),
        "global_std": float(stats["global"]["relative_signed_log"]["std"]),
        "global_logrms_mean": float(stats["global"]["log_context_rms"]["mean"]),
        "global_logrms_std": float(stats["global"]["log_context_rms"]["std"]),
        "per_city": {},
    }
    for city, cstats in stats.get("per_city", {}).items():
        out["per_city"][city] = {
            "mean": float(cstats["relative_signed_log"]["mean"]),
            "std": float(cstats["relative_signed_log"]["std"]),
            "logrms_mean": float(cstats["log_context_rms"]["mean"]),
            "logrms_std": float(cstats["log_context_rms"]["std"]),
        }
    return out


def trajectory_features(
    pos: np.ndarray,
    bs_position: Tuple[float, float, float],
    pos_scale: float,
    mode: str,
) -> np.ndarray:
    bs = np.asarray(bs_position, dtype=np.float32)[None, :]
    centered = (pos.astype(np.float32) - bs) / float(pos_scale)
    if mode == "pos":
        return centered.astype(np.float32)
    if mode != "pos_delta":
        raise ValueError("Unknown trajectory feature mode: %s" % mode)
    delta = np.zeros_like(centered, dtype=np.float32)
    delta[1:] = (pos[1:].astype(np.float32) - pos[:-1].astype(np.float32)) / float(pos_scale)
    return np.concatenate([centered, delta], axis=1).astype(np.float32)


class WWMDataset(Dataset):
    def __init__(
        self,
        scenarios: List[ScenarioFile],
        city_root: Path,
        context_steps: int,
        future_steps: int,
        point_count: int,
        point_pool_count: int,
        point_pool_mode: str,
        point_normalization: str,
        trajectory_features_mode: str,
        pos_scale: float,
        point_scale: float,
        csi_transform: str,
        signed_log_eps: float,
        csi_mean: float,
        csi_std: float,
        drop_zero_csi_samples: bool,
        seed: int,
        normalization_scope: str = "global",
        per_city_stats: Optional[Dict[str, Any]] = None,
        context_rms_normalization: bool = False,
        context_rms_feature: bool = False,
        quality_lookup: Optional[Dict[Tuple[str, int], Dict[str, float]]] = None,
        filter_accepted: bool = False,
        use_quality_context_rms: bool = True,
        em_rt_sidecar_root: Optional[Path] = None,
    ) -> None:
        self.scenarios = scenarios
        self.city_root = city_root
        self.context_steps = int(context_steps)
        self.future_steps = int(future_steps)
        self.point_count = int(point_count)
        self.point_pool_count = int(point_pool_count)
        self.point_pool_mode = str(point_pool_mode)
        self.point_normalization = str(point_normalization)
        self.trajectory_features_mode = str(trajectory_features_mode)
        self.pos_scale = float(pos_scale)
        self.point_scale = float(point_scale)
        self.csi_transform = csi_transform
        self.signed_log_eps = float(signed_log_eps)
        self.csi_mean = float(csi_mean)
        self.csi_std = max(float(csi_std), 1e-6)
        self.drop_zero_csi_samples = bool(drop_zero_csi_samples)
        self.seed = int(seed)
        # --- balanced/normalized pipeline state ---
        self.normalization_scope = str(normalization_scope)
        self.per_city_stats = per_city_stats or {}
        self.context_rms_normalization = bool(context_rms_normalization)
        self.context_rms_feature = bool(context_rms_feature)
        self.quality_lookup = quality_lookup or {}
        self.filter_accepted = bool(filter_accepted)
        self.use_quality_context_rms = bool(use_quality_context_rms)
        self.em_rt_sidecar_root = Path(em_rt_sidecar_root).resolve() if em_rt_sidecar_root else None
        # Per-kept-index sampler weights (aligned with self.index), used by the
        # WeightedRandomSampler for power-stratified balanced sampling.
        self.sample_weights: List[float] = []
        self.index: List[Tuple[int, int]] = []
        self._arrays: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._em_path_arrays: Dict[int, Optional[np.ndarray]] = {}
        self._point_pools: Dict[str, np.ndarray] = {}
        self._point_trees: Dict[str, Any] = {}

        # The zero-CSI filtering scan below mmaps and inspects every one of the
        # ~344k samples serially, which costs several minutes on each startup.
        # Cache the resulting index to disk (keyed by scenarios + drop flag) so
        # subsequent launches load it instantly instead of re-scanning.
        idx_cache_path = None
        # When the quality index drives selection, filter to accepted samples and
        # record their sampler weights directly (this supersedes the zero-CSI scan,
        # since SNR acceptance already excludes zero/degenerate samples).
        if self.filter_accepted and self.quality_lookup:
            kept = 0
            for file_idx, scenario in enumerate(scenarios):
                for sample_idx in range(scenario.samples):
                    entry = self.quality_lookup.get((scenario.base, sample_idx))
                    if entry is None or not entry["accepted"]:
                        continue
                    self.index.append((file_idx, sample_idx))
                    self.sample_weights.append(float(entry["weight"]))
                    kept += 1
            if not self.index:
                raise RuntimeError(
                    "Quality index accepted 0 samples for these scenarios; check --csi-quality-index."
                )
            print("quality_index=applied accepted=%d total_scanned=%d" % (
                kept, sum(int(s.samples) for s in scenarios)))
        elif self.drop_zero_csi_samples and len(scenarios) > 0:
            key_payload = json.dumps(
                {
                    "drop_zero": True,
                    "scenarios": [
                        {"h": str(Path(s.h_path).resolve()), "n": int(s.samples)}
                        for s in scenarios
                    ],
                },
                sort_keys=True,
            ).encode("utf-8")
            key = hashlib.sha1(key_payload).hexdigest()[:16]
            idx_cache_dir = Path(scenarios[0].h_path).resolve().parent.parent / "sample_index_cache"
            idx_cache_path = idx_cache_dir / ("nonzero_index_%s.npy" % key)
            if idx_cache_path.exists():
                try:
                    arr = np.load(idx_cache_path)
                    self.index = [(int(a), int(b)) for a, b in arr]
                    print("nonzero_index_cache=loaded path=%s length=%d" % (idx_cache_path, len(self.index)))
                except Exception as exc:  # noqa: BLE001
                    print("nonzero_index_cache=load_failed reason=%s (rescanning)" % exc)
                    self.index = []

        if not self.index and not (self.filter_accepted and self.quality_lookup):
            for file_idx, scenario in enumerate(scenarios):
                if self.drop_zero_csi_samples:
                    h_arr = np.load(scenario.h_path, mmap_mode="r")
                    for sample_idx in range(scenario.samples):
                        if not bool(np.any(np.asarray(h_arr[sample_idx]) == 0)):
                            self.index.append((file_idx, sample_idx))
                else:
                    self.index.extend((file_idx, sample_idx) for sample_idx in range(scenario.samples))
            if idx_cache_path is not None and self.index:
                try:
                    idx_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(idx_cache_path, np.asarray(self.index, dtype=np.int64))
                    print("nonzero_index_cache=saved path=%s length=%d" % (idx_cache_path, len(self.index)))
                except Exception as exc:  # noqa: BLE001
                    print("nonzero_index_cache=save_failed reason=%s" % exc)
        for city in sorted({s.city_key for s in scenarios}):
            self._point_pools[city] = self._load_city_point_pool(city)
            if self.point_pool_mode == "full" and cKDTree is not None:
                self._point_trees[city] = cKDTree(self._point_pools[city])

    def _load_city_point_pool(self, city: str) -> np.ndarray:
        pc_path = self.city_root / city / "point_clouds" / "point_cloud.npy"
        if not pc_path.exists():
            raise FileNotFoundError("Missing point cloud for city %s: %s" % (city, pc_path))
        pc = np.load(pc_path, mmap_mode="r")
        if self.point_pool_mode == "full":
            return np.asarray(pc, dtype=np.float32)
        if self.point_pool_mode != "random_pool":
            raise ValueError("Unknown point_pool_mode: %s" % self.point_pool_mode)
        rng = np.random.default_rng(self.seed + stable_int(city))
        pool_count = min(max(self.point_count, self.point_pool_count), int(pc.shape[0]))
        replace = int(pc.shape[0]) < pool_count
        idx = rng.choice(int(pc.shape[0]), size=pool_count, replace=replace)
        return np.asarray(pc[idx], dtype=np.float32)

    def _get_arrays(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if file_idx not in self._arrays:
            scenario = self.scenarios[file_idx]
            h = np.load(scenario.h_path, mmap_mode="r")
            pos = np.load(scenario.pos_path, mmap_mode="r")
            self._arrays[file_idx] = (h, pos)
        return self._arrays[file_idx]

    def _resolve_em_path(self, scenario: ScenarioFile) -> Optional[Path]:
        """Resolve an RT path sidecar without changing the CSI dataset layout.

        Accepted layouts are ``<root>/<split>/path/<base>_path.(npy|npz)`` and
        ``<root>/<split>/<base>_path.(npy|npz)``.  A sidecar is optional for
        legacy runs; when requested by the CLI, missing files are reported at
        the first sample with the exact scenario stem.
        """
        if self.em_rt_sidecar_root is None:
            return None
        split_root = Path(scenario.meta_path).resolve().parent.parent.name
        base = scenario.base
        candidates = []
        for suffix in ("_path.npy", "_path.npz", "_rt_path.npy", "_rt_path.npz"):
            candidates.extend([
                self.em_rt_sidecar_root / split_root / "path" / (base + suffix),
                self.em_rt_sidecar_root / split_root / "rt_path" / (base + suffix),
                self.em_rt_sidecar_root / split_root / (base + suffix),
                self.em_rt_sidecar_root / "path" / (base + suffix),
                self.em_rt_sidecar_root / (base + suffix),
            ])
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _get_em_path(self, file_idx: int) -> Optional[np.ndarray]:
        if self.em_rt_sidecar_root is None:
            return None
        if file_idx not in self._em_path_arrays:
            scenario = self.scenarios[file_idx]
            path = self._resolve_em_path(scenario)
            if path is None:
                raise FileNotFoundError(
                    "RT path sidecar missing for %s under %s; expected *_path.npy or *.npz"
                    % (scenario.base, self.em_rt_sidecar_root)
                )
            if path.suffix == ".npz":
                loaded = np.load(path, allow_pickle=False)
                key = "residual" if "residual" in loaded else ("path" if "path" in loaded else "features")
                if key not in loaded:
                    raise ValueError("RT sidecar %s must contain residual/path/features" % path)
                arr = np.asarray(loaded[key], dtype=np.float32)
            else:
                arr = np.load(path, mmap_mode="r")
            if arr.ndim != 3:
                raise ValueError("RT sidecar %s must have shape [samples,time,features], got %s" % (path, arr.shape))
            if int(arr.shape[0]) < int(scenario.samples):
                raise ValueError("RT sidecar %s has %d samples but H has %d" % (path, arr.shape[0], scenario.samples))
            self._em_path_arrays[file_idx] = arr
        return self._em_path_arrays[file_idx]

    def __len__(self) -> int:
        return len(self.index)

    def _sample_local_point_cloud(self, city: str, center: np.ndarray) -> Tuple[np.ndarray, float]:
        pool = self._point_pools[city]
        if pool.shape[0] <= self.point_count:
            selected = pool
        elif self.point_pool_mode == "full" and city in self._point_trees:
            _, idx = self._point_trees[city].query(center.astype(np.float32), k=self.point_count)
            selected = pool[np.asarray(idx, dtype=np.int64)]
        else:
            diff = pool - center[None, :].astype(np.float32)
            dist2 = np.einsum("ij,ij->i", diff, diff)
            idx = np.argpartition(dist2, self.point_count - 1)[: self.point_count]
            selected = pool[idx]
        pc = selected.astype(np.float32) - center[None, :].astype(np.float32)
        if self.point_normalization == "fixed":
            normalization_scale = float(self.point_scale)
            pc = pc / normalization_scale
        elif self.point_normalization == "unit_sphere":
            radius = float(np.max(np.linalg.norm(pc, axis=1))) if pc.size else 0.0
            normalization_scale = max(radius, 1e-6)
            pc = pc / normalization_scale
        else:
            raise ValueError("Unknown point_normalization: %s" % self.point_normalization)
        if pc.shape[0] < self.point_count:
            reps = int(math.ceil(self.point_count / max(pc.shape[0], 1)))
            pc = np.tile(pc, (reps, 1))[: self.point_count]
        return pc.astype(np.float32), normalization_scale

    def __getitem__(self, item: int) -> Dict[str, Any]:
        file_idx, sample_idx = self.index[item]
        scenario = self.scenarios[file_idx]
        h_arr, pos_arr = self._get_arrays(file_idx)
        em_path_arr = self._get_em_path(file_idx)
        total_steps = self.context_steps + self.future_steps

        h = np.asarray(h_arr[sample_idx], dtype=np.float32)
        pos = np.asarray(pos_arr[sample_idx], dtype=np.float32)
        h = np.transpose(h, (1, 0, 2, 3))
        if h.shape[0] != pos.shape[0] or h.shape[0] < total_steps:
            raise ValueError(
                "Sample timestep mismatch: model needs at least %d aligned steps, CSI has %d, trajectory has %d"
                % (total_steps, h.shape[0], pos.shape[0])
            )
        if h.shape[0] > total_steps:
            h = h[-total_steps:]
            pos = pos[-total_steps:]
        em_path = None
        if em_path_arr is not None:
            em_path = np.asarray(em_path_arr[sample_idx], dtype=np.float32)
            if em_path.shape[0] < total_steps:
                raise ValueError("RT sidecar timestep mismatch for %s[%d]" % (scenario.base, sample_idx))
            if em_path.shape[0] > total_steps:
                em_path = em_path[-total_steps:]
            if not np.isfinite(em_path).all():
                raise ValueError("RT sidecar contains non-finite values for %s[%d]" % (scenario.base, sample_idx))

        # Look up this sample's context RMS (fit-time value) from the quality index.
        entry = self.quality_lookup.get((scenario.base, sample_idx)) if self.quality_lookup else None
        context_rms = (
            float(entry["context_rms"])
            if entry is not None and self.use_quality_context_rms
            else 0.0
        )
        if context_rms <= 0:
            # Held-out datasets do not have to share the training quality sidecar.
            # Computing the same context-only statistic is leakage-free and keeps
            # absolute-power conditioning and the 16->4 normalization contract
            # usable for external 40/80 km/h tests.
            context_rms = float(np.sqrt(np.mean(np.square(h[: self.context_steps], dtype=np.float64))))

        if self.csi_transform == "signed_log":
            # 1) context-RMS normalization: divide by per-sample RMS so |H_rel| is order-1.
            if self.context_rms_normalization:
                if context_rms <= 0:
                    raise ValueError(
                        "context_rms_normalization requires a positive context_rms for %s[%d]"
                        % (scenario.base, sample_idx)
                    )
                h = h / context_rms
            # 2) signed-log.
            h = np.sign(h) * np.log1p(np.abs(h) / self.signed_log_eps)
            # 3) per-city (or global) standardization of the relative signed-log CSI.
            if self.normalization_scope == "per_city":
                cstats = self.per_city_stats.get("per_city", {}).get(scenario.city_key)
                if cstats is None:
                    raise ValueError("Missing per-city CSI stats for %s" % scenario.city_key)
                h = (h - float(cstats["mean"])) / max(float(cstats["std"]), 1e-6)
            else:
                h = (h - self.csi_mean) / self.csi_std
        elif self.csi_transform == "none":
            pass
        else:
            raise ValueError("Unknown CSI transform: %s" % self.csi_transform)

        traj = trajectory_features(pos, scenario.bs_position, self.pos_scale, self.trajectory_features_mode)
        # 4) log10(context_rms) as an extra standardized trajectory feature so the
        # network still sees absolute power after RMS normalization strips it.
        if self.context_rms_feature:
            if context_rms <= 0:
                raise ValueError(
                    "context_rms_feature requires a positive context_rms for %s[%d]"
                    % (scenario.base, sample_idx)
                )
            if self.normalization_scope == "per_city":
                cstats = self.per_city_stats.get("per_city", {}).get(scenario.city_key, {})
                lr_mean = float(cstats.get("logrms_mean", self.per_city_stats.get("global_logrms_mean", 0.0)))
                lr_std = float(cstats.get("logrms_std", self.per_city_stats.get("global_logrms_std", 1.0)))
            else:
                lr_mean = float(self.per_city_stats.get("global_logrms_mean", 0.0))
                lr_std = float(self.per_city_stats.get("global_logrms_std", 1.0))
            log_rms = (math.log10(context_rms) - lr_mean) / max(lr_std, 1e-6)
            rms_col = np.full((traj.shape[0], 1), np.float32(log_rms), dtype=np.float32)
            traj = np.concatenate([traj, rms_col], axis=1).astype(np.float32)
        pc_center = pos[self.context_steps - 1]
        pc, pc_normalization_scale = self._sample_local_point_cloud(scenario.city_key, pc_center)
        bs = np.asarray(scenario.bs_position, dtype=np.float32)
        point_origin = (pc_center.astype(np.float32) - bs) / float(self.pos_scale)
        point_scale = np.asarray([pc_normalization_scale / float(self.pos_scale)], dtype=np.float32)
        out = {
            "h": torch.from_numpy(np.ascontiguousarray(h)),
            "pos": torch.from_numpy(np.ascontiguousarray(pos.copy())),
            "traj": torch.from_numpy(np.ascontiguousarray(traj.copy())),
            "point_cloud": torch.from_numpy(np.ascontiguousarray(pc.copy())),
            "point_origin": torch.from_numpy(np.ascontiguousarray(point_origin.copy())),
            "point_scale": torch.from_numpy(np.ascontiguousarray(point_scale)),
            "context_rms": torch.tensor(context_rms, dtype=torch.float32),
            "city_key": scenario.city_key,
        }
        if em_path is not None:
            out["em_path"] = torch.from_numpy(np.ascontiguousarray(em_path).copy())
        return out


class WWMPointCloudDataset(Dataset):
    """Point-cloud-only view of WWM data for Point-BERT dVAE pretraining."""

    def __init__(
        self,
        scenarios: List[ScenarioFile],
        city_root: Path,
        context_steps: int,
        point_count: int,
        point_pool_count: int,
        point_pool_mode: str,
        point_normalization: str,
        point_scale: float,
        drop_zero_csi_samples: bool,
        seed: int,
    ) -> None:
        self.scenarios = scenarios
        self.city_root = city_root
        self.context_steps = int(context_steps)
        self.point_count = int(point_count)
        self.point_pool_count = int(point_pool_count)
        self.point_pool_mode = str(point_pool_mode)
        self.point_normalization = str(point_normalization)
        self.point_scale = float(point_scale)
        self.drop_zero_csi_samples = bool(drop_zero_csi_samples)
        self.seed = int(seed)
        self.index: List[Tuple[int, int]] = []
        self._pos_arrays: Dict[int, np.ndarray] = {}
        self._point_pools: Dict[str, np.ndarray] = {}
        self._point_trees: Dict[str, Any] = {}

        for file_idx, scenario in enumerate(scenarios):
            if self.drop_zero_csi_samples:
                h_arr = np.load(scenario.h_path, mmap_mode="r")
                for sample_idx in range(scenario.samples):
                    if not bool(np.any(np.asarray(h_arr[sample_idx]) == 0)):
                        self.index.append((file_idx, sample_idx))
            else:
                self.index.extend((file_idx, sample_idx) for sample_idx in range(scenario.samples))
        for city in sorted({s.city_key for s in scenarios}):
            self._point_pools[city] = self._load_city_point_pool(city)
            if self.point_pool_mode == "full" and cKDTree is not None:
                self._point_trees[city] = cKDTree(self._point_pools[city])

    def _load_city_point_pool(self, city: str) -> np.ndarray:
        pc_path = self.city_root / city / "point_clouds" / "point_cloud.npy"
        if not pc_path.exists():
            raise FileNotFoundError("Missing point cloud for city %s: %s" % (city, pc_path))
        pc = np.load(pc_path, mmap_mode="r")
        if self.point_pool_mode == "full":
            return np.asarray(pc, dtype=np.float32)
        if self.point_pool_mode != "random_pool":
            raise ValueError("Unknown point_pool_mode: %s" % self.point_pool_mode)
        rng = np.random.default_rng(self.seed + stable_int(city))
        pool_count = min(max(self.point_count, self.point_pool_count), int(pc.shape[0]))
        replace = int(pc.shape[0]) < pool_count
        idx = rng.choice(int(pc.shape[0]), size=pool_count, replace=replace)
        return np.asarray(pc[idx], dtype=np.float32)

    def _get_pos_array(self, file_idx: int) -> np.ndarray:
        if file_idx not in self._pos_arrays:
            self._pos_arrays[file_idx] = np.load(self.scenarios[file_idx].pos_path, mmap_mode="r")
        return self._pos_arrays[file_idx]

    def __len__(self) -> int:
        return len(self.index)

    def _sample_local_point_cloud(self, city: str, center: np.ndarray) -> np.ndarray:
        pool = self._point_pools[city]
        if pool.shape[0] <= self.point_count:
            selected = pool
        elif self.point_pool_mode == "full" and city in self._point_trees:
            _, idx = self._point_trees[city].query(center.astype(np.float32), k=self.point_count)
            selected = pool[np.asarray(idx, dtype=np.int64)]
        else:
            diff = pool - center[None, :].astype(np.float32)
            dist2 = np.einsum("ij,ij->i", diff, diff)
            idx = np.argpartition(dist2, self.point_count - 1)[: self.point_count]
            selected = pool[idx]
        pc = selected.astype(np.float32) - center[None, :].astype(np.float32)
        if self.point_normalization == "fixed":
            pc = pc / float(self.point_scale)
        elif self.point_normalization == "unit_sphere":
            radius = float(np.max(np.linalg.norm(pc, axis=1))) if pc.size else 0.0
            pc = pc / max(radius, 1e-6)
        else:
            raise ValueError("Unknown point_normalization: %s" % self.point_normalization)
        if pc.shape[0] < self.point_count:
            reps = int(math.ceil(self.point_count / max(pc.shape[0], 1)))
            pc = np.tile(pc, (reps, 1))[: self.point_count]
        return pc.astype(np.float32)

    def __getitem__(self, item: int) -> torch.Tensor:
        file_idx, sample_idx = self.index[item]
        scenario = self.scenarios[file_idx]
        pos = np.asarray(self._get_pos_array(file_idx)[sample_idx], dtype=np.float32)
        if pos.shape[0] < self.context_steps:
            raise ValueError("Need %d positions, got %d" % (self.context_steps, pos.shape[0]))
        pc_center = pos[self.context_steps - 1]
        pc = self._sample_local_point_cloud(scenario.city_key, pc_center)
        return torch.from_numpy(np.ascontiguousarray(pc.copy()))


def point_group_cache_key(args: argparse.Namespace, dataset: Dataset) -> str:
    payload = {
        "version": 2,
        "dataset_kind": "wwm_local_point_group",
        "length": len(dataset),
        "split": getattr(args, "split", None),
        "context_steps": int(args.context_steps),
        "point_count": int(args.point_count),
        "point_pool_count": int(args.point_pool_count),
        "point_pool_mode": str(args.point_pool_mode),
        "point_normalization": str(args.point_normalization),
        "point_tokens": int(args.point_tokens),
        "point_group_size": int(args.point_group_size),
        "point_center_sampling": str(args.point_center_sampling),
        "point_scale": float(args.point_scale),
        "pos_scale": float(args.pos_scale),
        "drop_zero_csi_samples": bool(args.drop_zero_csi_samples),
        "seed": int(args.seed),
        "scenarios": [
            {
                "base": s.base,
                "city_key": s.city_key,
                "samples": int(s.samples),
                "h_path": str(Path(s.h_path).resolve()),
                "pos_path": str(Path(s.pos_path).resolve()),
            }
            for s in getattr(dataset, "scenarios", [])
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def point_group_cache_dir(args: argparse.Namespace, dataset: Dataset) -> Path:
    root_text = str(args.point_group_cache_dir or "").strip()
    root = Path(root_text).resolve() if root_text else Path(args.dataset_root).resolve().parent / "point_group_cache"
    return root / point_group_cache_key(args, dataset)


def expected_point_group_cache_meta(args: argparse.Namespace, dataset: Dataset) -> Dict[str, Any]:
    return {
        "version": 2,
        "length": int(len(dataset)),
        "point_tokens": int(args.point_tokens),
        "point_group_size": int(args.point_group_size),
        "point_count": int(args.point_count),
        "point_center_sampling": str(args.point_center_sampling),
        "dtype": "float16" if bool(args.point_group_cache_float16) else "float32",
        "cache_key": point_group_cache_key(args, dataset),
    }


def point_group_cache_ready(cache_dir: Path, expected: Dict[str, Any]) -> bool:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return (cache_dir / "neighborhood.dat").exists() and (cache_dir / "center.dat").exists()


def compute_point_groups_numpy(points: np.ndarray, num_group: int, group_size: int, center_sampling: str) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float32)
    n = int(xyz.shape[0])
    if n < num_group:
        raise ValueError("point_count must be >= point_tokens")
    if center_sampling == "linspace":
        center_idx = np.rint(np.linspace(0, n - 1, num_group)).astype(np.int64)
    elif center_sampling == "fps":
        center_idx = np.zeros(num_group, dtype=np.int64)
        distance = np.full(n, np.inf, dtype=np.float32)
        farthest = 0
        for idx in range(num_group):
            center_idx[idx] = farthest
            centroid = xyz[farthest]
            dist = np.sum((xyz - centroid[None, :]) ** 2, axis=1)
            distance = np.minimum(distance, dist)
            farthest = int(np.argmax(distance))
    else:
        raise ValueError("Unknown point center sampling: %s" % center_sampling)
    center = xyz[center_idx].astype(np.float32)
    k = min(int(group_size), n)
    diff = xyz[None, :, :] - center[:, None, :]
    dist2 = np.einsum("gnc,gnc->gn", diff, diff)
    group_idx = np.argpartition(dist2, k - 1, axis=1)[:, :k]
    neighborhood = xyz[group_idx].astype(np.float32) - center[:, None, :]
    return neighborhood.astype(np.float32), center.astype(np.float32)


def build_point_group_cache(dataset: Dataset, args: argparse.Namespace, cache_dir: Path) -> None:
    expected = expected_point_group_cache_meta(args, dataset)
    if point_group_cache_ready(cache_dir, expected):
        print("point_group_cache=ready path=%s length=%d" % (cache_dir, len(dataset)))
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if bool(args.point_group_cache_float16) else np.float32
    tmp_dir = Path(tempfile.mkdtemp(prefix="building_", dir=str(cache_dir)))
    neighborhood_shape = (len(dataset), int(args.point_tokens), int(args.point_group_size), 3)
    center_shape = (len(dataset), int(args.point_tokens), 3)
    neighborhood = np.memmap(tmp_dir / "neighborhood.dat", dtype=dtype, mode="w+", shape=neighborhood_shape)
    center = np.memmap(tmp_dir / "center.dat", dtype=dtype, mode="w+", shape=center_shape)
    start = time.time()
    log_every = max(1, int(args.point_group_cache_log_every))
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, dict):
            pc = sample["point_cloud"].numpy()
        elif torch.is_tensor(sample):
            pc = sample.numpy()
        else:
            pc = np.asarray(sample, dtype=np.float32)
        neigh_np, center_np = compute_point_groups_numpy(
            pc,
            num_group=int(args.point_tokens),
            group_size=int(args.point_group_size),
            center_sampling=str(args.point_center_sampling),
        )
        neighborhood[idx] = neigh_np.astype(dtype)
        center[idx] = center_np.astype(dtype)
        if idx == 0 or (idx + 1) % log_every == 0 or idx + 1 == len(dataset):
            elapsed = max(time.time() - start, 1e-6)
            rate = float(idx + 1) / elapsed
            eta_s = float(len(dataset) - idx - 1) / max(rate, 1e-12)
            print("point_group_cache_build item=%d/%d rate=%.2f eta_h=%.2f path=%s" % (idx + 1, len(dataset), rate, eta_s / 3600.0, cache_dir))
    neighborhood.flush()
    center.flush()
    meta = dict(expected)
    meta.update(
        {
            "neighborhood_shape": list(neighborhood_shape),
            "center_shape": list(center_shape),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    del neighborhood
    del center
    gc.collect()
    for name in ("neighborhood.dat", "center.dat", "meta.json"):
        os.replace(tmp_dir / name, cache_dir / name)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print("point_group_cache=built path=%s length=%d dtype=%s" % (cache_dir, len(dataset), meta["dtype"]))


class PointGroupCachedDataset(Dataset):
    def __init__(self, base: Dataset, cache_dir: Path, args: argparse.Namespace, include_point_cloud: bool) -> None:
        self.base = base
        self.cache_dir = cache_dir
        self.include_point_cloud = bool(include_point_cloud)
        expected = expected_point_group_cache_meta(args, base)
        if not point_group_cache_ready(cache_dir, expected):
            raise FileNotFoundError("Point group cache is missing or incompatible: %s" % cache_dir)
        dtype = np.float16 if expected["dtype"] == "float16" else np.float32
        self.neighborhood = np.memmap(
            cache_dir / "neighborhood.dat",
            dtype=dtype,
            mode="r",
            shape=(expected["length"], expected["point_tokens"], expected["point_group_size"], 3),
        )
        self.center = np.memmap(
            cache_dir / "center.dat",
            dtype=dtype,
            mode="r",
            shape=(expected["length"], expected["point_tokens"], 3),
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> Any:
        sample = self.base[item]
        neighborhood = torch.from_numpy(np.asarray(self.neighborhood[item], dtype=np.float32).copy())
        center = torch.from_numpy(np.asarray(self.center[item], dtype=np.float32).copy())
        grouped = {"neighborhood": neighborhood, "center": center}
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["point_group"] = grouped
            if not self.include_point_cloud:
                sample.pop("point_cloud", None)
            return sample
        if self.include_point_cloud:
            return {"point_cloud": sample, "point_group": grouped}
        return grouped


def make_multi_split_loader(
    scenario_groups: List[Tuple[str, List[ScenarioFile]]],
    city_root: Path,
    args: argparse.Namespace,
    shuffle: bool,
    drop_last: bool,
) -> Tuple[Dataset, DataLoader]:
    """Build one shuffled loader while retaining a reusable cache per source split."""
    datasets: List[Dataset] = []
    for split_name, scenarios in scenario_groups:
        group_args = copy.copy(args)
        group_args.split = split_name
        dataset: Dataset = WWMDataset(
            scenarios=scenarios,
            city_root=city_root,
            context_steps=args.context_steps,
            future_steps=args.future_steps,
            point_count=args.point_count,
            point_pool_count=args.point_pool_count,
            point_pool_mode=args.point_pool_mode,
            point_normalization=args.point_normalization,
            trajectory_features_mode=args.trajectory_features,
            pos_scale=args.pos_scale,
            point_scale=args.point_scale,
            csi_transform=args.csi_transform,
            signed_log_eps=args.signed_log_eps,
            csi_mean=float(args.csi_mean),
            csi_std=float(args.csi_std),
            drop_zero_csi_samples=args.drop_zero_csi_samples,
            seed=args.seed,
            normalization_scope=str(getattr(args, "csi_normalization_scope", "global")),
            per_city_stats=getattr(args, "_per_city_stats", None),
            context_rms_normalization=bool(getattr(args, "csi_context_rms_normalization", False)),
            context_rms_feature=bool(getattr(args, "context_rms_feature", False)),
            quality_lookup=getattr(args, "_quality_lookup", None),
            filter_accepted=bool(getattr(args, "_quality_lookup", None)),
            em_rt_sidecar_root=getattr(args, "_em_rt_sidecar_root", None),
        )
        if bool(args.point_group_cache):
            cache_dir = point_group_cache_dir(group_args, dataset)
            expected = expected_point_group_cache_meta(group_args, dataset)
            if not point_group_cache_ready(cache_dir, expected):
                if bool(args.point_group_cache_build):
                    build_point_group_cache(dataset, group_args, cache_dir)
                else:
                    raise FileNotFoundError(
                        "Point group cache missing; rerun with --point-group-cache-build: %s" % cache_dir
                    )
            print("point_group_cache=using split=%s path=%s include_point_cloud=False" % (split_name, cache_dir))
            dataset = PointGroupCachedDataset(dataset, cache_dir, group_args, include_point_cloud=False)
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError("No dataset groups were provided")
    combined: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    # Power-stratified balanced sampling: draw accepted samples with probability
    # proportional to their precomputed sampler_weight. Mutually exclusive with
    # shuffle (the sampler already randomizes order).
    sampler = None
    if bool(getattr(args, "balanced_sampling", False)):
        weights: List[float] = []
        for ds in datasets:
            base = ds.base if isinstance(ds, PointGroupCachedDataset) else ds
            sw = getattr(base, "sample_weights", None)
            if not sw or len(sw) != len(base):
                raise RuntimeError(
                    "--balanced-sampling needs per-sample weights; ensure --csi-quality-index is set."
                )
            weights.extend(float(w) for w in sw)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        print("balanced_sampling=on samples=%d weight_min=%.4f weight_max=%.4f" % (
            len(weights), min(weights), max(weights)))
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and len(combined) >= args.batch_size,
        persistent_workers=bool(args.num_workers > 0 and args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor) if args.num_workers > 0 else None,
    )
    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty; reduce batch size or increase samples.")
    return combined, loader


class PointBERTShapeNet55Dataset(Dataset):
    """ShapeNet55 dataset path compatible with the official Point-BERT dVAE."""

    def __init__(
        self,
        root: Path,
        subset: str,
        npoints: int,
        whole: bool = False,
    ) -> None:
        self.root = Path(root)
        self.data_root = self.root / "ShapeNet55-34" / "ShapeNet-55"
        self.pc_root = self.root / "ShapeNet55-34" / "shapenet_pc"
        self.subset = str(subset)
        self.sample_points_num = int(npoints)
        list_files = [self.data_root / ("%s.txt" % self.subset)]
        if whole and self.subset != "test":
            list_files.insert(0, self.data_root / "test.txt")
        lines: List[str] = []
        for list_file in list_files:
            if not list_file.exists():
                raise FileNotFoundError("Missing ShapeNet55 split file: %s" % list_file)
            with list_file.open("r", encoding="utf-8") as f:
                lines.extend(line.strip() for line in f if line.strip())
        if not lines:
            raise RuntimeError("No ShapeNet55 entries found under %s" % self.data_root)
        self.file_list = lines
        self.permutation: Optional[np.ndarray] = None

    @staticmethod
    def pc_norm(pc: np.ndarray) -> np.ndarray:
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        radius = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        return pc / max(float(radius), 1e-6)

    def random_sample(self, pc: np.ndarray) -> np.ndarray:
        if pc.shape[0] < self.sample_points_num:
            replace_idx = np.random.choice(pc.shape[0], self.sample_points_num - pc.shape[0], replace=True)
            pc = np.concatenate([pc, pc[replace_idx]], axis=0)
        if self.permutation is None or self.permutation.shape[0] != pc.shape[0]:
            self.permutation = np.arange(pc.shape[0])
        np.random.shuffle(self.permutation)
        return pc[self.permutation[: self.sample_points_num]]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pc_path = self.pc_root / self.file_list[idx]
        if not pc_path.exists():
            raise FileNotFoundError("Missing ShapeNet55 point cloud: %s" % pc_path)
        data = np.load(pc_path).astype(np.float32)
        data = self.random_sample(data)
        data = self.pc_norm(data)
        return torch.from_numpy(np.ascontiguousarray(data)).float()


def estimate_signed_log_stats(scenarios: List[ScenarioFile], args: argparse.Namespace) -> Tuple[float, float, int]:
    if args.csi_transform != "signed_log":
        return 0.0, 1.0, 0
    if args.csi_mean is not None and args.csi_std is not None:
        return float(args.csi_mean), float(args.csi_std), 0

    max_per_file = optional_limit(args.stats_samples_per_file)
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    for scenario in scenarios:
        h_arr = np.load(scenario.h_path, mmap_mode="r")
        sample_count = scenario.samples if max_per_file is None else min(scenario.samples, max_per_file)
        if sample_count <= 0:
            continue
        h = np.asarray(h_arr[:sample_count], dtype=np.float32)
        if args.drop_zero_csi_samples:
            keep = ~np.any(h == 0, axis=tuple(range(1, h.ndim)))
            h = h[keep]
            if h.size == 0:
                continue
        h = np.sign(h) * np.log1p(np.abs(h) / float(args.signed_log_eps))
        total_count += int(h.size)
        total_sum += float(h.sum(dtype=np.float64))
        total_sumsq += float(np.square(h, dtype=np.float64).sum(dtype=np.float64))
    if total_count == 0:
        return 0.0, 1.0, 0
    mean = total_sum / total_count
    var = max(total_sumsq / total_count - mean * mean, 1e-12)
    return float(mean), float(math.sqrt(var)), int(total_count)
