#!/usr/bin/env python
"""Paper-grade downstream-5 protocol: matched retraining + independent-city test.

The protocol deliberately separates three sources of information:

1. CSI + trajectory, encoded by a frozen WWM;
2. a light, architecture-matched downstream completion head;
3. an optional position-indexed sparse map built from a fixed support partition.

The downstream head is selected only on scenario-file-held-out samples from the
seven-city training package.  Frankfurt is never used for optimization or model
selection.  At final evaluation, a deterministic 5% Frankfurt support partition
forms the optional map memory and every remaining accepted sample is a query.

Matched arms use the same head architecture, loss, optimizer, data, and budget:

``v19``             point-grounded WWM whose pretraining also decodes the height
                    distribution of the masked cloud
``v18``             point-grounded pretrained WWM
``v17``             pretrained WWM without point-grounding masks
``random_frozen``   random frozen wireless representation; same frozen Point-dVAE
``map_only``        learned null wireless tokens during both training and inference
``shuffled_train``  v18 wireless tokens shuffled across samples during training

The head is trained with map dropout.  Consequently every checkpoint supports both
``sparse map + wireless`` and the strict ``CSI + trajectory only`` path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import wwm_csi_traj_point_reconstruction as R  # noqa: E402
import wwm_downstream5_center_set_refine as C  # noqa: E402
import wwm_downstream5_env_pointcloud as D  # noqa: E402
import wwm_downstream5_position_map_memory as M  # noqa: E402
from wwm.data import discover_scenarios, load_per_city_csi_stats, load_quality_index  # noqa: E402
from wwm.model import PaperWWM  # noqa: E402


ARMS = ("v19", "v18", "v17", "random_frozen", "map_only", "shuffled_train")


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def string_list_sha256(values: Sequence[str]) -> str:
    """Stable digest for protocol partitions without exposing platform ordering."""
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_city_seed(city: str, seed: int) -> int:
    return int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(city))


def atomic_json(path: Path, payload: Dict[str, Any], attempts: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    raise PermissionError("Could not publish %s" % path)


def atomic_torch_save(path: Path, payload: Dict[str, Any], attempts: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    raise PermissionError("Could not publish %s" % path)


class CrossFusionBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_mult: int) -> None:
        super().__init__()
        heads = next(
            value for value in (heads, 8, 6, 4, 2, 1) if hidden_dim % value == 0
        )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.wireless_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * int(ffn_mult)),
            nn.GELU(),
            nn.Linear(hidden_dim * int(ffn_mult), hidden_dim),
        )

    def forward(self, query: torch.Tensor, wireless: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(query)
        query = query + self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        query = query + self.cross_attention(
            self.query_norm(query),
            self.wireless_norm(wireless),
            self.wireless_norm(wireless),
            need_weights=False,
        )[0]
        return query + self.ffn(self.ffn_norm(query))


class MatchedCompletionHead(nn.Module):
    """One architecture for all representation controls and both map protocols."""

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        point_tokens: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        ffn_mult: int,
        max_center_delta_m: float,
        max_absolute_center_m: float,
        initial_scale_m: float,
    ) -> None:
        super().__init__()
        self.point_tokens = int(point_tokens)
        self.max_center_delta_m = float(max_center_delta_m)
        self.max_absolute_center_m = float(max_absolute_center_m)
        self.map_feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim)
        )
        self.map_center_projection = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.wireless_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim)
        )
        self.null_query = nn.Parameter(torch.zeros(1, point_tokens, hidden_dim))
        self.null_wireless = nn.Parameter(torch.zeros(1, point_tokens, hidden_dim))
        nn.init.trunc_normal_(self.null_query, std=0.02)
        nn.init.trunc_normal_(self.null_wireless, std=0.02)
        self.blocks = nn.ModuleList(
            [CrossFusionBlock(hidden_dim, heads, ffn_mult) for _ in range(int(depth))]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.feature_residual = nn.Linear(hidden_dim, feature_dim)
        self.feature_absolute = nn.Linear(hidden_dim, feature_dim)
        self.center_residual = nn.Linear(hidden_dim, 3)
        self.center_absolute = nn.Linear(hidden_dim, 3)
        self.scale_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.feature_residual.weight)
        nn.init.zeros_(self.feature_residual.bias)
        nn.init.zeros_(self.center_residual.weight)
        nn.init.zeros_(self.center_residual.bias)
        nn.init.zeros_(self.center_absolute.weight)
        nn.init.zeros_(self.center_absolute.bias)
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.constant_(self.scale_head[-1].bias, math.log(float(initial_scale_m)))

    def forward(
        self,
        wireless_latents: torch.Tensor,
        map_features: torch.Tensor,
        map_centers_m: torch.Tensor,
        map_scale_m: torch.Tensor,
        nearest_distance_m: torch.Tensor,
        map_present: torch.Tensor,
        wireless_present: bool,
    ) -> Dict[str, torch.Tensor]:
        batch = int(wireless_latents.shape[0])
        present = map_present.reshape(batch, 1, 1).float()
        scale = map_scale_m.reshape(batch, 1, 1).float().clamp_min(1e-6)
        normalized_centers = map_centers_m.float() / scale
        distance = torch.log1p(nearest_distance_m.float()).reshape(batch, 1, 1)
        distance = distance.expand(-1, self.point_tokens, -1) / math.log(101.0)
        presence_column = present.expand(-1, self.point_tokens, -1)
        map_query = self.map_feature_projection(map_features.float())
        map_query = map_query + self.map_center_projection(
            torch.cat([normalized_centers, distance, presence_column], dim=-1)
        )
        null_query = self.null_query.expand(batch, -1, -1)
        query = present * map_query + (1.0 - present) * null_query
        if wireless_present:
            wireless = self.wireless_projection(wireless_latents.float())
        else:
            wireless = self.null_wireless.expand(batch, -1, -1)
        for block in self.blocks:
            query = block(query, wireless)
        hidden = self.output_norm(query)
        predicted_scale_m = self.scale_head(hidden.mean(dim=1)).squeeze(-1).exp().clamp(1.0, 200.0)
        residual_features = map_features.float() + self.feature_residual(hidden)
        absolute_features = self.feature_absolute(hidden)
        features = present * residual_features + (1.0 - present) * absolute_features
        residual_centers = map_centers_m.float() + torch.tanh(
            self.center_residual(hidden)
        ) * self.max_center_delta_m
        absolute_centers = torch.tanh(self.center_absolute(hidden)) * self.max_absolute_center_m
        centers_m = present * residual_centers + (1.0 - present) * absolute_centers
        effective_scale_m = (
            map_present.float() * map_scale_m.float()
            + (1.0 - map_present.float()) * predicted_scale_m
        )
        return {
            "features": features,
            "centers_m": centers_m,
            "predicted_scale_m": predicted_scale_m,
            "scale_m": effective_scale_m,
            "map_present": map_present.float(),
            "center_residual_m": (centers_m - map_centers_m.float()).abs().mean(dim=(1, 2)),
        }


def reset_module(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def load_backbone(
    arm: str,
    v18_checkpoint: Path,
    v17_checkpoint: Path,
    dataset_root: Path,
    device: torch.device,
    v19_checkpoint: Path | None = None,
) -> Tuple[PaperWWM, argparse.Namespace, Path]:
    # The representation controls (random_frozen / map_only / shuffled_train) keep
    # reading the v18 file so their architecture and frozen Point-dVAE stay identical
    # to the published arms; only `v19` and `v17` select a different backbone.
    if arm == "v17":
        source = v17_checkpoint
    elif arm == "v19":
        if v19_checkpoint is None:
            raise ValueError("arm=v19 requires --v19-checkpoint")
        source = v19_checkpoint
    else:
        source = v18_checkpoint
    payload = torch.load(source, map_location="cpu", weights_only=False)
    args = R.checkpoint_args(payload, dataset_root)
    model = PaperWWM(args)
    if arm == "random_frozen":
        # Preserve only the frozen Point-dVAE interface.  Every module that maps
        # CSI/trajectory to point latents remains at its random initialization.
        dvae_state = {
            name: value
            for name, value in payload["model"].items()
            if name.startswith("online_stem.point.dvae.")
        }
        incompatible = model.load_state_dict(dvae_state, strict=False)
        if not dvae_state:
            raise RuntimeError("Random control could not locate Point-dVAE weights")
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise RuntimeError("Unexpected random-control keys: %s" % unexpected)
    else:
        incompatible = model.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Backbone mismatch for arm=%s" % arm)
    del payload
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, args, source


def dataset_for_root(
    args: argparse.Namespace,
    root: Path,
    scenarios: Sequence[Any],
    quality_lookup: Dict[Tuple[str, int], Dict[str, float]] | None,
    train_stats: Dict[str, Any] | None,
) -> Dataset:
    local_args = argparse.Namespace(**vars(args))
    local_args.dataset_root = str(root)
    local_args.city_root = str(root / "cities")
    return R.make_dataset(local_args, scenarios, quality_lookup, train_stats)


def choose_support_query(
    dataset: Dataset,
    fraction: float,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    by_city: Dict[str, List[int]] = defaultdict(list)
    for flat_index, (file_index, _sample_index) in enumerate(dataset.index):
        city = str(dataset.scenarios[int(file_index)].city_key)
        by_city[city].append(int(flat_index))
    support: List[int] = []
    query: List[int] = []
    per_city: Dict[str, Any] = {}
    for city in sorted(by_city):
        rows = np.asarray(by_city[city], dtype=np.int64)
        rng = np.random.default_rng(stable_city_seed(city, seed))
        order = rng.permutation(len(rows))
        keep = max(1, min(len(rows) - 1, int(round(len(rows) * float(fraction)))))
        support_city = np.sort(rows[order[:keep]]).astype(np.int64).tolist()
        query_city = np.sort(rows[order[keep:]]).astype(np.int64).tolist()
        support.extend(int(value) for value in support_city)
        query.extend(int(value) for value in query_city)
        per_city[city] = {
            "available": int(len(rows)),
            "support": int(len(support_city)),
            "query": int(len(query_city)),
        }
    if set(support) & set(query):
        raise AssertionError("Support/query overlap")
    return sorted(support), sorted(query), per_city


def subset_fixed(indices: Sequence[int], count: int, seed: int) -> List[int]:
    if count <= 0 or count >= len(indices):
        return list(indices)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(indices), size=int(count), replace=False)
    return sorted(int(value) for value in selected)


def shuffled_fixed(indices: Sequence[int], seed: int) -> List[int]:
    """Deterministically randomize evaluation order for a strong batch derangement."""
    values = np.asarray(list(indices), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return values[rng.permutation(len(values))].astype(np.int64).tolist()


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": False,
        "generator": generator,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 3})
    return DataLoader(Subset(dataset, list(indices)), **kwargs)


def query_map(
    memory: M.TrainPositionMemory,
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cities = [str(value) for value in batch["city_key"]]
    positions = batch["pos"][:, memory.context_steps - 1, :3].numpy().astype(np.float32)
    points, scales, distances = memory.query(cities, positions, "train_position_memory")
    return (
        torch.from_numpy(points).to(device, non_blocking=True),
        torch.from_numpy(scales).to(device, non_blocking=True),
        torch.from_numpy(distances).to(device, non_blocking=True),
    )


def prepare_wireless(
    model: PaperWWM,
    h: torch.Tensor,
    traj: torch.Tensor,
    arm: str,
    training: bool,
) -> Tuple[torch.Tensor, bool]:
    if arm == "map_only":
        shape = (h.shape[0], model.online_stem.point.point_tokens, model.latent_dim)
        return torch.zeros(shape, dtype=h.dtype, device=h.device), False
    with torch.no_grad():
        latent = model.predict_point_latents(h, traj)
    if arm == "shuffled_train" and training and latent.shape[0] > 1:
        offset = int(torch.randint(1, latent.shape[0], (1,), device=latent.device).item())
        latent = latent.roll(offset, dims=0)
    return latent, True


def decode_cloud(
    decoder: nn.Module,
    prediction: Dict[str, torch.Tensor],
    point_count: int,
) -> torch.Tensor:
    scale = prediction["scale_m"].reshape(-1, 1, 1).float().clamp_min(1e-6)
    normalized_centers = prediction["centers_m"].float() / scale
    _, fine = decoder(prediction["features"].float())
    normalized = R.sampled_whole_cloud(fine, normalized_centers, int(point_count))
    return normalized.float() * scale


def assign_targets(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    true_scale_m: torch.Tensor,
    temperature: float,
    iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_centers_m = target["centers"].float() * true_scale_m[:, None, None]
    center_cost = torch.cdist(
        prediction["centers_m"].float() / true_scale_m[:, None, None].clamp_min(1.0),
        target_centers_m / true_scale_m[:, None, None].clamp_min(1.0),
    )
    assignment = C.sinkhorn_assignment(center_cost, temperature, iterations)
    return torch.bmm(assignment, target_centers_m), torch.bmm(
        assignment, target["features"].float()
    )


def height_edges_tensor(text: str, device: torch.device) -> torch.Tensor:
    values = [float(x) for x in str(text).split(",") if str(x).strip()]
    return torch.tensor(values, dtype=torch.float32, device=device)


def height_histogram(
    cloud_m: torch.Tensor, edges: torch.Tensor, sigma_m: float
) -> torch.Tensor:
    """Differentiable per-sample height-layer occupancy, [B, len(edges)+1].

    Hard bucketize has zero gradient, so layer membership is a difference of
    sigmoids: sigma_m controls how far a point's mass leaks across an edge. The
    pretraining target uses hard bins, so sigma trades gradient quality against a
    bias that grows with the density contrast at an edge; the self-test bounds that
    bias on a worst-case step distribution sitting exactly on the z=0 edge.
    """
    z = cloud_m[..., 2].float()
    above = torch.sigmoid((z.unsqueeze(-1) - edges.reshape(1, 1, -1)) / max(sigma_m, 1e-3))
    weights = torch.cat(
        [1.0 - above[..., :1], above[..., :-1] - above[..., 1:], above[..., -1:]], dim=-1
    )
    hist = weights.mean(dim=1)
    return hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def height_loss(
    predicted_cloud_m: torch.Tensor,
    target_cloud_m: torch.Tensor,
    edges: torch.Tensor,
    sigma_m: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KL on the vertical mass distribution plus [mean z, std z] agreement.

    Chamfer cannot see this: 73.8% of the ground-truth points lie below z=0, so a
    cloud flattened onto that plane is a Chamfer optimum. Every previous head
    converged to exactly that sheet (side views showed pred z~0 against truth
    spanning 18 m), which is why the vertical structure needs its own term.
    """
    predicted_hist = height_histogram(predicted_cloud_m, edges, sigma_m)
    with torch.no_grad():
        target_hist = height_histogram(target_cloud_m, edges, sigma_m)
    kl = F.kl_div(predicted_hist.clamp_min(1e-8).log(), target_hist, reduction="none")
    kl = kl.sum(dim=-1).mean()
    predicted_z = predicted_cloud_m[..., 2].float()
    target_z = target_cloud_m[..., 2].float().detach()
    moments = F.smooth_l1_loss(
        torch.stack([predicted_z.mean(dim=1), predicted_z.std(dim=1, unbiased=False)], dim=-1),
        torch.stack([target_z.mean(dim=1), target_z.std(dim=1, unbiased=False)], dim=-1),
        beta=1.0,
    )
    return kl, moments


def training_loss(
    prediction: Dict[str, torch.Tensor],
    predicted_cloud_m: torch.Tensor,
    target: Dict[str, torch.Tensor],
    target_cloud_m: torch.Tensor,
    true_scale_m: torch.Tensor,
    cli: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    matched_centers, matched_features = assign_targets(
        prediction, target, true_scale_m, cli.sinkhorn_temperature, cli.sinkhorn_iterations
    )
    center = F.smooth_l1_loss(prediction["centers_m"].float(), matched_centers, beta=0.5)
    feature = F.smooth_l1_loss(prediction["features"].float(), matched_features, beta=0.05)
    stride = max(int(cli.loss_point_stride), 1)
    whole_per = R.chamfer_l1_per_sample(
        predicted_cloud_m[:, ::stride], target_cloud_m[:, ::stride]
    )
    center_set = R.chamfer_l1_per_sample(
        prediction["centers_m"], target["centers"].float() * true_scale_m[:, None, None]
    ).mean()
    scale = F.smooth_l1_loss(
        prediction["predicted_scale_m"].float(), true_scale_m.float(), beta=1.0
    )
    residual = prediction["center_residual_m"].mean()
    loss = (
        float(cli.whole_weight) * whole_per.mean()
        + float(cli.center_weight) * center
        + float(cli.center_set_weight) * center_set
        + float(cli.feature_weight) * feature
        + float(cli.scale_weight) * scale
        + float(cli.residual_weight) * residual
    )
    components = {
        "whole_m": whole_per.mean(),
        "center": center,
        "center_set_m": center_set,
        "feature": feature,
        "scale_m": scale,
        "residual_m": residual,
        "map_rate": prediction["map_present"].mean(),
    }
    if float(cli.height_weight) > 0:
        edges = height_edges_tensor(cli.height_edges, predicted_cloud_m.device)
        kl, moments = height_loss(
            predicted_cloud_m[:, ::stride],
            target_cloud_m[:, ::stride],
            edges,
            float(cli.height_sigma_m),
        )
        loss = loss + float(cli.height_weight) * (kl + float(cli.height_moment_weight) * moments)
        components["height_kl"] = kl
        components["height_moment_m"] = moments
    return loss, components


def per_sample_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, np.ndarray]:
    distance = torch.cdist(predicted.float(), target.float(), p=2)
    forward = distance.min(dim=2).values
    reverse = distance.min(dim=1).values
    out: Dict[str, torch.Tensor] = {
        "chamfer_m": 0.5 * (forward.mean(dim=1) + reverse.mean(dim=1)),
        "accuracy_m": forward.mean(dim=1),
        "completeness_m": reverse.mean(dim=1),
    }
    for threshold in (1.0, 2.0, 5.0):
        precision = (forward <= threshold).float().mean(dim=1)
        recall = (reverse <= threshold).float().mean(dim=1)
        out["fscore_%dm" % int(threshold)] = 2.0 * precision * recall / (
            precision + recall
        ).clamp_min(1e-8)
    # Vertical-structure diagnostics. Chamfer is blind to the flat-sheet failure
    # (points collapsed onto z~0 still match the 73.8% of ground-truth mass that
    # lies below zero), so these are reported for every protocol and baseline.
    edges = torch.tensor([0.0, 3.0, 6.0, 10.0, 15.0, 25.0], device=predicted.device)
    predicted_z = predicted[..., 2].float()
    target_z = target[..., 2].float()
    predicted_hist = torch.zeros(predicted_z.shape[0], edges.numel() + 1, device=predicted.device)
    target_hist = torch.zeros_like(predicted_hist)
    predicted_hist.scatter_add_(1, torch.bucketize(predicted_z, edges), torch.ones_like(predicted_z))
    target_hist.scatter_add_(1, torch.bucketize(target_z, edges), torch.ones_like(target_z))
    predicted_hist = predicted_hist / predicted_hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_hist = target_hist / target_hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    out["height_kl"] = (
        target_hist * (target_hist.clamp_min(1e-8).log() - predicted_hist.clamp_min(1e-8).log())
    ).sum(dim=1)
    out["z_std_m"] = predicted_z.std(dim=1, unbiased=False)
    out["z_std_error_m"] = (predicted_z.std(dim=1, unbiased=False)
                            - target_z.std(dim=1, unbiased=False)).abs()
    out["z_above3m_fraction"] = (predicted_z > 3.0).float().mean(dim=1)
    out["z_above3m_fraction_true"] = (target_z > 3.0).float().mean(dim=1)
    return {name: value.detach().cpu().numpy() for name, value in out.items()}


def append_arrays(store: Dict[str, List[np.ndarray]], prefix: str, metrics: Dict[str, np.ndarray]) -> None:
    for name, value in metrics.items():
        store["%s_%s" % (prefix, name)].append(np.asarray(value))


def summarize_arrays(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {name: float(np.mean(value)) for name, value in arrays.items()}


@torch.no_grad()
def evaluate(
    model: PaperWWM,
    head: MatchedCompletionHead,
    loader: DataLoader,
    memory: M.TrainPositionMemory,
    arm: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    pos_scale: float,
    include_baselines: bool,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    model.eval()
    head.eval()
    decoder = model.online_stem.point.dvae.decoder
    collected: Dict[str, List[np.ndarray]] = defaultdict(list)
    for batch in loader:
        h, traj, points, point_scale = R.move_batch(batch, device)
        true_scale_m = point_scale.float().reshape(-1) * float(pos_scale)
        target_cloud_m = points.float() * true_scale_m[:, None, None]
        map_points, map_scale_m, nearest_distance_m = query_map(memory, batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            map_target = R.target_point_features(model, map_points)
            wireless, wireless_present = prepare_wireless(model, h, traj, arm, training=False)
            for protocol, map_present, shuffle_wireless in (
                ("sparse_real", torch.ones(h.shape[0], device=device), False),
                ("sparse_shuffled", torch.ones(h.shape[0], device=device), True),
                ("wireless_real", torch.zeros(h.shape[0], device=device), False),
                ("wireless_shuffled", torch.zeros(h.shape[0], device=device), True),
            ):
                controlled = wireless
                if shuffle_wireless and wireless.shape[0] > 1 and wireless_present:
                    controlled = wireless.roll(1, dims=0)
                prediction = head(
                    controlled,
                    map_target["features"],
                    map_target["centers"].float() * map_scale_m[:, None, None],
                    map_scale_m,
                    nearest_distance_m,
                    map_present,
                    wireless_present,
                )
                cloud = decode_cloud(decoder, prediction, points.shape[1])
                append_arrays(collected, protocol, per_sample_metrics(cloud, target_cloud_m))
            if include_baselines:
                raw = map_points.float() * map_scale_m[:, None, None]
                _, map_fine = decoder(map_target["features"].float())
                map_dvae = R.sampled_whole_cloud(
                    map_fine, map_target["centers"], points.shape[1]
                ).float() * map_scale_m[:, None, None]
                true_target = R.target_point_features(model, points)
                _, ceiling_fine = decoder(true_target["features"].float())
                ceiling = R.sampled_whole_cloud(
                    ceiling_fine, true_target["centers"], points.shape[1]
                ).float() * true_scale_m[:, None, None]
                append_arrays(collected, "map_raw", per_sample_metrics(raw, target_cloud_m))
                append_arrays(collected, "map_dvae", per_sample_metrics(map_dvae, target_cloud_m))
                append_arrays(collected, "dvae_ceiling", per_sample_metrics(ceiling, target_cloud_m))
        collected["nearest_position_m"].append(nearest_distance_m.detach().cpu().numpy())
        collected["true_scale_m"].append(true_scale_m.detach().cpu().numpy())
    arrays = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    summary = summarize_arrays(arrays)
    summary["samples"] = float(len(arrays["nearest_position_m"]))
    return summary, arrays


def save_checkpoint(
    path: Path,
    head: MatchedCompletionHead,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    best: float,
    config: Dict[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "best_validation_chamfer_m": float(best),
            "config": config,
        },
    )


def load_head_checkpoint(path: Path, head: MatchedCompletionHead) -> Tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(payload["head"], strict=True)
    return int(payload.get("step", 0)), float(
        payload.get("best_validation_chamfer_m", float("inf"))
    )


def build_protocol(
    cli: argparse.Namespace,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    train_root = Path(cli.train_root).resolve()
    test_root = Path(cli.test_root).resolve()
    train_scenarios = discover_scenarios(train_root, cli.train_split, None, None)
    head_train_scenarios, head_val_scenarios = D.split_files(
        train_scenarios, cli.val_files_per_city, cli.protocol_seed
    )
    train_quality_path = train_root / "training_ready_v3" / "quality_index.npz"
    test_quality_path = test_root / "training_ready_v3" / "quality_index.npz"
    stats_path = train_root / "training_ready_v3" / "csi_stats_global.json"
    if str(getattr(args, "csi_normalization_scope", "global")) != "global":
        raise RuntimeError(
            "Independent-city evaluation requires train-fitted global CSI statistics; "
            "checkpoint requests normalization_scope=%s"
            % getattr(args, "csi_normalization_scope", None)
        )
    if not stats_path.exists():
        raise FileNotFoundError("Missing train-fitted CSI statistics: %s" % stats_path)
    train_quality = load_quality_index(train_quality_path) if train_quality_path.exists() else None
    test_quality = load_quality_index(test_quality_path) if test_quality_path.exists() else None
    train_stats = load_per_city_csi_stats(stats_path) if stats_path.exists() else None
    train_dataset = dataset_for_root(
        args, train_root, head_train_scenarios, train_quality, train_stats
    )
    val_dataset = dataset_for_root(args, train_root, head_val_scenarios, train_quality, train_stats)
    test_scenarios = [
        row
        for row in discover_scenarios(test_root, cli.test_split, None, None)
        if str(row.city_key) == str(cli.test_city)
    ]
    if not test_scenarios:
        raise RuntimeError("No independent test scenarios for city=%s" % cli.test_city)
    test_dataset = dataset_for_root(args, test_root, test_scenarios, test_quality, train_stats)
    support_indices, query_indices, support_manifest = choose_support_query(
        test_dataset, cli.test_support_fraction, cli.protocol_seed
    )
    train_indices = subset_fixed(
        list(range(len(train_dataset))), cli.train_samples, cli.protocol_seed + 1
    )
    val_indices = subset_fixed(
        list(range(len(val_dataset))), cli.val_samples, cli.protocol_seed + 2
    )
    query_indices = subset_fixed(query_indices, cli.test_samples, cli.protocol_seed + 3)
    query_evaluation_indices = shuffled_fixed(query_indices, cli.protocol_seed + 4)
    train_memory = M.TrainPositionMemory(
        train_dataset,
        list(range(len(train_dataset))),
        memory_fraction=cli.train_memory_fraction,
        seed=cli.protocol_seed,
    )
    test_memory = M.TrainPositionMemory(
        test_dataset, support_indices, memory_fraction=1.0, seed=cli.protocol_seed
    )
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "query_indices": query_evaluation_indices,
        "support_indices": support_indices,
        "train_memory": train_memory,
        "test_memory": test_memory,
        "manifest": {
            "train_root": str(train_root),
            "test_root": str(test_root),
            "test_city": str(cli.test_city),
            "train_cities": sorted({str(row.city_key) for row in train_scenarios}),
            "test_cities": sorted({str(row.city_key) for row in test_scenarios}),
            "city_disjoint": str(cli.test_city)
            not in {str(row.city_key) for row in train_scenarios},
            "head_train_files": int(len(head_train_scenarios)),
            "head_validation_files": int(len(head_val_scenarios)),
            "head_train_file_sha256": string_list_sha256(
                [str(row.base) for row in head_train_scenarios]
            ),
            "head_validation_file_sha256": string_list_sha256(
                [str(row.base) for row in head_val_scenarios]
            ),
            "head_train_samples_available": int(len(train_dataset)),
            "head_train_samples_used": int(len(train_indices)),
            "head_validation_samples_available": int(len(val_dataset)),
            "head_validation_samples_used": int(len(val_indices)),
            "independent_samples_available": int(len(test_dataset)),
            "test_support_fraction": float(cli.test_support_fraction),
            "test_support_samples": int(len(support_indices)),
            "test_query_samples": int(len(query_indices)),
            "support_query_overlap": int(len(set(support_indices) & set(query_indices))),
            "support_index_sha256": string_list_sha256(
                [str(value) for value in support_indices]
            ),
            "query_index_sha256": string_list_sha256(
                [str(value) for value in query_indices]
            ),
            "query_evaluation_order_sha256": hashlib.sha256(
                "\n".join(str(value) for value in query_evaluation_indices).encode("utf-8")
            ).hexdigest(),
            "wireless_shuffle_protocol": (
                "fixed globally randomized query order, then non-identity roll within each batch"
            ),
            "support_partition": support_manifest,
            "csi_normalization_scope": str(args.csi_normalization_scope),
            "csi_context_rms_normalization": bool(args.csi_context_rms_normalization),
            "context_rms_feature": bool(args.context_rms_feature),
            "csi_statistics_source": str(stats_path),
            "csi_statistics_sha256": file_sha256(stats_path),
            "test_statistics_fitted": False,
            "checkpoint_selection": "seven-city scenario-file validation only",
            "independent_test_usage": "one final evaluation after checkpoint selection",
        },
    }


def real_data_dry_run(cli: argparse.Namespace) -> None:
    """Exercise one real train/validation/test batch without reporting test metrics."""
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device if cli.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(cli.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if cli.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    model, args, source_checkpoint = load_backbone(
        cli.arm,
        Path(cli.v18_checkpoint).resolve(),
        Path(cli.v17_checkpoint).resolve(),
        Path(cli.train_root).resolve(),
        device,
        Path(cli.v19_checkpoint).resolve() if cli.v19_checkpoint else None,
    )
    protocol = build_protocol(cli, args)
    manifest = protocol["manifest"]
    if not manifest["city_disjoint"] or manifest["support_query_overlap"] != 0:
        raise RuntimeError("Real-data protocol isolation check failed")
    dvae = model.online_stem.point.dvae
    head = MatchedCompletionHead(
        model.latent_dim, dvae.decoder_dims, dvae.num_group,
        cli.hidden_dim, cli.depth, cli.heads, cli.ffn_mult,
        cli.max_center_delta_m, cli.max_absolute_center_m, cli.initial_scale_m,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cli.lr)
    train_loader = make_loader(
        protocol["train_dataset"], protocol["train_indices"][: cli.batch_size],
        min(cli.batch_size, len(protocol["train_indices"])), 0, False, cli.seed,
    )
    checked: Dict[str, Any] = {}
    batch = next(iter(train_loader))
    h, traj, points, point_scale = R.move_batch(batch, device)
    true_scale_m = point_scale.float().reshape(-1) * float(args.pos_scale)
    target_cloud_m = points.float() * true_scale_m[:, None, None]
    map_points, map_scale_m, nearest_distance_m = query_map(
        protocol["train_memory"], batch, device
    )
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_amp
    ):
        map_target = R.target_point_features(model, map_points)
        target = R.target_point_features(model, points)
        wireless, wireless_present = prepare_wireless(model, h, traj, cli.arm, True)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        prediction = head(
            wireless, map_target["features"],
            map_target["centers"].float() * map_scale_m[:, None, None],
            map_scale_m, nearest_distance_m, torch.ones(h.shape[0], device=device),
            wireless_present,
        )
        predicted_cloud_m = decode_cloud(dvae.decoder, prediction, points.shape[1])
        loss, components = training_loss(
            prediction, predicted_cloud_m, target, target_cloud_m, true_scale_m, cli
        )
    if not torch.isfinite(loss) or not torch.isfinite(predicted_cloud_m).all():
        raise FloatingPointError("Non-finite real training batch")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(head.parameters(), cli.grad_clip)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Non-finite real training gradient")
    checked["train"] = {
        "batch": int(h.shape[0]), "loss_finite": True,
        "gradient_finite": True, "loss": float(loss.detach()),
        "components": {name: float(value.detach()) for name, value in components.items()},
    }
    # Validation and Frankfurt are shape/finite audits only.  No metric is computed,
    # logged, or used for a model/hyperparameter decision here.
    for split_name, dataset, indices, memory in (
        ("validation", protocol["val_dataset"], protocol["val_indices"], protocol["train_memory"]),
        ("frankfurt_query", protocol["test_dataset"], protocol["query_indices"], protocol["test_memory"]),
    ):
        loader = make_loader(dataset, indices[:1], 1, 0, False, cli.protocol_seed)
        audit_batch = next(iter(loader))
        ah, atraj, apoints, _ = R.move_batch(audit_batch, device)
        amap, ascale, adistance = query_map(memory, audit_batch, device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            amap_target = R.target_point_features(model, amap)
            awireless, apresent = prepare_wireless(model, ah, atraj, cli.arm, False)
            outputs_finite = True
            shapes: Dict[str, List[int]] = {}
            for path_name, map_present in (
                ("sparse_map", torch.ones(ah.shape[0], device=device)),
                ("csi_only", torch.zeros(ah.shape[0], device=device)),
            ):
                aprediction = head(
                    awireless, amap_target["features"],
                    amap_target["centers"].float() * ascale[:, None, None],
                    ascale, adistance, map_present, apresent,
                )
                acloud = decode_cloud(dvae.decoder, aprediction, apoints.shape[1])
                outputs_finite = outputs_finite and bool(torch.isfinite(acloud).all())
                shapes[path_name] = list(acloud.shape)
        if not outputs_finite:
            raise FloatingPointError("Non-finite %s audit output" % split_name)
        checked[split_name] = {"outputs_finite": True, "cloud_shapes": shapes}
    report = {
        "status": "PASS", "arm": cli.arm, "seed": int(cli.seed),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "trainable_parameters": int(sum(p.numel() for p in head.parameters())),
        "protocol": manifest, "real_batch_checks": checked,
        "independent_test_metrics_inspected": False,
    }
    atomic_json(output_dir / "real_data_dry_run.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def train_and_evaluate(cli: argparse.Namespace) -> None:
    if cli.arm not in ARMS:
        raise ValueError("Unknown arm: %s" % cli.arm)
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cli.device if cli.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(cli.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if cli.amp_dtype == "bf16" else torch.float16
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    random.seed(cli.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    model, args, source_checkpoint = load_backbone(
        cli.arm,
        Path(cli.v18_checkpoint).resolve(),
        Path(cli.v17_checkpoint).resolve(),
        Path(cli.train_root).resolve(),
        device,
        Path(cli.v19_checkpoint).resolve() if cli.v19_checkpoint else None,
    )
    protocol = build_protocol(cli, args)
    manifest = protocol["manifest"]
    if not manifest["city_disjoint"]:
        raise RuntimeError("Independent test city overlaps WWM training cities")
    if manifest["support_query_overlap"] != 0:
        raise RuntimeError("Independent support/query leakage")
    atomic_json(output_dir / "protocol_manifest.json", manifest)
    train_loader = make_loader(
        protocol["train_dataset"], protocol["train_indices"], cli.batch_size,
        cli.num_workers, True, cli.seed
    )
    val_loader = make_loader(
        protocol["val_dataset"], protocol["val_indices"], cli.eval_batch_size,
        cli.num_workers, False, cli.protocol_seed
    )
    test_loader = make_loader(
        protocol["test_dataset"], protocol["query_indices"], cli.eval_batch_size,
        cli.num_workers, False, cli.protocol_seed
    )
    dvae = model.online_stem.point.dvae
    head = MatchedCompletionHead(
        model.latent_dim,
        dvae.decoder_dims,
        dvae.num_group,
        cli.hidden_dim,
        cli.depth,
        cli.heads,
        cli.ffn_mult,
        cli.max_center_delta_m,
        cli.max_absolute_center_m,
        cli.initial_scale_m,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cli.lr, betas=(0.9, 0.95), eps=1e-10,
        weight_decay=cli.weight_decay
    )
    config = vars(cli).copy() | {
        "task": "paper_grade_wireless_sparse_map_completion",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "trainable_parameters": int(sum(parameter.numel() for parameter in head.parameters())),
        "protocol": manifest,
    }
    atomic_json(output_dir / "config.json", config)
    print(
        "arm=%s params=%d train=%d val=%d frankfurt_support=%d query=%d"
        % (
            cli.arm, config["trainable_parameters"], len(protocol["train_indices"]),
            len(protocol["val_indices"]), len(protocol["support_indices"]),
            len(protocol["query_indices"]),
        ), flush=True
    )
    best = float("inf")
    step = 0
    history: List[Dict[str, float]] = []
    start = time.time()
    stop_marker = output_dir / "STOP_REQUESTED"
    for epoch in range(cli.epochs):
        head.train()
        for batch in train_loader:
            if step >= cli.max_steps:
                break
            h, traj, points, point_scale = R.move_batch(batch, device)
            true_scale_m = point_scale.float().reshape(-1) * float(args.pos_scale)
            target_cloud_m = points.float() * true_scale_m[:, None, None]
            map_points, map_scale_m, nearest_distance_m = query_map(
                protocol["train_memory"], batch, device
            )
            map_present = (
                torch.rand(h.shape[0], device=device) >= float(cli.map_dropout)
            ).float()
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                map_target = R.target_point_features(model, map_points)
                target = R.target_point_features(model, points)
                wireless, wireless_present = prepare_wireless(
                    model, h, traj, cli.arm, training=True
                )
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                prediction = head(
                    wireless,
                    map_target["features"],
                    map_target["centers"].float() * map_scale_m[:, None, None],
                    map_scale_m,
                    nearest_distance_m,
                    map_present,
                    wireless_present,
                )
                predicted_cloud_m = decode_cloud(dvae.decoder, prediction, points.shape[1])
                loss, components = training_loss(
                    prediction, predicted_cloud_m, target, target_cloud_m,
                    true_scale_m, cli
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss at step=%d" % step)
            lr = R.lr_at_step(
                step, cli.max_steps, cli.warmup_steps, cli.start_lr, cli.lr, cli.final_lr
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(head.parameters(), cli.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient at step=%d" % step)
            optimizer.step()
            elapsed = max(time.time() - start, 1e-6)
            row: Dict[str, float] = {
                "step": float(step), "epoch": float(epoch), "loss": float(loss.detach()),
                "grad_norm": float(grad_norm), "lr": float(lr),
                "samples_per_s": float((step + 1) * cli.batch_size / elapsed),
                "eta_s": float(max(cli.max_steps - step - 1, 0) * elapsed / max(step + 1, 1)),
            }
            row.update({"loss/" + name: float(value.detach()) for name, value in components.items()})
            history.append(row)
            step += 1
            if step % cli.log_every == 0 or step == 1:
                print(
                    "step=%05d loss=%.4f whole=%.3f center=%.3f feature=%.3f "
                    "scale=%.3f map=%.2f grad=%.2f samples_s=%.2f eta_h=%.2f"
                    % (
                        step, row["loss"], row["loss/whole_m"], row["loss/center_set_m"],
                        row["loss/feature"], row["loss/scale_m"], row["loss/map_rate"],
                        row["grad_norm"], row["samples_per_s"], row["eta_s"] / 3600.0,
                    ), flush=True
                )
            atomic_json(
                output_dir / "metrics_live.json",
                {"step": step, "arm": cli.arm, "last": row, "best_validation_cd_m": best},
            )
            if step % cli.val_every == 0:
                summary, _ = evaluate(
                    model, head, val_loader, protocol["train_memory"], cli.arm,
                    device, amp_dtype, use_amp, args.pos_scale, False
                )
                val_cd = float(summary["sparse_real_chamfer_m"])
                print(
                    "validation step=%d sparseCD=%.4f wirelessCD=%.4f sparseShuffle=%.4f n=%d"
                    % (
                        step, val_cd, summary["wireless_real_chamfer_m"],
                        summary["sparse_shuffled_chamfer_m"], int(summary["samples"]),
                    ), flush=True
                )
                if val_cd < best:
                    best = val_cd
                    save_checkpoint(
                        output_dir / "checkpoint_best.pt", head, optimizer,
                        step, epoch, best, config
                    )
                head.train()
            if step % cli.save_every == 0:
                save_checkpoint(
                    output_dir / "checkpoint_latest.pt", head, optimizer,
                    step, epoch, best, config
                )
            if stop_marker.exists():
                save_checkpoint(
                    output_dir / "checkpoint_latest.pt", head, optimizer,
                    step, epoch, best, config
                )
                print("stop requested; checkpoint saved", flush=True)
                return
        if step >= cli.max_steps:
            break
    save_checkpoint(
        output_dir / "checkpoint_latest.pt", head, optimizer, step,
        epoch if "epoch" in locals() else 0, best, config
    )
    best_path = output_dir / "checkpoint_best.pt"
    if not best_path.exists():
        raise RuntimeError("Training produced no validation-selected checkpoint")
    selected_step, selected_metric = load_head_checkpoint(best_path, head)
    # This is deliberately the first and only independent-city evaluation in the run.
    test_summary, test_arrays = evaluate(
        model, head, test_loader, protocol["test_memory"], cli.arm,
        device, amp_dtype, use_amp, args.pos_scale, True
    )
    np.savez_compressed(output_dir / "independent_test_per_sample.npz", **test_arrays)
    results = {
        "task": "paper_grade_wireless_sparse_map_completion",
        "arm": cli.arm,
        "seed": int(cli.seed),
        "selected_step": int(selected_step),
        "selected_validation_chamfer_m": float(selected_metric),
        "independent_test": test_summary,
        "protocol": manifest,
        "history_tail": history[-20:],
    }
    atomic_json(output_dir / "results.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def self_test() -> None:
    torch.manual_seed(7)
    batch, tokens, latent_dim, feature_dim = 3, 8, 16, 12
    head = MatchedCompletionHead(
        latent_dim, feature_dim, tokens, 16, 1, 4, 2, 5.0, 25.0, 30.0
    )
    latent = torch.randn(batch, tokens, latent_dim)
    features = torch.randn(batch, tokens, feature_dim)
    centers = torch.randn(batch, tokens, 3)
    scale = torch.full((batch,), 30.0)
    distance = torch.ones(batch)
    mixed = torch.tensor([1.0, 0.0, 1.0])
    prediction = head(latent, features, centers, scale, distance, mixed, True)
    if prediction["features"].shape != features.shape:
        raise AssertionError("Feature shape mismatch")
    if prediction["centers_m"].shape != centers.shape:
        raise AssertionError("Center shape mismatch")
    if not torch.allclose(prediction["scale_m"][[0, 2]], scale[[0, 2]]):
        raise AssertionError("Map-present scale must be preserved")
    null_a = head(latent, features, centers, scale, distance, torch.zeros(batch), False)
    null_b = head(torch.randn_like(latent), features, centers, scale, distance, torch.zeros(batch), False)
    if not torch.allclose(null_a["features"], null_b["features"]):
        raise AssertionError("map_only head depends on discarded wireless values")
    fake = type("Fake", (), {})()
    fake.index = [(0, index) for index in range(100)]
    fake.scenarios = [type("Scenario", (), {"city_key": "x"})()]
    support, query, manifest = choose_support_query(fake, 0.05, 42)
    if len(support) != 5 or len(query) != 95 or set(support) & set(query):
        raise AssertionError("Support/query split invalid: %s" % manifest)
    loss = prediction["features"].square().mean() + prediction["centers_m"].square().mean()
    loss.backward()
    if not any(parameter.grad is not None for parameter in head.parameters()):
        raise AssertionError("No head gradients")

    # Height term: the soft histogram must agree with hard binning, must punish the
    # flat-sheet degenerate solution far harder than a faithful cloud, and must pass
    # gradient back to the point coordinates.
    edges = height_edges_tensor("0,3,6,10,15,25", torch.device("cpu"))
    truth = torch.cat(
        [torch.rand(2, 700, 3) * torch.tensor([40.0, 40.0, -4.0]),
         torch.rand(2, 300, 3) * torch.tensor([40.0, 40.0, 20.0])], dim=1
    )
    soft = height_histogram(truth, edges, 0.1)
    hard = torch.zeros_like(soft)
    hard.scatter_add_(1, torch.bucketize(truth[..., 2], edges), torch.ones_like(truth[..., 2]))
    hard = hard / hard.sum(dim=1, keepdim=True)
    if float((soft - hard).abs().max()) > 0.02:
        raise AssertionError("Soft height histogram disagrees with hard binning: %.4f"
                             % float((soft - hard).abs().max()))
    flat = truth.clone()
    flat[..., 2] = 0.0
    flat.requires_grad_(True)
    faithful_kl, _ = height_loss(truth + 0.05, truth, edges, 1.0)
    flat_kl, flat_moment = height_loss(flat, truth, edges, 1.0)
    if float(flat_kl) <= float(faithful_kl) * 3.0:
        raise AssertionError("Height KL fails to punish the flat sheet: %.4f vs %.4f"
                             % (float(flat_kl), float(faithful_kl)))
    (flat_kl + flat_moment).backward()
    if flat.grad is None or float(flat.grad[..., 2].abs().sum()) <= 0.0:
        raise AssertionError("Height loss produced no gradient on z")
    print("PASS paper_protocol_self_test params=%d height_kl(flat/faithful)=%.3f/%.3f"
          % (sum(p.numel() for p in head.parameters()), float(flat_kl), float(faithful_kl)))


def parser() -> argparse.ArgumentParser:
    root = ROOT.parent
    out = ROOT / "outputs" / "training_runs" / "wwm_downstream5_paper_protocol"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arm", choices=ARMS, default="v18")
    p.add_argument("--v18-checkpoint", default=str(ROOT / "outputs/training_runs/wwm_pretrain_v18_point_grounding/checkpoint.pt"))
    p.add_argument("--v19-checkpoint", default=str(ROOT / "outputs/training_runs/wwm_pretrain_v19_height_grounding/checkpoint.pt"))
    p.add_argument("--v17-checkpoint", default=str(ROOT / "outputs/training_runs/wwm_pretrain_v17_em_physics/checkpoint.pt"))
    p.add_argument("--train-root", default=str(root / "训练集/WWM_七城训练集_16to4_contextRMS_v3"))
    p.add_argument("--test-root", default=str(root / "测试集/WWM_两城测试集_16to4_contextRMS_v3"))
    p.add_argument("--output-dir", default=str(out / "v18_seed42"))
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--test-city", default="frankfurt_bankenviertel")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--train-samples", type=int, default=0)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--test-samples", type=int, default=0)
    p.add_argument("--val-files-per-city", type=int, default=2)
    p.add_argument("--train-memory-fraction", type=float, default=0.05)
    p.add_argument("--test-support-fraction", type=float, default=0.05)
    p.add_argument("--map-dropout", type=float, default=0.25)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn-mult", type=int, default=3)
    p.add_argument("--max-center-delta-m", type=float, default=12.0)
    p.add_argument("--max-absolute-center-m", type=float, default=35.0)
    p.add_argument("--initial-scale-m", type=float, default=30.0)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--start-lr", type=float, default=1e-5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--final-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--whole-weight", type=float, default=0.5)
    p.add_argument("--center-weight", type=float, default=1.0)
    p.add_argument("--center-set-weight", type=float, default=1.0)
    p.add_argument("--feature-weight", type=float, default=0.5)
    p.add_argument("--scale-weight", type=float, default=0.2)
    p.add_argument("--residual-weight", type=float, default=0.001)
    # Decode-space height supervision. Default 0 so every previously published arm
    # reproduces bit-for-bit; the v19 method arm turns it on and is compared against
    # the same-config controls, never against a run with a different head objective.
    p.add_argument("--height-weight", type=float, default=0.0)
    p.add_argument("--height-moment-weight", type=float, default=0.5)
    p.add_argument("--height-edges", default="0,3,6,10,15,25")
    p.add_argument("--height-sigma-m", type=float, default=1.0)
    p.add_argument("--loss-point-stride", type=int, default=1)
    p.add_argument("--sinkhorn-temperature", type=float, default=0.05)
    p.add_argument("--sinkhorn-iterations", type=int, default=8)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocol-seed", type=int, default=20260812)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return p


if __name__ == "__main__":
    cli = parser().parse_args()
    if cli.self_test:
        self_test()
    elif cli.dry_run:
        real_data_dry_run(cli)
    else:
        train_and_evaluate(cli)
