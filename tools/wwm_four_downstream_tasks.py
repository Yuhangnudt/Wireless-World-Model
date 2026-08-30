#!/usr/bin/env python
"""Train and evaluate the four WWM downstream tasks on Singapore data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wwm.common import (atomic_json_dump, inverse_csi_transform, patchify_csi_tokens,
                        seed_everything, unpatchify_csi_tokens)
from wwm.data import WWMDataset, discover_scenarios, load_per_city_csi_stats, load_quality_index
from wwm.engine import extract_point_input, move_point_input
from wwm.metrics import nmse_db_metric, sgcs_metric_at_step
from wwm.model import PaperWWM
from wwm.task_heads import BeamPredictionHead, CSICompressionHead, LocalizationHead


PAPER = {
    "temporal_velocity_sgcs": 0.776,
    "compression_velocity_sgcs": {1024: 0.6468, 512: 0.6623, 256: 0.6598, 128: 0.7454},
    "beam_velocity_top1": 0.978,
    "beam_velocity_gain_ratio": 0.999,
    "localization_velocity_mean_m": 1.226949,
}
RATIO_LAYOUT = {1024: (1, 32), 512: (2, 32), 256: (4, 32), 128: (8, 32)}
CACHE_PROTOCOL_VERSION = 6
COMPRESSION_FUSION_MULTIPLIER = 0.25
BEAM_FUSION_MULTIPLIER = 0.25
LOCALIZATION_FUSION_MULTIPLIER = 1.0
MULTILEVEL_FEATURE_NORMALIZATION = "per_token_layernorm"
SPEED_RE = re.compile(r"_(\d+(?:p\d+)?)kmh_", re.IGNORECASE)


def json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def resolve_training_sidecar(root: Path, explicit: Optional[str], filename: str) -> Optional[Path]:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError("Missing sidecar: %s" % path)
        return path
    for version in ("training_ready_v3", "training_ready_v2"):
        candidate = root / version / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def speed_from_base(base: str) -> int:
    match = SPEED_RE.search(base)
    if not match:
        raise ValueError("Cannot infer speed from scenario name: %s" % base)
    return int(round(float(match.group(1).replace("p", "."))))


def singapore_scenarios(dataset_root: Path, split: str, city: str = "singapore_cbd") -> List[Any]:
    scenarios = discover_scenarios(dataset_root, split, None, None)
    if city and city != "any":
        selected = [item for item in scenarios if item.city_key == city]
    else:
        selected = list(scenarios)
    if not selected:
        raise RuntimeError("No scenarios found in split %s (city=%s)" % (split, city))
    return selected


def fixed_file_split(scenarios: List[Any], validation_fraction: float) -> Tuple[List[Any], List[Any]]:
    train: List[Any] = []
    val: List[Any] = []
    by_speed: Dict[int, List[Any]] = {}
    for scenario in scenarios:
        by_speed.setdefault(speed_from_base(scenario.base), []).append(scenario)
    for speed, group in sorted(by_speed.items()):
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha1(("wwm-val:%d:%s" % (speed, item.base)).encode("utf-8")).hexdigest(),
        )
        count = max(1, int(round(len(ordered) * validation_fraction)))
        val.extend(ordered[:count])
        train.extend(ordered[count:])
    if not train or not val:
        raise RuntimeError("Fixed train/validation split is empty")
    return sorted(train, key=lambda item: item.base), sorted(val, key=lambda item: item.base)


def scenario_content_hashes(scenarios: List[Any], attribute: str) -> set[str]:
    hashes: set[str] = set()
    for scenario in scenarios:
        digest = hashlib.sha256()
        with Path(getattr(scenario, attribute)).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes.add(digest.hexdigest())
    return hashes


class ProtocolDataset(Dataset):
    def __init__(self, base: WWMDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_index, sample_index = self.base.index[index]
        row = self.base[index]
        row["speed_kmh"] = torch.tensor(speed_from_base(self.base.scenarios[file_index].base), dtype=torch.int16)
        row["sample_index"] = torch.tensor(sample_index, dtype=torch.int32)
        return row


def make_protocol_dataset(
    scenarios: List[Any],
    dataset_root: Path,
    model_args: argparse.Namespace,
    city_root: Optional[Path] = None,
    quality_lookup: Optional[Dict[Tuple[str, int], Dict[str, float]]] = None,
    filter_quality: bool = True,
    use_quality_context_rms: bool = True,
) -> ProtocolDataset:
    base = WWMDataset(
        scenarios=scenarios,
        city_root=city_root if city_root is not None else dataset_root / "cities",
        context_steps=int(model_args.context_steps),
        future_steps=int(model_args.future_steps),
        point_count=int(model_args.point_count),
        point_pool_count=int(model_args.point_pool_count),
        point_pool_mode=str(model_args.point_pool_mode),
        point_normalization=str(model_args.point_normalization),
        trajectory_features_mode=str(model_args.trajectory_features),
        pos_scale=float(model_args.pos_scale),
        point_scale=float(model_args.point_scale),
        csi_transform=str(model_args.csi_transform),
        signed_log_eps=float(model_args.signed_log_eps),
        csi_mean=float(model_args.csi_mean),
        csi_std=float(model_args.csi_std),
        drop_zero_csi_samples=bool(model_args.drop_zero_csi_samples),
        seed=int(model_args.seed),
        normalization_scope=str(getattr(model_args, "csi_normalization_scope", "global")),
        per_city_stats=getattr(model_args, "_per_city_stats", None),
        context_rms_normalization=bool(getattr(model_args, "csi_context_rms_normalization", False)),
        context_rms_feature=bool(getattr(model_args, "context_rms_feature", False)),
        quality_lookup=quality_lookup,
        filter_accepted=bool(filter_quality and quality_lookup),
        use_quality_context_rms=use_quality_context_rms,
    )
    return ProtocolDataset(base)


def load_backbone(checkpoint_path: Path, device: torch.device) -> Tuple[PaperWWM, argparse.Namespace, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", payload.get("config", {}))
    if not isinstance(saved_args, dict):
        raise RuntimeError("Checkpoint has no configuration dictionary")
    model_args = argparse.Namespace(**saved_args)
    if (int(model_args.context_steps), int(model_args.future_steps)) != (16, 4):
        raise RuntimeError("Expected a 16->4 checkpoint")
    model = PaperWWM(model_args)
    state = payload.get("model", payload)
    incompatible = model.load_state_dict(state, strict=False)
    # Older checkpoints predate the optional EM tangent projections.  When
    # EM physics is disabled those layers are never called, so allowing their
    # default initialization preserves compatibility without weakening checks
    # for any active or unexpected parameter.
    optional_missing = {
        "em_tangent_latent_proj",
        "em_tangent_feature_proj",
    }
    missing = [
        key for key in incompatible.missing_keys
        if not (not bool(getattr(model_args, "em_physics_enable", False))
                and any(key == name or key.startswith(name + ".") for name in optional_missing))
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Backbone checkpoint mismatch: missing=%d unexpected=%d"
            % (len(missing), len(incompatible.unexpected_keys))
        )
    model.set_point_dvae_frozen(True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, model_args, int(payload.get("step", -1))


def full_mask(batch_size: int, lengths: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, sum(lengths), dtype=torch.bool, device=device)


def task_masks(batch_size: int, lengths: Tuple[int, int, int], device: torch.device) -> Dict[str, torch.Tensor]:
    nc, np_, nt = lengths
    if (nc, nt) != (640, 20):
        raise RuntimeError("Expected 640 CSI and 20 trajectory tokens, got %s" % (lengths,))
    tubelet_tokens = 64
    compression = torch.ones(batch_size, nc + np_ + nt, dtype=torch.bool, device=device)
    compression[:, 3 * tubelet_tokens : 4 * tubelet_tokens] = False
    compression[:, nc : nc + np_] = False
    compression[:, nc + np_ : nc + np_ + 16] = False

    localization = full_mask(batch_size, lengths, device)
    localization[:, 512:nc] = True
    localization[:, nc + np_ :] = True

    beam = full_mask(batch_size, lengths, device)
    beam[:, 512:nc] = True
    beam[:, nc + np_ + 16 :] = True

    if int((~compression[:, :nc]).sum(1)[0]) != 64:
        raise AssertionError("Compression must expose exactly 64 CSI tokens")
    if bool((~localization[:, nc + np_ :]).any()):
        raise AssertionError("Localization trajectory leakage detected")
    if int((~localization).sum(1)[0]) != 768:
        raise AssertionError("Localization must expose 512 CSI + 256 point tokens")
    if int((~beam).sum(1)[0]) != 784:
        raise AssertionError("Beam task must expose 512 CSI + 256 point + 16 trajectory tokens")
    return {"compression": compression, "localization": localization, "beam": beam}


def select_visible(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = (~mask).sum(1)
    if not bool(torch.all(count == count[0])):
        raise RuntimeError("Variable visible-token count is not supported in the cache")
    return encoded[~mask].reshape(encoded.shape[0], int(count[0]), encoded.shape[-1])


def normalize_multilevel_context(context: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return context
    return F.layer_norm(context, (context.shape[-1],))


def dft_beam_power(h_raw_t16: torch.Tensor, num_beams: int = 32) -> torch.Tensor:
    h = torch.complex(h_raw_t16[:, 0], h_raw_t16[:, 1])
    h = h.reshape(h.shape[0], 32, 4, 8)
    antenna = torch.arange(32, device=h.device, dtype=torch.float32)
    beam = torch.arange(num_beams, device=h.device, dtype=torch.float32)
    codebook = torch.exp(2j * math.pi * beam[:, None] * antenna[None, :] / float(num_beams)) / math.sqrt(32.0)
    received = torch.einsum("bnus,kn->bkus", h, codebook)
    return received.abs().square().sum(dim=(2, 3)).float()


def forecast_with_pretrain_head(
    model: PaperWWM,
    tokens: torch.Tensor,
    lengths: Tuple[int, int, int],
    h: torch.Tensor,
) -> torch.Tensor:
    """Decode the predictor's masked CSI tokens with the head PRE-TRAINING actually trained.

    Why this exists
    ---------------
    forecast_without_teacher() decodes through model.decoder -> decoder.to_patch.
    That projection is zero-initialised when zero_init_residual_decoder=True (the
    default whenever temporal_anchor != "none") and NEVER receives gradient during
    wwm_pretrain, because pretrain_loss reconstructs CSI through a different module,
    pretrain_csi_head.  Verified 2026-08-18 on v15/v16/v17/v19/v21: decoder.to_patch
    weight and bias are all-zero in every checkpoint, so residual == 0 and the
    "forecast" collapses onto the copy anchor.  Consequence: three checkpoints with
    demonstrably different predictor weights (predictor output sums 15044 vs 7913)
    produced bit-identical temporal SGCS to 16 significant digits, and the reported
    copy_gain (-0.012 for v17, -0.009 for v16) was pure anchor round-trip error
    rather than a model-vs-persistence comparison.

    This path mirrors pretrain_loss (model.py ~2064) exactly:
        pred_patch = pretrain_csi_head(pred[:, :nc])            # + anchor if residual head
        pred_h     = unpatchify_csi_tokens(pred_patch, ...)
    so it measures the temporal capability that pre-training actually optimised.
    """
    mask = model.future_csi_mask(h.shape[0], h.device)
    use_deep = model.deep_supervision_weight > 0 and model.deep_supervision_layers > 0
    context, context_levels = model.encode_visible_tokens(
        tokens, lengths, mask, return_intermediates=use_deep
    )
    context = model.fuse_multilevel_context(context, context_levels)
    template = model.online_stem.mask_template(h.shape[0], h.device, tokens.dtype)
    masked = model.mask_token.to(tokens.dtype) + template
    predictor_input = torch.where(mask.unsqueeze(-1), masked, context)
    predicted_tokens = model.predictor(predictor_input, lengths)
    nc = lengths[0]
    pred_patch = model.pretrain_csi_head(predicted_tokens[:, :nc])
    if bool(getattr(model, "pretrain_residual_head", False)):
        # Same anchor construction as pretrain_loss: last visible context frame's
        # patch row, tiled across every t_grid row, with the delta clamped to +-8.
        total_steps = model.context_steps + model.future_steps
        t_grid = total_steps // model.patch_t
        last_ctx = h[:, model.context_steps - model.patch_t : model.context_steps]
        anchor_row = patchify_csi_tokens(last_ctx, model.patch_t, model.patch_h, model.patch_w)
        anchor_row = anchor_row.to(pred_patch.dtype)
        anchor = anchor_row.repeat(1, t_grid, 1)[:, :nc, :]
        if getattr(model, "temporal_anchor", "copy") == "phasor":
            # Same phasor inductive bias as pretrain_loss.
            ph = model.phasor_anchor_patch(h).to(pred_patch.dtype)
            anchor[:, -ph.shape[1]:] = ph
        elif getattr(model, "temporal_anchor", "copy") == "coherent":
            # Same WRONG-EM static-channel anchor as pretrain_loss.
            mean_c = model._to_raw_csi(h)[:, : model.context_steps].mean(dim=1, keepdim=True)
            stored = model._from_raw_csi(mean_c.expand(-1, model.future_steps, -1, -1, -1).contiguous())
            co_patch = patchify_csi_tokens(stored, model.patch_t, model.patch_h, model.patch_w)
            co_rows = co_patch.reshape(h.shape[0], -1, co_patch.shape[-1]).to(pred_patch.dtype)
            anchor[:, -co_rows.shape[1]:] = co_rows
        pred_patch = anchor + torch.clamp(pred_patch, min=-8.0, max=8.0)
    total_steps = model.context_steps + model.future_steps
    full = unpatchify_csi_tokens(
        pred_patch.float(), total_steps, model.patch_t, model.patch_h, model.patch_w
    )
    return full[:, model.context_steps : total_steps]


def forecast_without_teacher(
    model: PaperWWM,
    tokens: torch.Tensor,
    lengths: Tuple[int, int, int],
    h: torch.Tensor,
) -> torch.Tensor:
    mask = model.future_csi_mask(h.shape[0], h.device)
    use_deep = model.deep_supervision_weight > 0 and model.deep_supervision_layers > 0
    context, context_levels = model.encode_visible_tokens(
        tokens, lengths, mask, return_intermediates=use_deep
    )
    context = model.fuse_multilevel_context(context, context_levels)
    template = model.online_stem.mask_template(h.shape[0], h.device, tokens.dtype)
    masked = model.mask_token.to(tokens.dtype) + template
    predictor_input = torch.where(mask.unsqueeze(-1), masked, context)
    predicted_tokens = model.predictor(predictor_input, lengths)
    nc = lengths[0]
    future_csi_mask = mask[:, :nc]
    if model.decoder_token_input == "all_csi":
        residual = model.decoder(predicted_tokens[:, :nc], future_token_mask=future_csi_mask)
    else:
        future = predicted_tokens[:, :nc][future_csi_mask].reshape(h.shape[0], -1, model.latent_dim)
        residual = model.decoder(future)
    if model.temporal_anchor == "none":
        return residual
    anchor = model.build_temporal_anchor(h).to(residual.dtype)
    return anchor + model.anchor_residual_scale.to(residual.dtype) * residual


def temporal_accumulator() -> Dict[str, Any]:
    return {
        "count": 0,
        "pred_steps": [0.0] * 4,
        "copy_steps": [0.0] * 4,
        "pred_error": 0.0,
        "copy_error": 0.0,
        "target_power": 0.0,
    }


def update_temporal(
    accumulator: Dict[str, Any], pred: torch.Tensor, copy_h: torch.Tensor, target: torch.Tensor
) -> None:
    count = int(target.shape[0])
    accumulator["count"] += count
    for step in range(4):
        accumulator["pred_steps"][step] += float(sgcs_metric_at_step(pred, target, step, mode="svd")) * count
        accumulator["copy_steps"][step] += float(sgcs_metric_at_step(copy_h, target, step, mode="svd")) * count
    accumulator["pred_error"] += float((pred - target).square().sum())
    accumulator["copy_error"] += float((copy_h - target).square().sum())
    accumulator["target_power"] += float(target.square().sum())


def finish_temporal(accumulator: Dict[str, Any]) -> Dict[str, Any]:
    count = int(accumulator["count"])
    pred_steps = [value / count for value in accumulator["pred_steps"]]
    copy_steps = [value / count for value in accumulator["copy_steps"]]
    target_power = max(float(accumulator["target_power"]), 1e-20)
    return {
        "samples": count,
        "sgcs_h": pred_steps,
        "sgcs_avg": float(np.mean(pred_steps)),
        "copy_sgcs_h": copy_steps,
        "copy_sgcs_avg": float(np.mean(copy_steps)),
        "copy_gain": float(np.mean(pred_steps) - np.mean(copy_steps)),
        "nmse_db": 10.0 * math.log10(max(float(accumulator["pred_error"]) / target_power, 1e-20)),
        "copy_nmse_db": 10.0 * math.log10(max(float(accumulator["copy_error"]) / target_power, 1e-20)),
    }


def cache_array_specs(count: int, latent_dim: int, save_levels: bool = False) -> Dict[str, Tuple[Tuple[int, ...], Any]]:
    specs = {
        "compression_tokens": ((count, 64, latent_dim), np.float16),
        "compression_target": ((count, 2, 2, 32, 32), np.float16),
        "localization_tokens": ((count, 768, latent_dim), np.float16),
        "localization_power": ((count, 1), np.float32),
        "localization_target": ((count, 2), np.float32),
        "beam_tokens": ((count, 784, latent_dim), np.float16),
        "beam_power": ((count, 32), np.float32),
        "speed_kmh": ((count,), np.int16),
    }

    # Save the final-layer endpoint needed by the task-specific residual gate.
    if save_levels:
        specs["compression_final_tokens"] = ((count, 64, latent_dim), np.float16)

    return specs


def protocol_dataset_digest(dataset: ProtocolDataset) -> str:
    digest = hashlib.sha256()
    for file_index, sample_index in dataset.base.index:
        digest.update(dataset.base.scenarios[file_index].base.encode("utf-8"))
        digest.update(b"\0")
        digest.update(int(sample_index).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def open_cache_arrays(cache_dir: Path, count: int, latent_dim: int, save_levels: bool = False) -> Dict[str, np.memmap]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: np.lib.format.open_memmap(cache_dir / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in cache_array_specs(count, latent_dim, save_levels=save_levels).items()
    }


def cache_complete(
    cache_dir: Path,
    checkpoint: Path,
    count: int,
    latent_dim: int,
    dataset_digest: Optional[str] = None,
    temporal_decoder: Optional[str] = None,
    save_levels: bool = False,
) -> bool:
    meta = json_read(cache_dir / "meta.json")
    if not checkpoint.is_file():
        return False
    checkpoint_stat = checkpoint.stat()
    # The temporal block is computed during caching, so a cache built with a
    # different decoder path carries different temporal numbers and must not be
    # reused. Pre-fix caches have no temporal_decoder field at all; treat the
    # absence as "legacy residual_decoder" so they only satisfy that request.
    recorded_decoder = meta.get("temporal_decoder", "residual_decoder")
    decoder_matches = temporal_decoder is None or recorded_decoder == temporal_decoder

    # Check if cache has multi-level features
    cache_has_levels = meta.get("save_levels", False)
    levels_match = save_levels == cache_has_levels

    metadata_matches = bool(
        meta.get("complete")
        and int(meta.get("protocol_version", -1)) == CACHE_PROTOCOL_VERSION
        and meta.get("checkpoint") == str(checkpoint)
        and int(meta.get("checkpoint_size", -1)) == checkpoint_stat.st_size
        and int(meta.get("checkpoint_mtime_ns", -1)) == checkpoint_stat.st_mtime_ns
        and int(meta.get("samples", -1)) == count
        and (dataset_digest is None or meta.get("dataset_index_sha256") == dataset_digest)
        and decoder_matches
        and levels_match
    )
    if not metadata_matches:
        return False
    for name, (shape, dtype) in cache_array_specs(count, latent_dim, save_levels=save_levels).items():
        path = cache_dir / (name + ".npy")
        if not path.is_file():
            return False
        array = np.load(path, mmap_mode="r")
        valid = tuple(array.shape) == shape and array.dtype == np.dtype(dtype)
        del array
        if not valid:
            return False
    return True


@torch.inference_mode()
def extract_split_cache(
    name: str,
    dataset: ProtocolDataset,
    model: PaperWWM,
    model_args: argparse.Namespace,
    checkpoint: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    temporal_decoder: str = "pretrain_head",
    save_levels: bool = False,
) -> Dict[str, Any]:
    cache_dir = output_root / "cache" / name
    dataset_digest = protocol_dataset_digest(dataset)
    # Only the test split carries the temporal block, so only it is decoder-sensitive.
    decoder_gate = temporal_decoder if name == "test" else None
    if cache_complete(cache_dir, checkpoint, len(dataset), int(model_args.latent_dim), dataset_digest,
                      temporal_decoder=decoder_gate, save_levels=save_levels):
        print("cache=ready split=%s samples=%d" % (name, len(dataset)), flush=True)
        return json_read(cache_dir / "meta.json")
    arrays = open_cache_arrays(cache_dir, len(dataset), int(model_args.latent_dim), save_levels=save_levels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    temporal = {40: temporal_accumulator(), 80: temporal_accumulator()}
    written = 0
    started = time.time()
    use_amp = device.type == "cuda"
    for batch_index, batch in enumerate(loader):
        h = batch["h"].to(device, non_blocking=True)
        traj = batch["traj"].to(device, non_blocking=True)
        point_full = move_point_input(extract_point_input(batch), device)
        point_local = batch["point_cloud"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            tokens, lengths = model.online_stem(h, point_full, traj)
            nc, np_, _ = lengths
            masks = task_masks(h.shape[0], lengths, device)
            use_deep = model.deep_supervision_weight > 0 and model.deep_supervision_layers > 0
            compression_ctx, compression_levels = model.encode_visible_tokens(
                tokens, lengths, masks["compression"], return_intermediates=use_deep
            )
            beam_ctx, beam_levels = model.encode_visible_tokens(
                tokens, lengths, masks["beam"], return_intermediates=use_deep
            )
            compression_ctx = model.fuse_multilevel_context(
                compression_ctx, compression_levels, COMPRESSION_FUSION_MULTIPLIER
            )
            beam_ctx = model.fuse_multilevel_context(
                beam_ctx, beam_levels, BEAM_FUSION_MULTIPLIER
            )

            compression_ctx = normalize_multilevel_context(compression_ctx, use_deep)
            beam_ctx = normalize_multilevel_context(beam_ctx, use_deep)

            local_point_tokens = model.online_stem.point(point_local) + model.online_stem.point_modality
            local_tokens = tokens.clone()
            local_tokens[:, nc : nc + np_] = local_point_tokens
            localization_ctx, localization_levels = model.encode_visible_tokens(
                local_tokens, lengths, masks["localization"], return_intermediates=use_deep
            )
            localization_ctx = model.fuse_multilevel_context(
                localization_ctx, localization_levels, LOCALIZATION_FUSION_MULTIPLIER
            )

            localization_ctx = normalize_multilevel_context(localization_ctx, use_deep)

            compression_tokens = compression_ctx[:, 3 * 64 : 4 * 64]
            localization_tokens = select_visible(localization_ctx, masks["localization"])
            beam_tokens = select_visible(beam_ctx, masks["beam"])

            if name != "test":
                temporal_pred = None
            elif temporal_decoder == "pretrain_head":
                temporal_pred = forecast_with_pretrain_head(model, tokens, lengths, h)
            else:
                temporal_pred = forecast_without_teacher(model, tokens, lengths, h)

        end = written + int(h.shape[0])
        arrays["compression_tokens"][written:end] = compression_tokens.float().cpu().numpy().astype(np.float16)
        arrays["compression_target"][written:end] = h[:, 6:8].float().cpu().numpy().astype(np.float16)

        # The other endpoint is compression_tokens, i.e. the normalized pretrained
        # multi-level MLP output. Only the final endpoint is additionally cached.
        if save_levels and use_deep and compression_levels:
            final_tokens = compression_levels[-1][:, 3 * 64 : 4 * 64]
            final_tokens = normalize_multilevel_context(final_tokens, True)
            arrays["compression_final_tokens"][written:end] = final_tokens.float().cpu().numpy().astype(np.float16)

        arrays["localization_tokens"][written:end] = localization_tokens.float().cpu().numpy().astype(np.float16)
        context_rms = batch["context_rms"].numpy().astype(np.float32)
        if bool(np.any(context_rms <= 0)):
            raise RuntimeError("Localization requires positive context RMS")
        arrays["localization_power"][written:end, 0] = np.log10(context_rms)
        arrays["localization_target"][written:end] = batch["traj"][:, 15, :2].numpy().astype(np.float32)
        arrays["beam_tokens"][written:end] = beam_tokens.float().cpu().numpy().astype(np.float16)

        raw_t16 = inverse_csi_transform(h[:, 15].float(), model_args)
        power = dft_beam_power(raw_t16)
        arrays["beam_power"][written:end] = power.cpu().numpy().astype(np.float32)
        speeds = batch["speed_kmh"].numpy().astype(np.int16)
        arrays["speed_kmh"][written:end] = speeds

        if temporal_pred is not None:
            pred_raw = inverse_csi_transform(temporal_pred.float(), model_args)
            target_raw = inverse_csi_transform(h[:, 16:20].float(), model_args)
            copy_raw = inverse_csi_transform(h[:, 15:16].float(), model_args).expand_as(target_raw)
            # Bucket by whatever speeds the split actually contains. Hardcoding
            # (40, 80) silently produced an EMPTY temporal dict on the cross-frequency
            # package (speeds 5/30/60), which then failed report assembly with
            # "One or more downstream results are missing" after all four heads had
            # already trained successfully.
            for speed in sorted({int(s) for s in speeds.tolist()}):
                selected = torch.from_numpy(speeds == speed).to(device)
                if bool(selected.any()):
                    if speed not in temporal:
                        temporal[speed] = temporal_accumulator()
                    update_temporal(temporal[speed], pred_raw[selected], copy_raw[selected], target_raw[selected])

        written = end
        if batch_index % 10 == 0 or written == len(dataset):
            elapsed = max(time.time() - started, 1e-6)
            eta = elapsed * (len(dataset) - written) / max(written, 1)
            print(
                "cache split=%s samples=%d/%d rate=%.1f/s eta_min=%.1f"
                % (name, written, len(dataset), written / elapsed, eta / 60.0),
                flush=True,
            )
    for array in arrays.values():
        array.flush()
    meta: Dict[str, Any] = {
        "complete": True,
        "protocol_version": CACHE_PROTOCOL_VERSION,
        "split": name,
        "samples": len(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "dataset_index_sha256": dataset_digest,
        "feature_dtype": "float16",
        "save_levels": save_levels,
        "quality_filter": "accepted quality-index samples" if dataset.base.filter_accepted else "disabled",
        "localization_protocol": "2D target; trajectory/origin masked; log10(context RMS) retained as non-coordinate conditioning",
        "beam_protocol": "same-frequency 3.5 GHz 32-beam DFT proxy labels",
        "temporal_decoder": temporal_decoder,
        "fusion_multipliers": {
            "compression": COMPRESSION_FUSION_MULTIPLIER,
            "beam": BEAM_FUSION_MULTIPLIER,
            "localization": LOCALIZATION_FUSION_MULTIPLIER,
        },
        "multilevel_feature_normalization": MULTILEVEL_FEATURE_NORMALIZATION,
        "temporal_decoder_note": (
            "pretrain_head: decode predictor CSI tokens with pretrain_csi_head, the module "
            "wwm_pretrain actually trains. residual_decoder: legacy model.decoder path, whose "
            "to_patch projection is zero-init and never gets gradient, so the forecast collapses "
            "onto the copy anchor and is identical across checkpoints."
        ),
    }
    if name == "test":
        # Temporal forecast is keyed by the 40/80 km/h buckets; skip any bucket with no samples
        # (e.g. cross-scenario test sets at other speeds) to avoid a divide-by-zero. This metric
        # only feeds the report stage and is unused for localization-only runs.
        meta["temporal"] = {
            str(speed): finish_temporal(value)
            for speed, value in temporal.items()
            if int(value["count"]) > 0
        }
    atomic_json_dump(meta, cache_dir / "meta.json")
    return meta


@torch.inference_mode()
def extract_localization_origin_cache(
    name: str,
    dataset: ProtocolDataset,
    model: PaperWWM,
    checkpoint: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    cache_dir = output_root / "cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "localization_origin_meta.json"
    state = json_read(state_path)
    output_path = cache_dir / "localization_origin_tokens.npy"
    checkpoint_stat = checkpoint.stat()
    dataset_digest = protocol_dataset_digest(dataset)
    if (
        bool(state.get("complete"))
        and int(state.get("protocol_version", -1)) == CACHE_PROTOCOL_VERSION
        and state.get("checkpoint") == str(checkpoint)
        and int(state.get("checkpoint_size", -1)) == checkpoint_stat.st_size
        and int(state.get("checkpoint_mtime_ns", -1)) == checkpoint_stat.st_mtime_ns
        and state.get("dataset_index_sha256") == dataset_digest
        and int(state.get("samples", -1)) == len(dataset)
        and output_path.exists()
    ):
        print("localization_origin_cache=ready split=%s samples=%d" % (name, len(dataset)), flush=True)
        return
    array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), 768, int(model.latent_dim)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    written = 0
    started = time.time()
    for batch_index, batch in enumerate(loader):
        h = batch["h"].to(device, non_blocking=True)
        traj = batch["traj"].to(device, non_blocking=True)
        point = move_point_input(extract_point_input(batch), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens, lengths = model.online_stem(h, point, traj)
            mask = task_masks(h.shape[0], lengths, device)["localization"]
            use_deep = model.deep_supervision_weight > 0 and model.deep_supervision_layers > 0
            encoded, encoded_levels = model.encode_visible_tokens(
                tokens, lengths, mask, return_intermediates=use_deep
            )
            encoded = model.fuse_multilevel_context(encoded, encoded_levels)
            encoded = normalize_multilevel_context(encoded, use_deep)
            visible = select_visible(encoded, mask)
        end = written + len(h)
        array[written:end] = visible.float().cpu().numpy().astype(np.float16)
        written = end
        if batch_index % 20 == 0 or written == len(dataset):
            elapsed = max(time.time() - started, 1e-6)
            eta = elapsed * (len(dataset) - written) / max(written, 1)
            print(
                "localization_origin_cache split=%s samples=%d/%d rate=%.1f/s eta_min=%.1f"
                % (name, written, len(dataset), written / elapsed, eta / 60.0),
                flush=True,
            )
    array.flush()
    atomic_json_dump(
        {
            "complete": True,
            "protocol_version": CACHE_PROTOCOL_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "dataset_index_sha256": dataset_digest,
            "samples": len(dataset),
            "multilevel_feature_normalization": MULTILEVEL_FEATURE_NORMALIZATION,
            "protocol": "trajectory tokens masked; absolute point_origin/point_scale retained",
        },
        state_path,
    )


def array_batches(
    count: int, batch_size: int, shuffle: bool, seed: int
) -> Iterator[np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


def load_arrays(cache_dir: Path, names: Iterable[str]) -> Dict[str, np.ndarray]:
    return {name: np.load(cache_dir / (name + ".npy"), mmap_mode="r") for name in names}


def to_device(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array[indices])).to(device, non_blocking=True)


def compression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    model_args: Optional[argparse.Namespace] = None,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    raw = F.mse_loss(pred, target)
    pred_complex = torch.complex(pred[:, :, 0], pred[:, :, 1])
    target_complex = torch.complex(target[:, :, 0], target[:, :, 1])
    magnitude = F.mse_loss(pred_complex.abs(), target_complex.abs())
    phase_error = 1.0 - torch.cos(torch.angle(pred_complex) - torch.angle(target_complex))
    return raw + magnitude + 0.5 * phase_error.mean()


def new_compression_heads(latent_dim: int, device: torch.device) -> Dict[int, CSICompressionHead]:
    return {
        ratio: CSICompressionHead(
            latent_dim=latent_dim,
            bottleneck_dim=width,
            total_steps=2,
            patch_t=2,
            patch_h=4,
            patch_w=4,
            quantization_bits=4,
            input_tokens=64,
            compressed_tokens=tokens,
            decoder_layers=4,
            decoder_heads=6,
            decoder_ffn_mult=4,
        ).to(device)
        for ratio, (width, tokens) in RATIO_LAYOUT.items()
    }


class TaskSpecificResidualGate(torch.nn.Module):
    def __init__(self, initial_alpha: float = 0.0) -> None:
        super().__init__()
        initial_alpha = min(max(float(initial_alpha), 1e-4), 1.0 - 1e-4)
        self.logit = torch.nn.Parameter(torch.tensor(math.log(initial_alpha / (1.0 - initial_alpha))))

    def forward(self, final_tokens: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit).to(dtype=final_tokens.dtype)
        return final_tokens + alpha * (fused_tokens - final_tokens)

    def alpha(self) -> float:
        return float(torch.sigmoid(self.logit.detach()).cpu())


def compression_input(
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    gate: Optional[TaskSpecificResidualGate] = None,
) -> torch.Tensor:
    fused = to_device(arrays["compression_tokens"], indices, device).float()
    if gate is None:
        return fused
    final_tokens = to_device(arrays["compression_final_tokens"], indices, device).float()
    return gate(final_tokens, fused)


def initialize_compression_heads(
    heads: Dict[int, CSICompressionHead],
    train_arrays: Dict[str, np.ndarray],
    checkpoint_path: Path,
    gates: Optional[Dict[int, TaskSpecificResidualGate]] = None,
    supervised_1024_projection: bool = False,
) -> Dict[str, Any]:
    sample_count = min(1024, len(train_arrays["compression_tokens"]))
    sample_indices = np.linspace(
        0, len(train_arrays["compression_tokens"]) - 1, sample_count, dtype=np.int64
    )
    fused_latent = torch.from_numpy(
        np.asarray(train_arrays["compression_tokens"][sample_indices], dtype=np.float32)
    )
    latent = fused_latent
    if gates:
        final_latent = torch.from_numpy(
            np.asarray(train_arrays["compression_final_tokens"][sample_indices], dtype=np.float32)
        )
        initial_alpha = gates[1024].alpha()
        latent = final_latent + initial_alpha * (fused_latent - final_latent)
    flat_latent = latent.reshape(-1, int(train_arrays["compression_tokens"].shape[-1]))
    mean = flat_latent.mean(0)
    centered = flat_latent - mean
    covariance = centered.transpose(0, 1).matmul(centered) / float(max(centered.shape[0] - 1, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    supervised_direction = None
    supervised_singular_value = None
    supervised_pca_cosine = None
    if supervised_1024_projection:
        target = torch.from_numpy(
            np.asarray(train_arrays["compression_target"][sample_indices], dtype=np.float32)
        )
        target_patches = patchify_csi_tokens(target, 2, 4, 4).reshape(-1, 64)
        centered_target = target_patches - target_patches.mean(0)
        cross_covariance = centered.transpose(0, 1).matmul(centered_target)
        cross_covariance /= float(max(centered.shape[0] - 1, 1))
        left_vectors, singular_values, _ = torch.linalg.svd(cross_covariance, full_matrices=False)
        supervised_direction = left_vectors[:, 0]
        supervised_singular_value = float(singular_values[0])
        supervised_pca_cosine = float(
            torch.abs(torch.dot(supervised_direction, eigenvectors[:, 0]))
        )
    calibration: Dict[str, Any] = {}
    for ratio, head in heads.items():
        width = head.bottleneck_dim
        if ratio == 1024 and supervised_direction is not None:
            components = supervised_direction[None, :].contiguous()
            std = centered.matmul(supervised_direction).var(unbiased=False).clamp_min(1e-6).sqrt()[None]
        else:
            components = eigenvectors[:, :width].transpose(0, 1).contiguous()
            std = eigenvalues[:width].clamp_min(1e-6).sqrt()
        reduce_weight = components / std[:, None]
        expand_weight = components.transpose(0, 1) * std[None, :]
        with torch.no_grad():
            head.reduce.weight.copy_(reduce_weight.to(head.reduce.weight.device))
            head.reduce.bias.copy_((-reduce_weight.matmul(mean)).to(head.reduce.bias.device))
            head.expand.weight.copy_(expand_weight.to(head.expand.weight.device))
            head.expand.bias.copy_(mean.to(head.expand.bias.device))
        with torch.no_grad():
            calibration_tokens = latent[: min(512, sample_count)].to(head.reduce.weight.device)
            reduced = head.reduce(calibration_tokens)
            mixed = head.token_mix(reduced.transpose(1, 2)).transpose(1, 2)
            before_p99 = float(torch.quantile(mixed.abs().float().reshape(-1), 0.99).cpu())
            target_p99 = 0.5
            scale = target_p99 / max(before_p99, 1e-6)
            head.reduce.weight.mul_(scale)
            head.reduce.bias.mul_(scale)
            calibrated = head.token_mix(
                head.reduce(calibration_tokens).transpose(1, 2)
            ).transpose(1, 2)
            after_p99 = float(torch.quantile(calibrated.abs().float().reshape(-1), 0.99).cpu())
            calibration[str(ratio)] = {
                "before_p99": before_p99,
                "scale": scale,
                "after_p99": after_p99,
            }

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    weight = state.get("pretrain_context_csi_head.0.weight")
    bias = state.get("pretrain_context_csi_head.0.bias")
    initialized_decoder = torch.is_tensor(weight) and torch.is_tensor(bias)
    if initialized_decoder:
        for head in heads.values():
            with torch.no_grad():
                head.to_patch.weight.copy_(weight.to(head.to_patch.weight.device))
                head.to_patch.bias.copy_(bias.to(head.to_patch.bias.device))
    del payload, state
    return {
        "pca_samples": int(sample_count * 64),
        "pca_sampling": "uniform over full training cache; fitted on actual initial gated input",
        "top_eigenvalues": eigenvalues[:4].tolist(),
        "supervised_1024_projection": {
            "enabled": bool(supervised_1024_projection),
            "method": "PLS first left singular vector of Cov(token, aligned CSI patch)",
            "cross_covariance_singular_value": supervised_singular_value,
            "absolute_cosine_with_pca1": supervised_pca_cosine,
        },
        "quantization_calibration": calibration,
        "pretrain_context_decoder_initialized": bool(initialized_decoder),
    }


@torch.inference_mode()
def _compression_omega(context_stored: torch.Tensor, model_args: argparse.Namespace) -> torch.Tensor:
    """Per-sample phase rate omega from 16 context frames (stored domain)."""
    ctx = inverse_csi_transform(context_stored.float(), model_args)  # [B,16,2,32,32]
    c = torch.complex(ctx[:, :, 0], ctx[:, :, 1])
    acc = (c[:, 1:] * torch.conj(c[:, :-1])).sum(dim=(-2, -1)).sum(dim=1)
    return torch.angle(acc)


def _forward_csi(h_phys: torch.Tensor, model_args: argparse.Namespace) -> torch.Tensor:
    """Physical -> stored domain (inverse of inverse_csi_transform)."""
    if model_args.csi_transform == "signed_log":
        y = torch.sign(h_phys) * torch.log1p(torch.abs(h_phys) / float(model_args.signed_log_eps))
        y = (y - float(model_args.csi_mean)) / max(float(model_args.csi_std), 1e-6)
        return y
    return h_phys


def evaluate_compression_head(
    head: CSICompressionHead,
    arrays: Dict[str, np.ndarray],
    model_args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
    speed: int | None = None,
    gate: Optional[TaskSpecificResidualGate] = None,
    demod: bool = False,
) -> Dict[str, float]:
    count = len(arrays["compression_tokens"])
    step_sum = [0.0, 0.0]
    loss_sum = 0.0
    seen = 0
    for indices in array_batches(count, batch_size, False, 0):
        if speed is not None:
            keep = np.asarray(arrays["speed_kmh"][indices]) == speed
            indices = indices[keep]
            if not len(indices):
                continue
        tokens = compression_input(arrays, indices, device, gate)
        target = to_device(arrays["compression_target"], indices, device).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = head(tokens)["pred_h"]
        current = len(indices)
        loss_sum += float(compression_loss(pred, target, model_args)) * current
        pred_raw = inverse_csi_transform(pred.float(), model_args)
        target_raw = inverse_csi_transform(target.float(), model_args)
        if demod:
            # Re-modulate the second frame with exp(+j*omega) before scoring.
            omega = _compression_omega(
                to_device(arrays["context_h"], indices, device).float(), model_args)
            pc = torch.complex(pred_raw[:, :, 0], pred_raw[:, :, 1])
            pc = pc * torch.exp(1j * omega[:, None, None, None] *
                                torch.tensor([0.0, 1.0], device=device, dtype=pc.real.dtype)[None, :, None, None])
            pred_raw = torch.stack([pc.real, pc.imag], dim=2)
        for step in range(2):
            step_sum[step] += float(sgcs_metric_at_step(pred_raw, target_raw, step, mode="svd")) * current
        seen += current
    values = [value / max(seen, 1) for value in step_sum]
    return {"samples": seen, "loss": loss_sum / max(seen, 1), "sgcs_h1": values[0], "sgcs_h2": values[1], "sgcs_avg": float(np.mean(values))}


def single_batch_overfit_compression(
    head: CSICompressionHead,
    arrays: Dict[str, np.ndarray],
    model_args: argparse.Namespace,
    device: torch.device,
    gate: Optional[TaskSpecificResidualGate] = None,
) -> Dict[str, float]:
    state = copy.deepcopy(head.state_dict())
    gate_state = copy.deepcopy(gate.state_dict()) if gate is not None else None
    indices = np.arange(min(8, len(arrays["compression_tokens"])))
    target = to_device(arrays["compression_target"], indices, device).float()
    parameters = list(head.parameters()) + (list(gate.parameters()) if gate is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)
    head.train()
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        tokens = compression_input(arrays, indices, device, gate)
        pred = head(tokens)["pred_h"]
        loss = compression_loss(pred, target, model_args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    head.load_state_dict(state)
    if gate is not None and gate_state is not None:
        gate.load_state_dict(gate_state)
    if not losses[-1] < losses[0] * 0.9:
        raise RuntimeError("Compression one-batch overfit failed: %.6f -> %.6f" % (losses[0], losses[-1]))
    return {"initial": losses[0], "final": losses[-1]}


def train_compression(
    cache_root: Path,
    output_dir: Path,
    model_args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    batch_size: int,
    checkpoint_path: Path,
    use_learnable_fusion: bool = False,
    supervised_1024_projection: bool = False,
    demod: bool = False,
) -> Dict[str, Any]:
    array_names = ["compression_tokens", "compression_target", "speed_kmh"]
    if use_learnable_fusion:
        array_names.append("compression_final_tokens")
    if demod:
        array_names.append("context_h")
    train = load_arrays(cache_root / "train", array_names)
    val = load_arrays(cache_root / "val", array_names)
    test = load_arrays(cache_root / "test", array_names)
    heads = new_compression_heads(int(model_args.latent_dim), device)
    gates = {
        ratio: TaskSpecificResidualGate(initial_alpha=0.01).to(device)
        for ratio in heads
    } if use_learnable_fusion else {}
    initialization = initialize_compression_heads(
        heads,
        train,
        checkpoint_path,
        gates=gates if use_learnable_fusion else None,
        supervised_1024_projection=supervised_1024_projection,
    )
    overfit = single_batch_overfit_compression(
        heads[128], train, model_args, device, gates.get(128)
    )
    optimizers = {
        ratio: torch.optim.AdamW(
            [
                {"params": head.parameters(), "lr": 1e-4, "weight_decay": 0.01},
                *([{"params": gates[ratio].parameters(), "lr": 1e-2, "weight_decay": 0.0}]
                  if use_learnable_fusion else []),
            ]
        )
        for ratio, head in heads.items()
    }
    schedulers = {
        ratio: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-6
        )
        for ratio, optimizer in optimizers.items()
    }
    best = {ratio: -math.inf for ratio in heads}
    skipped_nonfinite = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        for head in heads.values():
            head.train()
        sums = {ratio: 0.0 for ratio in heads}
        seen = 0
        for indices in array_batches(len(train["compression_tokens"]), batch_size, True, 1700 + epoch):
            target = to_device(train["compression_target"], indices, device).float()
            if demod:
                # Align frame 8 onto frame 7 with exp(-j*omega) before training.
                omega = _compression_omega(
                    to_device(train["context_h"], indices, device).float(), model_args)
                phys = inverse_csi_transform(target, model_args)
                pc = torch.complex(phys[:, :, 0], phys[:, :, 1])
                pc = pc * torch.exp(-1j * omega[:, None, None, None] *
                                    torch.tensor([0.0, 1.0], device=device, dtype=pc.real.dtype)[None, :, None, None])
                phys = torch.stack([pc.real, pc.imag], dim=2)
                target = _forward_csi(phys, model_args).to(target.dtype)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            losses: Dict[int, torch.Tensor] = {}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                for ratio, head in heads.items():
                    tokens = compression_input(train, indices, device, gates.get(ratio))
                    output = head(tokens)
                    losses[ratio] = compression_loss(output["pred_h"], target)
                total = torch.stack(list(losses.values())).sum()
            if not bool(torch.isfinite(total)):
                skipped_nonfinite += 1
                continue
            total.backward()
            grad_norms = [
                torch.nn.utils.clip_grad_norm_(
                    list(head.parameters()) + (list(gates[ratio].parameters()) if use_learnable_fusion else []), 1.0
                )
                for ratio, head in heads.items()
            ]
            if not all(bool(torch.isfinite(norm)) for norm in grad_norms):
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                continue
            for optimizer in optimizers.values():
                optimizer.step()
            for ratio, loss in losses.items():
                sums[ratio] += float(loss.detach()) * len(indices)
            seen += len(indices)
        for scheduler in schedulers.values():
            scheduler.step()
        row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "lr": optimizers[1024].param_groups[0]["lr"],
            "skipped_nonfinite": skipped_nonfinite,
        }
        for ratio, head in heads.items():
            metrics = evaluate_compression_head(
                head, val, model_args, device, batch_size * 2, gate=gates.get(ratio),
                demod=demod,
            )
            row["train_loss_1_%d" % ratio] = sums[ratio] / seen
            row["val_sgcs_1_%d" % ratio] = metrics["sgcs_avg"]
            if use_learnable_fusion:
                row["fusion_alpha_1_%d" % ratio] = gates[ratio].alpha()
            if metrics["sgcs_avg"] > best[ratio]:
                best[ratio] = metrics["sgcs_avg"]
                torch.save(
                    {
                        "head": head.state_dict(),
                        "gate": gates[ratio].state_dict() if use_learnable_fusion else None,
                        "fusion_alpha": gates[ratio].alpha() if use_learnable_fusion else None,
                        "ratio": ratio,
                        "epoch": epoch + 1,
                        "val": metrics,
                    },
                    output_dir / ("compression_1_%d_best.pt" % ratio),
                )
        history.append(row)
        atomic_json_dump({"stage": "compression", "last": row, "history": history}, output_dir / "metrics_live.json")
        print("compression epoch=%d %s" % (epoch + 1, " ".join("1/%d=%.4f" % (r, row["val_sgcs_1_%d" % r]) for r in sorted(heads, reverse=True))), flush=True)
    results: Dict[str, Any] = {
        "single_batch_overfit": overfit,
        "fusion_protocol": (
            "per-ratio learned scalar alpha: LayerNorm(final) + alpha * "
            "(LayerNorm(pretrained multilevel fusion) - LayerNorm(final))"
            if use_learnable_fusion else "fixed pretrained multilevel fusion"
        ),
        "initialization": initialization,
        "loss_protocol": "paper: raw MSE + magnitude MSE + 0.5 * circular phase loss",
        "layout": {str(ratio): {"bottleneck_channels": width, "tokens": tokens} for ratio, (width, tokens) in RATIO_LAYOUT.items()},
        "ratios": {},
    }
    for ratio, head in heads.items():
        checkpoint = torch.load(output_dir / ("compression_1_%d_best.pt" % ratio), map_location=device, weights_only=False)
        head.load_state_dict(checkpoint["head"])
        if use_learnable_fusion:
            gates[ratio].load_state_dict(checkpoint["gate"])
            gates[ratio].eval()
        head.eval()
        results["ratios"][str(ratio)] = {
            "payload_bits": head.payload_bits,
            "measured_ratio": head.compression_ratio,
            "validation": checkpoint["val"],
            "fusion_alpha": gates[ratio].alpha() if use_learnable_fusion else None,
            "test_40": evaluate_compression_head(head, test, model_args, device, batch_size * 2, speed=40, gate=gates.get(ratio), demod=demod),
            "test_80": evaluate_compression_head(head, test, model_args, device, batch_size * 2, speed=80, gate=gates.get(ratio), demod=demod),
            "test_all": evaluate_compression_head(head, test, model_args, device, batch_size * 2, gate=gates.get(ratio), demod=demod),
        }
    results["history"] = history
    atomic_json_dump(results, output_dir / "compression_results.json")
    return results


@torch.inference_mode()
def evaluate_localization(
    head: LocalizationHead,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    pos_scale: float,
    power_mean: float = 0.0,
    power_std: float = 1.0,
    speed: int | None = None,
    feature_key: str = "localization_tokens",
) -> Dict[str, Any]:
    errors: List[np.ndarray] = []
    count = len(arrays[feature_key])
    for indices in array_batches(count, batch_size, False, 0):
        if speed is not None:
            keep = np.asarray(arrays["speed_kmh"][indices]) == speed
            indices = indices[keep]
            if not len(indices):
                continue
        tokens = to_device(arrays[feature_key], indices, device).float()
        power = None
        if head.conditioning_dim:
            power = to_device(arrays["localization_power"], indices, device).float()
            power = (power - float(power_mean)) / max(float(power_std), 1e-6)
        target = to_device(arrays["localization_target"], indices, device).float()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = head(tokens, power)
        diff = (pred.float() - target) * pos_scale
        # 2D horizontal (x,y) — comparable to the paper's planar localization metric.
        err2d = torch.linalg.vector_norm(diff, dim=1)
        # 3D euclidean (x,y,z) — z carries BS-relative height, non-trivial here.
        errors.append(err2d.cpu().numpy())
    values = np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)

    def _stats(v: np.ndarray) -> Dict[str, Any]:
        return {
            "mean_error_m": float(v.mean()) if v.size else None,
            "median_error_m": float(np.quantile(v, 0.5)) if v.size else None,
            "p90_error_m": float(np.quantile(v, 0.9)) if v.size else None,
            "p95_error_m": float(np.quantile(v, 0.95)) if v.size else None,
            "cdf": {str(q): float(np.quantile(v, q)) for q in np.linspace(0.05, 1.0, 20)} if v.size else {},
        }

    s2d = _stats(values)
    return {
        "samples": int(values.size),
        # Back-compat: top-level keys remain the 2D horizontal metric (paper-comparable).
        "mean_error_m": s2d["mean_error_m"],
        "median_error_m": s2d["median_error_m"],
        "p90_error_m": s2d["p90_error_m"],
        "p95_error_m": s2d["p95_error_m"],
        "cdf": s2d["cdf"],
        "error_2d": s2d,
    }


def single_batch_overfit_localization(
    head: LocalizationHead,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
    feature_key: str,
    power_mean: float,
    power_std: float,
) -> Dict[str, float]:
    state = copy.deepcopy(head.state_dict())
    indices = np.arange(min(16, len(arrays[feature_key])))
    tokens = to_device(arrays[feature_key], indices, device).float()
    power = to_device(arrays["localization_power"], indices, device).float()
    power = (power - float(power_mean)) / max(float(power_std), 1e-6)
    target = to_device(arrays["localization_target"], indices, device).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
    losses = []
    head.train()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(head(tokens, power), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    head.load_state_dict(state)
    if not losses[-1] < losses[0] * 0.2:
        raise RuntimeError("Localization one-batch overfit failed: %.6f -> %.6f" % (losses[0], losses[-1]))
    return {"initial": losses[0], "final": losses[-1]}


def train_localization(
    cache_root: Path,
    output_dir: Path,
    model_args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    batch_size: int,
    feature_key: str = "localization_tokens",
    artifact_prefix: str = "localization",
    protocol: str = "CSI frames 1-16 + local point cloud; all trajectory tokens masked; point_origin omitted",
) -> Dict[str, Any]:
    names = (feature_key, "localization_power", "localization_target", "speed_kmh")
    train = load_arrays(cache_root / "train", names)
    val = load_arrays(cache_root / "val", names)
    test = load_arrays(cache_root / "test", names)
    train_power = np.asarray(train["localization_power"], dtype=np.float64)
    power_mean = float(train_power.mean())
    power_std = float(train_power.std())
    head = LocalizationHead(int(model_args.latent_dim), out_dim=2, conditioning_dim=1).to(device)
    overfit = single_batch_overfit_localization(
        head, train, device, feature_key, power_mean, power_std
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    best = math.inf
    patience = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        head.train()
        loss_sum = 0.0
        seen = 0
        for indices in array_batches(len(train[feature_key]), batch_size, True, 2700 + epoch):
            tokens = to_device(train[feature_key], indices, device).float()
            power = to_device(train["localization_power"], indices, device).float()
            power = (power - power_mean) / max(power_std, 1e-6)
            target = to_device(train["localization_target"], indices, device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.mse_loss(head(tokens, power), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
        scheduler.step()
        validation = evaluate_localization(
            head, val, device, batch_size * 2, float(model_args.pos_scale),
            power_mean, power_std, feature_key=feature_key,
        )
        row = {"epoch": epoch + 1, "train_mse": loss_sum / seen, "val_mean_error_m": validation["mean_error_m"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if float(validation["mean_error_m"]) < best:
            best = float(validation["mean_error_m"])
            patience = 0
            torch.save({"head": head.state_dict(), "epoch": epoch + 1, "val": validation}, output_dir / (artifact_prefix + "_best.pt"))
        else:
            patience += 1
        atomic_json_dump({"stage": "localization", "last": row, "history": history}, output_dir / "metrics_live.json")
        print("localization epoch=%d val_mean_m=%.4f" % (epoch + 1, validation["mean_error_m"]), flush=True)
        if epoch + 1 >= 15 and patience >= 8:
            break
    checkpoint = torch.load(output_dir / (artifact_prefix + "_best.pt"), map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    result = {
        "single_batch_overfit": overfit,
        "protocol": protocol + "; direct 2D regression; standardized log10(context RMS) conditioning",
        "power_normalization": {"fit_split": "train", "mean": power_mean, "std": power_std},
        "validation": checkpoint["val"],
        "test_40": evaluate_localization(head, test, device, batch_size * 2, float(model_args.pos_scale), power_mean, power_std, speed=40, feature_key=feature_key),
        "test_80": evaluate_localization(head, test, device, batch_size * 2, float(model_args.pos_scale), power_mean, power_std, speed=80, feature_key=feature_key),
        "test_all": evaluate_localization(head, test, device, batch_size * 2, float(model_args.pos_scale), power_mean, power_std, feature_key=feature_key),
        "history": history,
    }
    atomic_json_dump(result, output_dir / (artifact_prefix + "_results.json"))
    return result


@torch.inference_mode()
def evaluate_beam(
    head: BeamPredictionHead,
    arrays: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    speed: int | None = None,
) -> Dict[str, Any]:
    correct = 0
    gain_sum = 0.0
    power_gain_sum = 0.0
    seen = 0
    count = len(arrays["beam_tokens"])
    for indices in array_batches(count, batch_size, False, 0):
        if speed is not None:
            keep = np.asarray(arrays["speed_kmh"][indices]) == speed
            indices = indices[keep]
            if not len(indices):
                continue
        tokens = to_device(arrays["beam_tokens"], indices, device).float()
        powers = to_device(arrays["beam_power"], indices, device).float()
        labels = powers.argmax(1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted = head(tokens).argmax(1)
        correct += int((predicted == labels).sum())
        selected_gain = powers.gather(1, predicted[:, None]).squeeze(1)
        optimal_gain = powers.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-20)
        # The paper's beam gain ratio (Supplementary Note 3) is an AMPLITUDE ratio,
        # R_BG = mean |H w_b| / |H w_b*|, while beam_power here stores |H w|^2. Taking
        # the ratio of powers reports the SQUARE of the paper's metric and is therefore
        # pessimistic and not comparable. Use the amplitude ratio; the power ratio is
        # kept alongside so older numbers remain traceable.
        power_ratio = (selected_gain / optimal_gain).clamp(0.0, 1.0)
        gain_sum += float(power_ratio.sqrt().sum())
        power_gain_sum += float(power_ratio.sum())
        seen += len(indices)
    return {
        "samples": seen,
        "top1": correct / max(seen, 1),
        "beam_gain_ratio": gain_sum / max(seen, 1),
        "beam_gain_ratio_power": power_gain_sum / max(seen, 1),
        "gain_ratio_definition": "amplitude |Hw_b|/|Hw_b*| per paper Supplementary Note 3",
    }


def single_batch_overfit_beam(
    head: BeamPredictionHead, arrays: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, float]:
    state = copy.deepcopy(head.state_dict())
    indices = np.arange(min(128, len(arrays["beam_tokens"])))
    tokens = to_device(arrays["beam_tokens"], indices, device).float()
    powers = to_device(arrays["beam_power"], indices, device).float()
    labels = powers.argmax(1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
    losses = []
    head.train()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(tokens), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    head.load_state_dict(state)
    if not losses[-1] < losses[0] * 0.5:
        raise RuntimeError("Beam one-batch overfit failed: %.6f -> %.6f" % (losses[0], losses[-1]))
    return {"initial": losses[0], "final": losses[-1]}


def train_beam(
    cache_root: Path,
    output_dir: Path,
    model_args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    batch_size: int,
    head_variant: str = "paper",
) -> Dict[str, Any]:
    names = ("beam_tokens", "beam_power", "speed_kmh")
    train = load_arrays(cache_root / "train", names)
    val = load_arrays(cache_root / "val", names)
    test = load_arrays(cache_root / "test", names)
    head = BeamPredictionHead(
        int(model_args.latent_dim), num_beams=32, variant=head_variant).to(device)
    overfit = single_batch_overfit_beam(head, train, device)
    train_labels = np.asarray(train["beam_power"]).argmax(1)
    counts = np.bincount(train_labels, minlength=32)
    majority = float(counts.max() / counts.sum())
    active_classes = int(np.count_nonzero(counts))
    if active_classes < 2:
        raise RuntimeError("Beam labels contain fewer than two active classes")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    best = -math.inf
    patience = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        head.train()
        loss_sum = 0.0
        seen = 0
        for indices in array_batches(len(train["beam_tokens"]), batch_size, True, 3700 + epoch):
            tokens = to_device(train["beam_tokens"], indices, device).float()
            powers = to_device(train["beam_power"], indices, device).float()
            labels = powers.argmax(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(head(tokens), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
        scheduler.step()
        validation = evaluate_beam(head, val, device, batch_size * 2)
        row = {"epoch": epoch + 1, "train_ce": loss_sum / seen, "val_top1": validation["top1"], "val_gain_ratio": validation["beam_gain_ratio"], "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if float(validation["top1"]) > best:
            best = float(validation["top1"])
            patience = 0
            torch.save({"head": head.state_dict(), "epoch": epoch + 1, "val": validation}, output_dir / "beam_best.pt")
        else:
            patience += 1
        atomic_json_dump({"stage": "beam", "last": row, "history": history}, output_dir / "metrics_live.json")
        print("beam epoch=%d val_top1=%.4f gain=%.4f" % (epoch + 1, validation["top1"], validation["beam_gain_ratio"]), flush=True)
        if epoch + 1 >= 12 and patience >= 6:
            break
    checkpoint = torch.load(output_dir / "beam_best.pt", map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    result = {
        "single_batch_overfit": overfit,
        "head_variant": head_variant,
        "head_protocol": ("1-layer attentive classifier per WWM Supplementary Note 3"
                          if head_variant == "paper" else
                          "deeper MLP readout (our improvement, deviates from paper)"),
        "label_audit": {"active_classes": active_classes, "majority_baseline_top1": majority, "class_counts": counts.tolist()},
        "strict_paper_protocol": False,
        "protocol": "3.5 GHz input and synchronized 3.5 GHz 32-beam DFT labels (same-frequency proxy)",
        "missing_for_strict_protocol": "paired 2.6 GHz input and 6.62505 GHz Type-I Single-Panel labels",
        "validation": checkpoint["val"],
        "test_40": evaluate_beam(head, test, device, batch_size * 2, speed=40),
        "test_80": evaluate_beam(head, test, device, batch_size * 2, speed=80),
        "test_all": evaluate_beam(head, test, device, batch_size * 2),
        "history": history,
    }
    atomic_json_dump(result, output_dir / "beam_results.json")
    return result


def f6(value: Any) -> str:
    try:
        return "%.6f" % float(value)
    except (TypeError, ValueError):
        return "N/A"


def write_report(output_dir: Path, payload: Dict[str, Any]) -> None:
    temporal = payload["temporal"]
    compression = payload["compression"]["ratios"]
    beam = payload["beam"]
    localization = payload["localization"]
    quality_filter = payload.get("split_audit", {}).get("quality_filter", {}).get("enabled", False)
    compression_rows = []
    for ratio in (1024, 512, 256, 128):
        item = compression[str(ratio)]
        current = item["test_all"]["sgcs_avg"]
        target = PAPER["compression_velocity_sgcs"][ratio]
        compression_rows.append(
            "| 1/%d | %d | %s | %s | %s | %+.6f |"
            % (ratio, item["payload_bits"], f6(item["test_40"]["sgcs_avg"]), f6(item["test_80"]["sgcs_avg"]), f6(current), current - target)
        )
    temporal_rows = []
    # Temporal buckets vary by test set (in-distribution 40/80; cross-scenario 5/30/60).
    # Iterate over whatever buckets were actually populated instead of hardcoding.
    temporal_speeds = sorted(int(s) for s in temporal.keys())
    for speed in temporal_speeds:
        item = temporal[str(speed)]
        temporal_rows.append(
            "| %d | %d | %s | %s | %+.6f | %s |"
            % (speed, item["samples"], " / ".join(f6(v) for v in item["sgcs_h"]), f6(item["sgcs_avg"]), item["copy_gain"], f6(item["nmse_db"]))
        )
    all_temporal_samples = sum(temporal[str(speed)]["samples"] for speed in temporal_speeds)
    all_temporal_sgcs = sum(temporal[str(speed)]["sgcs_avg"] * temporal[str(speed)]["samples"] for speed in temporal_speeds) / all_temporal_samples
    report = f"""# WWM 四个下游任务：新加坡复现评估

生成时间：{payload['generated_at']}

## 结论

- 本次使用冻结 WWM checkpoint（step {payload['checkpoint_step']}）训练轻量任务头；训练只使用新加坡 5/30/60 km/h，40/80 km/h 仅用于最终速度泛化测试。
- CSI 时序预测沿用本地 16→4 协议，40/80 km/h 合并 SGCS 为 **{all_temporal_sgcs:.6f}**。论文是 14→2，Velocity Generalization 为 {PAPER['temporal_velocity_sgcs']:.3f}，只能比较量级，不能作为同协议复现声明。
- 压缩任务执行第 4 个 tubelet（第 7–8 帧）、固定 32-token 嵌套 payload、4-bit μ-law 和论文重建损失。
- 定位主结果直接回归 2D，严格屏蔽全部轨迹 token 和 `point_origin`，仅额外保留不含坐标的 CSI context RMS；40/80 km/h 合并平均误差为 **{localization['test_all']['mean_error_m']:.6f} m**。
- 质量索引 accepted 过滤：**{quality_filter}**；实际样本数和跨 split 重名审计见 `split_audit.json`。
- 波束网络已经训练和测试，但当前结果是 **3.5 GHz 同频 DFT 代理任务**。本地没有同步 2.6/6.62505 GHz CSI，因此其数值不能与论文跨频 Type-I 结果作等价比较。

## 1. CSI 时序预测

| 速度 km/h | 样本 | h1 / h2 / h3 / h4 SGCS | 平均 SGCS | 相对 copy 增益 | NMSE dB |
|---:|---:|---|---:|---:|---:|
{chr(10).join(temporal_rows)}

论文 Velocity Generalization（14→2）SGCS：**{PAPER['temporal_velocity_sgcs']:.6f}**。

## 2. CSI 压缩与反馈

| 压缩率 | payload bit | 40 km/h SGCS | 80 km/h SGCS | 合并 SGCS | 相对论文速度泛化差值 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(compression_rows)}

## 3. 波束预测

| 测试集 | Top-1 | Beam gain ratio |
|---|---:|---:|
| 40 km/h | {f6(beam['test_40']['top1'])} | {f6(beam['test_40']['beam_gain_ratio'])} |
| 80 km/h | {f6(beam['test_80']['top1'])} | {f6(beam['test_80']['beam_gain_ratio'])} |
| 合并 | {f6(beam['test_all']['top1'])} | {f6(beam['test_all']['beam_gain_ratio'])} |

论文严格跨频 Velocity Generalization：Top-1 **{PAPER['beam_velocity_top1']:.6f}**，beam gain ratio **{PAPER['beam_velocity_gain_ratio']:.6f}**。当前代理任务的多数类 Top-1 基线为 {beam['label_audit']['majority_baseline_top1']:.6f}，活跃类别 {beam['label_audit']['active_classes']}/32。

## 4. 用户定位（2D 水平）

**2D 水平定位**（x,y；与论文 1.23 m 对标）：

| 测试集 | 平均误差 m | 中位误差 m | P90 m | P95 m |
|---|---:|---:|---:|---:|
| 40 km/h | {f6(localization['test_40']['mean_error_m'])} | {f6(localization['test_40']['median_error_m'])} | {f6(localization['test_40']['p90_error_m'])} | {f6(localization['test_40']['p95_error_m'])} |
| 80 km/h | {f6(localization['test_80']['mean_error_m'])} | {f6(localization['test_80']['median_error_m'])} | {f6(localization['test_80']['p90_error_m'])} | {f6(localization['test_80']['p95_error_m'])} |
| 合并 | {f6(localization['test_all']['mean_error_m'])} | {f6(localization['test_all']['median_error_m'])} | {f6(localization['test_all']['p90_error_m'])} | {f6(localization['test_all']['p95_error_m'])} |

论文 Velocity Generalization 平均 2D 定位误差：**{PAPER['localization_velocity_mean_m']:.6f} m**。

## 性能与协议审计

1. 波束严格复现的缺口是数据，不是分类头：需要同轨迹同步生成 2.6 GHz 输入和 6.62505 GHz 标签，并使用论文 Type-I Single-Panel codebook。
2. 当前源数据中心频率为 3.5 GHz；论文预训练及波束输入口径为 2.6 GHz。该差异会影响所有绝对性能对照。
3. 正式定位头不接收 `point_origin`；保留原点的 `localization-origin` 仅是泄漏诊断，永不进入四任务正式报告。
4. 本地时序任务为 16→4，而论文表 5 为 14→2；预测跨度更长，四帧平均与论文两帧平均不完全可比。
5. 预训练 predictor 的 masked-future probe 明显低于 encoder probe，时序任务的上限仍受预训练细粒度表征退化影响；本次任务头训练无法修复主干表征。
6. 质量筛选与预训练保持一致；验证集按文件固定划分，40/80 km/h 未参与模型选择，测试集仍是已反复查看过的开发测试集，最终论文结论需要新的封存测试集。
"""
    (output_dir / "WWM四个下游任务新加坡完整评估.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "cache", "train", "compression", "localization", "beam", "localization-origin", "report"),
        default="all",
    )
    parser.add_argument("--test-root", default=None,
                        help="Optional separate root for the test split (e.g. 测试集/). Defaults to --dataset-root.")
    parser.add_argument("--test-split", default="test_gen_velocity",
                        help="Test split directory name under --test-root (flat H/pos/meta supported).")
    parser.add_argument("--test-city", default="singapore_cbd",
                        help="City key filter for the test split. Use 'any' to keep all cities.")
    parser.add_argument("--train-city", default="singapore_cbd",
                        help="City key filter for the train/val split. For cross-scenario eval, keep "
                             "this as the training city (e.g. singapore_cbd) while --test-city differs.")
    parser.add_argument("--stats-path", default=None,
                        help="Override path to csi_stats_per_city.json (e.g. an augmented file that adds "
                             "the test city's per-city stats). Auto-detects training_ready_v3 then v2.")
    parser.add_argument("--quality-index", default=None,
                        help="Training quality_index.npz; auto-detected under training_ready_v3/v2.")
    parser.add_argument("--test-quality-index", default=None,
                        help="Test quality_index.npz; auto-detected from --test-root.")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable accepted-sample filtering for an explicit unfiltered diagnostic.")
    parser.add_argument("--audit-content-hashes", action="store_true",
                        help="Hash every H/pos file for cross-split content-overlap auditing (slow).")
    parser.add_argument("--temporal-decoder", choices=("pretrain_head", "residual_decoder"),
                        default="pretrain_head",
                        help="Which module decodes predicted CSI tokens for the temporal forecast. "
                             "'pretrain_head' uses pretrain_csi_head, i.e. the head wwm_pretrain "
                             "actually trains. 'residual_decoder' is the legacy model.decoder path "
                             "whose to_patch projection is zero-init and never trained, making the "
                             "forecast identical to the copy baseline for every checkpoint; keep it "
                             "only to reproduce historical numbers.")
    parser.add_argument("--feature-batch-size", type=int, default=12)
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--compression-epochs", type=int, default=25)
    parser.add_argument("--localization-epochs", type=int, default=50)
    parser.add_argument("--beam-epochs", type=int, default=30)
    parser.add_argument(
        "--beam-head-variant", choices=("paper", "mlp"), default="paper",
        help="'paper' = 1-layer attentive classifier exactly as WWM Supplementary "
             "Note 3 (use for any paper comparison); 'mlp' = our deeper readout, "
             "higher accuracy but a protocol deviation.")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-levels", action="store_true",
                        help="Save multi-level encoder features for learnable fusion experiments")
    parser.add_argument("--use-learnable-fusion", action="store_true",
                        help="Use learnable task-specific fusion for compression (requires --save-levels cache)")
    parser.add_argument("--supervised-1024-projection", action="store_true",
                        help="Initialize only the 1/1024 projection from token-to-CSI-patch cross-covariance")
    parser.add_argument("--fusion-mode", choices=("static", "dynamic"), default="static",
                        help="Fusion mode: 'static' for fixed learnable weights, 'dynamic' for content-based")
    cli = parser.parse_args()

    seed_everything(cli.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checkpoint = Path(cli.checkpoint).resolve()
    dataset_root = Path(cli.dataset_root).resolve()
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "cache"

    model_args: argparse.Namespace
    checkpoint_step: int
    test_root = Path(cli.test_root).resolve() if cli.test_root else dataset_root
    if cli.stage in ("all", "cache", "localization", "localization-origin"):
        model, model_args, checkpoint_step = load_backbone(checkpoint, device)
        # Replicate training-time CSI preprocessing (per-city norm + log_context_rms trajectory
        # feature) so the frozen encoder sees in-distribution inputs.
        stats_path = resolve_training_sidecar(dataset_root, cli.stats_path, "csi_stats_global.json")
        if stats_path is None:
            stats_path = resolve_training_sidecar(dataset_root, cli.stats_path, "csi_stats_per_city.json")
        model_args._per_city_stats = load_per_city_csi_stats(stats_path) if stats_path is not None else None
        if getattr(model_args, "csi_normalization_scope", "global") == "per_city" and not model_args._per_city_stats:
            raise FileNotFoundError("Per-city normalization requires stats at %s" % stats_path)
        quality_enabled = not cli.no_quality_filter
        train_quality_path = resolve_training_sidecar(dataset_root, cli.quality_index, "quality_index.npz")
        test_quality_path = resolve_training_sidecar(test_root, cli.test_quality_index, "quality_index.npz")
        if quality_enabled and (train_quality_path is None or test_quality_path is None):
            raise FileNotFoundError(
                "Quality filtering is the default protocol, but a train/test quality_index.npz is missing. "
                "Provide --quality-index/--test-quality-index or use --no-quality-filter for a diagnostic."
            )
        train_quality = load_quality_index(train_quality_path) if quality_enabled and train_quality_path else None
        test_quality = load_quality_index(test_quality_path) if quality_enabled and test_quality_path else None
        train_all = singapore_scenarios(dataset_root, "train", cli.train_city)
        train_scenarios, val_scenarios = fixed_file_split(train_all, cli.validation_fraction)
        test_scenarios = singapore_scenarios(test_root, cli.test_split, cli.test_city)
        scenario_audit = {
            "train_files": len(train_scenarios),
            "validation_files": len(val_scenarios),
            "test_files": len(test_scenarios),
            "train_speeds": sorted({speed_from_base(item.base) for item in train_scenarios}),
            "validation_speeds": sorted({speed_from_base(item.base) for item in val_scenarios}),
            "test_speeds": sorted({speed_from_base(item.base) for item in test_scenarios}),
            "test_root": str(test_root),
            "test_split": cli.test_split,
            "test_city": cli.test_city,
            "train_city": cli.train_city,
            "quality_filter": {
                "enabled": quality_enabled,
                "train_index": str(train_quality_path) if train_quality_path else None,
                "test_index": str(test_quality_path) if test_quality_path else None,
            },
        }
        base_sets = {
            "train": {item.base for item in train_scenarios},
            "validation": {item.base for item in val_scenarios},
            "test": {item.base for item in test_scenarios},
        }
        scenario_audit["base_name_overlap"] = {
            "train_validation": len(base_sets["train"] & base_sets["validation"]),
            "train_test": len(base_sets["train"] & base_sets["test"]),
            "validation_test": len(base_sets["validation"] & base_sets["test"]),
        }
        if any(scenario_audit["base_name_overlap"].values()):
            raise RuntimeError("Cross-split scenario base-name leakage: %s" % scenario_audit["base_name_overlap"])
        if cli.audit_content_hashes:
            hash_sets = {
                split: {
                    "h": scenario_content_hashes(items, "h_path"),
                    "pos": scenario_content_hashes(items, "pos_path"),
                }
                for split, items in (
                    ("train", train_scenarios), ("validation", val_scenarios), ("test", test_scenarios)
                )
            }
            scenario_audit["content_hash_overlap"] = {
                "%s_%s_%s" % (left, right, kind): len(hash_sets[left][kind] & hash_sets[right][kind])
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
                for kind in ("h", "pos")
            }
        # Resolve the point-cloud city_root for the test split. Some external test folders
        # (e.g. Frankfurt) ship their own <city>/point_clouds/ inside test_root/test_split;
        # others (e.g. Singapore) reuse the curated dataset_root/cities. Auto-detect the former.
        test_city_root: Optional[Path] = None
        nested_root = test_root / cli.test_split
        if (nested_root / cli.test_city / "point_clouds" / "point_cloud.npy").exists():
            test_city_root = nested_root
            print("test city_root (nested): %s" % test_city_root, flush=True)
        datasets: Dict[str, ProtocolDataset] = {}
        for name, scenarios in (("train", train_scenarios), ("val", val_scenarios), ("test", test_scenarios)):
            # Scenarios carry absolute H/pos/meta paths from discovery (test set lives under
            # test_root). city_root supplies point clouds: curated cities/ for train/val, and
            # the auto-detected test_city_root for the test split when it ships its own clouds.
            cr = test_city_root if (name == "test" and test_city_root is not None) else None
            lookup = test_quality if name == "test" else train_quality
            datasets[name] = make_protocol_dataset(
                scenarios, dataset_root, model_args, city_root=cr,
                quality_lookup=lookup, filter_quality=quality_enabled,
            )
        scenario_audit["samples"] = {
            "train": len(datasets["train"]),
            "validation": len(datasets["val"]),
            "test": len(datasets["test"]),
        }
        atomic_json_dump(scenario_audit, output_dir / "split_audit.json")
        for name, dataset in datasets.items():
            if cli.stage == "localization-origin":
                extract_split_cache(name, dataset, model, model_args, checkpoint, output_dir, device,
                                    cli.feature_batch_size, temporal_decoder=cli.temporal_decoder,
                                    save_levels=cli.save_levels)
                extract_localization_origin_cache(
                    name, dataset, model, checkpoint, output_dir, device, cli.feature_batch_size
                )
            else:
                extract_split_cache(name, dataset, model, model_args, checkpoint, output_dir, device,
                                    cli.feature_batch_size, temporal_decoder=cli.temporal_decoder,
                                    save_levels=cli.save_levels)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        atomic_json_dump({"args": vars(model_args), "checkpoint_step": checkpoint_step}, output_dir / "backbone_config.json")
        if cli.stage == "cache":
            return
    else:
        config = json_read(output_dir / "backbone_config.json")
        if not config:
            raise FileNotFoundError("Missing backbone_config.json; run --stage cache first")
        model_args = argparse.Namespace(**config["args"])
        checkpoint_step = int(config["checkpoint_step"])
        for split in ("train", "val", "test"):
            split_dir = cache_root / split
            meta = json_read(split_dir / "meta.json")
            count = int(meta.get("samples", -1))
            if count < 0 or not cache_complete(
                split_dir, checkpoint, count, int(model_args.latent_dim), save_levels=cli.save_levels
            ):
                raise RuntimeError(
                    "Cache protocol is stale or incomplete for %s; rebuild with --stage cache" % split
                )

    if cli.stage == "localization":
        # Localization-only: train the no-origin head and exit (skip compression/beam/report).
        result = train_localization(cache_root, output_dir, model_args, device, cli.localization_epochs, cli.head_batch_size)
        ta = result.get("test_all", {})
        print(
            "localization-only done: 2D_mean=%.4f m (results=%s)"
            % (ta.get("mean_error_m") or -1.0, output_dir / "localization_results.json"),
            flush=True,
        )
        return
    if cli.stage in ("all", "train"):
        compression = train_compression(cache_root, output_dir, model_args, device, cli.compression_epochs, cli.head_batch_size, checkpoint, use_learnable_fusion=cli.use_learnable_fusion, supervised_1024_projection=cli.supervised_1024_projection)
        localization = train_localization(cache_root, output_dir, model_args, device, cli.localization_epochs, cli.head_batch_size)
        beam = train_beam(cache_root, output_dir, model_args, device, cli.beam_epochs,
                          cli.head_batch_size, head_variant=cli.beam_head_variant)
    elif cli.stage == "compression":
        compression = train_compression(cache_root, output_dir, model_args, device, cli.compression_epochs, cli.head_batch_size, checkpoint, use_learnable_fusion=cli.use_learnable_fusion, supervised_1024_projection=cli.supervised_1024_projection)
        localization = json_read(output_dir / "localization_results.json")
        beam = json_read(output_dir / "beam_results.json")
        if not localization or not beam:
            print(
                "compression-only done: results=%s" % (output_dir / "compression_results.json"),
                flush=True,
            )
            return
    elif cli.stage == "beam":
        compression = json_read(output_dir / "compression_results.json")
        localization = json_read(output_dir / "localization_results.json")
        beam = train_beam(cache_root, output_dir, model_args, device, cli.beam_epochs,
                          cli.head_batch_size, head_variant=cli.beam_head_variant)
    elif cli.stage == "localization-origin":
        localization = train_localization(
            cache_root,
            output_dir,
            model_args,
            device,
            cli.localization_epochs,
            cli.head_batch_size,
            feature_key="localization_origin_tokens",
            artifact_prefix="localization_origin",
            protocol=(
                "CSI frames 1-16 + absolute point-cloud centers; all trajectory tokens masked; "
                "point_origin/point_scale retained to match pretraining geometry semantics"
            ),
        )
        ta = localization.get("test_all", {})
        print(
            "localization-origin diagnostic done: 2D_mean=%.4f m (invalid as a formal result; results=%s)"
            % (ta.get("mean_error_m") or -1.0, output_dir / "localization_origin_results.json"),
            flush=True,
        )
        return
    else:
        compression = json_read(output_dir / "compression_results.json")
        localization = json_read(output_dir / "localization_results.json")
        beam = json_read(output_dir / "beam_results.json")
    temporal = json_read(cache_root / "test" / "meta.json").get("temporal", {})
    if not temporal or not compression or not localization or not beam:
        raise RuntimeError("One or more downstream results are missing")
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "dataset_root": str(dataset_root),
        "split_audit": json_read(output_dir / "split_audit.json"),
        "paper_targets": PAPER,
        "temporal": temporal,
        "compression": compression,
        "localization": localization,
        "beam": beam,
        "strict_four_task_reproduction": False,
        "protocol_notes": {
            "temporal": "local 16->4 backbone decoder diagnostic; use wwm_paper_strict_repro.py for local 16->4",
            "compression": "paper loss and nested payload layout",
            "localization": "formal 2D result excludes trajectory and point_origin coordinates",
            "beam": "paper head architecture, but same-frequency proxy labels",
        },
        "strict_blocker": "Temporal remains 16->4 here; paired 2.6/6.62505 GHz beam data and exact Type-I labels are missing",
    }
    atomic_json_dump(payload, output_dir / "four_task_final_results.json")
    write_report(output_dir, payload)
    atomic_json_dump({"stage": "complete", "results": str(output_dir / "four_task_final_results.json")}, output_dir / "metrics_live.json")
    print("complete report=%s" % (output_dir / "WWM四个下游任务新加坡完整评估.md"), flush=True)


if __name__ == "__main__":
    main()
