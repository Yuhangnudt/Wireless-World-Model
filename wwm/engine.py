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

if os.name != "nt":
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

from .common import *
from .metrics import *
from .pointbert import *
from .data import *
from .model import *
from .masking import *
from .losses import *


try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def unwrap_model(model: nn.Module) -> PaperWWM:
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_city_root(dataset_root: Path, configured: Optional[str]) -> Path:
    if configured:
        return Path(configured).resolve()
    nested = dataset_root / "cities"
    if nested.exists():
        return nested.resolve()
    return (dataset_root.parent / "cities").resolve()


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def print_model_summary(model: PaperWWM, args: argparse.Namespace, trainable_params: int) -> None:
    csi_tokens, point_tokens, traj_tokens = model.online_stem.lengths()
    total_tokens = csi_tokens + point_tokens + traj_tokens
    total_params = count_parameters(model, trainable_only=False)
    frozen_params = total_params - int(trainable_params)
    print(
        "model_summary total_params=%d trainable_params=%d frozen_params=%d latent_dim=%d total_tokens=%d "
        "tokens(csi/point/traj)=%d/%d/%d"
        % (
            total_params,
            int(trainable_params),
            frozen_params,
            int(args.latent_dim),
            total_tokens,
            csi_tokens,
            point_tokens,
            traj_tokens,
        )
    )
    print(
        "model_blocks encoder_layers=%d predictor_layers=%d decoder_layers=%d heads=%d ffn_mult=%d "
        "patch(t/h/w)=%d/%d/%d context=%d future=%d"
        % (
            int(args.mmoe_layers),
            int(args.predictor_layers),
            int(args.decoder_layers),
            int(args.mmoe_heads),
            int(args.ffn_mult),
            int(args.patch_t),
            int(args.patch_h),
            int(args.patch_w),
            int(args.context_steps),
            int(args.future_steps),
        )
    )
    print(
        "model_modes point_tokenizer=%s point_tokens=%d point_group_size=%d point_source=%s "
        "encoder_visible=%s decoder_input=%s temporal_anchor=%s"
        % (
            str(args.point_tokenizer),
            int(args.point_tokens),
            int(args.point_group_size),
            str(args.point_dvae_token_source),
            str(args.encoder_visible_mode),
            str(args.decoder_token_input),
            str(args.temporal_anchor),
        )
    )
    if float(getattr(model, "deep_supervision_weight", 0.0)) > 0:
        encoder_indices = model.online_encoder.intermediate_indices(
            len(model.online_encoder.blocks),
            int(model.deep_supervision_layers),
            str(model.deep_layer_selection),
        )
        predictor_indices = model.predictor.intermediate_indices(
            len(model.predictor.blocks),
            int(model.deep_supervision_layers),
            str(model.deep_layer_selection),
        )
        print(
            "model_multilevel enabled=1 selection=%s encoder_blocks_1based=%s predictor_blocks_1based=%s "
            "fusion=%s levels=%d deep_weight=%.6g context_source=%s context_weight=%.6g"
            % (
                str(model.deep_layer_selection),
                ",".join(str(index + 1) for index in encoder_indices),
                ",".join(str(index + 1) for index in predictor_indices),
                str(model.deep_context_fusion),
                int(model.deep_supervision_layers),
                float(model.deep_supervision_weight),
                str(model.context_loss_source),
                float(model.context_loss_weight),
            )
        )
    print(
        "model_collapse_guard jepa_mode=%s ema_target=%d sigreg_enabled=%d visreg_enabled=%d "
        "sigreg_weights(csi/point/traj)=%.6g/%.6g/%.6g"
        % (
            str(model.jepa_mode),
            int(model.has_ema_target),
            int(model.sigreg_enable),
            int(model.visreg_enable),
            float(model.sigreg_weight_csi),
            float(model.sigreg_weight_point),
            float(model.sigreg_weight_traj),
        )
    )


def parse_gpu_ids(value: str) -> List[int]:
    if not value:
        return list(range(torch.cuda.device_count()))
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("--gpu-ids did not contain any valid CUDA device ids.")
    available = torch.cuda.device_count()
    invalid = [idx for idx in ids if idx < 0 or idx >= available]
    if invalid:
        raise ValueError("Invalid CUDA device ids %s; only %d device(s) are visible." % (invalid, available))
    return ids


def get_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    start_lr: float,
    peak_lr: float,
    final_lr: float,
    schedule: str = "cosine",
) -> float:
    if schedule == "constant":
        return peak_lr
    if schedule != "cosine":
        raise ValueError("Unknown lr schedule: %s" % schedule)
    if total_steps <= 1:
        return peak_lr
    if warmup_steps > 0 and step < warmup_steps:
        p = float(step + 1) / float(warmup_steps)
        return start_lr + p * (peak_lr - start_lr)
    denom = max(total_steps - warmup_steps, 1)
    p = min(max((step - warmup_steps) / denom, 0.0), 1.0)
    return final_lr + 0.5 * (peak_lr - final_lr) * (1.0 + math.cos(math.pi * p))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def resume_epoch_position(step: int, steps_per_epoch: int, grad_accum: int) -> Tuple[int, int]:
    """Map a completed optimizer-step count to the next logical batch position."""
    if steps_per_epoch <= 0 or grad_accum <= 0:
        raise ValueError("steps_per_epoch and grad_accum must be positive")
    completed_updates = max(int(step), 0)
    epoch = completed_updates // int(steps_per_epoch)
    updates_in_epoch = completed_updates % int(steps_per_epoch)
    return epoch, updates_in_epoch * int(grad_accum)


def to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def first_nonfinite_module_state(module: nn.Module) -> Optional[str]:
    for name, tensor in module.named_parameters():
        if tensor is not None and not bool(torch.isfinite(tensor.detach()).all().item()):
            return "parameter:%s" % name
    for name, tensor in module.named_buffers():
        if tensor is not None and torch.is_tensor(tensor) and not bool(torch.isfinite(tensor.detach()).all().item()):
            return "buffer:%s" % name
    return None


def set_batchnorm_eval(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def move_point_input(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_point_input(subvalue, device) for key, subvalue in value.items()}
    raise TypeError("Unsupported point input type: %s" % type(value).__name__)


def extract_point_input(batch: Any, origin_dropout: float = 0.0) -> Any:
    """Assemble the point-modality input.

    origin_dropout > 0 zeroes point_origin for a random subset of samples and returns
    a boolean `origin_dropped` flag alongside. Rationale: point_origin equals
    traj[:, context_steps-1, :3] -- the localization label -- and the point embedder
    adds it to every point token, so leaving it always present lets the model read the
    UE position off the geometry pathway and never learn to infer it from CSI.
    """
    if not isinstance(batch, dict):
        return batch
    metadata = {
        key: batch[key]
        for key in ("point_origin", "point_scale")
        if key in batch
    }
    if origin_dropout > 0.0 and "point_origin" in metadata:
        origin = metadata["point_origin"]
        keep = torch.rand(origin.shape[0], device=origin.device) >= float(origin_dropout)
        metadata["point_origin"] = origin * keep.reshape(-1, *([1] * (origin.dim() - 1))).to(origin.dtype)
        metadata["origin_dropped"] = ~keep
    if "point_group" in batch:
        return {"point_group": batch["point_group"], **metadata} if metadata else batch["point_group"]
    if "point_cloud" in batch:
        return {"points": batch["point_cloud"], **metadata} if metadata else batch["point_cloud"]
    if "neighborhood" in batch and "center" in batch:
        return {"neighborhood": batch["neighborhood"], "center": batch["center"], **metadata}
    raise KeyError("Batch does not contain point_group, point_cloud, or neighborhood/center.")


def make_loader(
    scenarios: List[ScenarioFile],
    city_root: Path,
    args: argparse.Namespace,
    shuffle: bool,
    drop_last: bool,
) -> Tuple[WWMDataset, DataLoader]:
    dataset = WWMDataset(
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
    )
    if bool(args.point_group_cache):
        cache_dir = point_group_cache_dir(args, dataset)
        if not point_group_cache_ready(cache_dir, expected_point_group_cache_meta(args, dataset)):
            if bool(args.point_group_cache_build):
                build_point_group_cache(dataset, args, cache_dir)
            else:
                raise FileNotFoundError("Point group cache missing; rerun with --point-group-cache-build: %s" % cache_dir)
        print("point_group_cache=using path=%s include_point_cloud=False" % cache_dir)
        dataset = PointGroupCachedDataset(dataset, cache_dir, args, include_point_cloud=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last and len(dataset) >= args.batch_size,
        persistent_workers=bool(args.num_workers > 0 and args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor) if args.num_workers > 0 else None,
    )
    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty; reduce batch size or increase samples.")
    return dataset, loader


def make_point_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool, drop_last: bool) -> DataLoader:
    if bool(args.point_group_cache):
        cache_dir = point_group_cache_dir(args, dataset)
        if not point_group_cache_ready(cache_dir, expected_point_group_cache_meta(args, dataset)):
            if bool(args.point_group_cache_build):
                build_point_group_cache(dataset, args, cache_dir)
            else:
                raise FileNotFoundError("Point group cache missing; rerun with --point-group-cache-build: %s" % cache_dir)
        print("point_group_cache=using path=%s include_point_cloud=False" % cache_dir)
        dataset = PointGroupCachedDataset(dataset, cache_dir, args, include_point_cloud=False)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=pointbert_worker_init_fn,
        persistent_workers=bool(args.num_workers > 0 and args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor) if args.num_workers > 0 else None,
    )


def schedule_linear(step: int, start: float, target: float, ntime: int) -> float:
    if ntime <= 0:
        return float(target)
    ratio = min(max(float(step) / float(ntime), 0.0), 1.0)
    return float(start) + ratio * (float(target) - float(start))


def pointbert_cosine_value(step: int, start: float, target: float, ntime: int) -> float:
    if ntime <= 0 or step > ntime:
        return float(target)
    return float(target) + (float(start) - float(target)) * (1.0 + math.cos(math.pi * float(step) / float(ntime))) / 2.0


def pointbert_kld_weight(step: int, start: float, target: float, ntime: int, delay: int) -> float:
    shifted = int(step) - int(delay)
    if shifted < 0:
        return 0.0
    return pointbert_cosine_value(shifted, start, target, ntime)


def pointbert_add_weight_decay(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": float(weight_decay)}]


def get_pointbert_epoch_lr(
    epoch: int,
    max_epoch: int,
    warmup_epochs: int,
    start_lr: float,
    peak_lr: float,
    final_lr: float = 1e-6,
) -> float:
    """LR used at the start of a Point-BERT epoch.

    The official runner calls timm's scheduler at the end of each epoch, so epoch
    e trains with the value produced by scheduler.step(e-1), except epoch 0 which
    starts from warmup_lr_init.
    """
    if epoch <= 0:
        return float(start_lr)
    t = int(epoch) - 1
    if warmup_epochs > 0 and t < warmup_epochs:
        return float(start_lr) + (float(peak_lr) - float(start_lr)) * float(t) / float(warmup_epochs)
    t_initial = max(int(max_epoch), 1)
    if t >= t_initial:
        return float(final_lr)
    return float(final_lr) + 0.5 * (float(peak_lr) - float(final_lr)) * (
        1.0 + math.cos(math.pi * float(t) / float(t_initial))
    )


def save_point_dvae_state(
    output_dir: Path,
    name: str,
    dvae: PointBERTDiscreteVAE,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: List[Dict[str, Any]],
    step: int,
    epoch: int,
    required: bool = False,
) -> bool:
    payload = {
        "model": dvae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(args),
        "history": history,
        "step": step,
        "epoch": epoch,
    }
    checkpoint_path = output_dir / name
    try:
        atomic_torch_save(payload, checkpoint_path)
    except CheckpointSaveError as exc:
        if required:
            raise
        print("checkpoint_save_skipped path=%s reason=%s" % (checkpoint_path, exc))
        return False
    live = {"step": step, "epoch": epoch, "last": history[-1] if history else {}}
    atomic_json_dump(live, output_dir / "metrics_live.json")
    return True


def checkpoint_model_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("model"), dict):
            return ckpt["model"]
        if isinstance(ckpt.get("base_model"), dict):
            return ckpt["base_model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint payload type: %s" % type(ckpt).__name__)


def train_point_dvae(args: argparse.Namespace) -> Dict[str, Any]:
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.set_float32_matmul_precision("high")

    dataset_root = Path(args.dataset_root).resolve()
    city_root = resolve_city_root(dataset_root, args.city_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios: List[ScenarioFile] = []
    if args.point_dvae_data == "shapenet55":
        dataset = PointBERTShapeNet55Dataset(
            root=Path(args.shapenet55_root).resolve(),
            subset=args.shapenet55_subset,
            npoints=args.shapenet55_npoints,
            whole=args.shapenet55_whole,
        )
        loader = make_point_loader(dataset, args, shuffle=True, drop_last=True)
    elif args.point_dvae_data == "wwm":
        scenarios = discover_scenarios(
            dataset_root,
            args.split,
            optional_limit(args.max_files),
            optional_limit(args.max_samples_per_file),
        )
        args.csi_mean = 0.0 if args.csi_mean is None else args.csi_mean
        args.csi_std = 1.0 if args.csi_std is None else args.csi_std
        dataset = WWMPointCloudDataset(
            scenarios=scenarios,
            city_root=city_root,
            context_steps=args.context_steps,
            point_count=args.point_count,
            point_pool_count=args.point_pool_count,
            point_pool_mode=args.point_pool_mode,
            point_normalization=args.point_normalization,
            point_scale=args.point_scale,
            drop_zero_csi_samples=args.drop_zero_csi_samples,
            seed=args.seed,
        )
        loader = make_point_loader(dataset, args, shuffle=True, drop_last=True)
    else:
        raise ValueError("Unknown point dVAE data source: %s" % args.point_dvae_data)
    if len(loader) == 0:
        raise RuntimeError("Point dVAE DataLoader is empty.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_dvae = PointBERTDiscreteVAE(
        point_tokens=args.point_tokens,
        group_size=args.point_group_size,
        center_sampling=args.point_center_sampling,
        encoder_dims=args.point_dvae_encoder_dims,
        codebook_size=args.point_dvae_codebook_size,
        codebook_dim=args.point_dvae_codebook_dim,
        decoder_dims=args.point_dvae_decoder_dims,
    ).to(device)
    raw_dvae.logit_clip = float(args.point_dvae_logit_clip)
    point_dvae_ckpt: Optional[Dict[str, Any]] = None
    if args.point_dvae_resume:
        ckpt = torch.load(args.point_dvae_resume, map_location="cpu")
        point_dvae_ckpt = ckpt if isinstance(ckpt, dict) else None
        raw_state = checkpoint_model_state(ckpt)
        state = normalize_state_dict_keys(
            raw_state,
            prefixes=(
                "online_stem.point.dvae.",
                "module.online_stem.point.dvae.",
                "point.dvae.",
                "module.point.dvae.",
                "dvae.",
                "module.dvae.",
            ),
        )
        incompatible = raw_dvae.load_state_dict(state, strict=args.point_dvae_strict_resume)
        print(
            "point_dvae_resumed=%s missing=%d unexpected=%d"
            % (args.point_dvae_resume, len(incompatible.missing_keys), len(incompatible.unexpected_keys))
        )

    optimizer_params = pointbert_add_weight_decay(raw_dvae, args.point_dvae_weight_decay) if args.pointbert_official_optim else raw_dvae.parameters()
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.point_dvae_lr,
        betas=(0.9, 0.999) if args.pointbert_official_optim else (0.9, 0.95),
        eps=1e-8 if args.pointbert_official_optim else 1e-10,
        weight_decay=args.point_dvae_weight_decay if not args.pointbert_official_optim else 0.0,
        foreach=args.adamw_foreach,
    )
    history: List[Dict[str, Any]] = []
    step = 0
    start_epoch = 0
    if point_dvae_ckpt is not None:
        if not args.reset_optimizer and isinstance(point_dvae_ckpt.get("optimizer"), dict):
            try:
                optimizer.load_state_dict(point_dvae_ckpt["optimizer"])
                print("point_dvae_optimizer_state=loaded")
            except Exception as exc:
                print("point_dvae_optimizer_state=reset reason=%s" % exc)
        elif args.reset_optimizer:
            print("point_dvae_optimizer_state=reset")
        if args.reset_training_state:
            print("point_dvae_training_state=reset")
        else:
            history = list(point_dvae_ckpt.get("history", []))
            step = int(point_dvae_ckpt.get("step", 0))
            start_epoch = int(point_dvae_ckpt.get("epoch", 0))
            print("point_dvae_training_state=loaded step=%d epoch=%d history=%d" % (step, start_epoch, len(history)))

    adaptive_lr_scale = 1.0
    adaptive_lr_best: Optional[float] = None
    adaptive_lr_bad_windows = 0
    adaptive_lr_last_check_step = -1
    if args.point_dvae_adaptive_lr:
        for item in reversed(history):
            if "adaptive_lr_scale" in item:
                adaptive_lr_scale = float(item.get("adaptive_lr_scale", 1.0))
                adaptive_lr_best_value = item.get("adaptive_lr_best")
                adaptive_lr_best = None if adaptive_lr_best_value is None else float(adaptive_lr_best_value)
                adaptive_lr_bad_windows = int(item.get("adaptive_lr_bad_windows", 0))
                adaptive_lr_last_check_step = int(item.get("adaptive_lr_last_check_step", -1))
                break
        print(
            "point_dvae_adaptive_lr=on monitor=%s window=%d patience=%d factor=%.3f min_lr=%.2e scale=%.4f"
            % (
                args.point_dvae_adaptive_lr_monitor,
                args.point_dvae_adaptive_lr_window_steps,
                args.point_dvae_adaptive_lr_patience,
                args.point_dvae_adaptive_lr_factor,
                args.point_dvae_adaptive_min_lr,
                adaptive_lr_scale,
            )
        )

    gpu_ids: List[int] = []
    if device.type == "cuda" and args.multi_gpu:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if len(gpu_ids) > 1:
            raw_dvae = raw_dvae.to(torch.device("cuda:%d" % gpu_ids[0]))
            device = torch.device("cuda:%d" % gpu_ids[0])
            model: nn.Module = nn.DataParallel(raw_dvae, device_ids=gpu_ids, output_device=gpu_ids[0])
            print("multi_gpu=DataParallel device_ids=%s" % ",".join(str(i) for i in gpu_ids))
        else:
            model = raw_dvae
    else:
        model = raw_dvae

    grad_accum = int(args.grad_accum_steps)
    if grad_accum <= 0:
        grad_accum = max(1, int(math.ceil(float(args.global_batch_size) / float(args.batch_size))))
    steps_per_epoch = int(math.ceil(len(loader) / float(grad_accum)))
    max_steps = int(args.limit_steps) if args.limit_steps and args.limit_steps > 0 else int(args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch)) if args.warmup_epochs > 0 else 0
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    start = time.time()
    run_start_step = int(step)
    model.train()
    if args.point_dvae_freeze_bn_stats:
        set_batchnorm_eval(raw_dvae)
        print("point_dvae_freeze_bn_stats=on")
    optimizer.zero_grad(set_to_none=True)
    skipped_nonfinite = 0
    print(
        "stage=point_dvae data=%s device=%s samples=%d steps=%d groups=%d group_size=%d codebook=%d dim=%d"
        % (
            args.point_dvae_data,
            device,
            len(dataset),
            max_steps,
            args.point_tokens,
            args.point_group_size,
            args.point_dvae_codebook_size,
            args.point_dvae_codebook_dim,
        )
    )

    stop = False
    for epoch in range(start_epoch, args.epochs):
        if args.point_dvae_freeze_bn_stats:
            set_batchnorm_eval(raw_dvae)
        accum = 0
        accum_terms: Dict[str, float] = {}
        for batch in loader:
            point_cloud = move_point_input(extract_point_input(batch), device)
            if args.pointbert_official_schedule:
                temperature = pointbert_cosine_value(step, args.point_dvae_temp_start, args.point_dvae_temp_target, args.point_dvae_temp_steps)
                kld_weight = pointbert_kld_weight(
                    step,
                    args.point_dvae_kld_start,
                    args.point_dvae_kld_target,
                    args.point_dvae_kld_steps,
                    args.point_dvae_kld_delay,
                )
            else:
                temperature = schedule_linear(step, args.point_dvae_temp_start, args.point_dvae_temp_target, args.point_dvae_temp_steps)
                kld_weight = schedule_linear(step, args.point_dvae_kld_start, args.point_dvae_kld_target, args.point_dvae_kld_steps)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(point_cloud, temperature=temperature, hard=args.point_dvae_hard)
                base_dvae = model.module if isinstance(model, nn.DataParallel) else model
                recon_loss, kl_loss = base_dvae.get_loss(out)
                loss = recon_loss + float(kld_weight) * kl_loss
                scaled = loss / grad_accum
            if not torch.isfinite(loss):
                bad_state = first_nonfinite_module_state(raw_dvae)
                recon_value = to_float(recon_loss) if torch.is_tensor(recon_loss) else float(recon_loss)
                kl_value = to_float(kl_loss) if torch.is_tensor(kl_loss) else float(kl_loss)
                skipped_nonfinite += 1
                print(
                    "point_dvae_nonfinite step=%05d epoch=%d action=%s state=%s loss=%s recon=%s kl=%s temp=%.6f kld_w=%.6f skipped=%d"
                    % (
                        step,
                        epoch,
                        args.point_dvae_nonfinite_action,
                        str(bad_state),
                        str(to_float(loss) if torch.is_tensor(loss) else loss),
                        str(recon_value),
                        str(kl_value),
                        float(temperature),
                        float(kld_weight),
                        skipped_nonfinite,
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                accum_terms = {}
                if bad_state is not None:
                    raise FloatingPointError("Non-finite point dVAE model state at step %d: %s" % (step, bad_state))
                if args.point_dvae_nonfinite_action == "skip":
                    continue
                raise FloatingPointError("Non-finite point dVAE loss at step %d" % step)
            scaled.backward()
            accum += 1
            with torch.no_grad():
                accum_terms["loss"] = accum_terms.get("loss", 0.0) + to_float(loss)
                accum_terms["recon_loss"] = accum_terms.get("recon_loss", 0.0) + to_float(recon_loss)
                accum_terms["kl_loss"] = accum_terms.get("kl_loss", 0.0) + to_float(kl_loss)
                accum_terms["temperature"] = accum_terms.get("temperature", 0.0) + float(temperature)
                accum_terms["kld_weight"] = accum_terms.get("kld_weight", 0.0) + float(kld_weight)
            if accum < grad_accum:
                continue

            if args.pointbert_official_schedule:
                base_lr = get_pointbert_epoch_lr(
                    epoch,
                    args.epochs,
                    int(args.warmup_epochs),
                    args.point_dvae_start_lr,
                    args.point_dvae_lr,
                    args.point_dvae_final_lr,
                )
            else:
                base_lr = get_lr(step, max_steps, warmup_steps, args.point_dvae_start_lr, args.point_dvae_lr, args.point_dvae_final_lr)
            if args.point_dvae_adaptive_lr:
                lr = max(float(args.point_dvae_adaptive_min_lr), float(base_lr) * float(adaptive_lr_scale))
            else:
                lr = float(base_lr)
            set_optimizer_lr(optimizer, lr)
            if args.grad_clip and args.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(raw_dvae.parameters(), args.grad_clip)
                grad_norm_value = float(grad_norm.detach().cpu())
            else:
                grad_norm_value = 0.0
            if not math.isfinite(grad_norm_value):
                skipped_nonfinite += 1
                print(
                    "point_dvae_nonfinite_grad step=%05d epoch=%d action=%s grad_norm=%s skipped=%d"
                    % (step, epoch, args.point_dvae_nonfinite_action, str(grad_norm_value), skipped_nonfinite)
                )
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                accum_terms = {}
                if args.point_dvae_nonfinite_action == "skip":
                    continue
                raise FloatingPointError("Non-finite point dVAE grad norm at step %d" % step)
            optimizer.step()
            bad_state = first_nonfinite_module_state(raw_dvae)
            if bad_state is not None:
                print(
                    "point_dvae_nonfinite_state step=%05d epoch=%d state=%s lr=%.2e grad_norm=%s"
                    % (step, epoch, bad_state, lr, str(grad_norm_value))
                )
                save_point_dvae_state(
                    output_dir,
                    "point_dvae_nonfinite_state_diagnostic.pt",
                    raw_dvae,
                    optimizer,
                    args,
                    history,
                    step,
                    epoch,
                )
                raise FloatingPointError("Non-finite point dVAE model state at step %d: %s" % (step, bad_state))
            optimizer.zero_grad(set_to_none=True)
            elapsed_so_far = max(time.time() - start, 1e-6)
            run_steps_done = max(step - run_start_step + 1, 1)
            run_steps_per_s = run_steps_done / elapsed_so_far
            metrics: Dict[str, Any] = {
                "epoch": float(epoch),
                "step": float(step),
                "lr": float(lr),
                "base_lr": float(base_lr),
                "adaptive_lr_scale": float(adaptive_lr_scale),
                "adaptive_lr_best": adaptive_lr_best,
                "adaptive_lr_bad_windows": int(adaptive_lr_bad_windows),
                "adaptive_lr_last_check_step": int(adaptive_lr_last_check_step),
                "grad_norm": grad_norm_value,
                "skipped_nonfinite": int(skipped_nonfinite),
                "micro_batches": float(accum),
                "elapsed_s": float(elapsed_so_far),
                "steps_per_s": float(run_steps_per_s),
                "samples_per_s": float(run_steps_done * grad_accum * args.batch_size / elapsed_so_far),
                "eta_s": float(max(max_steps - step - 1, 0) / max(run_steps_per_s, 1e-12)),
            }
            metrics.update({key: value / max(accum, 1) for key, value in accum_terms.items()})
            history.append(metrics)
            if args.point_dvae_adaptive_lr:
                window = max(1, int(args.point_dvae_adaptive_lr_window_steps))
                if len(history) >= window and (adaptive_lr_last_check_step < 0 or step - adaptive_lr_last_check_step >= window):
                    monitor_key = args.point_dvae_adaptive_lr_monitor
                    recent_values = [float(item.get(monitor_key, item.get("loss", 0.0))) for item in history[-window:]]
                    current_window_metric = float(sum(recent_values) / max(len(recent_values), 1))
                    improved = False
                    if adaptive_lr_best is None:
                        adaptive_lr_best = current_window_metric
                        adaptive_lr_bad_windows = 0
                        improved = True
                    else:
                        min_delta = max(abs(adaptive_lr_best) * float(args.point_dvae_adaptive_lr_threshold), 1e-12)
                        if current_window_metric < adaptive_lr_best - min_delta:
                            adaptive_lr_best = current_window_metric
                            adaptive_lr_bad_windows = 0
                            improved = True
                        else:
                            adaptive_lr_bad_windows += 1
                    old_scale = adaptive_lr_scale
                    if (
                        not improved
                        and adaptive_lr_bad_windows >= max(1, int(args.point_dvae_adaptive_lr_patience))
                        and lr > float(args.point_dvae_adaptive_min_lr) * (1.0 + 1e-6)
                    ):
                        adaptive_lr_scale = max(0.0, adaptive_lr_scale * float(args.point_dvae_adaptive_lr_factor))
                        adaptive_lr_bad_windows = 0
                        next_lr = max(float(args.point_dvae_adaptive_min_lr), float(base_lr) * float(adaptive_lr_scale))
                        print(
                            "adaptive_lr_update step=%05d %s_window=%.6f best=%.6f scale=%.4f->%.4f next_lr=%.2e"
                            % (
                                step,
                                monitor_key,
                                current_window_metric,
                                adaptive_lr_best,
                                old_scale,
                                adaptive_lr_scale,
                                next_lr,
                            )
                        )
                    adaptive_lr_last_check_step = step
                    metrics["adaptive_lr_window_metric"] = current_window_metric
                    metrics["adaptive_lr_scale"] = float(adaptive_lr_scale)
                    metrics["adaptive_lr_best"] = adaptive_lr_best
                    metrics["adaptive_lr_bad_windows"] = int(adaptive_lr_bad_windows)
                    metrics["adaptive_lr_last_check_step"] = int(adaptive_lr_last_check_step)
            if step % args.log_every == 0:
                atomic_json_dump(
                    {"step": step, "epoch": epoch, "last": metrics, "extra": {"stage": "point_dvae", "lightweight": True}},
                    output_dir / "metrics_live.json",
                )
                print(
                    "step=%05d epoch=%d loss=%.6f recon=%.6f kl=%.6f temp=%.4f kld_w=%.4f lr=%.2e samples_s=%.2f eta_h=%.2f"
                    % (
                        step,
                        epoch,
                        metrics.get("loss", 0.0),
                        metrics.get("recon_loss", 0.0),
                        metrics.get("kl_loss", 0.0),
                        metrics.get("temperature", 0.0),
                        metrics.get("kld_weight", 0.0),
                        lr,
                        metrics.get("samples_per_s", 0.0),
                        metrics.get("eta_s", 0.0) / 3600.0,
                    )
                )
            step += 1
            accum = 0
            accum_terms = {}
            if args.latest_save_every_steps > 0 and step % args.latest_save_every_steps == 0:
                prune_old_checkpoints(output_dir, "point_dvae_step_*.pt", args.keep_last_checkpoints)
                save_point_dvae_state(output_dir, "point_dvae_latest.pt", raw_dvae, optimizer, args, history, step, epoch)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                if save_point_dvae_state(output_dir, "point_dvae_step_%06d.pt" % step, raw_dvae, optimizer, args, history, step, epoch):
                    removed = prune_old_checkpoints(output_dir, "point_dvae_step_*.pt", args.keep_last_checkpoints)
                    if removed:
                        print("checkpoint_pruned count=%d oldest=%s" % (len(removed), removed[0].name))
            if step >= max_steps:
                stop = True
                break
        if stop:
            break

    elapsed = time.time() - start
    # Exclude private runtime objects (e.g. _quality_lookup with tuple keys,
    # _per_city_stats) that are attached to args but are not JSON-serializable.
    serializable_config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    summary = {
        "config": serializable_config,
        "dataset_root": str(dataset_root),
        "city_root": str(city_root),
        "output_dir": str(output_dir),
        "point_dvae_data": args.point_dvae_data,
        "shapenet55_root": str(Path(args.shapenet55_root).resolve()) if args.point_dvae_data == "shapenet55" else None,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device.index or 0) if torch.cuda.is_available() else None,
        "gpu_ids": gpu_ids,
        "num_scenarios": len(scenarios),
        "num_samples": len(dataset),
        "steps": step,
        "elapsed_s": elapsed,
        "history": history,
        "scenarios": [asdict(s) for s in scenarios],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_point_dvae_state(
        output_dir,
        "point_dvae.pt",
        raw_dvae,
        optimizer,
        args,
        history,
        step,
        int(history[-1]["epoch"]) if history else 0,
        required=True,
    )
    print("saved_metrics=%s" % (output_dir / "metrics.json"))
    print("saved_point_dvae=%s" % (output_dir / "point_dvae.pt"))
    return summary


def configure_wwm_pretrain_trainable(model: PaperWWM, args: argparse.Namespace) -> int:
    for p in model.parameters():
        p.requires_grad = True
    model.freeze_target()
    for p in model.decoder.parameters():
        p.requires_grad = False
    model.anchor_residual_scale.requires_grad = False
    if bool(getattr(args, "pretrain_freeze_encoder", False)):
        # Predictor-focused fine-tune: freeze the (already-converged) input stem
        # and online encoder. In sigreg mode the target is a stop-grad pass of
        # the online stem+encoder, so freezing them turns the target into a fixed
        # strong teacher; the predictor then regresses toward a stationary goal.
        for p in model.online_stem.parameters():
            p.requires_grad = False
        for p in model.online_encoder.parameters():
            p.requires_grad = False
    if args.point_tokenizer == "pointbert_dvae" and args.freeze_point_dvae:
        model.set_point_dvae_frozen(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_wwm_pretrain(args: argparse.Namespace) -> Dict[str, Any]:
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.set_float32_matmul_precision("high")

    dataset_root = Path(args.dataset_root).resolve()
    city_root = resolve_city_root(dataset_root, args.city_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_names = list(dict.fromkeys([str(args.split)] + [str(value) for value in args.extra_train_splits]))
    scenario_groups: List[Tuple[str, List[ScenarioFile]]] = []
    for split_name in split_names:
        split_scenarios = discover_scenarios(
            dataset_root,
            split_name,
            optional_limit(args.max_files),
            optional_limit(args.max_samples_per_file),
        )
        split_scenarios = filter_scenarios_by_speeds(split_scenarios, args.train_speeds)
        scenario_groups.append((split_name, split_scenarios))
        speed_counts: Dict[str, int] = {}
        for scenario in split_scenarios:
            speed_key = "%g" % scenario_speed_kmh(scenario)
            speed_counts[speed_key] = speed_counts.get(speed_key, 0) + int(scenario.samples)
        print("pretrain_data split=%s scenarios=%d raw_samples=%d speeds=%s" % (
            split_name,
            len(split_scenarios),
            sum(int(scenario.samples) for scenario in split_scenarios),
            json.dumps(speed_counts, sort_keys=True),
        ))
    scenarios = [scenario for _, group in scenario_groups for scenario in group]

    # ── balanced/normalized pipeline setup ──────────────────────────────────
    from .data import load_quality_index, load_per_city_csi_stats
    args._quality_lookup = None
    args._per_city_stats = None
    args._em_rt_sidecar_root = None
    conditional_weights = (
        float(getattr(args, "em_deltar_weight", 0.0))
        + float(getattr(args, "em_conditional_sigreg_weight", 0.0))
        + float(getattr(args, "em_conditional_scale_weight", 0.0))
        + float(getattr(args, "em_conditional_covariance_weight", 0.0))
        + float(getattr(args, "em_conditional_rank_weight", 0.0))
        + float(getattr(args, "em_rt_path_weight", 0.0))
    )
    if conditional_weights > 0 and not bool(getattr(args, "em_physics_enable", False)):
        raise ValueError("EM conditional/path losses require --em-physics-enable")
    if getattr(args, "em_rt_sidecar_root", None):
        args._em_rt_sidecar_root = Path(str(args.em_rt_sidecar_root)).resolve()
        if not args._em_rt_sidecar_root.exists():
            raise FileNotFoundError("--em-rt-sidecar-root does not exist: %s" % args._em_rt_sidecar_root)
        if float(getattr(args, "em_rt_path_weight", 0.0)) <= 0:
            raise ValueError("--em-rt-sidecar-root requires --em-rt-path-weight > 0")
        if int(getattr(args, "em_rt_path_dim", 0)) <= 0:
            raise ValueError("--em-rt-path-dim must be positive when RT sidecar distillation is enabled")
    elif float(getattr(args, "em_rt_path_weight", 0.0)) > 0:
        raise ValueError("--em-rt-path-weight > 0 requires --em-rt-sidecar-root")
    if getattr(args, "csi_quality_index", None):
        qi_path = Path(str(args.csi_quality_index)).resolve()
        args._quality_lookup = load_quality_index(qi_path)
        print("quality_index=loaded path=%s entries=%d" % (qi_path, len(args._quality_lookup)))
    if getattr(args, "csi_stats_file", None):
        stats_path = Path(str(args.csi_stats_file)).resolve()
        args._per_city_stats = load_per_city_csi_stats(stats_path)
        recorded_eps = float(args._per_city_stats.get("signed_log_eps", 1.0))
        if abs(float(args.signed_log_eps) - recorded_eps) > 1e-6:
            raise ValueError(
                "signed_log_eps mismatch: --signed-log-eps %.6g but stats file records %.6g; "
                "pass --signed-log-eps %.6g to match." % (
                    float(args.signed_log_eps), recorded_eps, recorded_eps)
            )
        # Use global relative_signed_log stats from the file (skip estimation scan).
        if args.csi_mean is None:
            args.csi_mean = float(args._per_city_stats["global_mean"])
        if args.csi_std is None:
            args.csi_std = float(args._per_city_stats["global_std"])
        print("csi_stats_file=loaded path=%s global_mean=%.6g global_std=%.6g" % (
            stats_path, float(args.csi_mean), float(args.csi_std)))
    # ── end pipeline setup ───────────────────────────────────────────────────

    csi_mean, csi_std, stats_count = estimate_signed_log_stats(scenarios, args)
    args.csi_mean = csi_mean
    args.csi_std = csi_std
    dataset, loader = make_multi_split_loader(scenario_groups, city_root, args, shuffle=True, drop_last=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_model = PaperWWM(args).to(device)
    if args.point_tokenizer == "pointbert_dvae":
        if args.point_dvae_resume:
            load_point_dvae_checkpoint(raw_model, args.point_dvae_resume, args.point_dvae_strict_resume)
            raw_model.to(device)
        elif not args.allow_random_point_dvae:
            raise ValueError(
                "--train-stage wwm_pretrain with --point-tokenizer pointbert_dvae requires --point-dvae-resume. "
                "Run --train-stage point_dvae first."
            )
    trainable = configure_wwm_pretrain_trainable(raw_model, args)
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-10,
        weight_decay=args.weight_decay,
        foreach=args.adamw_foreach,
    )

    step = 0
    start_epoch = 0
    history: List[Dict[str, Any]] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_state = ckpt["model"]
        # Always tolerate new modules (e.g. pretrain_csi_head added for the hybrid
        # objective) so we can warm-start from an older checkpoint: filter to
        # shape-compatible keys, load non-strict, and report what stayed random.
        model_state, dropped_prefix, dropped_shape = filter_compatible_state_dict(raw_model, model_state)
        if dropped_prefix or dropped_shape:
            print("resume_filter dropped_prefix=%d dropped_shape=%d" % (dropped_prefix, dropped_shape))
        incompatible = raw_model.load_state_dict(model_state, strict=False)
        print("partial_resume missing=%d unexpected=%d" % (len(incompatible.missing_keys), len(incompatible.unexpected_keys)))
        if getattr(raw_model, "has_ema_target", False) and not any(
            key.startswith("target_encoder.") for key in model_state
        ):
            # Warm-starting an EMA-mode run from a sigreg-mode checkpoint: the
            # checkpoint has no target-branch keys, so the freshly deep-copied
            # EMA teacher is RANDOM. Sync it from the loaded online encoder,
            # otherwise the first epoch chases a random teacher.
            raw_model.sync_target_from_online()
            print("ema_target=synced_from_online (resumed sigreg-mode checkpoint)")
        if getattr(raw_model, "pretrain_residual_head", False) and not getattr(args, "no_rezero_residual_head", False):
            # The resumed checkpoint's pretrain_csi_head was trained to emit ABSOLUTE
            # CSI. In residual mode pred = anchor + head(delta); loading the old
            # absolute head makes pred ~= anchor + full_CSI (double magnitude) which
            # diverges to NaN within a few steps. Re-zero the head's last layer so it
            # restarts from delta=0 (pred == anchor == copy-baseline ~0.91) and learns
            # only the correction.
            # NOTE: when resuming a checkpoint that was ALREADY residual-trained, pass
            # --no-rezero-residual-head so the learned corrections are NOT wiped.
            last = raw_model.pretrain_csi_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            print("residual_head=rezeroed after resume (delta starts at 0)")
        elif getattr(raw_model, "pretrain_residual_head", False):
            print("residual_head=kept (--no-rezero-residual-head): resuming residual-trained weights")
        if not args.reset_optimizer:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("optimizer_state=loaded")
            except Exception as exc:
                print("optimizer_state=reset reason=%s" % exc)
        else:
            print("optimizer_state=reset")
        if args.reset_training_state:
            print("training_state=reset")
        else:
            history = list(ckpt.get("history", []))
            step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
        print("resumed_from=%s step=%d epoch=%d" % (args.resume, step, start_epoch))
        if args.point_tokenizer == "pointbert_dvae" and args.point_dvae_resume:
            load_point_dvae_checkpoint(raw_model, args.point_dvae_resume, args.point_dvae_strict_resume)
            raw_model.to(device)
        del model_state, ckpt

    gpu_ids: List[int] = []
    if device.type == "cuda" and args.multi_gpu:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if len(gpu_ids) > 1:
            raw_model = raw_model.to(torch.device("cuda:%d" % gpu_ids[0]))
            device = torch.device("cuda:%d" % gpu_ids[0])
            model: nn.Module = nn.DataParallel(raw_model, device_ids=gpu_ids, output_device=gpu_ids[0])
            print("multi_gpu=DataParallel device_ids=%s" % ",".join(str(i) for i in gpu_ids))
        else:
            model = raw_model
    else:
        model = raw_model

    grad_accum = int(args.grad_accum_steps)
    if grad_accum <= 0:
        grad_accum = max(1, int(math.ceil(float(args.global_batch_size) / float(args.batch_size))))
    steps_per_epoch = int(math.ceil(len(loader) / float(grad_accum)))
    max_steps = int(args.limit_steps) if args.limit_steps and args.limit_steps > 0 else int(args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch)) if args.warmup_epochs > 0 else 0
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    remaining_steps = max(max_steps - int(step), 0)
    resume_skip_batches = 0
    if args.resume and not args.reset_training_state and step > 0:
        checkpoint_epoch = int(start_epoch)
        start_epoch, resume_skip_batches = resume_epoch_position(step, steps_per_epoch, grad_accum)
        print(
            "resume_position checkpoint_epoch=%d logical_epoch=%d skip_batches=%d"
            % (checkpoint_epoch, start_epoch, resume_skip_batches)
        )
    print_model_summary(raw_model, args, trainable)
    print(
        "stage=wwm_pretrain device=%s samples=%d scenarios=%d steps_per_epoch=%d max_steps=%d "
        "resume_step=%d remaining_steps=%d grad_accum=%d global_batch~=%d csi_mean=%.6f csi_std=%.6f"
        % (
            device,
            len(dataset),
            len(scenarios),
            steps_per_epoch,
            max_steps,
            int(step),
            remaining_steps,
            grad_accum,
            grad_accum * args.batch_size,
            csi_mean,
            csi_std,
        )
    )
    print(
        "run_paths output_dir=%s checkpoint_latest=%s metrics_live=%s"
        % (output_dir, output_dir / "checkpoint_latest.pt", output_dir / "metrics_live.json")
    )

    start = time.time()
    run_start_step = int(step)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    stop_requested = False
    stop_marker = output_dir / "STOP_REQUESTED"
    for epoch in range(start_epoch, args.epochs):
        accum = 0
        accum_loss = 0.0
        accum_components: Dict[str, float] = {"latent": 0.0, "recon": 0.0, "sgcs1m": 0.0}
        task_counts = {"fine": 0, "coarse": 0, "traj": 0, "point": 0, "temporal": 0, "geo": 0}
        point_spread_reports = 0
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_skip_batches:
                continue
            h = batch["h"].to(device, non_blocking=True)
            traj = batch["traj"].to(device, non_blocking=True)
            em_path = batch.get("em_path")
            if em_path is not None:
                em_path = em_path.to(device, non_blocking=True)
            point_cloud = move_point_input(
                extract_point_input(batch, origin_dropout=float(getattr(args, "point_origin_dropout", 0.0))),
                device,
            )
            pre_mask, task = make_pretrain_mask(raw_model, args, h.shape[0], device)
            raw_model.set_pretrain_step(step)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    h, point_cloud, traj, pretrain_mask=pre_mask, pretrain_task=task,
                    em_path=em_path, mode="pretrain"
                )
                loss = out["pretrain_loss"].mean()
                scaled = loss / grad_accum
            if not torch.isfinite(loss):
                # Skip this micro-batch instead of killing a multi-hour run: zero any
                # partial grad from this iteration and continue. A persistent NaN
                # (many consecutive skips) means weights are already poisoned -> abort
                # so we don't waste the GPU spinning forever (the last good checkpoint
                # is on disk and can be resumed with --no-rezero-residual-head).
                nonfinite_skips = getattr(train_wwm_pretrain, "_nonfinite_skips", 0) + 1
                setattr(train_wwm_pretrain, "_nonfinite_skips", nonfinite_skips)
                consec = getattr(train_wwm_pretrain, "_consec_skips", 0) + 1
                setattr(train_wwm_pretrain, "_consec_skips", consec)
                print("WARN non-finite pretrain loss at step %d micro-batch skipped (total_skips=%d consec=%d)" % (step, nonfinite_skips, consec))
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                accum_loss = 0.0
                accum_components = {"latent": 0.0, "recon": 0.0, "sgcs1m": 0.0}
                if consec >= 50:
                    raise FloatingPointError(
                        "Model weights appear NaN-poisoned: %d consecutive non-finite micro-batches at step %d. "
                        "Resume from the last good checkpoint (add --no-rezero-residual-head) with lower LR / "
                        "sgcs-weight." % (consec, step))
                continue
            setattr(train_wwm_pretrain, "_consec_skips", 0)
            scaled.backward()
            if args.cuda_sync_backward and device.type == "cuda":
                # One barrier per completed autograd graph is substantially cheaper
                # than CUDA_LAUNCH_BLOCKING=1 and prevents Windows CUDA kernels from
                # overlapping the next micro-batch after the angular auxiliary path.
                torch.cuda.synchronize(device)
            accum += 1
            accum_loss += to_float(loss)
            # accumulate per-component breakdown for diagnostics
            step_components = getattr(raw_model, "_loss_components", {})
            for ck, cv in step_components.items():
                accum_components[ck] = accum_components.get(ck, 0.0) + cv
            # The point-spread diagnostics only exist on steps where the point task
            # fired, so they need their own denominator rather than `n`.
            if "pspread_log_ratio" in step_components:
                point_spread_reports += 1
            task_counts[task] += 1
            if accum < grad_accum:
                continue

            lr = get_lr(step, max_steps, warmup_steps, args.start_lr, args.lr, args.final_lr, args.lr_schedule)
            set_optimizer_lr(optimizer, lr)
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                # A non-finite GRADIENT (even with finite loss) would poison every
                # weight via optimizer.step(). Skip the update entirely so the model
                # stays clean; this is the root guard the loss-only check missed.
                grad_skips = getattr(train_wwm_pretrain, "_grad_skips", 0) + 1
                setattr(train_wwm_pretrain, "_grad_skips", grad_skips)
                print("WARN non-finite grad_norm at step %d update skipped (grad_skips=%d)" % (step, grad_skips))
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                accum_loss = 0.0
                accum_components = {"latent": 0.0, "recon": 0.0, "sgcs1m": 0.0}
                continue
            optimizer.step()
            raw_model.update_ema(args.ema_decay)
            optimizer.zero_grad(set_to_none=True)

            elapsed_so_far = max(time.time() - start, 1e-6)
            run_steps_done = max(step - run_start_step + 1, 1)
            metrics: Dict[str, Any] = {
                "epoch": float(epoch),
                "step": float(step),
                "lr": float(lr),
                "grad_norm": float(grad_norm.detach().cpu()),
                "micro_batches": float(accum),
                "pretrain_loss": float(accum_loss / max(accum, 1)),
                "elapsed_s": float(elapsed_so_far),
                "steps_per_s": float(run_steps_done / elapsed_so_far),
                "samples_per_s": float(run_steps_done * grad_accum * args.batch_size / elapsed_so_far),
                "eta_s": float(max(max_steps - step - 1, 0) / max(run_steps_done / elapsed_so_far, 1e-12)),
                "context_effective_weight": float(args.context_loss_weight) * (
                    min(max(float(step) / float(args.context_loss_warmup_steps), 0.0), 1.0)
                    if int(args.context_loss_warmup_steps) > 0 else 1.0
                ),
                "sgcs_effective_weight": float(args.pretrain_sgcs_weight) * (
                    min(max(float(step) / float(args.pretrain_sgcs_warmup_steps), 0.0), 1.0)
                    if int(args.pretrain_sgcs_warmup_steps) > 0 else 1.0
                ),
            }
            metrics.update({"task_count/%s" % key: float(value) for key, value in task_counts.items()})
            # Per-component loss breakdown (averaged over the accumulated micro-batches)
            # so the dashboard can show latent vs SIGReg vs context — critical for the
            # EMA-free sigreg run, where SIGReg is the sole collapse guard and its
            # magnitude relative to latent must stay visible.
            _n = max(accum, 1)
            for _ck, _cv in accum_components.items():
                metrics["loss/%s" % _ck] = float(_cv / _n)
            history.append(metrics)
            atomic_json_dump(
                {"step": step, "epoch": epoch, "last": metrics, "extra": {"stage": "wwm_pretrain"}},
                output_dir / "metrics_live.json",
            )
            if step % args.log_every == 0:
                n = max(accum, 1)
                lat = accum_components.get("latent", 0.0) / n
                rec = accum_components.get("recon", 0.0) / n
                visible_recon = accum_components.get("visible_recon", 0.0) / n
                context_loss = accum_components.get("context", 0.0) / n
                visreg_loss_value = accum_components.get("visreg", 0.0) / n
                deep_loss = accum_components.get("deep", 0.0) / n
                sgc = accum_components.get("sgcs1m", 0.0) / n
                phase_loss = accum_components.get("recon_phase", 0.0) / n
                em_loss = accum_components.get("em", 0.0) / n
                kvis_time = accum_components.get("em_kvisreg_time_raw", 0.0) / n
                em_rank = accum_components.get("em_kvisreg_rank", 0.0) / n
                em_relation = accum_components.get("em_relation_raw", 0.0) / n
                ctx_phys = accum_components.get("context_weight_physical", 0.0) / n
                # Height aux: report the raw KL next to the constant-histogram KL a
                # head that ignores its input would score. hgt_kl alone cannot tell
                # learning from prior fitting.
                height_kl = accum_components.get("height_kl", 0.0) / n
                height_const_kl = accum_components.get("height_const_kl", 0.0) / n
                print(
                    "step=%05d epoch=%d pretrain_loss=%.6f lat=%.4f rec=%.4f phase=%.4f vis=%.4f ctx=%.4f deep=%.4f visreg=%.4f pspread=%.4f pspr_ratio=%.3f pspr_w=%.3f/%.3f sgcs1m=%.4f em=%.4f rel=%.4f kvis_t=%.4f rank=%.2f ctxphys=%.2f hgt_kl=%.4f/%.4f grad=%.4f lr=%.2e tasks(f/c/t/p/tmp/geo)=%d/%d/%d/%d/%d/%d repr_std(c/p/t)=%.3f/%.3f/%.3f repr_cos(c/p/t)=%.3f/%.3f/%.3f samples_s=%.2f eta_h=%.2f"
                    % (
                        step,
                        epoch,
                        metrics.get("pretrain_loss", 0.0),
                        lat, rec, phase_loss, visible_recon, context_loss, deep_loss, visreg_loss_value,
                        accum_components.get("pspread", 0.0) / n,
                        accum_components.get("pspread_log_ratio", 0.0) / max(point_spread_reports, 1),
                        accum_components.get("pspread_pred_within", 0.0) / max(point_spread_reports, 1),
                        accum_components.get("pspread_target_within", 0.0) / max(point_spread_reports, 1),
                        sgc,
                        em_loss, em_relation, kvis_time, em_rank, ctx_phys,
                        height_kl, height_const_kl,
                        metrics.get("grad_norm", 0.0),
                        lr,
                        task_counts["fine"],
                        task_counts["coarse"],
                        task_counts["traj"],
                        task_counts["point"],
                        task_counts["temporal"],
                        task_counts["geo"],
                        accum_components.get("repr_std_csi", 0.0) / n,
                        accum_components.get("repr_std_point", 0.0) / n,
                        accum_components.get("repr_std_traj", 0.0) / n,
                        accum_components.get("repr_cos_csi", 0.0) / n,
                        accum_components.get("repr_cos_point", 0.0) / n,
                        accum_components.get("repr_cos_traj", 0.0) / n,
                        metrics.get("samples_per_s", 0.0),
                        metrics.get("eta_s", 0.0) / 3600.0,
                    )
                )

            step += 1
            accum = 0
            accum_loss = 0.0
            accum_components = {"latent": 0.0, "recon": 0.0, "sgcs1m": 0.0}
            task_counts = {"fine": 0, "coarse": 0, "traj": 0, "point": 0, "temporal": 0, "geo": 0}
            point_spread_reports = 0
            if args.latest_save_every_steps > 0 and step % args.latest_save_every_steps == 0:
                prune_old_checkpoints(output_dir, "checkpoint_step_*.pt", args.keep_last_checkpoints)
                save_state(output_dir, "checkpoint_latest.pt", model, optimizer, args, history, step, epoch)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                if save_state(output_dir, "checkpoint_step_%06d.pt" % step, model, optimizer, args, history, step, epoch):
                    removed = prune_old_checkpoints(output_dir, "checkpoint_step_*.pt", args.keep_last_checkpoints)
                    if removed:
                        print("checkpoint_pruned count=%d oldest=%s" % (len(removed), removed[0].name))
            if stop_marker.exists():
                print("stop_requested marker=%s step=%d saving_checkpoint" % (stop_marker, step))
                checkpoint_saved = save_state(
                    output_dir,
                    "checkpoint_latest.pt",
                    model,
                    optimizer,
                    args,
                    history,
                    step,
                    epoch,
                    extra={"stage": "wwm_pretrain", "status": "stopped_by_request"},
                )
                if checkpoint_saved:
                    print("stop_checkpoint_saved=%s" % (output_dir / "checkpoint_latest.pt"))
                stop_requested = True
                stop = True
                break
            if step >= max_steps:
                stop = True
                break
        if stop:
            break
        resume_skip_batches = 0

    elapsed = time.time() - start
    # Exclude private runtime objects (e.g. _quality_lookup with tuple keys,
    # _per_city_stats) that are attached to args but are not JSON-serializable.
    serializable_config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    summary = {
        "config": serializable_config,
        "dataset_root": str(dataset_root),
        "city_root": str(city_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device.index or 0) if torch.cuda.is_available() else None,
        "gpu_ids": gpu_ids,
        "num_scenarios": len(scenarios),
        "num_samples": len(dataset),
        "csi_stats": {"mean": args.csi_mean, "std": args.csi_std, "count": stats_count},
        "schedule": {
            "steps_per_epoch": steps_per_epoch,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "grad_accum_steps": grad_accum,
        },
        "steps": step,
        "elapsed_s": elapsed,
        "history": history,
        "scenarios": [asdict(s) for s in scenarios],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if not stop_requested:
        save_state(
            output_dir,
            "checkpoint.pt",
            model,
            optimizer,
            args,
            history,
            step,
            int(history[-1]["epoch"]) if history else 0,
            required=True,
        )
    print("saved_metrics=%s" % (output_dir / "metrics.json"))
    if not stop_requested:
        print("saved_checkpoint=%s" % (output_dir / "checkpoint.pt"))
    return summary


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    split: str,
) -> Dict[str, float]:
    model.eval()
    base_model = unwrap_model(model)
    totals: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            h = batch["h"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if args.anchor_only_fast_eval and args.temporal_anchor != "none":
                    _, terms = compute_anchor_only_terms(base_model, h, args)
                else:
                    traj = batch["traj"].to(device, non_blocking=True)
                    point_cloud = move_point_input(extract_point_input(batch), device)
                    out = model(h, point_cloud, traj)
                    _, terms = compute_terms(out, h, args)
            bs = int(h.shape[0])
            count += bs
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + to_float(value) * bs
    model.train()
    result = {("%s/%s" % (split, key)): value / max(count, 1) for key, value in totals.items()}
    result["%s/samples" % split] = float(count)
    return result


EVAL_KEYS = (
    "samples",
    "loss",
    "recon_loss",
    "raw_csi_mse",
    "mag_loss",
    "phase_loss",
    "sgcs",
    "sgcs_t15",
    "sgcs_t16",
    "sgcs_h1",
    "sgcs_h2",
    "sgcs_h3",
    "sgcs_h4",
    "sgcs_final",
    "sgcs_avg",
    "sgcs_loss",
    "nmse_db",
    "transformed_nmse_db",
    "latent_loss",
)


def strip_prefix(metrics: Dict[str, float], split: str) -> Dict[str, float]:
    prefix = split + "/"
    return {key[len(prefix) :]: value for key, value in metrics.items() if key.startswith(prefix)}


def write_eval_summary(output_dir: Path, eval_results: Dict[str, Dict[str, float]]) -> None:
    path = output_dir / "eval_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("split",) + EVAL_KEYS)
        writer.writeheader()
        for split, metrics in eval_results.items():
            compact = strip_prefix(metrics, split)
            row = {"split": split}
            row.update({key: compact.get(key, "") for key in EVAL_KEYS})
            writer.writerow(row)


def save_state(
    output_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: List[Dict[str, Any]],
    step: int,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
    required: bool = False,
) -> bool:
    base_model = unwrap_model(model)
    payload = {
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # Strip private runtime objects (e.g. _quality_lookup, a 175k-entry
        # tuple-keyed dict) so checkpoints stay small and JSON-safe on resume.
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "history": history,
        "step": step,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    checkpoint_path = output_dir / name
    try:
        atomic_torch_save(payload, checkpoint_path)
    except CheckpointSaveError as exc:
        if required:
            raise
        print("checkpoint_save_skipped path=%s reason=%s" % (checkpoint_path, exc))
        return False
    live = {"step": step, "epoch": epoch, "last": history[-1] if history else {}, "extra": extra or {}}
    atomic_json_dump(live, output_dir / "metrics_live.json")
    return True


def filter_compatible_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    drop_prefixes: Tuple[str, ...] = (),
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    current = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    dropped_by_prefix = 0
    dropped_by_shape = 0
    for key, value in state_dict.items():
        if any(key.startswith(prefix) for prefix in drop_prefixes):
            dropped_by_prefix += 1
            continue
        if key not in current or tuple(current[key].shape) != tuple(value.shape):
            dropped_by_shape += 1
            continue
        filtered[key] = value
    return filtered, dropped_by_prefix, dropped_by_shape


def configure_trainable(model: PaperWWM, args: argparse.Namespace) -> int:
    mode = str(args.downstream_mode)
    if args.freeze_backbone:
        mode = "paper"
    if mode == "pretrain_joint":
        for p in model.parameters():
            p.requires_grad = True
        model.freeze_target()
    elif mode == "finetune":
        for p in model.parameters():
            p.requires_grad = False
        for module in (model.online_stem, model.online_encoder, model.predictor, model.decoder):
            for p in module.parameters():
                p.requires_grad = True
        model.mask_token.requires_grad = True
        model.anchor_residual_scale.requires_grad = True
        model.freeze_target()
    elif mode == "paper":
        for p in model.parameters():
            p.requires_grad = False
        # Supplementary Note 1 keeps the WWM encoder/predictor frozen and trains
        # the CSI decoder for channel prediction.
        for p in model.online_stem.parameters():
            p.requires_grad = False
        for p in model.online_encoder.parameters():
            p.requires_grad = False
        for p in model.predictor.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = True
        model.mask_token.requires_grad = False
        model.anchor_residual_scale.requires_grad = False
        model.freeze_target()
    else:
        raise ValueError("Unknown downstream_mode: %s" % mode)
    if args.point_tokenizer == "pointbert_dvae":
        if mode == "paper" or args.freeze_point_dvae:
            model.set_point_dvae_frozen(True)
        else:
            model.online_stem.point.set_dvae_frozen(False)
            if model.target_stem is not None:
                model.target_stem.point.set_dvae_frozen(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    for prefix in prefixes:
        selected = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if selected:
            return selected
    if any(key.startswith("module.") for key in state_dict):
        stripped = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
        return normalize_state_dict_keys(stripped, prefixes)
    return state_dict


def load_point_dvae_checkpoint(model: PaperWWM, path: str, strict: bool) -> None:
    if not path:
        return
    if model.online_stem.point.dvae is None:
        raise ValueError("--point-dvae-resume was supplied but --point-tokenizer is not pointbert_dvae.")
    ckpt = torch.load(path, map_location="cpu")
    raw_state = checkpoint_model_state(ckpt)
    if not isinstance(raw_state, dict):
        raise ValueError("Unsupported point dVAE checkpoint format: %s" % path)
    state = normalize_state_dict_keys(
        raw_state,
        prefixes=(
            "online_stem.point.dvae.",
            "module.online_stem.point.dvae.",
            "point.dvae.",
            "module.point.dvae.",
            "dvae.",
            "module.dvae.",
        ),
    )
    incompatible = model.online_stem.point.dvae.load_state_dict(state, strict=strict)
    history = ckpt.get("history", []) if isinstance(ckpt, dict) else []
    matched_temperature = None
    if history and isinstance(history[-1], dict) and history[-1].get("temperature") is not None:
        matched_temperature = float(history[-1]["temperature"])
        model.online_stem.point.dvae_temperature = matched_temperature
    if model.target_stem is not None and model.target_stem.point.dvae is not None:
        model.target_stem.point.dvae.load_state_dict(model.online_stem.point.dvae.state_dict(), strict=True)
        if matched_temperature is not None:
            model.target_stem.point.dvae_temperature = matched_temperature
    print(
        "point_dvae_loaded=%s missing=%d unexpected=%d inference_temperature=%s"
        % (
            path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            "config" if matched_temperature is None else "%.6f" % matched_temperature,
        )
    )


def train(args: argparse.Namespace) -> Dict[str, Any]:
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.set_float32_matmul_precision("high")

    dataset_root = Path(args.dataset_root).resolve()
    city_root = resolve_city_root(dataset_root, args.city_root)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(
        dataset_root,
        args.split,
        optional_limit(args.max_files),
        optional_limit(args.max_samples_per_file),
    )
    csi_mean, csi_std, stats_count = estimate_signed_log_stats(scenarios, args)
    args.csi_mean = csi_mean
    args.csi_std = csi_std
    dataset, loader = make_loader(scenarios, city_root, args, shuffle=True, drop_last=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_model = PaperWWM(args).to(device)
    if args.point_tokenizer == "pointbert_dvae":
        if args.point_dvae_resume:
            load_point_dvae_checkpoint(raw_model, args.point_dvae_resume, args.point_dvae_strict_resume)
            raw_model.to(device)
        elif not args.allow_random_point_dvae:
            raise ValueError(
                "--point-tokenizer pointbert_dvae requires --point-dvae-resume. "
                "Run --train-stage point_dvae first, or pass --allow-random-point-dvae for smoke tests only."
            )
        raw_model.set_point_dvae_frozen(args.freeze_point_dvae or args.downstream_mode == "paper")
    trainable = configure_trainable(raw_model, args)
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=float(getattr(args, "downstream_lr", args.lr)),
        betas=(0.9, 0.95),
        eps=1e-10,
        weight_decay=args.weight_decay,
        foreach=args.adamw_foreach,
    )

    step = 0
    start_epoch = 0
    history: List[Dict[str, Any]] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model_state = ckpt["model"]
        if args.reset_decoder_on_resume or not args.strict_resume:
            model_state, dropped_prefix, dropped_shape = filter_compatible_state_dict(
                raw_model,
                model_state,
                drop_prefixes=("decoder.",) if args.reset_decoder_on_resume else (),
            )
            if dropped_prefix or dropped_shape:
                print("resume_filter dropped_prefix=%d dropped_shape=%d" % (dropped_prefix, dropped_shape))
        incompatible = raw_model.load_state_dict(model_state, strict=args.strict_resume and not args.reset_decoder_on_resume)
        if not args.strict_resume:
            print("partial_resume missing=%d unexpected=%d" % (len(incompatible.missing_keys), len(incompatible.unexpected_keys)))
        if args.reset_decoder_on_resume:
            print("decoder_state=reset missing=%d unexpected=%d" % (len(incompatible.missing_keys), len(incompatible.unexpected_keys)))
        # `paper` describes which parameters are trainable; it must not discard
        # optimizer state when resuming the same downstream run after interruption.
        reset_optimizer = bool(args.reset_optimizer)
        if not reset_optimizer:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("optimizer_state=loaded")
            except Exception as exc:
                print("optimizer_state=reset reason=%s" % exc)
        else:
            print("optimizer_state=reset")
        if args.reset_training_state:
            history = []
            step = 0
            start_epoch = 0
            print("training_state=reset")
        else:
            history = list(ckpt.get("history", []))
            step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
        if args.resume_anchor_residual_scale is not None:
            with torch.no_grad():
                raw_model.anchor_residual_scale.fill_(float(args.resume_anchor_residual_scale))
            print("resume_anchor_residual_scale=%.6f" % float(args.resume_anchor_residual_scale))
        print("resumed_from=%s step=%d epoch=%d" % (args.resume, step, start_epoch))
        if args.point_tokenizer == "pointbert_dvae" and args.point_dvae_resume:
            load_point_dvae_checkpoint(raw_model, args.point_dvae_resume, args.point_dvae_strict_resume)
            raw_model.to(device)

    gpu_ids: List[int] = []
    if device.type == "cuda" and args.multi_gpu:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if len(gpu_ids) > 1:
            if device.index is not None and device.index != gpu_ids[0]:
                raise ValueError("--device cuda:%d conflicts with first --gpu-ids device %d" % (device.index, gpu_ids[0]))
            raw_model = raw_model.to(torch.device("cuda:%d" % gpu_ids[0]))
            device = torch.device("cuda:%d" % gpu_ids[0])
            model: nn.Module = nn.DataParallel(raw_model, device_ids=gpu_ids, output_device=gpu_ids[0])
            print("multi_gpu=DataParallel device_ids=%s" % ",".join(str(i) for i in gpu_ids))
        else:
            model = raw_model
            print("multi_gpu=requested single visible device=%s" % gpu_ids[0])
    else:
        model = raw_model

    grad_accum = int(args.grad_accum_steps)
    if grad_accum <= 0:
        grad_accum = max(1, int(math.ceil(float(args.global_batch_size) / float(args.batch_size))))
    steps_per_epoch = int(math.ceil(len(loader) / float(grad_accum)))
    max_steps = int(args.limit_steps) if args.limit_steps and args.limit_steps > 0 else int(args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch)) if args.warmup_epochs > 0 else 0
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    remaining_steps = max(max_steps - int(step), 0)

    resume_skip_batches = 0
    if args.resume and not args.reset_training_state and step > 0:
        checkpoint_epoch = int(start_epoch)
        start_epoch, resume_skip_batches = resume_epoch_position(step, steps_per_epoch, grad_accum)
        print(
            "resume_position checkpoint_epoch=%d logical_epoch=%d skip_batches=%d"
            % (checkpoint_epoch, start_epoch, resume_skip_batches)
        )

    print_model_summary(raw_model, args, trainable)
    print(
        "stage=wwm device=%s cuda=%s scenarios=%d samples=%d steps_per_epoch=%d max_steps=%d "
        "resume_step=%d remaining_steps=%d grad_accum=%d global_batch~=%d csi_mean=%.6f csi_std=%.6f"
        % (
            device,
            torch.cuda.get_device_name(device.index or 0) if torch.cuda.is_available() else "none",
            len(scenarios),
            len(dataset),
            steps_per_epoch,
            max_steps,
            int(step),
            remaining_steps,
            grad_accum,
            grad_accum * args.batch_size,
            csi_mean,
            csi_std,
        )
    )
    print(
        "run_paths output_dir=%s checkpoint_latest=%s metrics_live=%s"
        % (output_dir, output_dir / "checkpoint_latest.pt", output_dir / "metrics_live.json")
    )

    start = time.time()
    run_start_step = int(step)
    train_pretrain_aux = bool(args.pretrain_weight > 0 and args.downstream_mode != "paper")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, args.epochs):
        active_csi_weights = csi_loss_weights(args, epoch)
        active_csi_domain = resolve_csi_loss_domain(args, epoch)
        accum = 0
        accum_terms: Dict[str, float] = {}
        accum_pretrain = 0.0
        task_counts = {"fine": 0, "coarse": 0, "traj": 0, "point": 0, "temporal": 0, "geo": 0}
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_skip_batches:
                continue
            h = batch["h"].to(device, non_blocking=True)
            traj = batch["traj"].to(device, non_blocking=True)
            point_cloud = move_point_input(extract_point_input(batch), device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(h, point_cloud, traj)
                loss, terms = compute_terms(
                    out,
                    h,
                    args,
                    loss_weights=active_csi_weights,
                    metric_mode=args.train_metric_sgcs_mode,
                    loss_domain=active_csi_domain,
                )
                pretrain = h.new_zeros(())
                if train_pretrain_aux:
                    pre_mask, task = make_pretrain_mask(raw_model, args, h.shape[0], device)
                    pretrain_out = model(
                        h, point_cloud, traj, pretrain_mask=pre_mask, pretrain_task=task, mode="pretrain"
                    )
                    pretrain = pretrain_out["pretrain_loss"].mean()
                    task_counts[task] += 1
                    loss = loss + args.pretrain_weight * pretrain
                scaled = loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss at step %d" % step)
            scaled.backward()
            accum += 1
            accum_pretrain += to_float(pretrain)
            with torch.no_grad():
                for key, value in terms.items():
                    accum_terms[key] = accum_terms.get(key, 0.0) + to_float(value)
                accum_terms["total_loss"] = accum_terms.get("total_loss", 0.0) + to_float(loss)

            if accum < grad_accum:
                continue

            lr = get_lr(
                step,
                max_steps,
                warmup_steps,
                float(getattr(args, "downstream_start_lr", args.start_lr)),
                float(getattr(args, "downstream_lr", args.lr)),
                float(getattr(args, "downstream_final_lr", args.final_lr)),
                args.lr_schedule,
            )
            set_optimizer_lr(optimizer, lr)
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite downstream grad_norm at step %d" % step)
            optimizer.step()
            if args.downstream_mode != "paper":
                raw_model.update_ema(args.ema_decay)
            optimizer.zero_grad(set_to_none=True)

            metrics: Dict[str, Any] = {
                "epoch": float(epoch),
                "step": float(step),
                "lr": float(lr),
                "grad_norm": float(grad_norm.detach().cpu()),
                "micro_batches": float(accum),
                "pretrain_loss": float(accum_pretrain / max(accum, 1)),
                "csi_mag_weight": float(active_csi_weights[0]),
                "csi_phase_weight": float(active_csi_weights[1]),
                "csi_sgcs_weight": float(active_csi_weights[2]),
                "csi_loss_domain": active_csi_domain,
            }
            metrics.update({key: value / max(accum, 1) for key, value in accum_terms.items()})
            metrics.update({"task_count/%s" % key: float(value) for key, value in task_counts.items()})
            elapsed_so_far = max(time.time() - start, 1e-6)
            metrics["elapsed_s"] = float(elapsed_so_far)
            run_steps_done = max(step - run_start_step + 1, 1)
            metrics["steps_per_s"] = float(run_steps_done / elapsed_so_far)
            metrics["samples_per_s"] = float(run_steps_done * grad_accum * args.batch_size / elapsed_so_far)
            metrics["eta_s"] = float(max(max_steps - step - 1, 0) / max(metrics["steps_per_s"], 1e-12))
            history.append(metrics)
            if step % args.log_every == 0:
                atomic_json_dump(
                    {"step": step + 1, "epoch": epoch, "last": metrics, "extra": {"stage": "wwm"}},
                    output_dir / "metrics_live.json",
                )

            if step % args.log_every == 0:
                print(
                    "step=%05d epoch=%d loss=%.5f recon=%.5f latent=%.5f pre=%.5f "
                    "sgcs=%.4f sgcs_avg=%.4f nmse_db=%.2f lr=%.2e csi_w=%.1f/%.1f/%.1f samples_s=%.2f eta_h=%.2f"
                    % (
                        step,
                        epoch,
                        metrics.get("total_loss", 0.0),
                        metrics.get("recon_loss", 0.0),
                        metrics.get("latent_loss", 0.0),
                        metrics.get("pretrain_loss", 0.0),
                        metrics.get("sgcs", 0.0),
                        metrics.get("sgcs_avg", 0.0),
                        metrics.get("nmse_db", 0.0),
                        lr,
                        metrics.get("csi_mag_weight", 0.0),
                        metrics.get("csi_phase_weight", 0.0),
                        metrics.get("csi_sgcs_weight", 0.0),
                        metrics.get("samples_per_s", 0.0),
                        metrics.get("eta_s", 0.0) / 3600.0,
                    )
                )

            step += 1
            accum = 0
            accum_terms = {}
            accum_pretrain = 0.0
            task_counts = {"fine": 0, "coarse": 0, "traj": 0, "point": 0, "temporal": 0, "geo": 0}
            if args.latest_save_every_steps > 0 and step % args.latest_save_every_steps == 0:
                prune_old_checkpoints(output_dir, "checkpoint_step_*.pt", args.keep_last_checkpoints)
                save_state(output_dir, "checkpoint_latest.pt", model, optimizer, args, history, step, epoch)
            if args.save_every_steps > 0 and step % args.save_every_steps == 0:
                if save_state(output_dir, "checkpoint_step_%06d.pt" % step, model, optimizer, args, history, step, epoch):
                    removed = prune_old_checkpoints(output_dir, "checkpoint_step_*.pt", args.keep_last_checkpoints)
                    if removed:
                        print("checkpoint_pruned count=%d oldest=%s" % (len(removed), removed[0].name))
            if step >= max_steps:
                stop = True
                break
        if stop:
            break
        resume_skip_batches = 0

    elapsed = time.time() - start
    eval_results: Dict[str, Dict[str, float]] = {}
    if args.skip_eval:
        print("skip_eval=true")
    else:
        for split in args.eval_splits:
            try:
                eval_scenarios = discover_scenarios(
                    dataset_root,
                    split,
                    optional_limit(args.eval_max_files),
                    optional_limit(args.eval_max_samples_per_file),
                )
                _, eval_loader = make_loader(eval_scenarios, city_root, args, shuffle=False, drop_last=False)
                metrics = evaluate_model(model, eval_loader, args, device, use_amp, amp_dtype, split)
            except Exception as exc:
                print("skip_eval_split=%s reason=%s" % (split, exc))
                continue
            eval_results[split] = metrics
            compact = strip_prefix(metrics, split)
            print(
                "eval_split=%s samples=%s sgcs=%.4f sgcs_avg=%.4f sgcs_h1/h2/h3/h4=%.4f/%.4f/%.4f/%.4f nmse_db=%.2f loss=%.5f"
                % (
                    split,
                    compact.get("samples", 0.0),
                    compact.get("sgcs", 0.0),
                    compact.get("sgcs_avg", 0.0),
                    compact.get("sgcs_h1", 0.0),
                    compact.get("sgcs_h2", 0.0),
                    compact.get("sgcs_h3", 0.0),
                    compact.get("sgcs_h4", 0.0),
                    compact.get("nmse_db", 0.0),
                    compact.get("loss", 0.0),
                )
            )

    write_eval_summary(output_dir, eval_results)
    summary = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "dataset_root": str(dataset_root),
        "city_root": str(city_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device.index or 0) if torch.cuda.is_available() else None,
        "gpu_ids": gpu_ids,
        "num_scenarios": len(scenarios),
        "num_samples": len(dataset),
        "csi_stats": {"mean": args.csi_mean, "std": args.csi_std, "count": stats_count},
        "schedule": {
            "steps_per_epoch": steps_per_epoch,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "grad_accum_steps": grad_accum,
        },
        "steps": step,
        "elapsed_s": elapsed,
        "history": history,
        "eval": eval_results,
        "scenarios": [asdict(s) for s in scenarios],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_state(
        output_dir,
        "checkpoint.pt",
        model,
        optimizer,
        args,
        history,
        step,
        int(history[-1]["epoch"]) if history else 0,
        required=True,
    )
    print("saved_metrics=%s" % (output_dir / "metrics.json"))
    print("saved_checkpoint=%s" % (output_dir / "checkpoint.pt"))
    return summary
