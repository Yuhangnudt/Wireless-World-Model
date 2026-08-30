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

from .common import csi_complex_view, inverse_csi_transform
from .metrics import (sgcs_step_terms, select_sgcs, nmse_db_metric)
from .model import PaperWWM


def csi_loss_weights(args: argparse.Namespace, epoch: Optional[int] = None) -> Tuple[float, float, float]:
    if args.csi_loss_schedule == "constant":
        return float(args.mag_weight), float(args.phase_weight), float(args.sgcs_weight)
    if args.csi_loss_schedule != "paper":
        raise ValueError("Unknown csi_loss_schedule: %s" % args.csi_loss_schedule)
    if epoch is None:
        return 1.0, 0.2, 0.0
    epoch_one_based = int(epoch) + 1
    if epoch_one_based <= 10:
        return 1.0, 0.2, 0.0
    if epoch_one_based <= 15:
        return 1.0, 0.5, 0.0
    if epoch_one_based <= 20:
        return 1.0, 1.0, 0.0
    return 1.0, 0.2, 1.0


def resolve_csi_loss_domain(args: argparse.Namespace, epoch: Optional[int] = None) -> str:
    domain = str(args.csi_loss_domain)
    if domain != "curriculum":
        return domain
    if epoch is None:
        return "physical"
    return "transformed" if int(epoch) < int(args.physical_loss_start_epoch) else "physical"


def complex_csi_loss(
    pred_h: torch.Tensor,
    target_h: torch.Tensor,
    args: argparse.Namespace,
    loss_weights: Optional[Tuple[float, float, float]] = None,
    metric_mode: Optional[str] = None,
    loss_domain: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mag_weight, phase_weight, sgcs_weight = loss_weights if loss_weights is not None else csi_loss_weights(args)
    active_domain = loss_domain or resolve_csi_loss_domain(args)
    if active_domain == "physical":
        pred_physical_h = inverse_csi_transform(pred_h, args)
        target_physical_h = inverse_csi_transform(target_h, args)
        loss_scale = target_physical_h.detach().pow(2).mean().sqrt().clamp_min(float(args.physical_loss_eps))
        pred_loss_h = pred_physical_h / loss_scale
        target_loss_h = target_physical_h / loss_scale
    elif active_domain == "transformed":
        pred_loss_h = pred_h
        target_loss_h = target_h
    else:
        raise ValueError("Unknown csi_loss_domain: %s" % active_domain)
    huber_delta = float(getattr(args, "physical_loss_huber_delta", 0.0))
    if active_domain == "physical" and huber_delta > 0:
        raw_mse = F.huber_loss(pred_loss_h, target_loss_h, delta=huber_delta)
    else:
        raw_mse = F.mse_loss(pred_loss_h, target_loss_h)
    pred_c = csi_complex_view(pred_loss_h)
    target_c = csi_complex_view(target_loss_h)
    if active_domain == "physical" and huber_delta > 0:
        mag_loss = F.huber_loss(torch.abs(pred_c), torch.abs(target_c), delta=huber_delta)
    else:
        mag_loss = F.mse_loss(torch.abs(pred_c), torch.abs(target_c))
    phase_error = 1.0 - torch.cos(torch.angle(pred_c) - torch.angle(target_c))
    if active_domain == "physical" and args.phase_weight_mode != "plain":
        target_mag = torch.abs(target_c).detach()
        if args.phase_weight_mode == "mask":
            threshold = target_mag.mean().clamp_min(float(args.physical_loss_eps)) * float(args.phase_min_amplitude_ratio)
            phase_weight_tensor = (target_mag >= threshold).to(dtype=phase_error.dtype)
        elif args.phase_weight_mode == "amplitude":
            phase_weight_tensor = target_mag / target_mag.mean().clamp_min(float(args.physical_loss_eps))
        else:
            raise ValueError("Unknown phase_weight_mode: %s" % args.phase_weight_mode)
        phase_loss = (phase_error * phase_weight_tensor).sum() / phase_weight_tensor.sum().clamp_min(1.0)
        phase_active_ratio = (phase_weight_tensor > 0).float().mean()
    else:
        phase_loss = phase_error.mean()
        phase_active_ratio = pred_h.new_tensor(1.0)
    if sgcs_weight > 0:
        pred_metric = inverse_csi_transform(pred_h, args)
        target_metric = inverse_csi_transform(target_h, args)
        sgcs_steps = sgcs_step_terms(pred_metric, target_metric, mode=args.sgcs_mode)
        sgcs = select_sgcs(sgcs_steps, args.sgcs_report)
        sgcs_loss = 1.0 - select_sgcs(sgcs_steps, args.sgcs_loss_step)
    else:
        with torch.no_grad():
            pred_metric = inverse_csi_transform(pred_h.detach(), args)
            target_metric = inverse_csi_transform(target_h.detach(), args)
            sgcs_steps = sgcs_step_terms(pred_metric, target_metric, mode=metric_mode or args.eval_sgcs_mode)
            sgcs = select_sgcs(sgcs_steps, args.sgcs_report)
        sgcs_loss = pred_h.new_zeros(())
    loss = raw_mse + mag_weight * mag_loss + phase_weight * phase_loss + sgcs_weight * sgcs_loss
    terms = {
        "raw_csi_mse": raw_mse,
        "mag_loss": mag_loss,
        "phase_loss": phase_loss,
        "phase_active_ratio": phase_active_ratio,
        "sgcs": sgcs,
        "sgcs_loss": sgcs_loss,
    }
    if active_domain == "physical":
        terms["physical_loss_scale"] = loss_scale.detach()
    terms.update(sgcs_steps)
    return loss, terms


def compute_terms(
    out: Dict[str, torch.Tensor],
    h: torch.Tensor,
    args: argparse.Namespace,
    loss_weights: Optional[Tuple[float, float, float]] = None,
    metric_mode: Optional[str] = None,
    loss_domain: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    future_h = h[:, args.context_steps : args.context_steps + args.future_steps]
    recon_loss, csi_terms = complex_csi_loss(
        out["pred_h"], future_h, args, loss_weights=loss_weights, metric_mode=metric_mode, loss_domain=loss_domain
    )
    latent_loss = out["latent_loss"].mean()
    latent_weight = 0.0 if args.downstream_mode == "paper" else float(args.latent_weight)
    loss = recon_loss + latent_weight * latent_loss
    copy_guard_weight = float(getattr(args, "copy_guard_weight", 0.0))
    residual_l2_weight = float(getattr(args, "residual_l2_weight", 0.0))
    copy_sgcs = h.new_zeros(())
    copy_guard_loss = h.new_zeros(())
    residual_l2 = h.new_zeros(())
    if copy_guard_weight > 0:
        anchor_h = out.get("anchor_h")
        if anchor_h is None:
            raise ValueError("--copy-guard-weight requires --temporal-anchor copy or linear")
        active_sgcs_weight = (
            float(loss_weights[2]) if loss_weights is not None else float(csi_loss_weights(args)[2])
        )
        if active_sgcs_weight > 0:
            pred_sgcs = 1.0 - csi_terms["sgcs_loss"]
        else:
            pred_metric_guard = inverse_csi_transform(out["pred_h"], args)
            target_metric_guard = inverse_csi_transform(future_h, args)
            pred_sgcs_terms = sgcs_step_terms(pred_metric_guard, target_metric_guard, mode=args.sgcs_mode)
            pred_sgcs = select_sgcs(pred_sgcs_terms, args.sgcs_loss_step)
        with torch.no_grad():
            anchor_metric = inverse_csi_transform(anchor_h.detach(), args)
            target_metric_guard = inverse_csi_transform(future_h.detach(), args)
            anchor_sgcs_terms = sgcs_step_terms(anchor_metric, target_metric_guard, mode=args.sgcs_mode)
            copy_sgcs = select_sgcs(anchor_sgcs_terms, args.sgcs_loss_step)
        copy_guard_loss = F.relu(copy_sgcs - pred_sgcs)
        loss = loss + copy_guard_weight * copy_guard_loss
    if residual_l2_weight > 0:
        residual_h = out.get("residual_h")
        if residual_h is None:
            raise ValueError("--residual-l2-weight requires model residual output")
        residual_l2 = residual_h.square().mean()
        loss = loss + residual_l2_weight * residual_l2
    with torch.no_grad():
        pred_metric = inverse_csi_transform(out["pred_h"].detach(), args)
        target_metric = inverse_csi_transform(future_h.detach(), args)
        nmse_db = nmse_db_metric(pred_metric, target_metric)
        transformed_nmse_db = nmse_db_metric(out["pred_h"].detach(), future_h.detach())
    terms = {
        "loss": loss,
        "recon_loss": recon_loss,
        "latent_loss": latent_loss,
        "nmse_db": nmse_db,
        "transformed_nmse_db": transformed_nmse_db,
        "copy_sgcs": copy_sgcs,
        "copy_guard_loss": copy_guard_loss,
        "residual_l2": residual_l2,
    }
    terms.update(csi_terms)
    return loss, terms


def compute_anchor_only_terms(model: PaperWWM, h: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    future_h = h[:, args.context_steps : args.context_steps + args.future_steps]
    pred_h = model.build_temporal_anchor(h).to(dtype=future_h.dtype)
    recon_loss, csi_terms = complex_csi_loss(pred_h, future_h, args)
    with torch.no_grad():
        pred_metric = inverse_csi_transform(pred_h.detach(), args)
        target_metric = inverse_csi_transform(future_h.detach(), args)
        nmse_db = nmse_db_metric(pred_metric, target_metric)
        transformed_nmse_db = nmse_db_metric(pred_h.detach(), future_h.detach())
    terms = {
        "loss": recon_loss,
        "recon_loss": recon_loss,
        "latent_loss": h.new_zeros(()),
        "nmse_db": nmse_db,
        "transformed_nmse_db": transformed_nmse_db,
    }
    terms.update(csi_terms)
    return recon_loss, terms
