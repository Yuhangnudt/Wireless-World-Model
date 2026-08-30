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



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent
    default_dataset_root = data_dir / "outputs" / "dataset"
    default_output_dir = data_dir / "outputs" / "training_runs" / "wwm_paper_aligned"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-stage", choices=("wwm", "wwm_pretrain", "point_dvae"), default="wwm")
    p.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Lock architecture and preprocessing to the current journal-paper protocol.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and exit before touching the dataset.",
    )
    p.add_argument("--dataset-root", default=str(default_dataset_root))
    p.add_argument("--city-root", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--extra-train-splits", nargs="*", default=[],
                   help="Additional splits used only by wwm_pretrain (for example held-out 60 km/h files).")
    p.add_argument("--train-speeds", nargs="*", type=float, default=[],
                   help="Optional allowed speed list applied to every pretraining split.")
    p.add_argument("--output-dir", default=str(default_output_dir))
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-samples-per-file", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--global-batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--limit-steps", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=False)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--context-steps", type=int, default=16)
    p.add_argument("--future-steps", type=int, default=4)
    p.add_argument("--point-count", type=int, default=1024)
    p.add_argument("--point-pool-count", type=int, default=8192)
    p.add_argument("--point-pool-mode", choices=("full", "random_pool"), default="full")
    p.add_argument("--point-normalization", choices=("unit_sphere", "fixed"), default="unit_sphere")
    p.add_argument("--point-tokens", type=int, default=256)
    p.add_argument("--point-group-size", type=int, default=32)
    p.add_argument("--point-center-sampling", choices=("fps", "linspace"), default="fps")
    p.add_argument("--point-tokenizer", choices=("pointbert_dvae", "pointnet"), default="pointbert_dvae")
    p.add_argument("--point-dvae-encoder-dims", type=int, default=256)
    p.add_argument("--point-dvae-codebook-size", type=int, default=8192)
    p.add_argument("--point-dvae-codebook-dim", type=int, default=256)
    p.add_argument("--point-dvae-decoder-dims", type=int, default=256)
    p.add_argument("--point-dvae-temperature", type=float, default=0.0625)
    p.add_argument("--point-dvae-hard", action="store_true")
    p.add_argument("--point-dvae-token-source", choices=("refined", "sampled", "encoder"), default="refined")
    p.add_argument("--point-center-encoding", dest="point_center_encoding", action="store_true", default=True)
    p.add_argument("--no-point-center-encoding", dest="point_center_encoding", action="store_false")
    p.add_argument("--freeze-point-dvae", dest="freeze_point_dvae", action="store_true", default=True)
    p.add_argument("--train-point-dvae-with-wwm", dest="freeze_point_dvae", action="store_false")
    p.add_argument("--point-dvae-resume", default=None)
    p.add_argument("--point-dvae-strict-resume", dest="point_dvae_strict_resume", action="store_true", default=True)
    p.add_argument("--no-point-dvae-strict-resume", dest="point_dvae_strict_resume", action="store_false")
    p.add_argument("--allow-random-point-dvae", action="store_true")
    p.add_argument("--point-dvae-data", choices=("wwm", "shapenet55"), default="wwm")
    p.add_argument("--point-group-cache", action="store_true")
    p.add_argument("--point-group-cache-build", action="store_true")
    p.add_argument("--point-group-cache-dir", default=None)
    p.add_argument("--point-group-cache-float16", action="store_true")
    p.add_argument("--point-group-cache-log-every", type=int, default=500)
    p.add_argument("--shapenet55-root", default=str(data_dir / "reference_repos" / "Point-BERT" / "data"))
    p.add_argument("--shapenet55-subset", choices=("train", "test"), default="train")
    p.add_argument("--shapenet55-npoints", type=int, default=1024)
    p.add_argument("--shapenet55-whole", action="store_true")
    p.add_argument("--pointbert-official-schedule", dest="pointbert_official_schedule", action="store_true", default=True)
    p.add_argument("--no-pointbert-official-schedule", dest="pointbert_official_schedule", action="store_false")
    p.add_argument("--pointbert-official-optim", dest="pointbert_official_optim", action="store_true", default=True)
    p.add_argument("--no-pointbert-official-optim", dest="pointbert_official_optim", action="store_false")
    p.add_argument("--point-dvae-start-lr", type=float, default=1e-7)
    p.add_argument("--point-dvae-lr", type=float, default=1e-7)
    p.add_argument("--point-dvae-final-lr", type=float, default=1e-7)
    p.add_argument("--point-dvae-adaptive-lr", action="store_true")
    p.add_argument("--point-dvae-adaptive-lr-monitor", choices=("loss", "recon_loss"), default="recon_loss")
    p.add_argument("--point-dvae-adaptive-lr-window-steps", type=int, default=500)
    p.add_argument("--point-dvae-adaptive-lr-patience", type=int, default=2)
    p.add_argument("--point-dvae-adaptive-lr-factor", type=float, default=0.5)
    p.add_argument("--point-dvae-adaptive-lr-threshold", type=float, default=0.002)
    p.add_argument("--point-dvae-adaptive-min-lr", type=float, default=1e-7)
    p.add_argument("--point-dvae-weight-decay", type=float, default=5e-4)
    p.add_argument("--point-dvae-nonfinite-action", choices=("raise", "skip"), default="raise")
    p.add_argument("--point-dvae-logit-clip", type=float, default=0.0)
    p.add_argument("--point-dvae-freeze-bn-stats", action="store_true")
    p.add_argument("--point-dvae-temp-start", type=float, default=1.0)
    p.add_argument("--point-dvae-temp-target", type=float, default=0.0625)
    p.add_argument("--point-dvae-temp-steps", type=int, default=100000)
    p.add_argument("--point-dvae-kld-start", type=float, default=0.0)
    p.add_argument("--point-dvae-kld-target", type=float, default=0.1)
    p.add_argument("--point-dvae-kld-steps", type=int, default=100000)
    p.add_argument("--point-dvae-kld-delay", type=int, default=10000)
    p.add_argument("--trajectory-features", choices=("pos", "pos_delta"), default="pos")
    p.add_argument("--pos-scale", type=float, default=1000.0)
    p.add_argument("--point-scale", type=float, default=500.0)
    p.add_argument("--latent-dim", type=int, default=384)
    p.add_argument("--patch-t", type=int, default=1)
    p.add_argument("--patch-h", type=int, default=4)
    p.add_argument("--patch-w", type=int, default=8)
    p.add_argument("--mmoe-layers", type=int, default=12)
    p.add_argument("--predictor-layers", type=int, default=12)
    p.add_argument("--mmoe-heads", type=int, default=6)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=6)
    p.add_argument("--decoder-heads", type=int, default=6)
    p.add_argument("--decoder-ffn-mult", type=int, default=4)
    p.add_argument("--decoder-token-input", choices=("future", "all_csi"), default="all_csi")
    p.add_argument("--encoder-visible-mode", choices=("compact", "padded"), default="compact")
    # Change A: per-modality LayerNorm in the MoE transformer (shared attention kept).
    p.add_argument("--modality-layernorm", choices=("per_modality", "shared"), default="per_modality")
    p.add_argument("--temporal-anchor", choices=("none", "copy", "linear", "phasor", "coherent"), default="copy")
    p.add_argument("--anchor-delta-scale", type=float, default=0.25)
    p.add_argument("--anchor-residual-scale", type=float, default=1.0)
    p.add_argument("--zero-init-residual-decoder", dest="zero_init_residual_decoder", action="store_true", default=True)
    p.add_argument("--no-zero-init-residual-decoder", dest="zero_init_residual_decoder", action="store_false")
    p.add_argument("--start-lr", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--final-lr", type=float, default=1e-6)
    p.add_argument("--downstream-start-lr", type=float, default=1e-6)
    p.add_argument("--downstream-lr", type=float, default=1e-4)
    p.add_argument("--downstream-final-lr", type=float, default=1e-6)
    p.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--adamw-foreach", dest="adamw_foreach", action="store_true", default=None)
    p.add_argument("--no-adamw-foreach", dest="adamw_foreach", action="store_false")
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9925)
    p.add_argument("--ema-predictor-branch", action="store_true")
    # Anti-collapse mechanism for JEPA pre-training.
    #   ema    : EMA target encoder (teacher-student, baseline WWM behaviour).
    #   sigreg : NO EMA target. Target = stop-grad online encoder over the full
    #            input (SimSiam-style); SIGReg is the collapse guard and is
    #            REQUIRED (weights must be > 0). No target branch is built.
    p.add_argument("--jepa-mode", choices=("ema", "sigreg"), default="ema",
                   help="Collapse-prevention for wwm_pretrain: EMA teacher (default) or SIGReg-only (LeJEPA-style, no EMA).")
    p.add_argument("--jepa-no-target-detach", dest="jepa_no_target_detach", action="store_true", default=False,
                   help="sigreg mode only: let gradients flow through the target pass too (pure LeJEPA). Default keeps stop-grad.")
    p.add_argument("--latent-weight", type=float, default=0.5)
    p.add_argument("--pretrain-weight", type=float, default=0.25)
    p.add_argument("--csi-loss-schedule", choices=("constant", "paper"), default="paper")
    p.add_argument("--csi-loss-domain", choices=("physical", "transformed", "curriculum"), default="curriculum")
    p.add_argument("--physical-loss-start-epoch", type=int, default=5)
    p.add_argument("--physical-loss-huber-delta", type=float, default=10.0)
    p.add_argument("--physical-loss-eps", type=float, default=1e-12)
    p.add_argument("--phase-weight-mode", choices=("plain", "mask", "amplitude"), default="plain")
    p.add_argument("--phase-min-amplitude-ratio", type=float, default=0.05)
    p.add_argument("--mag-weight", type=float, default=1.0)
    p.add_argument("--phase-weight", type=float, default=0.2)
    p.add_argument("--sgcs-weight", type=float, default=0.0)
    p.add_argument("--sgcs-mode", choices=("svd", "power"), default="power")
    p.add_argument("--eval-sgcs-mode", choices=("svd", "power"), default="svd")
    p.add_argument("--train-metric-sgcs-mode", choices=("svd", "power"), default="power")
    p.add_argument("--sgcs-report", choices=("avg", "final", "h1", "h2", "h3", "h4", "t15", "t16"), default="avg")
    p.add_argument("--sgcs-loss-step", choices=("avg", "final", "h1", "h2", "h3", "h4", "t15", "t16"), default="avg")
    p.add_argument("--copy-guard-weight", type=float, default=0.0)
    p.add_argument("--residual-l2-weight", type=float, default=0.0)
    p.add_argument("--fine-mask-blocks", type=int, default=8)
    p.add_argument("--fine-mask-time-fraction", type=float, default=0.50)
    p.add_argument("--fine-mask-spatial-fraction", type=float, default=0.15)
    p.add_argument("--coarse-mask-blocks", type=int, default=2)
    p.add_argument("--coarse-mask-time-fraction", type=float, default=0.50)
    p.add_argument("--coarse-mask-spatial-fraction", type=float, default=0.70)
    # Pre-training mask-task sampling weights. The temporal task masks whole
    # timestep rows so the predictor learns cross-time inference (the ability
    # the downstream CSI temporal-prediction task needs). Default temporal
    # weight is 0.0 for backward compatibility; enable it for the fixed recipe.
    p.add_argument("--mask-fine-weight", type=float, default=1.0)
    p.add_argument("--mask-coarse-weight", type=float, default=1.0)
    p.add_argument("--mask-traj-weight", type=float, default=1.0)
    p.add_argument(
        "--mask-point-weight",
        type=float,
        default=0.0,
        help="Sampling weight for CSI+trajectory->point grounding: mask all point tokens.",
    )
    p.add_argument("--mask-temporal-weight", type=float, default=2.0)
    p.add_argument("--mask-geo-weight", type=float, default=0.0,
                   help="Sampling weight of the geometry->CSI grounding task: mask ALL CSI, "
                        "keep point+traj visible, forcing geometry to be fused into the shared "
                        "representation (helps localization/compression). Default 0 = off.")
    p.add_argument("--temporal-mask-min-rows", type=int, default=2)
    p.add_argument("--temporal-mask-max-rows", type=int, default=2)
    p.add_argument("--temporal-future-bias", type=float, default=0.75)
    # Hybrid-JEPA reconstruction: weight of the masked-position CSI reconstruction
    # loss added to the latent prediction loss during wwm_pretrain (0 = pure JEPA).
    p.add_argument("--pretrain-recon-weight", type=float, default=1.0)
    p.add_argument("--pretrain-geo-recon-weight", type=float, default=None,
                   help="Reconstruction weight used only by the all-CSI-masked geo task. "
                        "Defaults to --pretrain-recon-weight for backward compatibility; set 0 "
                        "to keep geo latent supervision without ill-posed raw-CSI reconstruction.")
    p.add_argument("--pretrain-visible-recon-weight", type=float, default=0.25,
                   help="Reconstruct visible CSI from encoder outputs so positionwise detail survives the backbone.")
    p.add_argument("--pretrain-recon-mag-weight", type=float, default=0.5)
    p.add_argument("--pretrain-recon-phase-weight", type=float, default=0.5)
    p.add_argument("--pretrain-phase-eps-ratio", type=float, default=0.05,
                   help="Physical-domain phase normalization floor as a fraction of target RMS amplitude.")
    p.add_argument("--pretrain-sgcs-weight", type=float, default=0.0,
                   help="Weight on explicit (1-SGCS) loss over head-reconstructed CSI during pretraining.")
    p.add_argument("--pretrain-sgcs-warmup-steps", type=int, default=0,
                   help="Linearly warm explicit physical-domain SGCS supervision over this many optimizer steps.")
    # Angular power spectrum auxiliary supervision. The beam downstream reads the
    # ENCODER output over visible CSI tokens, and a depth probe showed mid-layer
    # tokens are no better than the final layer (d12 test 0.7087 vs d16 0.7108),
    # so the angular resolution is lost because nothing in the pretraining
    # objective rewards it. This head forces the encoder context tokens to retain
    # the 32-beam DFT angular power spectrum that the beam label is built from.
    p.add_argument("--pretrain-angular-weight", type=float, default=0.0,
                   help="Weight on the angular power spectrum (32-beam DFT) auxiliary loss over encoder "
                        "context CSI tokens. 0 disables it.")
    p.add_argument("--pretrain-angular-warmup-steps", type=int, default=0,
                   help="Linearly warm the angular auxiliary supervision over this many optimizer steps.")
    p.add_argument("--pretrain-angular-beams", type=int, default=32,
                   help="DFT codebook size for the angular auxiliary target (must match the beam downstream).")
    # Localization root cause. point_origin == traj[:, context_steps-1, :3], i.e. the
    # localization label, and _common_centers adds it to EVERY point token. So during
    # pretraining the UE position is handed to the model for free on every step and
    # there is zero gradient pressure to extract position from CSI. Measured
    # consequence: the frozen backbone scores 40.7 m even on samples it trained on,
    # while a 1-NN fingerprint on raw CSI reaches 0.5 m median per BS.
    p.add_argument("--point-origin-dropout", type=float, default=0.0,
                   help="Per-sample probability of zeroing point_origin (and thus the "
                        "absolute-position shortcut) during pretraining. Forces the "
                        "encoder to derive position from CSI + local geometry.")
    p.add_argument("--pretrain-position-weight", type=float, default=0.0,
                   help="Weight on an explicit UE-position regression aux loss, applied "
                        "only to samples whose point_origin was dropped. 0 disables it.")
    p.add_argument("--pretrain-position-warmup-steps", type=int, default=0,
                   help="Linearly warm the position auxiliary supervision over this many steps.")
    # Point-cloud recovery (v19). --mask-point-weight alone (v18) hides every point
    # token but only supervises the recovered latents with L1 against the EMA target,
    # which measurably bought nothing downstream. These flags add an explicit,
    # physically-meaningful decoding target on the same masked slots: the vertical
    # mass distribution of the hidden cloud, the one geometry a frozen backbone
    # already predicts well (ridge R^2 0.53 vs 0.14 for BEV occupancy) and the one
    # Chamfer training never rewards.
    p.add_argument("--pretrain-height-weight", type=float, default=0.0,
                   help="Weight on the point-cloud height-distribution aux loss over PREDICTOR tokens at "
                        "fully-masked point slots. 0 disables it.")
    p.add_argument("--pretrain-height-warmup-steps", type=int, default=0,
                   help="Linearly warm the height auxiliary supervision over this many optimizer steps.")
    p.add_argument("--pretrain-height-stat-weight", type=float, default=1.0,
                   help="Relative weight of the [mean z, std z, p90 z] regression inside the height aux loss.")
    p.add_argument("--pretrain-height-edges", default="0,3,6,10,15,25",
                   help="Comma-separated height-layer edges in metres relative to the UE; L edges give L+1 layers.")
    p.add_argument("--pretrain-height-scale-m", type=float, default=20.0,
                   help="Metre scale that normalizes the height statistics target.")
    p.add_argument("--pretrain-residual-head", dest="pretrain_residual_head", action="store_true", default=False,
                   help="Head predicts residual delta over the previous-visible-timestep CSI patch (anchor); "
                        "pred = anchor + head(delta). Exploits copy-anchor ~0.91 as free starting point.")
    p.add_argument("--no-rezero-residual-head", dest="no_rezero_residual_head", action="store_true", default=False,
                   help="When resuming a checkpoint whose pretrain_csi_head was ALREADY residual-trained, keep its "
                        "weights instead of re-zeroing (re-zero is only needed when warm-starting from an absolute head).")
    p.add_argument("--pretrain-freeze-encoder", dest="pretrain_freeze_encoder", action="store_true", default=False,
                   help="Predictor-focused fine-tune: freeze online_stem + online_encoder (already converged) and "
                        "train ONLY the predictor (+ its heads). The sigreg stop-grad target then becomes a fixed "
                        "strong teacher, so the predictor chases a stationary target instead of a moving one.")
    # Change B: per-modality SIGReg (LeJEPA) anti-collapse regularizer (EMA retained).
    p.add_argument("--sigreg-enable", dest="sigreg_enable", action="store_true", default=False,
                   help="Enable per-modality SIGReg during wwm_pretrain (additive to the JEPA loss).")
    p.add_argument("--sigreg-apply-on", choices=("encoder_out", "predictor_out"), default="encoder_out",
                   help="Apply SIGReg to the online-encoder output (visible tokens) or the predictor output (all tokens).")
    p.add_argument("--sigreg-num-projections", type=int, default=512,
                   help="Number of random 1-D projections per SIGReg call (Cramér–Wold).")
    p.add_argument("--sigreg-weight-csi", type=float, default=0.0)
    p.add_argument("--sigreg-weight-point", type=float, default=0.0)
    p.add_argument("--sigreg-weight-traj", type=float, default=0.0)
    p.add_argument("--visreg-enable", action="store_true",
                   help="Enable pooled per-modality VISReg on disposable projection heads.")
    p.add_argument("--visreg-projection-dim", type=int, default=128)
    p.add_argument("--visreg-num-slices", type=int, default=128)
    p.add_argument("--visreg-scale-weight", type=float, default=1.0)
    p.add_argument("--visreg-shape-weight", type=float, default=1.0)
    p.add_argument("--visreg-center-weight", type=float, default=1.0)
    p.add_argument("--visreg-weight-csi", type=float, default=0.0)
    p.add_argument("--visreg-weight-point", type=float, default=0.0)
    # v22: within-sample point-token spread. The predictor emits near-identical
    # tokens inside a sample (within/between ratio 0.12 against a target of 2.95),
    # which blocks generative point-cloud decoding. See _point_spread_pretrain.
    p.add_argument("--point-spread-weight", type=float, default=0.0)
    p.add_argument("--point-spread-surplus-weight", type=float, default=0.25,
                   help="relative penalty for exceeding the target spread; deficits always cost 1.0")
    p.add_argument("--point-spread-share-encoder", dest="point_spread_predictor_only",
                   action="store_false",
                   help="let the spread term reach the shared encoder. The first v22 attempt did "
                        "this and collapsed the CSI branch (repr_cos_csi 0.135->0.568 in 80 steps); "
                        "the default detaches each sample's mean token instead.")
    p.set_defaults(point_spread_predictor_only=True)
    p.add_argument("--visreg-weight-traj", type=float, default=0.0)
    # V-JEPA 2.1 context loss: align encoder visible-token latents to the EMA target
    # (dense-detail preservation). loss = loss_pred + context_loss_weight * loss_context.
    p.add_argument("--context-loss-weight", type=float, default=0.1,
                   help="λ on the V-JEPA 2.1 context loss during wwm_pretrain (0 disables).")
    p.add_argument("--context-loss-exp", type=float, default=1.0,
                   help="Power p in |ctx-target|^p/p for the context loss (1.0 = L1, matches latent loss).")
    p.add_argument("--context-loss-warmup-steps", type=int, default=0,
                   help="Linearly warm the context coefficient to avoid early predictor copying.")
    p.add_argument("--context-loss-source", choices=("encoder", "predictor"), default="encoder",
                   help="Apply dense context alignment to visible online-encoder tokens (V-JEPA 2.1) "
                        "or retain the historical predictor-output behavior for ablation.")
    # --- EM physical constraints (v17). Off by default; see
    # 预训练指挥文档_电磁物理约束v17.md. Every switch below is inert unless
    # --em-physics-enable is given, so existing recipes (v10/v15/v16) are unaffected.
    p.add_argument("--em-physics-enable", action="store_true",
                   help="Master switch; every other --em-* is inert without it.")
    p.add_argument("--em-kernel", choices=("doppler", "identity", "shuffle"), default="doppler",
                   help="identity/shuffle are the negative controls of §10.")
    p.add_argument("--em-kernel-time-basis", choices=("center", "blockavg"), default="center",
                   help="center = kernel at tubelet centres (§3.2, keeps 30km/h alive, "
                        "spectral relstd 0.299). blockavg = mean-pool push-forward "
                        "(source-doc behaviour, relstd 0.046 at 30km/h; ablation only).")
    p.add_argument("--em-carrier-frequency-hz", type=float, default=3.5e9)
    p.add_argument("--em-sample-period-s", type=float, default=0.005)
    p.add_argument("--em-speed-scale", type=float, default=0.97,
                   help="Effective Doppler scale eta; fitted offline, do not re-fit.")
    p.add_argument("--em-kernel-energy", type=float, default=0.999,
                   help="Spectral energy retained by the truncated-KL whitener.")
    p.add_argument("--em-kernel-jitter", type=float, default=1e-4)
    p.add_argument("--em-physics-warmup-steps", type=int, default=500)
    p.add_argument("--em-apply-on", choices=("predictor", "encoder"), default="predictor",
                   help="predictor: dense, no visibility handling (source-doc behaviour). "
                        "encoder: restricted to fully-visible time groups.")
    p.add_argument("--em-context-weight-enable", action="store_true",
                   help="Physical alpha_i context weights; needs --context-loss-weight > 0.")
    p.add_argument("--em-relation-weight", type=float, default=0.0,
                   help="Gram matching ||G_Z - K_c||^2 weight (0 disables).")
    p.add_argument("--em-relation-centered", dest="em_relation_centered",
                   action="store_true", default=True,
                   help="Match the CENTRED kernel C_T K_c C_T (WWM_EM_METHOD §5.2). Default on: "
                        "measured on v16, 90%% of the time-pooled representation's energy is the "
                        "time-constant component, so the uncentred form mostly demands that "
                        "component be deleted (distance-to-kernel 0.883 uncentred vs 0.338 centred).")
    p.add_argument("--em-relation-uncentered", dest="em_relation_centered",
                   action="store_false",
                   help="Ablation: raw Gram vs raw kernel, as in the v17 command doc §4 snippet.")
    p.add_argument("--em-kvisreg-enable", action="store_true")
    p.add_argument("--em-kvisreg-weight", type=float, default=0.0)
    p.add_argument("--em-kvisreg-projection-dim", type=int, default=128)
    p.add_argument("--em-kvisreg-projections", type=int, default=256)
    p.add_argument("--em-kvisreg-time-weight", type=float, default=1.0)
    p.add_argument("--em-kvisreg-balance", dest="em_kvisreg_balance",
                   action="store_true", default=True,
                   help="Normalize feature/time branches by their sample counts before combining them.")
    p.add_argument("--em-kvisreg-no-balance", dest="em_kvisreg_balance",
                   action="store_false",
                   help="Ablation: reproduce the historical sample-count-imbalanced objective.")
    p.add_argument("--em-kvisreg-time-axis", dest="em_kvisreg_time_axis",
                   action="store_true", default=True)
    p.add_argument("--em-kvisreg-no-time-axis", dest="em_kvisreg_time_axis",
                   action="store_false",
                   help="Ablation: without the time axis the whitening is INVISIBLE to the "
                        "objective and K-SIGReg degenerates to plain VISReg (§3.3).")
    # Direct EM anti-collapse constraints.
    p.add_argument("--em-tangent-enable", action="store_true",
                   help="Match adjacent latent changes to trajectory/Doppler EM changes.")
    p.add_argument("--em-tangent-weight", type=float, default=0.0)
    p.add_argument("--em-tangent-vector-weight", type=float, default=0.05,
                   help="Small vector escape term; keeps nonzero gradient at exact collapse.")
    p.add_argument("--em-tangent-dim", type=int, default=32)
    p.add_argument("--em-tangent-target-scale", type=float, default=0.05)
    p.add_argument("--em-tangent-beta-r", type=float, default=1.0)
    p.add_argument("--em-tangent-beta-v", type=float, default=0.25)
    p.add_argument("--em-tangent-beta-f", type=float, default=1.0)
    p.add_argument("--em-tangent-position-scale", type=float, default=1.0)
    p.add_argument("--em-tangent-speed-scale", type=float, default=20.0)
    p.add_argument("--em-tangent-phase-scale", type=float, default=1.0)
    p.add_argument("--em-tangent-physics-mode", choices=("radial", "spread", "speed_spread"), default="radial",
                   help="Physical scale target: radial Doppler (legacy), speed, or speed plus Doppler spread.")
    p.add_argument("--em-tangent-spread-scale", type=float, default=200.0,
                   help="Doppler-spread normalization in Hz for the speed_spread tangent target.")
    p.add_argument("--em-tangent-eta", type=float, default=1.0,
                   help="Effective spread factor multiplying v/lambda; fitted from trajectory only.")
    p.add_argument("--em-tangent-delta-ksigreg-enable", action="store_true",
                   help="Apply SIGReg to adjacent latent changes to close the rank-1 direction escape.")
    p.add_argument("--em-tangent-delta-ksigreg-weight", type=float, default=0.0)
    p.add_argument("--em-tangent-delta-ksigreg-projections", type=int, default=128)
    p.add_argument("--em-tangent-delta-whiten", action="store_true",
                   help="Whiten delta features before delta-K-SIGReg (scale is still set by Tangent).")
    p.add_argument("--em-direct-apply-on", choices=("predictor", "context", "both"), default="predictor",
                   help="Apply direct EM constraints to predictor, fused context, or both.")
    p.add_argument("--em-direct-multilevel", action="store_true",
                   help="Also apply direct EM constraints to each V2 intermediate fusion level.")
    p.add_argument("--em-direct-level-weight", type=float, default=0.5,
                   help="Relative weight of each intermediate level when multilevel direct EM is enabled.")
    p.add_argument("--em-modal-enable", action="store_true",
                   help="Match de-DC temporal Doppler modal energy between CSI and latent.")
    p.add_argument("--em-modal-weight", type=float, default=0.0)
    p.add_argument("--em-modal-dim", type=int, default=32)
    p.add_argument("--em-modal-temperature", type=float, default=1.0)
    p.add_argument("--em-modal-domain", choices=("fft", "lag"), default="fft",
                   help="Modal target domain; lag avoids Doppler FFT aliasing (recommended).")
    p.add_argument("--em-modal-lags", type=str, default="1,2,3,4",
                   help="Comma-separated positive lags for lag-domain modal matching.")
    p.add_argument("--em-modal-smoothing", type=int, default=1,
                   help="Moving-average width for lag modal profiles.")
    p.add_argument("--em-modal-floor", type=float, default=1e-4,
                   help="Positive energy floor for stable modal probabilities and collapse gradients.")
    p.add_argument("--em-direct-shuffle", action="store_true",
                   help="Matched negative control: shuffle tangent/modal targets within speed bins.")
    p.add_argument("--em-conditional-enable", action="store_true",
                   help="Instantiate the matched low-dimensional DeltaR/conditional-K-SIGReg branch. "
                        "Keep its weights at zero for the parameter-matched A control.")
    p.add_argument("--em-conditional-dim", type=int, default=32)
    p.add_argument("--em-deltar-weight", type=float, default=0.0)
    p.add_argument("--em-deltar-lags", type=str, default="1,2,3,4,5,6,7,8")
    p.add_argument("--em-deltar-huber-delta", type=float, default=1.0)
    p.add_argument("--em-deltar-shuffle", action="store_true",
                   help="Negative control (arm D): permute the DeltaR target across samples "
                        "WITHIN each speed bin, so the label distribution per speed group is "
                        "matched but the sample<->physics correspondence is destroyed.")
    p.add_argument("--em-conditional-sigreg-weight", type=float, default=0.0)
    p.add_argument("--em-conditional-scale-weight", type=float, default=0.0)
    p.add_argument("--em-conditional-covariance-weight", type=float, default=0.0,
                   help="Weight on conditional covariance-to-identity loss in z_em after speed/rank whitening.")
    p.add_argument("--em-conditional-rank-weight", type=float, default=0.0,
                   help="Weight on the effective-rank floor for z_em within each physical condition group.")
    p.add_argument("--em-conditional-projections", type=int, default=256)
    p.add_argument("--em-rt-sidecar-root", type=str, default=None,
                   help="Root of aligned RT path sidecars. Files are resolved as split/path/<stem>_path.npy or .npz.")
    p.add_argument("--em-rt-path-weight", type=float, default=0.0,
                   help="Huber weight for privileged RT path residual distillation. Requires --em-rt-sidecar-root.")
    p.add_argument("--em-rt-path-dim", type=int, default=0,
                   help="RT path residual feature dimension; 0 infers it from the first sidecar batch.")
    p.add_argument("--deep-supervision-layers", type=int, default=4,
                   help="Number of encoder/predictor levels used for V-JEPA 2.1 supervision.")
    p.add_argument("--deep-supervision-weight", type=float, default=0.0,
                   help="Weight on masked-token auxiliary losses from intermediate predictor levels.")
    p.add_argument("--deep-layer-selection", choices=("last", "uniform"), default="last",
                   help="Select the final consecutive levels (legacy) or uniformly spaced levels, "
                        "including the final block, for V-JEPA 2.1 supervision.")
    p.add_argument("--deep-context-fusion", choices=("linear", "mlp"), default="linear",
                   help="Fuse concatenated encoder levels with the legacy identity-initialized Linear "
                        "or a residual LayerNorm-MLP.")
    p.add_argument("--deep-fusion-hidden-mult", type=float, default=1.0,
                   help="Hidden width multiplier relative to latent_dim for --deep-context-fusion mlp.")
    p.add_argument("--deep-fusion-residual-scale", type=float, default=0.1,
                   help="Initial learnable residual scale for the multi-level fusion MLP.")
    p.add_argument("--csi-transform", choices=("signed_log", "none"), default="signed_log")
    p.add_argument("--signed-log-eps", type=float, default=1.0)
    p.add_argument("--inverse-signed-log-clip", type=float, default=12.0)
    p.add_argument("--drop-zero-csi-samples", dest="drop_zero_csi_samples", action="store_true", default=True)
    p.add_argument("--keep-zero-csi-samples", dest="drop_zero_csi_samples", action="store_false")
    p.add_argument("--csi-mean", type=float, default=None)
    p.add_argument("--csi-std", type=float, default=None)
    # --- balanced/normalized dataset pipeline (training_ready_v2) ---
    p.add_argument("--csi-normalization-scope", choices=("global", "per_city"), default="global",
                   help="per_city standardizes signed-log CSI with per-city mean/std from --csi-stats-file.")
    p.add_argument("--csi-stats-file", type=str, default=None,
                   help="Path to csi_stats_per_city.json (per-city + global relative_signed_log stats).")
    p.add_argument("--csi-context-rms-normalization", dest="csi_context_rms_normalization",
                   action="store_true", default=False,
                   help="Divide each sample's CSI by its context RMS before signed-log (needs --csi-quality-index).")
    p.add_argument("--context-rms-feature", dest="context_rms_feature", action="store_true", default=False,
                   help="Append standardized log10(context_rms) as an extra trajectory feature (dim +1).")
    p.add_argument("--csi-quality-index", type=str, default=None,
                   help="Path to quality_index.npz (accepted mask, sampler_weight, context_rms).")
    p.add_argument("--csi-quality-index-split", type=str, default="train",
                   help="Split label recorded for the quality index (audit only).")
    p.add_argument("--balanced-sampling", dest="balanced_sampling", action="store_true", default=False,
                   help="WeightedRandomSampler over accepted samples using sampler_weight (needs --csi-quality-index).")
    p.add_argument("--stats-samples-per-file", type=int, default=0)
    p.add_argument("--eval-splits", nargs="*", default=["test_velocity_40_60"])
    p.add_argument("--eval-max-files", type=int, default=1)
    p.add_argument("--eval-max-samples-per-file", type=int, default=512)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--downstream-mode", choices=("paper", "finetune", "pretrain_joint"), default="paper")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--anchor-only-fast-eval", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--resume-anchor-residual-scale", type=float, default=None)
    p.add_argument("--strict-resume", dest="strict_resume", action="store_true", default=True)
    p.add_argument("--no-strict-resume", dest="strict_resume", action="store_false")
    p.add_argument("--reset-optimizer", action="store_true")
    p.add_argument("--reset-decoder-on-resume", action="store_true")
    p.add_argument("--reset-training-state", action="store_true")
    p.add_argument("--save-every-steps", type=int, default=200)
    p.add_argument("--latest-save-every-steps", type=int, default=0)
    p.add_argument("--keep-last-checkpoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--multi-gpu", action="store_true")
    p.add_argument("--gpu-ids", default="")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument(
        "--cuda-sync-backward",
        action="store_true",
        help="Synchronize once after each backward pass to avoid async CUDA races on Windows.",
    )
    p.add_argument("--checkpoint-activations", action="store_true")
    p.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args(argv)
