"""Protocol constants for the journal-paper Wireless World Model.

The training engine intentionally exposes many switches because it is also used
for ablations.  This module is the small, auditable layer used by the public
reproduction entry point: values stated in the paper are applied in one place
and validated before a run starts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PaperProtocol:
    """Reference architecture and preprocessing for the released WWM-V2 run."""

    # Observation protocol (Sec. II and Sec. IV-B).
    context_steps: int = 16
    future_steps: int = 4
    csi_channels: int = 2
    csi_time_steps: int = 20
    csi_subbands: int = 32
    csi_spatial: int = 32
    csi_patch_t: int = 2
    csi_patch_h: int = 4
    csi_patch_w: int = 4

    # Multimodal tokenization (Sec. III-D).
    latent_dim: int = 768
    mmoe_layers: int = 16
    predictor_layers: int = 16
    mmoe_heads: int = 12
    ffn_mult: int = 4
    point_count: int = 1024
    point_tokens: int = 256
    point_group_size: int = 32
    point_center_sampling: str = "fps"
    point_tokenizer: str = "pointbert_dvae"
    point_dvae_token_source: str = "refined"
    trajectory_features: str = "pos"

    # CSI processing (Sec. IV-B, Eq. 55).
    csi_transform: str = "signed_log"
    signed_log_eps: float = 1.0
    point_normalization: str = "unit_sphere"
    csi_context_rms_normalization: bool = True
    context_rms_feature: bool = True

    # Objective choices stated by the paper.  The scalar defaults below follow
    # the reference recipe; the paper itself specifies the terms and schedules
    # but omits a complete hyper-parameter table.
    jepa_mode: str = "sigreg"
    modality_layernorm: str = "per_modality"
    deep_supervision_layers: int = 4
    deep_layer_selection: str = "uniform"
    deep_supervision_weight: float = 0.1
    context_loss_weight: float = 0.1
    pretrain_recon_weight: float = 1.0
    sigreg_enable: bool = True
    sigreg_num_projections: int = 256
    sigreg_weight_csi: float = 2e-5
    sigreg_weight_point: float = 3e-5
    sigreg_weight_traj: float = 5e-4

    # The released V2 checkpoint uses the multimodal objective without the
    # optional EM branch. The switches remain available for declared ablations.
    em_physics_enable: bool = False
    em_kernel: str = "doppler"
    em_kernel_time_basis: str = "center"
    em_kvisreg_enable: bool = False
    em_kvisreg_weight: float = 0.0
    em_kvisreg_projections: int = 256
    em_kvisreg_time_weight: float = 0.0015826554124971457
    em_relation_weight: float = 0.0
    em_tangent_enable: bool = False
    em_tangent_weight: float = 0.0
    em_modal_enable: bool = False
    em_modal_weight: float = 0.0
    em_modal_domain: str = "lag"

    # The paper calls the downstream temporal readout lightweight.  The
    # Eight-layer decoder in the released V2 configuration.
    decoder_layers: int = 8
    decoder_heads: int = 12
    decoder_ffn_mult: int = 4

    # Optimisation defaults used by the reference training recipe.
    batch_size: int = 2
    global_batch_size: int = 128
    epochs: int = 30
    start_lr: float = 1e-6
    lr: float = 2e-5
    final_lr: float = 1e-6
    weight_decay: float = 0.04
    warmup_epochs: float = 2.0
    ema_decay: float = 0.9925
    seed: int = 42

    @property
    def csi_tokens(self) -> int:
        return (self.csi_time_steps // self.csi_patch_t) * (
            self.csi_subbands // self.csi_patch_h
        ) * (self.csi_spatial // self.csi_patch_w)

    @property
    def total_tokens(self) -> int:
        # Trajectory tokenizer emits one token per frame.
        return self.csi_tokens + self.point_tokens + self.csi_time_steps

    def as_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values.update(
            {
                "csi_tokens": self.csi_tokens,
                "total_tokens": self.total_tokens,
                "paper_source": "bare_jrnl.pdf",
            }
        )
        return values


PAPER_PROTOCOL = PaperProtocol()


def apply_paper_protocol(args: argparse.Namespace) -> argparse.Namespace:
    """Apply fixed paper fields to an existing CLI namespace.

    Runtime paths, worker counts, device selection, checkpoint paths and
    evaluation switches remain caller-controlled.  All architectural and data
    protocol fields are overwritten so a reproduction command cannot silently
    instantiate the older 10-layer/768-width experimental model.
    """

    fixed = PAPER_PROTOCOL.as_dict()
    fixed.pop("csi_tokens", None)
    fixed.pop("total_tokens", None)
    fixed.pop("paper_source", None)
    # Dataclass names differ from the historical CLI names for patch sizes.
    fixed.update(
        {
            "patch_t": fixed.pop("csi_patch_t"),
            "patch_h": fixed.pop("csi_patch_h"),
            "patch_w": fixed.pop("csi_patch_w"),
            "point_dvae_hard": False,
            "freeze_point_dvae": True,
            "point_center_encoding": True,
            "csi_normalization_scope": "global",
            "drop_zero_csi_samples": True,
            "decoder_token_input": "all_csi",
            "encoder_visible_mode": "compact",
            "deep_context_fusion": "mlp",
            "lr_schedule": "cosine",
            "em_relation_centered": True,
            "em_apply_on": "predictor",
        }
    )
    for name, value in fixed.items():
        if hasattr(args, name):
            setattr(args, name, value)
    validate_paper_args(args)
    return args


def validate_paper_args(args: argparse.Namespace) -> None:
    """Fail early when a namespace no longer describes the paper protocol."""

    expected = {
        "context_steps": PAPER_PROTOCOL.context_steps,
        "future_steps": PAPER_PROTOCOL.future_steps,
        "latent_dim": PAPER_PROTOCOL.latent_dim,
        "mmoe_layers": PAPER_PROTOCOL.mmoe_layers,
        "predictor_layers": PAPER_PROTOCOL.predictor_layers,
        "mmoe_heads": PAPER_PROTOCOL.mmoe_heads,
        "ffn_mult": PAPER_PROTOCOL.ffn_mult,
        "patch_t": PAPER_PROTOCOL.csi_patch_t,
        "patch_h": PAPER_PROTOCOL.csi_patch_h,
        "patch_w": PAPER_PROTOCOL.csi_patch_w,
        "point_tokens": PAPER_PROTOCOL.point_tokens,
        "signed_log_eps": PAPER_PROTOCOL.signed_log_eps,
        "em_physics_enable": PAPER_PROTOCOL.em_physics_enable,
        "em_kvisreg_enable": PAPER_PROTOCOL.em_kvisreg_enable,
        "em_kvisreg_projections": PAPER_PROTOCOL.em_kvisreg_projections,
        "em_modal_domain": PAPER_PROTOCOL.em_modal_domain,
    }
    mismatches = {
        name: (getattr(args, name, None), value)
        for name, value in expected.items()
        if hasattr(args, name) and getattr(args, name) != value
    }
    if mismatches:
        details = ", ".join(
            "%s=%r (expected %r)" % (name, actual, expected_value)
            for name, (actual, expected_value) in mismatches.items()
        )
        raise ValueError("Paper protocol mismatch: %s" % details)


def paper_summary(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a JSON-serialisable summary for dry-runs and run manifests."""

    validate_paper_args(args)
    summary = PAPER_PROTOCOL.as_dict()
    summary["dataset_root"] = str(getattr(args, "dataset_root", ""))
    summary["output_dir"] = str(getattr(args, "output_dir", ""))
    summary["train_stage"] = str(getattr(args, "train_stage", ""))
    summary["point_dvae_resume"] = getattr(args, "point_dvae_resume", None)
    return summary
