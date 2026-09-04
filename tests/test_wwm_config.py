from argparse import Namespace

from wwm.wwm_config import PAPER_PROTOCOL, apply_paper_protocol, validate_paper_args


def test_paper_token_geometry():
    assert PAPER_PROTOCOL.csi_tokens == 640
    assert PAPER_PROTOCOL.total_tokens == 916
    assert PAPER_PROTOCOL.context_steps + PAPER_PROTOCOL.future_steps == 20


def test_apply_paper_protocol_overwrites_legacy_architecture():
    args = Namespace(
        context_steps=14,
        future_steps=2,
        latent_dim=768,
        mmoe_layers=10,
        predictor_layers=10,
        mmoe_heads=6,
        ffn_mult=4,
        patch_t=2,
        patch_h=4,
        patch_w=4,
        point_tokens=256,
        signed_log_eps=1e-7,
    )
    apply_paper_protocol(args)
    validate_paper_args(args)
    assert args.context_steps == 16
    assert args.future_steps == 4
    assert args.latent_dim == 768
    assert args.mmoe_layers == args.predictor_layers == 16
    assert (args.patch_t, args.patch_h, args.patch_w) == (2, 4, 4)
    assert args.signed_log_eps == 1.0
