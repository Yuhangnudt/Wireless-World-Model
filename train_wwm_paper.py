#!/usr/bin/env python
"""Paper-aligned Wireless World Model training — entry point.

Thin dispatcher over the wwm/ package (refactored from the former single-file
monolith). Stages:
  --train-stage point_dvae     -> Point-BERT dVAE pretraining
  --train-stage wwm_pretrain   -> JEPA multimodal pretraining
  --train-stage wwm (default)  -> downstream CSI forecasting (frozen backbone)

Pass --paper-protocol to lock the namespace to the current journal-paper
architecture and preprocessing. train_paper.py enables this lock by default.

Network/method changes over baseline WWM (see TECHNICAL_DOC.md):
  A) per-modality LayerNorm in the MoE transformer (shared attention kept)
  B) per-modality SIGReg regularizer (EMA retained)
"""
from __future__ import annotations

import json

from wwm.cli import parse_args
from wwm.engine import train, train_wwm_pretrain, train_point_dvae
from wwm.paper_config import apply_paper_protocol, paper_summary


def main() -> None:
    parsed = parse_args()
    if parsed.paper_protocol:
        parsed = apply_paper_protocol(parsed)
    if parsed.dry_run:
        payload = paper_summary(parsed) if parsed.paper_protocol else vars(parsed)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if parsed.train_stage == "point_dvae":
        train_point_dvae(parsed)
    elif parsed.train_stage == "wwm_pretrain":
        train_wwm_pretrain(parsed)
    else:
        train(parsed)


if __name__ == "__main__":
    main()
