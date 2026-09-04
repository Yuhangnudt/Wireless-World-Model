#!/usr/bin/env python
"""Strict entry point for the current journal-paper WWM protocol.

This wrapper keeps operational options (dataset/output/device/checkpoint paths)
from :mod:`wwm.cli` but always applies :mod:`wwm.wwm_config`.  Use
``--train-stage wwm`` for a downstream forecasting head; the default stage is
multimodal JEPA pretraining.
"""
from __future__ import annotations

import json
import sys

from wwm.cli import parse_args
from wwm.engine import train, train_point_dvae, train_wwm_pretrain
from wwm.wwm_config import apply_paper_protocol, paper_summary


def main() -> None:
    argv = list(sys.argv[1:])
    if "--train-stage" not in argv:
        argv.extend(["--train-stage", "wwm_pretrain"])
    argv.append("--paper-protocol")
    args = apply_paper_protocol(parse_args(argv))
    if args.dry_run:
        print(json.dumps(paper_summary(args), indent=2, sort_keys=True, default=str))
        return
    if args.train_stage == "point_dvae":
        train_point_dvae(args)
    elif args.train_stage == "wwm_pretrain":
        train_wwm_pretrain(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
