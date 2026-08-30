"""Wireless World Model (WWM) — paper-aligned training package.

Refactored from the single-file train_wwm_paper.py monolith into cohesive modules:
  common   — shared helpers (init, sincos, complex/patch views, IO)
  metrics  — SGCS / NMSE
  pointbert— Point-BERT dVAE tokenizer stack
  data     – datasets, scenario discovery, signed-log stats, point-group cache
  splits   - deterministic scenario-level train/validation/test indexing
  sigreg   — per-modality SIGReg regularizer (LeJEPA)
  model    — embeddings, ModalityMoE transformer, CSI decoder, PaperWWM
  masking  — pretrain masking (fine/coarse/traj/temporal + geom2csi)
  losses   — LCSI complex loss + downstream term computation
  engine   — train/pretrain/point_dvae stages, eval, checkpointing
  cli      — argparse
See TECHNICAL_DOC.md for the network/method changes over baseline WWM.
"""
