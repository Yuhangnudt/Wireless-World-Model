# WWM-V2 Model Card

## Summary

WWM-V2 is a multimodal wireless-world model for CSI dynamics. It encodes complex CSI, scene point tokens, and user trajectory features with a modality-aware mixture-of-experts Transformer. A JEPA predictor learns masked future latents against a stop-gradient target (with EMA available as an explicit mode), and dense multi-level fusion exposes the representation to task-specific heads.

## Architecture

| Property | Value |
| --- | --- |
| Context / forecast | 16 frames -> 4 frames |
| Latent width | 768 |
| Shared MMoE depth | 16 blocks |
| Predictor depth | 16 blocks |
| Attention heads | 12 |
| CSI patch | 2 temporal x 4 x 4 spatial |
| Point tokens | 256 (Point-BERT dVAE, 32 points/group) |
| Fusion | 4 uniformly sampled levels, residual LayerNorm-MLP |
| Normalization | signed-log CSI, context-RMS, standardized context-RMS feature |
| Regularization | SIGReg/VISReg in the released sigreg mode; EMA target is available as an explicit alternative |

The exact constructor arguments are stored in `checkpoints/wwm_v2_architecture.json`. The model-only checkpoint is FP32 and excludes optimizer tensors and transient history.

## Reference artifact

- File: `checkpoints/wwm_v2_fp32.pt`
- Training step: 4543
- SHA-256: `checkpoints/SHA256SUMS`
- Payload: `format_version`, `model_name`, `model`, `args`, `step`

## Reference results

All rows use the V2 backbone. Downstream heads are trained independently with the protocol stated in each row; no cross-version selection is used.

| Task | Protocol | Metric |
| --- | --- | ---: |
| CSI temporal forecasting | 16 -> 4, pre-training CSI head diagnostic | SGCS 0.757493 |
| CSI compression | Frozen backbone, nested 1/1024 head | SGCS 0.622897 |
| CSI compression | Frozen backbone, nested 1/512 head | SGCS 0.700703 |
| CSI compression | Frozen backbone, nested 1/256 head | SGCS 0.729671 |
| CSI compression | Frozen backbone, nested 1/128 head | SGCS 0.745488 |
| Localization | CSI + point tokens, trajectory masked | mean error 1.36 m |
| Beam selection | Type-I W1 analytical transfer protocol | top-1 0.9651 |

These values are protocol anchors copied from the retained evaluation summary. They are not a claim that one downstream head serves every task.

## Intended use

Use WWM-V2 as a research backbone for multimodal channel prediction, compression, geometry-aware localization, and beam decision studies. The code is designed for controlled experiments and explicit protocol audits.

## Data protocol

The main recipe uses the seven-city 3.5-GHz/15-kHz corpus with file-level speed-stratified train/validation assignment and an independent test root. The three-city 3.5-GHz/30-kHz/5-ms partial corpus is retained as an optional numerology-transfer set and should be indexed separately.

## Responsible reporting

Metrics depend on numerology, context horizon, quality filter, split construction, and head protocol. Report those fields with every comparison. Raw CSI and point-cloud data remain private to the project.
