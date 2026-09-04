# Wireless World Model (WWM)

WWM is a multimodal foundation model for wireless-channel dynamics. It learns a shared latent state from complex CSI, 3-D scene geometry, and user trajectory tokens, then reuses that state for forecasting, CSI compression, localization, and beam selection.

The released reference is the single `WWM-V2` checkpoint. Its design combines modality-aware mixture-of-experts attention, JEPA-style masked latent prediction with a stop-gradient target, dense multi-level fusion, and protocol-aware CSI/geometry supervision. The reference uses 16 context frames to forecast 4 future frames and keeps weights in FP32 for deterministic evaluation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `wwm/` | Core model, losses, data pipeline, metrics, and deterministic split indexer |
| `train_wwm.py` | Strict V2 training entry point with locked architecture defaults |
| `train_wwm_dispatch.py` | General training dispatcher for declared ablations |
| `tools/wwm_four_downstream_tasks.py` | Frozen-backbone evaluation for the four paper tasks |
| `tools/build_split_index.py` | Portable train/validation/test index and leakage audit |
| `tools/export_release_checkpoint.py` | Strip optimizer/history from a training checkpoint |
| `reproducibility/figures/` | Figure-generation scripts and compact CSV/JSON inputs |
| `checkpoints/` | V2 FP32 backbone and selected downstream heads |
| `docs/` | Training recipe, data protocol, model card, and lessons learned |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -e .
```

Install a PyTorch build matching the host CUDA runtime. The Point-BERT dVAE transfer file is an explicit training input; no random tokenizer is silently created.

## Build a deterministic index

All windows from one trajectory stay together. Files are stratified by speed and assigned to validation with a stable SHA-1 key. The test root is independent and is never used for checkpoint selection.

```bash
python tools/build_split_index.py \\
  --dataset-root /data/WWM_7city_16to4 \\
  --test-root /data/WWM_test \\
  --test-split test_gen_velocity \\
  --output-dir artifacts/index \\
  --validation-fraction 0.10 \\
  --audit-contents
```

The command writes `index.csv` (one row per sample) and `manifest.json` (counts and overlap audit). The three-city 3.5-GHz/30-kHz/5-ms dataset is an optional extension set; keep it in a separate index because its numerology differs from the main 3.5-GHz/15-kHz protocol.

## Train the reference model

```bash
python train_wwm.py \\
  --dataset-root /data/WWM_7city_16to4 \\
  --output-dir runs/wwm_v2 \\
  --point-dvae-resume checkpoints/point_dvae.pt \\
  --limit-steps 4544 --seed 42
```

`train_wwm.py` applies the V2 architecture and preprocessing defaults from `wwm/wwm_config.py`; runtime paths, device, worker count, and resume options remain caller-controlled. Use `docs/TRAINING_RECIPE.md` for the full command and downstream head protocol.

## Load the released backbone

```python
from pathlib import Path
import torch
from tools.wwm_four_downstream_tasks import load_backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, args, step = load_backbone(Path("checkpoints/wwm_v2_fp32.pt"), device)
print(args.context_steps, args.future_steps, step)
```

The payload contains `model`, `args`, and `step`; optimizer tensors and transient training history are not part of the release artifact. SHA-256 digests are listed in `checkpoints/SHA256SUMS`.

## Reproduce figures

The plotting programs in `reproducibility/figures/` write publication-ready PDF/PNG files from compact CSV/JSON summaries. Existing historical comparison assets are retained for traceability; numerical claims in `docs/MODEL_CARD.md` identify the single V2 backbone and its task protocol.

## Data and license

Raw CSI, trajectory, and point-cloud data are not redistributed. Obtain them from the project owner and keep the generated index next to the data. Add the license required by your institution or data provider before publishing a public GitHub fork.
