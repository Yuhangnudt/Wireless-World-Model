# Local checkpoints

The paper-code release does not provide trained weights. Generate a Point-BERT
dVAE tokenizer and the WWM backbone locally, then pass their paths to
`train_wwm.py` with `--point-dvae-resume` and `--resume`. Some internal
workspaces may show legacy hard-linked `.pt` files from the simulation-results
folder; those artifacts are not part of this paper protocol and are ignored by
Git.

Model files are ignored by Git (`*.pt`, `*.pth`, `*.ckpt`) because they are
large, data-dependent artifacts and no trained weights are provided with this
release. Record the resolved paper configuration and a SHA-256 digest next to
any local artifact used for a reported result.
