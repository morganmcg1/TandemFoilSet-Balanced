# Baseline — `icml-appendix-charlie-pai2d-r1`

The baseline is the default Transolver in `train.py` on the current advisor branch.

## Configuration
- Transolver: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW (`lr=5e-4`, `weight_decay=1e-4`)
- Loss: `MSE_vol + 10.0 * MSE_surf` (normalized space)
- Schedule: cosine annealing over `epochs`
- Batch: `batch_size=4`, balanced-domain weighted sampling

## Primary ranking metric
- `val_avg/mae_surf_p` — mean surface-pressure MAE across the four val splits (lower is better)
- `test_avg/mae_surf_p` — final paper-facing number

## Status
- Round 1 (8 hypotheses) is establishing baseline numbers on this branch.
- This file will be updated with the best `val_avg/mae_surf_p` (and per-split and test numbers) after Round 1 reviews.
