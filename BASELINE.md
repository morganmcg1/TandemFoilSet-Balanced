# Baseline — icml-appendix-willow-pai2g-24h-r5

This is the per-launch baseline tracker. Branch `icml-appendix-willow-pai2g-24h-r5` was cut from `icml-appendix-willow` with no prior advisor work, so the starting point is `train.py` at HEAD.

## Starting configuration (train.py HEAD)

- Model: Transolver, `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (≈ 0.8M params)
- Optimizer: AdamW (`lr=5e-4, weight_decay=1e-4`)
- Schedule: CosineAnnealingLR(T_max=epochs)
- Loss: weighted MSE in normalized space, `surf_weight=10.0` (volume+surface losses summed)
- Batch size 4, default `epochs=50`
- Per-training cap: `SENPAI_TIMEOUT_MINUTES=30` wall-clock

## Primary ranking metrics

- val: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits)
- test: `test_avg/mae_surf_p` (equal-weight mean across 4 test splits, evaluated from the best-val checkpoint)

## Best result so far

| PR | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|----|--------------------|---------------------|-------|
| — (baseline `train.py` at HEAD) | TBD from first run | TBD from first run | No prior runs on this branch yet |

Whenever a PR improves on the current best, update this row in the same commit that runs `senpai:merge-winner`.
