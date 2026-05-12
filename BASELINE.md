# Baseline metrics (willow-pai2g-24h-r4)

**Branch:** `icml-appendix-willow-pai2g-24h-r4`
**Run cap:** `SENPAI_TIMEOUT_MINUTES=30` per training run, hard.

## Baseline config (`train.py` defaults)

| | |
|---|---|
| Optimizer | AdamW |
| LR | 5e-4 |
| Weight decay | 1e-4 |
| Batch size | 4 |
| Epochs | 50 (capped at 30 min wall) |
| Scheduler | CosineAnnealingLR(T_max=epochs) |
| Loss | MSE, `vol_loss + 10 * surf_loss` |
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |

## Primary ranking metric

`test_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four test splits. **Lower is better.**

Validation analogue (used for checkpoint selection): `val_avg/mae_surf_p`.

## Current best

No experiments have been merged into this branch yet — the first run that beats the baseline above sets the floor.
