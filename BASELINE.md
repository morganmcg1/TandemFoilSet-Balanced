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

### 2026-05-12 19:XX — PR #1396: Double Transolver slice tokens (slice_num 64 → 128)

- **val_avg/mae_surf_p:** 146.2510 (epoch 9 of 11 completed)
- **test_avg/mae_surf_p:** NaN ⚠️ — GT NaN in `test_geom_camber_cruise` sample 20 leaks through `err * mask` in `data/scoring.py:49`. Bug-fix PR pending; val number is valid.
- **Per-split val surface MAE (best epoch 9):**
  - `val_single_in_dist`: p=175.68, Ux=—, Uy=—
  - `val_geom_camber_rc`: p=158.18
  - `val_geom_camber_cruise`: p=115.62
  - `val_re_rand`: p=135.53
- **Test (3-split excl. cruise):** 147.07
- **W&B run:** `5qh8pj8v`
- **Peak GPU:** 54.5 GB | **Sec/epoch:** ~172s | **Epochs:** 11/50 (30-min cap)
- **Model diff vs original baseline:** `slice_num=128` (was 64); all other config unchanged.
- **Reproduce:**
  ```bash
  cd target
  python train.py --wandb_name willow-r4-frieren-slice128 --agent willowpai2g24h4-frieren
  ```

**Next target:** beat val_avg/mae_surf_p = 146.2510
