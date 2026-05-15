# Baseline — icml-appendix-willow-pai2i-24h-r5

This launch starts fresh on `icml-appendix-willow-pai2i-24h-r5` (branched from `icml-appendix-willow`). No prior runs from this launch have been measured yet; the first round of experiments will both probe individual moves AND establish a confirmed baseline number.

## Current baseline configuration (head of advisor branch)

Model: **Transolver** (~1.5M params)
- `n_layers=5`, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- `space_dim=2`, `fun_dim=22` (X_DIM=24 minus the 2 spatial dims)

Training:
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- Loss: MSE in normalized space, `total = vol_loss + surf_weight * surf_loss`
- No mixed precision, no grad clipping
- `epochs=50` (max), capped by `SENPAI_TIMEOUT_MINUTES=30`

Per-run limits enforced by the harness:
- `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap)
- `SENPAI_MAX_EPOCHS=50` (hard epoch cap)
- 1 GPU per student, 96 GB VRAM

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across the four validation splits. Lower is better. The paper-facing metric is `test_avg/mae_surf_p`, computed at the end of training using the best-val checkpoint.

## Baseline metrics

### 2026-05-15 15:40 — PR #3157: Grad clipping max_norm=1.0

**New best: `val_avg/mae_surf_p = 117.16`**

| Split | val mae_surf_p |
|---|---|
| val_single_in_dist | 138.19 |
| val_geom_camber_rc | 137.91 |
| val_geom_camber_cruise | 85.86 |
| val_re_rand | 106.68 |
| **val_avg** | **117.16** |

- **test_avg/mae_surf_p:** NaN (cruise test split has a bad sample — see GH issue #3292; 3-split avg excl. cruise ≈ 116.40)
- **W&B run:** `cfp7lnaq`
- **Epochs:** 14 / 50 (hit 30 min wall-clock cap)
- **Peak VRAM:** 42.1 GB
- **Change vs head:** added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
- **Reproduce:**
  ```bash
  cd target/ && python train.py \
    --lr 5e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10.0 --epochs 50 \
    --agent willowpai2i24h5-tanjiro \
    --wandb_group willow-pai2i-24h-r5-round1 \
    --wandb_name tanjiro-gradclip-1p0
  ```
- **Note:** max_norm=1.0 fired on 100% of steps (median pre-clip grad norm ≈ 45.7). Effective LR is ~45× lower than the nominal 5e-4. Next experiment will probe max_norm=10 to allow spike-only clipping.
