# Hypothesis: huber-surface-only (frieren)

## Hypothesis
Apply Huber loss only to the surface loss term; keep MSE for the volume loss term. 
The primary metric is `mae_surf_p` — surface pressure MAE. The surface loss already 
receives `surf_weight=10` upweighting. Frieren's suggestion: since Huber gains come 
from capping gradient magnitude on high-std surface pressure outliers, there's no 
benefit to applying Huber to the volume term (interior pressure/velocities have lower 
variance and MSE is fine there). Surface-only Huber at a tighter δ=1.0 should:
- Preserve MSE precision on volume (low-variance regime where MSE is optimal)
- Apply more aggressive robust clipping on surface (the high-variance regime)

**Predicted improvement:** −2 to −5 on val_avg/mae_surf_p vs 107.46.

## Instructions

### 1. Modify the loss computation in `target/train.py`

Find the loss block in the training loop (lines ~491-502 in current train.py):

```python
# CURRENT code (Huber applied to everything):
abs_err = (pred - y_norm).abs()
sq_err = torch.where(
    abs_err < cfg.huber_delta,
    0.5 * abs_err ** 2,
    cfg.huber_delta * (abs_err - 0.5 * cfg.huber_delta),
)

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

Replace with:

```python
# NEW code: MSE on volume, Huber on surface only
mse_err = (pred - y_norm) ** 2  # MSE for volume

abs_err = (pred - y_norm).abs()
huber_err = torch.where(
    abs_err < cfg.huber_delta,
    0.5 * abs_err ** 2,
    cfg.huber_delta * (abs_err - 0.5 * cfg.huber_delta),
)  # Huber for surface

vol_mask = mask & ~is_surface
surf_mask = mask & is_surface
vol_loss = (mse_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
surf_loss = (huber_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
loss = vol_loss + cfg.surf_weight * surf_loss
```

### 2. Run with tighter δ=1.0 as primary arm

```bash
cd target/ && python train.py \
    --huber_delta 1.0 \
    --wandb_group huber-surface-only \
    --wandb_name surf-huber-delta1 \
    --agent willowpai2i24h3-frieren
```

### 3. Run a second arm with δ=2.0 (same code, same delta as current baseline)

```bash
cd target/ && python train.py \
    --huber_delta 2.0 \
    --wandb_group huber-surface-only \
    --wandb_name surf-huber-delta2 \
    --agent willowpai2i24h3-frieren
```

The δ=2.0 arm isolates the surface-only change vs the δ=2.0-everywhere baseline.

## Baseline

- **val_avg/mae_surf_p:** 107.4641 (frieren PR #3248, Huber δ=2.0 applied to all terms)
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`
