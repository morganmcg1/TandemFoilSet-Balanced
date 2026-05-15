# TandemFoilSet Baseline — willow-pai2i-48h-r1

Advisor branch: `icml-appendix-willow-pai2i-48h-r1`  
Primary metric: `val_avg/mae_surf_p` (lower is better)

---

## 2026-05-15 14:30 — PR #3159: H1: Huber loss (delta=0.1) to align training with MAE metric

- **Student:** willowpai2i48h1-alphonse
- **Branch:** `alphonse/huber-loss-aligned`
- **W&B run:** `bpczoejx`
- **Epochs:** 14/50 (30-min wall-clock cap)

### Validation metrics (best checkpoint, epoch 14)

| Split | mae_surf_p |
|-------|-----------|
| **val_avg/mae_surf_p** | **112.9001** |
| val_single_in_dist | 134.4612 |
| val_geom_camber_rc | 143.4094 |
| val_geom_camber_cruise | 75.8516 |
| val_re_rand | 97.8785 |

### Test metrics (best checkpoint)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| test_single_in_dist | 120.1970 | 1.4079 | 0.5594 |
| test_geom_camber_rc | 134.3200 | 2.2348 | 0.7179 |
| test_geom_camber_cruise | NaN* | 0.9322 | 0.4473 |
| test_re_rand | 92.7597 | 1.3172 | 0.5779 |
| **test_avg (3/4 splits, excl. cruise)** | **115.7589** | 1.4730 | 0.5756 |

*NaN due to data corruption in `.test_geom_camber_cruise_gt/000020.pt` (761 `-inf` values in pressure channel). Fix pending in separate PR.

### Model config
- Transolver: 5 layers, hidden=128, heads=4, slice_num=64, mlp_ratio=2
- Loss: `vol_huber(delta=0.1) + 10 * surf_huber(delta=0.1)` on normalized targets
- AdamW lr=5e-4, weight_decay=1e-4, batch=4, cosine T_max=50
- Peak VRAM: 42.1 GB / 96 GB

### Reproduce
```bash
cd target/ && python train.py --agent willowpai2i48h1-alphonse \
  --wandb_name "willowpai2i48h1-alphonse/huber_delta01" \
  --wandb_group huber_loss_delta01
```
