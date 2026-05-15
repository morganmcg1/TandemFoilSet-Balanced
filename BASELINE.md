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

*NaN due to data corruption — fixed in PR #3309 (see entry below).

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

---

## 2026-05-15 17:00 — PR #3309: Bugfix: prevent inf*0=NaN in evaluate_split (cruise test fix)

- **Student:** willowpai2i48h1-thorfinn
- **Branch:** `thorfinn/nanbug-fix`
- **W&B run:** `g48284pc`
- **Epochs:** 12/14 best (30-min cap, model unchanged from PR #3159)
- **Type:** Infrastructure bugfix — val unchanged (within noise), test_avg now valid

### Validation metrics (same model as PR #3159)

| Split | mae_surf_p |
|-------|-----------|
| **val_avg/mae_surf_p** | **112.8295** |
| val_single_in_dist | 142.4737 |
| val_geom_camber_rc | 133.6949 |
| val_geom_camber_cruise | 77.0254 |
| val_re_rand | 98.1238 |

### Test metrics (all 4 splits now valid — cruise NaN fixed)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| test_single_in_dist | 129.2485 | — | — |
| test_geom_camber_rc | 118.9903 | — | — |
| test_geom_camber_cruise | **83.4377** ← was NaN | — | — |
| test_re_rand | 94.7221 | — | — |
| **test_avg (all 4 splits)** | **106.5996** | — | — |

### Fix applied
In `train.py:evaluate_split`, 4 lines added after `mask = mask.to(device)`:
```python
_y_fin = torch.isfinite(y).all(dim=-1)  # [B, N]
if not _y_fin.all():
    y = torch.where(_y_fin.unsqueeze(-1), y, torch.zeros_like(y))
    mask = mask & _y_fin
```

### Reproduce
```bash
cd target/ && python train.py --agent willowpai2i48h1-thorfinn \
  --wandb_name "willowpai2i48h1-thorfinn/nanbug_fix" \
  --wandb_group nanbug_fix
```
