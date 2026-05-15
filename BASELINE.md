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

---

## 2026-05-15 18:30 — PR #3317: H3b: Cosine T_max=15 tuned to actual epoch budget ← CURRENT BEST

- **Student:** willowpai2i48h1-askeladd
- **Branch:** `askeladd/cosine-tmax-tuned`
- **W&B run:** `kx17n4pn` (Arm A winner)
- **Epochs:** 14/50 (30-min wall-clock cap)

### Validation metrics (best checkpoint, epoch 14)

| Split | mae_surf_p |
|-------|-----------|
| **val_avg/mae_surf_p** | **91.3319** ← CURRENT BEST |
| val_single_in_dist | 108.1607 |
| val_geom_camber_rc | 98.4476 |
| val_geom_camber_cruise | 72.8700 |
| val_re_rand | 85.8493 |

### Test metrics (3/4 splits — cruise NaN, branch predates PR #3309 merge)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| test_single_in_dist | 96.7268 | 1.0136 | 0.5508 |
| test_geom_camber_rc | 88.3769 | 1.6032 | 0.7599 |
| test_geom_camber_cruise | NaN* | 0.5799 | 0.3970 |
| test_re_rand | 80.1744 | 0.9808 | 0.5792 |
| **test_avg (3/4 splits, excl. cruise)** | **88.4260** | 1.0444 | 0.5717 |

*Branch created before PR #3309 NaN fix was merged; cruise test NaN is the data corruption bug.

### Model config
- Transolver: 5 layers, hidden=128, heads=4, slice_num=64, mlp_ratio=2
- Loss: `vol_huber(delta=0.1) + 10 * surf_huber(delta=0.1)` on normalized targets
- AdamW lr=5e-4, weight_decay=1e-4, batch=4, **cosine T_max=15** ← key change
- Peak VRAM: ~78.5 GB / 96 GB

### Key insight
T_max=15 aligns the cosine LR schedule with the 14-epoch wall-clock budget. At T_max=50 the LR was only 79% decayed at training stop — effectively no annealing. At T_max=15, epoch 14 runs at ~1.1% of peak LR (fine-tuning pass). Arm B (T_max=12) scored 103.12 — LR crashed to zero at epoch 12 and left 2 epochs under-training.

### Reproduce
```bash
cd target/ && python train.py --agent willowpai2i48h1-askeladd \
  --wandb_name "willowpai2i48h1-askeladd/cosine_tmax15" \
  --wandb_group cosine_tmax_scan
```
