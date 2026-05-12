# TandemFoilSet Baseline

## 2026-05-12 20:10 — PR #1391: BF16 + batch 8: more epochs within 30-min cap via AMP

**Changes merged:** bf16 autocast on training forward+loss, `batch_size=8`, `lr=7e-4` (√2 scaled), fp32 eval kept; scoring bug workaround in `evaluate_split` for `test_geom_camber_cruise/000020` (761 inf values in ground-truth pressure y — skip non-finite-y samples before accumulation).

### Primary metrics (best val checkpoint, epoch 17)

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **133.7491** |
| **test_avg/mae_surf_p** | **121.2830** |

### Per-split test MAE (surface pressure)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---|---|---|---|
| test_single_in_dist | 166.1911 | 2.0391 | 0.9130 | 171.1126 |
| test_geom_camber_rc | 136.1980 | 3.1547 | 1.1068 | 133.5701 |
| test_geom_camber_cruise | 78.5697 | 1.2584 | 0.5572 | 78.2832 |
| test_re_rand | 104.1732 | 1.7889 | 0.8165 | 103.6310 |

### Run info

- **W&B run:** `s8kl6dza` — group `bf16-batch-8`
- **Epochs:** 17 / 50 (30-min timeout, ~107 s/epoch)
- **Peak GPU memory:** 65.9 GB
- **Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.67M params)

### Reproduce

```bash
cd "target/" && python train.py \
  --epochs 50 \
  --agent <student> \
  --wandb_name <run-name> \
  --wandb_group <group>
```

*(Config defaults in `train.py` now include `lr=7e-4`, `batch_size=8`, bf16 autocast, and the scoring-bug workaround — no extra flags needed.)*
