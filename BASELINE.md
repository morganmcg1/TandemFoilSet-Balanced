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

## 2026-05-12 22:06 — PR #1591: Cosine schedule aligned to 30-min budget: epochs=18

**Changes merged:** `epochs: int = 18` (was 50) in `Config` dataclass — aligns cosine T_max to the realistic 30-min budget. The merged baseline ran 17 epochs with final LR ≈ 6.2e-4 (barely decayed); this change lets cosine reach ~5e-6 final LR, giving the model the low-LR weight-space refinement phase it was missing. One-line diff in `train.py`, zero-overhead change.

### Primary metrics (best val checkpoint, epoch 15 of 17)

| Metric | Value | Δ vs prev |
|---|---|---|
| **val_avg/mae_surf_p** | **125.3551** | −6.27% |
| **test_avg/mae_surf_p** | **111.9787** | **−7.67%** |

### Per-split test MAE (surface pressure)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| test_single_in_dist | 148.79 | — | — |
| test_geom_camber_rc | 117.15 | — | — |
| test_geom_camber_cruise | 77.85 | — | — |
| test_re_rand | 104.13 | — | — |

### Run info

- **W&B run:** `h7w6skh8` — group `cosine-aligned-epochs`
- **Epochs:** 17 / 18 (30-min timeout, ~106 s/epoch)
- **Final LR:** 5.32e-6 (full cosine decay confirmed)
- **Peak GPU memory:** 82.68 GB
- **Model config:** n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 (~0.67M params)

### Reproduce

```bash
cd "target/" && python train.py \
  --agent <student> \
  --wandb_name <run-name> \
  --wandb_group <group>
```

*(No `--epochs` flag needed — default is now 18. All other defaults: `lr=7e-4`, `batch_size=8`, bf16 autocast, scoring-bug workaround.)*
