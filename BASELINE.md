# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16 18:40 UTC

## Current best — PR #4062: slice_num=8 — fern

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **56.8954** | run `vzpgr8us` (best epoch 18, timeout) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **55.9817** | run `vzpgr8us` |

Per-split validation (run `vzpgr8us` vs prior #3854 slice=16 baseline, 57.6953):

| Split | mae_surf_p | Δ vs #3854 |
|---|---|---|
| val_single_in_dist | 66.966 | +1.48% ⚠️ |
| val_geom_camber_rc | 70.071 | **−2.43%** |
| val_geom_camber_cruise | 35.324 | **−7.06%** |
| val_re_rand | 55.221 | +0.46% |
| **val_avg** | **56.895** | **−1.39%** |

Per-split test (run `vzpgr8us`):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 58.1525 |
| test_geom_camber_rc | 63.5973 |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) |
| test_re_rand | 46.2952 |

Key mechanistic finding: **slice_num=8 extends the winning slice axis (64→32: −3.02%, 32→16: −5.16%, 16→8: −1.39%; clearly decelerating).** With ~100 nodes per slice (vs ~50 at slice=16), each slice token aggregates a broader spatial neighborhood. The per-split signature is informative: in-distribution regresses slightly (+1.48%) while OOD-camber splits improve substantially (rc −2.43%, cruise −7.06%). Coarser slicing forces the model to learn more abstract, geometry-invariant features at the slice-aggregation stage — trades fine-grained precision for OOD generalization. Test (−1.55%) tracks val improvement closely, validating the win as paper-facing. Direction is alive but decelerating; the next datapoint should bracket toward saturation (slice=4 or slice=12).

Merged from PR #4062, student `willowpai2i48h2-fern`.

---

## Current best configuration

slice_num=8 + Huber δ=0.5 + vel-asinh s=0.5 + n_head=2 + SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping:
- **`--slice_num 8`** ← NEW (PR #4062): 8× coarser than original baseline 64; ~100 nodes per slice
- **`--huber_delta 0.5`** (PR #3854 stack): tighter quadratic transition for small residuals
- **`--asinh_vel_scale 0.5`** (PR #3789): applies `asinh(vel / 0.5)` to velocity channels (Ux, Uy); pressure unchanged
- **`--n_head 2`** (PR #3794): wider per-head attention dim (64 vs 32); also 14% faster per epoch
- **`--use_swiglu --mlp_ratio 1.333`** (PR #3723): SwiGLU in all TransolverBlock MLPs; param-count matched
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- **NO SGDR** in this run; SGDR + slice=16 + δ=0.5 super-compound is untested

## Reproduce

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 0.5 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --use_swiglu \
  --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --agent <student>
```

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=2, slice_num=8, mlp_ratio=1.333, use_swiglu=True, asinh_vel_scale=0.5`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~15 epochs with n_head=2 at ~107 s/epoch)
- Loss: `F.huber_loss(delta=0.5)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: **`ema_decay=0.99`**, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

---
