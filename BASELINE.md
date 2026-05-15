# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-15

## Current best — PR #3366: EMA + grad_clip=5 + Huber δ=1.0 (fern)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **94.4199** | run `m6hkf8el` (best @ epoch 14) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | **92.3626** | run `m6hkf8el` |

Per-split validation (best @ epoch 14):

| Split | mae_surf_p | Δ vs prev baseline (EMA 121.685) |
|---|---|---|
| val_single_in_dist | 111.794 | **−24.2%** |
| val_geom_camber_rc | 110.162 | **−20.0%** |
| val_geom_camber_cruise | 69.012 | **−25.3%** |
| val_re_rand | 86.712 | **−20.5%** |

Per-split test (best ckpt):

| Split | mae_surf_p | Δ vs prev baseline |
|---|---|---|
| test_single_in_dist | 99.797 | −20.1% |
| test_geom_camber_rc | 96.252 | −21.0% |
| test_geom_camber_cruise | NaN (data/scoring.py bug — `inf * 0 = NaN`) | — |
| test_re_rand | 81.040 | −25.0% |

Reproducibility: run `eq4osquw` (replicate) = val_avg **94.868**, test 3-split **93.39** — within 0.45 of primary. Both landed at epoch 14 (wall-clock timeout). **All 4 val splits improve by ≥20%.**

W&B runs: `m6hkf8el` (primary), `eq4osquw` (replicate); smoke/debug run `afqw9g6s` excluded.
Merged from PR #3366, student `willowpai2i48h2-fern`.

## Current best configuration

EMA + gradient clipping + Huber loss on top of the EMA baseline config:
- `ema_decay = 0.999` (unchanged from PR #3186)
- **`grad_clip = 5.0`**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `optimizer.step()`
- **`huber_delta = 1.0`**: `F.huber_loss(pred, y_norm, delta=1.0, reduction="none")` replaces MSE element-wise loss
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- Checkpoint (`model_path`) saves EMA `state_dict`
- All other settings unchanged from EMA baseline config

**Key mechanistic finding:** At clip=5, the gradient norm threshold bites ~92–99% of steps (median pre-clip norm ~16–29×). Raising clip from 1 → 5 allows 5× larger effective steps; Huber caps per-sample influence from high-Re outliers; EMA smooths the noisy trajectory. The three mechanisms compound orthogonally.

## Baseline configuration (EMA + clip + Huber)

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~14 epochs)
- Loss: `F.huber_loss(delta=1.0)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: `ema_decay=0.999`, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

## Reproduce

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 1.0 \
  --wandb_group ema-grad-clip-huber \
  --wandb_name ema-grad-clip-huber \
  --agent <student>
```

## History

| Date | PR | val_avg/mae_surf_p | Δ | Notes |
|---|---|---|---|---|
| 2026-05-15 (seed) | ref run `07efagec` | 136.8873 | — | askeladd baseline-w1 reference arm |
| 2026-05-15 17:30 | #3186 fern EMA | 121.6850 | −11.10% | All 4 val splits improve; 3 reproducible runs |
| 2026-05-15 20:40 | #3366 fern EMA+clip+Huber | **94.4199** | **−22.4%** | All 4 val splits ≥−20%; 2 reproducible runs; val still monotone at epoch 14 |

Update this file every time a PR improves on `val_avg/mae_surf_p` and is merged. Record the PR number and the new metric value with the W&B run id.
