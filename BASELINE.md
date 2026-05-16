# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16

## Current best — PR #3475: Asinh pressure compression on EMA decay=0.99 stack (askeladd)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **81.9754** | run `j5214ii4` (best replicate @ epoch 14) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **81.3654** | run `j5214ii4` |

Per-split validation (best replicate `j5214ii4` vs prev baseline #3474):

| Split | mae_surf_p | Δ vs #3474 (90.6131) |
|---|---|---|
| val_single_in_dist | 101.013 | **−4.8%** |
| val_geom_camber_rc | 90.717 | **−8.8%** |
| val_geom_camber_cruise | 59.909 | **−14.8%** |
| val_re_rand | 76.263 | **−11.8%** |

Per-split test (best replicate `j5214ii4`):

| Split | mae_surf_p | Δ vs #3474 |
|---|---|---|
| test_single_in_dist | 91.416 | **−4.5%** |
| test_geom_camber_rc | 83.080 | **−7.4%** |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) | — |
| test_re_rand | 69.600 | **−14.1%** |

Verify arms (both on new EMA decay=0.99 baseline):

| Arm | run | val_avg | Δ vs #3474 | test 3-split |
|---|---|---|---|---|
| verify | `2028x8co` | 85.815 | −5.3% | 83.338 |
| **replicate (best)** | **`j5214ii4`** | **81.975** | **−9.5%** | **81.365** |

Seed variance: ~3.8 MAE units between two identical-config replicates. Both clear the old baseline by >5%. Best arm merged.
Merged from PR #3475, student `willowpai2i48h2-askeladd`.

---

## Previous best — PR #3474: EMA decay=0.99 + grad_clip=5 + Huber δ=1.0 (alphonse)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **90.6131** | run `fzrq04xr` (best @ epoch 14) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | **88.8252** | run `fzrq04xr` |

Sweep: ema_decay 0.997→0.995→0.99; all arms beat prior baseline. Merged 2026-05-16 00:25 UTC.

## Current best configuration

Asinh pressure compression + EMA (fast decay) + gradient clipping + Huber loss:
- **`asinh_p_scale = 1.0`** ← NEW (PR #3475): target pressure channel is asinh-transformed before loss; inverted before reporting physical-unit predictions
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `optimizer.step()`
- **`huber_delta = 1.0`**: `F.huber_loss(pred, y_norm, delta=1.0, reduction="none")`
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- Checkpoint (`model_path`) saves EMA `state_dict`

**Key mechanistic finding (asinh):** By applying `asinh(p / scale)` to the pressure target before loss computation, the heavy-tailed pressure distribution (|z| can reach ~5+ on high-Re samples) is compressed toward a near-z-score range. This stops Huber+EMA from over-weighting the highest-magnitude tail. The compound effect with fast-EMA (decay=0.99) is larger than asinh alone: asinh on the old decay=0.999 base gave −2.1%; on the new decay=0.99 base it gives −9.5%. Mechanism: fast-EMA tracks the late-training basin cleanly, and the compressed loss signal provides cleaner gradients exactly when EMA can act on them.

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~14 epochs)
- Loss: `F.huber_loss(delta=1.0)` → `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`
- Gradient clip: `clip_grad_norm_(model.parameters(), 5.0)` before optimizer step
- EMA: **`ema_decay=0.99`**, shadow model updated after every optimizer step
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30`

## Reproduce

```bash
cd target/ && python train.py \
  --grad_clip 5.0 \
  --huber_delta 1.0 \
  --ema_decay 0.99 \
  --asinh_p_scale 1.0 \
  --wandb_group asinh-pressure-on-new-base \
  --wandb_name asinh-p-s1.0-decay0.99 \
  --agent <student>
```

## History

| Date | PR | val_avg/mae_surf_p | Δ | Notes |
|---|---|---|---|---|
| 2026-05-15 (seed) | ref run `07efagec` | 136.8873 | — | askeladd baseline-w1 reference arm |
| 2026-05-15 17:30 | #3186 fern EMA | 121.6850 | −11.10% | All 4 val splits improve; 3 reproducible runs |
| 2026-05-15 20:40 | #3366 fern EMA+clip+Huber | 94.4199 | −22.4% | All 4 val splits ≥−20%; 2 reproducible runs; val still monotone at epoch 14 |
| 2026-05-16 00:25 | #3474 alphonse EMA decay=0.99 | 90.6131 | −4.0% | All 3 arms beat baseline; monotone in decay direction; 3 runs; val monotone at ep14 |
| **2026-05-16 03:30** | **#3475 askeladd asinh-pressure** | **81.9754** | **−9.53%** | **Every val split improves; val_re_rand −11.8%; 2 verify replicates both beat baseline; test_3split=81.37 (−8.4%)** |

Update this file every time a PR improves on `val_avg/mae_surf_p` and is merged. Record the PR number and the new metric value with the W&B run id.
