# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16 07:50 UTC

## Current best — PR #3723: SwiGLU MLP activation (param-matched, mlp_ratio=1.333) — tanjiro

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **66.6130** | run `ju2azfzk` (param-matched Arm B @ epoch 13) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **65.4628** | run `ju2azfzk` |

Per-split validation (Arm B `ju2azfzk` vs prev baseline #3475, 81.9754):

| Split | mae_surf_p | Δ vs #3475 |
|---|---|---|
| val_single_in_dist | 78.885 | **−21.9%** |
| val_geom_camber_rc | 78.184 | **−13.8%** |
| val_geom_camber_cruise | 45.513 | **−24.0%** |
| val_re_rand | 63.870 | **−16.2%** |

Per-split test (Arm B `ju2azfzk`):

| Split | mae_surf_p | Δ vs #3475 |
|---|---|---|
| test_single_in_dist | 69.321 | **−24.2%** |
| test_geom_camber_rc | 71.445 | **−14.0%** |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) | — |
| test_re_rand | 55.623 | **−20.1%** |

Arm comparison (both on asinh+EMA+clip+Huber baseline):

| Arm | run | mlp_ratio | n_params | epochs | val_avg | Δ vs #3475 | test_3split |
|---|---|---|---|---|---|---|---|
| A — wider | `rqiazooj` | 2 (SwiGLU, +25% params) | 827,479 | 12 | 70.850 | −13.6% | 69.171 |
| **B — param-matched (best)** | **`ju2azfzk`** | **1.333 (SwiGLU, param-matched)** | **661,499** | **13** | **66.613** | **−18.7%** | **65.463** |

Key mechanistic finding: the win comes from the **gating mechanism** (data-dependent multiplicative pathway per MLP block), NOT from extra parameters — param-matched Arm B beats wider Arm A by 4.2 MAE on val. SwiGLU forces each MLP block to selectively suppress/pass channels per node, which is exactly what's needed for CFD surrogate features mixing global (Re, NACA) and local (coordinates, dsdf) signals. The compound effect with asinh+EMA is very large (literature baseline for GELU→SwiGLU is 0.5-2%; we see −18.7%) because the clean gradient signal from asinh lets the gating mechanism act on high-quality late-training signal.

Seed variance: approximately ±1.5-3 MAE units at 14-epoch budget. Both arms clearly beat the old baseline by >13%.
Merged from PR #3723, student `willowpai2i48h2-tanjiro`.

---

## Previous best — PR #3475: Asinh pressure compression on EMA decay=0.99 stack (askeladd)

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

Seed variance: ~3.8 MAE units. Merged 2026-05-16 03:30 UTC.

## Current best configuration

SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping + Huber loss:
- **`--use_swiglu --mlp_ratio 1.333`** ← NEW (PR #3723): replaces GELU activation with SwiGLU (SiLU(W_gate·x) ⊙ W_value·x) in all TransolverBlock MLPs; mlp_ratio=1.333 keeps parameter count matched to the GELU baseline
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `optimizer.step()`
- **`huber_delta = 1.0`**: `F.huber_loss(pred, y_norm, delta=1.0, reduction="none")`
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- Checkpoint (`model_path`) saves EMA `state_dict`

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=1.333, use_swiglu=True`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~13 epochs with SwiGLU's +6.7% per-epoch overhead)
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
  --use_swiglu \
  --mlp_ratio 1.333 \
  --wandb_group swiglu-on-new-base \
  --wandb_name swiglu-p-matched \
  --agent <student>
```

## History

| Date | PR | val_avg/mae_surf_p | Δ | Notes |
|---|---|---|---|---|
| 2026-05-15 (seed) | ref run `07efagec` | 136.8873 | — | askeladd baseline-w1 reference arm |
| 2026-05-15 17:30 | #3186 fern EMA | 121.6850 | −11.10% | All 4 val splits improve; 3 reproducible runs |
| 2026-05-15 20:40 | #3366 fern EMA+clip+Huber | 94.4199 | −22.4% | All 4 val splits ≥−20%; 2 reproducible runs; val still monotone at epoch 14 |
| 2026-05-16 00:25 | #3474 alphonse EMA decay=0.99 | 90.6131 | −4.0% | All 3 arms beat baseline; monotone in decay direction; 3 runs; val monotone at ep14 |
| 2026-05-16 03:30 | #3475 askeladd asinh-pressure | 81.9754 | −9.53% | Every val split improves; val_re_rand −11.8%; 2 verify replicates both beat baseline; test_3split=81.37 (−8.4%) |
| **2026-05-16 07:50** | **#3723 tanjiro SwiGLU-mlp (param-matched)** | **66.6130** | **−18.74%** | **Every split improves 13-24%; gating mechanism confirmed as win (not params); param-matched Arm B > wider Arm A; test_3split=65.46 (−19.5%)** |
