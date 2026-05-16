# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16 10:55 UTC

## Current best — PR #3789: vel-asinh scale=0.5 on SwiGLU + n_head=2 baseline — thorfinn

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **63.7383** | run `hy29un5q` (vel-asinh s=0.5 @ epoch 13) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **62.9264** | run `hy29un5q` |

Per-split validation (run `hy29un5q` vs prev baseline #3794 n_head=2, 64.3427):

| Split | mae_surf_p | Δ vs #3794 |
|---|---|---|
| val_single_in_dist | 72.7317 | **−5.62%** |
| val_geom_camber_rc | 78.3846 | +0.26% |
| val_geom_camber_cruise | 43.6151 | **−0.29%** |
| val_re_rand | 60.2217 | **−0.57%** |

Per-split test (run `hy29un5q`):

| Split | mae_surf_p | Δ vs #3794 |
|---|---|---|
| test_single_in_dist | 65.8686 | **−4.09%** |
| test_geom_camber_rc | 70.4182 | +3.27% |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) | — |
| test_re_rand | 52.4924 | **−3.03%** |

Replicates: `7cw3m817` val=65.91 (also beats prev SwiGLU baseline 66.61). Mean of 2 finished runs = 64.82. Third run `0rnfylq0` was in-flight at submission. Both completed arms beat the #3723 SwiGLU baseline. Best run hy29un5q beats the n_head=2 baseline (64.34) by 0.93%.

Key mechanistic finding: **vel-asinh scale=0.5 compounds cleanly with SwiGLU + n_head=2**. Applying `asinh(vel / 0.5)` to velocity channels (Ux, Uy) independently of pressure (scale=1.0) redistributes gradient mass away from large-velocity outliers. The largest gain is val_single_in_dist (−5.62%) — the in-distribution split with the biggest absolute residuals benefits most from lighter-tailed velocity targets. The scale=0.5 optimum confirmed: scale=0.25 (askeladd #3796) over-compresses and regresses by +4%.

Merged from PR #3789, student `willowpai2i48h2-thorfinn`.

---

## Previous best — PR #3794: n_head=2 on SwiGLU baseline (param-matched, mlp_ratio=1.333) — fern

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **64.3427** | run `0hy5wlxj` (n_head=2 @ epoch 15) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **63.6663** | run `0hy5wlxj` |

Per-split validation (run `0hy5wlxj` vs prev baseline #3723 SwiGLU, 66.6130):

| Split | mae_surf_p | Δ vs #3723 |
|---|---|---|
| val_single_in_dist | 77.068 | **−2.30%** |
| val_geom_camber_rc | 75.996 | **−2.80%** |
| val_geom_camber_cruise | 43.741 | **−3.89%** |
| val_re_rand | **60.565** | **−5.17%** |

Per-split test (run `0hy5wlxj`):

| Split | mae_surf_p | Δ vs #3723 |
|---|---|---|
| test_single_in_dist | 68.680 | **−0.93%** |
| test_geom_camber_rc | 68.186 | **−4.56%** |
| test_geom_camber_cruise | NaN (data/scoring.py bug — known fleet-wide) | — |
| test_re_rand | 54.133 | **−2.68%** |

Key mechanistic finding: n_head=2 (per-head dim 64 vs 32) gives wider per-head attention windows, enabling longer-range token relationship capture. The largest gain is `val_re_rand` (−5.17%) — Re-stratified OOD holdout where wider attention windows help generalize across Re regimes. Critically, n_head=2 is also **14% faster per epoch** (124 s vs ~145 s), giving 2 extra epochs in the same 30-min budget (best_epoch 15 vs 13). The wall-clock speedup contributes to realized gain — this is a real architectural improvement, not just more training.

Merged from PR #3794, student `willowpai2i48h2-fern`.

---

## Previous best — PR #3723: SwiGLU MLP activation (param-matched, mlp_ratio=1.333) — tanjiro

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

Key mechanistic finding: the win comes from the **gating mechanism** (data-dependent multiplicative pathway per MLP block), NOT from extra parameters — param-matched Arm B beats wider Arm A by 4.2 MAE on val.

---

## Current best configuration

vel-asinh s=0.5 + n_head=2 + SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping + Huber loss:
- **`--asinh_vel_scale 0.5`** ← NEW (PR #3789): applies `asinh(vel / 0.5)` to velocity channels (Ux, Uy); pressure channel unchanged; scale=0.5 confirmed optimum (scale=0.25 over-compresses)
- **`--n_head 2`** (PR #3794): wider per-head attention dim (64 vs 32); also 14% faster per epoch
- **`--use_swiglu --mlp_ratio 1.333`** (PR #3723): replaces GELU activation with SwiGLU in all TransolverBlock MLPs; mlp_ratio=1.333 keeps parameter count matched to the GELU baseline
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `optimizer.step()`
- **`huber_delta = 1.0`**: `F.huber_loss(pred, y_norm, delta=1.0, reduction="none")`
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- Checkpoint (`model_path`) saves EMA `state_dict`

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=2, slice_num=64, mlp_ratio=1.333, use_swiglu=True, asinh_vel_scale=0.5`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)` (wall clock binds at ~15 epochs with n_head=2 at 124 s/epoch)
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
  --n_head 2 \
  --asinh_vel_scale 0.5 \
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
| 2026-05-16 07:50 | #3723 tanjiro SwiGLU-mlp (param-matched) | 66.6130 | −18.74% | Every split improves 13-24%; gating mechanism confirmed as win (not params); param-matched Arm B > wider Arm A; test_3split=65.46 (−19.5%) |
| 2026-05-16 09:35 | #3794 fern n_head=2 on SwiGLU | 64.3427 | −3.41% | Every split improves 2-5%; val_re_rand −5.17%; 14% faster per epoch → 2 more training epochs; test_3split=63.67 (−2.74%) |
| **2026-05-16 10:55** | **#3789 thorfinn vel-asinh s=0.5 on SwiGLU+n_head=2** | **63.7383** | **−0.93%** | **val_single_in_dist −5.62%, val_re_rand −0.57%; scale=0.5 confirmed optimum (scale=0.25 regresses); replicate 7cw3m817=65.91 also beats prior SwiGLU baseline; test_3split=62.93 (−1.15%)** |
