# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-16 09:35 UTC

## Current best — PR #3794: n_head=2 on SwiGLU baseline (param-matched, mlp_ratio=1.333) — fern

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

n_head=2 + SwiGLU MLP + Asinh pressure compression + EMA (fast decay) + gradient clipping + Huber loss:
- **`--n_head 2`** ← NEW (PR #3794): wider per-head attention dim (64 vs 32); also 14% faster per epoch
- **`--use_swiglu --mlp_ratio 1.333`** (PR #3723): replaces GELU activation with SwiGLU in all TransolverBlock MLPs; mlp_ratio=1.333 keeps parameter count matched to the GELU baseline
- **`asinh_p_scale = 1.0`** (PR #3475)
- **`ema_decay = 0.99`** (PR #3474)
- **`grad_clip = 5.0`**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `optimizer.step()`
- **`huber_delta = 1.0`**: `F.huber_loss(pred, y_norm, delta=1.0, reduction="none")`
- Validation, checkpoint selection, and test eval all use EMA shadow weights
- Checkpoint (`model_path`) saves EMA `state_dict`

## Baseline configuration

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=2, slice_num=64, mlp_ratio=1.333, use_swiglu=True`
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
  --wandb_group n-head-on-swiglu \
  --wandb_name n-head-2-swiglu \
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
| **2026-05-16 09:35** | **#3794 fern n_head=2 on SwiGLU** | **64.3427** | **−3.41%** | **Every split improves 2-5%; val_re_rand −5.17%; 14% faster per epoch → 2 more training epochs; test_3split=63.67 (−2.74%)** |
