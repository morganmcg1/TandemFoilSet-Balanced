# TandemFoilSet Baseline

**Branch:** `icml-appendix-willow-pai2i-48h-r2`
**Last updated:** 2026-05-15

## Current best — PR #3186: EMA weights (fern)

| Metric | Value | Source |
|--------|-------|--------|
| `val_avg/mae_surf_p` | **121.6850** | run `2i7tmbir` (best @ epoch 14) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | **118.2810** | run `2i7tmbir` |

Per-split validation (best @ epoch 14):

| Split | mae_surf_p | Δ vs prev baseline |
|---|---|---|
| val_single_in_dist | 147.552 | −2.83% |
| val_geom_camber_rc | 137.679 | −20.83% |
| val_geom_camber_cruise | 92.418 | −8.86% |
| val_re_rand | 109.092 | −9.38% |

Per-split test (best ckpt):

| Split | mae_surf_p | Δ vs prev baseline |
|---|---|---|
| test_single_in_dist | 124.921 | −8.50% |
| test_geom_camber_rc | 121.909 | −22.64% |
| test_geom_camber_cruise | NaN (data/scoring.py bug — `inf * 0 = NaN`) | — |
| test_re_rand | 108.013 | −9.21% |

W&B runs: `2i7tmbir` (primary), `kji1tmn4`, `no0se6tm` — all three within ±0.7 of each other.
Merged from PR #3186, student `willowpai2i48h2-fern`.

## Current best configuration

EMA (Polyak) shadow-weight averaging added on top of the baseline config:
- `ema_decay = 0.999`; EMA model updated after every `optimizer.step()`
- Validation, checkpoint selection, and test eval all use the **EMA shadow weights**, not the live model
- Checkpoint (`model_path`) saves EMA `state_dict`
- All other settings unchanged from baseline config

## Baseline configuration (before EMA)

- Model: Transolver — `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2`
- Optimizer: AdamW — `lr=5e-4, weight_decay=1e-4`
- Schedule: `CosineAnnealingLR(T_max=epochs)`
- Loss: `vol_loss + surf_weight * surf_loss` with `surf_weight=10.0`, uniform channel weighting
- Sampler: balanced 3-domain `WeightedRandomSampler`
- Batch size: 4
- Hard caps: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30` (wall clock typically binds first; ~14 epochs land before timeout)

## Reproduce

```bash
cd target/ && python train.py \
  --wandb_group ema-weights \
  --wandb_name ema-weights \
  --agent <student>
```

## History

| Date | PR | val_avg/mae_surf_p | Δ | Notes |
|---|---|---|---|---|
| 2026-05-15 (seed) | ref run `07efagec` | 136.8873 | — | askeladd baseline-w1 reference arm |
| 2026-05-15 17:30 | #3186 fern EMA | **121.6850** | **−11.10%** | All 4 val splits improve; 3 reproducible runs |

Update this file every time a PR improves on `val_avg/mae_surf_p` and is merged. Record the PR number and the new metric value with the W&B run id.
