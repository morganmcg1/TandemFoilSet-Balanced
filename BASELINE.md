# Baseline — `icml-appendix-willow-pai2i-24h-r1`

Primary ranking metric: **`val_avg/mae_surf_p`** (equal-weight mean surface
pressure MAE across the four validation splits).
Paper-facing metric: **`test_avg/mae_surf_p`** (same metric on the test splits,
computed at end-of-run from the best-val checkpoint).

Lower is better. Per-split diagnostic metrics (`{split}/mae_surf_{Ux,Uy,p}`,
`{split}/mae_vol_*`) are also reported in W&B for every run.

## Current baseline configuration (head of advisor branch `train.py`)

- Model: Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2
  (~570K params)
- Loss: **Charbonnier `sqrt(diff² + ε²)`, ε=1e-3** (**merged from PR #3143**),
  `loss = vol_loss + surf_weight * surf_loss`, `surf_weight=10.0`
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`
- Schedule: `SequentialLR(LinearLR warmup → CosineAnnealingLR)`,
  `warmup_epochs=3`, `eta_min=1e-6` (**merged from PR #3150**)
- Batch size: 4 (mesh-padded by `pad_collate`)
- Sampler: `WeightedRandomSampler` with `sample_weights` from `load_data` for
  balanced raceCar single / raceCar tandem / cruise tandem domain coverage
- Run budget: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`

## Current best baseline result (PR #3143 winner)

| Source | wandb run | val_avg/mae_surf_p | test 3-split avg (cruise excl.) | Notes |
|--------|-----------|--------------------|---------------------------------|-------|
| PR #3143 arm `charbonnier_eps1e-3` | lukq8jry | **98.60** | **98.03** | 50-epoch / 30-min cap; ran on pre-warmup base |

Per-split val at best epoch (charbonnier_eps1e-3):
- `val_single_in_dist`     mae_surf_p = 126.05
- `val_geom_camber_rc`     mae_surf_p = 106.97
- `val_geom_camber_cruise` mae_surf_p =  73.34
- `val_re_rand`            mae_surf_p =  88.04

Per-split partial test (cruise excluded due to NaN bug):
- `test_single_in_dist`    mae_surf_p = 115.46
- `test_geom_camber_rc`    mae_surf_p =  93.44
- `test_re_rand`           mae_surf_p =  85.18
- `test_geom_camber_cruise` mae_surf_p = NaN (cruise-Inf data bug, known)

**Composition caveat.** PR #3143 was developed off the pre-warmup base (before
PR #3150 was merged), so the 98.60 number is _Charbonnier alone_ vs the
MSE+no-warmup control (121.14). The merge into the advisor branch composes
Charbonnier with the already-merged warmup+cosine schedule. The two levers
are orthogonal (loss form vs LR schedule) so we expect at least the
Charbonnier gain to persist; the next assignment for any student that
trains a control run will let us confirm the composed baseline.

## Pre-merge history

| Source | wandb run | val_avg/mae_surf_p | Notes |
|--------|-----------|--------------------|-------|
| PR #3150 arm `lr5e-4_wu3` | sb39atyp | 125.83 | warmup+cosine merge winner (over MSE+no-warmup ~130) |
| frieren w128 #3148 | qmyih0vv | 128.46 | pre-warmup baseline-equivalent control |
| fern depth5 #3145  | 0g36hqgg | 129.07 | pre-warmup baseline-equivalent control |
| nezuko surfp1 #3149| 7d1rlw4w | 132.33 | pre-warmup baseline-equivalent control |
| edward mse_baseline #3143 | 9npuojl6 | 121.14 | within-PR MSE control, pre-warmup |

Implicit pre-warmup baseline ≈ **130 ± 3**, run-to-run variance ~3-4 units.
Warmup landed the schedule baseline at 125.83. Charbonnier compresses that
further to **98.60** by linearizing the loss for large residuals (the surface
pressure outliers near boundary layers were dominating MSE).

**Note on test_avg.** `test_avg/mae_surf_p` is currently `None` for every
run because `test_geom_camber_cruise` produces NaN due to Inf in the hidden
test GT (issues #1569 / #1567 / #3292, multi-launch known data bug). Until
that is resolved, rank by `val_avg/mae_surf_p` and use a 3-split partial
test mean (`test_avg_3splits/mae_surf_p`) as a companion paper-facing
metric.

W&B project: `wandb-applied-ai-team/senpai-v1` — research tag
`willow-pai2i-24h-r1`.

Reproduce baseline:
```
cd target/
python train.py --wandb_name baseline_default --wandb_group baseline
```
