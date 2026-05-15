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
- Loss: MSE in normalized space, `loss = vol_loss + surf_weight * surf_loss`,
  `surf_weight=10.0`
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`
- Schedule: `SequentialLR(LinearLR warmup → CosineAnnealingLR)`,
  `warmup_epochs=3`, `eta_min=1e-6` (**merged from PR #3150**)
- Batch size: 4 (mesh-padded by `pad_collate`)
- Sampler: `WeightedRandomSampler` with `sample_weights` from `load_data` for
  balanced raceCar single / raceCar tandem / cruise tandem domain coverage
- Run budget: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`

## Current best baseline result (PR #3150 winner)

| Source | wandb run | val_avg/mae_surf_p | test 3-split avg (cruise excl.) | Notes |
|--------|-----------|--------------------|---------------------------------|-------|
| PR #3150 arm `lr5e-4_wu3` | sb39atyp | **125.83** | **122.01** | 50-epoch / 30-min cap, best epoch 13 |

Per-split val at best epoch (lr5e-4_wu3, epoch 13):
- `val_single_in_dist`     mae_surf_p = 156.63
- `val_geom_camber_rc`     mae_surf_p = 135.87
- `val_geom_camber_cruise` mae_surf_p =  94.04
- `val_re_rand`            mae_surf_p = 116.79

Per-split partial test (cruise excluded due to NaN bug):
- `test_single_in_dist`    mae_surf_p = 133.07
- `test_geom_camber_rc`    mae_surf_p = 121.81
- `test_re_rand`           mae_surf_p = 111.15
- `test_geom_camber_cruise` mae_surf_p = NaN (cruise-Inf data bug, known)

## Pre-merge history

Three baseline-equivalent control arms across closed PRs produced 128.46
(frieren w128, #3148), 129.07 (fern depth5, #3145), 132.33 (nezuko surfp1,
#3149) — an implicit pre-warmup baseline of **130 ± 3**. The merged warmup
arm at 125.83 beats this by ~3.2%, and beats its own internal no-warmup
reference (142.28, run bww3uk1z) by -11.6% — the larger within-PR gap is
partly attributable to single-seed noise on the wu0 arm but the consistent
per-split wins (warmup helps on every split) confirm the merge.

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
