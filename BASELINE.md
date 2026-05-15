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
- Schedule: `CosineAnnealingLR(T_max=epochs)` (no warmup)
- Batch size: 4 (mesh-padded by `pad_collate`)
- Sampler: `WeightedRandomSampler` with `sample_weights` from `load_data` for
  balanced raceCar single / raceCar tandem / cruise tandem domain coverage
- Run budget: `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`

## Round-1 status (this launch)

No merged improvements yet. Two PRs reviewed so far (#3148, #3149) — both
contained a baseline-equivalent control arm trained from scratch on this
launch's pods.

| Source | wandb run | val_avg/mae_surf_p | test_avg (avail. splits) | Notes |
|--------|-----------|--------------------|--------------------------|-------|
| PR #3148 arm `w128`  | qmyih0vv | 128.46 | rc=141.6 / sid=129.3 / re=114.4 | 50-epoch / 30-min cap, best epoch 14 |
| PR #3149 arm `surfp1` | 7d1rlw4w | 132.33 | rc=136.9 / sid=139.1 / re=122.5 | 50-epoch / 30-min cap, best epoch 13 |

Take **val_avg/mae_surf_p ≈ 130 ± 3** as the implicit round-1 baseline (mean
of the two baseline-equivalent runs above, std-dev ≈ run-to-run noise).
This anchor will be replaced with a single confirmed reference run as soon
as a merged winner appears.

**Note on test_avg.** `test_avg/mae_surf_p` is currently `None` for every
run because `test_geom_camber_cruise` produces NaN due to Inf in the hidden
test GT (see `research/EXPERIMENTS_LOG.md`). Rank by `val_avg/mae_surf_p`
until that's resolved.

W&B project: `wandb-applied-ai-team/senpai-v1` — research tag
`willow-pai2i-24h-r1`.

Reproduce baseline:
```
cd target/
python train.py --wandb_name baseline_default --wandb_group baseline
```
