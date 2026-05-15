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

- No merged improvements yet. Baseline numbers will be established by the first
  wave of student PRs; once we have a confirmed reference run we will fill in
  the table below and update it after each merged winner.

| Round | PR | Wandb run id | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|-------|----|--------------|--------------------|---------------------|-------|
| _to fill_ | – | – | – | – | – |

W&B project: `wandb-applied-ai-team/senpai-v1` — research tag
`willow-pai2i-24h-r1`.

Reproduce baseline:
```
cd target/
python train.py --wandb_name baseline_default --wandb_group baseline
```
