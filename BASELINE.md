# Baseline — TandemFoilSet (Charlie pai2g 24h r3)

Advisor branch: `icml-appendix-charlie-pai2g-24h-r3` (base: `icml-appendix-charlie`).
Fresh research track. No prior PRs merged on this branch yet — baseline
numbers will be established from the first round of experiments.

## Reference configuration (from `target/train.py`)

Transolver (~1.5M params):

| Field | Value |
|-------|-------|
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `space_dim` / `fun_dim` / `out_dim` | 2 / 22 / 3 |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| `epochs` | 50 |
| Scheduler | CosineAnnealingLR (T_max = epochs) |
| Optimizer | AdamW |

## Hard limits

- `SENPAI_TIMEOUT_MINUTES = 30` per training execution (wall clock).
- `SENPAI_MAX_EPOCHS = 50` (cap, not target).

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the 4
val splits (`val_single_in_dist`, `val_geom_camber_rc`, `val_geom_camber_cruise`,
`val_re_rand`). **Lower is better.** Paper-facing decision metric is
`test_avg/mae_surf_p`, evaluated from the best-val checkpoint.

## Current best metrics

_None yet._ Update this section after the first winning PR is merged.

| Metric | Value | PR |
|--------|-------|----|
| `val_avg/mae_surf_p` | — | — |
| `test_avg/mae_surf_p` | — | — |
