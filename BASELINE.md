<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Current best — `icml-appendix-willow-pai2g-24h-r2`

Primary ranking metric: **`val_avg/mae_surf_p`** (lower is better)
Test-time metric: **`test_avg/mae_surf_p`** (lower is better)

## Active baseline (config to beat)

Stock Transolver from `train.py` at HEAD on `icml-appendix-willow-pai2g-24h-r2`:

| Hyperparam | Value |
|---|---|
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| `epochs` (ceiling) | 50 |
| Wall clock cap | `SENPAI_TIMEOUT_MINUTES=30` |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Schedule | `CosineAnnealingLR(T_max=epochs)` |
| Optimizer | AdamW |
| Loss | MSE on normalized targets, `vol_loss + 10.0 * surf_loss` |

Reproduce: `cd target/ && python train.py --agent <name> --wandb_name "<name>/baseline"`.

## Best metrics on the willow r2 W&B project

W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`

Baseline measured from two alphonse stock-config runs (W&B `hqj9bt84`, `89653mip`); the spread between them is the noise floor for single-run comparisons.

- `val_avg/mae_surf_p`: **131.79 / 132.73** (~132, ±~0.5%)
- `test_avg/mae_surf_p`: **NaN** in both — cruise test split's pressure channel overflowed and propagated NaN through the test_avg aggregator. **This NaN is systemic: nearly every finished run in the project has NaN test_avg, including baseline.** Decisions on this branch are made on val_avg + the three finite per-test-split numbers below.

Per-split `mae_surf_p` (best of `hqj9bt84` baseline):

| Split | val | test |
|---|---|---|
| `single_in_dist` | 136.34 | 152.32 *(rerun read; may differ slightly)* |
| `geom_camber_rc` | 129.59 | TBD from per-split test logs |
| `geom_camber_cruise` | 117.71 | **NaN** (systemic) |
| `re_rand` | 121.79 / 117.71 *(varies)* | TBD from per-split test logs |

Best W&B baseline run: `hqj9bt84` (alphonse, `r2-baseline`, 14 epochs / 30.7 min / val_avg=131.79)
Backup baseline: `89653mip` (alphonse, `r2-baseline`, 12 epochs / 30.8 min / val_avg=132.73)

## Notes for students

- **Single-run noise floor is ~0.5–1%** based on the two baseline reps. A hypothesis needs to clear that bar to be a clean winner.
- **Primary decision metric on this branch is `val_avg/mae_surf_p`** (lower is better). Aim to beat ~132 by a margin larger than noise.
- **`test_avg/mae_surf_p` will be NaN for almost all runs** until someone fixes the cruise overflow upstream. Until then, decisions are made on val_avg plus the three finite per-test-split metrics (`test_single_in_dist`, `test_geom_camber_rc`, `test_re_rand`) — those three should not regress meaningfully.
- Every PR should report `val_avg/mae_surf_p` and all four per-test-split `mae_surf_p` values (even if cruise is NaN — log the NaN explicitly).
- Per-split metrics matter for diagnosis — flag splits where your change helps or hurts disproportionately.
