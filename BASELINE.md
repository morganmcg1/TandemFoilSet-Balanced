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

- `val_avg/mae_surf_p`: **TBD** (no baseline run logged yet — alphonse is establishing it)
- `test_avg/mae_surf_p`: **TBD**

Per-split `mae_surf_p`:

| Split | val | test |
|---|---|---|
| `single_in_dist` | TBD | TBD |
| `geom_camber_rc` | TBD | TBD |
| `geom_camber_cruise` | TBD | TBD |
| `re_rand` | TBD | TBD |

Best W&B run: TBD
PR that established this baseline: TBD

## Notes for students

- Baseline metrics will be filled in once alphonse's baseline-only run completes. Until then, compare against the **published Transolver behavior**: a stock-config training in the 30-min cap typically lands `val_avg/mae_surf_p` in the 30–80 range on this dataset, but ranking your hypothesis vs baseline within the W&B project is what matters — not absolute numbers.
- Every PR should report both `val_avg/mae_surf_p` and `test_avg/mae_surf_p` (the test metric is computed automatically at the end of every training run by `train.py`).
- Per-split metrics matter for diagnosis — flag splits where your change helps or hurts disproportionately.
