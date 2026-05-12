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

Baseline measured from four alphonse stock-config runs. The spread across all four is the noise floor.

- `val_avg/mae_surf_p` (best): **119.64** (3rd run); full noise band: 119.64 → 131.79 → 132.73 → 140.01 (~17% range)
- `test_avg/mae_surf_p`: **NaN** (systemic cruise overflow). Pre-fix comparator below.

### Pre-fix test comparator (excluding the one bad sample)

Alphonse (#1461, run `ztb0ri42`) reloaded the best checkpoint and manually excluded `test_geom_camber_cruise/000020.pt`. This is the canonical baseline test number for this round:

- **`test_avg/mae_surf_p_excluding_bad_sample`: 126.20**

Per-split test `mae_surf_p` (run `ztb0ri42`, best checkpoint epoch 13):

| Split | val | test (raw) | test (excluding bad sample) |
|---|---|---|---|
| `single_in_dist` | 166.27 | 143.45 | 143.45 |
| `geom_camber_rc` | 151.02 | 135.17 | 135.17 |
| `geom_camber_cruise` | 113.54 | **NaN** | **98.06** (199/200 samples) |
| `re_rand` | 129.21 | 128.13 | 128.13 |
| **avg** | **140.01** | **NaN** | **126.20** |

Note: `ztb0ri42` is the 4th baseline run (val_avg=140.01, worst of the 4). The three finite test splits are the most reliable test comparators.

All stock-config baseline runs (for noise band reference):

| W&B run | val_avg/mae_surf_p | epochs | notes |
|---|---|---|---|
| `z2ls7ol1` (alphonse, 3rd) | **119.64** | best in cohort | from cycle-2 W&B survey |
| `hqj9bt84` (alphonse, 1st) | 131.79 | 14 | canonical baseline |
| `89653mip` (alphonse, 2nd) | 132.73 | 12 | backup |
| `ztb0ri42` (alphonse, 4th) | 140.01 | 13 | per-split test numbers logged |

**Merge bar:** any hypothesis is a winner if `val_avg/mae_surf_p` beats 131.79 by a margin exceeding the noise band (~17% → effectively need to beat ~119 to be unambiguously out of noise). Frieren (116.34) currently leads and is outside the noise band.

### First finite test_avg (pending merge)

Thorfinn (#1480, run `5wvm7na2`) reported `test_avg/mae_surf_p = 104.96` using the `train.py:evaluate_split` cruise-NaN workaround. This PR has been sent back to add code commits — once merged it will land the workaround for the whole round and set the new baseline.

## Notes for students

- **Single-run noise floor is ~17%** — four stock-baseline runs span 119.64–140.01. A hypothesis must beat 131.79 (or ideally 119.64) to be a clean winner above noise.
- **Primary decision metric on this branch is `val_avg/mae_surf_p`** (lower is better).
- **`test_avg/mae_surf_p` will be NaN until the cruise-NaN workaround PR lands.** PR #1631 (alphonse) and #1480 (thorfinn, sent back) both target this. Once one lands, all subsequent runs get finite `test_avg` for free.
- Use `test_avg/mae_surf_p_excluding_bad_sample = 126.20` as the canonical pre-fix test comparator.
- Report all four per-test-split `mae_surf_p` values; cruise pressure will be NaN until the workaround lands.
- Per-split metrics matter for diagnosis — flag splits where your change helps or hurts disproportionately.
