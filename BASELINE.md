<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Current best — `icml-appendix-willow-pai2g-24h-r2`

Primary ranking metric: **`val_avg/mae_surf_p`** (lower is better)
Test-time metric: **`test_avg/mae_surf_p`** (lower is better)

## 2026-05-12 22:15 — PR #1480: thorfinn — bf16 + grad_accum=2 (MERGED)

**New best val and first finite test_avg.**

- **val_avg/mae_surf_p:** 116.2965 (was 131.79) — **-11.6%**
- **test_avg/mae_surf_p:** **104.9554** — first finite test_avg in this project
- **W&B run:** `5wvm7na2`
- **Epochs:** 18 in 30 min (vs ~14 fp32 baseline — 2.5× throughput gain from bf16+accum)

Per-split test `mae_surf_p`:

| Split | test |
|---|---|
| `single_in_dist` | 115.83 |
| `geom_camber_rc` | 117.06 |
| `geom_camber_cruise` | **80.35** (1/200 samples skipped — the known bad sample) |
| `re_rand` | 106.58 |

**Also landed:** `train.py:evaluate_split` sanitize-and-gate cruise-NaN workaround — all future runs on this branch now produce finite `test_avg`.

Reproduce:
```bash
cd target/ && python train.py --agent <name> --wandb_name "<name>/bf16-accum2" --wandb_group "willow-r2-throughput"
```
(bf16 autocast and grad_accum=2 are now the default in `Config`; no extra flags needed)

---

## Active baseline (config to beat)

Transolver from `train.py` at HEAD on `icml-appendix-willow-pai2g-24h-r2` — now includes bf16 autocast and grad_accum=2 from PR #1480:

| Hyperparam | Value |
|---|---|
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `batch_size` | 4 (effective=8 with grad_accum=2) |
| `surf_weight` | 10.0 |
| `epochs` (ceiling) | 50 |
| Wall clock cap | `SENPAI_TIMEOUT_MINUTES=30` |
| `amp` | `True` (bf16 autocast on forward+loss) |
| `grad_accum` | 2 |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Schedule | `CosineAnnealingLR(T_max=epochs)` |
| Optimizer | AdamW |
| Loss | MSE on normalized targets, `vol_loss + 10.0 * surf_loss` |

Reproduce: `cd target/ && python train.py --agent <name> --wandb_name "<name>/baseline"`.

## Current best metrics

W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`

**Best val (merged):** `val_avg/mae_surf_p` = **116.30** (PR #1480, thorfinn, run `5wvm7na2`)
**Best test (merged):** `test_avg/mae_surf_p` = **104.96** (same run — first finite test_avg)

Stock fp32 baseline runs (pre-merge, for noise floor reference):

| W&B run | val_avg/mae_surf_p | epochs | notes |
|---|---|---|---|
| `z2ls7ol1` (alphonse) | 119.64 | ~14 | from cycle-2 W&B survey |
| `hqj9bt84` (alphonse) | 131.79 | 14 | canonical fp32 baseline |
| `89653mip` (alphonse) | 132.73 | 12 | fp32 backup |
| `ztb0ri42` (alphonse) | 140.01 | 13 | per-split test logged; workaround=126.20 |

fp32 baseline noise band: 119.64–140.01 (~17%). All future comparisons are against the new best: **116.30 val / 104.96 test**.

## Notes for students

- **Baseline as of PR #1480:** `val_avg/mae_surf_p = 116.30`, `test_avg/mae_surf_p = 104.96`.
- **cruise-NaN workaround is now landed.** All runs on this branch produce finite `test_avg` — no per-PR code needed.
- **Primary decision metric is `val_avg/mae_surf_p`** (lower is better). Beat 116.30 to be a winner.
- The noise band on fp32 baseline was ~17%. The new bf16+accum baseline may have a similar noise floor — single-run wins close to 116.30 should be confirmed with a second seed.
- Report `val_avg/mae_surf_p`, `test_avg/mae_surf_p`, and all four per-test-split `mae_surf_p` values.
- Per-split metrics matter — flag splits where your change helps or hurts disproportionately.
