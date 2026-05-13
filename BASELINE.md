<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Current best — `icml-appendix-willow-pai2g-24h-r2`

Primary ranking metric: **`val_avg/mae_surf_p`** (lower is better)
Test-time metric: **`test_avg/mae_surf_p`** (lower is better)

## 2026-05-13 00:10 — PR #1471: frieren — p_weight=2.0 + clip_grad_norm=1.0 (MERGED)

**New best val and test.**

- **val_avg/mae_surf_p:** 110.2732 (was 116.30) — **-5.19%**
- **test_avg/mae_surf_p:** **99.4108** (was 104.96) — **-5.29%**
- **W&B run:** `krsv4c21`
- **Epochs:** 19 in 30 min (best at epoch 18)

Per-split test `mae_surf_p`:

| Split | test | vs prior baseline | Δ% |
|---|---|---|---|
| `single_in_dist` | 116.69 | 115.83 | +0.74% (within noise) |
| `geom_camber_rc` | 110.01 | 117.06 | **-6.02%** |
| `geom_camber_cruise` | **72.77** | 80.35 | **-9.43%** (1/200 samples skipped) |
| `re_rand` | 98.17 | 106.58 | **-7.89%** |

Changes vs prior baseline (#1480):
- `p_weight: float = 2.0` — per-channel pressure upweight in squared-error map (before surf/vol mask sum)
- `clip_grad_norm_(model.parameters(), max_norm=1.0)` — after `loss.backward()`, before `optimizer.step()`

Note: grad clip is binding on nearly every step (mean norm 114, max 1203). The Transolver training loop runs in a high-gradient-magnitude regime; clip acts as a step-size cap.

Reproduce:
```bash
cd target/ && python train.py --agent <name> --wandb_name "<name>/p-weight-2-clip" --wandb_group "willow-r2-p-weight"
```
(p_weight=2.0 and clip_grad_norm=1.0 are now defaults in `Config` and train loop; no extra flags needed)

---

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

Transolver from `train.py` at HEAD on `icml-appendix-willow-pai2g-24h-r2` — includes bf16 autocast (PR #1480), grad_accum=2 (PR #1480), p_weight=2.0 + clip_grad_norm=1.0 (PR #1471):

| Hyperparam | Value |
|---|---|
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `batch_size` | 4 (effective=8 with grad_accum=2) |
| `surf_weight` | 10.0 |
| `p_weight` | 2.0 (per-channel pressure upweight in loss) |
| `grad_clip` | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |
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
| Loss | `p_weight`-weighted per-channel sq_err, `vol_loss + 10.0 * surf_loss` |

Reproduce: `cd target/ && python train.py --agent <name> --wandb_name "<name>/baseline"`.

## Current best metrics

W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`

**Best val (merged):** `val_avg/mae_surf_p` = **110.27** (PR #1471, frieren, run `krsv4c21`)
**Best test (merged):** `test_avg/mae_surf_p` = **99.41** (same run)

Prior merged baselines (for reference):

| PR | What landed | val_avg | test_avg |
|---|---|---|---|
| #1471 (frieren) | p_weight=2.0 + clip_grad_norm=1.0 | **110.27** | **99.41** |
| #1480 (thorfinn) | bf16 autocast + grad_accum=2 + cruise-NaN fix | 116.30 | 104.96 |

## Notes for students

- **Baseline as of PR #1471:** `val_avg/mae_surf_p = 110.27`, `test_avg/mae_surf_p = 99.41`.
- **cruise-NaN workaround is landed.** All runs produce finite `test_avg` — no per-PR code needed.
- **Primary decision metric is `val_avg/mae_surf_p`** (lower is better). Beat **110.27** to be a winner.
- Grad clip at max_norm=1.0 is now in the training loop default (binding on nearly every step — the model runs in a high-gradient-magnitude regime).
- Report `val_avg/mae_surf_p`, `test_avg/mae_surf_p`, and all four per-test-split `mae_surf_p` values.
- Per-split metrics matter — flag splits where your change helps or hurts disproportionately.
