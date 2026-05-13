<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Current best — `icml-appendix-willow-pai2g-24h-r2`

Primary ranking metric: **`val_avg/mae_surf_p`** (lower is better)
Test-time metric: **`test_avg/mae_surf_p`** (lower is better)

## 2026-05-13 05:30 — PR #1666: tanjiro — smooth_l1(β=1) loss replaces MSE (MERGED)

**New best val and test. Second compounding winner on top of OneCycleLR: -9.3% val / -8.5% test.**

- **val_avg/mae_surf_p:** 88.06 (was 97.07) — **-9.3%**
- **test_avg/mae_surf_p:** **78.46** (was 85.71) — **-8.5%**
- **W&B run:** `fihyl2d5` (rebased on OneCycleLR baseline)
- **Surface channel MAE:** Ux=1.3378, Uy=0.6521, p=78.46
- **Epochs:** 18 in ~30 min

Per-split test `mae_surf_p` (run `fihyl2d5`):

| Split | test | vs OneCycleLR baseline (#1655) | Δ% |
|---|---|---|---|
| `single_in_dist` | 85.74 | 99.24 | **-13.6%** |
| `geom_camber_rc` | 90.31 | 95.85 | **-5.8%** |
| `geom_camber_cruise` | 58.96 | 61.71 | **-4.5%** |
| `re_rand` | 78.83 | 86.04 | **-8.4%** |

Changes vs prior baseline (#1655 OneCycleLR):
- `F.smooth_l1_loss(pred_norm, y_norm, beta=1.0)` per-element, replacing `(pred - y_norm) ** 2`. p_weight multiplier applied to smooth_l1 output (same ch_weights logic). Grad clip + surf_weight + vol/surf split unchanged.

Analysis: smooth_l1 caps per-element gradient at 1.0 for large residuals (MAE gradient shape for outliers), vs MSE's unbounded per-element gradient. This eval/train-alignment benefit is largest for the p-channel (heavy-tailed distribution) and visible on all splits. The pre-clip global grad norm dropped ~3-4× (mean 64→17, max 852→202) but the clip still binds on nearly every step — the two mechanisms are not redundant.

Stack: smooth_l1 + OneCycleLR + p_weight=2 + grad_clip=1.0 + bf16 + grad_accum=2.

Reproduce:
```bash
cd target/ && python train.py --agent <name> --wandb_name "<name>/smooth-l1" --wandb_group "willow-r2-loss-shape"
```
(smooth_l1_loss(β=1) is now the default loss in train.py; no extra flags needed)

---

## 2026-05-13 05:00 — PR #1655: alphonse — OneCycleLR max_lr=2e-3 (MERGED)

**New best val and test. Strongest single improvement of the launch: -12% val / -14% test.**

- **val_avg/mae_surf_p:** 97.07 (was 110.27) — **-12.0%**
- **test_avg/mae_surf_p:** **85.71** (was 99.41) — **-13.8%**
- **W&B runs:** `d29igs7w` (primary, seed 1), `r7pd9bmk` (seed 2: val=101.18, test=89.99 — both beats beat baseline)
- **Epochs:** ~19 in 30 min

Per-split test `mae_surf_p` (run `d29igs7w`):

| Split | test | vs baseline (#1471) | Δ% |
|---|---|---|---|
| `single_in_dist` | 99.24 | 116.69 | **-15.0%** |
| `geom_camber_rc` | 95.85 | 110.01 | **-12.9%** |
| `geom_camber_cruise` | 61.71 | 72.77 | **-15.2%** |
| `re_rand` | 86.04 | 98.17 | **-12.4%** |

Changes vs prior baseline (#1471):
- `scheduler = OneCycleLR(optimizer, max_lr=2e-3, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4)` — replacing `CosineAnnealingLR(T_max=50)`

Analysis: OneCycleLR with max_lr=2e-3 imposes a short warmup (10% of steps, LR rises from 8e-5 to 2e-3) then a smooth cosine anneal to near-zero. Combined with p_weight=2.0 + grad_clip=1.0, the improvement is ~4× larger than either mechanism alone — confirming orthogonality. Seed variance is ~4 MAE (d29igs7w: val=97.07, r7pd9bmk: val=101.18).

Reproduce:
```bash
cd target/ && python train.py --agent <name> --wandb_name "<name>/onecycle-lr" --wandb_group "willow-r2-schedule"
```
(OneCycleLR is now the default scheduler in train.py; no extra flags needed)

---

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

Transolver from `train.py` at HEAD — includes bf16 (PR #1480), grad_accum=2 (#1480), p_weight=2.0 + clip_grad_norm=1.0 (#1471), OneCycleLR (#1655), smooth_l1_loss(β=1) (#1666):

| Hyperparam | Value |
|---|---|
| `lr` (initial / base) | 5e-4 (overridden by OneCycleLR) |
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
| Schedule | `OneCycleLR(max_lr=2e-3, pct_start=0.1, anneal_strategy="cos")` |
| Loss | `smooth_l1(β=1)` per-element, then `p_weight`-weighted, `vol_loss + 10.0 * surf_loss` |
| Optimizer | AdamW (default betas, eps=1e-8) |

Reproduce: `cd target/ && python train.py --agent <name> --wandb_name "<name>/baseline"`.

## Current best metrics

W&B project: `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`

**Best val (merged):** `val_avg/mae_surf_p` = **88.06** (PR #1666, tanjiro, run `fihyl2d5`)
**Best test (merged):** `test_avg/mae_surf_p` = **78.46** (same run)

Prior merged baselines (for reference):

| PR | What landed | val_avg | test_avg |
|---|---|---|---|
| #1666 (tanjiro) | smooth_l1(β=1) loss | **88.06** | **78.46** |
| #1655 (alphonse) | OneCycleLR max_lr=2e-3 | 97.07 | 85.71 |
| #1471 (frieren) | p_weight=2.0 + clip_grad_norm=1.0 | 110.27 | 99.41 |
| #1480 (thorfinn) | bf16 autocast + grad_accum=2 + cruise-NaN fix | 116.30 | 104.96 |

## Notes for students

- **Baseline as of PR #1666:** `val_avg/mae_surf_p = 88.06`, `test_avg/mae_surf_p = 78.46`.
- **cruise-NaN workaround is landed.** All runs produce finite `test_avg` — no per-PR code needed.
- **Primary decision metric is `val_avg/mae_surf_p`** (lower is better). Beat **88.06** to be a winner.
- OneCycleLR is the default scheduler (max_lr=2e-3, pct_start=0.1, cosine anneal).
- smooth_l1_loss(β=1) is the default per-element loss (replaces MSE).
- Grad clip at max_norm=1.0 is in the training loop default. Note: pre-clip grad_norm is ~3-4× lower with smooth_l1 vs MSE baseline, but clip still binds on nearly every step.
- Report `val_avg/mae_surf_p`, `test_avg/mae_surf_p`, and all four per-test-split `mae_surf_p` values.
