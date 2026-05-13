<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Best Baseline — `icml-appendix-willow-pai2g-24h-r3` (willow-pai2g-24h-r3)

Primary metric (lower is better): `val_avg/mae_surf_p` (equal-weight mean surface-pressure MAE across 4 val splits).
Paper-facing metric: `test_avg/mae_surf_p` (4 test splits; the cruise-NaN-y bug is now fixed in `train.py::evaluate_split` via PR #1433 — first clean 4-split test pass measured at `test_avg/mae_surf_p = 86.87` on the current advisor branch, see Supplemental section).

## 2026-05-12 21:05 — PR #1441: Replace MSE with SmoothL1 (Huber, beta=0.1) loss

- **`val_avg/mae_surf_p` (primary):** **104.6982** (epoch 13)
- **`test_avg/mae_surf_p` (3 splits, ex-cruise):** **101.0793**
- **`test_avg/mae_surf_Ux`** (4-split): 1.6194
- **`test_avg/mae_surf_Uy`** (4-split): 0.5998
- **Per-split surface-p MAE (val, best-val epoch):**
  - val_single_in_dist: 120.6339
  - val_geom_camber_rc: 117.4537
  - val_geom_camber_cruise: 82.3642
  - val_re_rand: 98.3412
- **W&B run:** `d53f0jn4` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3/runs/d53f0jn4)
- **Reproduce:**
  ```bash
  cd target && python train.py --loss_fn smooth_l1 --smooth_l1_beta 0.1
  ```

Mechanism summary: SmoothL1 caps the per-element gradient magnitude at `beta=0.1` in normalized space, so high-Re/high-`|p|` outlier samples no longer dominate the quadratic MSE gradient. Training curve becomes monotonically descending instead of oscillating ±20–40 MAE points per epoch under MSE. Predicted delta was 2–5%; observed delta was −20.6% on the primary metric.

## 2026-05-12 21:06 — PR #1433: Add gradient norm clipping (max_norm=1.0)

Merged on top of #1441. PR #1433 was measured under MSE (PR's own baseline arm = 131.96, clip=1.0 variant = **114.18**, −13.5%) so it does not dethrone the 104.70 high-water mark from #1441, but it ships:

- `clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()` (orthogonal hygiene lever)
- **Inline cruise-test NaN fix** in `train.py::evaluate_split` — drops non-finite-`y` samples before the forward pass and `accumulate_batch`. Once a 4-split test pass succeeds end-to-end, future PRs will report the true `test_avg/mae_surf_p` rather than 3-split-ex-cruise.

- **W&B run:** `qof1cbki` (https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r3/runs/qof1cbki) — recorded for reference; not the branch's empirical best.

Current advisor-branch code = SmoothL1(β=0.1) + grad_clip(1.0) + inline cruise-NaN fix. The combined-config run has not been measured yet (PR #1441 was SmoothL1-only, PR #1433 was grad-clip-only). Best-verified empirical metric remains **104.70**; expect future PRs that re-establish a baseline arm under the current advisor code to land near or below that point.

## 2026-05-13 00:02 — Supplemental: Current advisor-branch combined-config measurements (NOT a new merged PR)

The following measurements come from baseline arms of in-flight Round 2 PRs and characterize the current state of the advisor branch (SmoothL1 + grad_clip + cruise-NaN fix). These are not new code changes — they measure the code already merged via #1441 + #1433.

| Source | Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Note |
|---|---|---|---|---|
| #1616 (askeladd uniform-baseline, closed) | `eztvtkxc` | **90.91** | **86.87 (4-split, all finite!)** | Cleanest measurement; first paper-facing 4-split test pass |
| #1615 (tanjiro smooth-l1-v2 baseline, WIP) | `x0ud9i0a` | 102.17 | (TBD) | Independent run |
| #1437 (fern baseline-newbase, WIP) | `r7ysmbfi` | 104.84 | (TBD) | Independent run |

Range: 90.91 to 104.84 (15% spread; ±7 noise around mean ~99).

**Per-split val surface-p MAE (askeladd `eztvtkxc`, best-val epoch 14):**
- val_single_in_dist: 103.90
- val_geom_camber_rc: 105.34
- val_geom_camber_cruise: 68.99
- val_re_rand: 85.40

**Per-split test surface-p MAE (askeladd `eztvtkxc`, 4-split clean):**
- test_single_in_dist: 94.28
- test_geom_camber_rc: 95.81
- test_geom_camber_cruise: **76.61** (finite — cruise-NaN-y filter from #1433 worked)
- test_re_rand: 80.78

Reproduce (current advisor-branch HEAD):
```bash
cd target && python train.py --loss_fn smooth_l1 --grad_clip 1.0
```

Implication for Round 2 hypothesis ranking: any claim of <5% improvement is in the single-seed noise band. Practical merge bar = around the lower end of the variance band (~91); verified high-water-mark for *merged* code stays at 104.70 until a winning hypothesis PR with a terminal `SENPAI-RESULT` marker lands.
