<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2i-24h-r3`

## 2026-05-15 14:30 — PR #3243: Deeper Transolver (n_layers 5 → 8)

- Branch: `willowpai2i24h3-alphonse/deeper-transolver`
- Student: willowpai2i24h3-alphonse
- Hypothesis: increasing depth from L=5 to L=8 (paper's reference depth) reduces
  `val_avg/mae_surf_p` because the baseline is capacity-limited. Predicted −8% to −15%.

### Results

| Metric | Value | W&B |
|---|---|---|
| W&B run | `sof2eicn` (deeper-transolver-L8) | https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/sof2eicn |
| best `val_avg/mae_surf_p` | **147.85** at epoch 9 of 9 | |
| `test_avg/mae_surf_p` (nansafe) | **138.60** | (in-tree scorer NaN — see bug note) |
| best epoch / total epochs | 9 / 9 | timeout hit at 30.89 min |
| mean epoch wall-clock | 208 s/epoch (≈3.47 min) | |
| total train minutes | 30.89 (cut by `SENPAI_TIMEOUT_MINUTES=30`) | |
| params | 1.03 M (+56% vs L=5) | |
| peak VRAM | 64.5 GB / 96 GB (not logged in W&B summary; reported by student) | |
| batch_size | 4 (no OOM) | |

### Per-split val at best epoch

| Split | `mae_surf_p` |
|---|---|
| val_single_in_dist | 176.27 |
| val_geom_camber_rc | 172.55 |
| val_geom_camber_cruise | 112.02 |
| val_re_rand | 130.55 |
| **val_avg** | **147.85** |

### Per-split test (nansafe)

| Split | `mae_surf_p` |
|---|---|
| test_single_in_dist | 154.06 |
| test_geom_camber_rc | 156.95 |
| test_geom_camber_cruise | 112.50 |
| test_re_rand | 130.88 |
| **test_avg (nansafe)** | **138.60** |

### Analysis

- Train loss decreased monotonically over all 9 epochs; no instability — the deeper
  model trained cleanly.
- Hard `SENPAI_TIMEOUT_MINUTES=30` cap → 9 of 50 epochs only; cosine `T_max=50` meant
  LR never annealed (still at ~peak when the run was killed).
- Result is an **undertrained L=8 number**. The hypothesis cannot be falsified or
  confirmed from this run alone — we never reached the regime where L=8's extra capacity
  would matter most (late-cosine fine-tuning).
- Student identified a critical bug in `data/scoring.py`: a `-inf` in interior pressure
  of one cruise test sample propagates NaN into the surface metric via `NaN * 0 = NaN`,
  making in-tree `test_avg/mae_surf_p = None`. Documented in `CURRENT_RESEARCH_STATE.md`.
- The student's W&B summary correctly logs nansafe variants and `data_bug/*` diagnostics.

### Advisor action

- **Held in `status:review`** — not merged. Awaiting the round-3 cohort (7 in-flight
  PRs) for direct comparison. Once cohort completes, rank by `val_avg/mae_surf_p` and
  merge the strongest.
- **Reassigned alphonse to PR #3282** (`bf16-mixed-precision`) — attacks the time-budget
  constraint that distorted this run. If bf16 ~doubles throughput, the depth hypothesis
  can be re-tested with proper epoch counts.
