<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2g-24h-r2`

Primary metric: `val_avg/mae_surf_p` (lower is better; baseline ~132).
Test metric (`test_avg/mae_surf_p`) is structurally NaN on this branch due to a cruise-test overflow systemic to the codebase; rankings use val_avg + the three finite per-test-split metrics.

---

## 2026-05-12 20:00 — PR #1471: frieren — pressure channel weight=3 in loss (sent back, not merged)

- Branch: `willowpai2g24h2-frieren/p-channel-weight-3`
- Hypothesis: up-weight pressure (dim 2) inside per-channel `sq_err` by `p_weight=3.0` to direct more gradient at the ranking metric.
- Result: monotone val descent 241 → 130.98 over 14 epochs; W&B runs `ftuclvqz` (first arm, 148.57) and `ph14bsim` (second arm, 130.98, canonical).

### Metrics

| Metric | Value | Baseline | Delta |
|---|---|---|---|
| `val_avg/mae_surf_p` (best `ph14bsim`) | 130.98 | 131.79 / 132.73 | ~-0.6 to -1.3% |
| `val_single_in_dist/mae_surf_p` | 166.71 | 136.34 | +22% (worse) |
| `val_geom_camber_rc/mae_surf_p` | 140.45 | ~130 | +8% (worse) |
| `val_geom_camber_cruise/mae_surf_p` | 98.51 | 117.71 | **-16% (better)** |
| `val_re_rand/mae_surf_p` | 118.23 | 121.79 / 117.71 | mixed |
| `test_single_in_dist/mae_surf_p` | 140.14 | ~135 | mixed |
| `test_geom_camber_rc/mae_surf_p` | 126.71 | TBD | n/a |
| `test_geom_camber_cruise/mae_surf_p` | **NaN** | **NaN** (systemic) | n/a |
| `test_re_rand/mae_surf_p` | 116.95 | TBD | n/a |

### Analysis

The val_avg gain (~1%) is inside noise (the two baseline arms differ by 0.7%) and the first frieren arm regressed by 13% — high variance. The mean-improvement signal is dominated by `val_geom_camber_cruise` (-16%), which is precisely the split where the test counterpart blew up. Up-weighting pressure made cruise val better but pushed the model's p output into overflow territory on the larger test cruise set (200 samples vs 100 val).

Student's diagnostic is correct: the cruise NaN traces to `accumulate_batch` propagating `inf - y` through `mask` arithmetic. The systemic cruise-test NaN affects every run including baseline, so it's not a frieren-specific veto — but the *magnitude* of the p-output blowup at `p_weight=3` is what makes this hypothesis risky.

### Decision

Sent back to student with two changes: drop `p_weight` to 2.0 (less aggressive) and add `clip_grad_norm_(model.parameters(), 1.0)` as a baseline-hardening numerical safety. Same `--wandb_group "willow-r2-p-weight"` so the arms remain comparable. Acceptance criterion for re-review: val_avg cleanly below baseline AND no regression on the three finite per-test-splits.
