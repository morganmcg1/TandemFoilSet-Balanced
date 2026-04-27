# SENPAI Research Results

Branch: `icml-appendix-willow-pai2d-r2`. Primary metric:
`val_avg/mae_surf_p` (lower is better). Wall-clock cap:
`SENPAI_TIMEOUT_MINUTES=30` per run.

Note on `test_avg/mae_surf_p`: every round-1 run reports `NaN` on
`test_geom_camber_cruise/mae_surf_p`, which propagates to the
test-average. This is a `data/scoring.py::accumulate_batch`
NaN-propagation issue (no guard against non-finite predictions; the
existing guard only skips samples whose **GT** is non-finite).
Tracked separately for a dedicated bug-fix PR after round-1 settles.

## 2026-04-27 23:30 — PR #311: Round 1 axis: model width — n_hidden 128 → 192

- Branch: `willowpai2d2-alphonse/width-192`
- Hypothesis: 3–7% reduction in `val_avg/mae_surf_p` from
  `n_hidden 128 → 192` (+50% width, ~2.25× params).
- Run: `oahab4iy` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/oahab4iy)

### Results (best checkpoint, epoch 10 / 50 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 160.79 | 2.29 | 0.91 |
| val_geom_camber_rc | 148.93 | 3.43 | 1.26 |
| val_geom_camber_cruise | 105.93 | 1.64 | 0.65 |
| val_re_rand | 120.87 | 2.38 | 0.91 |
| **val_avg** | **134.13** | 2.43 | 0.93 |
| test_single_in_dist | 137.18 | 2.14 | 0.85 |
| test_geom_camber_rc | 132.05 | 3.22 | 1.17 |
| test_geom_camber_cruise | NaN ⚠ | 1.57 | 0.60 |
| test_re_rand | 121.55 | 2.14 | 0.89 |
| **test_avg** | NaN ⚠ | 2.27 | 0.88 |

### Conclusion

**Send back** for compute-equal follow-up. The wider model is
clearly undertrained at the 30-min cap (10 of 50 epochs reached;
~184s/epoch vs the ~36s/epoch of baseline ⇒ ~5× slower per step,
not the predicted 2–2.25×). Peak GPU memory at 92.89% leaves no
headroom for stacking. Val curve still descending steeply at epoch
10 (258 → 134), so the metric reflects undertraining rather than
the asymptotic capacity of width-192. The 134.13 number is at
the front of the round-1 cohort but not interpretable as a clean
"width helps" signal.

Sent back with: try **width-160** (1.55× params, divisible by 4),
expected 20–25 epochs in budget; optionally a same-PR AMP-only
baseline at width-128 to disentangle precision from architecture.

## 2026-04-27 23:30 — PR #335: Round 1 axis: LR schedule — 5-epoch warmup + cosine, peak 1e-3

- Branch: `willowpai2d2-tanjiro/warmup-cosine-1e3`
- Hypothesis: 3–7% reduction in `val_avg/mae_surf_p` from
  `lr 5e-4 → 1e-3` with 5-epoch linear warmup + cosine decay.
- Run: `ri332d19` (W&B
  https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-d-r2/runs/ri332d19)

### Results (best checkpoint, epoch 13 / 14 — wall-clock cut)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-|-:|-:|-:|
| val_single_in_dist | 212.25 | 2.11 | 1.01 |
| val_geom_camber_rc | 149.98 | 2.96 | 1.24 |
| val_geom_camber_cruise | 120.32 | 1.67 | 0.62 |
| val_re_rand | 135.73 | 2.26 | 0.95 |
| **val_avg** | **154.57** | 2.27 | 0.95 |
| test_single_in_dist | 178.60 | 2.00 | 0.96 |
| test_geom_camber_rc | 138.07 | 2.86 | 1.17 |
| test_geom_camber_cruise | NaN ⚠ | 1.52 | 0.57 |
| test_re_rand | 137.11 | 2.11 | 0.89 |
| **test_avg** | NaN ⚠ | 2.12 | 0.90 |

### Conclusion

**Send back** for schedule-shape iteration. The warmup wiring is
correct (verified from W&B `lr` panel: 1e-4 → 1e-3 over epochs
1–5, then cosine decay engages). But the 30-min cap only allows
14 epochs, so cosine `T_max=50` decays only ~9.5% of its arc —
the schedule is effectively "warmup + flat 1e-3," not the
warmup+cosine the hypothesis was testing. 154.57 sits at the
bottom of the round-1 cohort (133.55–154.57 range), consistent
with a flat high LR overshooting the local optima that lower
LRs can navigate in a short budget.

Sent back with: parametrize `--cosine_t_max` as a CLI flag, run a
small sweep `(lr 7e-4, T_max 18)` and `(lr 1e-3, T_max 15)` on
a shared `--wandb_group "willow-r2-tanjiro-sched-v2"`. Optional
third variant `(lr 8e-4, T_max 18)`.

## Round-1 cohort context (W&B observation, not advisor decisions)

These W&B runs from sibling PRs are visible on this project but are
not part of this advisor invocation (entrypoint surfaced only #311
and #335 for review; the others are still WIP or in-flight). Listed
here only to document the cohort's metric range as of this writing:

| W&B name | best_val_avg/mae_surf_p | best_epoch | state |
|-|-:|-:|-|
| willow-r2-fern-slice-128 | 133.55 | 11 | finished |
| willow-r2-alphonse-width-192 | 134.13 | 10 | finished |
| willow-r2-nezuko-surf-15 | 137.42 | 13 | finished |
| willow-r2-edward-mlp-ratio-4 | 137.83 | 11 | finished |
| willow-r2-thorfinn-bs8-lr7e-4 | 139.39 | 14 | finished |
| willow-r2-askeladd-depth-8 | 150.06 | 9 | finished |
| willow-r2-tanjiro-warmup-cos-1e3 | 154.57 | 13 | finished |

No baseline (unmodified train.py) finished run exists yet — every
finished round-1 run was a hypothesis variant.
