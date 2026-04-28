# SENPAI Research Results — charlie-pai2d-r3

## 2026-04-28 00:03 — PR #280: L1 surface loss to align gradient with reported MAE metric
- Branch: `charliepai2d3-alphonse/l1-surface-loss`
- Hypothesis: switching the surface loss from MSE to L1 (volume MSE
  unchanged) aligns the gradient with the reported `val_avg/mae_surf_p`
  metric and is more robust to the heavy-tailed high-Re samples.
- Config: `bs=4`, `lr=5e-4`, all other knobs at defaults; only loss changed.

### Per-epoch validation (`val_avg/mae_surf_p`)

| epoch | val_avg/mae_surf_p | best |
|------:|-------------------:|:----:|
| 1  | 244.06 | * |
| 4  | 198.10 | * |
| 8  | 131.46 | * |
| 11 | 113.55 | * |
| 13 | 105.84 | * |
| 14 | **102.64** | * |

Best epoch 14 (the final epoch); curve was still descending at termination.
Stopped at epoch 14 by the 30-min timeout. Full per-epoch metrics committed
at `models/model-charliepai2d3-alphonse-l1-surface-loss-20260427-223604/`
and centralised at `research/EXPERIMENT_METRICS.jsonl`.

### Per-split (best-val checkpoint, epoch 14)

| split | val mae_surf_p | test mae_surf_p (NaN-safe) |
|-------|---------------:|---------------------------:|
| single_in_dist     | 121.18 | 109.80 |
| geom_camber_rc     | 125.01 | 114.60 |
| geom_camber_cruise |  73.22 |  79.92 |
| re_rand            |  91.14 |  86.58 |
| **avg**            | **102.64** | **97.73** |

### Analysis

- **Wins on all four val splits vs the prior baseline (PR #306, val 135.20).**
  Biggest improvement on the hardest split, `val_single_in_dist`: 121.18 vs
  190.14 (−36%). The high-Re raceCar singles dominated the surface error
  before; L1 cuts that error sharply.
- −24.1% on `val_avg/mae_surf_p`, −20.6% on `test_avg/mae_surf_p`. Test < val
  on three of four splits → no overfit.
- The bs=4 / lr=5e-4 config used here is *different* from the prior baseline
  (bs=8 / lr=7.07e-4 / MSE). So the headline 24% win conflates L1 vs MSE
  with bs=4 vs bs=8. Per the bs-only test (PR #306 vs unknown bs=4 MSE),
  the bs effect was at most ~5%; L1 carries the rest.
- Peak memory only 42.13 GB at bs=4 — round 4 has plenty of room to push
  bs and capacity in combination with L1.

### Decision

**Merged** as the new round 3 baseline. Old baseline (PR #306, val 135.20)
becomes round-3 reference 1. New baseline `val_avg/mae_surf_p = 102.64`,
`test_avg/mae_surf_p = 97.73`. The seven other in-flight r3 PRs branched off
the pre-L1 advisor; their results need to clear 102.64 (val) to be winners.
Several are likely orthogonal to L1 and useful for round 4 composition even
if they don't beat the new baseline outright.

---

## 2026-04-27 23:26 — PR #306: Batch size 8 with sqrt LR scaling (lr=7.07e-4)
- Branch: `charliepai2d3-thorfinn/batch8-sqrt-lr`
- Hypothesis: doubling `batch_size` to 8 with √2-scaled LR (`5e-4 → 7.07e-4`)
  reduces gradient noise without changing the data budget; tests whether
  gradient quality alone improves convergence within the 30-min wallclock.

### Per-epoch validation (`val_avg/mae_surf_p`)

| epoch | val_avg/mae_surf_p | best | sec | peak_mem |
|------:|-------------------:|:----:|----:|---------:|
| 1  | 264.88 | * | 134.5 | 84.2 GB |
| 2  | 215.25 | * | 129.2 | 84.2 GB |
| 5  | 212.78 | * | 130.0 | 84.2 GB |
| 7  | 188.54 | * | 129.9 | 84.2 GB |
| 8  | 155.47 | * | 128.8 | 84.2 GB |
| 11 | 142.97 | * | 129.2 | 84.2 GB |
| 13 | **135.20** | * | 129.7 | 84.2 GB |
| 14 | 142.03 |   | 127.1 | 84.2 GB |

Best epoch 13/14. Stopped at epoch 14 by 30-min timeout (cosine T_max was
50 → never reached the tail). Full per-epoch metrics committed at
`models/model-charliepai2d3-thorfinn-batch8_sqrt_lr-20260427-223454/metrics.jsonl`
and centralised at `research/EXPERIMENT_METRICS.jsonl`.

### Per-split (best-val checkpoint)

| split | val mae_surf_p | test mae_surf_p (corrected) |
|-------|---------------:|----------------------------:|
| single_in_dist     | 190.14 | 173.01 |
| geom_camber_rc     | 138.39 | 120.22 |
| geom_camber_cruise |  97.95 |  82.83 |
| re_rand            | 114.32 | 116.53 |
| **avg**            | **135.20** | **123.15** |

### Analysis

- Run is stable; bs=8 + lr=7.07e-4 fits comfortably (peak 84.2 GB of 96 GB).
- The val curve was still descending at termination (epoch 13 = 135.20 vs
  epoch 11 = 142.97), so this is a **partially-trained model on a truncated
  cosine** — not a converged result.
- Test < val on three of four splits (single, rc, cruise) → no overfit.
- Cruise track (97.95 / 82.83) is by far the easiest; single-in-dist is the
  hardest (190.14 / 173.01) — high-Re raceCar singles dominate the surface
  error.
- **Critical infrastructure bug found and fixed:** `data/scoring.py` had an
  `Inf*0=NaN` reduction bug on the test path (a single sample in
  `test_geom_camber_cruise` has 761 Inf values in its pressure GT). Fix
  applied as advisor commit `2eb5c7f` (attribution to thorfinn).

### Decision

**Merged** as the round 3 measured baseline. No prior r3 baseline existed,
so this becomes the reference for the seven other in-flight r3 PRs. Per-PR
follow-ups for round 4: bs=12 (√3 LR scaling) if "larger batch" wins;
`--epochs ≤ 14` to actually decay cosine inside the wallclock cap;
activation checkpointing for bs=16+. These are recorded in
`research/CURRENT_RESEARCH_STATE.md`.
