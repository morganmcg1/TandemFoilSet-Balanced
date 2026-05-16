# SENPAI Research State

- **Last updated:** 2026-05-16 04:00 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 03:45 UTC — no open issues).

## Current best baseline (UPDATED)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **81.9754** | run `j5214ii4` (PR #3475 asinh-pressure, merged 03:30 UTC) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **81.3654** | run `j5214ii4` |

Per-split validation (best @ epoch 14):

| Split | mae_surf_p | Δ vs prev baseline (#3474, 90.6131) |
|---|---|---|
| val_single_in_dist | 101.013 | **−4.8%** |
| val_geom_camber_rc | 90.717 | **−8.8%** |
| val_geom_camber_cruise | 59.909 | **−14.8%** |
| val_re_rand | 76.263 | **−11.8%** |

**Reproduce new baseline**:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --agent <student>
```

## Active PRs (zero idle students)

### Round-4 carry-over (WIP, pre-merge, OLD BASELINE)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3543 | alphonse | ema-decay-push (0.97, 0.95) | 0.97 arm running; 0.95 pending |
| #3649 | fern | n_head sweep (4→8, cond. 16) | running on OLD baseline (90.61) |

### Round-5 (WIP, NEW baseline: val=81.9754)

| PR | Student | Hypothesis | Key test |
|----|---------|-----------|----------|
| #3659 | askeladd | asinh-scale-sweep (1.5, 2.0) | Optimal compression strength |
| #3660 | frieren | re-sinusoidal-corrected | Fix log_re normalization bug → target val_re_rand |
| #3661 | nezuko | wd-on-asinh (1e-3, 5e-3) | Regularization compound with asinh |
| #3662 | thorfinn | vel-asinh (scale=1.0) | asinh on Ux/Uy channels |
| #3663 | edward | dropout-sweep (0.05, 0.1) | MLP dropout for OOD |
| #3664 | tanjiro | slice-num-on-asinh (128) | Retest capacity on cleaner loss |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| **#3475 (askeladd)** | **asinh-pressure (scale=1.0)** | **81.9754** | **−9.53%** | **Every val split improves; val_re_rand −11.8%; test_3split=81.37** |

## Closed PRs (Round 4)

| PR | Student | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| #3477 | thorfinn | physics-continuity | all REGRESS 98-106 | CLOSED |
| #3571 | fern | depth-sweep n_layers=6 | val=93.83 (+3.55%) | CLOSED — wall-clock bound |
| #3610 | thorfinn | mlp_ratio=4 | val=93.12 (+2.76%) | CLOSED — wall-clock bound |
| #3576 | nezuko | wd-sweep (old baseline) | val=90.46 (TIED) | CLOSED — superseded |
| #3575 | edward | p_surf_weight=3/5 | val=94.65 (+4.5%) | CLOSED — decisive regression |
| #3578 | frieren | re-sinusoidal-embed | val=130.82 (+44%) | CLOSED — frequency mismatch bug |
| #3577 | tanjiro | slice-num=128 (old stack) | val=101.18 (+11.6%) | CLOSED — stale, pre-asinh |

## Key findings (cumulative)

### Optimization stack (Rounds 1-3)
EMA → EMA+clip+Huber → faster EMA decay (0.99) compounds cleanly. Combined: 136.89 → 90.61 (−33.8%).

### Target-side transform (Round 4 winner)
**asinh(pressure)**: heavy-tail compression of the pressure target. Compounds super-additively with fast-EMA. scale=1.0 → val=81.97 (−9.53% vs 90.61).
- val_re_rand drops 86.49 → 76.26 (−11.8%) — largest OOD improvement yet
- Every val split improves

### Architecture/regularization axes (Round 4 — falsified on OLD baseline)
- Depth (n_layers=6): +3.55% REGRESS — wall-clock bound, fewer epochs
- MLP width (mlp_ratio=4): +2.76% REGRESS — wall-clock bound  
- Weight decay (wd sweep): TIED — 90.46, marginal; worth retesting on new baseline
- Channel loss weighting (p_surf_weight): decisive REGRESS (+4-18%) — gradient norm explosion

### Bad implementations (closed, correctable)
- Re-sinusoidal: catastrophic failure due to wrong normalization (log_re/16 spans [0.78, 0.96] not [0,1]). Fix: normalize to actual [10.8, 13.4] range. Reassigned as corrected version.
- Slice_num=128 on old stack: run completed but not reported; retesting on new baseline.

## Strategic outlook (Round 5)

**Target**: val < 78 (−5% improvement from 81.97). To get there:
1. **asinh scale** might find 1-2% via optimal compression
2. **Corrected Re-embed** could improve val_re_rand toward 70
3. **wd + asinh** might add 1-2% OOD generalization
4. **velocity asinh** — likely marginal (vel less heavy-tailed) but cheap to test
5. **dropout** — regularization, moderate OOD expected gain
6. **slice_num on asinh** — uncertain, previous regression was pre-asinh

**alphonse #3543 ema-decay-push**: 0.97/0.95 arms — if they beat 0.99, could compound with asinh for another improvement. Note: these will be benchmarked against new baseline (81.97). Val=90.xx on old baseline won't beat it.

**fern #3649 n_head**: same caveat — benchmarked against 81.97. If n_head=8 gives val~85, it's a clear close.

## Operational notes

- **data/scoring.py NaN bug**: cruise=NaN fleet-wide. Known, dataset issue.
- Per-run budget: 30 min wall clock, 50 epoch cap (~14 epochs).
- **Zero idle students**: 8 WIP PRs (3543, 3649, 3659-3664).
- REST API: core exhausted (0/5000), resets ~04:19 UTC. GraphQL: 3000+ remaining. PR creation works via GraphQL.
