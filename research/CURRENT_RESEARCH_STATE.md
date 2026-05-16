# SENPAI Research State

- **Last updated:** 2026-05-16 05:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 05:30 UTC — no open issues).

## Current best baseline (UNCHANGED)

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

### Round-4 carry-over (WIP, OLD BASELINE — benchmarked vs 81.97)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3649 | fern | n_head sweep (4→8) | Pod active since 05:21 UTC; no results yet |

### Round-5 active (WIP, NEW baseline: val=81.9754)

| PR | Student | Hypothesis | Key test |
|----|---------|-----------|----------|
| #3659 | askeladd | asinh-scale-sweep (1.5, 2.0) | Optimal compression strength |
| #3660 | frieren | re-sinusoidal-corrected | Fix log_re normalization bug → target val_re_rand |
| #3661 | nezuko | wd-on-asinh (1e-3, 5e-3) | Regularization compound with asinh |
| #3662 | thorfinn | vel-asinh (scale=1.0) | asinh on Ux/Uy channels |
| #3663 | edward | dropout-sweep: rerun at 0.025 | Lighter dose preserves OOD signal, avoids rc regression |
| #3679 | alphonse | Huber δ sweep (0.5, 0.3) | Loss-shape companion to asinh transform |

### Round-5 fresh assignments

| PR | Student | Hypothesis | Key test |
|----|---------|-----------|----------|
| #3723 | tanjiro | SwiGLU MLP activation | GELU→SwiGLU in TransolverBlocks (LLaMA/PaLM-style gating) |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| **#3475 (askeladd)** | **asinh-pressure (scale=1.0)** | **81.9754** | **−9.53%** | **Every val split improves; val_re_rand −11.8%; test_3split=81.37** |

## Closed PRs (all rounds)

| PR | Student | Hypothesis | Best val | Verdict |
|---|---|---|---|---|
| #3477 | thorfinn | physics-continuity | 98-106 all REGRESS | CLOSED |
| #3571 | fern | depth-sweep n_layers=6 | 93.83 (+3.55%) | CLOSED — wall-clock bound |
| #3610 | thorfinn | mlp_ratio=4 | 93.12 (+2.76%) | CLOSED — wall-clock bound |
| #3576 | nezuko | wd-sweep (old baseline) | 90.46 (TIED) | CLOSED — superseded |
| #3575 | edward | p_surf_weight=3/5 | 94.65 (+4.5%) | CLOSED — decisive regression |
| #3578 | frieren | re-sinusoidal-embed | 130.82 (+44%) | CLOSED — frequency mismatch bug |
| #3577 | tanjiro | slice-num=128 (old stack) | 101.18 (+11.6%) | CLOSED — stale, pre-asinh |
| #3543 | alphonse | ema-decay-push (0.98/0.97/0.95) | 90.84 | CLOSED — exhausted below 0.99; +10.8% vs new baseline |
| **#3664** | **tanjiro** | **slice-num=128 on asinh** | **90.77 (+10.7%)** | **CLOSED — wall-clock bind; axis exhausted pre- AND post-asinh** |

## Key findings (cumulative)

### Optimization stack (Rounds 1-3)
EMA → EMA+clip+Huber → faster EMA decay (0.99) compounds cleanly. Combined: 136.89 → 90.61 (−33.8%).

### Target-side transform (Round 4 winner)
**asinh(pressure)**: heavy-tail compression of the pressure target. Compounds super-additively with fast-EMA. scale=1.0 → val=81.97 (−9.53% vs 90.61).
- val_re_rand drops 86.49 → 76.26 (−11.8%) — largest OOD improvement yet
- Every val split improves

### Exhausted axes
- **EMA decay**: optimum firmly at 0.99. 0.997→0.99 improved; 0.98→0.95 all regress.
- **Depth** (n_layers=6): +3.55% REGRESS — wall-clock bound
- **MLP width** (mlp_ratio=4): +2.76% REGRESS — wall-clock bound
- **slice_num=128**: +10.7-11.6% REGRESS both stacks — wall-clock bind is structural (4× attention cost)
- **Channel loss weighting** (p_surf_weight): decisive REGRESS (+4-18%)

### Signal-positive axes (pending confirmation)
- **Dropout** (edward #3663, v1): val +0.59% (within noise), test −0.64%, val_re_rand improved. Dose too high on smallest-support split. Retesting at 0.025 (#3663 v2).

### Open axes (Round 5)
- **asinh scale** (#3659): optimal compression in 1.5/2.0 range
- **Corrected Re-embed** (#3660): fixed normalization log_re in [10.8, 13.4]
- **wd on asinh** (#3661): regularization with clean loss landscape
- **vel-asinh** (#3662): apply asinh to Ux/Uy (less heavy-tailed but free)
- **dropout 0.025** (#3663 v2): lighter dose of proven-signal regularization
- **n_head sweep** (#3649): architecture attention granularity (pod active, results pending)
- **Huber δ on asinh** (#3679): loss-shape companion — δ=1.0 was tuned for raw pressure
- **SwiGLU MLP** (#3723): GELU→SwiGLU in all TransolverBlock MLPs; high prior from LLaMA/PaLM

## Strategic outlook

**Target**: val < 78 (−5% from 81.97). Most promising paths:
1. **Huber δ** (#3679 alphonse): mechanistically motivated; expected 1-3% gain.
2. **SwiGLU** (#3723 tanjiro): transformer literature consensus, novel axis; expected 1-3%.
3. **Corrected Re-embed** (#3660 frieren): could push val_re_rand below 70.
4. **asinh scale** (#3659 askeladd): 1-2% if optimal scale ≠ 1.0.
5. **Compounding**: Huber δ + SwiGLU + asinh scale could stack for val < 78.

## Operational notes

- **data/scoring.py NaN bug**: cruise=NaN fleet-wide. Known dataset issue.
- Per-run budget: 30 min wall clock, 50 epoch cap (~14 epochs).
- **Zero idle students**: 8 WIP PRs (#3649, #3659-#3663, #3679, #3723). All r2 pods healthy.
- **Stale_wip resolved**: #3649 fern pod was idle 03:26-05:21 UTC, now confirmed active.
