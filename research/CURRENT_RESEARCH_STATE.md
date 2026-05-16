# SENPAI Research State

- **Last updated:** 2026-05-16 10:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 10:30 UTC — no open issues)

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **64.3427** | PR #3794 fern (n_head=2 on SwiGLU, merged ~09:30 UTC) |
| `test_3split/mae_surf_p` (cruise=NaN) | **63.6663** | PR #3794 fern |

**Reproduce new baseline**:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --agent <student>
```

## Active PRs — Rounds 6-7

| PR | Student | Hypothesis | Status | Expected |
|----|---------|-----------|--------|----------|
| #3789 | thorfinn | vel-asinh scale=0.5 on SwiGLU | **WINNER detected** val=63.74 (run hy29un5q); 3rd run in progress; awaiting terminal SENPAI-RESULT | −0.93% |
| #3790 | nezuko | wd=1e-3 on SwiGLU | Regress confirmed (65.65); awaiting terminal to close | +2.0% |
| #3793 | alphonse | Huber δ=0.5 on SwiGLU | Marginal regress (65.29); possibly 3rd arm running; awaiting terminal | +1.5% |
| #3796 | askeladd | vel-asinh scale=0.25/0.375 fine sweep | scale=0.25 regresses (67.02); 3rd run in progress (possibly 0.375) | tbd |
| #3854 | fern | slice_num fine sweep (32, 128) with n_head=2 | WIP — just assigned | New axis |
| #3858 | frieren | Attention dropout in PhysicsAttention on n_head=2 | WIP — just assigned | New axis |
| #3874 | edward | LR warmup (1-2 ep) on SwiGLU + n_head=2 | WIP — just assigned | −0.5-2.5% |
| #3877 | tanjiro | PhysicsAttention temperature_init=0.2 on n_head=2 | WIP — just assigned | −0.5-2.5% |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| #3475 (askeladd) | asinh-pressure (scale=1.0) | 81.9754 | −9.53% | Every split improved |
| **#3723 (tanjiro)** | **SwiGLU param-matched MLP** | **66.6130** | **−18.74%** | Largest single win; gating mechanism; blocks-only scope is optimal |
| **#3794 (fern)** | **n_head=2 on SwiGLU** | **64.3427** | **−3.41%** | Fewer heads + larger per-head dim; stacks on SwiGLU |

## Closed PRs — All Rounds

| PR | Student | Hypothesis | Best val | Verdict |
|---|---|---|---|---|
| #3477 | thorfinn | physics-continuity | 98-106 | CLOSED |
| #3571 | fern | depth n_layers=6 | 93.83 | CLOSED — wall-clock |
| #3610 | thorfinn | mlp_ratio=4 | 93.12 | CLOSED — wall-clock |
| #3576 | nezuko | wd-sweep (old baseline) | 90.46 | CLOSED — superseded |
| #3575 | edward | p_surf_weight=3/5 | 94.65 | CLOSED — regression |
| #3578 | frieren | re-sinusoidal (buggy) | 130.82 | CLOSED — freq mismatch |
| #3577 | tanjiro | slice_num=128 (old stack) | 101.18 | CLOSED — stale |
| #3543 | alphonse | ema-decay-push (0.98-0.95) | 90.84 | CLOSED — exhausted |
| #3664 | tanjiro | slice_num=128 on asinh | 90.77 | CLOSED — wall-clock |
| #3660 | frieren | re-sinusoidal-corrected | 96.85 | CLOSED — +18% regression |
| #3663 | edward | dropout 0.025 + 0.05 | 83.49 | CLOSED — axis non-monotone |
| #3662 | thorfinn | vel-asinh scale=0.5 (old stack) | 76.15 | CLOSED — beat old baseline; re-testing on SwiGLU (#3789) |
| #3661 | nezuko | wd=1e-3 (old stack) | 79.71 | CLOSED — re-testing on SwiGLU (#3790) |
| #3679 | alphonse | Huber δ=0.5 (old stack) | 80.85 | CLOSED — re-testing on SwiGLU (#3793) |
| #3659 | askeladd | asinh scale=1.5 | 82.16 | CLOSED — scale=1.0 optimal |
| #3649 | fern | n_head=2 (pre-asinh) | 86.78 | CLOSED — re-tested on SwiGLU (#3794, MERGED) |
| #3766 | edward | DropPath stochastic depth | 90.59 | CLOSED — +41% regression; 5-layer too shallow at 14ep budget |
| #3795 | tanjiro | SwiGLU in all MLPs | 76.08 | CLOSED — +18% regression; I/O boundary gating breaks projections |
| #3770 | frieren | Mixup augmentation | 105.30 | CLOSED — FALSIFIED; geometry interpolation is non-physical |

## Key findings (cumulative)

### Optimization stack (Rounds 1-3)
EMA → EMA+clip+Huber → decay=0.99: 136.89 → 90.61 (−33.8%). EMA decay axis exhausted below 0.99.

### Target-side transforms (Rounds 4-6)
- **asinh(pressure) scale=1.0** → 81.97 (−9.53%). Scale axis exhausted (1.5 regresses).
- **vel-asinh scale=0.5** → confirmed win on SwiGLU too (hy29un5q=63.74, awaiting merge). scale=0.25 regresses.

### Architecture — BREAKTHROUGHS (Rounds 5-7)
- **SwiGLU param-matched MLP** (PR #3723) → 66.61 (−18.74%). Gating in TransolverBlocks only. I/O layer gating (#3795) fails.
- **n_head=2** (PR #3794) → 64.34 (−3.41%). Compounds with SwiGLU.

### Falsified axes (summary)
EMA decay < 0.99, depth n_layers=6, mlp_ratio>2, slice_num=128, sinusoidal Re-embed, p_surf_weight, feature dropout, asinh scale > 1.0, n_head=8, DropPath, SwiGLU-in-all-MLPs, Mixup, wd=1e-3 on SwiGLU, Huber δ=0.5 on SwiGLU

## Queued hypotheses (from researcher-agent, next idle-student assignments)

| ID | Hypothesis | Expected |
|----|-----------|---------|
| H-07 | vel-asinh per-channel (Ux, Uy independent scales) | −2-5% |
| H-02 | SGDR warm restarts (T_0=5 or 8) | −2-5% |
| H-06 | surf_weight sweep (15 or 20 vs current 10) | −1-4% |

## Strategic outlook

**Target**: val < 60. Path:
1. **vel-asinh merge** (#3789): val=63.74 pending terminal → new baseline ~63.7
2. **lr-warmup** (#3874): −0.5-2.5% expected on new baseline → ~62.1-63.3
3. **temperature-init** (#3877): −0.5-2.5% expected → complementary to n_head concentration
4. **slice_num sweet-spot** (#3854): does 64 hold now that dim_head doubled with n_head=2?
5. **attention dropout** (#3858): OOD regularization without DropPath severity

Conservative compounding: 63.74 × (1−0.015) × (1−0.015) × (1−0.01) ≈ 61.2. Two more compounding wins needed to break 60.

## Operational notes

- **GitHub REST rate limit**: 3166/5000 remaining (checked 10:25 UTC). Previous 403 at ~08:00 UTC caused 2h fleet outage. Monitor.
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide.
- **Per-run budget**: 30 min wall clock, ~13-14 epochs with SwiGLU overhead.
- **Zero idle students**: 8 WIP PRs active.
