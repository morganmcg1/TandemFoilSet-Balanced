# SENPAI Research State

- **Last updated:** 2026-05-16 06:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 06:30 UTC — no open issues).

## Current best baseline (UNCHANGED, but two winner candidates pending SENPAI-RESULT)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **81.9754** | run `j5214ii4` (PR #3475 asinh-pressure, merged 03:30 UTC) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **81.3654** | run `j5214ii4` |

## 🚨 Pending merges (W&B confirmed, awaiting student SENPAI-RESULT)

Students' gh CLI has been hitting HTTP 403 rate limits — runs completed on GPU but SENPAI-RESULT comments haven't posted. Advisor comments left on all 5 stale PRs noting the W&B observations and asking students to retry submission.

| PR | Student | W&B run | val_avg | test_3split | Verdict | Status |
|---|---|---|---|---|---|---|
| **#3662** | **thorfinn** | **`699fhd8k` vel-asinh-scale-0.5** | **76.15** | **87.80** | **−7.1% — STRONGEST WIN** | Awaiting SENPAI-RESULT |
| **#3661** | **nezuko** | **`ymfjl55c` wd-1e-3-asinh** | **79.71** | **92.51** | **−2.77%** | Awaiting SENPAI-RESULT (wd=5e-3 arm still running) |
| #3659 | askeladd | `2muknt29` asinh-p-scale-1.5 | 82.16 | 99.92 | +0.22% (tied, scale=1.0 optimal) | Awaiting SENPAI-RESULT (close) |
| #3660 | frieren | `sqlj9vu5` re-sinusoidal-corrected | 96.85 | 121.77 | +18.1% regression | Awaiting SENPAI-RESULT (close) |
| #3649 | fern | `dabfzga5` n-head-8 | 98.44 | 119.06 | +20.1% regression; n_head=2 still running | Wait for n_head=2 arm |

## Active PRs (zero idle students)

### Round-5 active (training in progress, no W&B finished runs yet)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3663 | edward | dropout-sweep v2 (0.025) | Running on resubmitted arm |
| #3679 | alphonse | Huber δ sweep (0.5, 0.3) | Running |
| #3723 | tanjiro | SwiGLU MLP activation | Recently assigned, debug/Arm A in progress |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| **#3475 (askeladd)** | **asinh-pressure (scale=1.0)** | **81.9754** | **−9.53%** | Every val split improves; val_re_rand −11.8% |

## Closed PRs (all rounds)

| PR | Student | Hypothesis | Best val | Verdict |
|---|---|---|---|---|
| #3477 | thorfinn | physics-continuity | 98-106 all REGRESS | CLOSED |
| #3571 | fern | depth-sweep n_layers=6 | 93.83 (+3.55%) | CLOSED — wall-clock bound |
| #3610 | thorfinn | mlp_ratio=4 | 93.12 (+2.76%) | CLOSED — wall-clock bound |
| #3576 | nezuko | wd-sweep (old baseline) | 90.46 (TIED) | CLOSED — superseded |
| #3575 | edward | p_surf_weight=3/5 | 94.65 (+4.5%) | CLOSED — decisive regression |
| #3578 | frieren | re-sinusoidal-embed (buggy) | 130.82 (+44%) | CLOSED — frequency mismatch |
| #3577 | tanjiro | slice-num=128 (old stack) | 101.18 (+11.6%) | CLOSED — stale, pre-asinh |
| #3543 | alphonse | ema-decay-push (0.98/0.97/0.95) | 90.84 | CLOSED — exhausted below 0.99 |
| #3664 | tanjiro | slice-num=128 on asinh | 90.77 (+10.7%) | CLOSED — wall-clock bind |

## Key findings (cumulative)

### Optimization stack (Rounds 1-3)
EMA → EMA+clip+Huber → faster EMA decay (0.99) compounds cleanly. Combined: 136.89 → 90.61 (−33.8%).

### Target-side transform (Rounds 4-5)
**asinh(pressure)** at scale=1.0 → val=81.97 (−9.53%). **scale=1.0 confirmed optimal** by askeladd's Round-5 sweep (scale=1.5 ties, scale=2.0 regresses 2%).

**asinh on velocity channels** (Round 5, thorfinn) at scale=0.5 → val~76.15 — **the largest single-arm gain since the original asinh win**. Velocity is less heavy-tailed than pressure, so a softer compression (smaller scale = stronger compression for small values) hits the right point. Awaiting SENPAI-RESULT to merge.

### Regularization (Round 5)
**Weight decay 1e-3** (nezuko) → val~79.71 — solid 2.8% improvement compounding with asinh. Awaiting SENPAI-RESULT.

**Dropout 0.05** (edward) → val +0.59% (within noise) but test −0.64%, val_re_rand improved. Dose over-pressures geom_camber_rc. Retesting at 0.025.

### Falsified axes (closed in Round 5)
- **n_head=8** (fern): +20% regression, wall-clock bind to 11 epochs. n_head=2 arm pending.
- **Re-sinusoidal (corrected normalization)** (frieren): +18% regression. The sinusoidal expansion of log_re adds noise the model can't filter; raw scalar already clean.
- **asinh scale > 1.0** (askeladd): scale=1.0 confirmed optimum.

## Strategic outlook

**Imminent merges (if students post SENPAI-RESULT)**:
1. **vel-asinh scale=0.5** (#3662) → new baseline ~76.15 (−7.1%)
2. **wd=1e-3** (#3661) → could compound for ~74-75 baseline

**Still in flight**:
- Huber δ sweep (#3679 alphonse): expected 1-3% gain on top of new baseline
- SwiGLU MLP (#3723 tanjiro): expected 1-3% gain
- dropout 0.025 (#3663 edward v2): expected 0.5-1% gain

**Target**: val < 74. With three winners potentially compounding (vel-asinh, wd, Huber-δ or SwiGLU), this is in reach.

## Operational notes

- **GitHub rate limit issue**: student pods hitting HTTP 403 on gh REST. Advisor (`gh api rate_limit`: 1232/5000 core, 4111/5000 graphql) is healthy. Resets hourly (~06:40 UTC).
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide. Known dataset issue.
- Per-run budget: 30 min wall clock, 50 epoch cap (~14 epochs).
- **Zero idle students**: 8 WIP PRs (#3649, #3659-#3663, #3679, #3723).
