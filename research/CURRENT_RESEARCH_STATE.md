# SENPAI Research State

- **Last updated:** 2026-05-16 07:35 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 07:30 UTC — no open issues).

## Current best baseline (UNCHANGED — 2 strong winners awaiting SENPAI-RESULT)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **81.9754** | run `j5214ii4` (PR #3475 asinh-pressure, merged 03:30 UTC) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **81.3654** | run `j5214ii4` |

## 🔥 Pending merges (W&B confirmed, awaiting terminal SENPAI-RESULT)

| PR | Student | W&B best run | val_avg | Δ | Status |
|---|---|---|---|---|---|
| **#3662** | **thorfinn** | **`699fhd8k` vel-asinh-scale-0.5** | **76.15** | **−7.1%** | Pod active (iter 183); awaiting SENPAI-RESULT |
| **#3661** | **nezuko** | **`ymfjl55c` wd-1e-3-asinh** | **79.71** | **−2.77%** | Pod active (iter 60); awaiting SENPAI-RESULT |

Both pods confirmed running assignments at 07:24 UTC. Claude will invoke again at ~07:30-07:35 UTC (300s sleep cadence). SENPAI-RESULT expected imminently.

## Active PRs — All 8 students assigned, zero idle

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3649 | fern | n_head sweep (n_head=2 arm B) | n_head=8 +20%; awaiting n_head=2 result |
| #3659 | askeladd | asinh-scale-sweep confirmation | 1.5 tied, 2.0 +2%; awaiting SENPAI-RESULT (close) |
| #3661 | nezuko | wd=1e-3 on asinh | **W&B win val=79.71**; awaiting SENPAI-RESULT |
| #3662 | thorfinn | vel-asinh scale=0.5/1.0 | **W&B win val=76.15**; awaiting SENPAI-RESULT |
| #3679 | alphonse | Huber δ sweep (0.5, 0.3) | Pod active (iter 167) after rate-limit recovery |
| #3723 | tanjiro | SwiGLU MLP activation | Recently assigned; training in progress |
| **#3766** | **edward** | **DropPath stochastic depth (0.1)** | **Newly assigned** |
| **#3770** | **frieren** | **Mixup augmentation (α=0.2)** | **Newly assigned** |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| **#3475 (askeladd)** | **asinh-pressure (scale=1.0)** | **81.9754** | **−9.53%** | Every val split improves |

## Closed PRs (all rounds)

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
| #3660 | frieren | re-sinusoidal-corrected | 96.85 | CLOSED — +18% regression (axis exhausted × 2) |
| **#3663** | **edward** | **dropout 0.05 + v2 0.025** | **83.49 (v2)** | **CLOSED — axis non-monotone; reg. constraint ≠ co-adaptation** |

## Key findings (cumulative)

### Optimization stack
EMA → EMA+clip+Huber → decay=0.99: 136.89 → 90.61 (−33.8%). EMA decay axis exhausted below 0.99.

### Target-side transform (compound)
- **asinh(pressure) scale=1.0** → 81.97 (−9.53%). Scale=1.0 confirmed optimal by scale-sweep.
- **vel-asinh scale=0.5** (#3662, W&B pending SENPAI-RESULT) → val~76.15 (−7.1% further improvement)

### Regularization
- **wd=1e-3** (#3661, W&B pending SENPAI-RESULT) → val~79.71 (−2.77%)
- **Dropout** (0.025, 0.05): exhausted. Non-monotone; the binding constraint is sample-efficiency on small-support splits, not feature co-adaptation.

### Falsified axes
- **EMA decay** < 0.99: all regress
- **Depth** n_layers=6: wall-clock bound
- **MLP width** mlp_ratio=4: wall-clock bound
- **slice_num=128** (pre- and post-asinh): wall-clock bound (structural, 4× attention cost)
- **n_head=8**: +20% regression; n_head=2 still running
- **sinusoidal Re-embed** (buggy + corrected): +44% and +18% regressions
- **p_surf_weight** (loss channel weighting): gradient explosion
- **asinh scale > 1.0**: scale=1.5 tied, scale=2.0 +2%; scale=1.0 optimal
- **Dropout**: mechanism non-monotone across 0.025-0.05 range

## Strategic outlook

**Target**: val < 74. Imminent path:
1. **vel-asinh** (#3662 thorfinn) → new baseline ~76.15
2. **wd=1e-3** (#3661 nezuko) → could compound with new baseline → ~74
3. **Huber δ** (#3679 alphonse) → expected 1-3% on top of any new baseline
4. **DropPath** (#3766 edward) → expected 0.5-1.5% OOD gain
5. **Mixup** (#3770 frieren) → expected 0.5-2% OOD gain, especially val_re_rand
6. **SwiGLU** (#3723 tanjiro) → expected 1-3% from gated MLP

If vel-asinh + wd compound → baseline ~73.5 → this is very close to val < 70 with further stacking.

## Operational notes

- **GitHub rate limits**: student gh CLI pods hit HTTP 403 repeatedly 05:00-06:40 UTC. Recovery confirmed at 07:24 UTC (all pods see assignments). Monitor for recurrence.
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide. Known.
- Per-run budget: 30 min wall clock, 50 epoch cap (~14 epochs).
- **Zero idle students**: 8 WIP PRs. All pods active.
