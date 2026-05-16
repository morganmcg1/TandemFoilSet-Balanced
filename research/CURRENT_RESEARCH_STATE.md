# SENPAI Research State

- **Last updated:** 2026-05-16 08:00 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 07:55 UTC — no open issues).

## Current best baseline (UPDATED — SwiGLU merge 07:50 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **66.6130** | run `ju2azfzk` (PR #3723 SwiGLU param-matched, merged 07:50 UTC) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **65.4628** | run `ju2azfzk` |

Per-split validation (best arm `ju2azfzk` vs prev baseline #3475, 81.9754):

| Split | mae_surf_p | Δ vs #3475 |
|---|---|---|
| val_single_in_dist | 78.885 | **−21.9%** |
| val_geom_camber_rc | 78.184 | **−13.8%** |
| val_geom_camber_cruise | 45.513 | **−24.0%** |
| val_re_rand | 63.870 | **−16.2%** |

**Reproduce new baseline**:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --agent <student>
```

## Active PRs — Round-6 (all on new SwiGLU baseline val=66.61)

| PR | Student | Hypothesis | Key test |
|----|---------|-----------|----------|
| #3789 | thorfinn | vel-asinh scale=0.5 on SwiGLU | Does vel-asinh compound with SwiGLU? (~−7% expected) |
| #3790 | nezuko | wd=1e-3 on SwiGLU | Does wd=1e-3 compound? (~−2.8% expected) |
| #3793 | alphonse | Huber δ=0.5 on SwiGLU | Does δ=0.5 compound? (~−1.4% expected) |
| #3794 | fern | n_head=2 on SwiGLU | Larger per-head dim + gated MLP interaction |
| #3795 | tanjiro | SwiGLU in all MLPs (preprocess+readout) | Extend gating to I/O layers |
| #3796 | askeladd | vel-asinh fine scale sweep (0.25, 0.375) | Is scale < 0.5 better on lighter-tailed velocity? |

### Round-5 carry-overs still WIP (assigned before SwiGLU)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3766 | edward | DropPath stochastic depth (rate=0.1) | Running |
| #3770 | frieren | Mixup augmentation (α=0.2) | Running |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| #3475 (askeladd) | asinh-pressure (scale=1.0) | 81.9754 | −9.53% | Every split improved |
| **#3723 (tanjiro)** | **SwiGLU param-matched MLP** | **66.6130** | **−18.74%** | **Biggest single win; gating mechanism, not params; every split −13-24%** |

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
| #3662 | thorfinn | vel-asinh scale=0.5 | 76.15 | CLOSED — beat old baseline; re-testing on SwiGLU |
| #3661 | nezuko | wd=1e-3 | 79.71 | CLOSED — beat old baseline; re-testing on SwiGLU |
| #3679 | alphonse | Huber δ=0.5 | 80.85 | CLOSED — beat old baseline; re-testing on SwiGLU |
| #3659 | askeladd | asinh scale=1.5 | 82.16 | CLOSED — scale=1.0 optimal |
| #3649 | fern | n_head=2 (pre-asinh) | 86.78 | CLOSED — merge conflict; re-testing on SwiGLU |

## Key findings (cumulative)

### Optimization stack (Rounds 1-3)
EMA → EMA+clip+Huber → decay=0.99: 136.89 → 90.61 (−33.8%). EMA decay axis exhausted below 0.99.

### Target-side transforms (Rounds 4-5)
- **asinh(pressure) scale=1.0** → 81.97 (−9.53%). Scale=1.0 confirmed optimal.
- **vel-asinh scale=0.5** → 76.15 on old stack (−7.1%). Mechanism confirmed real. Re-testing on SwiGLU (#3789).

### Architecture — BREAKTHROUGH (Round 5/6)
- **SwiGLU param-matched MLP** (PR #3723) → val=66.61 (−18.74%). Largest single improvement. Gating mechanism (not params) is the win — data-dependent channel selection per MLP block, each CFD node independently decides what features to use.

### Confirmed real on old stack, pending re-test on SwiGLU
- **wd=1e-3** (nezuko): −2.77%. Re-testing (#3790).
- **Huber δ=0.5** (alphonse): −1.37%. Re-testing (#3793).
- **n_head=2** (fern): −4.2% on OLD pre-asinh stack. Re-testing (#3794).

### Falsified axes (closed)
EMA decay < 0.99, depth n_layers=6, mlp_ratio>2, slice_num=128 (both stacks), sinusoidal Re-embed (both normalizations), p_surf_weight, feature dropout (0.025-0.05), asinh scale > 1.0, n_head=8.

## Strategic outlook

**Target**: val < 60. Path:
1. **vel-asinh compound** (#3789): if −7.1% on old stack holds on SwiGLU → val ≈ 62
2. **wd compound** (#3790): if −2.77% holds → val ≈ 64.8 (on its own), possibly compounding with vel-asinh
3. **SwiGLU-all-MLPs** (#3795): preprocess/readout gating could unlock 1-2% more
4. **n_head=2** (#3794): −4.2% on old stack; if holds on SwiGLU → val ≈ 63.9
5. **vel-scale < 0.5** (#3796): may unlock 1-2% additional

If vel-asinh + wd + Huber-δ compound: 66.61 × (1−0.071) × (1−0.028) × (1−0.014) ≈ 60.4.

## Operational notes

- **GitHub REST rate limit exhausted** (07:43 UTC, resets ~08:37 UTC). GraphQL: 2710/5000 remaining. PRs created via GraphQL successfully.
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide.
- Per-run budget: 30 min wall clock, 50 epoch cap (~13 epochs with SwiGLU +6.7% per-epoch overhead).
- **Zero idle students**: 8 WIP PRs (#3766, #3770, #3789-#3790, #3793-#3796).
