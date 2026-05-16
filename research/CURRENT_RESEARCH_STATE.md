# SENPAI Research State

- **Last updated:** 2026-05-16 09:40 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 09:30 UTC — no open issues).

## Current best baseline (UPDATED — n_head=2 merge 09:35 UTC)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **64.3427** | run `0hy5wlxj` (PR #3794 n_head=2 on SwiGLU, merged 09:35 UTC) |
| `test_3split/mae_surf_p` (3 valid splits; cruise=NaN) | **63.6663** | run `0hy5wlxj` |

Per-split validation (best arm `0hy5wlxj` vs prev baseline #3723 SwiGLU, 66.6130):

| Split | mae_surf_p | Δ vs #3723 |
|---|---|---|
| val_single_in_dist | 77.068 | **−2.30%** |
| val_geom_camber_rc | 75.996 | **−2.80%** |
| val_geom_camber_cruise | 43.741 | **−3.89%** |
| val_re_rand | 60.565 | **−5.17%** |

**Reproduce new baseline**:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 \
  --agent <student>
```

## Active PRs — Round-6 (on SwiGLU baseline val=66.61, compare against new baseline 64.34)

| PR | Student | Hypothesis | Key test |
|----|---------|-----------|----------|
| #3789 | thorfinn | vel-asinh scale=0.5 on SwiGLU | Does vel-asinh compound with SwiGLU+n_head=2? (~−7% expected) |
| #3790 | nezuko | wd=1e-3 on SwiGLU | Does wd=1e-3 compound? (~−2.8% expected) |
| #3793 | alphonse | Huber δ=0.5 on SwiGLU | Does δ=0.5 compound? (~−1.4% expected) |
| #3795 | tanjiro | SwiGLU in all MLPs (preprocess+readout) | Extend gating to I/O layers |
| #3796 | askeladd | vel-asinh fine scale sweep (0.25, 0.375) | Is scale < 0.5 better? |
| #3766 | edward | DropPath stochastic depth (rate=0.1) | Just started training (picked up at 09:21 UTC after rate-limit recovery) |

Note: Round-6 PRs were assigned against the old SwiGLU baseline (66.61). If they beat THAT baseline, they may or may not beat the new n_head=2 baseline (64.34). Review each against the current best.

## Active PRs — Round-7 (on new n_head=2+SwiGLU baseline val=64.34)

| PR | Student | Hypothesis | Key test |
|----|---------|-----------|----------|
| #3854 | fern | slice_num sweep (32, 128) with n_head=2 | Is slice_num=64 optimal for dim_head=64? |
| #3858 | frieren | attention dropout (attn_drop=0.1) in PhysicsAttention | Does attn dropout improve OOD generalization? |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| #3475 (askeladd) | asinh-pressure (scale=1.0) | 81.9754 | −9.53% | Every split improved |
| #3723 (tanjiro) | SwiGLU param-matched MLP | 66.6130 | −18.74% | Biggest single win; gating mechanism, not params |
| **#3794 (fern)** | **n_head=2 on SwiGLU** | **64.3427** | **−3.41%** | **Every split improves; val_re_rand −5.17%; also 14% faster/epoch** |

## Closed PRs — notable

| PR | Student | Hypothesis | Best val | Verdict |
|---|---|---|---|---|
| #3770 | frieren | Mixup augmentation (α=0.1, 0.2) | 105.30 | CLOSED — +28% regression; geometry-mixing breaks physical manifold |
| #3663 | edward | feature dropout 0.025/0.05 | 83.49 | CLOSED — non-monotone |
| #3660 | frieren | re-sinusoidal-corrected | 96.85 | CLOSED — +18% regression |
| #3662 | thorfinn | vel-asinh scale=0.5 (OLD stack) | 76.15 | CLOSED — re-testing on SwiGLU (#3789) |
| #3649 | fern | n_head=2 (OLD stack) | 86.78 | CLOSED — re-tested on SwiGLU (#3794, MERGED) |

## Key findings (cumulative)

### Optimization stack (Rounds 1-3)
EMA → EMA+clip+Huber → decay=0.99: 136.89 → 90.61 (−33.8%). EMA decay axis exhausted below 0.99.

### Target-side transforms (Rounds 4-5)
- **asinh(pressure) scale=1.0** → 81.97 (−9.53%). Scale=1.0 confirmed optimal.
- **vel-asinh scale=0.5** → 76.15 on old stack (−7.1%). Mechanism confirmed real. Re-testing on SwiGLU (#3789).

### Architecture — BREAKTHROUGH (Rounds 5-6)
- **SwiGLU param-matched MLP** (PR #3723) → val=66.61 (−18.74%). Gating mechanism (data-dependent channel selection) is the win.
- **n_head=2** (PR #3794) → val=64.34 (−3.41%). Wider per-head dim (64 vs 32) + 14% faster/epoch. **Every split improves.**

### Falsified axes (closed)
- EMA decay < 0.99, depth n_layers=6, mlp_ratio>2, slice_num=128 (old stacks), sinusoidal Re-embed (both), p_surf_weight, feature dropout, asinh scale > 1.0, n_head=8, Mixup augmentation

## Strategic outlook

**Target**: val < 60. Path from current best 64.34:
1. **vel-asinh compound** (#3789): if −7.1% on old stack holds on n_head=2+SwiGLU → val ≈ 59.8 (TARGET HIT)
2. **wd compound** (#3790): if −2.77% holds → val ≈ 62.6 (on its own)
3. **SwiGLU-all-MLPs** (#3795): preprocess/readout gating could unlock 1-2% more
4. **DropPath** (#3766): just started; 0.5-1.5% expected from literature
5. **slice_num fine sweep** (#3854): 0.5-3% if slice-head alignment was suboptimal at dim_head=64
6. **attention dropout** (#3858): 0.3-1.0% from attention regularization
7. **Huber δ=0.5** (#3793), **wd=1e-3** (#3790): each ~1-3% if they compound

If vel-asinh + wd compound: 64.34 × (1−0.071) × (1−0.028) ≈ 58.7 (val < 60 reached).

## Operational notes

- **GitHub REST rate limit** still periodically hitting HTTP 403 (as of 09:40 UTC). GraphQL (used by gh pr create) still works. PRs created via GraphQL successfully.
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide.
- Per-run budget: 30 min wall clock (~15 epochs at 124 s/epoch with current n_head=2 config).
- **Zero idle students**: 8 active WIP PRs.
- Edward (#3766) recovered from rate-limit storm at 09:21 UTC — just started training.
- Round-6 PRs (#3789, #3790, #3793, #3795, #3796) assigned against old baseline (66.61); review against new baseline (64.34) when results arrive.
