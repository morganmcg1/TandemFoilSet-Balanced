# SENPAI Research State

- **Last updated:** 2026-05-16 12:30 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 12:25 UTC — no open issues)

## ⚠️ MASSIVE round in progress — 6 pending wins observed in W&B, all blocked by student-pod rate limit

Rate-limit struck student GH credentials again ~11:50 UTC. All students have completed runs in W&B but cannot post terminal SENPAI-RESULT markers. Advisor has nudged each PR with W&B-observed metrics. Once students recover (likely 1-2h), merging round begins.

### W&B-observed results (PENDING terminal markers + test_3split)

| PR | Student | Hypothesis | val_avg (W&B) | Δ vs 63.74 | Status |
|----|---------|-----------|---------------|-----------|--------|
| **#3907** | **thorfinn** | **surf_weight=15** | **60.885** | **−4.48%** | **BIG WIN; replicate running** |
| **#3901** | **alphonse** | **Huber δ=0.5 compound** | **61.611** | **−3.34%** | **WIN; single run** |
| **#3854** | **fern** | **slice_num=32** | **62.40** | **−2.10%** | **WIN but 3 followup crashed** |
| **#3902** | **nezuko** | **wd=1e-3 compound** | **62.670** | **−1.68%** | **WIN; Arm B (5e-3) running** |
| **#3877** | **tanjiro** | **temperature_init=0.2** | **62.826** | **−1.43%** | **WIN; Arm B (0.1) running** |
| #3903 | askeladd | per-channel vel-asinh ux=0.5 uy=0.3 | 63.546 | −0.30% | Marginal (within variance) |
| #3874 | edward | LR warmup 1ep | 65.211 | +2.31% | REGRESSION; 2 replicates diverged |
| #3924 | frieren | SGDR T_0=5 | running | — | Just started |

**Critical gap**: NO student has logged `test_3split/mae_surf_p` in W&B summary. Without test metrics I cannot merge — nudges explicitly request test metric via checkpoint re-eval or `--skip_test False` re-run.

**Merge plan (when terminals land)**:
1. Merge thorfinn #3907 first (biggest win)
2. After merge, send back alphonse/nezuko/tanjiro/fern for rebase+re-test on new baseline (60.88)
3. Close edward (regression) and possibly askeladd (marginal)

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **63.7383** | PR #3789 thorfinn (vel-asinh s=0.5 + n_head=2 + SwiGLU, merged 10:55 UTC) |
| `test_3split/mae_surf_p` (cruise=NaN) | **62.9264** | PR #3789 thorfinn |

Per-split validation (hy29un5q vs prev n_head=2 baseline 64.34):

| Split | mae_surf_p | Δ |
|---|---|---|
| val_single_in_dist | 72.7317 | **−5.62%** |
| val_geom_camber_rc | 78.3846 | +0.26% |
| val_geom_camber_cruise | 43.6151 | **−0.29%** |
| val_re_rand | 60.2217 | **−0.57%** |

**Reproduce new baseline**:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 \
  --n_head 2 \
  --asinh_vel_scale 0.5 \
  --agent <student>
```

## Active PRs — All rounds (8 WIP, 0 idle)

| PR | Student | Hypothesis | Status | Expected |
|----|---------|-----------|--------|----------|
| #3854 | fern | slice_num fine sweep (32, 128) with n_head=2 | WIP — training | New axis |
| #3874 | edward | LR warmup (1-2 ep) on SwiGLU + n_head=2 | WIP — training | −0.5-2.5% |
| #3877 | tanjiro | PhysicsAttention temperature_init=0.2 | WIP — training | −0.5-2.5% |
| #3901 | alphonse | Huber δ=0.5 compound on full stack | WIP — training | −1-2% |
| #3902 | nezuko | wd=1e-3 compound on full stack | WIP — training | −1-2% |
| #3903 | askeladd | vel-asinh per-channel (Ux≠Uy) | WIP — training | −1-3% |
| #3907 | thorfinn | surf_weight sweep (15, 20) on full stack | WIP — training | −1-4% |
| #3924 | frieren | SGDR warm restarts (T_0=5) on full stack | WIP — just assigned | −2-5% |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| #3475 (askeladd) | asinh-pressure (scale=1.0) | 81.9754 | −9.53% | Every split improved |
| **#3723 (tanjiro)** | **SwiGLU param-matched MLP** | **66.6130** | **−18.74%** | Gating mechanism; blocks-only scope |
| **#3794 (fern)** | **n_head=2 on SwiGLU** | **64.3427** | **−3.41%** | 14% faster/epoch → 2 bonus epochs |
| **#3789 (thorfinn)** | **vel-asinh s=0.5 on n_head=2+SwiGLU** | **63.7383** | **−0.93%** | Single_in_dist −5.62%; scale=0.5 confirmed optimum |

## Key findings (cumulative)

### Merged stack (Rounds 1-7, chronological)
136.89 → 90.61 → 66.61 → 64.34 → **63.74** (−53.5% total from seed)

### What works on the full stack
- EMA, grad_clip, Huber δ=1.0 (Round 1-3)
- asinh(pressure) scale=1.0 (Round 4)
- SwiGLU gated MLP in TransolverBlocks only (Round 5)
- n_head=2 wider per-head dim (Round 6-7)
- vel-asinh scale=0.5 on Ux+Uy (Round 7-8)

### What does NOT work
- EMA decay < 0.99, depth n_layers=6, mlp_ratio>2, slice_num=128, sinusoidal Re-embed, p_surf_weight, feature dropout, asinh scale > 1.0, n_head=8, DropPath, SwiGLU-in-all-MLPs, Mixup, vel-asinh scale < 0.5 (over-compresses), attention dropout (rate=0.1 — OOD gain offset by in-dist regression at this scale)

### Confirmed mechanisms on SwiGLU-only baseline (need re-test on full stack)
- Huber δ=0.5: −1.62% on SwiGLU-only → compound test #3901
- wd=1e-3: −1.46% on SwiGLU-only → compound test #3902
- Both mechanisms are real; whether they compound on the full stack is the open question

## Strategic outlook

**Target**: val < 60. Current: 63.74. Need −5.9% more.

Path:
1. **Compound tests** (#3901 Huber, #3902 wd): if each delivers ~1.5%, that's ~3% → val ~61.7
2. **Per-channel vel-asinh** (#3903): if Ux≠Uy optimal, possibly −1-2% more
3. **LR warmup** (#3874): untested schedule axis, −0.5-2.5% expected
4. **Temperature-init** (#3877): architecture-internal, −0.5-2.5% expected
5. **Slice_num sweep** (#3854): does 64 still hold with dim_head=64 in n_head=2?
6. **Attention dropout** (#3858): OOD regularization axis

Conservative compound path: 63.74 × (1−0.015) × (1−0.015) × (1−0.01) ≈ 61.2. Three more compounding wins needed for <60.

## Operational notes

- thorfinn re-assigned post-merge → #3907 surf_weight; frieren re-assigned post-close → #3924 SGDR
- **GitHub REST rate limit**: 2900/5000 remaining (checked 10:55 UTC). Monitor.
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide (affects test_avg; use test_3split everywhere).
- **Per-run budget**: 30 min wall clock, ~15 epochs with n_head=2 at 124s/epoch.
- **Zero idle students**: 8 WIP PRs active (thorfinn technically idle but getting fresh assignment).

## Queued hypotheses (next idle assignments)

(All H-01 through H-07 from researcher-agent now in flight. Need researcher-agent re-run if more students go idle.)
| Source | Hypothesis | Expected |
|----|-----------|---------|
| frieren follow-up | Slice-diagonal-preserving attention dropout | Speculative — addresses #3858 failure mode |
| askeladd follow-up | Box-Cox power transform on velocity channels (not asinh shape) | Speculative — different target compression family |
| researcher needed | Fresh axes when next idle student appears | tbd |
