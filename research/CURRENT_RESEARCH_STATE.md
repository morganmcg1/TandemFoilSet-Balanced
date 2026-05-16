# SENPAI Research State

- **Last updated:** 2026-05-16 17:45 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 17:45 UTC)

## Current best baseline (after fern #3854 merge — MASSIVE WIN)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **57.6953** | PR #3854 fern (slice=16 + δ=0.5, run `bg8etivu`) |
| `test_3split/mae_surf_p` | **56.8613** | PR #3854 fern |

Per-split val (PR #3854):

| Split | mae_surf_p | Δ vs #3924 baseline |
|---|---|---|
| val_single_in_dist | 65.990 | −4.95% |
| val_geom_camber_rc | 71.815 | −3.24% |
| val_geom_camber_cruise | 38.006 | −6.17% |
| val_re_rand | 54.970 | −7.45% |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 16 \
  --agent <student>
```

**NO SGDR** in current baseline. Frieren #4013 confirmed SGDR+δ=0.5 super-compound conflicts.

## Active PRs (2 WIP, 6 idle students)

| PR | Student | Hypothesis | Status | Notes |
|----|---------|-----------|--------|-------|
| #4062 | fern | slice_num=8 (axis extension) | WIP | Highest confidence; direct extension of winning axis |
| #3877 | tanjiro | temperature_init=0.1 on slice=16 baseline | WIP — draft rebase | Mechanism real (-1.86% to -3.20% per-split val on old baseline); needs re-test on new baseline |

**Idle students** (6 — awaiting research-agent ideas due back ~17:50 UTC):
willowpai2i48h2-alphonse, willowpai2i48h2-askeladd, willowpai2i48h2-edward, willowpai2i48h2-frieren, willowpai2i48h2-nezuko, willowpai2i48h2-thorfinn

## Round-11 results (17:35 UTC)

| PR | Student | Hypothesis | val | test_3split | Action |
|----|---------|-----------|-----|-------------|--------|
| **#3854** | **fern** | **slice=16+δ=0.5** | **57.6953** | **56.8613** | ✓ **MERGED — NEW BASELINE** |
| #3877 | tanjiro | temp_init=0.1+δ=0.5+SGDR | 59.9942 | 59.4763 | → Send back: rebase to slice=16 baseline |
| #4017 | edward | p_weight=3.0+SGDR (2 seeds) | 60.29 / 62.50 | 60.34 / 61.00 | ✗ Closed — student verdict |
| #3986 | alphonse | surf_weight=20+δ=0.5 | 61.93 | 60.48 | ✗ Closed — regression |
| #4013 | frieren | SGDR+δ=0.5 super-compound | 62.61 | 61.10 | ✗ Closed — needs more low-lr time than budget |
| #3907 | thorfinn | surf_weight=15+δ=0.5 | 62.96 / 69.70 | 69.86 | ✗ Closed — mechanism non-compounding with δ=0.5 |
| #4035 | nezuko | asinh_p_scale=2.0 | 73.02 | — | ✗ Closed — over-compression (hit close threshold) |
| #3987 | askeladd | lr=1e-3+δ=0.5 | 74.14 | — | ✗ Closed — destabilizes loss landscape |

## Key findings (cumulative)

### Merged stack progression
136.89 → 90.61 → 66.61 → 64.34 → 63.74 → 61.61 → 60.89 → **57.70** (−57.85% total from seed; **sub-58 achieved**)

### What works on the full stack
- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0
- SwiGLU gated MLP in TransolverBlocks only
- n_head=2 wider per-head dim (dim_head=64)
- vel-asinh scale=0.5 on Ux+Uy
- Huber δ=0.5 (tighter quadratic transition; merged via PR #3901 alphonse)
- **slice_num=16 (coarser slicing): val=57.70, test=56.86 (NEW)** — ~50 nodes per slice; biggest single win since SwiGLU

### Confirmed mechanisms with broad-split improvement (pending re-test on new baseline)
- **temperature_init=0.1** (tanjiro #3877): −2.62% val vs alphonse baseline; rebased to slice=16 (in-flight)

### What does NOT work
- SGDR (any T_0) with δ=0.5 (insufficient low-lr time at 15-epoch budget)
- p_weight=3.0 per-channel pressure upweight (overfits val_cruise; regresses test)
- surf_weight=15, 20 with δ=0.5 (axis non-compounding)
- lr=1e-3 with δ=0.5 (destabilizes)
- wd=1e-3 super-compound (over-regularizes)
- asinh_p_scale=2.0 (over-compresses)
- n_layers=6, mlp_ratio>2, slice_num=128, n_head=8, DropPath, Mixup, LR warmup, vel-asinh scale<0.5

## Strategic outlook

**Target**: val < 56 (we just broke sub-58 with fern's slice=16). Current: 57.70. Need −2.9% more.

### High-confidence direct extension paths
1. **slice_num=8** (assigned to fern #4062): direct extension of the winning axis; 64→32 gave −3.02%, 32→16 gave −5.16% (accelerating). Expected val ∈ [54.5, 58.0].
2. **temperature_init=0.1 on slice=16** (in-flight tanjiro #3877): val=59.99 on old baseline; expect val=[55.5, 57.5] on new baseline.

### Open mechanisms not yet tested on new baseline
- **SGDR T_0=15 single-cycle + δ=0.5**: frieren's own suggestion; removes the destructive restart-bump that killed PR #4013. Single cycle covers whole budget.
- **AdamW betas tuning (β1=0.95)**: different momentum may help slice=16 dynamics.
- **EMA decay=0.995**: slower decay for sharper-minimum new baseline.
- **Best-checkpoint averaging (SWA across last 3 epochs)**: untested averaging technique.
- **Surface-only p_weight (edward's follow-up)**: decoupled from volume.
- **MLP ratio ↑ at slice=16**: coarser slicing may have unlocked MLP capacity.
- **n_hidden=192 at slice=16**: more channels per node.

### Speculative directions
- Mixed-δ schedule (δ=1.0 early, δ=0.5 late) — frieren's suggestion
- Data augmentation (mesh rotation, Re jitter)
- RMSNorm vs LayerNorm
- Lookahead optimizer

## Queued hypotheses
- **Research-agent ideas** (due ~17:50 UTC) — pending background completion

## Operational notes

- **GitHub REST rate limit**: still constrained (shared user ID 20516801 — students periodically hit 403); use GraphQL when possible
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide; use test_3split everywhere
- **Per-run budget**: 30 min wall clock, ~15-17 epochs at slice=16 (~107s/epoch — slightly faster than slice=64)
- **slice=16 dynamics**: epoch_time ~107s vs slice=64's ~124s → 14% faster per epoch (more capacity for SGDR or longer scheduling)
