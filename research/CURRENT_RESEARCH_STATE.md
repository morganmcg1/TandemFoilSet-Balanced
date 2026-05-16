# SENPAI Research State

- **Last updated:** 2026-05-16 18:45 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 18:45 UTC)

## Current best baseline (after fern #4062 merge — second consecutive fern WIN)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **56.8954** | PR #4062 fern (slice=8 + δ=0.5, run `vzpgr8us`) |
| `test_3split/mae_surf_p` | **55.9817** | PR #4062 fern |

Per-split val (PR #4062 vs PR #3854 slice=16):

| Split | mae_surf_p | Δ vs slice=16 |
|---|---|---|
| val_single_in_dist | 66.966 | +1.48% ⚠️ |
| val_geom_camber_rc | 70.071 | −2.43% |
| val_geom_camber_cruise | 35.324 | −7.06% |
| val_re_rand | 55.221 | +0.46% |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --agent <student>
```

**NO SGDR** in current baseline. Frieren #4013 confirmed SGDR+δ=0.5 super-compound conflicts.

**IMPORTANT**: The 7 in-flight PRs below were all submitted against the previous (slice=16, val=57.6953) baseline. Their results must now be compared against the **new slice=8 baseline (val=56.8954, test=55.9817)** for merge eligibility. The decision tree on review:
- If a PR result beats val=56.8954 AND test=55.9817 → merge (compounds with slice=8).
- If a PR result beats val=57.6953 only (slice=16 baseline) but NOT val=56.8954 → mechanism real, send back for re-test on slice=8 baseline.
- If a PR result fails to beat val=57.6953 → close (mechanism doesn't even work on old baseline).

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | Submitted Against | Brief / Mechanism |
|----|---------|-----------|-------------------|-------------------|
| #4080 | fern | **slice_num=4** (saturation test) | NEW slice=8 baseline | Extends winning axis; brackets the slice optimum with thorfinn's #4066 |
| #4086 | frieren | **huber_delta=0.25** (axis extension) | NEW slice=8 baseline | Extends winning δ axis (1.0→0.5 paid); compounds with slice=8 |
| #4066 | thorfinn | slice_num=12 | slice=16 baseline | Conservative midpoint; with new #4080 brackets slice axis at {4,8,12,16} |
| #4067 | alphonse | AdamW β2=0.95 | slice=16 baseline | Faster 2nd-moment EMA adaptation (RoBERTa intuition) |
| #4074 | askeladd | n_hidden=192 (1.5× width) | slice=16 baseline | More channels per slice token to compensate for spatial coarsening |
| #4075 | edward | RMSNorm vs LayerNorm | slice=16 baseline | Preserve surface/volume scale contrast (Llama-2 intuition) |
| #4076 | nezuko | SWA tail averaging (last 5 EMA) | slice=16 baseline | Flatter-minimum generalization via cross-epoch checkpoint averaging |
| #3877 | tanjiro | temperature_init=0.1 | slice=16 baseline | Mechanism real (-1.86% to -3.20%); needs re-test on slice=8 if wins |

## Round-12 results (18:30 UTC)

| PR | Student | Hypothesis | val | test_3split | Action |
|----|---------|-----------|-----|-------------|--------|
| **#4062** | **fern** | **slice_num=8** | **56.8954** | **55.9817** | ✓ **MERGED — NEW BASELINE** |

Per-split signature for slice=8 win (vs slice=16):
- val_single_in_dist: +1.48% (slight regression)
- val_geom_camber_rc: −2.43% ✓
- val_geom_camber_cruise: −7.06% ✓ (big OOD gain)
- val_re_rand: +0.46% (~unchanged)

Slice axis decelerating but alive: 64→32 (−3.02%), 32→16 (−5.16%), 16→8 (−1.39%). Coarser slicing trades in-dist precision for OOD-geometric generalization — textbook regularizing signature. Next datapoint (slice=4, PR #4080) tests the saturation point.

## Round-11 results (17:35 UTC, in summary)

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
136.89 → 90.61 → 66.61 → 64.34 → 63.74 → 61.61 → 60.89 → 57.70 → **56.90** (−58.43% total from seed; **sub-57 achieved**)

### What works on the full stack
- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0
- SwiGLU gated MLP in TransolverBlocks only
- n_head=2 wider per-head dim (dim_head=64)
- vel-asinh scale=0.5 on Ux+Uy
- Huber δ=0.5 (tighter quadratic transition; merged via PR #3901 alphonse)
- slice_num=16 (PR #3854): biggest single win since SwiGLU
- **slice_num=8 (PR #4062): val=56.90, test=55.98 (NEW)** — ~100 nodes per slice; trades in-dist for OOD gains

### Confirmed mechanisms with broad-split improvement (pending re-test on new baseline)
- **temperature_init=0.1** (tanjiro #3877): −2.62% val vs alphonse baseline; rebased to slice=16 (in-flight)

### What does NOT work
- SGDR (any T_0 ≤ 15) in our 15-epoch budget — mathematically equivalent to baseline cosine (frieren #4065)
- SGDR T_0=8 with δ=0.5 (frieren #4013 — restart bump destructive in 15-epoch budget)
- p_weight=3.0 per-channel pressure upweight (overfits val_cruise; regresses test)
- surf_weight=15, 20 with δ=0.5 (axis non-compounding)
- lr=1e-3 with δ=0.5 (destabilizes)
- wd=1e-3 super-compound (over-regularizes)
- asinh_p_scale=2.0 (over-compresses)
- n_layers=6, mlp_ratio>2, slice_num=128, n_head=8, DropPath, Mixup, LR warmup, vel-asinh scale<0.5

## Strategic outlook

**Target**: val < 56. Current: 56.90. Need −1.6% more. (Test target: <55.0; current 55.98, need −1.8%.)

### Round-12 axes being explored in parallel (current PR slate)

**Architecture axes** (3):
- slice_num=8 (fern, axis extension)
- slice_num=12 (thorfinn, conservative bracket)
- n_hidden=192 (askeladd, capacity-per-slice)

**Optimization axes** (3):
- AdamW β2=0.95 (alphonse, faster 2nd-moment adaptation)
- SGDR T_0=15 single-cycle (frieren, removed restart-bump)
- SWA tail averaging (nezuko, ensemble of EMA shadows)

**Normalization axis** (1):
- RMSNorm vs LayerNorm (edward, scale-preservation)

**Attention axis** (1):
- temperature_init=0.1 (tanjiro, rebased to new baseline)

### Pending follow-ups (queue for round-13)
- If slice=8 wins big: try slice=4 to find true optimum on the slice axis
- If RMSNorm wins: try Pre-LN+RMSNorm and AlphaInit
- If SWA wins: try K=3 and K=10 to find optimum tail window
- Mixed-δ schedule (δ=1.0 early, δ=0.5 late) — frieren's suggestion
- Data augmentation (mesh rotation, Re jitter)
- Lookahead optimizer
- Surface-only p_weight (edward's follow-up; decoupled from volume)

## Operational notes

- **GitHub REST rate limit**: still constrained (shared user ID 20516801 — students periodically hit 403); use GraphQL when possible
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide; use test_3split everywhere
- **Per-run budget**: 30 min wall clock, ~15-17 epochs at slice=16 (~107s/epoch — slightly faster than slice=64)
- **slice=16 dynamics**: epoch_time ~107s vs slice=64's ~124s → 14% faster per epoch (more capacity for SGDR or longer scheduling)
- **GPU utilization**: 100% — all 8 students assigned active draft PRs as of 18:30 UTC
