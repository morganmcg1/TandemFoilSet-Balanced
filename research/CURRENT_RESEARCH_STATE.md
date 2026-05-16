# SENPAI Research State

- **Last updated:** 2026-05-16 20:55 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 20:55 UTC)

## Current best baseline (after fern #4062 merge — second consecutive fern WIN)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **56.8954** | PR #4062 fern (slice=8 + δ=0.5, run `vzpgr8us`) |
| `test_3split/mae_surf_p` | **55.9817** | PR #4062 fern |

Per-split val (PR #4062):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 66.966 |
| val_geom_camber_rc | 70.071 ← **dominant residual** |
| val_geom_camber_cruise | 35.324 |
| val_re_rand | 55.221 |

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --slice_num 8 \
  --agent <student>
```

**NO SGDR** in current baseline. Frieren #4013 confirmed SGDR+δ=0.5 super-compound conflicts.

## Plateau status

**8 consecutive closes since the fern #4062 merge at 18:40 UTC** (no improvement for ~2h15m):
1. #4065 frieren SGDR T_0=15 — math equivalent to baseline cosine
2. #4080 fern slice=4 — capacity cliff at 4 slice tokens
3. #4075 edward RMSNorm — mixed (helps OOD test, hurts in-dist val)
4. #3877 tanjiro temp_init=0.1 (slice=16 retest) — slice/temp coupled on softmax-sharpness axis
5. #4100 fern n_head=4 — dim_head=32 lost OOD-camber generalization
6. #4086 frieren δ=0.25 (3-seed) — δ axis saturated past 0.5
7. #4076 nezuko SWA K=5 — SWA needs converged trajectory we don't have
8. **#4066 thorfinn slice=12 (2-seed, val=59.22/60.52) — slice axis bracket closed; non-monotonic between 8 and 16**

**Slice axis bracket** is now fully closed: {4 cliff, 8 winner, 12 cliff, 16 prior, 32 worse, 64 worse}. Non-monotonicity between 8 and 16 is a clean finding — single-seed variance (±3) doesn't fully explain a 2-seed mean of ~59.9 between two endpoints at 56.9 and 57.7.

Strategy: 1 more round of orthogonal-axis exploration before invoking researcher-agent. Round-13 now has 4 new orthogonal axes (regularization/asymHuber/Lookahead/LLRD) — if ALL fail, escalate.

## Active PRs (8 WIP, 0 idle — zero idle GPUs)

| PR | Student | Hypothesis | Submitted Against | Brief / Mechanism |
|----|---------|-----------|-------------------|-------------------|
| **#4138** | **fern** | **attn_dropout=mlp_dropout=0.1** | NEW slice=8 baseline | Regularization: orthogonal to EMA/wd; targets dropout axis (untested ever) |
| **#4141** | **frieren** | **asymmetric Huber (δ_pos=0.25, δ_neg=1.0)** | NEW slice=8 baseline | Loss: pushes under-prediction harder than over-prediction (their suggestion) |
| **#4142** | **nezuko** | **Lookahead optimizer (k=5, α=0.5)** | NEW slice=8 baseline | Optimizer: in-training averaging (k-step inner loop); fixes SWA's convergence requirement |
| **#4151** | **thorfinn** | **Layer-wise LR decay (factor=0.85)** | NEW slice=8 baseline | Optimizer: per-layer LR scaling; preserves early features, adapts late layers (BERT/ViT proven) |
| #4101 | edward | asinh_vel_scale=1.0 | NEW slice=8 baseline | Data: velocity-scale axis extension; symmetric with asinh_p_scale |
| #4102 | tanjiro | temperature_init=0.7 (diffuse) | NEW slice=8 baseline | Architecture: dead-slice hypothesis; sign-flip from closed #3877 |
| #4067 | alphonse | AdamW β2=0.95 | slice=16 baseline | Optimizer: faster 2nd-moment EMA adaptation; healthy `3pc74k8f` ETA terminal ~21:05 UTC |
| #4074 | askeladd | n_hidden=192 (1.5× width) | slice=16 baseline | Capacity: more channels per slice token; 3rd attempt `hapwhewl` started 20:43 UTC |

## Round-12 results (18:30-20:30 UTC, 8 PRs)

| PR | Student | Hypothesis | val | test_3split | Action |
|----|---------|-----------|-----|-------------|--------|
| **#4062** | **fern** | **slice_num=8** | **56.8954** | **55.9817** | ✓ **MERGED — NEW BASELINE** |
| #4065 | frieren | SGDR T_0=15 single cycle | 59.32 | — | ✗ Closed — equiv to baseline cosine |
| #4080 | fern | slice_num=4 | 61.5+ | — | ✗ Closed — capacity cliff at 4 slice tokens |
| #4075 | edward | RMSNorm replacing LayerNorm | 58.0+ | mixed | ✗ Closed — partial mechanism (OOD-test only) |
| #3877 | tanjiro | temp_init=0.1 (slice=16 retest) | 58.21 | — | ✗ Closed — slice/temp coupling confirmed |
| #4100 | fern | n_head=4 (dim_head=32) | 58.23 | 57.16 | ✗ Closed — head-dim too narrow at slice=8 |
| #4086 | frieren | huber_delta=0.25 (3-seed) | 60.02 | — | ✗ Closed — δ axis fully saturated past 0.5 |
| #4076 | nezuko | SWA K=5 tail averaging | 60.49 (swa) / 59.28 (final) | — | ✗ Closed — SWA needs converged trajectory |

## Key findings (cumulative)

### Merged stack progression
136.89 → 90.61 → 66.61 → 64.34 → 63.74 → 61.61 → 60.89 → 57.70 → **56.90** (−58.43% total from seed; **sub-57 achieved**)

### What works on the full stack
- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0
- SwiGLU gated MLP in TransolverBlocks only
- n_head=2 wider per-head dim (dim_head=64)
- vel-asinh scale=0.5 on Ux+Uy
- Huber δ=0.5 (tighter quadratic transition; PR #3901 alphonse)
- slice_num=16 (PR #3854): biggest single win since SwiGLU
- **slice_num=8 (PR #4062): val=56.90, test=55.98 (NEW)** — ~100 nodes per slice; trades in-dist for OOD gains

### What does NOT work
- SGDR (any T_0 ≤ 15) in our 15-epoch budget — mathematically equivalent to baseline cosine
- SGDR T_0=8 with δ=0.5 — restart bump destructive in 15-epoch budget
- slice_num=4 on slice=8 stack — capacity cliff at 4 slice tokens
- slice_num=12 (in-flight) likely small/no effect — slice axis bracketed by 8 and 16
- RMSNorm replacing LayerNorm — partial mechanism; in-dist val regresses
- temperature_init=0.1 at low slice_num — coupled with slice_num on softmax-sharpness axis; helpful only at slice_num ≥ 32
- p_weight=3.0 per-channel pressure upweight — overfits val_cruise; regresses test
- surf_weight=15, 20 with δ=0.5 — axis non-compounding
- huber_delta=0.25 (3-seed confirmation) — δ axis saturated past 0.5
- lr=1e-3 with δ=0.5 — destabilizes
- n_head=4 (dim_head=32) at slice=8 — lost OOD-camber generalization
- SWA K=5 tail averaging — needs converged trajectory; val swing −7 MAE in last 5 epochs
- wd=1e-3 super-compound — over-regularizes
- asinh_p_scale=2.0 — over-compresses
- n_layers=6, mlp_ratio>2, slice_num=128, n_head=8, DropPath, Mixup, LR warmup, vel-asinh scale<0.5 (old baseline closures)

## Strategic outlook

**Target**: val < 56. Current: 56.90. Need −1.6% more. (Test target: <55.0; current 55.98, need −1.8%.)

### Round-13 axes being explored in parallel (current PR slate)

**Regularization axis** (1, NEW):
- attn_dropout=mlp_dropout=0.1 (fern #4138) — untested ever

**Loss axis** (1, NEW):
- Asymmetric Huber δ_pos=0.25 / δ_neg=1.0 (frieren #4141) — pushes under-prediction harder

**Optimizer axes** (3):
- AdamW β2=0.95 (alphonse #4067, slice=16)
- Lookahead k=5 α=0.5 wrapping AdamW (nezuko #4142, NEW) — in-training averaging
- **Layer-wise LR decay 0.85** (thorfinn #4151, NEW) — per-layer LR scaling (BERT/ViT)

**Architecture axes** (2):
- n_hidden=192 (askeladd #4074, capacity-per-slice on slice=16 stack)
- temperature_init=0.7 (tanjiro #4102, diffuse-softmax hypothesis at slice=8)

**Data axis** (1):
- asinh_vel_scale=1.0 (edward #4101, axis extension upward)

### Pending follow-ups (queue for round-14)
- If dropout wins: sweep at {0.05, 0.1, 0.2} to find optimum
- If asymmetric Huber wins: try sign-flip ({1.0, 0.25}) to confirm direction
- If Lookahead wins: try k=10 and α=0.3 to find optimum cadence
- If multiple round-13 fail: **invoke researcher-agent for bold new directions** (per plateau protocol — bigger swings: SAM, layer-wise LR decay, AGC, divergence-free physics loss, knowledge distillation)
- Mixed-δ schedule (δ=1.0 early, δ=0.5 late) — frieren's prior suggestion
- Data augmentation (mesh rotation, Re jitter, target perturbation)
- Surface-only p_weight (edward's follow-up; decoupled from volume)

## Operational notes

- **GitHub REST rate limit**: still constrained (shared user ID 20516801); use GraphQL when possible
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide; use test_3split everywhere
- **Per-run budget**: 30 min wall clock, ~15-17 epochs at slice=8 (~107s/epoch)
- **Single-seed variance**: ≈±3 val_avg units (frieren 3-seed measurement)
- **stale_wip handling**: bump with status comment; do not assume crash — verify via kubectl pods + W&B run state
- **GPU utilization**: 100% — all 8 students assigned active draft PRs as of 20:35 UTC
