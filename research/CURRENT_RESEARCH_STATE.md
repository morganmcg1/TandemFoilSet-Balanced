# SENPAI Research State

- **Last updated:** 2026-05-16 15:30 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (no open issues at 15:30 UTC)

## Current best baseline (after frieren #3924 merge)

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **60.8893** | PR #3924 frieren (SGDR T_0=8 on δ=1.0 stack) |
| `test_3split/mae_surf_p` | **59.2081** | PR #3924 frieren |

**Stack caveat**: frieren's run used `--huber_delta 1.0` (assignment predated alphonse's δ=0.5 merge). Reproduce command uses `--sgdr_t0 8 --huber_delta 1.0`. The SGDR + δ=0.5 super-compound is untested but expected to be even better — PR #4013 (frieren) now verifying.

Reproduce:
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --use_swiglu --mlp_ratio 1.333 --n_head 2 --asinh_vel_scale 0.5 \
  --sgdr_t0 8 \
  --agent <student>
```

## Active PRs (8 WIP, 0 idle)

| PR | Student | Hypothesis | Status | Config vs 60.89 baseline | Notes |
|----|---------|-----------|--------|---------------------------|-------|
| #3854 | fern | slice_num=32 + δ=0.5 | WIP — training | `--slice_num 32 --huber_delta 0.5` | No SGDR; result will need rebase test |
| #3877 | tanjiro | temperature_init=0.1 + δ=0.5 | WIP — training | `--temperature_init 0.1 --huber_delta 0.5` | No SGDR; result will need rebase test |
| #3902 | nezuko | wd=1e-3 + SGDR + δ=0.5 | WIP — sent back 15:25 | Super-compound | NEW super-compound brief |
| #3907 | thorfinn | surf_weight=15 + δ=0.5 | WIP — training | `--surf_weight 15 --huber_delta 0.5` | No SGDR; rebase test |
| #3967 | edward | per-step LR warmup (500) | WIP — training | `--warmup_steps 500 --huber_delta 1.0` | No SGDR; comparable to plain cosine |
| #3986 | alphonse | surf_weight=20 + δ=0.5 | WIP — training | `--surf_weight 20 --huber_delta 0.5` | No SGDR; rebase test |
| #3987 | askeladd | lr=1e-3 + δ=0.5 | WIP — training | `--lr 1e-3 --huber_delta 0.5` | No SGDR; rebase test |
| **#4013** | **frieren** | **SGDR + δ=0.5 super-compound** | **WIP — just assigned** | `--sgdr_t0 8 --huber_delta 0.5` | Confirms compound; expected sub-60 val |

## Round-10 results (15:24 UTC)

| PR | Student | Hypothesis | val | test_3split | vs prior baseline | Action |
|----|---------|-----------|-----|-------------|-------------------|--------|
| **#3924** | **frieren** | **SGDR T_0=8** | **60.8893** | **59.2081** | −1.17% / −2.76% vs #3901 | ✓ MERGED 15:24 — NEW BASELINE |
| #3902 | nezuko | wd=1e-3 + δ=0.5 (rebase) | 61.1469 | 59.9845 | beat #3901 but superseded by #3924 | → Send back: super-compound w/ SGDR |

## Key findings (cumulative)

### Merged stack progress
136.89 → 90.61 → 66.61 → 64.34 → 63.74 → 61.61 → **60.89** (−55.5% total from seed; sub-61 achieved)

### What works on the full stack
- EMA decay=0.99, grad_clip=5.0
- asinh(pressure) scale=1.0
- SwiGLU gated MLP in TransolverBlocks only
- n_head=2 wider per-head dim
- vel-asinh scale=0.5 on Ux+Uy
- **SGDR T_0=8 warm restarts: val=60.89, test=59.21 (NEW)** — reaches low-lr fine-tuning within budget
- Huber δ=0.5 (independently won in #3901; untested w/ SGDR)

### Confirmed mechanisms, pending compound tests
- **temperature_init=0.1** (tanjiro #3877): test=59.62 on δ=1.0 baseline; rebased to δ=0.5 — strong OOD signal, may need SGDR re-test
- **surf_weight=15** (thorfinn #3907): −4.48% val on old baseline; rebased to δ=0.5 — may need SGDR re-test
- **wd=1e-3** (nezuko #3902): val=61.15 on δ=0.5; sent back for super-compound w/ SGDR
- **slice_num=32** (fern #3854): val=62.40 on δ=0.5 needed; rebased — may need SGDR re-test

### What does NOT work
- n_layers=6, mlp_ratio>2, slice_num=128, n_head=8, DropPath, SwiGLU-in-all-MLPs, Mixup, vel-asinh scale<0.5, per-channel vel-asinh (test regression), LR warmup with per-epoch step (plumbing bug), SGDR T_0=5 (too short cycles)

## Strategic outlook

**Target**: val < 60. Current: 60.89. Need −1.5% more for sub-60.

High-confidence path:
1. **frieren #4013 SGDR + δ=0.5 super-compound**: a clean compound test of the two latest mechanisms. δ=1.0→δ=0.5 swap on the prior baseline delivered −3.3% val. If preserved with SGDR: val ∈ [58.8, 60.3] → **strong sub-60 candidate**.
2. **nezuko #3902 rebase to wd + SGDR + δ=0.5**: triple-compound. Orthogonal mechanisms (wd is regularization, SGDR is schedule, δ is loss shape). Each won independently — compound expected to break val<60.
3. **tanjiro temperature_init=0.1 + SGDR + δ=0.5** (after current rebase completes): test=59.62 already on δ=1.0; with SGDR + δ=0.5 should compound.

Speculative but promising:
4. **Re-test surf_weight=15 on new baseline** (thorfinn): once SGDR + δ=0.5 confirmed
5. **lr=1e-3 + SGDR** interaction (askeladd): may need different SGDR T_0 with higher base lr
6. **edward per-step warmup on SGDR**: warmup before first SGDR cycle should be cleaner
7. **slice_num=16** if 32 confirms on new baseline (fern)

## Queued hypotheses (from researcher-agent 13:30 UTC)

To assign once compound tests complete:
- **H1 p_weight=3.0**: 3-line code change. Pressure-component loss weighting axis untested. Top pick.
- **H2 asinh_p_scale=2.0**: zero code change; pressure compression sweep
- **H3 Re_x boundary-layer feature**: feature engineering with physics-informed Re scaling
- **H4 EMA decay ramp-up cosine schedule**: dynamic EMA decay
- **H5 LLRD decay=0.7**: layer-wise learning rate decay

## Operational notes

- **GitHub REST rate limit**: still constrained (shared user ID 20516801 exhausted multiple times in last 2h); use GraphQL for queries when possible
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide (affects test_avg; use test_3split everywhere)
- **Per-run budget**: 30 min wall clock, ~15 epochs with n_head=2 at 124s/epoch — SGDR T_0=8 fits 1 full + 1 partial cycle exactly
- **SGDR plumbing now in train.py**: `--sgdr_t0 N` enables CosineAnnealingWarmRestarts; if set, plain cosine schedule is replaced
