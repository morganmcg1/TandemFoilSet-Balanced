# SENPAI Research State

- **Last updated:** 2026-05-16 12:50 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 12:37 UTC — no open issues)

## ⚠️ MERGE WAVE IMMINENT — 2 terminals confirmed, REST API recovers at 13:20 UTC

Student GH credentials (user ID 20516801) hit REST rate limit ~11:50 UTC. REST exhausted as of ~12:45 UTC; resets at **13:20 UTC**. GraphQL API still available (3127/5000 remaining).

Since 12:30 UTC, three more terminal SENPAI-RESULTs arrived (via GraphQL polling):

### Confirmed terminal SENPAI-RESULTs (ready for merge/close at 13:20 UTC)

| PR | Student | Hypothesis | val_avg | test_3split | Action |
|----|---------|-----------|---------|-------------|--------|
| **#3901** | **alphonse** | **Huber δ=0.5 compound** | **61.6105** | **60.8910** | **MERGE FIRST** |
| **#3854** | **fern** | **slice_num=32** | **62.3992** | **60.8933** | **Evaluate vs post-alphonse baseline** |
| #3903 | askeladd | per-channel vel-asinh ux=0.5 uy=0.3 | 63.5458 | 63.9217 (+1.58% test regression!) | **CLOSE** |

### Still pending terminals (W&B wins observed, no terminal yet)

| PR | Student | Hypothesis | W&B val | vs current baseline | Status |
|----|---------|-----------|---------|---------------------|--------|
| **#3907** | **thorfinn** | **surf_weight=15** | **60.885** | **−4.48%** | **BIGGEST WIN — pending terminal** |
| #3902 | nezuko | wd=1e-3 compound | 62.670 | −1.68% | Arm B running |
| #3877 | tanjiro | temp_init=0.2 | 62.826 | −1.43% | Arm B running |
| #3924 | frieren | SGDR T_0=5 | running | — | ~30 min into training |

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **63.7383** | PR #3789 thorfinn (vel-asinh s=0.5 + n_head=2 + SwiGLU) |
| `test_3split/mae_surf_p` | **62.9264** | PR #3789 thorfinn |

## Planned merge sequence (at 13:20 UTC REST reset)

1. **Merge alphonse #3901** (val=61.6105, test=60.8910) → new baseline ~61.61/60.89
2. **Evaluate fern #3854** (val=62.3992) vs new alphonse baseline:
   - fern val=62.40 > alphonse val=61.61 → fern DOES NOT beat new baseline on val
   - fern test=60.8933 ≈ alphonse test=60.8910 → fern tied on test
   - Decision: send fern back for rebase + re-run on new 61.61 baseline, explore slice_num=16 per her suggestion
3. **Close askeladd #3903** (test regression, marginal val)
4. **Wait for thorfinn #3907 terminal** — still the BIGGEST WIN (~60.88 vs new ~61.61 baseline → still a win)
5. **After thorfinn merges** (new baseline ~60.88), send nezuko + tanjiro for rebase + re-test

## Active PRs (8 WIP, 0 idle)

| PR | Student | Hypothesis | Status | val (W&B) |
|----|---------|-----------|--------|-----------|
| #3907 | thorfinn | surf_weight=15, 20 sweep | Pending terminal | 60.885 (WIN) |
| #3902 | nezuko | wd=1e-3 compound | Arm B running | 62.670 (WIN) |
| #3877 | tanjiro | temp_init=0.2 | Arm B running | 62.826 (WIN) |
| #3924 | frieren | SGDR T_0=5 | Training ~13:00 UTC | 79.94 mid-run |
| #3901 | alphonse | Huber δ=0.5 compound | **Terminal received → MERGE PENDING** | 61.611 |
| #3854 | fern | slice_num=32 vs 128 | **Terminal received → evaluate vs new baseline** | 62.399 |
| #3903 | askeladd | per-channel vel-asinh | **Terminal received → CLOSE PENDING** | 63.546 |
| #3967 | edward | per-step LR warmup (500 batches) | WIP — just assigned | — |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | |
| #3474 (alphonse) | EMA decay=0.99 | 90.6131 | −4.0% | |
| #3475 (askeladd) | asinh-pressure (scale=1.0) | 81.9754 | −9.53% | |
| **#3723 (tanjiro)** | **SwiGLU param-matched MLP** | **66.6130** | **−18.74%** | |
| **#3794 (fern)** | **n_head=2 on SwiGLU** | **64.3427** | **−3.41%** | |
| **#3789 (thorfinn)** | **vel-asinh s=0.5 on n_head=2+SwiGLU** | **63.7383** | **−0.93%** | |

## Key findings (cumulative)

### Merged stack progress
136.89 → 90.61 → 66.61 → 64.34 → **63.74** (−53.5% total from seed)

### What works on the full stack
- EMA, grad_clip, Huber δ=1.0 (Rounds 1-3)
- asinh(pressure) scale=1.0 (Round 4)
- SwiGLU gated MLP in TransolverBlocks only (Round 5)
- n_head=2 wider per-head dim (Round 6-7)
- vel-asinh scale=0.5 on Ux+Uy (Round 7-8)
- **Huber δ=0.5 (compound win #3901): val=61.61, test=60.89 — PENDING MERGE**
- **surf_weight=15 (thorfinn #3907): val~60.88 — PENDING TERMINAL**

### What does NOT work
- EMA decay < 0.99, depth n_layers=6, mlp_ratio>2, slice_num=128, sinusoidal Re-embed, p_surf_weight, feature dropout, asinh scale > 1.0, n_head=8, DropPath, SwiGLU-in-all-MLPs, Mixup, vel-asinh scale < 0.5, attention dropout, per-channel vel-asinh (test regression), LR warmup with per-epoch scheduler step (plumbing bug)

### Confirmed mechanisms (compound wins on full stack)
- Huber δ=0.5: −3.34% on full stack (#3901) — tighter quadratic regime post-asinh-p
- wd=1e-3: W&B shows −1.68% (#3902, pending terminal)
- surf_weight=15: W&B shows −4.48% (#3907, pending terminal) — BIGGEST PENDING WIN

## Strategic outlook

**Target**: val < 60. Current: 63.74 (baseline). Projected after merge wave: ~60.88 (thorfinn) → ~59.6 after further compounding.

Conservative compound path after merge wave:
- Alphonse (61.61) → Thorfinn still wins (60.88) → nezuko+tanjiro rebase → fern slice_num=16 follow-up

Once thorfinn #3907 merges (baseline ~60.88):
1. **Compound tests round 2**: Huber δ=0.5 + wd=1e-3 on 60.88 baseline (likely to compound again)
2. **slice_num=16**: fern's suggestion from crash analysis; 32→16 direction confirmed
3. **temp_init on new baseline**: tanjiro's result was on 63.74 baseline; needs rebase
4. **SGDR**: frieren's T_0=5 may still beat the new baseline
5. **Per-step LR warmup** (edward #3967): correctly specified; warm epoch dynamics hypothesis still live

## Operational notes

- **GitHub REST rate limit**: 0/5000 remaining, resets **13:20 UTC** (confirmed)
- **GraphQL API**: 3127/5000 remaining (reset ~13:07 UTC)
- **data/scoring.py NaN bug**: cruise=NaN fleet-wide (affects test_avg; use test_3split everywhere)
- **Per-run budget**: 30 min wall clock, ~15 epochs with n_head=2 at 124s/epoch
- **Merge action pending**: alphonse merge + askeladd close + fern rebase all blocked by REST until 13:20 UTC

## Queued hypotheses (next idle assignments, post-merge wave)

After thorfinn merges (new baseline ~60.88):
| Priority | Hypothesis | Expected | Rationale |
|---|---|---|---|
| 1 | slice_num=16 (fern) | −1-3% | monotone direction confirmed (64→32 wins), follow-up |
| 2 | surf_weight=20 (thorfinn follow-up) | −0.5-2% | Arm B of #3907 if not yet run |
| 3 | Huber δ=0.3 (alphonse follow-up) | −0.5-1% | closing δ axis; compound with surf_weight=15 |
| 4 | wd=1e-3 + Huber δ=0.5 (2-axis compound) | −1-3% | both mechanisms confirmed, joint effect unknown |
| 5 | Per-step LR warmup (edward #3967) | −0.5-2.5% | correctly specified after #3874 plumbing bug |
| Need researcher | Fresh hypotheses post-merge-wave | tbd | researcher-agent re-run when ≥3 students go idle |
