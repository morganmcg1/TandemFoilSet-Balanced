# SENPAI Research State

- **Last updated:** 2026-05-16 02:50 UTC
- **Branch:** `icml-appendix-willow-pai2i-48h-r2`
- **Most recent direction from human researcher team:** None (checked 02:45 UTC — no open issues).

## Current best baseline

| Metric | Value | Source |
|---|---|---|
| `val_avg/mae_surf_p` | **90.6131** | PR #3474 alphonse EMA decay=0.99+clip5+Huber (merged 00:25 UTC) |
| `test_avg/mae_surf_p` (3 valid splits; cruise=NaN) | 88.8252 | run `fzrq04xr` |

Per-split validation:

| Split | mae_surf_p | Δ vs prev baseline (#3366, 94.42) |
|---|---|---|
| val_single_in_dist | 106.135 | −5.1% |
| val_geom_camber_rc | 99.466 | **−9.7%** |
| val_geom_camber_cruise | 70.358 | +1.9% |
| val_re_rand | 86.494 | −0.2% ← weakest split, targeted by Round-4 |

## 🔥 Imminent merge candidate

**askeladd #3475 asinh-pressure verify** (rebased onto decay=0.99 stack):
- Verify run `2028x8co` finished at **val_avg/mae_surf_p = 85.82** → **−5.3%** vs baseline 90.6131
- Replicate `j5214ii4` finishing ~02:55 UTC (started 02:22 UTC)
- PR is MERGEABLE/CLEAN (rebase complete, branch verified clean)
- Awaiting terminal SENPAI-RESULT comment from askeladd to invoke senpai:merge-winner

## Active PRs (zero idle students)

### Tier-1 — EMA decay bracketing (WIP)

| PR | Student | Hypothesis | Best so far | Status |
|----|---------|-----------|-----------------|-------|
| #3543 | alphonse | ema-decay-push (0.98, 0.97, 0.95) | 0.98=90.84 (TIED); 0.97 running (step 805); 0.95 not started | Extra 0.98 replicate (klk1qnks) also active |

### Tier-2 — Winner awaiting merge

| PR | Student | Hypothesis | W&B best | Vs baseline | Status |
|----|---------|-----------|----------|------------|--------|
| **#3475** | **askeladd** | **asinh-pressure verify on decay=0.99** | **85.82** (`2028x8co`) | **−5.3% WIN** | **Replicate running; awaiting SENPAI-RESULT** |

### Tier-2 — Round-4 in-flight (WIP)

| PR | Student | Hypothesis | Best so far | Status |
|----|---------|-----------|----------|--------|
| #3575 | edward | p-surf-weight (3.0, 5.0) | 3.0=94.65 (REGRESS +4.5%) | 5.0 running |
| #3576 | nezuko | wd-sweep (1e-3, 5e-3) | 1e-3=90.75 (TIED) | 5e-3 running |
| #3577 | tanjiro | slice-num-128 | mid-train (~5 epochs) | running |
| #3578 | frieren | re-sinusoidal-embed (d=8) | early train (~2.5 epochs) | running |
| #3610 | thorfinn | mlp-ratio-sweep (4) | just started | running |
| **#3649** | **fern** | **n_head sweep (4→8, conditional 16)** | — | **Newly assigned** |
| #3543 | alphonse | ema-decay-push (0.97, 0.95) | 0.97 running | ongoing |

## Confirmed winners (merged)

| PR | Hypothesis | val_avg | Δ | Notes |
|---|---|---|---|---|
| #3186 (fern) | EMA weights decay=0.999 | 121.685 | −11.10% | All 4 splits improve, 3 runs |
| #3366 (fern) | EMA + grad_clip=5 + Huber δ=1.0 | 94.4199 | −22.4% | All 4 splits ≥−20%; 2 runs |
| **#3474 (alphonse)** | **EMA decay=0.99 (faster shadow tracking)** | **90.6131** | **−4.0%** | **Monotone sweep; 3 arms all beat prior baseline** |

## Closed PRs (this round)

| PR | Student | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| #3477 | thorfinn | physics-continuity (3 arms) | all REGRESS 98-106 | CLOSED |
| **#3571** | **fern** | **depth-sweep n_layers=6** | **val=93.83 (+3.55%)** | **CLOSED — wall-clock bound** |

## Round-4 status snapshot (as of 02:50 UTC)

| Student | PR | Hypothesis | Best val | Verdict |
|---|---|---|---|---|
| askeladd | #3475 | asinh-pressure verify | **85.82** | ✅ **WIN — merge pending** |
| alphonse | #3543 | ema-decay-push | 0.98=90.84 | ≈ TIED; 0.97/0.95 pending |
| nezuko | #3576 | wd-sweep | 1e-3=90.75 | ≈ TIED; 5e-3 running |
| tanjiro | #3577 | slice-num-128 | running | ⏳ |
| frieren | #3578 | re-sinusoidal-embed | running | ⏳ |
| thorfinn | #3610 | mlp-ratio=4 | running | ⏳ |
| edward | #3575 | p-surf-weight=3/5 | 3.0=94.65 | ❌ Arm 1 REGRESS; arm 2 running |
| fern | **#3649** | **n_head sweep** | — | **NEW** |
| ~~fern~~ | ~~#3571~~ | ~~depth-sweep~~ | ~~93.83 (+3.55%)~~ | ~~CLOSED~~ |

## Key findings (cumulative)

### Round 1: EMA wins
EMA trajectory averaging (decay=0.999): −11.1%, all 4 splits improve.

### Round 2: grad_clip + Huber compounds with EMA
EMA+clip5+Huber δ=1.0: −22.4%. Three mechanisms orthogonal.

### Round 3: Faster EMA decay wins; hyperparameter sweeps falsified
- EMA decay 0.99 beats slower decays within 14-epoch budget
- LR, CosineAnnealingLR truncation, Huber-delta, SWA, geometry-mirror all falsified
- asinh-pressure: WINNER (val~85.82 verify on new baseline)
- physics-continuity: CLOSED (all 3 arms regress)

### Round 4 status
- **Target-side transform wins**: asinh(pressure) at scale=1.0 → −5.3% (largest single-PR gain since EMA stack)
- **Architecture-via-depth**: FALSIFIED within wall-clock budget (depth=6 regressed)
- **Architecture-via-attention**: n_head sweep just assigned (#3649, fern)
- **Regularization (wd)**: TIED so far (wd=1e-3); 5e-3 running
- **OOD Re bottleneck** (val_re_rand): askeladd improved from 86.49 → 82.20 (−4.9%); frieren's re-embed specifically targets this

## Strategic outlook

**After askeladd merges** (expected next wakeup): new baseline val ~85.82. Round-5 compounds to explore:
1. **asinh + larger model**: if mlp_ratio/slice_num/n_head any of them win in Round 4
2. **asinh + stronger wd**: if nezuko 5e-3 arm wins
3. **asinh + ema-decay-push**: if alphonse 0.97/0.95 beats 0.99
4. **asinh scale sweep** (askeladd's own suggestion): scale=1.5, 2.0

## Operational notes

- **data/scoring.py NaN bug**: `test_geom_camber_cruise_gt/000020.pt` has inf GT → cruise=NaN fleet-wide. Known issue.
- Per-run budget: 30 min wall clock, 50 epoch cap. Wall clock binds (~14 epochs).
- **Zero idle students**: 8 WIP PRs (3543, 3475, 3575, 3576, 3577, 3578, 3610, 3649).
- REST API stable since 01:19 UTC recovery.
