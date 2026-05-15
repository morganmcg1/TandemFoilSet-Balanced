<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-15 (~23:50 UTC, Round 2 closing / Round 3 active, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received as of this writing.

## Current best (merged)

**val_avg/mae_surf_p = 93.1996** (PR #3444 thorfinn cosine T_max=14 on Huber + Fourier σ=10, run `1hx2rm1n`)
**test_avg/mae_surf_p = 83.5377** (same run, clean 4-split thanks to merged #3296)

Per-split test for #3444: in_dist 105.93, camber_rc 90.03, camber_cruise 57.65 (199/200), re_rand 80.55. **Every test split improves vs prior best**, with biggest gain on `geom_camber_rc` (−12.8%).

**Primary metric:** `val_avg/mae_surf_p` (lower better).
**Binding constraint:** SENPAI_TIMEOUT_MINUTES=30.0, SENPAI_MAX_EPOCHS=50, 1 GPU per student. 30-min wall clock binds at ~epoch 14.
**Two-pronged NaN guard:** pred `nan_to_num` + y-side sample mask, merged in #3296.

## Central findings (Round 1 + Round 2)

1. **Huber β=0.05** is dominant (−26% vs MSE). #3098 merged.
2. **NaN guard** stabilized test_avg measurement. #3296 merged.
3. **Cosine T_max recalibration 50→14** is the second major mechanism (PR #3444): the previous schedule never decayed below 82% of peak LR within the 14-epoch wall-clock budget. Setting T_max=14 gives proper end-of-run fine-tuning.
4. **EMA(0.999) is also a real mechanism** — alphonse's standalone EMA on Huber+Fourier gave val 92.41 / test 83.45 (rebase-rerun `os1cw09u`). Pending merge — see #3379.
5. **Run-to-run σ ≈ 4.6** on val_avg. Need >5 delta for significance.

## Open question

The two big Round 2 mechanisms — **EMA(0.999)** and **cosine T_max=14** — are nearly identical in magnitude (val 92.41 vs 93.20) and via different mechanisms. **Round 4 compound stack** (Huber + Fourier + EMA + T_max=14) is the natural next experiment. First we need alphonse's #3379 to land on the new T_max=14 baseline.

## Round 2 — Closed results

| PR | Student | Hypothesis | Best val_avg | Outcome |
|----|---------|------------|--------------|---------|
| #3444 | thorfinn | Cosine T_max recalibration (50→14) | **93.20** | ✅ **MERGED** |
| #3379 | alphonse | Compound stack (EMA WINNER, val 92.41 on old base; 94.16 on rebase) | 94.16 (rerun) | needs rebase + rerun on T_max=14 |
| #3405 | nezuko | FiLM on log(Re) H2 (Arm B mean 99.33, Arm C 97.09) | 88.79 (single-run best) | WIP — trying compound FiLM+EMA |
| #3412 | askeladd | DropPath H7 (Arm B 102.34, Arm C 115.02) | 102.34 | ❌ **CLOSED** (+9.8% regression) |
| #3380 | frieren | Fourier sigma sweep H4 (re-run on Huber config) | TBD | WIP — sigma-4 nearly done |
| #3407 | edward | Relative L2 H3 (val 117.69) | 117.69 | ❌ CLOSED |
| #3410 | tanjiro | 1st-Order SAM H5 (val 142.86) | 142.86 | ❌ CLOSED |
| #3409 | fern | AoA reflection aug H6 (val 119.28) | 119.28 | ❌ CLOSED |

## Round 3 — Active assignments (~23:50 UTC)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3379 | alphonse | EMA(0.999) + Fourier + grad-clip (rebase v3 against T_max=14 advisor HEAD) | WIP — rebase pending |
| #3483 | edward | H10: EMA-only (no Fourier), 3 arms | WIP — Arm A running step 2439 |
| #3484 | tanjiro | H11: EMA decay sweep (0.997/0.9995/0.9999) | WIP — Arm A 0.997 running step 3557 |
| #3486 | fern | H12: Fourier σ under EMA (σ∈{3,5,7}) | WIP — Arm A σ=3 running step 2851 |
| #3380 | frieren | H4 redo: Huber + Fourier σ sweep (σ∈{4,10,20}) | WIP — sigma-4 step 4461/5264 |
| #3405 | nezuko | H2: FiLM on log(Re); now also trying FiLM+EMA compound | WIP — `iqzilhif` FiLM+EMA running step 886 |
| #3537 | askeladd | H13: Lion optimizer (sign-based) vs AdamW, 3 LR arms | **NEW** — just assigned |

**Coordinated themes for Round 3:**
- **EMA cluster** (alphonse #3379, edward #3483, tanjiro #3484, fern #3486): mapping (Fourier σ × EMA decay) operating point on top of Huber.
- **Optimizer family swap** (askeladd #3537): Lion vs AdamW — fresh angle orthogonal to averaging schemes.
- **Conditioning** (nezuko #3405): FiLM (log Re) — exploring compound with EMA.

## Round 4+ / reserved hypotheses

- **EMA + Cosine T_max=14 compound** (Round 4 natural next): pending alphonse #3379 merge.
- **Lion + EMA compound** (Round 5 candidate): pending askeladd Lion results.
- **Sobolev loss on surface ∂p/∂s** — physics-motivated. Hold for plateau.
- **Test-time augmentation (TTA)** via geometric symmetries — free inference gain on test_avg.
- **Best-checkpoint test eval** — paper-facing improvement decoupled from val_avg.
- **Layer-wise LR decay (LLRD)** — per-Transolver-block LR.
- **MixUp on mesh batches** — strong OOD regularizer.
