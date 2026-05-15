<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-15 (~22:30 UTC, Round 2 closing / Round 3 launching, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received as of this writing.

## Current best

**val_avg/mae_surf_p = 96.05** (PR #3098, run `md6so639`, SmoothL1 β=0.05) ← formal merged best
**val_avg in-review = 92.41** (PR #3379 alphonse Arm C — Huber + Fourier + clip + EMA(0.999), run `hat7m2bl`) ← pending rebase + clean test_avg confirm
**test_avg/mae_surf_p = 90.00** (PR #3296 NaN guard, run `xvn4gllg`) ← first valid test_avg

Per-split val for #3379 Arm C: in_dist 119.72, camber_rc 104.00, camber_cruise 62.39, re_rand 83.51.
Per-split test for #3296: in_dist 109.30, camber_rc 103.19, camber_cruise 60.61 (199/200), re_rand 86.90.

**Primary metric:** `val_avg/mae_surf_p` (lower better).
**Binding constraint:** SENPAI_TIMEOUT_MINUTES=30.0, SENPAI_MAX_EPOCHS=50, 1 GPU per student.
**Note:** 30-min wall clock binds at ~epoch 14. All runs severely under-trained.

## Central finding: EMA(0.999) is the dominant Round 2 mechanism

From alphonse's #3379 per-arm decomposition:
- Huber + Fourier σ=10 alone: **100.76** (+4.7 regression vs 96.05)
- + grad_clip 1.0: **100.32** (no additional effect)
- + EMA(0.999): **92.41** (−3.78%) ← single mechanism, compensates Fourier regression

Fourier PE n=16 σ=10 is **net-negative on Huber without EMA.** With EMA it's net-neutral at best. Round 3 probes: (a) does EMA-without-Fourier beat 92.41? (b) does lower σ fix Fourier's regression? (c) is EMA decay 0.999 optimal at 14-epoch budget?

## Round 1 — Validated mechanisms

1. **Huber loss (SmoothL1 β=0.05)** — 26% improvement. Dominant lever. Merged (#3098).
2. **Grad-clip + EMA standalone** — val 102.67 (nezuko #3114). Subsumed by compound stack.
3. **Capacity scaling kills convergence** — askeladd #3100, edward #3103 both fail hard.
4. **Per-channel weighting counterproductive** — hurts encoder → hurts pressure (tanjiro #3118).
5. **Warmup incompatible** — cosine T_max=50 stays near peak LR at epoch 14 (fern #3105).
6. **NaN guard merged** — pred `nan_to_num` + y-side sample mask (thorfinn #3296). test_avg clean.
7. **Run-to-run σ ≈ 4.6** on val_avg. Need >5 delta for significance.

## Round 2 — Closed results

| PR | Student | Hypothesis | Best val_avg | Outcome |
|----|---------|------------|--------------|---------|
| #3379 | alphonse | Compound stack (EMA WINNER) | **92.41** | **NEEDS REBASE → MERGE** |
| #3405 | nezuko | FiLM on log(Re) H2 | 88.79 (single run) | WIP — Arm C running |
| #3412 | askeladd | DropPath H7 | TBD | WIP |
| #3444 | thorfinn | Cosine T_max recalibration H9 | TBD | WIP |
| #3380 | frieren | Fourier sigma sweep H4 | invalid (MSE config) | SEND BACK (Huber fix) |
| #3407 | edward | Relative L2 H3 | 117.69 | CLOSED ❌ (+22%) |
| #3410 | tanjiro | 1st-Order SAM H5 | 142.86 | CLOSED ❌ (+49%) |
| #3409 | fern | AoA reflection aug H6 | 119.28 | CLOSED ❌ (+13%) |

**Round 2 dead-end findings:** SAM is wall-clock-incompatible (halves step count); Rel L2 fights Huber's soft-cap; AoA aug is redundant with existing data distribution; frieren's sigma sweep was on MSE, not Huber — results invalid.

## Round 3 — Active assignments (~22:30 UTC)

| PR | Student | Hypothesis | Target |
|----|---------|------------|--------|
| #3483 | edward | H10: EMA-only, no Fourier (3 arms: EMA+clip, EMA-only, EMA decay 0.9995) | beats val 92.41 |
| #3484 | tanjiro | H11: EMA decay sweep (0.997 / 0.9995 / 0.9999) | beats val 92.41 |
| #3486 | fern | H12: Fourier sigma under EMA (σ∈{3, 5, 7}, full stack) | beats val 92.41 |

**Coordinated theme:** all 3 target the Fourier-vs-EMA interaction. H10 removes Fourier; H12 lowers σ; H11 adjusts EMA window. Results will give a clear 2D map of the optimal (Fourier σ, EMA decay) operating point.

## Still active WIP

- **#3379 alphonse**: Rebase in progress. Arm C only, on merged NaN guard. Will produce clean 4-split test_avg. On merge: BASELINE.md updates to val 92.41.
- **#3444 thorfinn**: Cosine T_max recalibration 50→14/18. Orthogonal to EMA — addresses the LR schedule mismatch vs wall-clock cap. Synergy: EMA + T_max compound is the next natural experiment.
- **#3412 askeladd**: DropPath stochastic depth. Orthogonal regularization. OOD splits target.
- **#3405 nezuko**: FiLM-output-only Arm B (88.79 best run, mean 99.33 across 3 runs — within noise). Arm C running (ETA ~22:55). Decision deferred until Arm C result.
- **#3380 frieren**: Sent back to re-run sigma sweep ∈ {4, 10, 20} with `--loss_type smooth_l1 --loss_beta 0.05`. Results will complement fern's σ∈{3,5,7} under EMA.

## Round 3+ / reserved hypotheses

- **EMA + FiLM combination**: assign after nezuko closes (pending Arm C).
- **EMA + cosine T_max compound**: integrate thorfinn's T_max fix with alphonse's EMA stack.
- **H8: Sobolev loss on surface ∂p/∂s** — physics-motivated. Hold for plateau.
- **Test-time augmentation (TTA)** via geometric symmetries — free inference gain.
- **Best-checkpoint test eval** — paper-facing improvement decoupled from val_avg.
