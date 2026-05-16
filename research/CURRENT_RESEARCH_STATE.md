<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~02:25 UTC, Round 4 underway — Lion merged, all students pivoting to Lion substrate, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 77.5788** (PR #3537 askeladd Lion optimizer lr=5e-5 wd=1e-3 on Huber + Fourier σ=10 + T_max=14, run `yvkf9glr`)
**test_avg/mae_surf_p = 68.8764** (same run, clean 4-split)

Per-split val: in_dist 90.85, camber_rc 87.72, camber_cruise 58.81, re_rand 72.93
Per-split test: in_dist 81.69, camber_rc 77.94, camber_cruise 48.83, re_rand 67.04

**Δ vs prior best (T_max=14 AdamW baseline, PR #3444): −15.62 val / −14.66 test (3.4σ/3.2σ).**

## Round 3 — Completed on AdamW, not yet rebased onto Lion

These experiments confirmed important mechanisms on the AdamW baseline. They are **still valuable results** but don't directly merge since they don't beat Lion. Re-validation on Lion is in progress.

| PR | Student | Config | val_avg | test_avg | Status |
|----|---------|--------|---------|----------|--------|
| #3484 | tanjiro | EMA(0.997) — Arm A winner, confirmed 02:22Z | **86.42** | **75.60** | Terminal SENPAI-RESULT posted, rebasing onto Lion |
| #3486 | fern | σ=3 + EMA(0.999) — Arm A winner, confirmed 02:23Z | **87.83** | **77.88** | Terminal SENPAI-RESULT posted, rebasing onto Lion |

**EMA key findings (AdamW baseline):**
- EMA decay 0.997 >> 0.9995 >> 0.9999 at 14-effective-epoch budget (tighter window ~333 steps wins)
- Fourier σ=3 >> σ=5 >> σ=7 under EMA — lower frequency better
- Trend: σ=3 + EMA(0.997) + T_max=14 is the right operating point

## Round 4 — Active assignments (8 students)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3609 | askeladd | H15: Lion + LR warmup (warmup_steps ∈ {0, 500, 1000}) | NEW — just assigned 02:24Z |
| #3379 | alphonse | Lion + EMA(0.997) compound (σ=10, T_max=14) | Rebasing — CONFLICTING, instructions posted 01:47Z |
| #3484 | tanjiro | Lion + EMA(0.997) rerun after AdamW Arm C complete | Terminal posted 02:22Z; pivoting to Lion-rebase rerun |
| #3486 | fern | Lion + EMA(0.997) + σ=3 rerun | Terminal posted 02:23Z; pivoting to Lion-rebase rerun |
| #3544 | thorfinn | Lookahead-wrapping-Lion (k=5 α=0.5, k=10 α=0.5, k=5 α=0.8) | Pivot instructions posted 01:48Z |
| #3405 | nezuko | FiLM-output + EMA(0.997) + Lion + clean cruise eval | Pivot instructions posted 01:50Z; CONFLICTING needs rebase |
| #3483 | edward | Lion + EMA-only ablation vs Fourier variants (3 arms) | Instructions posted 01:53Z; was GPU-active 01:48Z |
| #3380 | frieren | Multi-resolution Fourier σ∈{3,10,30} + Lion + EMA(0.997) | Pivot instructions posted 02:00Z; was doing AdamW σ sweep |

## Mechanisms in flight (what each student is testing)

| Mechanism | Who |
|-----------|-----|
| Lion + EMA(0.997) compound | alphonse, tanjiro, fern (3 independent variance samples) |
| Lion + EMA + σ=3 | fern, edward (partial overlap) |
| FiLM + Lion + EMA | nezuko |
| Lookahead + Lion | thorfinn |
| Multi-resolution Fourier + Lion + EMA | frieren |
| Lion warmup (LR schedule) | askeladd |

## Expected Round 4 outcomes (~02:30-04:30 UTC)

- **alphonse/tanjiro/fern Lion+EMA reruns:** If EMA(0.997) compounds with Lion, expected val ~70-75. Three independent runs give excellent variance estimates.
- **nezuko FiLM+Lion+EMA:** If FiLM adds ~10 pts as on AdamW, expected val ~65-70. Biggest potential upside.
- **thorfinn Lookahead+Lion:** Unknown — Lookahead regressed on AdamW. Lion may change the picture. 3 arms ~90 min.
- **askeladd Lion+warmup:** Paper recommendation. Unknown if small batch benefits. 3 arms ~90 min.
- **edward Lion+EMA ablations:** Answers "Is Fourier still needed under Lion?" Potentially shows EMA+Lion is sufficient without Fourier.
- **frieren multi-scale Fourier:** Potentially strongest compound if {3,10,30} concatenation captures all length scales simultaneously.

## Round 5+ / reserved hypotheses

- **Lion β1, β2 sweep** — β1=0.95 is an alternative recommendation in the paper.
- **Lion + gradient accumulation** — tests effective-batch sensitivity at batch=4.
- **Sobolev loss on surface ∂p/∂s** — physics-motivated, completely orthogonal.
- **Test-time augmentation (TTA)** via reflection symmetry — free inference-time gain.
- **Best-checkpoint test eval** — paper-facing improvement decoupled from val_avg.
- **Layer-wise LR decay (LLRD)** — per-block Transolver LR.
- **EMA + Lion multi-decay sweep** — confirm 0.997 is still optimal under Lion (tanjiro's AdamW finding may not transfer).

## Round 2 history

| PR | Hypothesis | Outcome |
|----|------------|---------|
| #3537 | Lion optimizer H13 | ✅ MERGED 01:43Z (val 77.58 / test 68.88) |
| #3444 | Cosine T_max=14 | ✅ Merged (val 93.20 / test 83.54) |
| #3296 | Two-pronged NaN guard | ✅ Merged |
| #3098 | SmoothL1 β=0.05 | ✅ Merged (val 96.05) |
| #3412 | DropPath | ❌ Closed |
| #3407 | Relative L2 | ❌ Closed |
| #3410 | 1st-Order SAM | ❌ Closed |
| #3409 | AoA reflection aug | ❌ Closed |
