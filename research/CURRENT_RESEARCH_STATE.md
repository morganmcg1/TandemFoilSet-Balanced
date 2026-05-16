<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~03:45 UTC, Round 5 underway — FiLM+Lion+EMA #3405 merged as new baseline, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 71.6544** (PR #3405 nezuko FiLM-output on log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Fourier σ=10 + Huber + T_max=14, run `ksltdq7a`)
**test_avg/mae_surf_p = 62.1091** (same run, clean 4-split)

Per-split val: in_dist 81.17, camber_rc 84.45, camber_cruise 51.99, re_rand 69.01
Per-split test: in_dist 71.30, camber_rc 73.87, camber_cruise 42.84, re_rand 60.43

**Δ vs prior best (PR #3537 Lion, val 77.58 / test 68.88): −5.93 val / −6.77 test (−7.9% / −9.8%).**

## Merged sequence (improvement cascade)

| PR | Description | val → val | test → test | Δ |
|----|-------------|-----------|-------------|---|
| #3098 | Huber loss | 135.23 → 96.05 | — | −29.1% |
| #3296 | NaN guard | — → 90.00 | first clean test | — |
| #3444 | cosine T_max=14 | 96.05 → 93.20 | 90.00 → 83.54 | −3.0% / −7.2% |
| #3537 | Lion optimizer | 93.20 → 77.58 | 83.54 → 68.88 | −16.8% / −17.5% |
| **#3405** | **FiLM+Lion+EMA** | **77.58 → 71.65** | **68.88 → 62.11** | **−7.9% / −9.8%** |

**Total improvement from starting baseline (135.23 / ~130 est.):**
- val: 135 → 71.65 (−47%)
- test: ~130 → 62.11 (−52%)

## Round 5 — Active assignments (6 of 8 students WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3609 | askeladd | Lion + LR warmup ablation (warmup ∈ {0, 500, 1000}) | WIP — Arm A control done (79.13), Arms B/C running |
| #3544 | thorfinn | Lookahead-on-Lion (dead end result: val 89.39) | Pending close; thorfinn idle after close → NEW ASSIGNMENT |
| #3486 | fern | Lion+EMA+σ=3 rerun done (val 73.81) | PR still open, informative ablation. Reassign fern. |
| #3483 | edward | Lion+EMA ablation (no Fourier Arm A done, σ=3 Arm B running) | WIP |
| #3380 | frieren | Multi-σ Fourier (config bug: n_fourier=0, val 76.95) | Pending re-run or close; reassign |
| — | alphonse | #3379 closed — fresh assignment needed | IDLE |
| — | nezuko | #3405 merged — fresh assignment needed | IDLE |
| — | tanjiro | #3484 closed — fresh assignment needed | IDLE |

## Key Round 4 findings

1. **FiLM on log(Re) is the biggest new mechanism on top of Lion+EMA.** Adds ~5.9 val / 6.8 test over Lion+EMA-only.
2. **EMA(0.997) clearly compounds with Lion** (alphonse + tanjiro avg ~77.7; with EMA vs 77.58 without). Effect is noisy but real.
3. **Fourier σ is marginal under Lion+EMA.** edward no-Fourier (73.10) ≈ fern σ=3 (73.81) ≈ alphonse σ=10 (76.15). FiLM dominates. **Round 5 should test FiLM+Lion+EMA without Fourier** to see if Fourier can be dropped.
4. **Lookahead does NOT compound with Lion.** (thorfinn 89.39, +11.8 regression). Dead end.
5. **Multi-scale Fourier had a config bug (n_fourier=0).** Need a clean run. Round 5 can test FiLM + multi-σ.
6. **Lion LR warmup Arm A control (val 79.13) ≈ Lion baseline.** Arms B/C with actual warmup still running.

## Round 5 hypotheses in flight / planned

**Mechanism targets on new FiLM+Lion+EMA baseline (val 71.65 / test 62.11):**

| Hypothesis | Expected yield | Assignment |
|------------|----------------|------------|
| FiLM + Lion + EMA — drop Fourier entirely | Does FiLM replace Fourier? | Planned for nezuko |
| FiLM + Lion + EMA + σ=3 (fern's σ but under FiLM) | Is σ=3 better than σ=10 under FiLM? | Planned for alphonse |
| FiLM conditioned on both log(Re) AND camber arc-length features | Extra context for camber OOD | Planned for tanjiro |
| Lion LR warmup ablation Arms B/C | Does warmup help Lion? | askeladd (#3609) |
| Lion + EMA ablation arms (edward Arms B/C) | No-Fourier vs σ=3 under FiLM | edward (#3483) |
| Multi-σ Fourier with FiLM (fix config bug) | Multi-scale under FiLM | frieren |

## Round 6+ / reserved hypotheses

- **EMA decay sweep under FiLM+Lion** — confirm 0.997 is still optimal with FiLM
- **Lion β sweep (β1 ∈ {0.9, 0.95})** — paper ablation
- **Sobolev loss on surface ∂p/∂s** — physics-motivated, orthogonal
- **TTA via geometric reflection symmetry** — free inference gain
- **Layer-wise LR decay (LLRD)** — per-block LR

## History

| PR | Hypothesis | Outcome |
|----|------------|---------|
| #3405 | FiLM+Lion+EMA H4 | ✅ MERGED 03:40Z (val 71.65 / test 62.11) |
| #3537 | Lion optimizer H13 | ✅ MERGED 01:43Z (val 77.58 / test 68.88) |
| #3444 | cosine T_max=14 | ✅ Merged (val 93.20) |
| #3296 | NaN guard | ✅ Merged |
| #3098 | Huber loss | ✅ Merged (val 96.05) |
| #3379 | alphonse EMA compound | Closed (superseded by nezuko) |
| #3484 | tanjiro EMA decay sweep | Closed (superseded) |
| #3544 | thorfinn Lookahead | Closing (dead end) |
| #3412/#3407/#3410/#3409 | Various R2 dead ends | ❌ Closed |
