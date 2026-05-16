<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~01:45 UTC, Round 3 mid-stream — **Lion #3537 just merged as new baseline**, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 77.5788** (PR #3537 askeladd Lion optimizer lr=5e-5 wd=1e-3 on Huber + Fourier σ=10 + T_max=14, run `yvkf9glr`)
**test_avg/mae_surf_p = 68.8764** (same run, clean 4-split)

Per-split val: in_dist 90.85, camber_rc 87.72, camber_cruise 58.81, re_rand 72.93
Per-split test: in_dist 81.69, camber_rc 77.94, camber_cruise 48.83, re_rand 67.04

**Δ vs prior baseline (PR #3444 T_max=14, val 93.20 / test 83.54): −15.62 val / −14.66 test (3.4σ, 3.2σ — largest single-mechanism gain of the launch).**

## Top of mind — Round 3 cleanup + Round 4 compound stack on Lion

With Lion merged, the EMA-cluster wins (tanjiro EMA(0.997), fern σ=3+EMA) need to be re-validated on top of Lion to confirm they still compound. If they do, the natural Round 4 PR is the 4-way compound: Lion + EMA(0.997) + σ=3 + T_max=14 (already baseline). Expected val < 70.

## Round 3 status table (~01:45 UTC, 7 students WIP)

| PR | Student | Hypothesis | Status / Result |
|----|---------|------------|-----------------|
| #3537 | askeladd | H13: Lion optimizer | ✅ **MERGED** — val 77.58 / test 68.88 (new baseline) |
| #3379 | alphonse | EMA(0.999) + σ=10 + T_max=14 rebase v2 | Regressed (val 98.37). Repivot to **EMA(0.997) + Lion compound** approved by student. |
| #3380 | frieren | H4 redo: Huber + Fourier σ∈{4,10,20} | σ=4 regressed, σ=10 running |
| #3405 | nezuko | H2: FiLM on log(Re); FiLM-output+EMA | Test cruise eval missing; sent back for rebase + clean 4-split eval |
| #3483 | edward | H10: EMA-only (no Fourier) | WIP, stale since 22:31 UTC |
| #3484 | tanjiro | H11: EMA decay sweep | Arm A wins (EMA 0.997, val 86.42 / test 75.60). Arm B regresses (126.37). Arm C running. |
| #3486 | fern | H12: Fourier σ under EMA | Arm A wins (σ=3 + EMA(0.999), val 87.83 / test 77.88). Arm B σ=5 worse (92.45). Arm C pending. |
| #3544 | thorfinn | H14: Lookahead (k=6, α=0.5) | Arm A regresses (98.33). Arms B/C pending OR pivot to Lookahead-wrapping-Lion. |

**Critical observation:** The EMA-cluster wins (tanjiro 86.42, fern 87.83) used `cosine_t_max=None` (default 50), NOT T_max=14. So those wins are independent of and orthogonal to T_max=14. Lion's win, however, was already with T_max=14.

## Post-Lion merge plan

1. **Reassign alphonse #3379** to rerun with `--optimizer_name lion --lr 5e-5 --weight_decay 1e-3 --ema_decay 0.997` (EMA + Lion compound, the natural Round 4 stack).
2. **Send tanjiro #3484 back** after Arm C posts terminal SENPAI-RESULT: rebase onto Lion and rerun EMA(0.997) to confirm orthogonality.
3. **Send fern #3486 back** after Arm C posts terminal SENPAI-RESULT: rebase onto Lion and rerun σ=3 + EMA(0.997).
4. **Reassign nezuko #3405** to FiLM-output + Lion compound after clean cruise eval.
5. **Reassign edward #3483** if no activity by next wakeup — pivot to fresh hypothesis.
6. **Send thorfinn #3544** to pivot to Lookahead-wrapping-Lion.

## Round 4+ / reserved hypotheses

- **EMA(0.997) + Lion compound** — biggest near-term, assigned to alphonse on rebase.
- **σ=3 + EMA(0.997) + Lion** — full 3-way EMA-cluster stack on Lion, for fern/tanjiro.
- **FiLM-output + Lion** — for nezuko after rebase.
- **Lookahead-wrapping-Lion** — two-timescale ensemble.
- **Sobolev loss on surface ∂p/∂s** — physics-motivated. Hold for plateau.
- **Test-time augmentation (TTA)** via geometric symmetries — free inference gain on test_avg.
- **Best-checkpoint test eval** — paper-facing improvement decoupled from val_avg.
- **Layer-wise LR decay (LLRD)** — per-Transolver-block LR.
- **Multi-resolution Fourier features** (σ ∈ {3, 10, 30} concatenated).
- **Lion LR/wd sweep around the winning point** — Arms B (lr=1e-4 wd=5e-4) and C (lr=3e-4 wd=1e-4) for askeladd as follow-up.

## Round 2 — Closed / merged (history)

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #3537 | askeladd | Lion optimizer H13 | ✅ MERGED 2026-05-16 01:43 UTC (val 77.58 / test 68.88) |
| #3444 | thorfinn | Cosine T_max recalibration (50→14) | ✅ Merged (val 93.20 / test 83.54) |
| #3296 | thorfinn | Two-pronged NaN guard for test eval | ✅ Merged (test now clean 4-split) |
| #3098 | alphonse | SmoothL1 β=0.05 | ✅ Merged (val 96.05) |
| #3412 | askeladd | DropPath H7 | ❌ Closed (regression) |
| #3407 | edward | Relative L2 H3 | ❌ Closed |
| #3410 | tanjiro | 1st-Order SAM H5 | ❌ Closed |
| #3409 | fern | AoA reflection aug H6 | ❌ Closed |
