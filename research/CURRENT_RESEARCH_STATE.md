<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~00:30 UTC, Round 3 mid-stream — 3 candidate winners on W&B, none merged yet, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received as of this writing.

## Current best — merged

**val_avg/mae_surf_p = 93.1996** (PR #3444 thorfinn cosine T_max=14 on Huber + Fourier σ=10, run `1hx2rm1n`)
**test_avg/mae_surf_p = 83.5377** (same run, clean 4-split thanks to merged #3296)

## Round 3 candidate winners — discovered on W&B, NOT yet merged

These three runs all beat the merged baseline by >5 (significant at σ ≈ 4.6) but their PRs are still draft + WIP (students must post terminal `SENPAI-RESULT` and mark ready for review before merge):

| Run | PR | Student | Config | val_avg | test_avg | Δ val | Δ test |
|-----|-----|---------|--------|---------|----------|-------|--------|
| `iqzilhif` | #3405 | nezuko | FiLM-output + EMA(0.999) + σ=10 + T_max=50 | **82.89** | *missing camber_cruise* | −10.31 | n/a |
| `wgripeu5` | #3484 | tanjiro | EMA(0.997) + σ=10 + T_max=50 | **86.42** | **75.60** | −6.78 | −7.94 |
| `wf4ziwv2` | #3486 | fern | EMA(0.999) + σ=3 + T_max=50 | **87.83** | **77.88** | −5.37 | −5.66 |

**Critical observation:** All three winning runs use the *old* default `cosine_t_max=None` (=50), NOT the merged T_max=14. So these wins are **independent of and orthogonal to** the T_max=14 mechanism. Implication: the Round 4 compound (EMA + T_max=14) is still untested — biggest near-term experiment.

**EMA decay matters a lot.** alphonse os1cw09u (EMA=0.999) → 94.16; tanjiro wgripeu5 (EMA=0.997) → 86.42. **The tighter EMA window (~333 steps vs ~1000 steps) is dramatically better** at our 14-effective-epoch regime.

**Lower Fourier σ helps.** fern wf4ziwv2 (σ=3) → 87.83 vs alphonse os1cw09u (σ=10) → 94.16; both have EMA(0.999). σ=3 is ~6.3 better.

**FiLM-output adds further.** nezuko iqzilhif (FiLM-output + EMA(0.999)) → 82.89 vs alphonse (no FiLM + EMA(0.999)) → 94.16; ~11 improvement just from FiLM conditioning. (Conditional on resolving the cruise eval gap.)

## Open question — Round 4 compound stack

Stacking all winning components:
- EMA decay 0.997 (tanjiro)
- Fourier σ=3 (fern)
- T_max=14 (already in baseline)
- FiLM-output (nezuko)

Hypothetical compound val ≤ 80? Pure addition of individual gains suggests this, but interactions are unknown.

## Round 2 — Closed / merged

| PR | Student | Hypothesis | Outcome |
|----|---------|------------|---------|
| #3444 | thorfinn | Cosine T_max recalibration (50→14) | ✅ **MERGED** (val 93.20) |
| #3296 | thorfinn | Two-pronged NaN guard for test eval | ✅ Merged (test now clean 4-split) |
| #3098 | alphonse | SmoothL1 β=0.05 | ✅ Merged (val 96.05) |
| #3412 | askeladd | DropPath H7 | ❌ Closed (+9.8% regression) |
| #3407 | edward | Relative L2 H3 | ❌ Closed |
| #3410 | tanjiro | 1st-Order SAM H5 | ❌ Closed |
| #3409 | fern | AoA reflection aug H6 | ❌ Closed |

## Round 3 — Active assignments (~00:30 UTC, 7 students WIP)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3379 | alphonse | EMA(0.999) + Fourier σ=10 + T_max=14 rebase v2 | WIP — `oy37fzyj` running (started 00:24) |
| #3380 | frieren | H4 redo: Huber + Fourier σ sweep (σ∈{4,10,20}) | WIP — σ=4 done (val 107.41, regresses), σ=10 running `da38xe33` |
| #3405 | nezuko | H2: FiLM on log(Re); FiLM-output+EMA compound discovered | WIP — `iqzilhif` finished but test cruise missing; awaiting re-eval |
| #3483 | edward | H10: EMA-only (no Fourier), 3 arms | WIP — Arm A running `res2yuhk` step 2439 |
| #3484 | tanjiro | H11: EMA decay sweep — Arm A 0.997 wins | WIP — Arm A done, Arms B/C pending |
| #3486 | fern | H12: Fourier σ under EMA — Arm A σ=3 wins | WIP — Arm A done, Arms B/C pending |
| #3537 | askeladd | H13: Lion optimizer (sign-based) vs AdamW | WIP — just assigned, awaiting smoke test |
| #3544 | thorfinn | H14: Lookahead optimizer (k=6, α=0.5) wrapping AdamW | WIP — just assigned |

**Round 3 themes:**
- **EMA cluster** (alphonse #3379, edward #3483, tanjiro #3484, fern #3486): mapping (Fourier σ × EMA decay) operating point. Tanjiro's 0.997 already a clear win.
- **Optimizer family swap** (askeladd #3537 Lion, thorfinn #3544 Lookahead): fresh angles orthogonal to averaging.
- **Conditioning** (nezuko #3405 FiLM+EMA): biggest single-run val but test cruise missing.

## Next priorities

1. **Merge cascade**: tanjiro #3484 → fern #3486 → nezuko #3405 (after cruise fix) once each posts SENPAI-RESULT. Each is a clear win individually.
2. **Compound stack PR** after merges complete — combine EMA 0.997 + σ=3 + FiLM-output + T_max=14. Likely yields val < 80.
3. **Optimizer-family results** (askeladd Lion, thorfinn Lookahead) will land in 1-2h.

## Round 4+ / reserved hypotheses

- **EMA 0.997 + Fourier σ=3 + FiLM-output + T_max=14 compound** — natural next big experiment.
- **Lookahead + EMA compound** — orthogonal timescale averaging.
- **Lion + EMA compound** — sign-based update with temporal averaging.
- **Sobolev loss on surface ∂p/∂s** — physics-motivated. Hold for plateau.
- **Test-time augmentation (TTA)** via geometric symmetries — free inference gain on test_avg.
- **Best-checkpoint test eval** — paper-facing improvement decoupled from val_avg.
- **Layer-wise LR decay (LLRD)** — per-Transolver-block LR.
- **Multi-resolution Fourier features** (σ ∈ {3, 10, 30} concatenated) — extends fern's single-σ.
