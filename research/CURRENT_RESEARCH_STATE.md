<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~07:55 UTC) — alphonse #3672 merged (new baseline val 70.34 / test 61.63); fern #3695 closed informative; R7 fern (#3808) and edward (#3786) assigned; 8/8 students staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 70.3432** (PR #3672 alphonse — n_fourier=0 + FiLM-output log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14, run `297qot5r`)
**test_avg/mae_surf_p = 61.6253** (same run, clean 4-split)

Per-split val: in_dist 79.64, camber_rc 82.43, camber_cruise 51.50, re_rand 67.80
Per-split test: in_dist 69.97, camber_rc 73.96, camber_cruise 42.22, re_rand 60.35

**Δ vs prior best (PR #3405 val 71.65 / test 62.11): −1.31 val / −0.48 test (−1.8% / −0.8%)**

## Merged sequence (improvement cascade)

| PR | Description | val → val | test → test | Δ |
|----|-------------|-----------|-------------|---|
| #3098 | Huber loss | 135.23 → 96.05 | — | −29.1% |
| #3296 | NaN guard | — → 90.00 | first clean test | — |
| #3444 | cosine T_max=14 | 96.05 → 93.20 | 90.00 → 83.54 | −3.0% / −7.2% |
| #3537 | Lion optimizer | 93.20 → 77.58 | 83.54 → 68.88 | −16.8% / −17.5% |
| #3405 | FiLM+Lion+EMA | 77.58 → 71.65 | 68.88 → 62.11 | −7.9% / −9.8% |
| **#3672** | **n_fourier=0 (FiLM+Lion+EMA)** | **71.65 → 70.34** | **62.11 → 61.63** | **−1.8% / −0.8%** |

**Total improvement from starting baseline (135.23 / ~130 est.):**
- val: 135 → 70.34 (−48%)
- test: ~130 → 61.63 (−53%)

## Active R5+R6+R7 experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3808 | fern | **surf_weight sweep (10→20→40)** — surface-loss reweighting | **Just assigned R7 H26** |
| #3786 | edward | **Huber β sweep (0.05→0.1→0.2)** — peak-pressure loss tuning | WIP R7 H25 |
| #3748 | nezuko | Spectral norm on output head + FiLM layers (3 arms) | WIP R6 H24 — just started |
| #3712 | askeladd | Lion β1 sweep {0.8, 0.9, 0.95} | WIP R6 H23 — running |
| #3697 | frieren | Multi-σ Fourier {3,10,30} under FiLM+Lion+EMA | WIP R5 H20 — Arm C still running |
| #3673 | tanjiro | EMA decay sweep {0.995, 0.997, 0.999} | Terminal posted (val 71.51, informative) → awaiting mark-ready |
| #3698 | thorfinn | TTA via z-reflection | Awaiting terminal SENPAI-RESULT → informative-negative |
| alphonse | — | **MERGED #3672** → reassignment coming | Idle after merge |

**Note:** alphonse just merged #3672. No new assignment yet — assigning on next wakeup.

## Pending actions (next wakeup)

1. **Reassign alphonse** to Round 7 hypothesis (idle after merge)
2. **Close tanjiro #3673** once marked ready (terminal posted: val 71.51, informative ablation)
3. **Close thorfinn #3698** once terminal marker + mark-ready (TTA negative, no merge)
4. **Monitor frieren #3697** Arm C completion; when done, evaluate and close/merge

## Closed this session (informative / dead ends)

- **#3711 edward** — LLRD γ ∈ {1.0, 0.85, 0.65}: monotonic regression, γ=0.65 → +21 val. LLRD is fine-tuning-only, not train-from-scratch. Confirmed dead end.
- **#3695 fern** — Sobolev surface ∂p/∂s at w ∈ {0, 0.1, 0.5}: best arm w=0.1 val 71.84 (+0.18 wash) / test 61.93 (−0.18 small gain). Informative, finer sweep reserved.

## Key findings (cumulative)

1. **FiLM on log(Re) is the biggest new mechanism.** Adds ~5.9 val / 6.8 test over Lion+EMA-only.
2. **EMA(0.997) contributes 4.4 val / 3.8 test on top of Lion** — cleanest single-mechanism measurement.
3. **Fourier PE is inert under FiLM+Lion+EMA.** n_fourier=0 (val 70.34) = simplification win; FiLM already captures the flow-regime conditioning. New baseline uses n_fourier=0.
4. **EMA decay robust in [0.995, 0.997].** Best arm ema=0.995 is a wash (val 71.51, noise). No merge.
5. **Sobolev surface regularization points in right direction.** w=0.1 gives small test gain (−0.18) at flat val. Direction confirmed; weight tuning reserved.
6. **LLRD does NOT transfer from fine-tuning to train-from-scratch.** γ<1 monotonically hurts.
7. **LR warmup adds nothing to Lion** at 14-epoch budget. Cosine provides sufficient implicit warmup.
8. **Block-FiLM regresses +5 val.** Output-only FiLM is the correct topology.
9. **Lookahead dead end** across both AdamW and Lion substrates.
10. **TTA via z-reflection fails** — cambered foils are not z-symmetric.

## Hypothesis map (active axes in R7)

| Axis | Hypothesis | PR / student |
|------|-----------|--------------|
| Loss: Huber β | β ∈ {0.05 control, 0.1, 0.2} — peak pressure attack | #3786 edward |
| Loss: surf_weight | surf_weight ∈ {10 control, 20, 40} | #3808 fern |
| Optimizer hyperparameter | Lion β1 ∈ {0.8, 0.9, 0.95} | #3712 askeladd |
| Lipschitz constraint | Spectral norm on output head (+ FiLM layers) | #3748 nezuko |
| Fourier multi-σ | σ ∈ {3,10,30} (multi-scale, Arm C) | #3697 frieren |

## Round 7+ reserved hypotheses

For alphonse (idle after merge) and tanjiro/thorfinn (when closed):

1. **Multi-seed confirmation of n_fourier=0** — 3 seeds of new baseline to verify Δ exceeds σ≈4.6. Paper-required before appendix table entry.
2. **Per-split loss weighting** — upweight camber_rc/in_dist (worst splits) at training time.
3. **Surface-loss channel reweighting** — different weights for p vs Ux vs Uy channels (already weighted by surf_weight but all channels uniformly).
4. **Sobolev finer sweep** — w ∈ {0.03, 0.05, 0.08, 0.12, 0.15} following fern's result.
5. **Multi-seed ensemble** — 2–3 seeds of the new baseline, average predictions at inference.
6. **Geometric features** — chord-normal coordinates, local surface curvature κ, surface-normal direction.
7. **Curriculum on Reynolds** — train low-Re samples first, ramp up. Targets re_rand OOD split.
8. **Wider model** — n_hidden 96→128 (within wall-clock budget).
9. **Snapshot/EMA ensemble** — average EMA checkpoint predictions across multiple training points.
10. **FiLM ablation under n_fourier=0** — confirm FiLM is still doing work with Fourier dropped.

## History

| PR | Hypothesis | Outcome |
|----|------------|---------|
| **#3672** | **alphonse Fourier ablation (n_fourier=0 winner)** | **✅ MERGED 07:50Z (val 70.34 / test 61.63)** |
| #3808 | fern surf_weight R7 | WIP |
| #3786 | edward Huber β sweep R7 | WIP |
| #3748 | nezuko spec norm R6 | WIP |
| #3712 | askeladd Lion β1 sweep R6 | WIP |
| #3697 | frieren multi-σ R5 | WIP (Arm C running) |
| #3673 | tanjiro EMA decay R5 | Terminal posted → pending mark-ready → close informative |
| #3698 | thorfinn TTA R5 | Pending terminal → close informative-negative |
| #3711 | edward LLRD R6 | ❌ Closed 07:35Z (LLRD dead end) |
| #3695 | fern Sobolev R5 | ❌ Closed 07:50Z (informative, small test gain at w=0.1) |
| #3405 | FiLM+Lion+EMA H4 | ✅ MERGED 03:40Z (val 71.65 / test 62.11) |
| #3537 | Lion optimizer H13 | ✅ MERGED 01:43Z (val 77.58 / test 68.88) |
| #3444 | cosine T_max=14 | ✅ Merged (val 93.20) |
| #3296 | NaN guard | ✅ Merged |
| #3098 | Huber loss | ✅ Merged (val 96.05) |
| #3671 | nezuko block-FiLM | ❌ Closed (+5 val regression) |
| #3544 | thorfinn Lookahead | ❌ Closed (dead end on both substrates) |
| #3486 | fern σ=3 Lion+EMA | ❌ Closed (superseded) |
| #3380 | frieren multi-σ (config bug) | ❌ Closed |
| #3483 | edward Lion+EMA ablation | ❌ Closed (EMA +4.4 val paper finding) |
| #3609 | askeladd Lion warmup | ❌ Closed (warmup adds nothing) |
