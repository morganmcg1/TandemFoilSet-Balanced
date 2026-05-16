<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~09:35 UTC) — 3 R5 closes (#3673 tanjiro EMA, #3697 frieren multi-σ, #3698 thorfinn TTA); 3 new R7 assignments made; 8/8 students staffed.
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

## Active R7 experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3817** | **alphonse** | **FiLM ablation under n_fourier=0 (2 arms: FiLM on vs off)** | WIP R7 H27 |
| #3808 | fern | surf_weight sweep (10→20→40) — surface-loss reweighting | WIP R7 H26 |
| #3786 | edward | Huber β sweep (0.05→0.1→0.2) — peak-pressure loss tuning | WIP R7 H25 |
| #3748 | nezuko | Spectral norm on output head + FiLM layers (3 arms) | WIP R6 H24 |
| #3712 | askeladd | Lion β1 sweep {0.8, 0.9, 0.95} | WIP R6 H23 |
| **#3842** | **tanjiro** | **Sobolev finer sweep w ∈ {0.05, 0.10, 0.15} — extend fern's signal** | **Just assigned R7 H28** |
| **#3843** | **frieren** | **Lion LR sweep {2e-5, 5e-5 ctrl, 1e-4} — basic LR ablation** | **Just assigned R7 H29** |
| **#3845** | **thorfinn** | **Train-time z-reflection aug (p=0.5/0.25) — close loop on TTA failure** | **Just assigned R7 H30** |

## Closed this session (informative / dead ends)

- **#3673 tanjiro** — EMA decay sweep {0.995, 0.997, 0.999}: best arm ema=0.995 val 71.51 (wash with baseline 71.65, within noise σ≈4.6). Informative: EMA decay robust in [0.995, 0.997].
- **#3697 frieren** — Multi-σ Fourier {3, 10, 30}: superseded by n_fourier=0 merge. No individual σ beat baseline.
- **#3698 thorfinn** — TTA z-reflection: catastrophic regression (val 72→307). Dataset asymmetry confirmed: raceCar AoA ∈ [-10°, 0°] → z-flipped inputs are OOD. Thorfinn's diagnostic subset sweep confirmed the finding.
- **#3711 edward** — LLRD γ ∈ {1.0, 0.85, 0.65}: monotonic regression, γ=0.65 → +21 val. LLRD is fine-tuning-only, not train-from-scratch. Dead end.
- **#3695 fern** — Sobolev surface ∂p/∂s at w ∈ {0, 0.1, 0.5}: best arm w=0.1 val 71.84 (wash) / test 61.93 (small gain). Informative.

## Key findings (cumulative)

1. **FiLM on log(Re) is the biggest new mechanism.** Adds ~5.9 val / 6.8 test over Lion+EMA-only.
2. **EMA(0.997) contributes 4.4 val / 3.8 test on top of Lion** — cleanest single-mechanism measurement.
3. **Fourier PE is inert under FiLM+Lion+EMA.** n_fourier=0 (val 70.34) = simplification win; FiLM already captures the flow-regime conditioning. New baseline uses n_fourier=0.
4. **EMA decay robust in [0.995, 0.997].** Best arm ema=0.995 is a wash (val 71.51, noise). No merge.
5. **Sobolev surface regularization points in right direction.** w=0.1 gives small test gain (−0.18) at flat val. Finer sweep reserved for R7.
6. **LLRD does NOT transfer from fine-tuning to train-from-scratch.** γ<1 monotonically hurts.
7. **LR warmup adds nothing to Lion** at 14-epoch budget. Cosine provides sufficient implicit warmup.
8. **Block-FiLM regresses +5 val.** Output-only FiLM is the correct topology.
9. **Lookahead dead end** across both AdamW and Lion substrates.
10. **TTA via z-reflection fails** — cambered foils are not z-symmetric. Train-time augmentation (H30) is the follow-up.

## R7 hypothesis map (all 8 axes covered)

| Axis | Hypothesis | PR / student |
|------|-----------|--------------|
| FiLM ablation (paper-critical) | FiLM on vs off under n_fourier=0 | #3817 alphonse |
| Loss: surf_weight | surf_weight ∈ {10 ctrl, 20, 40} | #3808 fern |
| Loss: Huber β | β ∈ {0.05 ctrl, 0.1, 0.2} | #3786 edward |
| Optimizer hyperparameter | Lion β1 ∈ {0.8, 0.9, 0.95} | #3712 askeladd |
| Lipschitz constraint | Spectral norm on output head (+ FiLM layers) | #3748 nezuko |
| Loss: Sobolev surface grad | w ∈ {0.05, 0.10, 0.15} finer sweep | #3842 tanjiro |
| Optimizer hyperparameter | Lion lr ∈ {2e-5, 5e-5 ctrl, 1e-4} | #3843 frieren |
| Data augmentation | Train-time z-reflection (p ∈ {0, 0.25, 0.5}) | #3845 thorfinn |

## Round 7+ reserved hypotheses

1. **Multi-seed confirmation of n_fourier=0** — 3 seeds to verify Δ > σ≈4.6 (paper appendix table). Blocked: no `--seed` arg in train.py.
2. **Per-split loss weighting** — upweight camber_rc/in_dist at training time.
3. **Multi-seed ensemble** — average predictions from 2–3 seeds of new baseline.
4. **Geometric features** — chord-normal coordinates, local surface curvature κ, surface-normal direction.
5. **Curriculum on Reynolds** — train low-Re samples first, ramp up. Targets re_rand OOD split.
6. **Wider model** — n_hidden 96→128 (if within wall-clock budget).
7. **Snapshot/EMA ensemble** — average EMA checkpoint predictions across multiple training points.

## History

| PR | Hypothesis | Outcome |
|----|------------|---------|
| **#3845** | **thorfinn train-time z-aug R7** | WIP |
| **#3843** | **frieren Lion LR sweep R7** | WIP |
| **#3842** | **tanjiro Sobolev finer sweep R7** | WIP |
| #3817 | alphonse FiLM ablation n_fourier=0 | WIP R7 |
| #3808 | fern surf_weight R7 | WIP |
| #3786 | edward Huber β sweep R7 | WIP |
| **#3672** | **alphonse Fourier ablation (n_fourier=0 winner)** | **✅ MERGED ~07:50Z (val 70.34 / test 61.63)** |
| #3748 | nezuko spec norm R6 | WIP |
| #3712 | askeladd Lion β1 sweep R6 | WIP |
| #3698 | thorfinn TTA R5 | ❌ Closed 09:24Z (catastrophic regression, dataset asymmetry) |
| #3697 | frieren multi-σ R5 | ❌ Closed 09:24Z (superseded by n_fourier=0) |
| #3673 | tanjiro EMA decay R5 | ❌ Closed 09:24Z (val 71.51 wash, informative) |
| #3711 | edward LLRD R6 | ❌ Closed 07:35Z (dead end) |
| #3695 | fern Sobolev R5 | ❌ Closed ~07:50Z (informative, test gain noted) |
| #3405 | FiLM+Lion+EMA H4 | ✅ MERGED 03:40Z (val 71.65 / test 62.11) |
| #3537 | Lion optimizer H13 | ✅ MERGED 01:43Z (val 77.58 / test 68.88) |
| #3444 | cosine T_max=14 | ✅ Merged (val 93.20) |
| #3296 | NaN guard | ✅ Merged |
| #3098 | Huber loss | ✅ Merged (val 96.05) |
| #3671 | nezuko block-FiLM | ❌ Closed (+5 val regression) |
| #3544 | thorfinn Lookahead | ❌ Closed (dead end) |
| #3486 | fern σ=3 Lion+EMA | ❌ Closed (superseded) |
| #3380 | frieren multi-σ (config bug) | ❌ Closed |
| #3483 | edward Lion+EMA ablation | ❌ Closed (EMA +4.4 val paper finding) |
| #3609 | askeladd Lion warmup | ❌ Closed (warmup adds nothing) |
