<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~10:45 UTC) — edward #3786 closed (Huber β dead-end); FiLM ablation confirmed by alphonse #3817 (−4.35 val contribution); fern #3808 internal signal at surf_weight=20 (−3.28 val); edward assigned R8 H31 (#3913 Reynolds sampler); 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 70.3432** (PR #3672 alphonse — n_fourier=0 + FiLM-output log(Re) + Lion lr=5e-5 wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14, run `297qot5r`)
**test_avg/mae_surf_p = 61.6253** (same run, clean 4-split)

Per-split val: in_dist 79.64, camber_rc 82.43, camber_cruise 51.50, re_rand 67.80
Per-split test: in_dist 69.97, camber_rc 73.96, camber_cruise 42.22, re_rand 60.35

## Merged sequence (improvement cascade)

| PR | Description | val → val | test → test | Δ |
|----|-------------|-----------|-------------|---|
| #3098 | Huber loss | 135.23 → 96.05 | — | −29.1% |
| #3296 | NaN guard | — → 90.00 | first clean test | — |
| #3444 | cosine T_max=14 | 96.05 → 93.20 | 90.00 → 83.54 | −3.0% / −7.2% |
| #3537 | Lion optimizer | 93.20 → 77.58 | 83.54 → 68.88 | −16.8% / −17.5% |
| #3405 | FiLM+Lion+EMA | 77.58 → 71.65 | 68.88 → 62.11 | −7.9% / −9.8% |
| **#3672** | **n_fourier=0 (FiLM+Lion+EMA)** | **71.65 → 70.34** | **62.11 → 61.63** | **−1.8% / −0.8%** |

**Total improvement:** val 135 → 70.34 (−48%), test ~130 → 61.63 (−53%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3817** | **alphonse** | **FiLM ablation under n_fourier=0** | WIP — terminal needed. FiLM confirmed −4.35 val. Asked to post SENPAI-RESULT. |
| #3808 | fern | surf_weight sweep (10→20→40) | WIP — stale. Ran on OLD baseline (n_fourier=16). Internal w20 signal −3.28 val. Asked for n_fourier=0 confirmation arm. |
| #3748 | nezuko | Spectral norm on output head (3 arms) | WIP — ran on OLD baseline. Arm B −2.71 val internally. Asked for n_fourier=0 confirmation arm. |
| #3712 | askeladd | Lion β1 sweep {0.8, 0.9, 0.95} | WIP — β1=0.9 confirmed optimal over β1=0.8; β1=0.95 arm still missing. |
| #3913 | edward | **R8 H31: Re-extremity WeightedRandomSampler {α=0, 0.5, 1.0}** | **Just assigned** |
| #3842 | tanjiro | Sobolev finer sweep w ∈ {0.05, 0.10, 0.15} | WIP R7 H28 |
| #3843 | frieren | Lion lr sweep {2e-5, 5e-5 ctrl, 1e-4} | WIP R7 H29 |
| #3845 | thorfinn | Train-time z-reflection aug (p=0.5/0.25) | WIP R7 H30 |

## Closed this session (informative / dead ends)

- **#3786 edward** — Huber β sweep {0.05, 0.10, 0.20}: β=0.05 locally optimal, β=0.10 gains +0.47 val vs control but within seed noise (σ≈4.6). On OLD baseline (n_fourier=16). Paper-negative: widening Huber β ∈ [0.05, 0.20] does not improve surface-pressure MAE.
- **#3673 tanjiro** — EMA decay: within noise. Informative.
- **#3697 frieren** — Multi-σ Fourier: superseded by n_fourier=0. Informative.
- **#3698 thorfinn** — TTA z-reflection: catastrophic failure. Dataset asymmetry confirmed.
- **#3711 edward** — LLRD: dead end.
- **#3695 fern** — Sobolev R5: small test gain noted.

## Baseline-shift issue — R6 students

R6 students (edward #3786, fern #3808, nezuko #3748, askeladd #3712) all ran on n_fourier=16 (OLD baseline), not n_fourier=0 (PR #3672 merged baseline). Their internal ablations are valid, but raw metrics don't compare to the new baseline. Needed: n_fourier=0 confirmation arms for promising results.

## Candidate confirmations needed

| Student | PR | Signal | Confirmation arm status |
|---------|----|--------|------------------------|
| **nezuko** | #3748 | Arm B spec_norm=output: −2.71 val vs internal ctrl | **Pending — asked to run n_fourier=0 arm** |
| **fern** | #3808 | Arm B surf_weight=20: −3.28 val vs internal ctrl | **Pending — asked to run n_fourier=0 arm** |
| alphonse | #3817 | FiLM ablation confirmed (−4.35 val); best FiLM-on arm val 70.05 (lucky seed) | No new mechanism; close as informative once terminal posted |

## Key findings (cumulative)

1. **FiLM on log(Re) is the biggest mechanism.** Under n_fourier=0 substrate: FiLM-on val 70.05 vs FiLM-off val 74.40 = **−4.35 val / −4.56 test**. Paper-critical ablation confirmed.
2. **EMA(0.997) contributes 4.4 val / 3.8 test on top of Lion** — cleanest single-mechanism measurement.
3. **Fourier PE inert under FiLM+Lion+EMA.** n_fourier=0 wins. New baseline confirmed by alphonse's arm A reproduction.
4. **EMA decay robust in [0.995, 0.997].** No merge.
5. **Sobolev surface regularization: small test gain at w=0.1.** Finer sweep pending (#3842 tanjiro).
6. **LLRD doesn't transfer** from fine-tuning to train-from-scratch.
7. **LR warmup adds nothing** to Lion at 14-epoch budget.
8. **Block-FiLM regresses +5 val.** Output-only is correct topology.
9. **Lookahead dead end** across both substrates.
10. **TTA z-reflection fails** — dataset AoA asymmetry.
11. **Huber β=0.05 locally optimal** in [0.05, 0.20] under FiLM+Lion+EMA.
12. **Seed noise floor measured: ≈2.7 val** (two identical FiLM-on/n_fourier=0 runs = val 70.05 vs 72.82).

## R8+ reserved hypotheses

1. **Per-split loss weighting** — upweight camber_rc training samples
2. **Multi-seed ensemble** — average 2–3 seeds of new baseline (blocked: no --seed arg)
3. **Geometric features** — chord-normal coords, local surface curvature κ
4. **Curriculum on Reynolds** — train low-Re first
5. **Wider model** — n_hidden 96→128
6. **Snapshot/EMA ensemble** — average EMA checkpoints across multiple training points
7. **Fern surf_weight=20 on new baseline** — highest priority if confirmation arm wins
8. **Nezuko spec_norm output on new baseline** — high priority if confirmation arm wins

## History

| PR | Hypothesis | Outcome |
|----|------------|---------|
| **#3913** | **edward R8 H31: Re-extremity sampler** | WIP |
| #3845 | thorfinn train-z-aug R7 | WIP |
| #3843 | frieren Lion lr sweep R7 | WIP |
| #3842 | tanjiro Sobolev finer R7 | WIP |
| #3817 | alphonse FiLM ablation n_fourier=0 | WIP (terminal pending — FiLM confirmed) |
| #3808 | fern surf_weight R7 | WIP (confirmation arm needed) |
| **#3786** | **edward Huber β sweep R7** | **❌ Closed 10:35Z (informative — β=0.05 optimal)** |
| #3748 | nezuko spec norm R6 | WIP (confirmation arm needed) |
| #3712 | askeladd Lion β1 R6 | WIP (β1=0.95 arm missing) |
| **#3672** | **alphonse Fourier ablation** | **✅ MERGED ~07:50Z (val 70.34 / test 61.63)** |
| #3698 | thorfinn TTA R5 | ❌ Closed (catastrophic) |
| #3697 | frieren multi-σ R5 | ❌ Closed (superseded) |
| #3673 | tanjiro EMA decay R5 | ❌ Closed (informative) |
| #3711 | edward LLRD R6 | ❌ Closed (dead end) |
| #3695 | fern Sobolev R5 | ❌ Closed (informative) |
| #3405 | FiLM+Lion+EMA H4 | ✅ MERGED (val 71.65 / test 62.11) |
| #3537 | Lion optimizer H13 | ✅ MERGED (val 77.58 / test 68.88) |
