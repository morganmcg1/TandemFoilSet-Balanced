<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~04:50 UTC, Round 5 fully staffed — 8 of 8 students have active hypotheses on `willow-pai2i-48h-r5`)
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

## Round 5 — fully staffed (8 of 8 students active)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3609 | askeladd | Lion + LR warmup ablation (warmup ∈ {0, 500, 1000}) | WIP — Arms A (79.13) + B (79.89) done, Arm C running; warmup hurts |
| #3483 | edward | Lion+EMA ablation (no-Fourier 73.10 done, σ=3 + pure-Lion arms running) | WIP |
| #3671 | nezuko | **FiLM at intermediate Transolver blocks (layer-wise conditioning)** | WIP (R5 H16) |
| #3672 | alphonse | **Fourier ablation under FiLM+Lion+EMA (n_fourier=0 vs σ=3 vs σ=10)** | WIP (R5 H17) |
| #3673 | tanjiro | **EMA decay sweep under FiLM+Lion (0.995 vs 0.997 vs 0.999)** | WIP (R5 H18) |
| **#3695** | **fern** | **Sobolev loss on surface ∂p/∂s (physics-motivated)** | WIP (R5 H19, just assigned) |
| **#3697** | **frieren** | **Multi-σ Gaussian Fourier features under FiLM+Lion+EMA (proper wiring)** | WIP (R5 H20, just assigned) |
| **#3698** | **thorfinn** | **TTA via z-reflection symmetry (free inference gain)** | WIP (R5 H21, just assigned) |

## Closed this turn (informative negative / superseded)

- **#3544 thorfinn** — Lookahead-on-Lion val 89.39 (+17.7 regression). Dead end across both AdamW and Lion substrates. Paper-relevant negative for optimizer-family ablation.
- **#3486 fern** — σ=3 + Lion + EMA val 73.81. Beats Lion baseline (−3.77) but loses to new FiLM baseline. Confirms σ-monotonic finding does NOT transfer from AdamW to Lion. Paper-valuable.
- **#3380 frieren** — Multi-σ config bug (n_fourier=0 at runtime). Effectively Lion+EMA no-Fourier sample (val 76.95). Student agreed to close + reassign.

## Key findings (cumulative)

1. **FiLM on log(Re) is the biggest new mechanism on top of Lion+EMA.** Adds ~5.9 val / 6.8 test over Lion+EMA-only.
2. **EMA(0.997) clearly compounds with Lion** (alphonse + tanjiro avg ~77.7 with EMA vs 77.58 without). Effect is noisy but real.
3. **Fourier σ is marginal under Lion+EMA.** edward no-Fourier (73.10) ≈ fern σ=3 (73.81) ≈ alphonse σ=10 (76.15). FiLM dominates.
4. **Lookahead does NOT compound with Lion or AdamW.** Two distinct optimizer substrates, both regress by ≥17 val points. Confirmed dead end.
5. **LR warmup hurts Lion** at our 14-effective-epoch budget. askeladd Arm B (warmup=500, val 79.89) > Arm A (warmup=0, val 79.13). Cosine schedule already provides implicit warmup.
6. **σ-monotonic finding does NOT transfer from AdamW to Lion.** Under AdamW+EMA, σ=3 was best; under Lion+EMA, σ doesn't matter much (and no-Fourier slightly wins).

## Round 5 hypothesis map (mechanism axes being tested)

| Axis | Hypothesis | PR / student |
|------|-----------|--------------|
| FiLM topology | FiLM at intermediate layers (layer-wise conditioning) | #3671 nezuko |
| Fourier under FiLM | n_fourier=0 vs σ=3 vs σ=10 (single-scale) | #3672 alphonse |
| Fourier under FiLM | Multi-σ {3,10,30} (multi-scale) | #3697 frieren |
| EMA decay | 0.995 vs 0.997 vs 0.999 | #3673 tanjiro |
| Loss regularizer | Sobolev surface ∂p/∂s | #3695 fern |
| Inference-time | TTA via z-reflection | #3698 thorfinn |
| Optimizer | Lion + LR warmup | #3609 askeladd |
| Ablation | Lion+EMA without Fourier (full sweep) | #3483 edward |

## Round 6+ / reserved hypotheses

- **Lion β1 sweep (β1 ∈ {0.9, 0.95})** — paper ablation. Reserved for askeladd's next round.
- **Layer-wise LR decay (LLRD)** — per-block LR multiplier.
- **Per-split loss weighting** — boost cruise/camber splits in training loss.
- **Architectural variants** — wider slice_num, deeper layers (if budget allows).
- **Geometric features** — chord-normal coordinates, curvature, etc., conditioning beyond log(Re).
- **Distillation from longer-trained teacher** — if any arm starts converging well past 14 epochs.

## History

| PR | Hypothesis | Outcome |
|----|------------|---------|
| #3405 | FiLM+Lion+EMA H4 | ✅ MERGED 03:40Z (val 71.65 / test 62.11) |
| #3537 | Lion optimizer H13 | ✅ MERGED 01:43Z (val 77.58 / test 68.88) |
| #3444 | cosine T_max=14 | ✅ Merged (val 93.20) |
| #3296 | NaN guard | ✅ Merged |
| #3098 | Huber loss | ✅ Merged (val 96.05) |
| #3544 | thorfinn Lookahead | ❌ Closed 04:33Z (val 89.39, dead end on both substrates) |
| #3486 | fern σ=3 Lion+EMA | ❌ Closed 04:33Z (val 73.81, superseded by FiLM merge) |
| #3380 | frieren multi-σ (config bug) | ❌ Closed 04:33Z (val 76.95, student agreed) |
| #3379 | alphonse EMA compound | Closed (superseded by nezuko) |
| #3484 | tanjiro EMA decay sweep | Closed (superseded) |
| #3412/#3407/#3410/#3409 | Various R2 dead ends | ❌ Closed |
