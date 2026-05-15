<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-15 (~18:55 UTC, Round 2 launching, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received as of this writing.

## Current best

**val_avg/mae_surf_p = 96.05** (PR #3098, merged — SmoothL1 beta=0.05 Huber loss, W&B: `md6so639`)
**test partial avg (excl. cruise) = 93.41** — in_dist 96.04, camber_rc 100.16, re_rand 84.02
**test_avg/mae_surf_p = NaN** ⚠️ — cruise GT bug; fix in PR #3296 (sent back for rebase onto Huber baseline)

Previous best: PR #3123 Fourier PE n=16 → val_avg = 130.46
Unmodified baseline: val_avg = 135.23

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across 4 splits.
**Binding constraint:** SENPAI_TIMEOUT_MINUTES=30.0, SENPAI_MAX_EPOCHS=50, 1 GPU per student.
**Note:** 30-min wall clock binds at ~epoch 14. All runs severely under-trained.

## Round 1 — Final results

| PR | Student | Hypothesis | Best val_avg | Δ vs new baseline (96.05) | Status |
|----|---------|------------|--------------|---------------------------|--------|
| #3098 | alphonse | SmoothL1/Huber loss β=0.05 | **96.05** | (set new baseline) | **MERGED** ✅ |
| #3114 | nezuko | Grad-clip(1.0) + EMA(0.999) | 102.67 | +6.9% | CLOSED (subsumed by #3379 stack) |
| #3103 | edward | Slice-num scaling | 124.39 | +29% | CLOSED |
| #3105 | fern | Linear warmup + cosine LR | 127.82 | +33% | CLOSED |
| #3118 | tanjiro | Per-channel surface loss | 130.51 | +36% | CLOSED |
| #3109 | frieren | bf16 + bigger batch | 133.72 | +39% | CLOSED |
| #3100 | askeladd | Scale-up h128→256 | 150.94 | +57% | CLOSED |
| #3296 | thorfinn | Two-pronged NaN guard | 142.20 (pre-Huber) | n/a — code fix | REBASE-IN-FLIGHT |

## Round 1 — Validated mechanisms

1. **Huber loss is the dominant Round 1 mechanism** — 26% improvement over PR #3123 baseline, 30% improvement over unmodified. Pressure is the dominant heavy-tailed channel; β=0.05 transition keeps more gradients in linear regime during under-training.
2. **Grad-clip + EMA is a confirmed orthogonal optimization layer** — 2nd place standalone. Will appear stacked on Huber in alphonse's Round 2 #3379.
3. **Capacity is NOT the binding constraint** — both scale-up (askeladd) and slice scaling (edward) regress significantly. 30-min wall clock binds; bigger models cannot converge.
4. **Per-channel weighting is counter-productive** — multi-task coupling is doing useful work; deprioritizing Ux/Uy hurts pressure too.
5. **Warmup is incompatible with our regime** — cosine T_max=50 already keeps LR near peak; warmup throws away early gradient signal.
6. **bf16 speedup is real but batch scaling requires LR scaling** — bs=10 viable, bs=12 OOMs on cruise meshes.
7. **NaN root cause is two-pronged** — pred overflow AND inf in GT sample 000020.pt. Two-pronged guard works (thorfinn #3296).
8. **Run-to-run variance estimate (fern's 3 arm-A reruns):** σ ≈ 4.6 on val_avg, ~3.6% relative. Any single-arm delta below this is noise.
9. **First valid test_avg of launch:** askeladd arm-A 136.70 (with Edward's NaN-guard data path).

## Round 2 — Active assignments

| PR | Student | Hypothesis | Target Split | Priority |
|----|---------|------------|--------------|----------|
| #3379 | alphonse | **H1: Compound stack** — EMA(0.999) + grad-clip(1.0) + Huber(β=0.05) + Fourier PE | val_avg overall | 1 |
| #3380 | frieren | **H4: Fourier sigma sweep** — n=16 fixed, sigma ∈ {4, 10(ref), 20} | camber_rc | 2 |
| #3296 | thorfinn | **NaN guard rebase + clean test_avg confirm** on Huber baseline | (infrastructure) | 1 |

## Round 2 — Full assignment roster (all 8 students WIP as of ~19:25 UTC)

| PR | Student | Hypothesis | Target Split |
|----|---------|------------|--------------|
| #3379 | alphonse | **H1: Compound stack** — EMA(0.999) + grad-clip(1.0) + Huber(β=0.05) + Fourier PE | val_avg overall |
| #3380 | frieren | **H4: Fourier sigma sweep** — n=16 fixed, sigma ∈ {4, 10, 20} | camber_rc |
| #3296 | thorfinn | **NaN guard rebase** — two-pronged guard on Huber baseline | (infrastructure) |
| #3405 | nezuko | **H2: FiLM on log(Re)** — explicit Re-regime conditioning | re_rand |
| #3407 | edward | **H3: Per-sample Relative L2** — cross-sample scale invariance | re_rand, camber_cruise |
| #3409 | fern | **H6: AoA reflection augmentation** — double RaceCar training data | in_dist, camber_rc |
| #3410 | tanjiro | **H5: 1st-Order SAM** — flat-minima OOD optimizer | OOD splits |
| #3412 | askeladd | **H7: DropPath stochastic depth** — implicit ensemble regularizer | OOD splits |

## Reserved for Round 3 / plateau triggers

- **H8: Sobolev loss on surface ∂p/∂s** — physics-motivated gradient matching. Hold until H1–H7 plateau.
- **Best-checkpoint test eval (not terminal-epoch)** — paper-facing improvement, decoupled from val_avg gains.
- **Re-conditioned positional features** (DOS-friendly): radial basis around airfoil leading edge for camber_rc recovery.
- **Cosine T_max=14 fix** — schedule recalibration to actual wall-clock epoch count (revisit if relative gains plateau).

## Operational notes

- **All 8 students active as of 19:25 UTC.** Zero idle GPUs.
- **#3296 thorfinn** — rebase-in-flight. Has detailed advisor instructions. Once thorfinn pushes rebase + confirmation run on Huber baseline, this merges immediately (critical NaN guard for paper test_avg).
- **Expected merge order**: #3296 first (fixes test_avg), then whichever of H1–H7 beats 96.05.
- **Round 1 mechanisms confirmed orthogonal**: Huber (loss), clip+EMA (optimization), Fourier PE (positional). Compound stack (alphonse #3379) will confirm stacking.
