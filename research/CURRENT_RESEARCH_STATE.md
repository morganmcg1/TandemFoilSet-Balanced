<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-15 (~18:20 UTC, Round 5, 48h launch, `willow-pai2i-48h-r5`)
- **Human researcher directives:** None received as of this writing.

## Current best

**val_avg/mae_surf_p = 96.05** (PR #3098, merged — SmoothL1 beta=0.05 Huber loss, W&B: `md6so639`)  
**test partial avg (excl. cruise) = 93.41** — in_dist 96.04, camber_rc 100.16, re_rand 84.02  
**test_avg/mae_surf_p = NaN** ⚠️ — cruise GT bug (000020.pt has 761 Inf p-values); fix in progress (PR #3296, thorfinn)

Previous best: PR #3123 Fourier PE n=16 → val_avg = 130.46  
Unmodified baseline (no changes): val_avg = 135.23

**Primary metric:** `val_avg/mae_surf_p` — equal-weight mean surface pressure MAE across 4 splits.  
**Binding constraint:** SENPAI_TIMEOUT_MINUTES=30.0, SENPAI_MAX_EPOCHS=50, 1 GPU per student.  
**Note:** 30-min wall clock binds at ~epoch 14. All runs severely under-trained.

## Round 1 — Results (started ~15:20 UTC 2026-05-15)

| PR | Student | Hypothesis | Best val_avg | Δ vs baseline | Status |
|----|---------|------------|--------------|----------------|--------|
| #3098 | alphonse | SmoothL1/Huber loss β=0.05 | **96.05** | **-26.4%** | **MERGED** ✅ |
| #3109 | frieren | bf16 + bigger batch | 133.72 | +2.5% worse | CLOSED ❌ |
| #3114 | nezuko | Grad-clip(1.0) + EMA(0.999) | **102.67** | **-21.3%** | WIP — result ready |
| #3103 | edward | Slice-num scaling (64→128→192) | 124.39 | -4.6% | WIP — result ready |
| #3105 | fern | Linear warmup + cosine LR | 127.82 | -2.0% | WIP — result ready |
| #3118 | tanjiro | Per-channel surface loss | 130.51 | -0.0% | WIP — result ready |
| #3100 | askeladd | Scale-up h128→256 | 142.46 | +8.9% worse | WIP — result ready |
| #3296 | thorfinn | Diagnose & fix cruise NaN | N/A | critical fix | WIP — debugging |

## Round 1 — Key findings

1. **Huber/SmoothL1 loss is the dominant mechanism** — 26% improvement at 14 epochs. Effect size far exceeded prediction. Pressure is the dominant heavy-tailed channel; β=0.05 transition keeps more gradients in linear regime during under-training.
2. **Grad-clip + EMA also strong** (nezuko, 21% improvement) — pure optimization mechanism, orthogonal to Huber.
3. **Architecture and schedule changes failed under-training constraint**: slice scaling, scale-up, warmup-cosine, per-channel weighting all didn't beat baseline. The 30-min wall-clock is the dominant bottleneck.
4. **bf16 speedup real but batch scaling failed**: 18% faster epochs, but larger batches without LR scaling hurt convergence.
5. **NaN root cause fully identified**: two contributors — (a) model pred overflow on cruise OOD samples; (b) GT sample 000020.pt has 761 Inf p-values. Fix requires both pred and y guards.
6. **Neither winner used Fourier PE** — both ran on pre-Fourier merge baseline (n_fourier=0). Round 2 compound stack combining Fourier PE + Huber + EMA + clip expected to push further.

## Round 2 — Active assignments

| PR | Student | Hypothesis | Priority |
|----|---------|------------|----------|
| (new) | alphonse | **H1: Compound stack** — EMA(0.999) + grad-clip(1.0) + Huber(β=0.05) + Fourier PE(n=16, σ=10) | 1 |
| (new) | frieren | **H4: Fourier sigma sweep** — n=16 fixed, sigma ∈ {4, 10(ref), 20} | 2 |

## Round 2 — Pending assignments (students WIP → idle on merge)

When nezuko, edward, fern, tanjiro, askeladd, thorfinn become idle:

| Priority | Hypothesis | Best student |
|----------|------------|--------------|
| 2 | H2: FiLM conditioning on log(Re) — target re_rand | nezuko |
| 2 | H3: Per-sample relative L2 loss — complement to Huber | edward or fern |
| 3 | H5: 1st-Order SAM — flat minima, OOD | tanjiro or askeladd |
| 3 | H6: AoA reflection symmetry augmentation | fern or tanjiro |
| 4 | H7: Cosine T_max fix (T_max=14 to match wall-clock cap) | thorfinn |
| 4 | H8: β sweep (β=0.025) to find optimal Huber knee | askeladd |

See `research/RESEARCH_IDEAS_2026-05-15_round2.md` for full hypothesis details.
