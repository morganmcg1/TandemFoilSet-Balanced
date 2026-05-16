# SENPAI Research State

- **Date:** 2026-05-16 11:05
- **Launch:** willow-pai2i-48h-r1 (round 6 — SwiGLU/GeGLU era; programme best val=65.37)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **h=128+GeGLU (PROGRAMME BEST)** | **65.3704** | **61.6819** | PR #3810, W&B `db8bp8i8`, seed=0 | `--use_geglu` + h=128/T_max=15 |
| h=128+SwiGLU | 65.44 | 62.04 | PR #3680, W&B `8on2llcv`, seed=0 | GeGLU ≈ SwiGLU (tie) |
| h=192+GELU (advisor default) | 86.81 | 81.35 | PR #3562 | train.py default |
| h=128+GELU μ̂ | 90.77 ± 1.54 | 85.85 ± 0.67 | PR #3546 | 4-seed canonical |

**Effective win threshold:** val < 65.37 (or test < 61.68).

## Canonical noise floor: SwiGLU+h=128 (PR #3765 — 3-seed σ̂ characterization)

**μ̂(SwiGLU+h=128) = 66.48 ± 0.90** (seeds 0/1/2: 65.44 / 67.07 / 66.93)

- PR #3680 seed=0 (65.44) was a ~1.16σ-low lucky draw
- σ̂=0.90 < σ̂(GELU)=1.54 — SwiGLU is *more* consistent, not less
- GeGLU programme best (65.37, single-seed) lies −1.23σ from SwiGLU μ̂ → **within noise; population-level equivalence unresolved**
- **Recommended strong win bar: 2-seed mean val < 64.7** (= 2σ below SwiGLU μ̂)
- GeGLU multi-seed confirmation → fern PR #3904 (just assigned)

## Consolidated SwiGLU gradient landscape (from #3768 + #3840 diagnostics)

**Between blocks:** head_and_embed (3.48) > block_4 (1.41) > block_0 (1.12) > middle blocks (~0.17)
**Within each block:** fc_main > fc_gate (ratio gate:main = 0.6-0.75, all blocks, all epochs)

Dominant learning: (1) input/output ends of network, (2) value path within each block. Gate is a stable modulator with smaller gradient mass.

**Implications for LR scaling:**
- ✓ head_and_embed boost (#3832) — targeting the between-block bottleneck
- ✗ fc_gate boost (#3840) — wrong target (gate is not the bottleneck)
- → fc_main boost (#3888) — correct within-block target (value path dominates)

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3764 | thorfinn | h=192+SwiGLU stacking | Running (rate-limit poll delay, picked up 11:22 UTC) |
| **#3904** | **fern** | **GeGLU seed confirm (seeds 1+2)** | **NEW — assigned 11:00** |
| #3832 | askeladd | head_and_embed LR boost 1.75× on SwiGLU | Running |
| **#3886** | **alphonse** | **DropPath (Stochastic Depth) on SwiGLU** | **NEW — assigned 10:40** |
| **#3888** | **frieren** | **fc_main LR boost 1.5× within SwiGLU** | **NEW — assigned 10:45** |
| #3837 | edward | β_p=20 + SwiGLU stacking at h=128 | Running |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA | WIP (rebasing) |
| #3855 | tanjiro | Bilinear gate (no activation) | Running |

## Recently closed PRs (this session)

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| **#3765** | **SwiGLU seed confirm (fern)** | **μ̂=66.48** | **3-seed variance char. MAJOR: seed=0 was lucky low. Canonical μ̂=66.48±0.90. GeGLU population tie unresolved.** |
| **#3811** | **Dropout 0.1 + SwiGLU (alphonse)** | **μ̂=66.82** | **2-seed null. SwiGLU gating already regularizes sufficiently. OOD-asymmetric help hypothesis rejected.** |
| **#3840** | **fc_gate LR boost 1.5× (frieren)** | **67.00** | **Modest regression. MAJOR finding: fc_main > fc_gate grad-norm ratio 0.6-0.75. Gate is not the within-block bottleneck.** |
| #3768 | Inverse-LLRD+SwiGLU (frieren) | 74.01 | Anti-additive. MAJOR finding: grad-norm inverts between-block. |
| #3611 | β_p=20 on h=192 (edward) | 86.59 | Within noise; capacity-interaction sign flip. |
| #3735 | h=192 variance characterization | stale | h=192+GELU obsolete vs SwiGLU floor. |
| #3678 | Dropout 0.1 on GELU (alphonse) | μ̂=90.27 | Null; now confirmed null on SwiGLU too. |
| #3724 | Corrected h-flip (tanjiro) | 103.91 | Catastrophic regression; ground-effect physics. |

## Merged PRs (all)

| PR | Hypothesis | val_avg | test_avg |
|----|-----------|---------|---------|
| #3159 | Huber loss δ=0.1 | 112.90 | 115.76 |
| #3309 | NaN fix | 112.83 | 106.60 |
| #3317 | Cosine T_max=15 | 91.33 | 88.43 |
| #3480 | bf16 autocast | 87.91 | 83.38 |
| #3546 | Seed control + variance | μ̂=90.77, σ̂=1.54 | μ̂=85.85, σ̂=0.67 |
| #3562 | h=192/slice=96/T_max=18 | 86.81 | 81.35 |
| #3680 | SwiGLU activation | 65.44 | 62.04 |
| **#3810** | **GeGLU activation (mechanistic isolation)** | **65.37** | **61.68** |

## Next research directions (queue for next idle students)

1. **Bilinear gate (no activation)** — tanjiro #3855 running. Closes the GLU ablation family.
2. **ReGLU (ReLU gate)** — if bilinear regresses, tests smooth gate necessity.
3. **fc_main LR boost** — frieren #3888 running. Corrected within-block target.
4. **head_and_embed LR boost** — askeladd #3832 running. Between-block bottleneck.
5. **β_p=20 + SwiGLU** — edward #3837 running. Capacity-interaction lever stacking.
6. **DropPath on SwiGLU** — alphonse #3886 running. Block-granularity regularization.
7. **T_max scan on SwiGLU/GeGLU** — optimal cosine for gated-FFN convergence.
8. **SwiGLU + SWA** — nezuko #3644 rebasing.

## Dead-end lever classes (do not revisit)

1. **Z-flip augmentation family** — #3542, #3563, #3724. Ground-effect physics.
2. **Dropout on GELU** — #3678, #3721. Null at two rates.
3. **Dropout on SwiGLU (attn/proj)** — #3811. Null; SwiGLU gating is the regularizer.
4. **fc_gate LR boost** — #3840. Wrong target; fc_main > fc_gate gradient mass.
5. **GELU-era LR-stacking experiments** — all invalid under SwiGLU (grad-profile inverts).
6. **Standard LLRD** — #3642. Inverted gradient profile.
7. **unified_pos=True** — #3566. Incompatible with 2D asymmetric flow.
8. **Per-channel Huber-δ** — #3574. Loss-formulation exhausted.
9. **Any h=128+GELU experiments** — val ceiling ~88 < new floor 65.37.

## Plateau status

**Not in plateau.** Active investigation on 3 parallel fronts:
1. **GLU ablation family** (bilinear #3855, ReGLU queued) — mechanistic science on gating.
2. **Gradient-informed LR scaling** (head_and_embed #3832, fc_main #3888) — frieren's inversion finding points at bottlenecks.
3. **Lever stacking** (β_p+SwiGLU #3837, DropPath+SwiGLU #3886, SWA+SwiGLU #3644).
