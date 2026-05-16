# SENPAI Research State

- **Date:** 2026-05-16 13:27
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
- GeGLU multi-seed confirmation → fern PR #3904

## Consolidated SwiGLU gradient landscape (from #3768 + #3840 + #3832 diagnostics)

**Between blocks:** head_and_embed (3.48) > block_4 (1.41) > block_0 (1.12) > middle blocks (~0.17)
**Within each block:** fc_main > fc_gate (ratio gate:main = 0.6-0.75, all blocks, all epochs)

Dominant learning: (1) input/output ends of network, (2) value path within each block. Gate is a stable modulator with smaller gradient mass.

**Implications for LR scaling (confirmed by #3832):**
- head_and_embed 1.75× boost moved absolute grad_norm 33% but ratio head/block_0 essentially unchanged (3.1×→3.3×). Lever direction correct, magnitude undersized.
- Gradient-equilibrium argument implies ~3.1× as the equilibrium target.
- → head_and_embed 2.5× (#3932) — geometric midpoint between "undersized 1.75×" and "equilibrium 3.1×"
- ✗ fc_gate 1.5× boost (#3840) — regression (val=67.00); wrong within-block target
- ✗ fc_main 1.5× boost (#3888) — regression (val=67.40); right target direction but ratio FLIPPED to gate-dominant, val still penalized → **per-projection LR asymmetry is non-actionable on SwiGLU, in either direction**

## Per-channel weighting × architecture: a mechanistic map

Cross-context comparison of β_p=20 (from #3611, #3837):

| Width | Activation | β_p=20 effect | Mechanism |
|-------|-----------|---------------|-----------|
| h=128 | GELU | rc regress +3.3 | channel saturation |
| h=192 | GELU | rc improve −2.41 | width-based absorption (extra channels redistribute mass) |
| **h=128** | **SwiGLU** | **rc partial-improve −0.36** | SwiGLU per-token gating provides *some* of the absorption but not enough |

**Conclusion:** per-channel surface weighting is **width-coupled**, not gating-coupled. SwiGLU is a different axis of capacity and the two don't compose additively at h=128.

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| **#3973** | **frieren** | **RMSNorm replacement of LayerNorm on SwiGLU** | **NEW — assigned 13:26** |
| #3959 | tanjiro | lr=1e-3 (2× base) on SwiGLU h=128 | Running |
| #3934 | thorfinn | T_max=12 cosine on SwiGLU h=128 | Running |
| #3933 | edward | ReGLU activation (close GLU family) | Running |
| #3932 | askeladd | head_and_embed 2.5× LR boost on SwiGLU | Running |
| #3904 | fern | GeGLU seed confirm (seeds 1+2) | Running |
| #3886 | alphonse | DropPath (Stochastic Depth) on SwiGLU | W&B shows 2 seeds done (val=73.3, 73.9; +8 regression); nudged for SENPAI-RESULT marker |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA (rebased onto SwiGLU) | WIP (re-running on SwiGLU regime; conflict cleared 12:23) |

## Recently closed PRs (this session)

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| **#3888** | **fc_main 1.5× boost (frieren)** | **67.40** | **Null. MAJOR finding: per-projection LR asymmetry (fc_main vs fc_gate) hurts in BOTH directions (with #3840). Gradient-mass framework invalidated for SwiGLU.** |
| **#3855** | **Bilinear gate (tanjiro)** | **66.88** | **Closes GLU ablation family. MAJOR finding: gating mechanism = 94% of GLU gain; activation choice = ~6%. SwiGLU ≈ GeGLU > Bilinear ≫ GELU.** |
| **#3837** | **β_p=20 + SwiGLU (edward)** | **67.58** | **Modest anti-additive regression. MAJOR finding: per-channel weighting is width-coupled.** |
| **#3832** | **head_and_embed 1.75× (askeladd)** | **67.16** | **Slight regression; lever direction correct, magnitude undersized (head/block_0 ratio essentially unchanged at 3.3×).** |
| **#3764** | **h=192+SwiGLU stacking (thorfinn)** | **79.22** | **Anti-additive; compute-starvation (12 ep vs 17 at h=128), schedule mismatch — not architectural antagonism.** |
| #3765 | SwiGLU seed confirm (fern) | μ̂=66.48 | 3-seed variance char. MAJOR: seed=0 was lucky low. Canonical μ̂=66.48±0.90. GeGLU population tie unresolved. |
| #3811 | Dropout 0.1 + SwiGLU (alphonse) | μ̂=66.82 | 2-seed null. SwiGLU gating already regularizes sufficiently. |
| #3840 | fc_gate LR boost 1.5× (frieren) | 67.00 | Modest regression. MAJOR finding: fc_main > fc_gate grad-norm ratio. |
| #3768 | Inverse-LLRD+SwiGLU (frieren) | 74.01 | Anti-additive. Grad-norm inverts between-block. |
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

1. **head_and_embed 3.0× LR boost** — if 2.5× wins or is undersized, push to equilibrium ratio.
2. **head_and_embed + block_4 dual boost** — block_4 is 2nd-largest grad-norm group (1.41), might benefit from concurrent boost.
3. **T_max scan extended** — if T_max=12 wins, scan T_max ∈ {10, 12, 14} for the sweet spot.
4. **LeakyReGLU** — if ReGLU dies due to dead-gate, LeakyReLU(0.01) rescue.
5. **LR scan extended (1.5e-3, 2e-3)** — if lr=1e-3 (tanjiro #3959) wins, find the stability edge.
6. **RMSNorm vs LayerNorm swap** — modern LLaMA-style. Param-matched, single-knob test of normalization geometry.
7. **AdamW betas scan** — (0.9, 0.95) vs default (0.9, 0.999); LLaMA-style β2.
8. **slice_num scan on SwiGLU h=128** — current slice_num=64 was inherited; test 32, 128 for attention granularity.
9. **Combine winners (Round 7)** — if multiple PRs win independently, test their stacks.
10. **SwiGLU + SWA over SwiGLU-converged checkpoint** — extends nezuko's mechanism finding if her #3644 SWA wins on SwiGLU.

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
10. **Per-channel weighting (β_p=20) on h=128 width** — #3611, #3837. Width-coupled; needs h=192-class capacity to absorb redistribution.
11. **h=192+SwiGLU stacking under current budget** — #3764. Compute-starved at 12 epochs; needs faster-converging h=192 setup before retesting.
12. **head_and_embed 1.75× LR boost** — #3832. Right direction, undersized lever; superseded by 2.5× #3932.
13. **Bilinear gate (no activation)** — #3855. Closes GLU ablation family: works mechanistically (94% of GLU gain) but does not improve on GeGLU/SwiGLU; reduced complexity does not translate to better minima.
14. **Per-projection LR asymmetry on SwiGLU** — #3840 (fc_gate boost) + #3888 (fc_main boost). Both directions regress; ratio actually flipped under fc_main boost (Adam did not normalize away the asymmetry), but val penalized either dynamics shift. Gate/main grad-mass asymmetry is a non-actionable invariant of healthy SwiGLU optimization.

## Plateau status

**Not in plateau.** Active investigation on 5 parallel fronts:
1. **GLU ablation family** (ReGLU #3933) — formally closing with the last activation choice (ReLU); Bilinear #3855 confirmed 94% of gain from gating mechanism alone.
2. **Gradient-informed LR scaling — between-block** (head_and_embed 2.5× #3932) — within-block per-projection axis exhausted (#3840 + #3888 both null in both directions).
3. **Schedule tuning** (T_max=12 #3934, SWA tail #3644 on SwiGLU regime) — aligning cosine to actual budget on h=128/gated-FFN.
4. **Regularization layered with gating** (DropPath #3886) — early signal shows +8 val regression at drop_prob=0.1; nudged for SENPAI-RESULT marker. May close soon.
5. **Base LR scaling** (lr=1e-3 #3959) — using SwiGLU's measured σ̂=0.90 stability headroom that 5e-4 inherits from GELU era.
6. **Normalization geometry** (RMSNorm #3973) — LLaMA-style architectural test of "does mean-centering matter for this model?"
