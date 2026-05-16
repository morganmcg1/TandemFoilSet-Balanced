# SENPAI Research State

- **Date:** 2026-05-16 10:25
- **Launch:** willow-pai2i-48h-r1 (round 6 — SwiGLU/GeGLU era; new all-time best 65.37)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/gated-FFN, ~13ep at h=192+gated-FFN)
- **Latest direction from human team:** None (no open issues scoped to this launch)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **h=128+GeGLU (PROGRAMME BEST)** | **65.3704** | **61.6819** | PR #3810, W&B `db8bp8i8`, seed=0 | `--use_geglu` + h=128/T_max=15 override |
| h=128+SwiGLU (former best) | 65.44 | 62.04 | PR #3680, W&B `8on2llcv`, seed=0 | GeGLU ≈ SwiGLU (statistical tie) |
| h=192+GELU (advisor default) | 86.81 | 81.35 | PR #3562 | Current train.py default |
| h=128+GELU μ̂ (canonical) | 90.77 ± 1.54 | 85.85 ± 0.67 | PR #3546 | Old 4-seed canonical |

**Effective win threshold:** val < 65.37 (or test < 61.68).

**Key mechanistic insight from PR #3810 (GeGLU):** The gating architecture — not SiLU specifically — drives the +25pt gain. GeGLU and SwiGLU are equivalent for CFD pressure prediction. The multiplicative `main × gate(x)` interaction is the lever. This opens the bilinear/ReGLU/gating-ablation research line.

**Key mechanistic insight from PR #3768 (frieren):** Under SwiGLU, the gradient profile inverts — head_and_embed (3.48) > block_4 (1.41) > block_0 (1.12). Middle blocks (1-3) are ~0.17 (10× lower). All GELU-era LR-stacking experiments are invalid under SwiGLU; the block_0-dominant assumption was falsified.

**Unresolved questions:**

1. **Seed reproducibility:** val=65.37 is single-seed — fern #3765 running SwiGLU seeds 1+2. GeGLU has not been seed-confirmed yet.
2. **h=192 stacking:** does gating compound with capacity? — thorfinn #3764 (SwiGLU+h=192).
3. **grad bottleneck targeting:** head_and_embed LR boost (askeladd #3832) and fc_gate LR boost (frieren #3840) directly motivated by frieren's inversion finding.

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

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3764 | thorfinn | **h=192+SwiGLU stacking** | Running |
| #3765 | fern | **SwiGLU h=128 seed confirm (seeds 1+2)** | Running |
| #3832 | askeladd | **head_and_embed LR boost 1.75x on SwiGLU** | Running (NEW — motivated by frieren's grad-norm inversion finding) |
| #3811 | alphonse | **Dropout 0.1 on SwiGLU (2-seed)** | Running |
| #3840 | frieren | **fc_gate LR boost 1.5x within SwiGLU blocks** | Running (NEW — surgical within-block LR targeting) |
| #3837 | edward | **β_p=20 + SwiGLU stacking at h=128** | Running (NEW — capacity-interaction sign-flip finding) |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA on SwiGLU | WIP (rebasing) |
| **#3855** | **tanjiro** | **Bilinear gate (no activation) — closes GLU ablation** | **NEW — assigned 09:39** |

## Recently closed PRs (this session)

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| **#3768** | **Inverse-LLRD+SwiGLU (frieren)** | **74.01** | **Anti-additive; MAJOR finding: grad-norm inverts under SwiGLU** |
| **#3611** | **β_p=20 on h=192 (edward)** | **86.59** | **Within noise; but capacity-interaction sign-flip finding** |
| **#3735** | **h=192 variance characterization (askeladd)** | stale | Lower-priority pivot; h=192+GELU obsolete vs SwiGLU floor |
| #3678 | Dropout 0.1 on h=128+GELU | μ̂=90.27 | Null result; follow-up #3811 on SwiGLU |
| #3724 | Corrected h-flip | 103.91 | Catastrophic regression; ground-effect physics |
| #3721 | DropPath rate=0.1 | 92.06 | Regression |
| #3722 | Inverse-LLRD γ_inv=1.176 | 88.03 | GELU-era; below SwiGLU floor |
| #3644 (GELU arm) | LLRD+GELU | 91.68 | Null vs GELU baseline |

## The SwiGLU/GeGLU era — key insights

1. **Gating is the dominant lever.** PR #3680 (SwiGLU) gave −25pt val vs GELU (90.77→65.44). PR #3810 (GeGLU) confirms the gating architecture — not SiLU — is what matters.

2. **Gating mechanism is ideally suited to CFD pressure.** Pressure fields have sharp spatial gradients (boundary layers, wake regions). The multiplicative gate selectively suppresses irrelevant tokens per-step. GELU cannot do this.

3. **Param parity is exact.** The 2/3 hidden-dim factor (171 vs 256) preserves n_params=663K. This is a pure architectural expressiveness gain, not capacity.

4. **Grad profile inverts under gating (frieren #3768 finding).** head_and_embed (3.48) > block_4 (1.41) > block_0 (1.12). Middle blocks at 0.17. GELU-era block_0-dominant assumption is falsified. LR scaling must be redesigned for gated-FFN regime.

5. **β_p=20 capacity-interaction sign flip (edward #3611).** The per-channel surf weighting reverses sign between h=128 (regression) and h=192 (improvement). Suggests capacity mediates how extra gradient mass is absorbed.

6. **All GELU-era experiments are sub-threshold.** Any result > 65.37 from GELU-regime work (h=192+GELU, dropout+GELU, LLRD+GELU) is irrelevant — the new floor is set by gated FFN.

## Immediate next priorities

1. **#3764 thorfinn h=192+SwiGLU** — determines if capacity and gating compound.
2. **#3765 fern seed confirm** — CRITICAL for seed reproducibility of the 65.44/65.37 era.
3. **#3832 askeladd head_and_embed LR boost** — direct follow-up to grad-norm inversion finding.
4. **#3840 frieren fc_gate LR boost** — within-block surgical LR targeting.
5. **#3811 alphonse dropout+SwiGLU** — regularization interaction with gating.
6. **#3855 tanjiro Bilinear gate** — closes the GLU ablation family. If bilinear ≈ SwiGLU, multiplicative interaction alone explains gain.

## Next research directions (queue for next idle students)

1. **Bilinear gate (no activation):** `fc_out(fc_main(x) * fc_gate(x))`. If this also lands at ~65, the multiplicative interaction alone is the lever. Strongest discriminator in the GLU ablation family.
2. **ReGLU (ReLU gate):** sharp/discontinuous gate. Resolves whether smooth gates specifically matter.
3. **GeGLU/SwiGLU at h=192** — does gating still compound with wider capacity? (Also covered by thorfinn #3764 with SwiGLU.)
4. **T_max scan on SwiGLU/GeGLU** — optimal cosine schedule may differ under gated-FFN convergence profile.
5. **β_p=20 + SwiGLU at h=128** — edward's #3837. Capacity-interaction sign flip hints this may work.
6. **SwiGLU + constant-tail SWA** — nezuko #3644 rebasing; directional SWA gain may persist in gated basin.
7. **Wider SwiGLU (h=256+SwiGLU)** — extend stacking if thorfinn confirms h=192+SwiGLU > h=128+SwiGLU.
8. **Per-layer gating** — selective gating in only some blocks (bottom 2-3 layers).
9. **Non-z-flip augmentations** (parked): Re-jitter, inlet velocity perturbations.

## Dead-end lever classes (do not revisit)

1. **Z-flip augmentation family (ALL VARIANTS)** — #3542, #3563, #3724. Ground-effect physics (no-slip wall at z=0) makes z-flip non-physical for raceCar. 3 independent failures.
2. **Dropout on GELU** — #3678, #3721. Null result. Follow-up dropout+SwiGLU (#3811) still live.
3. **SWA/EMA on frozen cosine tail at T_max<15** — #3644 GELU arm, #3580, #3521. T_cosine=10 undertrained.
4. **Standard LLRD** — #3642. Inverted gradient profile in Transolver.
5. **Inverse-LLRD on SwiGLU** — #3768. Falsified: block_0 is not the SwiGLU bottleneck.
6. **GELU-era LR-stacking experiments** — all invalid under SwiGLU (grad profile inverts).
7. **unified_pos=True** — #3566. Incompatible with 2D asymmetric flow.
8. **Per-channel Huber-δ** — #3574. Loss-formulation exhausted.
9. **Any h=128+GELU experiments** — val ceiling ~88 < new floor 65.37.
10. **β_p=20 on h=192+GELU** — #3611. Within noise + below SwiGLU floor; but β_p+SwiGLU still live (#3837).

## Plateau status

**Decidedly not in plateau.** The GeGLU result confirms the gating mechanism is the lever. Three active lines of investigation:
1. **GLU ablation family** (Bilinear, ReGLU) — pure gate-mechanism science.
2. **Gradient-informed LR scaling** (head_and_embed, fc_gate boosts) — frieren's inversion finding points directly at the bottleneck.
3. **Lever stacking** (capacity × gating, β_p × gating, SWA × gating) — any of these may compound.
