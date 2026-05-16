# SENPAI Research State

- **Date:** 2026-05-16 08:35
- **Launch:** willow-pai2i-48h-r1 (round 6 — SwiGLU era; new all-time best 65.44)
- **Advisor branch:** `icml-appendix-willow-pai2i-48h-r1`
- **Budget per run:** 30 min wall clock, 50 epochs max (~17ep at h=128/SwiGLU, ~13ep at h=192+SwiGLU)
- **Latest direction from human team:** None (no open issues as of 08:35)

## Research contract

Beat the Transolver baseline on `val_avg/mae_surf_p` (lower is better). Paper-facing metric: `test_avg/mae_surf_p`.

## Current best baselines

| Config | val_avg | test_avg | Source | Note |
|--------|---------|---------|--------|------|
| **h=128+SwiGLU (PROGRAMME BEST)** | **65.44** | **62.04** | PR #3680, W&B `8on2llcv`, seed=0 | Requires `--use_swiglu` + h=128/T_max=15 override |
| h=192+GELU (advisor default) | 86.81 | 81.35 | PR #3562 | Current train.py default |
| h=128+GELU μ̂ (canonical) | 90.77 ± 1.54 | 85.85 ± 0.67 | PR #3546 | Old 4-seed canonical |

**⚠️ Two unresolved questions before declaring a new canonical:**
1. **Seed reproducibility**: val=65.44 is single seed=0 — fern #3765 running seeds 1+2 to confirm
2. **h=192 stacking**: does SwiGLU compound with capacity? — thorfinn #3764

**Effective win threshold** (pending seed confirmation): val < 65.44 on the h=128+SwiGLU or h=192+SwiGLU config.

## Merged PRs (all)

| PR | Hypothesis | val_avg | test_avg |
|----|-----------|---------|---------|
| #3159 | Huber loss δ=0.1 | 112.90 | 115.76 |
| #3309 | NaN fix | 112.83 | 106.60 |
| #3317 | Cosine T_max=15 | 91.33 | 88.43 |
| #3480 | bf16 autocast | 87.91 | 83.38 |
| #3546 | Seed control + variance | μ̂=90.77, σ̂=1.54 | μ̂=85.85, σ̂=0.67 |
| #3562 | h=192/slice=96/T_max=18 | 86.81 | 81.35 |
| **#3680** | **SwiGLU activation** | **65.44** | **62.04** |

## Active WIP — 8/8 students (zero idle)

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3764 | thorfinn | **h=192+SwiGLU stacking** | Running (2 seeds) |
| #3765 | fern | **SwiGLU h=128 seed confirm (seeds 1+2)** | Running |
| #3768 | frieren | **Inverse-LLRD+SwiGLU stacking** | Running |
| #3735 | askeladd | h=192 4-seed σ̂ variance char | Running |
| **#3811** | **alphonse** | **Dropout 0.1 on h=128+SwiGLU (2-seed)** | **NEW — assigned 08:32 (follow-up to #3678)** |
| #3611 | edward | Per-channel β_p=20 (rebasing onto h=192) | WIP (rebasing) |
| #3644 | nezuko | Cosine T_max=10 + constant tail + SWA on SwiGLU | WIP (rebasing onto SwiGLU; committed to option (a) at 08:22) |
| **#3810** | **tanjiro** | **GeGLU activation (SiLU vs GELU gate ablation)** | **NEW — assigned 08:30** |

## Recently closed PRs

| PR | Hypothesis | val | Reason |
|----|-----------|-----|--------|
| **#3678** | **Dropout 0.1 on h=128+GELU (2-seed)** | **μ̂=90.27** | **2-seed mean within ±0.3σ of GELU μ̂=90.77 — null result. Follow-up: dropout+SwiGLU (PR #3811).** |
| **#3724** | **Corrected h-flip (z-flip + Uy/AoA/gap sign-flip, skip cruise)** | **103.91** | **Catastrophic regression; per-split diagnostic decisive. Ground-effect physics breaks z-symmetry — 3rd z-flip-class failure.** |
| #3721 | DropPath rate=0.1 | 92.06 | Regression on h=128 GELU; regularization not the bottleneck |
| #3722 | Inverse-LLRD γ_inv=1.176 | 88.03 | 1σ win on GELU but far above SwiGLU 65.44. Queued as #3768 on SwiGLU. |
| #3642 | LLRD γ=0.85 | 92.45 | Inverted gradient profile; block_0 needs highest LR |
| #3566 | unified_pos=True | 102.63 | Encoding mismatch in 2D asymmetric flow |
| #3574 | Per-channel Huber-δ | 91.78 | Loss-formulation lever exhausted |

## SwiGLU discovery — key implications

The SwiGLU result (val 90.77→65.44, −27.9%) reframes the entire programme:

1. **The FFN nonlinearity was the dominant unexplored lever.** All prior work (loss shape, LR schedule, weight averaging, regularization) operated within the old GELU regime. The activation class was untouched.

2. **Selective gating is ideally matched to CFD pressure.** Pressure fields have sharp spatial gradients (boundary layers, wake regions). SwiGLU's gate learns to suppress "irrelevant" tokens (far-field, wake) and amplify "relevant" ones (surface boundary layer) per-step. GELU cannot do this.

3. **Param parity is exact.** The 2/3 hidden-dim factor (171 vs 256) preserves n_params=663K vs 663K. This is not a capacity gain — it's purely an architectural expressiveness gain.

4. **All prior h=128+GELU experiments are now sub-threshold.** PRs that showed gains vs GELU μ̂=90.77 (inverse-LLRD 88.03, h=192 89.70) are below the new floor. Future experiments compare to 65.44.

## Immediate next priorities

1. **#3765 fern seed confirmation** — CRITICAL. Must confirm val=65.44 is reproducible before any follow-on work.
2. **#3764 thorfinn h=192+SwiGLU** — determines the canonical production config.
3. **#3768 frieren inverse-LLRD+SwiGLU** — lever stacking test.
4. **#3810 tanjiro GeGLU** — ablation: is gating-in-general or SiLU specifically the lever?
5. **#3811 alphonse dropout+SwiGLU** — does regularization compound with gating?

## Next research directions (queue for next idle students)

1. **Wider SwiGLU (h=256+SwiGLU)** — extend stacking test to even larger capacity (if thorfinn confirms h=192+SwiGLU > h=128+SwiGLU).
2. **SwiGLU + per-channel surf weight β_p=20** — stacking edward's #3611 lever with the new activation.
3. **SwiGLU + constant-tail SWA** — nezuko #3644 rebasing; directional SWA gain (0.35σ) may persist on SwiGLU's converged basin.
4. **ReGLU / Bilinear gate alternatives** — follow-up to GeGLU (#3810) if gating-mechanism theory confirmed.
5. **Per-layer SwiGLU** — selective gating in only some blocks (bottom 2-3 layers where geometry encoding is primary).
6. **T_max scan on SwiGLU** — optimal cosine schedule may differ under the new activation's convergence profile.
7. **Non-z-flip augmentations** (parked): x-translation, Re-jitter, inlet velocity perturbations — preserve ground-effect physics.

## Dead-end lever classes (do not revisit)

1. **Z-flip augmentation family (ALL VARIANTS)** — #3542, #3563, #3724. Ground-effect physics (no-slip wall at z=0) makes z-flip non-physical for raceCar. 3 independent failures.
2. **Dropout on GELU** — #3678, #3721. Regularization is not the bottleneck on h=128+GELU. Closed with null result.
3. **SWA/EMA on frozen cosine tail at T_max<15** — #3644 GELU arm, #3580, #3521. T_cosine=10 undertrained the model.
4. **LLRD standard** — #3642. Inverted gradient profile in Transolver.
5. **unified_pos=True** — #3566. Incompatible with 2D asymmetric flow.
6. **Per-channel Huber-δ** — #3574. Loss-formulation exhausted.
7. **Uniform surf_weight scan** — #3428, #3174, #3522.
8. **Any h=128+GELU experiments** — val ceiling ~88 < new floor 65.44.

## Plateau status

**Decidedly not in plateau.** SwiGLU represents the first major architectural insight in this programme — switching from GELU to gated linear units gave a 28% relative val improvement. The SwiGLU era is just beginning; the programme is now exploring what stacks with gating (capacity, LR scaling, regularization, gate activation form).
