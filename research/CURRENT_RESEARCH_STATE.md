<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-17 (~07:30 UTC) — #4240 #4274 #4256 CLOSED (Findings #33-35); #4326 #4328 #4329 assigned (R12 H75-H77: Huber β, EMA, Lion β1 at new BL). 8/8 staffed; #4255 fern nudged to submit.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 53.0764** (PR #4201 nezuko — Lion lr=2e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + **T_max=20** + **grad_clip=1.0** + **layer_scale_init=1e-4**; NO spec_norm; run `d3qlknrv`)
**test_avg/mae_surf_p = 44.8874** (same run, clean 4-split)

Per-split val: in_dist 55.86, camber_rc 65.64, camber_cruise 36.68, re_rand 54.13
Per-split test: in_dist 46.83, camber_rc 57.22, camber_cruise 30.65, re_rand 44.85

**Δ vs prior best (PR #4145 T_max=24+clip=1.0, val 53.81 / test 45.49): −0.73 val / −0.60 test**

Key mechanism: Four-way composition. layer_scale=1e-4 + T_max=20 + lr=2e-4 + clip=1.0 compose additively. Win concentrated in camber_rc (−4.9 val / −4.9 test — the hardest OOD split). layer_scale + clip=1.0 together provide direction-sensitive step stabilization in high-gradient-norm geometry domains. camber_cruise regresses (+2.5/+2.8) — likely over-tight clip for cruise's gradient distribution. σ_val=1.55 across 3 seeds; 2/3 seeds beat BL; median (53.27/45.35) also beats BL.

**Weakness**: camber_cruise regression (+2.5 val / +2.8 test) and re_rand test (+0.92) need attention in R12.

## Merged sequence (improvement cascade)

| PR | Description | val → val | test → test | Δ |
|----|-------------|-----------|-------------|---|
| #3098 | Huber loss | 135.23 → 96.05 | — | −29.1% |
| #3296 | NaN guard | — → 90.00 | first clean test | — |
| #3444 | cosine T_max=14 | 96.05 → 93.20 | 90.00 → 83.54 | −3.0% / −7.2% |
| #3537 | Lion optimizer | 93.20 → 77.58 | 83.54 → 68.88 | −16.8% / −17.5% |
| #3405 | FiLM+Lion+EMA | 77.58 → 71.65 | 68.88 → 62.11 | −7.9% / −9.8% |
| #3672 | n_fourier=0 | 71.65 → 70.34 | 62.11 → 61.63 | −1.8% / −0.8% |
| #3748 | spec_norm(output) | 70.34 → 68.96 | 61.63 → 60.82 | −2.0% / −1.3% |
| #3843 | Lion lr=1e-4 | 68.96 → 65.41 | 60.82 → 56.06 | −5.2% / −7.8% |
| #3954 | spec_norm + lr=1e-4 | 65.41 → 64.68 | 56.06 → 56.17 | −1.1% / +0.2% |
| #3976 | Lion lr=1.5e-4 | 64.68 → 63.05 | 56.17 → 53.60 | −2.5% / −4.6% |
| #4056 | grad_clip=1.0 | 63.05 → 61.18 | 53.60 → 52.09 | −3.0% / −2.8% |
| #4063 | T_max=20 | 61.18 → 57.66 | 52.09 → 49.45 | −5.7% / −5.1% |
| #4120 | lr=2e-4 at clip=1.0 | 57.66 → 56.89 | 49.45 → 49.03 | −1.3% / −0.8% |
| #4015 | layer_scale_init=1e-4 + T_max=20 | 56.89 → 54.30 | 49.03 → 47.29 | −4.6% / −3.5% |
| #4145 | T_max=24 + grad_clip=1.0 | 54.30 → 53.81 | 47.29 → 45.49 | −0.9% / −3.8% |
| **#4201** | **ls=1e-4 + T_max=20 + lr=2e-4 + clip=1.0** | **53.81 → 53.08** | **45.49 → 44.89** | **−1.4% / −1.3%** |

**Total improvement:** val 135 → 53.08 (−60.7%), test ~130 → 44.89 (−65.5%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #4255 | fern | R11 H67: LR sweep at T_max=24+clip=1.0 {1.3e-4, 1.5e-4, 1.7e-4, 2.0e-4} | WIP — nudged to submit; all arms done |
| #4315 | tanjiro | R12 H71: LR sweep {1.7e-4, 2.3e-4} at new BL (ls+T_max=20+clip+lr=2e-4) | WIP |
| #4318 | askeladd | R12 H72: ls magnitude {1e-3, 5e-5} at new BL substrate | WIP |
| #4319 | nezuko | R12 H73: WD sweep {5e-4, 2e-3} at new BL substrate | WIP |
| #4320 | thorfinn | R12 H74: T_max {16, 22} at new BL substrate (T_max=24 excluded: diverges at ls) | WIP |
| **#4326** | **alphonse** | **R12 H75: Huber β {0.03, 0.10} at new BL substrate (targets camber_cruise & re_rand)** | **Just assigned** |
| **#4328** | **frieren** | **R12 H76: EMA decay {0.995, 0.999} at new BL substrate (ls unblocks ema=0.999?)** | **Just assigned** |
| **#4329** | **edward** | **R12 H77: Lion β1 {0.85, 0.95} at new BL substrate (extends Finding #30)** | **Just assigned** |

## Recent closures

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4256 | edward | Fine-grained clip {0.85, 1.15} at T_max=24+no-ls: both arms +2 val vs clip=1.0. **Finding #35**: clip=1.0 locally optimal; fine-grid asymmetry (Finding #25) does not replicate. Substrate superseded. | CLOSED |
| #4274 | frieren | EMA {0.995, 0.999} at T_max=24+clip+no-ls: both within σ of BL. **Finding #34**: EMA tightens around 0.997 at old substrate. Follow-up at new BL → #4328. | CLOSED |
| #4240 | alphonse | Triple composition (ls=1e-4 + T_max=24 + clip=1.0): 3/4 diverge (val>100). **Finding #33**: T_max=24 EXCLUDED when layer_scale present. Safe T_max ≤ 20 with ls. | CLOSED |
| #4258 | thorfinn | β1 sweep: both arms +5 val vs BL. **Finding #30**: β1=0.9 robust at old substrate. Lion-state axis closed at old BL; follow-up at new BL → #4329. | CLOSED |
| #4212 | askeladd | ls magnitude: ls=1e-3 −0.18 val within σ at old substrate; all arms regress vs new BL. **Finding #31**: non-monotone ls landscape. Follow-up at new BL → #4318. | CLOSED |
| #4231 | tanjiro | LR recalibration: lr=1.7e-4 directional winner at old substrate. Test regresses vs new BL. **Finding #32**: LR finding for follow-up at new BL → #4315. | CLOSED |
| #4214 | frieren | EMA@layer_scale+T_max=20: timeout-truncated. **Finding #28**: layer_scale stabilises ema=0.999 but slow. | CLOSED |

## Key findings (cumulative, 35)

1. **FiLM on log(Re)** contributes −4.35 val / −4.56 test under n_fourier=0.
2. **EMA(0.997)** contributes +4.4 val on top of Lion.
3. **Fourier PE inert** under FiLM+Lion+EMA. n_fourier=0 wins.
4. **EMA decay robust** in [0.995, 0.997].
5. **Sobolev surface regularization**: catastrophically destabilizes at w=0.05+.
6. **LLRD doesn't transfer** from fine-tuning to scratch.
7. **LR warmup adds nothing** to Lion.
8. **Block-FiLM regresses.** Output-only FiLM is correct topology.
9. **Lookahead dead end.**
10. **TTA z-reflection fails** — AoA asymmetry.
11. **Huber β=0.05 locally optimal** in [0.05, 0.20].
12. **Seed noise floor ≈ 2.77 val** (two identical runs). *Updated: 3-seed σ=1.55 at new BL.*
13. **Output-only spectral norm**: −1.39 val at lr=5e-5; contribution diminishes with lr.
14. **Lion lr=1.5e-4 optimal WITHOUT clip**: monotone trend, inflection at 2e-4. **UPDATED by finding #22.**
15. **Train-time z-aug fails** — AoA asymmetry.
16. **surf_weight optimum substrate-dependent**.
17. **Lion β1=0.9 optimal** in {0.8, 0.9, 0.95}. **Extended to new substrate by Finding #30.**
18. **spec_norm Lipschitz contribution diminishes as lr grows**: closed at lr=1.5e-4.
19. **Input MixUp catastrophic** on CFD pressure fields (+23-28 val).
20. **T_max=20 optimal** within 30-min budget at old substrate.
21. **Multi-FiLM global conditioning FALSIFIED**: Global γ/β hurts camber_rc OOD.
22. **LR optimum shifts from 1.5e-4 to 2e-4 at clip=1.0**: clip changes step direction; interaction is direction-sensitive.
23. **layer_scale_init=1e-4 (CaiT/DeiT-III) composes ~80% additively with T_max=20**.
24. **(clip × T_max) interaction super-additive at T_max=24**: clip essential at T_max≥24; late-LR endpoint ~1.35e-4 amplifies gradient outliers.
25. **clip=1.0 valley is asymmetric (lr=2e-4+T_max=14)**: clip=0.7 → +2.46, clip=1.4 → +5.35. Two co-located optima (scale AND direction-stability).
26. **lr response monotone at T_max=20+clip=1.0 in [1.5, 2.0]**: no interior minimum.
27. **T_max scan non-monotone at lr=2e-4+clip=1.0**: T_max=18 worst (bimodal terminal-LR at 14-epoch wall-clock).
28. **layer_scale=1e-4 stabilises ema=0.999 (no divergence) but slow-converging**: vs no-layer_scale T_max=20 where ema=0.999 diverged.
29. **Four-way composition (ls=1e-4 + T_max=20 + lr=2e-4 + clip=1.0) beats BL on both metrics**: win concentrated in camber_rc (−4.9 val / −4.9 test). σ_val=1.55 across 3 seeds; 2/3 beat BL; median also beats BL. camber_cruise regresses (+2.5/+2.8). New best val 53.08 / test 44.89.
30. **β1=0.9 robust at new substrate (T_max=24+clip=1.0)**: both arms +5 val. Rate-of-momentum-warmup dominates in 13-epoch budget. Lion optimizer-state axis closed.
31. **ls=1e-3 marginal winner at old T_max=20 substrate**: −0.18 val / −0.57 test within σ. Non-monotone landscape on log(layer_scale): 3e-4 is worst. Against new BL, all arms regress.
32. **lr=1.7e-4 directional winner at layer_scale+T_max=20+no-clip substrate**: −1.55 val / −1.22 test vs old BL. Cross-substrate: val beats new BL (52.75 < 53.08) but test regresses (+1.18) due to camber_cruise/re_rand divergence.
33. **layer_scale + T_max=24 + clip=1.0 is UNSTABLE**: 3/4 runs diverge (val>100). T_max=24 EXCLUDED from all future layer_scale experiments. Safe range with layer_scale: T_max ≤ 20.
34. **EMA tightens around 0.997 at T_max=24+clip=1.0+no-layer_scale**: both 0.995 and 0.999 within σ of BL. EMA axis closed at old substrate. Distinct from Finding #28 (layer_scale enables ema=0.999, but slow).
35. **Fine-grained clip (0.85, 1.15) at T_max=24+no-ls both regress ~2 val vs clip=1.0**: the asymmetric regression from Finding #25 (clip 0.5 vs 2.0 favored 2.0) does NOT replicate at the fine grid. clip=1.0 is locally optimal at {0.85, 1.0, 1.15}. Clip axis at old substrate closed.

## R12 focus: closing all hyperparameter axes at the new BL substrate

The new BL (PR #4201, val 53.08 / test 44.89) uses: ls=1e-4 + T_max=20 + lr=2e-4 + clip=1.0 + EMA(0.997) + Huber β=0.05 + FiLM + wd=1e-3 + Lion β1=0.9.

**Main weakness**: camber_cruise regression (+2.5 val / +2.8 test), re_rand test (+0.92). R12 systematically closes every hyperparameter axis at this substrate:

| Axis | PR | Student | Values tested |
|------|----|---------|---------------|
| LR | #4315 | tanjiro | {1.7e-4, 2.3e-4} vs 2e-4 |
| ls magnitude | #4318 | askeladd | {1e-3, 5e-5} vs 1e-4 |
| WD | #4319 | nezuko | {5e-4, 2e-3} vs 1e-3 (FIRST at new BL) |
| T_max | #4320 | thorfinn | {16, 22} vs 20 (T_max=24 EXCLUDED) |
| Huber β | #4326 | alphonse | {0.03, 0.10} vs 0.05 (FIRST at new BL) |
| EMA decay | #4328 | frieren | {0.995, 0.999} vs 0.997 (FIRST at new BL with ls) |
| Lion β1 | #4329 | edward | {0.85, 0.95} vs 0.9 default (extends Finding #30) |

T_max=24 EXCLUDED from all future ls experiments (Finding #33 — diverges at ls substrate).

Still in-flight from old BL substrate (T_max=24+clip, no-ls): #4255 fern (LR sweep — all arms done, nudged to submit).

**Note on guard parser issue**: Advisor template comments containing `SENPAI-RESULT: {...}` (literal brace placeholder) trip the merge guard parser. Future advisor comments should use "SENPAI-RESULT JSON marker" or similar instead of the literal template.
