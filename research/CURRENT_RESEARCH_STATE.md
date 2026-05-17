<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-17 (~08:30 UTC) — **#4320 thorfinn MERGED — NEW BEST val 49.75 / test 42.89** (T_max=22, first sub-50 val, camber_cruise fixed). #4372 thorfinn assigned (R12 H79: T_max={21,23} fine grid). 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 49.7515** (PR #4320 thorfinn — Lion lr=2e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + **T_max=22** + **grad_clip=1.0** + **layer_scale_init=1e-4**; NO spec_norm; run `1neonugr`)
**test_avg/mae_surf_p = 42.8929** (same run, clean 4-split)

Per-split val: in_dist 50.61, camber_rc 63.77, **camber_cruise 32.88**, re_rand 51.75
Per-split test: in_dist 44.05, camber_rc 57.45, **camber_cruise 27.56**, re_rand 42.51

**Δ vs prior best (PR #4201 T_max=20, val 53.08 / test 44.89): −3.32 val / −1.99 test**

Key mechanism: Lower cosine-endpoint LR at T_max=22 (~1.24e-4 within 14 epochs vs ~1.34e-4 at T_max=20). Improvement is broad-based: all 4 val splits improve, 3 of 4 test splits improve. **camber_cruise regression from PR #4201 (+2.5/+2.8) is FIXED**: cruise val 36.68→32.88 (−3.80), cruise test 30.65→27.56 (−3.09). T_max=22 is safe — T_max=24 diverges at ls substrate (Finding #33). T_max=16 regresses (+1.97 val, concentrated in camber_rc).

**Current weakness**: camber_rc test (57.45) remains the weakest split — ~15 higher than other splits. Next immediate probe: T_max=23 (cliff-edge probe, PR #4372).

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
| #4201 | ls=1e-4 + T_max=20 + lr=2e-4 + clip=1.0 | 53.81 → 53.08 | 45.49 → 44.89 | −1.4% / −1.3% |
| **#4320** | **T_max=22 at new BL (endpoint LR −3.32 val — BIGGEST R12 WIN)** | **53.08 → 49.75** | **44.89 → 42.89** | **−6.2% / −4.5%** |

**Total improvement:** val 135 → 49.75 (−63.1%), test ~130 → 42.89 (−67.0%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status | Substrate |
|----|---------|------------|--------|-----------|
| **#4372** | **thorfinn** | **R12 H79: T_max {21,23} fine grid around T_max=22 BL** | **Just assigned** | T_max=22 BL |
| #4315 | tanjiro | R12 H71: LR sweep {1.7e-4, 2.3e-4} | WIP | T_max=20 (old BL — directional finding) |
| #4318 | askeladd | R12 H72: ls magnitude {1e-3, 5e-5} | WIP | T_max=20 (directional finding) |
| #4319 | nezuko | R12 H73: WD sweep {5e-4, 2e-3} | WIP | T_max=20 (directional finding) |
| #4326 | alphonse | R12 H75: Huber β {0.03, 0.10} | WIP | T_max=20 (directional finding) |
| #4328 | frieren | R12 H76: EMA decay {0.995, 0.999} | WIP | T_max=20 (directional finding) |
| #4329 | edward | R12 H77: Lion β1 {0.85, 0.95} | WIP | T_max=20 (directional finding) |
| #4346 | fern | R12 H78: Multi-seed BL replication (seed=42, 2026) | WIP | T_max=20 (tighten PR #4201 σ) |

**Note on substrate shift**: #4315–#4346 test at T_max=20 and will be compared against the current BL (val 49.75, T_max=22). They are unlikely to beat the new BL but will yield directional axis findings — any winners at T_max=20 that clear the old BL (53.08) are valuable leads for R13 at T_max=22.

## Recent closures

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4255 | fern | LR {1.3, 1.7, 2.0} × 1e-4 at T_max=24+clip+no-ls: all regress +2-3 val vs old BL. **Finding #36**: lr=1.5e-4 is sharp local minimum at this substrate; ±0.2e-4 wings each cost ≥2 val. Finding #22 (clip→lr=2e-4) does NOT dominate when T_max=24 active. | CLOSED |
| #4256 | edward | Fine-grained clip {0.85, 1.15} at T_max=24+no-ls: both arms +2 val vs clip=1.0. **Finding #35**: clip=1.0 locally optimal; fine-grid asymmetry (Finding #25) does not replicate. Substrate superseded. | CLOSED |
| #4274 | frieren | EMA {0.995, 0.999} at T_max=24+clip+no-ls: both within σ of BL. **Finding #34**: EMA tightens around 0.997 at old substrate. Follow-up at new BL → #4328. | CLOSED |
| #4240 | alphonse | Triple composition (ls=1e-4 + T_max=24 + clip=1.0): 3/4 diverge (val>100). **Finding #33**: T_max=24 EXCLUDED when layer_scale present. Safe T_max ≤ 20 with ls. | CLOSED |
| #4258 | thorfinn | β1 sweep: both arms +5 val vs BL. **Finding #30**: β1=0.9 robust at old substrate. Lion-state axis closed at old BL; follow-up at new BL → #4329. | CLOSED |
| #4212 | askeladd | ls magnitude: ls=1e-3 −0.18 val within σ at old substrate; all arms regress vs new BL. **Finding #31**: non-monotone ls landscape. Follow-up at new BL → #4318. | CLOSED |
| #4231 | tanjiro | LR recalibration: lr=1.7e-4 directional winner at old substrate. Test regresses vs new BL. **Finding #32**: LR finding for follow-up at new BL → #4315. | CLOSED |
| #4214 | frieren | EMA@layer_scale+T_max=20: timeout-truncated. **Finding #28**: layer_scale stabilises ema=0.999 but slow. | CLOSED |

## Key findings (cumulative, 37)

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
36. **lr=1.5e-4 is a SHARP local minimum at T_max=24+clip+no-ls**: all wider LR arms ({1.3, 1.7, 2.0}×1e-4) regress ≥2.2 val. Finding #22 (clip shifts lr optimum to 2e-4 at T_max=14) does NOT generalize to T_max=24; elevated late-schedule LR at T_max=24 already provides effective high-LR. LR axis fully closed at old substrate.
37. **T_max=22 is the new optimum — first sub-50 val, camber_cruise regression fixed**: lower cosine-endpoint LR (~1.24e-4 at epoch 14) vs T_max=20 (~1.34e-4) gives the small-gradient cruise domain better convergence. Win is broad-based (all 4 val splits, 3 of 4 test). T_max=16 regresses (+1.97 val) confirming the mechanism: higher time-averaged LR (more T_max) is beneficial IF the endpoint LR is simultaneously lower. Safe range with ls: T_max ∈ [20, 22]; T_max=24 diverges (Finding #33). Follow-up: T_max=23 cliff-edge probe (#4372).

## R12 focus: T_max=22 established as new BL — fine-grid probe ongoing

**Immediate priority**: #4372 thorfinn T_max={21, 23} probe.
- T_max=23 is the cliff-edge test: does improvement continue toward T_max=24 divergence point?
- T_max=21 completes the 1-step grid (20→21→22→23) for a clean paper-facing comparison.

**R12 hyperparameter sweeps in flight at T_max=20**: All 6 axis sweeps (#4315–#4346 minus thorfinn) are testing at T_max=20 (the now-superseded substrate). They will yield directional findings but are unlikely to beat the new BL (49.75 at T_max=22). Plan:
- Let them complete — GPU time already committed
- Close as informative findings with directional notes for R13
- R13 will re-test the most promising axis winners at T_max=22

**T_max axis summary**:
- T_max=14: old start (BL chain)
- T_max=16: regresses (Finding in PR #4320 — more sustained high LR hurts camber_rc)  
- T_max=20: current BL chain intermediate (val 53.08)
- **T_max=22: CURRENT BEST (val 49.75, −3.32)**
- T_max=23: next probe (cliff edge)
- T_max=24: DIVERGES at ls substrate (Finding #33)

T_max=24 EXCLUDED from all future ls experiments.

Old BL substrate (T_max=24+clip, no-ls) now fully closed — all students reassigned to new BL substrate axes.

**Note on guard parser issue**: Advisor template comments containing `SENPAI-RESULT: {...}` (literal brace placeholder) trip the merge guard parser. Future advisor comments should use "SENPAI-RESULT JSON marker" or similar instead of the literal template.
