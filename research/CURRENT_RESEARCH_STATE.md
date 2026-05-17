<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-17 (~10:50 UTC) — #4505 MERGED (Finding #56: spec_norm output pi=3 is NEW BL val 46.80/test 40.49 — biggest single win this round). #4503 CLOSED (Finding #55: Huber β axis fully closed). edward→#4560 (R13 H98 spec_norm pi=5/10), alphonse→#4557 (R13 H97 grad_clip). All 8 students staffed. ~3h left.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 46.7952** (PR #4505 edward — Lion lr=2e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=22 + grad_clip=1.0 + layer_scale_init=1e-4 + **spec_norm_target=output** + **spec_norm_n_power_iter=3**; run `b4txs5yb`)
**test_avg/mae_surf_p = 40.4866** (same run, clean 4-split)

Per-split val: in_dist 49.41, camber_rc 60.76, **camber_cruise 28.72**, re_rand 48.29
Per-split test: in_dist 44.45, camber_rc 54.86, **camber_cruise 24.21**, re_rand 38.43

**Δ vs prior best (PR #4320 T_max=22 no spec_norm, val 49.75 / test 42.89): −2.96 val / −2.41 test**

Key mechanism: spec_norm output (Lipschitz constraint on final layer) cooperates with layer_scale_init=1e-4. ls=1e-4 keeps residual block contributions tiny early in training → output layer carries proportionally more signal → Lipschitz constraint has higher leverage. pi=3 (tighter Lipschitz bound) better than pi=1 at this high-mean-lr substrate. No overhead (~142s/epoch). Finding #13 (diminishing returns with lr) does NOT extend to T_max=22+ls=1e-4 substrate.

**Current weakness**: camber_rc test (54.86) remains the weakest split — ~10 higher than camber_cruise. spec_norm pi=5/10 in flight (#4560) — testing if more power iterations continue to help.

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
| #4320 | T_max=22 at new BL (endpoint LR −3.32 val) | 53.08 → 49.75 | 44.89 → 42.89 | −6.2% / −4.5% |
| **#4505** | **spec_norm output pi=3 at T_max=22 (ls×spec_norm synergy — BIGGEST R13 WIN so far)** | **49.75 → 46.80** | **42.89 → 40.49** | **−6.0% / −5.6%** |

**Total improvement:** val 135 → 46.80 (−65.3%), test ~130 → 40.49 (−68.9%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status | Substrate |
|----|---------|------------|--------|-----------|
| **#4560** | **edward** | **R13 H98: spec_norm output {pi=5,pi=10} at T_max=22** | **Just assigned** | T_max=22 NEW BL — power-iter axis |
| **#4557** | **alphonse** | **R13 H97: grad_clip {0.7,1.5} at T_max=22** | **Just assigned** | T_max=22 — clip untested axis |
| **#4531** | **thorfinn** | **R13 H95: T_max×lr basin pairing {T_max=24,T_max=26}×lr=2.5e-4** | In flight | T_max=22 — basin pairing |
| **#4533** | **nezuko** | **R13 H96: surf_weight downward {7.5, 5} at T_max=22** | In flight | T_max=22 — surf_weight downward |
| **#4507** | **fern** | **R13 H93: spec_norm INPUT {pi=1, pi=3} at T_max=22** | In flight | T_max=22 — spec_norm input |
| **#4508** | **frieren** | **R13 H94: n_fourier=8 {σ=5, σ=10} at T_max=22** | In flight | T_max=22 — n_fourier axis |
| **#4495** | **askeladd** | **R13 H90: ls upward {2e-4, 3e-4} at T_max=22** | In flight | T_max=22 — ls upward |
| **#4470** | **tanjiro** | **R13 H88: batch_size {2, 8} at T_max=22** | In flight | T_max=22 — batch-size |

**All 8 students at T_max=22 substrate.** R13 in full swing. **NEW BL = 46.7952 val / 40.4866 test (spec_norm pi=3).** All in-flight students should compare against OLD BL (49.75/42.89) since they were assigned before the merge.

**σ_T20 established** (PR #4346 fern, Finding #45): 5-seed σ_val = 1.70, σ_test = 1.40. Merge thresholds: <1 val = noise, 1-2 val = weak signal, ≥2 val = robust. camber_rc has σ ~3.4 (extra caution for rc-only improvements).

**σ_T22 established** (PR #4437 fern, Finding #51): σ_val ≈ 1.86, σ_test ≈ 1.27. BL `1neonugr` at z=-1.27σ (seed-lucky). True mean ≈ 52.11. Merge threshold: <2 val gain = noise, ≥3 val = robust.

**STRUCTURAL META-FINDING (Findings #46-50)**: All 4 attempts to transfer T_max=20 directional wins to T_max=22 have failed with inversions: lr +1.91 (#4391), ls-smaller +3.10 (#4419), Huber β-up +2.69 (#4434), wd-up +3.63 (#4436). T_max=22 is NOT a linear extension of T_max=20 — the substrate change absorbs the regularization/optimization budget that prior wins were adding. **R13 must search T_max=22 as a fresh problem.**

**Active fresh axes at T_max=22**:
- **Huber β downward** {0.03, 0.04}: in flight (#4503 alphonse) — student's own hint "β=0.05 closer to L1 better matched to T_max=22"
- **spec_norm output** at T_max=22: in flight (#4505 edward) — closing Finding #13 at new substrate
- **lr axis** {1.5e-4, 1.7e-4}: in flight (#4471 thorfinn) — downward probe, completes 5-point grid
- **ls upward** {2e-4, 3e-4}: in flight (#4495 askeladd) — faster residual ramp under truncation
- **batch_size** {2, 8}: in flight (#4470 tanjiro) — never swept in this launch
- **surf_weight** {15, 20}: in flight (#4420 nezuko)
- **Lion β2** {0.95, 0.995}: in flight (#4408 frieren, GPU 100%)
- **σ_T22 calibration**: in flight (#4437 fern — CRITICAL, needed for merge threshold)

## Recent closures

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4505 | edward | spec_norm output {pi=1, pi=3}: pi=1 −1.71 val, pi=3 −2.96 val (NEW BL). **Finding #56**: spec_norm wins at T_max=22 — ls×spec_norm positive synergy. pi=5/10 in flight (#4560). | **MERGED** |
| #4503 | alphonse | Huber β {0.04, 0.03}: +4.63/+5.57 val. **Finding #55**: β axis fully closed — β=0.05 local minimum, asymmetric (L1 side 2× worse). grad_clip axis in flight (#4557). | CLOSED |
| #4471 | thorfinn | lr downward {1.7e-4, 1.5e-4}: +4.48/+7.07 val. **Finding #53**: 5-point lr×T_max=22 grid closed — asymmetric U-shape, lr=2e-4 minimum. Basin in (peak_LR, LR_end) plane. Basin pairing test in flight (#4531). | CLOSED |
| #4420 | nezuko | surf_weight {15, 20}: +3.76/+5.76 val. **Finding #54**: upward direction closed — strictly worse on both heads. camber_rc representation-limited. Downward probe {7.5, 5} in flight (#4533). | CLOSED |
| #4437 | fern | 3-seed σ-calib at T_max=22: seeds 42/2026/default → mean 52.11, σ_val=1.86. **Finding #51**: BL seed-lucky (z=-1.27σ). Merge threshold: ≥3 val above BL = robust. | CLOSED |
| #4408 | frieren | Lion β2 {0.95, 0.995}: β2=0.95 +11.23, β2=0.995 +6.46. **Finding #52**: β2 axis closed — β2=0.99 optimal; Lion optimizer-state fully pinned. | CLOSED |
| #4434 | alphonse | Huber β {0.10, 0.15} at T_max=22: +2.69/+2.21 val. **Finding #49**: Huber β upward fails — cruise inverts. Downward probe in flight (#4503). | CLOSED |
| #4436 | edward | wd {2e-3, 3e-3} at T_max=22: +3.63/+2.27 val. **Finding #50**: wd upward fails — cruise inverts. Spec_norm orthogonal in flight (#4505). | CLOSED |
| #4419 | askeladd | ls {5e-5, 1e-5} at T_max=22: +3.10/+4.48 val. **Finding #48**: ls smaller direction inverts at T_max=22 — faster residual ramp needed under truncation. Upward probe in flight (#4495). | CLOSED |
| #4391 | tanjiro | lr {2.3e-4, 2.5e-4} at T_max=22: both regress (+1.91/+0.12 val). **Finding #46**: lr composition inverts at T_max=22 — T_max=20 cruise gain does not transfer. lr upward direction closed. | CLOSED |
| #4372 | thorfinn | T_max {21, 23} at BL: T_max=23 +3.92 val, T_max=21 +5.77 val. **Finding #47**: T_max=22 is sharp resonance optimum under 13-epoch timeout (LR 7.18e-5 @ epoch 13 is the sweet spot). T_max axis closed. | CLOSED |
| #4346 | fern | 5-seed replication at T_max=20: σ_val=1.70, σ_test=1.40, σ_camber_rc=3.43. **Finding #45**: T_max=20 noise floor established — <1 val gain = noise, ≥2 = robust. T_max=22 σ in flight (#4437). | CLOSED |
| #4329 | edward | Lion β1 {0.85, 0.95} at T_max=20+clip: β1=0.85 unstable (2/4 diverge), β1=0.95 +3.83 vs new BL. **Finding #44**: β1=0.9 robustly optimal, axis closed at all substrates. | CLOSED |
| #4326 | alphonse | Huber β {0.03, 0.10} at T_max=20+clip: β=0.10 wins −1.15 val with **camber_cruise −2.96 val**. **Finding #43**: β=0.10 directional cruise signal; R13 follow-up at T_max=22 (#4434). | CLOSED |
| #4319 | nezuko | WD {5e-4, 2e-3} at T_max=20+clip: both regress vs old BL. wd=2e-3 per-split signal (cruise/re_rand win, rc/in_dist lose). **Finding #42**: wd=1e-3 robust; asymmetric trade for R13 follow-up. | CLOSED |
| #4318 | askeladd | ls {5e-5, 1e-3} at T_max=20+clip: ls=5e-5 wins −1.04 val vs old BL but camber_rc regresses +1.96. **Finding #41**: non-monotone ls landscape; optimum shifting toward smaller ls with higher clip+LR. | CLOSED |
| #4328 | frieren | EMA {0.995, 0.999} at T_max=20+ls+clip: ema=0.999 +6.46 val (worst on all splits — budget-bottlenecked). **Finding #38**: EMA axis closed at new BL substrate. ema=0.997 robust across all 3 tested substrates. | CLOSED |
| #4315 | tanjiro | LR {1.7, 2.3} × 1e-4 at T_max=20+ls+clip: lr=2.3e-4 directional winner (val −0.41 / test −0.48 vs old BL). **Finding #39**: lr=2.3e-4 best at T_max=20 substrate; camber_cruise −2.45 val. Lead → #4391 (lr×T_max=22). | CLOSED |
| #4255 | fern | LR {1.3, 1.7, 2.0} × 1e-4 at T_max=24+clip+no-ls: all regress +2-3 val vs old BL. **Finding #36**: lr=1.5e-4 is sharp local minimum at this substrate; ±0.2e-4 wings each cost ≥2 val. Finding #22 (clip→lr=2e-4) does NOT dominate when T_max=24 active. | CLOSED |
| #4256 | edward | Fine-grained clip {0.85, 1.15} at T_max=24+no-ls: both arms +2 val vs clip=1.0. **Finding #35**: clip=1.0 locally optimal; fine-grid asymmetry (Finding #25) does not replicate. Substrate superseded. | CLOSED |
| #4274 | frieren | EMA {0.995, 0.999} at T_max=24+clip+no-ls: both within σ of BL. **Finding #34**: EMA tightens around 0.997 at old substrate. Follow-up at new BL → #4328. | CLOSED |
| #4240 | alphonse | Triple composition (ls=1e-4 + T_max=24 + clip=1.0): 3/4 diverge (val>100). **Finding #33**: T_max=24 EXCLUDED when layer_scale present. Safe T_max ≤ 20 with ls. | CLOSED |
| #4258 | thorfinn | β1 sweep: both arms +5 val vs BL. **Finding #30**: β1=0.9 robust at old substrate. Lion-state axis closed at old BL; follow-up at new BL → #4329. | CLOSED |
| #4212 | askeladd | ls magnitude: ls=1e-3 −0.18 val within σ at old substrate; all arms regress vs new BL. **Finding #31**: non-monotone ls landscape. Follow-up at new BL → #4318. | CLOSED |
| #4231 | tanjiro | LR recalibration: lr=1.7e-4 directional winner at old substrate. Test regresses vs new BL. **Finding #32**: LR finding for follow-up at new BL → #4315. | CLOSED |
| #4214 | frieren | EMA@layer_scale+T_max=20: timeout-truncated. **Finding #28**: layer_scale stabilises ema=0.999 but slow. | CLOSED |

## Key findings (cumulative, 56)

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
38. **EMA axis closed at new BL substrate** (T_max=20+ls+clip+lr=2e-4): ema=0.995 regresses +1.48 val, ema=0.999 catastrophically bad +6.46 val (worst on ALL 4 splits). Root cause: 13-epoch wall-clock budget means ema=0.999's long averaging window is dominated by early-training weights — starvation-bottlenecked, not stability-bottlenecked. Layer_scale does NOT allow ema=0.999 to compete within this budget. EMA axis robustly closed across all 3 substrates (Findings #28+#34+#38): ema=0.997 is the optimum everywhere.
56. **spec_norm output pi=3 wins at T_max=22 — NEW BEST val 46.7952 / test 40.4866** (Finding #56, edward #4505): pi=1 −1.71 val, pi=3 −2.96 val vs old BL. Finding #13 (diminishing returns with lr) does NOT extend to T_max=22+ls=1e-4 — the ls×spec_norm synergy is a positive interaction. Tighter Lipschitz (pi=3) better than loose (pi=1) at high-mean-lr substrate. No overhead (~142s/epoch). Win is generalization-oriented (camber_rc, cruise, re_rand improve most; in_dist marginal). Power-iteration axis not yet closed — pi=5/10 in flight (#4560).
55. **Huber β axis fully closed at T_max=22 — β=0.05 local minimum, asymmetric landscape** (Finding #55, alphonse #4503): β=0.03(+5.57), β=0.04(+4.63), β=0.05(BL), β=0.10(+2.69), β=0.15(+2.21). L1 side hurts ~2× more than L2 side. Mechanism: Lion sign-update + L1-corner = flat-magnitude gradient on small residuals → slows late convergence. Per-split: in_dist regressed most (+7-9 val) under narrowing — highest dynamic range, most sensitive to β=0.05 smoothing. β axis definitively closed.
54. **surf_weight upward direction closed at T_max=22** (Finding #54, nezuko #4420): w=15 +3.76 val, w=20 +5.76 val — monotonically worse on BOTH heads (not a vol→surf trade). camber_rc hit hardest (representation-limited, not gradient-balance-limited). Volumetric loss is useful auxiliary signal for shared encoder. Optimum likely below w=10 at this substrate. Downward probe in flight (#4533).
53. **lr × T_max=22 grid fully closed — asymmetric U-shape** (Finding #53, thorfinn #4471): 5-point grid complete. lr=2e-4 is the operating point. Downward regressions far steeper than upward (+4.48, +7.07 at -15/-25% vs +1.91, +0.12 at +15/+25%). Refined mechanism: **basin in (peak_LR, LR_end) plane** — peak LR controls representation learning, LR_end ≥ ~6e-5 controls final fine-tuning. lr=2.5e-4 within σ_T22. Basin pairing test (T_max=24+lr=2.5e-4) in flight (#4531).
52. **Lion β2 axis closed at T_max=22** (Finding #52, frieren #4408): β2=0.95 catastrophic (+11.23 val), β2=0.995 regresses (+6.46 val). Combined with Finding #44 (β1 axis), **Lion optimizer-state fully pinned**: β1=0.9, β2=0.99 optimal across all substrates. n_fourier axis test in flight (#4508).
51. **σ_T22 ≈ 1.86 val / 1.27 test** (Finding #51, fern #4437): 3-seed mean val 52.11, test 44.53. BL `1neonugr` (49.75) at z=-1.27σ from mean — seed-lucky. Merge-threshold update: <2 val gain = noise, ≥3 val = robust. Slightly noisier than σ_T20=1.70 (Finding #45).
50. **wd upward composition fails at T_max=22** (Finding #50, edward #4436): wd=2e-3 +3.63 val, wd=3e-3 +2.27 val. cruise inverts (+4.09/+2.79 — opposite of Finding #42). T_max=22 already absorbed the regularization budget that wd=2e-3 was providing. 4th consecutive meta-finding: T_max=20 directional wins don't compose at T_max=22. Spec_norm orthogonal axis in flight (#4505).
49. **Huber β upward composition fails at T_max=22** (Finding #49, alphonse #4434): β=0.10 +2.69 val, β=0.15 +2.21 val. cruise inverts (+2.94/+2.59 — opposite of Finding #43). Mechanism: T_max=22 higher mean lr × wider Huber L2 region = over-emphasis on small residuals. β=0.05 correctly matches T_max=22. Student hint: β smaller direction may work → downward probe in flight (#4503).
48. **ls smaller-direction inverts at T_max=22** (Finding #48, askeladd #4419): ls=5e-5 +3.10 val, ls=1e-5 +4.48 val. Arm-vs-arm: ls=1e-4 > ls=5e-5 > ls=1e-5. camber_rc regression worsens monotonically with smaller ls (+3.06, +5.09 val). Root cause: smaller ls slows residual ramp; under T_max=22's 13-epoch truncation, residual capacity comes online too slowly. Opposite direction (ls={2e-4, 3e-4}) now in flight (#4495) — 'faster residual ramp' mechanism.
47. **T_max=22 is sharp resonance optimum under 13-epoch timeout** (Finding #47, thorfinn #4372): ±1 step costs ≥3.9 val. Root cause: cosine LR schedule delivers LR=7.18e-5 @ epoch 13 at T_max=22 — T_max=23 overshoots (7.97e-5), T_max=21 undershoots (6.35e-5). T_max axis fully closed. Follow-up: lr downward probe (#4471) tests whether the resonance point shifts.
46. **lr composition inverts at T_max=22** (Finding #46, tanjiro #4391): lr=2.3e-4 val 51.66 (+1.91 regression), lr=2.5e-4 val 49.87 (+0.12). The T_max=20 directional cruise win (Finding #39) does not transfer at T_max=22. LR upward direction exhausted at the new BL substrate. Downward lr probe in flight (#4471).
45. **σ_val ≈ 1.7 / σ_test ≈ 1.4 at T_max=20 substrate** (Finding #45, fern #4346 5-seed): per-split σ highest on camber_rc (3.4 val). Merge thresholds: <1 val = noise, ≥2 val = robust. T_max=22 σ TBD (fern #4437 in flight).
44. **Lion β1 axis closed at T_max=20+clip substrate** (Finding #44, edward #4329): β1=0.85 diverges/crashes (2/4 launches) — too-fast momentum at clip=1.0+ls+lr=2e-4. β1=0.95 marginal (+0.50 val vs old BL, +3.83 vs new). β1=0.9 robustly optimal. Axis closed across all substrates (Findings #30+#44).
43. **Huber β=0.10 wins at T_max=20 substrate** (Finding #43, alphonse #4326): beats old BL by −1.15 val / −0.45 test. **camber_cruise −2.96 val / −2.29 test** — largest cruise gain across all R12 sweeps. in_dist −1.42 val, re_rand −0.52 val (improvements). Only camber_rc hurt (+0.32 val test +1.40). Does NOT beat new BL (+2.18 val) as run at T_max=20. Strong R13 candidate → #4434.
42. **wd=1e-3 robustly optimal at T_max=20+clip substrate** (Finding #42): wd=5e-4 regresses +0.75 val; wd=2e-3 regresses +0.44 val. Key per-split signal: wd=2e-3 trades **camber_cruise/re_rand wins (−3.73/−0.86 val) for camber_rc/in_dist losses (+5.24/+1.11 val)**. High wd penalizes sharp-curve feature learning (camber_rc) but helps generalization on smooth domains (cruise/re_rand). Lead for R13: wd=2e-3 at T_max=22 may unlock cruise gain without the rc regression seen here.
41. **ls=5e-5 directional winner at T_max=20+clip+lr=2e-4 substrate** (Finding #41): beats ls=1e-4 BL by −1.04 val / −0.17 test. Non-monotone ls landscape: optimum has shifted monotonically toward smaller ls as clip+LR increased across substrates. Critical: camber_rc regresses (+1.96 val/test) for both ls={5e-5, 1e-3} arms — ls=1e-4 is uniquely camber_rc-best. Does NOT beat new BL (+2.29 val). Lead for R13: ls=5e-5 transfer test at T_max=22.
40. **Effective warmup_epochs=0 throughout programme** (Finding #40): train.py:799 constructs plain `CosineAnnealingLR` — no LambdaLR/SequentialLR wrapper, no warmup flag. Every BL run has used zero linear warmup; LR starts at peak on epoch 0. Zero warmup is stable at the new BL substrate (lr=2e-4 + clip=1.0 + ls=1e-4). If implemented, warmup would be a novel untested positive axis, not a correction of broken status quo.
39. **lr=2.3e-4 directional winner at T_max=20+ls+clip substrate**: val −0.41 / test −0.48 vs lr=2e-4 BL (old BL, doesn't reach new BL). Key per-split finding: camber_cruise improves strongly (−2.45 val / −2.11 test) while camber_rc regresses (+1.83 val / +1.95 test). The endpoint LR mechanism is consistent with Finding #37: higher lr at T_max=20 → slightly lower noise at convergence for the small-gradient regime. Lead for R13: lr=2.3e-4 at T_max=22 may compose additively (#4391 tanjiro).

## R13 focus: Sweeping T_max=22 substrate hyperparameter axes

**T_max axis CLOSED** (Finding #47). T_max=22 is the sharp resonance optimum — no further T_max sweeps.

**Closed axes at T_max=22** (Findings #46-56):
- lr (full 5-point grid, Finding #53) — CLOSED, basin pairing in flight (#4531)
- T_max around 22 (Finding #47) — CLOSED
- Lion β1, β2 (Findings #44, #52) — CLOSED
- Huber β BOTH directions (Findings #49, #55) — **FULLY CLOSED** — β=0.05 is local min
- wd upward (Finding #50) — CLOSED
- ls smaller (Finding #48) — CLOSED, upward in flight (#4495)
- surf_weight upward (Finding #54) — CLOSED, downward in flight (#4533)
- σ_T22 calibration (Finding #51) — CLOSED
- spec_norm output pi=1,pi=3 (Finding #56) — pi=3 MERGED (new BL); pi=5/10 in flight (#4560)

**Open axes at T_max=22** (in flight):
- **spec_norm output pi higher**: {pi=5, pi=10} — in flight (#4560 edward)
- **grad_clip**: {0.7, 1.5} — in flight (#4557 alphonse) — untested at new substrate
- **batch_size**: {2, 8} — in flight (#4470 tanjiro)
- **ls upward**: {2e-4, 3e-4} — in flight (#4495 askeladd)
- **spec_norm input**: {pi=1, pi=3} — in flight (#4507 fern) [pi=1 slightly worse than BL]
- **n_fourier=8**: {σ=5, σ=10} — in flight (#4508 frieren) [σ=5 worse, σ=10 not seen yet]
- **T_max×lr basin pairing**: {T_max=24,T_max=26}×lr=2.5e-4 — in flight (#4531 thorfinn)
- **surf_weight downward**: {7.5, 5} — in flight (#4533 nezuko)

**T_max axis summary** (CLOSED):
- T_max=14: old start (BL chain)
- T_max=16: regresses
- T_max=20: BL chain intermediate (val 53.08)
- **T_max=22: CURRENT BEST (val 49.75) — sharp resonance under 13-epoch timeout**
- T_max=21: +5.77 val regression (LR too fast-decaying at epoch 13)
- T_max=23: +3.92 val regression (LR too slow-decaying at epoch 13)
- T_max=24: DIVERGES at ls substrate (Finding #33)

**Note on guard parser issue**: Advisor template comments containing `SENPAI-RESULT: {...}` (literal brace placeholder) trip the merge guard parser. Future advisor comments should use "SENPAI-RESULT JSON marker" or similar instead of the literal template.
