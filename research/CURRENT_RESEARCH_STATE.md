<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~15:30 UTC) — #3954 nezuko MERGED (val 64.68 / test 56.17); nezuko assigned R10 H39 (layer scale #4015); 7 R8/R9 WIP arms running; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 64.6812** (PR #3954 nezuko — Lion lr=1e-4 + spec_norm(output, n_power_iter=1) + n_fourier=0 + FiLM + Lion wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14; run `pc7lsis0`)
**test_avg/mae_surf_p = 56.1746** (same run, clean 4-split)

Per-split val: in_dist 69.26, camber_rc 78.64, camber_cruise 46.37, re_rand 64.47
Per-split test: in_dist 61.06, camber_rc 69.24, camber_cruise 38.56, re_rand 55.83

**Δ vs prior best (PR #3843 lr=1e-4 no-spec_norm, val 65.41 / test 56.06): −0.73 val / +0.11 test**

Note: val improvement is within seed noise (σ≈2.77). spec_norm at lr=1e-4 is orthogonal to Lion's sign-based update bounding but does not add meaningfully. Multiple reproductions of lr=1e-4 without spec_norm cluster at val 64.18–64.79 (mean ~64.5).

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
| **#3954** | **spec_norm + lr=1e-4** | **65.41 → 64.68** | **56.06 → 56.17** | **−1.1% / +0.2%** |

**Total improvement:** val 135 → 64.68 (−52%), test ~130 → 56.17 (−57%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#4015** | **nezuko** | **R10 H39: Layer scale init {1e-4, 1e-5} on Transolver blocks** | **Just assigned** |
| **#3976** | **frieren** | **R9 H36: Lion lr push {1.5e-4, 2e-4}** | jurrwig2 (lr=1.5e-4) near terminal; lr=2e-4 pending |
| **#3977** | **fern** | **R9 H37: Stochastic depth p={0.1, 0.2}** | 8zhftd2l (p=0.1) near terminal; p=0.2 pending |
| **#3978** | **askeladd** | **R9 H38: Input MixUp alpha={0.2, 0.5}** | u1k8cpqz (α=0.2) running; α=0.5 needs launch (nudge posted) |
| #3958 | thorfinn | R8 H35: Lion wd sweep {0.5e-3, 1e-3, 2e-3} | wd=5e-4 finished (val 64.79 — above new BL 64.68); wd=2e-3 running |
| #3957 | tanjiro | R8 H34: T_max sweep {10, 14, 20} under spec_norm | d3jr861j T_max=10 near terminal; T_max=20 pending |
| #3955 | alphonse | R8 H33: n_power_iter sweep {1, 3, 5} under spec_norm | npow=5 running; npow=3 finished val 71.91 |
| #3913 | edward | R8 H31: Re-extremity WeightedRandomSampler α={0, 0.5, 1.0} | alpha=0.5+lr=1e-4 started (step 84); alpha=0 val 64.53 (no sampler effect) |

## R8 round interpretation (under new baseline 64.68)

These PRs test spec_norm variations at old lr=5e-5 substrate. Most findings will be informative:
- **alphonse #3955**: n_power_iter has diminishing returns; n_power_iter=1 looks optimal at lr=5e-5 (consistent with now being in lr=1e-4 world where spec_norm itself contributes ~0)
- **tanjiro #3957**: T_max sweep informative about schedule sensitivity under spec_norm
- **thorfinn #3958**: wd=5e-4 (val 64.79) no longer beats new BL 64.68; wd=2e-3 still running (could regress or match)
- **edward #3913**: alpha=0 reproduces lr=1e-4 baseline; alpha=0.5 will reveal if Re-stratified sampling has any signal at optimal lr

## R9 hypothesis map (current round)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| LR push | lr ∈ {1.5e-4, 2e-4} | #3976 frieren | −1-3 val if monotone trend continues |
| Residual regularization | Stochastic depth p ∈ {0.1, 0.2} | #3977 fern | Uncertain; ~−1-2 val if effective |
| Data augmentation | Input MixUp α ∈ {0.2, 0.5} | #3978 askeladd | OOD generalization; uncertain |

## R10 hypothesis map (new round — nezuko assigned)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| Architecture stabilization | Layer scale init ∈ {1e-4, 1e-5} | #4015 nezuko | −1-3 val if effective; moderate confidence |

## Next priorities

1. **frieren #3976 lr=1.5e-4** (jurrwig2) — near terminal. If val < 62, this is the new frontier. If val ~64, confirms 1e-4 was the inflection.
2. **fern #3977 stoch depth** — near terminal. First fresh architectural regularizer tested.
3. **askeladd #3978** — alpha=0.2 running; alpha=0.5 needs launch (duplicate ctrl y7l9wliq should be killed).
4. **R8 PRs (thorfinn, tanjiro, alphonse)** — close as informative when arms finish; substrate is now outdated.
5. **edward #3913** — alpha=0.5+lr=1e-4 running; close as informative if similar to alpha=0.

## Key findings (cumulative, 18)

1. **FiLM on log(Re)** contributes −4.35 val / −4.56 test under n_fourier=0 (paper-critical ablation confirmed).
2. **EMA(0.997)** contributes +4.4 val on top of Lion.
3. **Fourier PE inert** under FiLM+Lion+EMA. n_fourier=0 wins.
4. **EMA decay robust** in [0.995, 0.997].
5. **Sobolev surface regularization**: catastrophically destabilizes at w=0.05+ on spec_norm substrate.
6. **LLRD doesn't transfer** from fine-tuning to scratch.
7. **LR warmup adds nothing** to Lion.
8. **Block-FiLM regresses.** Output-only FiLM is correct topology.
9. **Lookahead dead end.**
10. **TTA z-reflection fails** — AoA asymmetry.
11. **Huber β=0.05 locally optimal** in [0.05, 0.20].
12. **Seed noise floor ≈ 2.77 val** (two identical runs).
13. **Output-only spectral norm (−1.39 val / −0.81 test)**: Lipschitz constraint on head MLP reduces peak-pressure over-fitting. Arm C (output+FiLM spec_norm) hurts — FiLM's adaLN-Zero init incompatible with Lipschitz bounding.
14. **Lion lr=1e-4 wins massively (−3.55 val / −4.76 test vs spec_norm baseline)**: sign-based updates tolerate 2× LR. lr=2e-5 significantly worse. Monotone trend: val 78.93→69.69→65.41 across 2× steps.
15. **Train-time z-aug fails** for same reason as TTA — AoA asymmetry is training-data deep.
16. **surf_weight optimum substrate-dependent**: w=20 wins on n_fourier=16 (−3.28 val) but regresses on n_fourier=0 (+7.30 vs new baseline). Input encoding changes which loss signals the model prioritizes.
17. **Lion β1=0.9 optimal** in {0.8, 0.9, 0.95}: asymmetric — lower momentum hurts more (+4.57) than higher (+1.75).
18. **spec_norm Lipschitz contribution diminishes as lr grows**: −1.39 val at lr=5e-5, ~0 val at lr=1e-4. Lion's sign-based step size naturally bounds output gradient growth; spectral cap redundant at higher lr.
