<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~13:30 UTC) — #3843 frieren MERGED (new baseline val 65.41 / test 56.06); #3808 fern closed informative (surf_weight not substrate-independent); #3712 askeladd closed informative (β1=0.9 confirmed optimal); 3 R9 assignments issued; edward #3913 debugging; 4 R8 students in WIP; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 65.4142** (PR #3843 frieren — Lion lr=1e-4 + n_fourier=0 + FiLM + Lion wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14; NO spec_norm, run `bw38ym4h`)
**test_avg/mae_surf_p = 56.0627** (same run, clean 4-split)

Per-split val: in_dist 69.60, camber_rc 80.18, camber_cruise 46.19, re_rand 65.69
Per-split test: in_dist 61.03, camber_rc 70.47, camber_cruise 37.84, re_rand 54.91

**Δ vs prior best (PR #3748 spec_norm, val 68.96 / test 60.82): −3.55 val / −4.76 test**

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
| **#3843** | **Lion lr=1e-4** | **68.96 → 65.41** | **60.82 → 56.06** | **−5.2% / −7.8%** |

**Total improvement:** val 135 → 65.41 (−52%), test ~130 → 56.06 (−57%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3976** | **frieren** | **R9 H36: Lion lr push {1.5e-4, 2e-4}** | **Just assigned** |
| **#3977** | **fern** | **R9 H37: Stochastic depth p={0, 0.1, 0.2}** | **Just assigned** |
| **#3978** | **askeladd** | **R9 H38: Input MixUp alpha={0, 0.2, 0.5}** | **Just assigned** |
| #3954 | nezuko | R8 H32: spec_norm + lr=1e-4 combined | WIP — HIGHEST PRIORITY watch |
| #3955 | alphonse | R8 H33: spec_norm n_power_iter sweep {1, 3, 5} | WIP |
| #3957 | tanjiro | R8 H34: cosine T_max sweep {10, 14 ctrl, 20} under spec_norm | WIP |
| #3958 | thorfinn | R8 H35: Lion wd sweep at lr=1e-4 {0.5e-3, 1e-3 ctrl, 2e-3} | WIP |
| #3913 | edward | R8 H31: Re-extremity WeightedRandomSampler α ∈ {0, 0.5, 1.0} | WIP — 8 failed runs, implementation issue, debug posted |

## Closed this session

- **#3843 frieren** — Lion lr=1e-4: val 65.41 / test 56.06. **MERGED** — largest gain since R3 Lion. New baseline.
- **#3808 fern** — surf_weight sweep: confirmation arm D (w=20, n_fourier=0) val 72.71 = regression vs new baseline 65.41. Closed informative. Key finding: surf_weight optimum is substrate-dependent.
- **#3712 askeladd** — β1 sweep {0.8, 0.9, 0.95}: β1=0.9 optimal, β1=0.8 +4.57 worse, β1=0.95 +1.75 worse. Closed informative. Paper-default confirmed.
- **#3817 alphonse** — FiLM ablation: confirmed −4.35 val / −4.56 test. Paper-critical.
- **#3842 tanjiro** — Sobolev: catastrophic val 212. Loss scaling broken.
- **#3845 thorfinn** — Train-time z-aug: val 93. AoA asymmetry root cause.
- **#3786 edward** — Huber β: β=0.05 locally optimal.

## Key findings (cumulative)

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

## R9 hypothesis map (current round)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| LR push | lr ∈ {1.5e-4, 2e-4} | #3976 frieren | −1-3 val if trend continues; bounds LR search |
| Residual regularization | Stochastic depth p ∈ {0.1, 0.2} | #3977 fern | Fresh mechanism, uncertain; ~−1-2 val if effective |
| Data augmentation | Input MixUp α ∈ {0.2, 0.5} | #3978 askeladd | OOD generalization target; uncertain |
| Compound: spec_norm + lr=1e-4 | Stack two winners | #3954 nezuko | val ~62-65 (HIGHEST PRIORITY — watch closely) |
| Spec_norm constraint | n_power_iter {1, 3, 5} | #3955 alphonse | Marginal refinement |
| LR schedule under spec_norm | T_max {10, 14, 20} | #3957 tanjiro | Informative ablation |
| LR × wd at lr=1e-4 | wd {0.5, 1.0, 2.0}×1e-3 | #3958 thorfinn | Calibration check |
| Re-stratified sampler | α ∈ {0, 0.5, 1.0} | #3913 edward | Implementation debugging — recovery pending |

## Next priorities

1. **nezuko #3954** (spec_norm + lr=1e-4 combined) — if val < 62, this is the new frontier. Watch closely.
2. **frieren #3976** (lr push) — could push val below 63 if trend continues; monotone signal is strong.
3. **thorfinn #3958** (wd calibration at lr=1e-4) — informs whether current wd=1e-3 is still optimal at higher lr.
4. **edward #3913** — needs implementation fix; 8 failures is a lot. If still unresolved at next wakeup, close and reassign fresh hypothesis.
