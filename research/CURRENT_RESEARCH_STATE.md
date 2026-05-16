<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~20:00 UTC) — **#4056 thorfinn MERGED** (grad_clip=1.0: val 61.18 / test 52.09, new best); #4057 edward CLOSED; #4120 thorfinn LR@clip1 / #4122 edward wd@clip1 assigned; tanjiro T_max=18 (val 59.22) WINNING, pending terminal; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 61.1778** (PR #4056 thorfinn — Lion lr=1.5e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14 + **grad_clip=1.0**; NO spec_norm; run `y5tua53k`)
**test_avg/mae_surf_p = 52.0853** (same run, clean 4-split)

Per-split val: in_dist 65.37, camber_rc 76.90, camber_cruise 41.74, re_rand 60.70
Per-split test: in_dist 56.81, camber_rc 66.84, camber_cruise 34.22, re_rand 50.47

**Δ vs prior best (PR #3976 lr=1.5e-4 no-clip, val 63.05 / test 53.60): −1.87 val / −1.51 test**

⚠️ **PENDING**: tanjiro #4063 T_max=18 (val 59.22 / test 50.79) may set a new best once terminal is posted.

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
| **#3976** | **Lion lr=1.5e-4** | **64.68 → 63.05** | **56.17 → 53.60** | **−2.5% / −4.6%** |
| **#4056** | **grad_clip=1.0** | **63.05 → 61.18** | **53.60 → 52.09** | **−3.0% / −2.8%** |

**Total improvement:** val 135 → 63.05 (−53%), test ~130 → 53.60 (−59%)

**LR monotone trend:** val(2e-5)=78.93 → val(5e-5)=69.69 → val(1e-4)=65.41 → val(1.5e-4)=63.05 → val(2e-4)=63.84 ← inflects back up. Optimum confirmed in [1.2e-4, 1.7e-4].

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#4084** | **fern** | **R10 H48: Dropout sweep {0.05, 0.10} on Transolver blocks at lr=1.5e-4** | **Just assigned** |
| **#4085** | **askeladd** | **R10 H49: Batch size sweep {8, 16} with Lion at lr=1.5e-4** | **Just assigned** |
| **#4120** | **thorfinn** | **R10 H52: LR sweep at clip=1.0 {1.5e-4 ctrl, 2e-4, 2.5e-4}** | **Just assigned** |
| **#4122** | **edward** | **R10 H53: wd sweep at clip=1.0 {3e-4, 5e-4, 1e-3 ctrl, 2e-3}** | **Just assigned** |
| #4096 | frieren | R10 H50: SGDR cosine warm restarts {T_0=7, T_0=4 T_mult=2} | WIP |
| #4063 | tanjiro | R10 H47: T_max=18 WINNING (val 59.22) — T_max=20 still running | WIP, **near terminal** |
| #4015 | nezuko | R10 H39: Layer scale — sent back, Arm D+E (new substrate) | Sent back |
| #4044 | alphonse | R10 H40: Multi-param FiLM — ctrl-only, treatment not launched | Nudged |

## Recent closures (informative nulls — recent sessions)

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4057 | edward | Surfrouting: vec arm val 62.76 (+1.58 vs new BL 61.18); falsification triggered (bias→0) | CLOSED |
| #4049 | frieren | spec_norm at lr=1.5e-4: Arm B −0.27 val vs ctrl (within noise) | CLOSED |
| #4046 | askeladd | p_weight upweighting monotone hurts (1→2→3 worsens) | CLOSED |
| #4045 | fern | Capacity bottleneck = wall clock (n=128 ctrl best within budget) | CLOSED |
| #3957 | tanjiro | T_max=20 best within-substrate (val 67.48 lr=5e-5) — above new BL | CLOSED |
| #3958 | thorfinn | wd=5e-4 best (val 64.79) — above new BL 63.05 | CLOSED |
| #3913 | edward | Re-sampler monotonically hurts; alpha=0 ctrl=64.53 | CLOSED |
| #3977 | fern | Stochastic depth hurts at 5-block depth | CLOSED |
| #3978 | askeladd | MixUp catastrophic (+23-27 val) — non-physical targets | CLOSED |
| #3955 | alphonse | n_power_iter=1 optimal; higher = over-regularizes | CLOSED |

## R10/R11 hypothesis map (current round)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| Conditioning signal | Multi-FiLM 11 global params | #4044 alphonse | −2 to −4 val; camber_rc target |
| Architecture capacity | n_hidden 192/256 vs 128 | #4045 fern | −1 to −4 val |
| Metric alignment | p_weight {2x, 3x} Huber loss | #4046 askeladd | −1 to −3 val |
| Architecture stability | Layer scale init {1e-4, 1e-5} | #4015 nezuko | −1 to −3 val |
| Optimizer stability | Grad clip {0.5, 1.0, 2.0} at lr=1.5e-4 | #4056 thorfinn | −0 to −2 val mean; variance reduction |
| Architecture routing | Surface-biased PhysicsAttention routing | #4057 edward | −1 to −3 val; camber_rc target |
| Schedule depth | T_max sweep {14, 18, 20} at lr=1.5e-4 | #4063 tanjiro | ≤−1 to +1 val; tests within-substrate T_max=20 transfer |
| Block regularization | Dropout sweep {0.05, 0.10} in PhysicsAttention + FFN | #4084 fern | −0 to −2 val; targets camber_rc generalization |
| Gradient signal cleanliness | Batch size {8, 16} with Lion at lr=1.5e-4 | #4085 askeladd | −0 to −1.5 val; risk: wall-clock budget |
| LR schedule reformulation | SGDR cosine warm restarts {T_0=7×2, T_0=4 T_mult=2} | #4096 frieren | −0 to −2 val; tests Lion+EMA × restarts compositional |

## Key findings (cumulative, 19)

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
13. **Output-only spectral norm**: Lipschitz constraint on head MLP reduces peak-pressure over-fitting at lr=5e-5 (−1.39 val / −0.81 test). Contribution diminishes with higher lr.
14. **Lion lr=1.5e-4 optimal**: monotone trend 2e-5→5e-5→1e-4→1.5e-4 (inflection at 2e-4). Largest single gain since Lion optimizer itself.
15. **Train-time z-aug fails** — AoA asymmetry.
16. **surf_weight optimum substrate-dependent**.
17. **Lion β1=0.9 optimal** in {0.8, 0.9, 0.95}.
18. **spec_norm Lipschitz contribution diminishes as lr grows**: −1.39 val at lr=5e-5 → ~0 at lr=1e-4 → −0.27 val (noise) at lr=1.5e-4. Output-head Lipschitz closed; Lion's sign-update already bounds per-step gradient magnitude.
19. **Input MixUp catastrophic** on CFD pressure fields: non-physical blended targets (+23-28 val). FiLM's log(Re) conditioning also gets mixed — further invalidating the augmentation.

## Next priorities

1. **Tanjiro #4063 T_max=18 terminal** — T_max=18 val 59.22 / test 50.79 beats new BL 61.18/52.09. T_max=20 still running. Merge when terminal posted → likely new best.
2. **Multi-FiLM treatment arm (alphonse #4044)** — highest-priority R10 hypothesis (camber_rc target). Currently ctrl-only; nudged for treatment launch.
3. **Layer scale at new substrate (nezuko Arms D+E)** — if layer_scale=1e-4 on clip=1.0+lr=1.5e-4 beats val 61.18.
4. **R10 in flight**: SGDR (#4096 frieren), dropout (#4084 fern), batchsize (#4085 askeladd), LR@clip1 (#4120 thorfinn), wd@clip1 (#4122 edward).
5. **Combine T_max=18 + clip=1.0** — the biggest follow-up after tanjiro's PR merges; both axes are independent.
6. **Unassigned hypotheses** (if more idle): H41 SWA, finer T_max sweep, EMA decay at new substrate.
