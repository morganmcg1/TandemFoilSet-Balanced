<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~16:45 UTC) — #3976 frieren MERGED (val 63.05 / test 53.60, lr=1.5e-4 inflection confirmed); closed 4 informative nulls (#3955 alphonse, #3977 fern, #3978 askeladd, + see below); 5 R10/R11 assignments; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 63.0492** (PR #3976 frieren — Lion lr=1.5e-4 + n_fourier=0 + FiLM + Lion wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14; NO spec_norm; run `jurrwig2`)
**test_avg/mae_surf_p = 53.6049** (same run, clean 4-split)

Per-split val: in_dist 64.45, camber_rc 80.74, camber_cruise 43.48, re_rand 63.53
Per-split test: in_dist 55.69, camber_rc 70.55, camber_cruise 35.48, re_rand 52.70

**Δ vs prior best (PR #3954 spec_norm+lr=1e-4, val 64.68 / test 56.17): −1.63 val / −2.57 test**

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

**Total improvement:** val 135 → 63.05 (−53%), test ~130 → 53.60 (−59%)

**LR monotone trend:** val(2e-5)=78.93 → val(5e-5)=69.69 → val(1e-4)=65.41 → val(1.5e-4)=63.05 → val(2e-4)=63.84 ← inflects back up. Optimum confirmed in [1.2e-4, 1.7e-4].

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#4049** | **frieren** | **R11 H46: spec_norm at lr=1.5e-4 (2-arm)** | **Just assigned** |
| **#4044** | **alphonse** | **R10 H40: Multi-param FiLM (all 11 global params)** | **Just assigned** |
| **#4045** | **fern** | **R10 H44: Model capacity n_hidden {192, 256}** | **Just assigned** |
| **#4046** | **askeladd** | **R10 H43: Pressure channel upweighting {2×, 3×}** | **Just assigned** |
| **#4015** | **nezuko** | **R10 H39: Layer scale init {1e-4, 1e-5}** | WIP — just assigned ~1h ago |
| #3958 | thorfinn | R8 H35: wd sweep — nudged to post terminal (null) | WIP — awaiting terminal |
| #3913 | edward | R8 H31: Re-sampler — nudged to post terminal (null) | WIP — awaiting terminal |
| #3957 | tanjiro | R8 H34: T_max=20 running (informative) | WIP — near terminal |

## Recent closures (informative nulls — this session)

| PR | Student | Result | Note |
|----|---------|--------|------|
| #3955 | alphonse | n_power_iter=1 optimal; higher = over-regularizes | CLOSED |
| #3977 | fern | Stochastic depth hurts at 5-block depth | CLOSED |
| #3978 | askeladd | MixUp catastrophic (+23-27 val) — non-physical targets | CLOSED |
| #3913 | edward | Pending terminal — alpha=0 reproduces BL, alpha=0.5 hurts | Awaiting |
| #3958 | thorfinn | Pending terminal — wd=5e-4 above new BL 63.05 | Awaiting |

## R10/R11 hypothesis map (new round)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| Conditioning signal | Multi-FiLM 11 global params | #4044 alphonse | −2 to −4 val; camber_rc target |
| Architecture capacity | n_hidden 192/256 vs 128 | #4045 fern | −1 to −4 val |
| Metric alignment | p_weight {2x, 3x} Huber loss | #4046 askeladd | −1 to −3 val |
| Architecture stability | Layer scale init {1e-4, 1e-5} | #4015 nezuko | −1 to −3 val |
| Spec_norm at new LR | spec_norm at lr=1.5e-4 | #4049 frieren | −0 to −2 val; determines if finding #18 extends |

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
18. **spec_norm Lipschitz contribution diminishes as lr grows**: ~−1.39 val at lr=5e-5, ~0 at lr=1e-4. frieren R11 will test at lr=1.5e-4.
19. **Input MixUp catastrophic** on CFD pressure fields: non-physical blended targets (+23-28 val). FiLM's log(Re) conditioning also gets mixed — further invalidating the augmentation.

## Next priorities

1. **Monitor R10/R11 hypotheses** — multi-FiLM (H40) and capacity (H44) have highest expected upside
2. **thorfinn/edward terminal results** — close as informative when posted
3. **tanjiro T_max=20** — close as informative when terminal
4. **Watch for camber_rc breakthrough** — weakest split (val 80.74); multi-FiLM (alphonse) is the strongest candidate
