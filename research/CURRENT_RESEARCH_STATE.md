<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~21:25 UTC) — **#4096 frieren SGDR** + **#4085 askeladd batchsize** CLOSED informative (no terminals posted; W&B-observed nulls); **#4152 frieren EMA-decay@T_max=20** + **#4153 askeladd Lion-β2@T_max=20** assigned; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 57.6606** (PR #4063 tanjiro — Lion lr=1.5e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + **T_max=20**; NO spec_norm; NO grad_clip; run `fh3jmkd1`)
**test_avg/mae_surf_p = 49.4491** (same run, clean 4-split)

Per-split val: in_dist 57.88, camber_rc 71.61, camber_cruise 40.47, re_rand 60.69
Per-split test: in_dist 51.04, camber_rc 64.76, camber_cruise 32.44, re_rand 49.55

**Δ vs prior best (PR #4056 grad_clip=1.0, val 61.18 / test 52.09): −3.52 val / −2.64 test**
**All 8 splits improve.** Monotone trend T_max=14→18→20 on all splits.

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
| **#4063** | **T_max=20** | **61.18 → 57.66** | **52.09 → 49.45** | **−5.7% / −5.1%** |

**Total improvement:** val 135 → 57.66 (−57%), test ~130 → 49.45 (−62%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#4145** | **alphonse** | **R11 H55: grad_clip=1.0 + T_max=20 composition + T_max=24 extension** | **Just assigned** |
| **#4015** | **nezuko** | **R10 H39: layer_scale=1e-4 + T_max=20 composition (Arms F+G)** | **Sent back for Arms F+G** |
| #4128 | fern | R10 H54: surf_weight recalibration at clip=1.0 {5, 10 ctrl, 20} | WIP |
| **#4153** | **askeladd** | **R11 H58: Lion β2 sweep at T_max=20 {0.98, 0.99 ctrl, 0.995}** | **Just assigned** |
| #4120 | thorfinn | R10 H52: LR re-optimisation at clip=1.0 {1.5e-4, 2e-4, 2.5e-4} | WIP |
| #4122 | edward | R10 H53: wd sweep at clip=1.0 {3e-4, 5e-4, 1e-3 ctrl, 2e-3} | WIP |
| **#4152** | **frieren** | **R11 H57: EMA decay sweep at T_max=20 {0.995, 0.997 ctrl, 0.999}** | **Just assigned** |
| **#4148** | **tanjiro** | **R11 H56: LR recalibration at T_max=20 {1.3e-4, 1.5e-4 ctrl, 1.7e-4}** | **Just assigned** |

**All 8 students now staffed.**

## Recent closures (informative nulls — recent sessions)

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4096 | frieren | SGDR cosine restarts: T_0=7 → val 64.14 (+6.48), T_0=4 T_mult=2 → 69.51 (+11.85). Restarts oppose T_max=20 mechanism. | CLOSED |
| #4085 | askeladd | Batch size: bs=8 catastrophic (3 reps best 76.93 = +19.27 above BL); bs=16 not launched. Lion+bs=4 correctly tuned. | CLOSED |
| #4044 | alphonse | Multi-FiLM {cond_dim=11, cond_dim=4}: both hurt camber_rc (target). Global γ/β can't substitute per-node geometry. | CLOSED |
| #4084 | fern | Dropout {0.05, 0.10}: monotone hurt; camber_rc −4.23 val (net-negative breadth) | CLOSED |
| #4057 | edward | Surfrouting: vec arm val 62.76 (+1.58 vs BL 61.18); scalar bias no-op | CLOSED |
| #4049 | frieren | spec_norm at lr=1.5e-4: −0.27 val (noise; finding #18 extends) | CLOSED |
| #4046 | askeladd | p_weight upweighting monotone hurts (1→2→3 worsens) | CLOSED |
| #4045 | fern | Capacity bottleneck = wall clock (n=128 ctrl best within budget) | CLOSED |
| #3977 | fern | Stochastic depth hurts at 5-block depth | CLOSED |
| #3978 | askeladd | MixUp catastrophic (+23-27 val) — non-physical targets | CLOSED |
| #3955 | alphonse | n_power_iter=1 optimal; higher = over-regularizes | CLOSED |

## R11 hypothesis map (active round)

| Axis | Hypothesis | PR / student | Expected outcome |
|------|-----------|-------------|-----------------|
| Schedule+optimizer composition | grad_clip=1.0 + T_max=20 + T_max=24 extension | #4145 alphonse | −1 to −3 val; tests orthogonal improvements composing |
| LR recalibration at T_max=20 | lr {1.3e-4, 1.5e-4 ctrl, 1.7e-4} at T_max=20 | #4148 tanjiro | −0 to −2 val; LR optimum may shift with T_max |
| EMA decay at T_max=20 | ema_decay {0.995, 0.997 ctrl, 0.999} at T_max=20 | #4152 frieren | −0 to −1.5 val; longer averaging at noisy endpoint |
| Lion β2 at T_max=20 | lion_beta2 {0.98, 0.99 ctrl, 0.995} at T_max=20 | #4153 askeladd | −0 to −1.5 val; untested optimizer-state axis |
| Architecture stability composition | layer_scale=1e-4 + T_max=20 | #4015 nezuko | −0 to −2 val; tests if both wins stack |
| Loss balance | surf_weight {5, 10 ctrl, 20} at clip=1.0 substrate | #4128 fern | −0 to −2 val; substrate-dependent finding |
| Gradient signal | Batch size {8, 16} at lr=1.5e-4 | #4085 askeladd | −0 to −1.5 val |
| LR recalibration | LR {1.5e-4, 2e-4, 2.5e-4} at clip=1.0 | #4120 thorfinn | −0 to −2 val; tests effective-LR shift from clip |
| WD recalibration | wd {3e-4, 5e-4, 1e-3 ctrl, 2e-3} at clip=1.0 | #4122 edward | −0 to −2 val |
| Schedule reformulation | SGDR cosine warm restarts | #4096 frieren | −0 to −2 val |

## Key findings (cumulative, 21)

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
13. **Output-only spectral norm**: −1.39 val at lr=5e-5; contribution diminishes with lr.
14. **Lion lr=1.5e-4 optimal**: monotone trend, inflection at 2e-4.
15. **Train-time z-aug fails** — AoA asymmetry.
16. **surf_weight optimum substrate-dependent**.
17. **Lion β1=0.9 optimal** in {0.8, 0.9, 0.95}.
18. **spec_norm Lipschitz contribution diminishes as lr grows**: closed at lr=1.5e-4.
19. **Input MixUp catastrophic** on CFD pressure fields (+23-28 val).
20. **T_max=20 optimal** within SENPAI_TIMEOUT_MINUTES=30 budget: monotone T_max=14→18→20, 5× larger effect than old substrate (6.76 vs 1.3 val). Mechanism: higher time-averaged LR within budget; EMA smooths late-training noise.
21. **Multi-FiLM global conditioning FALSIFIED**: Global γ/β conditioning on 11 params hurts camber_rc OOD (−5.45 val on target split). Per-node geometric variation must reach model through attention path, not global FiLM scalar.

## Next priorities

1. **alphonse #4145 composition**: grad_clip=1.0 + T_max=20 (Arm B) is highest-priority combination. If both improvements compose, potential new best ~55–56 val.
2. **nezuko #4015 composition**: layer_scale=1e-4 + T_max=20 (Arms F+G) — tests if architecture stability improvement stacks with schedule win.
3. **All 8 students now staffed.** Tanjiro assigned #4148 LR recalibration at T_max=20.
4. **In-flight recalibration PRs**: thorfinn #4120 LR@clip, edward #4122 wd@clip — note these used OLD T_max=14 substrate; results will need re-testing on T_max=20.
5. **SGDR frieren #4096**: LR schedule reformulation — may be less relevant now that T_max=20 is the baseline, but still informative.
6. **Paper completeness**: Need clean ablation of T_max (done), grad_clip (done), composition (in-flight). Consider EMA decay at T_max=20 substrate as a confirmatory sweep.
