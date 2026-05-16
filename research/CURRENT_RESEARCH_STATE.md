<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-16 (~22:35 UTC) — **#4173 thorfinn triple-compose REVIEWED+SENT BACK** (val tied +0.086, test better −0.69; sent back for lr=1.8e-4 + T_max=18 arms); **#4128 fern surf_weight CLOSED** (1/3 arms launched, obsolete substrate); **#4192 fern Huber β@lr=2e-4** assigned; 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 56.8913** (PR #4120 thorfinn — Lion **lr=2e-4** + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + T_max=14 + **grad_clip=1.0**; NO spec_norm; run `1c58zju8`)
**test_avg/mae_surf_p = 49.0322** (same run, clean 4-split)

Per-split val: in_dist 61.01, camber_rc 71.92, camber_cruise 37.30, re_rand 57.34
Per-split test: in_dist 52.64, camber_rc 64.54, camber_cruise 31.01, re_rand 47.94

**Δ vs prior best (PR #4063 T_max=20, val 57.66 / test 49.45): −0.77 val / −0.41 test**
**All 4 splits improve.** LR optimum at clip=1.0 shifts from 1.5e-4 to 2e-4 (direction-sensitive clip interaction).

Key mechanism: clip=1.0 clips every step (pre-clip ‖g‖ median ~23.7 >> 1.0). Clipped step direction (normalized gradient) ≠ sign(momentum), and their dot product depends on nominal lr. With clip=1.0, the LR-vs-val curve has same shape as no-clip but shifted upward in lr.

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
| **#4120** | **lr=2e-4 at clip=1.0** | **57.66 → 56.89** | **49.45 → 49.03** | **−1.3% / −0.8%** |

**Total improvement:** val 135 → 56.89 (−57.9%), test ~130 → 49.03 (−62.3%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #4145 | alphonse | R11 H55: grad_clip=1.0 + T_max=20 at lr=1.5e-4 (+ T_max=24 extension) | WIP |
| #4015 | nezuko | R10 H39: layer_scale=1e-4 + T_max=20 composition (Arms F+G) | WIP |
| **#4192** | **fern** | **R11 H61: Huber β recalibration at lr=2e-4 {0.03, 0.05 ctrl, 0.10}** | **Just assigned** |
| #4153 | askeladd | R11 H58: Lion β2 sweep at T_max=20 {0.98, 0.99 ctrl, 0.995} | WIP |
| #4173 | thorfinn | R11 H59 (extended): Triple composition + Arms D (lr=1.8e-4 + T_max=20) and E (lr=2e-4 + T_max=18) | Sent back |
| **#4180** | **edward** | **R11 H60: Clip ratio sweep at lr=2e-4 {0.7, 1.0 ctrl, 1.4}** | **Just assigned** |
| #4152 | frieren | R11 H57: EMA decay sweep at T_max=20 {0.995, 0.997 ctrl, 0.999} | WIP |
| #4148 | tanjiro | R11 H56: LR recalibration at T_max=20 {1.3e-4, 1.5e-4 ctrl, 1.7e-4} | WIP |

**All 8 students now staffed.**

## Recent closures (informative nulls — recent sessions)

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4128 | fern | surf_weight@clip=1.0 (old substrate): sw=5 → 60.59 val / 51.95 test (worse); sw=10 ctrl & sw=20 never launched; obsolete substrate | CLOSED |
| #4122 | edward | wd@clip=1.0 (old substrate): wd=3e-4 → 62.99 (+1.8 vs ctrl 61.18), wd=5e-4 → 61.45 (within noise); wd=1e-3 ctrl & wd=2e-3 never launched; substrate obsolete | CLOSED |
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
| **Triple composition (extended)** | **T_max=20 + lr=2e-4 + clip=1.0 → val 56.98 (tied); Arms D (lr=1.8e-4@T_max=20) + E (lr=2e-4@T_max=18)** | **#4173 thorfinn** | **First arm tied on val (+0.086) but test better −0.69; localizing the lr×T_max minimum** |
| Schedule+clip composition | grad_clip=1.0 + T_max=20 at lr=1.5e-4 (+ T_max=24) | #4145 alphonse | −0 to −2 val; tests if clip×T_max compose (suboptimal lr) |
| LR recalibration at T_max=20 | lr {1.3e-4, 1.5e-4 ctrl, 1.7e-4} at T_max=20 | #4148 tanjiro | −0 to −1 val; note lr=2e-4 tied — needs Arm D |
| EMA decay at T_max=20 | ema_decay {0.995, 0.997 ctrl, 0.999} at T_max=20 | #4152 frieren | −0 to −1 val; longer averaging at noisy endpoint |
| Lion β2 at T_max=20 | lion_beta2 {0.98, 0.99 ctrl, 0.995} at T_max=20 | #4153 askeladd | −0 to −1.5 val; untested optimizer-state axis |
| Architecture stability composition | layer_scale=1e-4 + T_max=20 | #4015 nezuko | −0 to −2 val; tests if both wins stack |
| **Huber β recalibration at lr=2e-4** | **β {0.03, 0.05 ctrl, 0.10} at lr=2e-4 + clip=1.0** | **#4192 fern** | **−0 to −1.5 val; tests whether finding #11 extends to new substrate; β=0.03 is novel** |
| Clip ratio recalibration at lr=2e-4 | clip {0.7, 1.0 ctrl, 1.4} at lr=2e-4 | #4180 edward | −0 to −1.5 val; parallel to finding #22 |

## Key findings (cumulative, 22)

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
14. **Lion lr=1.5e-4 optimal WITHOUT clip**: monotone trend, inflection at 2e-4. **UPDATED by finding #22.**
15. **Train-time z-aug fails** — AoA asymmetry.
16. **surf_weight optimum substrate-dependent**.
17. **Lion β1=0.9 optimal** in {0.8, 0.9, 0.95}.
18. **spec_norm Lipschitz contribution diminishes as lr grows**: closed at lr=1.5e-4.
19. **Input MixUp catastrophic** on CFD pressure fields (+23-28 val).
20. **T_max=20 optimal** within SENPAI_TIMEOUT_MINUTES=30 budget: monotone T_max=14→18→20. Mechanism: higher time-averaged LR within budget; EMA smooths late-training noise.
21. **Multi-FiLM global conditioning FALSIFIED**: Global γ/β conditioning on 11 params hurts camber_rc OOD (−5.45 val on target split). Per-node geometric variation must reach model through attention path, not global FiLM scalar.
22. **LR optimum shifts from 1.5e-4 to 2e-4 at clip=1.0**: clip=1.0 clips every step (pre-clip ‖g‖ median ~23.7 >> 1.0). Clipped step direction (g/‖g‖) ≠ sign(Lion momentum); their interaction is direction-sensitive, not a pure scale factor. The LR-vs-val curve at clip=1.0 has same shape as no-clip but shifted upward in lr. Inflection now at 2e-4 (regresses by 2.5e-4). All 4/4 splits improve.

## Next priorities

1. **thorfinn #4173 triple composition**: T_max=20 + lr=2e-4 + clip=1.0 — the highest-priority uncharted territory. Potential val ~54–55 if all three compose.
2. **alphonse #4145**: T_max=20 + clip=1.0 at lr=1.5e-4 — will tell us if clip×T_max compose even at suboptimal lr; result informs thorfinn #4173 interpretation.
3. **tanjiro #4148**: LR@T_max=20 — note that lr=2e-4 finding (#22) suggests Arm D (lr=2e-4) should be added if Arm C (lr=1.7e-4) looks promising. The LR optimum at T_max=20+clip=0 may be between 1.5e-4 and 2e-4.
4. **edward #4122 + fern #4128**: On old T_max=14 substrate. If they show wins, will need re-testing on T_max=20+lr=2e-4+clip=1.0 substrate.
5. **Paper completeness**: T_max ablation (done), clip ablation (done), LR shift finding (done), triple composition (in-flight #4173).
