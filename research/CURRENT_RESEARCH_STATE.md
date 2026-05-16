<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State

- **Date:** 2026-05-17 (~00:05 UTC) — **#4148 tanjiro LR@T_max=20 CLOSED** (informative null; all arms within noise of ctrl; substrate superseded by #4015); **#4231 tanjiro LR@new-substrate** assigned (lr=1.7e-4, 2.0e-4 at layer_scale+T_max=20 + seed-3 confirmation). 8/8 staffed.
- **Human researcher directives:** None received this launch.

## Current best — merged

**val_avg/mae_surf_p = 54.3009** (PR #4015 nezuko — Lion lr=1.5e-4 + n_fourier=0 + FiLM + wd=1e-3 + EMA(0.997) + Huber β=0.05 + **T_max=20** + **layer_scale_init=1e-4**; NO spec_norm; NO grad_clip; run `8m99yywe`)
**test_avg/mae_surf_p = 47.2883** (same run, clean 4-split)

Per-split val: in_dist 57.78, camber_rc 67.19, camber_cruise 38.28, re_rand 53.95
Per-split test: in_dist 49.18, camber_rc 61.28, camber_cruise 32.26, re_rand 46.43

**Δ vs prior best (PR #4120 lr=2e-4+clip=1.0, val 56.89 / test 49.03): −2.59 val / −1.74 test** (single seed Arm F)
**2-seed mean: val 55.97 / test 48.60** (both seeds beat prior BL; σ=1.67 val under T_max=20)

Key mechanism: layer_scale_init=1e-4 (CaiT/DeiT-III) initializes each block residual at 1e-4 and lets it grow during training — reduces initialization sensitivity. Composes ~80% additively with T_max=20 (observed −7.08 vs predicted −8.78 from #3976 BL).

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
| **#4015** | **layer_scale_init=1e-4 + T_max=20** | **56.89 → 54.30** | **49.03 → 47.29** | **−4.6% / −3.5%** |

**Total improvement:** val 135 → 54.30 (−59.8%), test ~130 → 47.29 (−63.6%)

## Active experiments (8 of 8 students staffed)

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #4145 | alphonse | R11 H55: grad_clip=1.0 + T_max=20 at lr=1.5e-4 (+ T_max=24 extension) | WIP |
| #4192 | fern | R11 H61: Huber β recalibration at lr=2e-4 {0.03, 0.05 ctrl, 0.10} | WIP |
| #4173 | thorfinn | R11 H59 (extended): triple comp Arms D (lr=1.8e-4+T_max=20) + E (lr=2e-4+T_max=18) | Sent back |
| #4201 | nezuko | R11 H62: layer_scale=1e-4 + clip=1.0 + lr=2e-4 + T_max=20 (four-way composition) | WIP |
| #4180 | edward | R11 H60: Clip ratio sweep at lr=2e-4 {0.7, 1.0 ctrl, 1.4} | WIP |
| **#4212** | **askeladd** | **R11 H63: layer_scale magnitude sweep {1e-3, 1e-5, 3e-4} at new BL substrate** | **Just assigned** |
| **#4214** | **frieren** | **R11 H64: EMA decay {0.995, 0.997 ctrl, 0.999} at layer_scale+T_max=20 substrate** | **Just assigned** |
| **#4231** | **tanjiro** | **R11 H65: LR recalibration at new substrate (layer_scale+T_max=20) {1.7e-4, 2.0e-4, seed-3}** | **Just assigned** |

**All 8 students now staffed.**

## Recent closures (informative nulls — recent sessions)

| PR | Student | Result | Note |
|----|---------|--------|------|
| #4148 | tanjiro | LR@T_max=20 no-clip (old substrate): all 3 arms within noise (|Δval|<1.5); lr=1.7e-4 directional best (−0.64 val); lr=2e-4 no longer diverges at T_max=20 (finding). Substrate superseded. | CLOSED |
| #4153 | askeladd | Lion β2@T_max=20 (old substrate): β2=0.995 σ huge (range 12.71 across 3 seeds); β2=0.98 has layer_scale_init=1.0 confound; ctrl never launched; substrate obsolete vs val 54.30 | CLOSED |
| #4152 | frieren | EMA@T_max=20 (old substrate): ema=0.995 within noise; ema=0.999 unstable (1 crashed/1 diverged/1 worse); ctrl never launched; substrate obsolete | CLOSED |
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
| **Four-way composition** | **layer_scale=1e-4 + T_max=20 + lr=2e-4 + clip=1.0** | **#4201 nezuko** | **val ~51–54; highest-priority composition** |
| **LR at new substrate** | **lr {1.7e-4, 2.0e-4, ctrl-seed3} at layer_scale+T_max=20 + no clip** | **#4231 tanjiro** | **−0 to −1.5 val; directional best from old sub (lr=1.7e-4, −0.64) may carry over** |
| Triple composition (extended) | T_max=20 + lr=2e-4 + clip=1.0 → Arms D (lr=1.8e-4) + E (T_max=18) | #4173 thorfinn | Localizing lr×T_max minimum at clip=1.0 |
| Schedule+clip composition | grad_clip=1.0 + T_max=20 at lr=1.5e-4 (+ T_max=24) | #4145 alphonse | Tests clip×T_max compose; Arm E adds layer_scale |
| EMA decay at new substrate | ema_decay {0.995, 0.997 ctrl, 0.999} at layer_scale+T_max=20 | #4214 frieren | −0 to −1 val; may stabilize σ=1.67 seed variance |
| layer_scale magnitude sweep | layer_scale {1e-3, 1e-5, 3e-4} at T_max=20 + lr=1.5e-4 (new BL substrate) | #4212 askeladd | −0 to −1.5 val; tests whether 1e-4 is locally optimal |
| Huber β at lr=2e-4 | β {0.03, 0.05 ctrl, 0.10} at lr=2e-4 + clip=1.0 | #4192 fern | −0 to −1.5 val; needs layer_scale to beat new BL |
| Clip ratio at lr=2e-4 | clip {0.7, 1.0 ctrl, 1.4} at lr=2e-4 | #4180 edward | −0 to −1.5 val; needs layer_scale to beat new BL |
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
23. **layer_scale_init=1e-4 (CaiT/DeiT-III) composes ~80% additively with T_max=20**: layer_scale=1e-4 initializes each block residual at 1e-4 and grows during training (block-0 attn 1e-4 → 0.019 by end). Reduces initialization sensitivity. Confirmed composing with T_max=20 substrate (observed −7.08 val from #3976 BL vs predicted −8.78). 2-seed agreement: D+E on T_max=14 σ=0.016 (very tight), F+G on T_max=20 σ=1.67 (wider). All 4/4 splits improve on both val and test. New best val 54.30 / test 47.29.

## Next priorities

1. **nezuko #4201 four-way composition**: layer_scale=1e-4 + T_max=20 + lr=2e-4 + clip=1.0. Highest-priority composition — four orthogonal wins stacked. Potential val ~51–54.
2. **tanjiro #4231 LR at new substrate**: lr=1.7e-4 was directional best at old substrate (−0.64 val). Does it carry over with layer_scale? Arm D is also a useful seed-3 confirmation for the new BL.
2. **askeladd #4212 layer_scale magnitude**: tests whether 1e-4 is locally optimal at the new BL substrate. If smaller magnitude (1e-5) wins, theory suggests even slower residual growth helps.
3. **frieren #4214 EMA on new substrate**: σ=1.67 seed variance on T_max=20 is concerning; heavier EMA (0.999) may stabilize, OR may be unstable (carried over from prior substrate). Resolves whether substrate change rescues ema=0.999.
4. **alphonse #4145** (WIP, awaiting Arm D+E): Arm B (T_max=20+clip at lr=1.5e-4 → val 56.71) doesn't beat new BL (54.30). Asked student to add Arm E (layer_scale + clip + T_max=20 at lr=1.5e-4) as the natural next composition.
5. **thorfinn #4173 sent back**: Arms D (lr=1.8e-4 + T_max=20 + clip) and E (lr=2e-4 + T_max=18 + clip) — localizing the lr×T_max minimum at clip=1.0. These won't beat new BL unless layer_scale further composes.
6. **tanjiro #4148**: LR@T_max=20 (no clip, no layer_scale). Results will now compare to val 54.30; only Arm D (lr=2e-4) has any hope.
7. **edward #4180**: Clip ratio @ lr=2e-4. Now compared against new BL — only if layer_scale composes with clip will this beat 54.30.
8. **fern #4192**: Huber β @ lr=2e-4. Same — needs to compose with layer_scale.
9. **Paper completeness**: T_max ablation (done), clip (done), layer_scale (done), LR shift finding (done), four-way composition (in-flight #4201), layer_scale magnitude sweep (in-flight #4212). Need EMA ablation on new substrate (in-flight #4214).
