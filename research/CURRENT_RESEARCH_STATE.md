# SENPAI Research State

- **Date:** 2026-05-16 14:55
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #3527 (tanjiro, merged):** BF16 + LayerScale γ=0.01 + n_freqs=10 (no EMA)
- **val_avg/mae_surf_p: 67.19** | **test_avg/mae_surf_p: 58.05**
- Per-split test surf_p: single=67.42, rc=69.79, cruise=38.66, re_rand=56.35
- **Cumulative improvement: -47.8% val from round-5 start (~128.69)**
- Key insight: BF16 buys 17 epochs vs 12 (FP32). At 17 epochs, n_freqs=10 BEATS n_freqs=14 — fewer aliased Fourier components outperforms at convergence. EMA+n14 (quad-compound) also beats old baseline at 68.50/60.15 but is weaker than n10 alone — EMA costs ~2 epochs overhead and n14 over-fits at the extended horizon.

## Improvement history

| PR | Method | val_avg | test_avg | Δ val |
|---|---|---|---|---|
| (round 5 baseline) | ~128 (no-clip no-Huber) | ~128.69 | — | — |
| #3213 (frieren, merged) | Huber delta=0.3 | 103.18 | 92.02 | -19% |
| #3182 (askeladd, merged) | Huber-0.3 + clip-0.25 | 98.62 | 88.14 | -4.4% |
| #3221 (nezuko, merged) | Fourier n=10 + Huber-0.3 | 89.27 | 79.43 | -9.5% |
| #3333 (frieren, merged) | Fourier+Huber+T_max=20+clip=0.25 | 84.59 | 73.89 | -5.2% |
| #3529 (frieren, merged) | clip=0.25→1.0 on full stack | 84.01 | 72.95 | -0.7% |
| #3438 (nezuko, merged) | n_freqs=14 on full stack | 81.08 | 71.52 | -3.5% |
| #3593 (alphonse, merged) | LayerScale γ-init=0.01 on full stack | 72.77 | 65.12 | -10.2% |
| #3192 (edward, merged) | LayerScale + n_freqs=14 + EMA 0.998 | 71.20 | 62.71 | -2.16% |
| **#3527 (tanjiro, merged)** | **BF16 + LayerScale + n_freqs=10** | **67.19** | **58.05** | **-5.6%** |

## Active WIP

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| edward | #3971 | EMA warm-up ramping {0.9→0.998 @100, 0.9→0.9995 @200} on FP32 triple | wave-10 WIP |
| alphonse | #3964 | LayerScale γ-init sweep {0.005, 0.02} on FP32 triple compound | wave-10 WIP |
| askeladd | #3983 | Huber δ sweep {0.15, 0.5} on FP32 triple compound | wave-10 WIP |
| tanjiro | #4005 | BF16+LS+n10+EMA 0.998 — missing cell | wave-11 NEW |
| fern | #4006 | n_freqs sweep {8, 12} on BF16+LS stack | wave-11 NEW |
| frieren | #4007 | Width scaling: n_hidden=144 on BF16+LS+n10 | wave-11 NEW |
| thorfinn | #4008 | surf_weight sweep {5.0, 20.0} on BF16+LS+n10 | wave-11 NEW |
| nezuko | #4009 | Gradient clip sweep {0.5, 1.0} on BF16+LS+n10 | wave-11 NEW |

## Closed this round

| PR | Reason |
|---|---|
| #3424 (askeladd) | clip=0.1 × Huber δ both arms regress 5%; LayerScale already gates gradients |
| #3878 (edward) | EMA decay sweep {0.995, 0.999}: both worse than 0.998 |
| #3882 (alphonse) | SAM ρ=0.05: 2× overhead halves epochs; no flat-min benefit |
| #3823 (nezuko) | Lookahead: both k=5/k=10 ~15-21% worse; disrupts LayerScale γ |
| #3740 (frieren) | Asymmetric LayerScale γ-init: γ converges naturally regardless of init |
| #3730 (alphonse) | LayerScale+n14 sub-additive WITHOUT EMA |
| #3782 (fern) | AdamW eps sweep falsified; default 1e-8 optimal |
| #3784 (thorfinn) | LR sweep on FP32 triple: clip_frac=1.0 throughout; superseded by BF16 |
| #3883 (fern) | T_max sweep: T_max=12 worst, T_max=20 optimal |
| #3909 (frieren) | Learnable Fourier: frequencies barely migrate; overhead costs ~2 epochs |
| #3941 (nezuko) | WD sweep {3e-5, 3e-4} both worse; WD=1e-4 confirmed optimal |

## Current research themes

### BF16 regime shift (wave-11 — all new assignments target BF16+LS+n10)

Key insight: **BF16 extended convergence horizon changes optimal hyperparameters**. At 12 epochs (FP32), n14>n10 and EMA helps. At 17 epochs (BF16), n10>n14 and EMA costs more than it gains. All sweepable hyperparameters need re-evaluation on the new BF16+LS+n10 stack.

1. **Missing cell — tanjiro #4005**: BF16+LS+n10+EMA 0.998. Does EMA help or hurt n10 at 17 epochs? EMA's half-life is 41% of training — potentially meaningful. One arm, cheap, decisive.

2. **n_freqs on BF16+LS — fern #4006**: n=8 and n=12. Aliasing regime reversed: n10>n14 at 17 epochs. Does n=8 win? n=12 interpolates.

3. **Width scaling — frieren #4007**: n_hidden=144 with BF16+LS+n10. BF16's −21% memory opens wider models. LayerScale's γ gating lets model stay sparse if extra width isn't needed. Expected ~14 epochs.

4. **surf_weight sweep — thorfinn #4008**: surf_weight={5.0, 20.0} vs default 10.0. **Never tested in this programme.** Direct control over surface vs volume loss weighting.

5. **Clip sweep — nezuko #4009**: clip={0.5, 1.0} vs 0.25. clip_frac=1.0 throughout all runs. LayerScale provides implicit gating; maybe clip=0.25 is double-regularizing. Prior clip=1.0 test (PR #3529) was pre-LayerScale.

### Old FP32 triple stack (wave-10, completing)

Edward, alphonse, askeladd still running on FP32 triple. Results informative — EMA warm-up and Huber δ findings will inform wave-12+ planning.

## Key insights accumulated

- **BF16 regime shift**: n_freqs=10 > n_freqs=14 at 17 epochs. Everything from FP32 triple needs re-evaluation on BF16+LS+n10.
- **clip=0.25 is 100% saturated**: Every step is clipped. Effective LR = clip × grad direction. clip sweep is high priority on new stack.
- **T_max=20 confirmed optimal** for 12-17 epoch runs. T_max=12 traps in suboptimal basin. No re-sweep needed.
- **surf_weight=10.0 never swept**: Only major hyperparameter unexplored. High priority.
- **AdamW fully confirmed**: β2=0.999, eps=1e-8, WD=1e-4 all optimal. Inner-optimizer exhausted.
- **Learnable Fourier dead**: Frequencies don't migrate; overhead too costly.
- **LayerScale γ=0.01 biggest single win**: −10.2% val from per-channel residual gating.
- **EMA at 12 epochs helps; at 17 epochs (BF16), its cost (~2 epochs) exceeds benefit on n10 stack.**

## Potential next research directions

- **n_layers depth scaling** — no current flag; requires train.py modification; BF16 memory headroom makes viable
- **Batch size 8** — BF16 memory freed; better gradient estimates
- **n_freqs=6** — if n=8 wins on fern's sweep, continue the aliasing trend lower
- **n_hidden=160** — if frieren's n=144 shows width helps, scale further
- **Huber δ on new BF16 stack** — askeladd's FP32 result will inform whether to re-test
- **Sub-60 target**: val=67.19 now; missing cell + width scaling could push to 63-65

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
