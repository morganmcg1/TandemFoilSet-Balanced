# SENPAI Research State

- **Date:** 2026-05-16 16:25
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

## Active WIP (8 students, full deck)

| Student | PR | Hypothesis | Status |
|---|---|---|---|
| edward | #3971 | EMA warm-up ramping {0.9→0.998 @100, 0.9→0.9995 @200} on FP32 triple | wave-10 WIP (training finished, awaiting results push) |
| tanjiro | #4033 | Huber δ sweep {0.15, 0.5} on BF16+LS+n10 — last unswept primary axis | wave-11 NEW |
| fern | #4006 | n_freqs sweep {8, 12} on BF16+LS stack | wave-11 WIP |
| thorfinn | #4008 | surf_weight sweep {5.0, 20.0} on BF16+LS+n10 | wave-11 WIP |
| nezuko | #4009 | Gradient clip sweep {0.5, 1.0} on BF16+LS+n10 | wave-11 WIP |
| frieren | #4014 | Width scaling narrower: n_hidden=120 on BF16+LS+n10 | wave-11 WIP |
| alphonse | #4026 | Batch size sweep {2, 8} on BF16+LS+n10 — first ever bs sweep | wave-11 WIP |
| askeladd | #4027 | LR sweep {7e-4, 1e-3} on BF16+LS+n10 | wave-11 WIP |

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
| #4007 (frieren) | Width n=144: timeout-bound at 15 epochs vs 17+ needed |
| #3983 (askeladd) | Huber δ {0.15, 0.5}: both regress vs δ=0.3 on FP32 stack |
| #3964 (alphonse) | LayerScale γ-init {0.005, 0.020}: both regress vs γ=0.01 |
| #4005 (tanjiro) | BF16+LS+n10+EMA 0.998 missing cell: EMA hurts at 17-epoch BF16 horizon (val 68.64 vs 67.19); EMA window covers only 41% of training. **Drop EMA entirely on BF16 stack.** |

## Current research themes

### BF16 regime shift (wave-11 — all new assignments target BF16+LS+n10)

Key insight: **BF16 extended convergence horizon changes optimal hyperparameters**. At 12 epochs (FP32), n14>n10 and EMA helps. At 17 epochs (BF16), n10>n14 and EMA costs more than it gains. All sweepable hyperparameters need re-evaluation on the new BF16+LS+n10 stack.

1. **Missing cell — CLOSED #4005**: BF16+LS+n10+EMA 0.998 underperforms (val=68.64 vs 67.19). Drop EMA on BF16 stack.

2. **n_freqs on BF16+LS — fern #4006** (WIP): n=8 and n=12. Aliasing regime reversed: n10>n14 at 17 epochs. Does n=8 win? n=12 interpolates.

3. **Width scaling (narrow bracket) — frieren #4014** (WIP): n_hidden=120 with BF16+LS+n10. n=144 closed (timeout-bound); n=120 brackets the n=128 baseline.

4. **surf_weight sweep — thorfinn #4008** (WIP): surf_weight={5.0, 20.0} vs default 10.0. **Never tested in this programme.** Direct control over surface vs volume loss weighting.

5. **Clip sweep — nezuko #4009** (WIP): clip={0.5, 1.0} vs 0.25. clip_frac=1.0 throughout all runs. LayerScale provides implicit gating; maybe clip=0.25 is double-regularizing. Prior clip=1.0 test (PR #3529) was pre-LayerScale.

6. **Batch size sweep — alphonse #4026** (WIP): batch_size={2, 8} vs default 4. Never tested in this programme. BF16's freed memory makes bs=8 viable; bs=2 tests if smaller-batch noise helps generalization.

7. **LR sweep on BF16 — askeladd #4027** (WIP): lr={7e-4, 1e-3} vs default 5e-4. Prior FP32 LR sweep (#3784) was clip-saturated; on BF16+LS+n10 with 17 epochs and unsaturated gradient dynamics, higher LR may now manifest.

8. **Huber δ on BF16 — tanjiro #4033** (NEW): δ={0.15, 0.5} on BF16+LS+n10. Last unswept primary axis. FP32 confirmed δ=0.3 (#3424, #3983 both bracket); BF16's smaller late-epoch residuals may shift optimum.

### Old FP32 triple stack (wave-10, completing)

Edward #3971 still running EMA warm-up ramp on FP32 triple. Result will inform whether EMA delivers when "born ramping". Askeladd #3983 and alphonse #3964 closed (δ and γ-init bracketed; both confirmed at original values).

## Key insights accumulated

- **BF16 regime shift**: n_freqs=10 > n_freqs=14 at 17 epochs. Everything from FP32 triple needs re-evaluation on BF16+LS+n10.
- **clip=0.25 is 100% saturated**: Every step is clipped. Effective LR = clip × grad direction. clip sweep is high priority on new stack.
- **T_max=20 confirmed optimal** for 12-17 epoch runs. T_max=12 traps in suboptimal basin. No re-sweep needed.
- **surf_weight=10.0 never swept**: Only major hyperparameter unexplored. High priority.
- **AdamW fully confirmed**: β2=0.999, eps=1e-8, WD=1e-4 all optimal. Inner-optimizer exhausted.
- **Huber δ=0.3 fully confirmed** on FP32 stack (#3424 and #3983 both bracket). Validate on BF16+LS+n10 next.
- **LayerScale γ=0.01 fully confirmed** (#3593 win, #3740 asymmetric closed, #3964 bracket both regress).
- **Learnable Fourier dead**: Frequencies don't migrate; overhead too costly.
- **LayerScale γ=0.01 biggest single win**: −10.2% val from per-channel residual gating.
- **EMA on BF16 stack is now DEFINITIVELY dead.** All three EMA tests on BF16+LS regress: n14+EMA (68.50), n10+EMA (68.64), quad-compound. EMA's smoothing window covers only 41% of training; cost (~2 epochs) exceeds benefit. #4005 closed the question.

## Potential next research directions

- **n_layers depth scaling** — no current flag; requires train.py modification; BF16 memory headroom makes viable
- **n_freqs=6** — if n=8 wins on fern's sweep, continue the aliasing trend lower
- **n_hidden=160 / 96** — depends on frieren #4014 narrow bracket outcome
- **Huber δ on new BF16 stack** — re-test {0.15, 0.5} at extended 17-epoch horizon
- **LR + clip co-tuning** — clip and LR interact strongly; once individual sweeps settle, run 2D grid
- **Sub-60 val target**: val=67.19 now; missing cell (#4005) + width narrowing + LR sweep could push to 63-65

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
