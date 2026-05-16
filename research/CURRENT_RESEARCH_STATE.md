# SENPAI Research State

- **Date:** 2026-05-16 17:35
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4006 (fern, merged):** BF16 + LayerScale γ=0.01 + **n_freqs=8** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **val_avg/mae_surf_p: 64.08** | **test_avg/mae_surf_p: 55.05**
- Per-split test surf_p: single=62.10, rc=68.13, cruise=36.63, re_rand=53.35 — **all 4 splits improve uniformly**
- **Cumulative improvement: -50.2% val from round-5 start (~128.69)** — breaks the -50% threshold
- Key insight: at 1499 train samples + 17-epoch BF16 horizon, lower n_freqs wins decisively. Non-monotonic ordering: n=8 < n=12 < n=14 ≈ n=10. The earlier "n=14 over-fits" framing was confounded by EMA overhead.

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
| #3527 (tanjiro, merged) | BF16 + LayerScale + n_freqs=10 | 67.19 | 58.05 | -5.6% |
| #4009 (nezuko, merged) | BF16 + LS + n10 + clip=1.0 | 65.70 | 57.80 | -2.22% |
| **#4006 (fern, merged)** | **BF16 + LS + n_freqs=8 (clip=0.25)** | **64.08** | **55.05** | **-2.47%** |

## Active WIP (8 students, full deck)

| Student | PR | Hypothesis | Status | Baseline |
|---|---|---|---|---|
| fern | #4058 | n_freqs lower {4, 6} on BF16+LS+n8 | wave-13 NEW | new 64.08 |
| thorfinn | #4059 | sw {2.5, 5.0} compound test on BF16+LS+n8 | wave-13 NEW | new 64.08 |
| frieren | #4060 | fourier_base {1.5, 2.5} on BF16+LS+n8 | wave-13 NEW | new 64.08 |
| edward | #4053 | n_freqs {8, 12} at clip=1.0 — INCLUDES the critical n=8+clip=1.0 compound | wave-12 WIP | 65.70 |
| nezuko | #4052 | clip ceiling {2.0, 4.0} on BF16+LS+n10+clip=1.0 | wave-12 WIP | 65.70 |
| tanjiro | #4033 | Huber δ {0.15, 0.5} on BF16+LS+n10+clip=0.25 | wave-11 WIP | 67.19 |
| alphonse | #4026 | batch_size {2, 8} on BF16+LS+n10+clip=0.25 | wave-11 WIP | 67.19 |
| askeladd | #4027 | LR {7e-4, 1e-3} on BF16+LS+n10+clip=0.25 | wave-11 WIP | 67.19 |

**Note:** wave-11 and wave-12 PRs were assigned against older baselines but any result <64.08 is a new winner. Otherwise their findings still inform variable optimums independently of n_freqs.

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
| #4005 (tanjiro) | BF16+LS+n10+EMA 0.998: val=68.64 (worse). EMA dead on BF16 |
| #3971 (edward) | EMA warm-up on FP32 triple: arm-1 +4.1%, arm-2 +50%. Superseded by BF16+clip=1.0 |
| #4008 (thorfinn) | sw=5: val=64.10 — independent win superseded by parallel #4006 (n=8) merge |
| #4014 (frieren) | Width n=120: throughput hypothesis fails (no speedup), test ties |

## Current research themes

### Wave-13: BF16+LS+n=8+clip=0.25 (new merged baseline)

The big shifts in the last two waves:
- **clip=1.0 (PR #4009)**: 4× larger effective step — wave-12 winner
- **n_freqs=8 (PR #4006)**: lower aliasing wins decisively — wave-13 winner

These two improvements have NOT been compounded yet. Edward #4053 is testing n=8+clip=1.0 right now — that's the highest-leverage in-flight experiment.

**Wave-13 new sweeps** (all on n=8+clip=0.25):
1. **fern #4058**: continue aliasing trend lower — n_freqs={4, 6}. Does the curve bottom at n=8 or keep going?
2. **thorfinn #4059**: surf_weight compound test — sw={2.5, 5.0} on n=8. Independent levers; expected to compound additively.
3. **frieren #4060**: fourier_base sweep — base={1.5, 2.5} at n=8. Per-band spacing matters more at fewer bands.

### Wave-12 (in flight, will be reviewed when terminal)

- **edward #4053**: n_freqs={8, 12} at clip=1.0 — answers the critical n=8+clip=1.0 compound question
- **nezuko #4052**: clip ceiling {2.0, 4.0} on n=10+clip=1.0 — pushes clip further

### Wave-11 (in flight, on old 67.19 baseline)

These were assigned before the n=8 / clip=1.0 wins. Results still inform optimums of independent variables (though sub-optimal effective baseline):
- **tanjiro #4033**: Huber δ {0.15, 0.5}
- **alphonse #4026**: batch_size {2, 8} — first ever bs sweep
- **askeladd #4027**: LR {7e-4, 1e-3}

## Key insights accumulated

- **n_freqs=8 is the new default** on BF16+LS stack. Lower aliasing wins at 1499 train samples.
- **n_freqs ordering (matched recipes)**: 8 < 12 < 14 ≈ 10 — non-monotonic; the "n=10 sweet spot" framing from PR #3527 was confounded by EMA overhead in the n=14 arm.
- **clip=1.0 wins on n=10 stack**. Compound with n=8 not yet tested (edward in flight).
- **Clip acts as lr-scale in fully-clipped regime.** clip_frac=1.0 throughout most runs; effective step ≈ clip × grad/‖grad‖.
- **BF16 regime shift**: extended convergence horizon (17 epochs vs 12) changes optimal hyperparameters.
- **EMA dead on BF16 stack**: 3 separate tests confirm. Smoothing window covers only 41% of training; cost exceeds benefit.
- **T_max=20 confirmed optimal** for 12-17 epoch runs.
- **surf_weight curve is monotonic-decreasing in 5-20 range** at clip=0.25, n=10 (sw=5 better than sw=10 > sw=20). Compound on n=8 untested.
- **AdamW fully confirmed**: β2=0.999, eps=1e-8, WD=1e-4 all optimal.
- **Huber δ=0.3 confirmed** on FP32 stack. Re-testing on BF16+clip=0.25 via tanjiro #4033.
- **LayerScale γ=0.01 fully confirmed** across multiple bracket experiments.
- **Width n_hidden=128 is the right point**: narrower (120, 144) both fail. Transolver attention slicing dominates runtime, so width changes don't buy epochs.

## Potential next research directions

- **n_freqs={2, 3} ultra-low** — if n=4 or n=6 wins on fern's #4058, push even further
- **fourier_base offset** — if base sweep shows a direction, refine around the winner
- **n=8 + clip=1.0 + sw=5 triple** — once all three are independently confirmed
- **n_layers depth scaling** — BF16 memory headroom; requires train.py modification
- **slice_num sweep** — Transolver's slice_num=64 default never swept; attention-bound runtime makes this high-leverage
- **mlp_ratio sweep** — currently 2; could try 4 (higher capacity) or 1 (regularization)
- **Sub-60 val target**: val=64.08 now; n=8+clip=1.0 compound could push to ~62; +sw=5 could push lower

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
