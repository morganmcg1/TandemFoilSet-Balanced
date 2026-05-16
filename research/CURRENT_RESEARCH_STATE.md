# SENPAI Research State

- **Date:** 2026-05-16 19:15
- **Branch:** `icml-appendix-charlie-pai2i-48h-r5`
- **Most recent human-team direction:** _(no issues specific to this arm)_

## Current best

- **PR #4026 (alphonse, merged):** BF16 + LayerScale γ=0.01 + n_freqs=10 + **batch_size=2** + Huber-0.3 + T_max=20 + clip=0.25 (no EMA)
- **val_avg/mae_surf_p: 60.67** | **test_avg/mae_surf_p: 53.11**
- Per-split test surf_p: single=57.99, rc=66.40, cruise=35.50, re_rand=52.54 — **all 4 splits improve uniformly**
- best_epoch=18/18 (timeout-bound, still descending — head-room remains)
- Peak memory **18.5 GB** (vs 33 GB baseline, -44%); throughput 102.6 s/epoch (-8%)
- **Cumulative improvement: -52.8% val from round-5 start (~128.69)**
- Key insight: in the **clip-saturation regime** (clip_frac=1.0 throughout), batch_size acts purely as a "steps in budget" lever — bs=2 gets 4.5× more updates than bs=8 in the same 30-min budget. The per-step update magnitude is fixed at `clip × dir(grad)`, so optimization progress is dominated by total update count.

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
| #4006 (fern, merged) | BF16 + LS + n_freqs=8 (clip=0.25) | 64.08 | 55.05 | -2.47% |
| **#4026 (alphonse, merged)** | **BF16 + LS + n10 + batch_size=2** | **60.67** | **53.11** | **-5.32%** |

## Active WIP (8 students)

| Student | PR | Hypothesis | Status | Baseline |
|---|---|---|---|---|
| alphonse | #4083 | bs=2 + n_freqs=8 compound (+ lr_t_max=18 arm) | wave-14 NEW | 60.67 |
| nezuko | #4095 | bs=2 + clip=1.0 compound; arm-2: triple bs=2+n=8+clip=1.0 | wave-14 NEW | 60.67 |
| fern | #4058 | n_freqs lower {4, 6} on BF16+LS+n8 | wave-13 WIP | 64.08 |
| thorfinn | #4059 | sw {2.5, 5.0} compound test on BF16+LS+n8 | wave-13 WIP | 64.08 |
| frieren | #4060 | fourier_base {1.5, 2.5} on BF16+LS+n8 | wave-13 WIP | 64.08 |
| edward | #4053 | n_freqs {8, 12} at clip=1.0 — INCLUDES critical n=8+clip=1.0 compound | wave-12 WIP | 65.70 |
| tanjiro | #4033 | Huber δ {0.15, 0.5} on BF16+LS+n10+clip=0.25 | wave-11 WIP | 67.19 |
| askeladd | #4027 | LR {7e-4, 1e-3} on BF16+LS+n10+clip=0.25 | wave-11 WIP | 67.19 |

**Note:** All in-flight PRs assigned before bs=2 merge → baseline now 60.67. Any result below that is a new winner. Wave-11 results still inform variable optimums independently (different lever axes from bs).

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
| #4052 (nezuko) | clip ceiling {2.0, 4.0}: both arms regress on val (68.21 / 66.46 vs 65.70). Ceiling confirmed at clip=1.0. Non-monotone: clip=2.0 worst (+3.81%), clip=4.0 mixed (+1.16% val / -1.04% test) |

## Current research themes

### Wave-14: BF16+LS+n10+bs=2 (new merged baseline)

The big shift this wave:
- **batch_size=2 (PR #4026)**: 4.5× more updates in same wall-clock budget — wave-14 winner. Mechanism is clip-saturation, not gradient noise.

**Critical implication:** bs=2 is independent of every other lever swept so far because it operates on *step count*, not on per-step quality. So it should compound with: n_freqs (representation), clip ceiling (step size), Huber δ (residual shape), surf_weight (loss balance).

**Wave-14 priorities (orthogonal compound tests):**
1. **alphonse new assignment**: bs=2 + n_freqs=8 — the most obvious compound (both #4006 and #4026 are independently merged winners)
2. **bs=2 + clip=1.0 compound** — clip=1.0 already merged at #4009; needs re-test now that bs=2 changes step count
3. **bs=1** — push the steps lever further; does it keep paying? Or does gradient variance start mattering once we're saturating the GPU pipeline overhead?
4. **bs=2 + longer epochs / different lr_t_max** — best_epoch=18/18 means we hit timeout still descending; lr_t_max=18 (per student suggestion) lets the cosine schedule finish

### Wave-13 (in flight, on n=8 baseline 64.08)

- **fern #4058**: n_freqs lower {4, 6} — does the curve bottom at n=8 or keep going?
- **thorfinn #4059**: sw {2.5, 5.0} compound test on n=8
- **frieren #4060**: fourier_base {1.5, 2.5} on n=8

These are still useful: any result <60.67 is a new winner, and they inform whether the optimum of each independent lever shifts on the n=8 stack.

### Wave-12 (in flight, on n=10+clip=1.0 baseline 65.70)

- **edward #4053**: n_freqs {8, 12} at clip=1.0 — answers the critical n=8+clip=1.0 compound question
- **nezuko #4052**: clip ceiling {2.0, 4.0} on n=10+clip=1.0 — pushes clip further

### Wave-11 (in flight, on older 67.19 baseline)

- **tanjiro #4033**: Huber δ {0.15, 0.5} — actively training at 100% GPU
- **askeladd #4027**: LR {7e-4, 1e-3} — recovering from GitHub rate-limit window (15:43–18:24 UTC)

## Key insights accumulated

- **batch_size=2 is the new default**. clip-saturation regime turns batch_size into pure steps-in-budget lever (mechanism is independent of gradient noise).
- **Memory headroom: 18.5 GB peak** — huge head-room for width/depth/Fourier expansion experiments without OOM risk.
- **best_epoch=18/18 at bs=2**: still descending at timeout → more epochs would help; lr_t_max=18 (per student note) lets cosine finish.
- **n_freqs=8 is the right default at bs=8/clip=0.25**. Compound with bs=2 not yet tested.
- **n_freqs ordering (matched recipes)**: 8 < 12 < 14 ≈ 10 — non-monotonic; the "n=10 sweet spot" framing from PR #3527 was confounded by EMA overhead in the n=14 arm.
- **clip=1.0 wins on n=10 stack** (#4009). Compound with n=8 and with bs=2 not yet tested.
- **Clip acts as lr-scale in fully-clipped regime.** clip_frac=1.0 throughout most runs; effective step ≈ clip × grad/‖grad‖.
- **BF16 regime shift**: extended convergence horizon (17-18 epochs vs 12) changes optimal hyperparameters.
- **EMA dead on BF16 stack**: 3 separate tests confirm. Smoothing window covers only 41% of training; cost exceeds benefit.
- **T_max=20 confirmed optimal** for 12-17 epoch runs — but bs=2 may want lr_t_max=18 since best_epoch=18.
- **surf_weight curve is monotonic-decreasing in 5-20 range** at clip=0.25, n=10. Compound on n=8 untested.
- **AdamW fully confirmed**: β2=0.999, eps=1e-8, WD=1e-4 all optimal.
- **Huber δ=0.3 confirmed** on FP32 stack. Re-testing on BF16+clip=0.25 via tanjiro #4033.
- **LayerScale γ=0.01 fully confirmed** across multiple bracket experiments.
- **Width n_hidden=128 is the right point**: narrower (120, 144) both fail. With bs=2's memory savings, 192/256 width experiments are now safe to try.

## Potential next research directions

- **bs=1** — push the steps lever further. ~27,000 updates in 30 min budget. Caveat: GPU compute headroom may not exist at bs=1.
- **bs=2 + n=8 + clip=1.0 triple** — combine all three independent wins
- **bs=2 + lr_t_max=18** — let cosine finish given best_epoch=18 (student-suggested)
- **bs=2 + EMA re-test** — EMA's "smoothing window covers <50%" critique partly inverts when steps double; the question is open at 13,500 updates
- **bs=2 + width expansion** — 18.5 GB headroom means n_hidden=256 fits, slice_num=128 fits
- **mlp_ratio sweep** — currently 2; with extra memory budget, mlp_ratio=4 is safe
- **slice_num sweep** — Transolver's slice_num=64 default never swept; attention-bound runtime makes this high-leverage; with bs=2 memory headroom we can try slice_num=128/256
- **Sub-55 val target**: val=60.67 now; bs=2 + n=8 compound is the obvious next push toward 55-57; +lr_t_max=18 could go lower

## Ideas dossier
- Full hypothesis catalogue: `research/RESEARCH_IDEAS_2026-05-15.md` (13 ranked entries)
